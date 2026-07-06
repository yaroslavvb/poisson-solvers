"""Neural Preconditioning Operator (NPO): toy NAMG model + PCG wrapper.

Toy-scale reimplementation of Li, Xiao, Lai & Wang, "Neural Preconditioning
Operator for Efficient PDE Solves", arXiv:2502.01337v2 (2025). The network
``M_theta`` learns an approximate inverse of the (scaled) discrete Poisson
operator; inside a Krylov solver the preconditioned residual is
``z = M_theta(r)`` (paper Sec. 3.2).

Architecture — NAMG (neural-attention multigrid, paper Sec. 3.4, Fig. 2),
reduced to toy scale but faithful in structure::

    lift (1x1 conv, width C=32, with appended x/y coordinate channels)
      -> pre-relaxation: 3x3 conv + ReLU (learned local smoother, the
         analogue of one weighted-Jacobi pre-smoothing sweep)
      -> LayerNorm -> NAMG former:
           restriction  (Eq. 11-12): m=64 LEARNED coarse queries
             cross-attend to the fine tokens — the attention weights play
             the role of the learned restriction R = A . E_theta. Learned
             queries keep the model resolution-agnostic: the fine token
             sequence may have any length N.
           coarse level (Eq. 13-14): one 4-head self-attention block over
             the m coarse tokens (+ an FFN, our addition beyond Eq. 13-14).
           prolongation (Eq. 15): fine tokens cross-attend back to the
             coarse tokens (P = A . E_theta^T analogue); the result is
             residual-added to the fine features — the coarse-grid
             correction x'^f = x^f + P x'^c.
      -> LayerNorm -> FeedForward
      -> post-relaxation: 3x3 conv + ReLU -> project to 1 channel (1x1 conv)

Hyperparameters follow paper Table 6 in spirit: feature width 32, one pre-
and one post-relaxation, ReLU activations, 4 attention heads; the paper's
``num_c = 128`` coarse tokens is halved to m = 64 for the 32x32 toy grid.
LayerNorms are applied pre-norm (inside the branches) so the trunk
preserves residual amplitude information.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]

# CRITICAL SCALING FACT: PCG iterates are invariant to scaling the
# preconditioner M by any positive constant.  We therefore train the network
# to invert the SCALED matrix A_hat = h^2 * A, whose spectrum for n = 32 is
# ~[0.018, 8] (comfortable float32 numerics), and use the resulting M_theta
# DIRECTLY as a preconditioner for A = A_hat / h^2: since
# M_theta ~ A_hat^{-1} = h^{-2} A^{-1} is a positive multiple of A^{-1},
# no rescaling is needed.


class NPO(nn.Module):
    """NAMG-lite neural preconditioning operator.

    Maps a residual field ``r`` of shape ``(B, 1, n, n)`` to an approximate
    error correction ``z ~ A_hat^{-1} r`` of the same shape, where
    ``A_hat = h^2 * A`` is the scaled 5-point Poisson operator.

    Parameters
    ----------
    width : int, optional
        Feature width C (paper Table 6 uses 32).
    num_coarse : int, optional
        Number m of learned coarse tokens (paper uses num_c=128; 64 here).
    num_heads : int, optional
        Attention heads (paper Table 6 uses 4).
    ffn_mult : int, optional
        Hidden-layer expansion factor of the feed-forward blocks.
    """

    def __init__(self, width=32, num_coarse=64, num_heads=4, ffn_mult=2):
        super().__init__()
        self.width = width
        # Lift: residual + 2 coordinate channels -> C features. Coordinate
        # channels supply position information without a fixed-size
        # positional embedding, preserving resolution-agnosticism.
        self.lift = nn.Conv2d(3, width, kernel_size=1)
        self.pre_relax = nn.Conv2d(width, width, kernel_size=3, padding=1)

        # NAMG former (paper Eq. 11-15).
        self.coarse_queries = nn.Parameter(
            torch.randn(num_coarse, width) / math.sqrt(width)
        )
        self.norm_fine_r = nn.LayerNorm(width)  # before restriction
        self.restrict = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.norm_coarse_a = nn.LayerNorm(width)
        self.coarse_self = nn.MultiheadAttention(width, num_heads, batch_first=True)
        self.norm_coarse_f = nn.LayerNorm(width)
        self.coarse_ffn = nn.Sequential(
            nn.Linear(width, ffn_mult * width),
            nn.ReLU(),
            nn.Linear(ffn_mult * width, width),
        )
        self.norm_fine_p = nn.LayerNorm(width)  # fine queries for prolongation
        self.norm_coarse_p = nn.LayerNorm(width)  # coarse keys/values
        self.prolong = nn.MultiheadAttention(width, num_heads, batch_first=True)

        self.norm_fine_f = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_mult * width),
            nn.ReLU(),
            nn.Linear(ffn_mult * width, width),
        )

        self.post_relax = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.proj = nn.Conv2d(width, 1, kernel_size=1)

    @staticmethod
    def _coord_channels(r):
        """Constant x/y coordinate channels in [0, 1], shape (B, 2, n1, n2)."""
        b, _, n1, n2 = r.shape
        xs = torch.linspace(0.0, 1.0, n1, device=r.device, dtype=r.dtype)
        ys = torch.linspace(0.0, 1.0, n2, device=r.device, dtype=r.dtype)
        gx = xs.view(1, 1, n1, 1).expand(b, 1, n1, n2)
        gy = ys.view(1, 1, 1, n2).expand(b, 1, n1, n2)
        return torch.cat([gx, gy], dim=1)

    def forward(self, r):
        """Apply M_theta to a batch of residual fields.

        Parameters
        ----------
        r : torch.Tensor
            Shape ``(B, 1, n, n)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1, n, n)``, approximation of ``A_hat^{-1} r``.
        """
        b, _, n1, n2 = r.shape
        v = self.lift(torch.cat([r, self._coord_channels(r)], dim=1))
        # Pre-relaxation: learned local smoothing step (weighted-Jacobi
        # analogue), residual form x' = x + smoother(x).
        v = v + F.relu(self.pre_relax(v))

        # Fine tokens (B, N, C).
        t = v.flatten(2).transpose(1, 2)

        # Restriction (Eq. 11-12): learned coarse queries attend to fine
        # tokens; softmax attention weights realize the learned restriction.
        tn = self.norm_fine_r(t)
        q = self.coarse_queries.unsqueeze(0).expand(b, -1, -1)
        c, _ = self.restrict(q, tn, tn, need_weights=False)

        # Coarse-level mixing: self-attention (Eq. 13-14) + FFN (ours).
        cn = self.norm_coarse_a(c)
        cs, _ = self.coarse_self(cn, cn, cn, need_weights=False)
        c = c + cs
        c = c + self.coarse_ffn(self.norm_coarse_f(c))

        # Prolongation + coarse-grid correction (Eq. 15): fine tokens attend
        # back to coarse tokens; correction is residual-added.
        cp = self.norm_coarse_p(c)
        corr, _ = self.prolong(self.norm_fine_p(t), cp, cp, need_weights=False)
        t = t + corr

        # LayerNorm -> FeedForward (pre-norm residual block).
        t = t + self.ffn(self.norm_fine_f(t))

        v = t.transpose(1, 2).reshape(b, self.width, n1, n2)
        # Post-relaxation (post-smoothing sweep) and projection.
        v = v + F.relu(self.post_relax(v))
        return self.proj(v)


class NPOPreconditioner:
    """Trained NPO as a PCG preconditioner callable ``z = M(r)``.

    Loads ``results/npo_checkpoint.pt`` (written by ``train_npo.py``) and
    exposes ``__call__(r)`` / ``apply(r)`` mapping float64 -> float64, the
    interface expected by ``pcg.pcg`` / ``pcg.flexible_pcg``.

    Because M_theta contains ReLU units it is NOT a fixed SPD matrix, so
    classical PCG theory does not apply; use ``pcg.flexible_pcg``
    (Polak-Ribiere beta, Notay 2000) for reliable convergence.

    The input is normalized to unit norm before the network and the output
    is rescaled by the same factor: the network only ever sees unit-norm
    residuals (matching training) and the resulting map is positively
    homogeneous of degree 1, ``M(c r) = c M(r)`` for ``c > 0``.

    Parameters
    ----------
    checkpoint : str or pathlib.Path, optional
        Path to the checkpoint (default ``<repo>/results/npo_checkpoint.pt``).
    device : str, optional
        Torch device (default ``"cpu"``).
    """

    def __init__(self, checkpoint=None, device="cpu"):
        if checkpoint is None:
            checkpoint = _REPO_ROOT / "results" / "npo_checkpoint.pt"
        ckpt = torch.load(checkpoint, map_location=device)
        self.model = NPO(**ckpt["config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device)
        self.model.eval()
        self.device = device

    def apply(self, r):
        """Apply the preconditioner: ``z = M_theta(r) ~ (h^2 A)^{-1} r``.

        Parameters
        ----------
        r : numpy.ndarray
            Residual vector of length ``n*n`` (float64).

        Returns
        -------
        numpy.ndarray
            Preconditioned residual, float64, same shape as ``r``.
        """
        r64 = np.asarray(r, dtype=np.float64)
        nrm = float(np.linalg.norm(r64))
        if nrm == 0.0:
            return np.zeros_like(r64)
        n = int(round(math.sqrt(r64.size)))
        t = torch.from_numpy((r64 / nrm).astype(np.float32))
        t = t.reshape(1, 1, n, n).to(self.device)
        with torch.no_grad():
            z = self.model(t)
        return z.double().cpu().numpy().ravel() * nrm

    __call__ = apply
