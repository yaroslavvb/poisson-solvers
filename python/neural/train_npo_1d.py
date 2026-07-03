"""Train a tiny 1D toy neural preconditioner for the browser demo (cg-explorer/).

Following the NPO recipe (arXiv:2502.01337): the network M_theta is trained so
that A M_theta(r) ~ r over sampled residual-like vectors (the paper's condition
loss, Eq. 9, which coincides with the residual loss, Eq. 10, when applied to
right-hand-side samples). The architecture is a miniature of the paper's
multigrid design: a local fine-scale conv stack plus a coarse-grid branch
(16 cells) that handles the low-frequency modes a local stencil cannot reach.
It is deliberately small so it accelerates CG visibly without trivializing
the solve - exactly what the interactive visualization needs.

Trains against A_hat = h^2 A = tridiag(-1,2,-1) (spectrum in (0,4), float-friendly);
since (F)CG iterates are invariant to positive scaling of the preconditioner,
the same network preconditions A unchanged.

The pooling / linear-upsample conventions here are mirrored line-for-line by
applyNPO() in cg-explorer/app.js; keep them in sync.

Outputs cg-explorer/npo_weights.js with the selected checkpoint's weights.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
N = 256
NC = 16          # coarse cells
POOL = N // NC
H = 1.0 / (N + 1)
TOL = 1e-4
TARGET_ITERS = (12, 60)   # visibly faster than CG, but nowhere near one step
MAX_ERR = 0.10            # solution error at stop must stay presentable


def apply_ahat_t(v: torch.Tensor) -> torch.Tensor:
    """A_hat v = tridiag(-1,2,-1) v with Dirichlet ends, batched (B, N)."""
    out = 2 * v.clone()
    out[:, 1:] -= v[:, :-1]
    out[:, :-1] -= v[:, 1:]
    return out


# explicit align-corners=False linear upsample indices, mirrored in app.js
_pos = (np.arange(N) + 0.5) * NC / N - 0.5
_j0 = np.clip(np.floor(_pos).astype(int), 0, NC - 1)
_j1 = np.clip(_j0 + 1, 0, NC - 1)
_w = np.clip(_pos - np.floor(_pos), 0.0, 1.0)
J0 = torch.from_numpy(_j0)
J1 = torch.from_numpy(_j1)
WUP = torch.from_numpy(_w.astype(np.float32))


class ToyNPO1D(nn.Module):
    """Fine conv stack + coarse-grid conv branch (a multigrid miniature)."""

    def __init__(self, channels=8, kernel=9, ckernel=5):
        super().__init__()
        p, cp = kernel // 2, ckernel // 2
        self.fine = nn.Sequential(
            nn.Conv1d(1, channels, kernel, padding=p), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=p), nn.ReLU(),
            nn.Conv1d(channels, 1, kernel, padding=p),
        )
        self.coarse = nn.Sequential(
            nn.Conv1d(1, channels, ckernel, padding=cp), nn.ReLU(),
            nn.Conv1d(channels, channels, ckernel, padding=cp), nn.ReLU(),
            nn.Conv1d(channels, 1, ckernel, padding=cp),
        )

    def forward(self, r):  # (B, N) -> (B, N)
        fine = self.fine(r[:, None, :])[:, 0, :]
        pooled = r.view(-1, NC, POOL).mean(-1)               # (B, NC)
        c = self.coarse(pooled[:, None, :])[:, 0, :]          # (B, NC)
        up = (1 - WUP) * c[:, J0] + WUP * c[:, J1]            # (B, N)
        return fine + up


GRID = torch.arange(1, N + 1, dtype=torch.float32) * H


def modes_batch(ks):
    """Rows sin(k pi x_i) for the given mode indices k (eigenvectors of A_hat)."""
    return torch.sin(torch.pi * ks.float()[:, None] * GRID[None, :])


def sample_batch(bs=96):
    """Residual-like vectors spanning the whole spectrum: white noise, smoothed
    noise, low-mode combos, high-mode combos, random single eigenmodes, spikes.
    Late-iteration FCG residuals concentrate in whatever modes the net handles
    worst, so training must cover all of them."""
    q = bs // 6
    white = torch.randn(q, N)
    smooth = torch.randn(q, N)
    for _ in range(10):  # damped-Jacobi smoothing passes mimic late-iteration residuals
        smooth = smooth - 0.25 * apply_ahat_t(smooth)
    low = torch.randn(q, 8) @ modes_batch(torch.arange(1, 9))
    high = torch.randn(q, 8) @ modes_batch(torch.arange(N - 8, N))
    single = modes_batch(torch.randint(1, N + 1, (q,))) * torch.randn(q, 1)
    spikes = torch.zeros(bs - 5 * q, N)
    for row in spikes:
        idx = torch.randint(0, N, (3,))
        row[idx] = torch.randn(3)
    r = torch.cat([white, smooth, low, high, single, spikes])
    return r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)


# ---------- numpy-side evaluation on the real heater/chiller problem ----------

def np_apply_a(v):
    out = 2 * v.copy()
    out[1:] -= v[:-1]
    out[:-1] -= v[1:]
    return out / H**2


def exact_solution(b):
    c = np.zeros(N); d = np.zeros(N); x = np.zeros(N)
    c[0] = -0.5; d[0] = H**2 * b[0] / 2
    for i in range(1, N):
        m = 2 + c[i - 1]
        c[i] = -1 / m
        d[i] = (H**2 * b[i] + d[i - 1]) / m
    x[-1] = d[-1]
    for i in range(N - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def make_np_precond(model):
    def M(r):
        nr = float(np.linalg.norm(r))
        if nr == 0:
            return r.copy()
        with torch.no_grad():
            z = model(torch.from_numpy((r / nr).astype(np.float32))[None, :])[0].numpy()
        return z.astype(np.float64) * nr  # unit-norm wrapper => positively homogeneous
    return M


def fcg_eval(M, maxit=500):
    """Flexible CG (Polak-Ribiere beta), tol 1e-2. Returns (iters, err_at_stop)."""
    b = np.zeros(N)
    b[0], b[-1] = 1.0, -1.0
    xstar = exact_solution(b)
    bn = np.linalg.norm(b)
    def safeguarded(r):
        # descent safeguard, mirrored in app.js: if the nonlinear net fails to
        # produce a descent direction, fall back to the raw residual this step
        z = M(r)
        if r @ z <= 1e-14 * np.linalg.norm(r) * np.linalg.norm(z):
            return r.copy()
        return z

    x, r = np.zeros(N), b.copy()
    z = safeguarded(r)
    p, rz = z.copy(), r @ z
    for k in range(1, maxit + 1):
        ap = np_apply_a(p)
        alpha = rz / (p @ ap)
        x += alpha * p
        r_new = r - alpha * ap
        if np.linalg.norm(r_new) / bn <= TOL:
            return k, float(np.linalg.norm(x - xstar) / np.linalg.norm(xstar))
        z = safeguarded(r_new)
        beta = (z @ (r_new - r)) / rz
        rz = z @ r_new
        p = z + beta * p
        r = r_new
    return maxit + 1, float(np.linalg.norm(x - xstar) / np.linalg.norm(xstar))


def stack_to_json(seq):
    layers = []
    for mod in seq:
        if isinstance(mod, nn.Conv1d):
            layers.append({
                "w": [[[float(f"{w:.7g}") for w in kern] for kern in cout] for cout in mod.weight.detach().tolist()],
                "b": [float(f"{v:.7g}") for v in mod.bias.detach().tolist()],
                "k": mod.kernel_size[0],
            })
    return layers


def export_js(model, path, meta):
    blob = {"fine": stack_to_json(model.fine), "coarse": stack_to_json(model.coarse),
            "nc": NC, **meta}
    js = ("// Toy 1D neural preconditioner weights (see python/neural/train_npo_1d.py).\n"
          "// Two-scale conv net trained with the condition/residual losses of arXiv:2502.01337.\n"
          f"const NPO_WEIGHTS = {json.dumps(blob)};\n"
          "if (typeof module !== 'undefined') module.exports = NPO_WEIGHTS;\n")
    path.write_text(js)
    return len(js)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    model = ToyNPO1D()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    checkpoints = []
    for step in range(1, 601):
        r = sample_batch()
        z = model(r)
        cond = ((apply_ahat_t(z) - r) ** 2).sum(dim=1).mean()  # condition loss, Eq. 9
        pd_pen = torch.relu(-(z * r).sum(dim=1)).mean()  # keep M a descent map: r.M(r) > 0
        loss = cond + pd_pen
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            checkpoints.append((step, {k: v.clone() for k, v in model.state_dict().items()}, float(loss)))
            print(f"step {step:4d}  condition loss {float(loss):.4f}")

    # pick the checkpoint with acceptable error whose iteration count sits in the window
    chosen = None
    for step, sd, loss in checkpoints:
        model.load_state_dict(sd)
        iters, err = fcg_eval(make_np_precond(model))
        ok = TARGET_ITERS[0] <= iters <= TARGET_ITERS[1] and err <= MAX_ERR
        print(f"checkpoint step {step:4d}: FCG iters = {iters:3d}, err at stop = {100*err:.1f}%{'  <- candidate' if ok else ''}")
        if ok and (chosen is None or err < chosen[3]):
            chosen = (step, sd, iters, err)
    if chosen is None:
        step, sd, loss = checkpoints[-1]
        model.load_state_dict(sd)
        iters, err = fcg_eval(make_np_precond(model))
        chosen = (step, sd, iters, err)
        print("WARNING: no checkpoint met the criteria; exporting the last one")

    step, sd, iters, err = chosen
    model.load_state_dict(sd)
    out = REPO / "cg-explorer" / "npo_weights.js"
    size = export_js(model, out, {"train_step": step, "fcg_iters_at_export": iters})
    nparams = sum(p.numel() for p in model.parameters())
    print(f"exported checkpoint step {step} ({nparams} params, {size} bytes) -> {out}")
    print(f"FCG with exported net: {iters} iterations, err at stop {100*err:.1f}% (plain CG: ~100 iters, ~21% err)")


if __name__ == "__main__":
    main()
