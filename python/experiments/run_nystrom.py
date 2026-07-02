"""Nystrom-PCG experiment: 32x32 2-D Poisson, GRF RHS, ranks [16, 64, 128, 256].

For each sketch size, builds the randomized Nystrom preconditioner
(Frangella, Tropp & Udell, arXiv:2110.02820, Alg. 2.1 + Eq. 5.3) at
``mu = 0`` (valid: A is positive definite), runs PCG to relative residual
1e-10, and computes the EXACT condition number of the preconditioned system
by dense eigensolves at N = 1024.

Honest expectation (see also the note in :mod:`nystrom`): the 2-D
Laplacian's spectrum decays SLOWLY from the top — the adversarial case for
Nystrom preconditioning, which the paper aims at fast-decay / regularized
problems with small effective dimension d_eff(mu). Deflating the top
``rank`` of N = 1024 modes only lowers kappa from lam_max/lam_min to
roughly lam_{rank+1}/lam_min, so expect modest speedups that grow with rank.

Run from the repo root: ``uv run python python/experiments/run_nystrom.py``.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nystrom import NystromPreconditioner
from pcg import pcg
from poisson import grf_rhs, poisson_2d

RANKS = [16, 64, 128, 256]
SPECTRUM_RANK = 128  # rank shown in the eigenvalue-comparison figure


def preconditioned_spectrum(pre, a_dense):
    """Exact eigenvalues of the symmetric preconditioned matrix.

    We use the symmetric form ``P^{-1/2} A P^{-1/2}`` rather than
    ``eig(P^{-1} A)``: the eigendecomposition of ``P^{-1}`` is known in
    closed form (eigenvalue ``(lam_ell + mu)/(lam_i + mu)`` on the i-th
    column of U, 1 on ``range(U)^perp``), so its principal square root is
    exact — ``P^{-1/2} = I + U (sqrt(s) - 1) U^T`` with
    ``s_i = (lam_ell + mu)/(lam_i + mu)`` — and ``eigvalsh`` on the
    symmetrized product returns guaranteed-real eigenvalues, avoiding the
    spurious imaginary parts of a nonsymmetric ``eig``.

    Parameters
    ----------
    pre : NystromPreconditioner
    a_dense : numpy.ndarray
        Dense symmetric ``A`` (plus ``mu*I`` already folded in if used).

    Returns
    -------
    numpy.ndarray
        Ascending eigenvalues of ``P^{-1/2} A P^{-1/2}``.
    """
    u = pre.U[:, : pre.rank_eff]
    s = (pre.lam_ell + pre.mu) / (pre.lams[: pre.rank_eff] + pre.mu)
    p_inv_half = np.eye(a_dense.shape[0]) + (u * (np.sqrt(s) - 1.0)) @ u.T
    sym = p_inv_half @ a_dense @ p_inv_half
    return np.linalg.eigvalsh(0.5 * (sym + sym.T))


def main():
    n = 32
    A = poisson_2d(n)
    b = grf_rhs(n, alpha=2.0, tau=3.0, seed=42)
    a_dense = A.toarray()

    eigs_a = np.linalg.eigvalsh(a_dense)
    kappa_a = float(eigs_a[-1] / eigs_a[0])
    print(f"kappa(A) unpreconditioned: {kappa_a:.2f}")

    x_ref, res_none = pcg(A, b, M=None)
    it_none = len(res_none) - 1
    print(f"CG (no preconditioner): {it_none} iterations, "
          f"final relres = {res_none[-1]:.3e}")

    ranks_out = {}
    histories = {}
    spectrum_eigs = None
    for rank in RANKS:
        pre = NystromPreconditioner(A, rank, mu=0.0, seed=0)
        x, res = pcg(A, b, M=pre)
        its = len(res) - 1
        histories[rank] = res

        eigs_p = preconditioned_spectrum(pre, a_dense)
        kappa_p = float(eigs_p[-1] / eigs_p[0])
        if rank == SPECTRUM_RANK:
            spectrum_eigs = eigs_p

        # Reference: the OPTIMAL rank-ell preconditioner (exact top-ell
        # eigenvectors, arXiv:2110.02820 Sec. 5.1) achieves exactly
        # kappa = lam_{ell+1}/lam_min — a lower bound on what any rank-ell
        # Nystrom preconditioner can do. Here lam_257/lam_min is still ~298
        # (vs kappa(A) = 441): the flat top of the Laplacian spectrum, not
        # the randomized sketch, is what caps the achievable gain.
        kappa_opt = float(eigs_a[::-1][rank] / eigs_a[0])

        err = float(np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref))
        ranks_out[str(rank)] = {
            "iterations": its,
            "kappa_precond": kappa_p,
            "kappa_optimal_rank_ell": kappa_opt,
        }
        print(f"Nystrom rank {rank:4d}: {its:3d} iterations, "
              f"kappa_precond = {kappa_p:8.2f} (optimal rank-ell "
              f"{kappa_opt:7.2f}), final relres = {res[-1]:.3e}, "
              f"relerr vs CG solution = {err:.3e}")

    root = Path(__file__).resolve().parents[2]
    (root / "figures").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)

    # Convergence curves.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogy(res_none, "k", label=f"CG (none), {it_none} its")
    for rank in RANKS:
        res = histories[rank]
        ax.semilogy(res, label=f"Nystrom rank {rank}, {len(res) - 1} its")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|r_k\| / \|b\|$")
    ax.set_title(f"2-D Poisson {n}x{n}: Nystrom-PCG convergence (mu = 0)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "figures" / "nystrom_convergence.png", dpi=150)
    plt.close(fig)

    # Spectrum comparison for rank 128: A vs preconditioned system.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.semilogy(np.sort(eigs_a)[::-1], label=r"$\lambda_j(A)$")
    ax.semilogy(np.sort(spectrum_eigs)[::-1],
                label=rf"$\lambda_j(P^{{-1/2}} A P^{{-1/2}})$, rank {SPECTRUM_RANK}")
    ax.set_xlabel("index $j$ (descending)")
    ax.set_ylabel("eigenvalue")
    ax.set_title(f"2-D Poisson {n}x{n}: spectrum before/after Nystrom "
                 f"preconditioning (rank {SPECTRUM_RANK})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "figures" / "nystrom_spectrum.png", dpi=150)
    plt.close(fig)

    summary = {
        "problem": {
            "n": n,
            "size": n * n,
            "mu": 0.0,
            "tol": 1e-10,
            "rhs": "grf_rhs(32, alpha=2.0, tau=3.0, seed=42)",
        },
        "kappa_unpreconditioned": kappa_a,
        "unpreconditioned_iterations": it_none,
        "ranks": ranks_out,
        "note": (
            "The 2-D Laplacian's spectrum decays slowly from the top "
            "(adversarial for Nystrom, cf. arXiv:2110.02820 which targets "
            "fast-decay/regularized problems), so kappa improves only as "
            "lam_{rank+1}/lam_min and speedups are modest, growing with rank."
        ),
    }
    with open(root / "results" / "nystrom.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved results/nystrom.json and figures/nystrom_*.png under {root}")


if __name__ == "__main__":
    main()
