"""Baseline experiment: 32x32 2-D Poisson, GRF right-hand side, CG vs Jacobi-PCG.

Python port of the Mathematica reference notebook (see repo README):
builds ``A = poisson_2d(32)`` and ``b = grf_rhs(32)``, solves with plain CG
and Jacobi-PCG, verifies against a direct sparse solve, and saves
convergence/field figures plus a JSON summary.

Expected finding: ``diag(A) = 4/h^2`` is constant for the Dirichlet
Laplacian, so Jacobi is a positive scalar multiple of the identity; PCG is
invariant under positive scaling of the preconditioner, hence Jacobi-PCG
converges iteration-identically to plain CG (up to float rounding).

Run from the repo root: ``uv run python python/experiments/run_baseline.py``.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcg import pcg
from poisson import grf_rhs, poisson_2d
from preconditioners import jacobi


def main():
    n = 32
    A = poisson_2d(n)
    b = grf_rhs(n, alpha=2.0, tau=3.0, seed=42)

    x_none, res_none = pcg(A, b, M=None)
    x_jac, res_jac = pcg(A, b, M=jacobi(A))

    it_none = len(res_none) - 1
    it_jac = len(res_jac) - 1
    print(f"CG (no preconditioner): {it_none} iterations, "
          f"final relres = {res_none[-1]:.3e}")
    print(f"Jacobi-PCG:             {it_jac} iterations, "
          f"final relres = {res_jac[-1]:.3e}")

    # Verify against direct sparse solve.
    x_direct = spla.spsolve(A.tocsc(), b)
    err_none = np.linalg.norm(x_none - x_direct) / np.linalg.norm(x_direct)
    err_jac = np.linalg.norm(x_jac - x_direct) / np.linalg.norm(x_direct)
    print(f"relative error vs spsolve: CG {err_none:.3e}, Jacobi {err_jac:.3e}")

    # Jacobi = (h^2/4) * I here, so the residual histories must coincide
    # up to float rounding.
    m = min(len(res_none), len(res_jac))
    max_dev = float(np.max(np.abs(np.array(res_none[:m]) - np.array(res_jac[:m]))))
    print(f"max |res_none - res_jacobi| over common iterations: {max_dev:.3e}")

    root = Path(__file__).resolve().parents[2]
    (root / "figures").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)

    # Convergence curves (log-scale residuals).
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(res_none, label=f"CG (none), {it_none} its")
    ax.semilogy(res_jac, "--", label=f"Jacobi-PCG, {it_jac} its")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|r_k\| / \|b\|$")
    ax.set_title(f"2-D Poisson {n}x{n}, GRF RHS: PCG convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "figures" / "baseline_convergence.png", dpi=150)
    plt.close(fig)

    # GRF right-hand side and solution fields.
    for field, fname, title in [
        (b, "grf_field.png", f"GRF right-hand side (alpha=2, tau=3, {n}x{n})"),
        (x_none, "solution_field.png", f"Poisson solution ({n}x{n})"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(field.reshape(n, n))
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(root / "figures" / fname, dpi=150)
        plt.close(fig)

    summary = {
        "none": {"iterations": it_none, "final_relres": float(res_none[-1])},
        "jacobi": {"iterations": it_jac, "final_relres": float(res_jac[-1])},
        "note": (
            "diag(A) = 4/h^2 is constant for the constant-coefficient "
            "Dirichlet Laplacian, so Jacobi is a positive scalar multiple of "
            "the identity; PCG is invariant to positive scaling of M, hence "
            "Jacobi-PCG matches plain CG iteration-for-iteration (max "
            f"residual-history deviation {max_dev:.3e})."
        ),
        "verification": {
            "relerr_vs_spsolve_none": float(err_none),
            "relerr_vs_spsolve_jacobi": float(err_jac),
        },
    }
    with open(root / "results" / "baseline.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved results/baseline.json and figures/*.png under {root}")


if __name__ == "__main__":
    main()
