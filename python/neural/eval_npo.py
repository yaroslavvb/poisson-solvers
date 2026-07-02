"""Evaluate the trained NPO preconditioner on the canonical Poisson problem.

Solves ``A x = b`` with ``A = poisson_2d(32)`` and ``b = grf_rhs(32)``
(seed 42 — held out from the training seeds 100+) three ways:

* plain CG (no preconditioner) — the baseline iteration count;
* flexible PCG (Notay 2000, Polak-Ribiere beta) with the NPO — the correct
  solver for a nonlinear preconditioner (Li et al., arXiv:2502.01337);
* plain PCG with the NPO — recorded deliberately: M_theta contains ReLUs,
  so it is not a fixed SPD matrix and the Fletcher-Reeves beta of classical
  PCG loses conjugacy; we log what actually happens.

Prints a comparison table, saves ``results/npo_eval.json`` and
``figures/npo_convergence.png``. Run from the repo root:
``uv run python python/neural/eval_npo.py``.
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

from pcg import flexible_pcg, pcg
from poisson import grf_rhs, poisson_2d
from npo import NPOPreconditioner

REPO_ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-10
MAXITER = 2000


def main():
    n = 32
    a = poisson_2d(n)
    b = grf_rhs(n, alpha=2.0, tau=3.0, seed=42)
    x_direct = spla.spsolve(a.tocsc(), b)
    npo = NPOPreconditioner()

    runs = {
        "cg_none": pcg(a, b, M=None, tol=TOL, maxiter=MAXITER),
        "npo_fcg": flexible_pcg(a, b, npo, tol=TOL, maxiter=MAXITER),
        "npo_pcg": pcg(a, b, M=npo, tol=TOL, maxiter=MAXITER),
    }
    labels = {
        "cg_none": "CG (no preconditioner)",
        "npo_fcg": "NPO + flexible PCG",
        "npo_pcg": "NPO + plain PCG",
    }

    summary = {}
    print(f"{'solver':<26} {'iters':>6} {'final relres':>14} "
          f"{'relerr vs spsolve':>18} {'converged':>10}")
    for key, (x, hist) in runs.items():
        iters = len(hist) - 1
        relres = float(hist[-1])
        relerr = float(np.linalg.norm(x - x_direct) / np.linalg.norm(x_direct))
        converged = relres <= TOL
        summary[key] = {
            "iterations": iters,
            "final_relres": relres,
            "relerr_vs_spsolve": relerr,
            "converged": bool(converged),
        }
        print(f"{labels[key]:<26} {iters:>6} {relres:>14.3e} "
              f"{relerr:>18.3e} {str(converged):>10}")

    it_none = summary["cg_none"]["iterations"]
    it_fcg = summary["npo_fcg"]["iterations"]
    speedup = it_none / it_fcg if summary["npo_fcg"]["converged"] else 0.0
    summary["speedup_fcg_vs_cg"] = speedup
    summary["tol"] = TOL
    summary["maxiter"] = MAXITER
    summary["problem"] = "poisson_2d(32), b = grf_rhs(32, alpha=2, tau=3, seed=42)"
    print(f"\nNPO-FCG speedup over plain CG: {speedup:.2f}x "
          f"({it_none} -> {it_fcg} iterations)")

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "npo_eval.json", "w") as f:
        json.dump(summary, f, indent=2)

    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {"cg_none": "-", "npo_fcg": "--", "npo_pcg": ":"}
    for key, (_, hist) in runs.items():
        its = len(hist) - 1
        ax.semilogy(hist, styles[key], label=f"{labels[key]}, {its} its")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|r_k\| / \|b\|$")
    ax.set_title(f"NPO preconditioning, 2-D Poisson {n}x{n}, GRF RHS")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "npo_convergence.png", dpi=150)
    plt.close(fig)

    print("saved results/npo_eval.json and figures/npo_convergence.png")


if __name__ == "__main__":
    main()
