"""Consolidated benchmark: every preconditioner on the canonical problems.

Runs the full method matrix on the canonical 32x32 Poisson problem
(``A = poisson_2d(32)``, ``b = grf_rhs(32)``, tol 1e-10, maxiter 2000):

* CG (none), CG (Jacobi), CG (ILU, ``spilu`` defaults);
* CG (Nystrom, ranks 16/64/128/256; Frangella-Tropp-Udell arXiv:2110.02820);
* flexible PCG (Notay 2000) and plain PCG with the trained NPO
  (Li et al., arXiv:2502.01337).

Plus the variable-coefficient problem ``variable_poisson_2d(32, 100)``
with CG (none) vs CG (Jacobi), where the jumping diagonal makes Jacobi
genuinely useful (on the constant-coefficient problem it is a scalar
multiple of the identity and PCG is scale-invariant in M).

Sanity checks (asserted; the Nystrom strict-monotonicity check is
report-only since randomized-sketch iteration counts may tie):

1. Jacobi iterations == plain-CG iterations on the constant-coefficient
   problem;
2. Jacobi iterations <  plain-CG iterations on the variable-coefficient
   problem;
3. Nystrom iterations decrease with rank (strict per-step check reported;
   asserted: non-increasing and strictly lower at rank 256 than rank 16);
4. every converged solve matches ``spsolve`` to 1e-8 relative error.

Outputs: results/results.json, figures/all_convergence.png.

Run from the repo root: ``uv run python python/experiments/run_all.py``.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))

from npo import NPOPreconditioner
from nystrom import NystromPreconditioner
from pcg import flexible_pcg, pcg
from poisson import grf_rhs, poisson_2d, variable_poisson_2d
from preconditioners import ilu, jacobi

TOL = 1e-10
MAXITER = 2000
NYSTROM_RANKS = [16, 64, 128, 256]


def run_method(solver, A, b, M, x_direct, setup_time_s=0.0):
    """Time one solve and package the standard result record.

    Parameters
    ----------
    solver : callable
        ``pcg`` or ``flexible_pcg``.
    A, b : system matrix and right-hand side.
    M : callable or None
        Preconditioner.
    x_direct : numpy.ndarray
        Reference ``spsolve`` solution for the error check.
    setup_time_s : float, optional
        Preconditioner construction time, recorded separately from the
        solve wall time.

    Returns
    -------
    record : dict
        ``{iterations, final_relres, wall_time_s, setup_time_s,
        relerr_vs_spsolve, converged}``.
    res_hist : list of float
        Relative-residual history (for the convergence figure).
    """
    t0 = time.perf_counter()
    x, res = solver(A, b, M=M, tol=TOL, maxiter=MAXITER)
    wall = time.perf_counter() - t0
    relerr = float(np.linalg.norm(x - x_direct) / np.linalg.norm(x_direct))
    record = {
        "iterations": len(res) - 1,
        "final_relres": float(res[-1]),
        "wall_time_s": wall,
        "setup_time_s": setup_time_s,
        "relerr_vs_spsolve": relerr,
        "converged": bool(res[-1] <= TOL),
    }
    return record, res


def main():
    n = 32
    root = Path(__file__).resolve().parents[2]
    (root / "figures").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)

    results = {"canonical": {}, "variable": {}}
    histories = {}

    # ------------------------------------------------------------------
    # Canonical constant-coefficient problem.
    # ------------------------------------------------------------------
    A = poisson_2d(n)
    b = grf_rhs(n, alpha=2.0, tau=3.0, seed=42)
    x_direct = spla.spsolve(A.tocsc(), b)

    results["canonical"]["cg_none"], histories["cg_none"] = run_method(
        pcg, A, b, None, x_direct
    )
    results["canonical"]["cg_jacobi"], histories["cg_jacobi"] = run_method(
        pcg, A, b, jacobi(A), x_direct
    )

    t0 = time.perf_counter()
    m_ilu = ilu(A)
    t_ilu = time.perf_counter() - t0
    results["canonical"]["cg_ilu"], histories["cg_ilu"] = run_method(
        pcg, A, b, m_ilu, x_direct, setup_time_s=t_ilu
    )

    for rank in NYSTROM_RANKS:
        t0 = time.perf_counter()
        pre = NystromPreconditioner(A, rank, mu=0.0, seed=0)
        t_nys = time.perf_counter() - t0
        key = f"cg_nystrom_rank{rank}"
        results["canonical"][key], histories[key] = run_method(
            pcg, A, b, pre, x_direct, setup_time_s=t_nys
        )

    t0 = time.perf_counter()
    npo = NPOPreconditioner()
    t_npo = time.perf_counter() - t0
    results["canonical"]["fcg_npo"], histories["fcg_npo"] = run_method(
        flexible_pcg, A, b, npo, x_direct, setup_time_s=t_npo
    )
    results["canonical"]["cg_npo"], histories["cg_npo"] = run_method(
        pcg, A, b, npo, x_direct, setup_time_s=t_npo
    )

    # ------------------------------------------------------------------
    # Variable-coefficient problem: Jacobi becomes nontrivial.
    # ------------------------------------------------------------------
    Av = variable_poisson_2d(n, contrast=100.0)
    xv_direct = spla.spsolve(Av.tocsc(), b)
    results["variable"]["cg_none"], _ = run_method(pcg, Av, b, None, xv_direct)
    results["variable"]["cg_jacobi"], _ = run_method(
        pcg, Av, b, jacobi(Av), xv_direct
    )

    # ------------------------------------------------------------------
    # Report table.
    # ------------------------------------------------------------------
    for problem, methods in results.items():
        print(f"\n=== {problem} problem "
              f"({'poisson_2d(32)' if problem == 'canonical' else 'variable_poisson_2d(32, contrast=100)'}) ===")
        print(f"{'method':<22} {'iters':>6} {'final relres':>14} "
              f"{'wall [s]':>10} {'setup [s]':>10} {'relerr':>10}")
        for name, rec in methods.items():
            print(f"{name:<22} {rec['iterations']:>6} "
                  f"{rec['final_relres']:>14.3e} {rec['wall_time_s']:>10.4f} "
                  f"{rec['setup_time_s']:>10.4f} "
                  f"{rec['relerr_vs_spsolve']:>10.3e}")

    # ------------------------------------------------------------------
    # Sanity checks.
    # ------------------------------------------------------------------
    can, var = results["canonical"], results["variable"]
    nys_iters = [can[f"cg_nystrom_rank{r}"]["iterations"] for r in NYSTROM_RANKS]

    checks = {
        "jacobi_equals_none_constant_coeff": (
            can["cg_jacobi"]["iterations"] == can["cg_none"]["iterations"]
        ),
        "jacobi_beats_none_variable_coeff": (
            var["cg_jacobi"]["iterations"] < var["cg_none"]["iterations"]
        ),
        "nystrom_strictly_decreasing_with_rank": all(
            a > b for a, b in zip(nys_iters, nys_iters[1:])
        ),
        "nystrom_noninc_and_overall_decrease": (
            all(a >= b for a, b in zip(nys_iters, nys_iters[1:]))
            and nys_iters[-1] < nys_iters[0]
        ),
        "converged_solves_match_spsolve_1e-8": all(
            rec["relerr_vs_spsolve"] < 1e-8
            for methods in results.values()
            for rec in methods.values()
            if rec["converged"]
        ),
    }
    print("\n=== sanity checks ===")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  Nystrom iterations by rank {NYSTROM_RANKS}: {nys_iters}")
    results["sanity_checks"] = checks
    results["nystrom_iterations_by_rank"] = dict(
        zip(map(str, NYSTROM_RANKS), nys_iters)
    )

    # ------------------------------------------------------------------
    # Convergence figure: all canonical-problem methods.
    # ------------------------------------------------------------------
    styles = {
        "cg_none": ("k-", "CG (none)"),
        "cg_jacobi": ("k--", "CG (Jacobi)"),
        "cg_ilu": ("g-", "CG (ILU)"),
        "cg_nystrom_rank16": ("C0-", "CG (Nystrom 16)"),
        "cg_nystrom_rank64": ("C1-", "CG (Nystrom 64)"),
        "cg_nystrom_rank128": ("C4-", "CG (Nystrom 128)"),
        "cg_nystrom_rank256": ("C3-", "CG (Nystrom 256)"),
        "fcg_npo": ("m-", "FCG (NPO)"),
        "cg_npo": ("m:", "CG (NPO, plain)"),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, (style, label) in styles.items():
        res = histories[key]
        its = len(res) - 1
        ax.semilogy(res, style, lw=1.3,
                    label=f"{label}, {its} its"
                          + ("" if results["canonical"][key]["converged"]
                             else " (no conv)"))
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|r_k\| / \|b\|$")
    ax.set_xlim(0, 200)  # cg_npo stalls to 2000; clip for readability
    ax.set_title(f"2-D Poisson {n}x{n}, GRF RHS: all methods (tol $10^{{-10}}$)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "figures" / "all_convergence.png", dpi=150)
    plt.close(fig)

    results["config"] = {
        "n": n,
        "tol": TOL,
        "maxiter": MAXITER,
        "rhs": "grf_rhs(32, alpha=2.0, tau=3.0, seed=42)",
        "nystrom_seed": 0,
        "note_wall_time": "wall_time_s is the solve only; "
                          "setup_time_s is preconditioner construction.",
    }
    with open(root / "results" / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved results/results.json and figures/all_convergence.png "
          f"under {root}")

    # Hard assertions (Nystrom strict monotonicity intentionally excluded —
    # reported above; ties between adjacent ranks are possible).
    hard = [
        "jacobi_equals_none_constant_coeff",
        "jacobi_beats_none_variable_coeff",
        "nystrom_noninc_and_overall_decrease",
        "converged_solves_match_spsolve_1e-8",
    ]
    failed = [name for name in hard if not checks[name]]
    assert not failed, f"sanity checks failed: {failed}"


if __name__ == "__main__":
    main()
