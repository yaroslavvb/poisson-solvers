"""Spectral verification of the discrete Poisson operators.

Checks the computed spectra of ``laplacian_1d`` and ``poisson_2d`` against
the closed-form eigenvalues of the Dirichlet second-difference stencil
(cf. LeVeque, "Finite Difference Methods for ODEs and PDEs", SIAM 2007,
Sec. 2.10)::

    lam_k(d1)          = 2 - 2 cos(k pi h),     k = 1..n,  h = 1/(n+1)
    lam_{k,l}(A)       = (lam_k + lam_l) / h^2
    kappa(A)           = lam_{n,n} / lam_{1,1}
                       = (1 - cos(n pi h)) / (1 - cos(pi h))
                       = sin^2(n pi h / 2) / sin^2(pi h / 2)   ~  4(n+1)^2/pi^2

The exact ratio (not the asymptote) is used everywhere; the ``O(n^2)``
growth of kappa is demonstrated on n in [8, 16, 32, 64, 128] with the
analytic formula alone.

Outputs: figures/d1_eigenvalues.png, figures/A_spectrum.png,
figures/kappa_scaling.png, results/spectra.json.

Run from the repo root: ``uv run python python/experiments/spectra.py``.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poisson import laplacian_1d, poisson_2d

KAPPA_NS = [8, 16, 32, 64, 128]


def d1_eigs_analytic(n):
    """Exact eigenvalues 2 - 2 cos(k pi/(n+1)), k = 1..n, ascending."""
    k = np.arange(1, n + 1)
    return 2.0 - 2.0 * np.cos(k * np.pi / (n + 1))


def kappa_analytic(n):
    """Exact kappa of poisson_2d(n): sin^2(n pi h/2) / sin^2(pi h/2)."""
    h = 1.0 / (n + 1)
    return float(np.sin(n * np.pi * h / 2.0) ** 2 / np.sin(np.pi * h / 2.0) ** 2)


def main():
    n = 32
    h = 1.0 / (n + 1)
    root = Path(__file__).resolve().parents[2]
    (root / "figures").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)

    # (a) 1-D stencil: computed vs analytic.
    d1 = laplacian_1d(n)
    eigs_d1 = np.linalg.eigvalsh(d1.toarray())  # ascending
    eigs_d1_ana = d1_eigs_analytic(n)  # ascending in k
    dev_d1 = float(np.max(np.abs(eigs_d1 - eigs_d1_ana)))
    print(f"d1 (n={n}): max |computed - analytic| = {dev_d1:.3e}")

    # (b) 2-D operator: computed vs analytic tensor sums (lam_k + lam_l)/h^2.
    A = poisson_2d(n)
    eigs_a = np.linalg.eigvalsh(A.toarray())  # ascending
    eigs_a_ana = np.sort(
        (eigs_d1_ana[:, None] + eigs_d1_ana[None, :]).ravel() / h**2
    )
    dev_a = float(np.max(np.abs(eigs_a - eigs_a_ana)))
    print(f"A  (n={n}): max |computed - analytic| = {dev_a:.3e}")

    kappa_comp = float(eigs_a[-1] / eigs_a[0])
    kappa_ana = kappa_analytic(n)
    print(f"kappa(A) computed  = {kappa_comp:.6f}")
    print(f"kappa(A) analytic  = {kappa_ana:.6f}  "
          f"(exact sin^2 ratio; asymptote 4(n+1)^2/pi^2 = "
          f"{4 * (n + 1) ** 2 / np.pi ** 2:.2f})")

    # (c) kappa vs n from the exact analytic formula: O(n^2) growth.
    kappa_vs_n = {m: kappa_analytic(m) for m in KAPPA_NS}
    print("kappa vs n (analytic):")
    for m, kap in kappa_vs_n.items():
        print(f"  n = {m:4d}: kappa = {kap:12.2f}   kappa/n^2 = {kap / m**2:.4f}")

    # (d1) 1-D eigenvalues: computed dots, analytic curve, continuum parabola.
    k = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(k, eigs_d1, "o", ms=4, label="computed (eigvalsh)")
    ax.plot(k, eigs_d1_ana, "-", lw=1.5,
            label=r"analytic $2 - 2\cos(k\pi h)$")
    ax.plot(k, (k * np.pi * h) ** 2, "--", lw=1.5,
            label=r"continuum $(k\pi h)^2$")
    ax.set_xlabel("mode index $k$")
    ax.set_ylabel("eigenvalue of $d_1$ (no $1/h^2$)")
    ax.set_title(f"1-D Dirichlet stencil eigenvalues, n = {n}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "figures" / "d1_eigenvalues.png", dpi=150)
    plt.close(fig)

    # (d2) 2-D sorted spectrum, computed vs analytic.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    idx = np.arange(1, n * n + 1)
    ax.semilogy(idx, eigs_a, lw=1.5, label="computed (eigvalsh)")
    ax.semilogy(idx, eigs_a_ana, "--", lw=1.0,
                label=r"analytic $(\lambda_k + \lambda_l)/h^2$")
    ax.set_xlabel("index (ascending)")
    ax.set_ylabel("eigenvalue of $A$")
    ax.set_title(f"2-D Poisson spectrum, n = {n} (N = {n * n}), "
                 f"$\\kappa$ = {kappa_comp:.1f}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "figures" / "A_spectrum.png", dpi=150)
    plt.close(fig)

    # (d3) kappa scaling: loglog with n^2 reference slope.
    ns = np.array(KAPPA_NS, dtype=float)
    kappas = np.array([kappa_vs_n[m] for m in KAPPA_NS])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(ns, kappas, "o-", label=r"$\kappa(A)$ (exact analytic)")
    ax.loglog(ns, kappas[-1] * (ns / ns[-1]) ** 2, "k--",
              label=r"$\propto n^2$ reference")
    ax.set_xlabel("grid size $n$")
    ax.set_ylabel(r"$\kappa(A)$")
    ax.set_title(r"Condition number growth: $\kappa \sim 4(n+1)^2/\pi^2$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(root / "figures" / "kappa_scaling.png", dpi=150)
    plt.close(fig)

    # (e) JSON summary.
    summary = {
        "n": n,
        "d1_max_abs_dev": dev_d1,
        "A_max_abs_dev": dev_a,
        "A_lam_min_computed": float(eigs_a[0]),
        "A_lam_max_computed": float(eigs_a[-1]),
        "A_lam_min_analytic": float(eigs_a_ana[0]),
        "A_lam_max_analytic": float(eigs_a_ana[-1]),
        "kappa_computed": kappa_comp,
        "kappa_analytic_exact": kappa_ana,
        "kappa_asymptotic_4n1sq_over_pisq": float(4 * (n + 1) ** 2 / np.pi**2),
        "kappa_vs_n_analytic": {str(m): kappa_vs_n[m] for m in KAPPA_NS},
        "kappa_over_nsq": {str(m): kappa_vs_n[m] / m**2 for m in KAPPA_NS},
    }
    with open(root / "results" / "spectra.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved results/spectra.json and figures/{{d1_eigenvalues,"
          f"A_spectrum,kappa_scaling}}.png under {root}")


if __name__ == "__main__":
    main()
