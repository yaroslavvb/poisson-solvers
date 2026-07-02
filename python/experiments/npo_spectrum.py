"""Linearized spectral analysis of the trained NPO preconditioner.

The NPO ``M_theta`` (Li et al., arXiv:2502.01337) is a nonlinear map, so it
has no spectrum in the strict sense. We linearize it by columns: applying
the preconditioner to the N = 1024 canonical basis vectors gives a dense
matrix ``M_tilde`` with ``M_tilde[:, j] = M_theta(e_j)`` — the exact linear
map ONLY IF M_theta were linear. Because the NPOPreconditioner wrapper
normalizes inputs to unit norm (and ``||e_j|| = 1``), each column is a raw
network evaluation.

The figure of merit for PCG is eigenvalue CLUSTERING of the preconditioned
operator ``M_tilde A``: PCG iterates are invariant under positive scaling
of M, so what matters is the relative spread of the spectrum, not its
location (classical bound: iterations ~ sqrt(lam_max/lam_min) for positive
real spectra, Saad 2003, Sec. 6.11). We therefore compare the spectrum of
``M_tilde A`` against the spectrum of ``A``, both scaled by their median,
and report:

* eigenvalues of ``M_tilde A`` via dense nonsymmetric ``scipy.linalg.eig``;
* nonsymmetry of the linearization, ``||M - M^T||_F / ||M||_F``;
* nonlinearity of the NPO: relative error between ``M_tilde @ b`` and
  ``M_theta(b)`` for the canonical GRF right-hand side ``b``.

Outputs: figures/npo_spectrum.png, results/npo_spectrum.json.

Run from the repo root: ``uv run python python/experiments/npo_spectrum.py``.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neural"))

from npo import NPOPreconditioner
from poisson import grf_rhs, poisson_2d


def main():
    n = 32
    N = n * n
    A = poisson_2d(n)
    b = grf_rhs(n, alpha=2.0, tau=3.0, seed=42)
    npo = NPOPreconditioner()

    # Column-wise linearization: M_tilde[:, j] = M_theta(e_j).
    m_tilde = np.empty((N, N))
    e = np.zeros(N)
    for j in range(N):
        e[j] = 1.0
        m_tilde[:, j] = npo(e)
        e[j] = 0.0

    # Nonsymmetry of the linearization (a fixed SPD M would give 0).
    nonsym = float(
        np.linalg.norm(m_tilde - m_tilde.T) / np.linalg.norm(m_tilde)
    )

    # Nonlinearity: does the column linearization reproduce M_theta(b)?
    z_lin = m_tilde @ b
    z_npo = npo(b)
    nonlin = float(np.linalg.norm(z_lin - z_npo) / np.linalg.norm(z_npo))

    # Spectrum of the (nonsymmetric) preconditioned operator.
    eigs_ma = sla.eig(m_tilde @ A.toarray())[0]
    re, im = eigs_ma.real, eigs_ma.imag
    eigs_a = np.linalg.eigvalsh(A.toarray())

    # Clustering measures (scale-invariant, matching PCG's invariance to
    # positive rescaling of M).
    med_ma = float(np.median(np.abs(eigs_ma)))
    med_a = float(np.median(eigs_a))
    spread_ma = float(np.abs(eigs_ma).max() / np.abs(eigs_ma).min())
    spread_a = float(eigs_a[-1] / eigs_a[0])
    frac_in_band = float(
        np.mean((np.abs(eigs_ma) > 0.5 * med_ma) & (np.abs(eigs_ma) < 2.0 * med_ma))
    )
    frac_in_band_a = float(
        np.mean((eigs_a > 0.5 * med_a) & (eigs_a < 2.0 * med_a))
    )

    print(f"nonsymmetry ||M - M^T||_F / ||M||_F        = {nonsym:.4f}")
    print(f"nonlinearity |M_tilde b - NPO(b)|/|NPO(b)| = {nonlin:.4f}")
    print(f"eig(M_tilde A): Re in [{re.min():.4f}, {re.max():.4f}], "
          f"max |Im| = {np.abs(im).max():.4f}, "
          f"negative real parts: {int(np.sum(re <= 0))}")
    print(f"spread max|lam|/min|lam|:  M_tilde A = {spread_ma:8.2f}   "
          f"A = {spread_a:8.2f}")
    print(f"fraction of eigenvalues within [0.5, 2] x median: "
          f"M_tilde A = {frac_in_band:.3f}   A = {frac_in_band_a:.3f}")

    root = Path(__file__).resolve().parents[2]
    (root / "figures").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)

    # Figure: complex-plane scatter + median-scaled clustering histogram.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.scatter(re, im, s=8, alpha=0.6)
    ax1.axvline(0.0, color="k", lw=0.5)
    ax1.set_xlabel(r"Re $\lambda$")
    ax1.set_ylabel(r"Im $\lambda$")
    ax1.set_title(r"eig($\tilde{M} A$) in the complex plane")
    ax1.grid(True, alpha=0.3)

    bins = np.logspace(-2, 2, 60)  # in units of the median
    ax2.hist(np.abs(eigs_ma) / med_ma, bins=bins, alpha=0.6,
             label=r"$|\lambda(\tilde{M} A)|$ / median")
    ax2.hist(eigs_a / med_a, bins=bins, alpha=0.6,
             label=r"$\lambda(A)$ / median")
    ax2.set_xscale("log")
    ax2.set_xlabel("eigenvalue / median (scale-invariant)")
    ax2.set_ylabel("count")
    ax2.set_title("clustering: NPO-preconditioned vs raw spectrum")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"NPO linearized spectrum, 2-D Poisson {n}x{n}")
    fig.tight_layout()
    fig.savefig(root / "figures" / "npo_spectrum.png", dpi=150)
    plt.close(fig)

    summary = {
        "n": n,
        "size": N,
        "nonsymmetry_relfro": nonsym,
        "nonlinearity_relerr_on_grf_b": nonlin,
        "eig_MA": {
            "re_min": float(re.min()),
            "re_max": float(re.max()),
            "abs_im_max": float(np.abs(im).max()),
            "num_nonpositive_real": int(np.sum(re <= 0)),
            "abs_min": float(np.abs(eigs_ma).min()),
            "abs_max": float(np.abs(eigs_ma).max()),
            "abs_median": med_ma,
            "spread_absmax_over_absmin": spread_ma,
            "frac_within_half_to_2x_median": frac_in_band,
        },
        "eig_A": {
            "min": float(eigs_a[0]),
            "max": float(eigs_a[-1]),
            "median": med_a,
            "spread_max_over_min": spread_a,
            "frac_within_half_to_2x_median": frac_in_band_a,
        },
    }
    with open(root / "results" / "npo_spectrum.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved results/npo_spectrum.json and figures/npo_spectrum.png "
          f"under {root}")


if __name__ == "__main__":
    main()
