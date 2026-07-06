"""Grid regressions and multiscale preconditioning: anatomy at n=8, solvers at n=32.

PART A (n=8, N=64, everything dense and inspectable), notation of reports
09/10: h = 1/(n+1), A = Kronecker-sum Laplacian / h^2 (poisson_2d), Sigma =
A^{-1} the covariance of the Gibbs field u ~ N(0, A^{-1}), B = I - D^{-1}A
the two-sided regression matrix (= Jacobi iteration matrix), phiL2R =
sequential regression on predecessors (modified Cholesky of Sigma), phiR2L =
regression on successors (= chol(A)), reversal identity
chol(A^{-1}) = P L^{-T} P (P = index reversal = 180-degree grid rotation).

  1. Green's function / dense A^{-1}: M-matrix positivity, symmetry, the
     center-column Green's bump; block norms; long-range correlation number.
  2. Neumann / random-walk path expansion A^{-1} = (sum_k B^k) D^{-1} at
     geometric rate rho(B) = cos(pi h) -- the many-paths story of report 10.
  3. Two-sided regression: rows of B are the 1/4 stencil averages;
     conditional variance 1/A_ii = h^2/4; notebook identity A = (I-B) D.
  4. One-sided regressions: L = chol(A) as regression-on-successors
     (u_i on u_j, j>i, coefficients -L[j,i]/L[i,i], innovation sd 1/L_ii);
     bandwidth-n fill; the wavefront coefficient profile; reversal identity;
     the 180-degree rotation automorphism making L2R and R2L mirror images.
  5. IC(0) = zero-fill incomplete Cholesky = truncated regression on the
     stencil successors only; Vecchia cross-check against the exact-Sigma
     truncated regressions (Schafer-Katzfuss-Owhadi).
  6. Multiscale regressors: the global mean and 2x2 block averages as
     least-squares regressors for the smooth component; residual correlation
     collapse -- the statistical case for two-level preconditioning.

PART B (n=32, N=1024): hot/cold-rod problem, PCG with none / IC(0) /
coarse-only / two-level (Jacobi+coarse) / two-level (IC(0)+coarse);
iteration counts, dense kappa(M^{-1}A), and error-field pedagogy.

Run from the repo root:
    uv run python python/experiments/grid_regressions_multiscale.py

Deterministic (no sampling); PASS/FAIL style follows
verify_statistical_identities.py. Figures -> figures/ (dpi=150), numbers ->
results/grid_multiscale.json.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcg import pcg
from poisson import poisson_2d
from preconditioners import block_average_matrix, ic0

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"
RESDIR = ROOT / "results"
np.set_printoptions(precision=4, suppress=True)

RESULTS = {}
N_FAIL = 0


def ok(name, cond):
    global N_FAIL
    if not cond:
        N_FAIL += 1
    print(f"{'PASS' if cond else 'FAIL'}: {name}")
    return bool(cond)


def info(msg):
    print(f"  [info] {msg}")


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


# ---------------------------------------------------------------------------
# IC(0) and block_average_matrix moved to python/preconditioners.py (shared
# with experiments/richardson_ar.py); imported above.
# ---------------------------------------------------------------------------
def kappa_from_Minv(Ad, Minv):
    """kappa(M^{-1}A) for SPD Minv: eig(Minv A) = eig(R^T A R), R = chol(Minv)."""
    R = np.linalg.cholesky(Minv)
    w = np.linalg.eigvalsh(R.T @ Ad @ R)
    return w[-1] / w[0], w[0], w[-1]


def add_block_grid(ax, N, nblk, **kw):
    for t in range(1, nblk):
        ax.axhline(t * (N // nblk) - 0.5, **kw)
        ax.axvline(t * (N // nblk) - 0.5, **kw)


# ===========================================================================
# PART A -- ANATOMY at n = 8
# ===========================================================================
t0 = time.time()
print("== PART A: anatomy at n=8 (N=64, lexicographic k = i*n + j) ==")
n8 = 8
N8 = n8 * n8
h8 = 1.0 / (n8 + 1)
A8s = poisson_2d(n8)
A8 = A8s.toarray()
S8 = np.linalg.inv(A8)  # covariance Sigma = A^{-1}
partA = {"n": n8, "N": N8, "h": h8}

# ---- 1. Green's function / inverse -----------------------------------------
kc = 3 * n8 + 3  # center-ish node (3,3)
col = S8[:, kc].reshape(n8, n8)
ok("A^{-1} elementwise positive (M-matrix)", S8.min() > 0)
ok("A^{-1} symmetric", np.allclose(S8, S8.T))
mono_row = all(col[3, j] > col[3, j + 1] for j in range(3, n8 - 1)) and all(
    col[3, j] < col[3, j + 1] for j in range(0, 3)
)
mono_col = all(col[i, 3] > col[i + 1, 3] for i in range(3, n8 - 1)) and all(
    col[i, 3] < col[i + 1, 3] for i in range(0, 3)
)
ok(
    "center column (node (3,3)) is a Green's bump: max at source, monotone decay along its row and column",
    np.unravel_index(col.argmax(), col.shape) == (3, 3) and mono_row and mono_col,
)
nn_cov = S8[0, 1]  # corner (0,0) with nearest neighbor (0,1)
corner_cov = S8[0, N8 - 1]  # corner (0,0) with opposite corner (7,7)
block_norms = np.array(
    [
        [np.linalg.norm(S8[8 * bi : 8 * bi + 8, 8 * bj : 8 * bj + 8]) for bj in range(8)]
        for bi in range(8)
    ]
)
block_mono = all(
    all(block_norms[bi, bj] > block_norms[bi, bj + 1] for bj in range(bi, n8 - 1))
    and all(block_norms[bi, bj] < block_norms[bi, bj + 1] for bj in range(0, bi))
    for bi in range(n8)
)
ok("A^{-1} block Frobenius norms decay monotonically away from the diagonal in every block row",
   bool(block_mono))
info(
    f"max(Ainv)={S8.max():.6e} (at diag, center), min(Ainv)={S8.min():.6e} (corner-to-opposite-corner)"
)
info(f"nearest-neighbor/opposite-corner covariance ratio = {nn_cov/corner_cov:.1f}")
partA["greens"] = {
    "center_node": [3, 3],
    "max_entry": S8.max(),
    "min_entry": S8.min(),
    "argmax_is_diagonal": bool(
        np.unravel_index(S8.argmax(), S8.shape)[0] == np.unravel_index(S8.argmax(), S8.shape)[1]
    ),
    "nn_covariance_corner": nn_cov,
    "opposite_corner_covariance": corner_cov,
    "nn_to_opposite_corner_ratio": nn_cov / corner_cov,
    "block_frobenius_norms_8x8": block_norms,
}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
im = axes[0].imshow(S8, cmap="viridis")
add_block_grid(axes[0], N8, n8, color="w", lw=0.6, alpha=0.7)
axes[0].set_xticks(np.arange(n8) * n8 + 3.5, [f"{b}" for b in range(n8)])
axes[0].set_yticks(np.arange(n8) * n8 + 3.5, [f"{b}" for b in range(n8)])
axes[0].set_xlabel("block column bj  (grid row bj)")
axes[0].set_ylabel("block row bi  (grid row bi)")
axes[0].set_title(r"$A^{-1}$ (64x64): block $(b_i,b_j)$ couples grid-row $b_i$ to grid-row $b_j$")
fig.colorbar(im, ax=axes[0], fraction=0.046)
im = axes[1].imshow(col, cmap="viridis")
axes[1].plot(3, 3, "r+", ms=12, mew=2)
axes[1].set_title("column of node (3,3), reshaped 8x8:\ndiscrete Green's bump")
axes[1].set_xlabel("grid col j")
axes[1].set_ylabel("grid row i")
fig.colorbar(im, ax=axes[1], fraction=0.046)
fig.tight_layout()
fig.savefig(FIGDIR / "grid8_Ainv.png", dpi=150)
plt.close(fig)

# ---- 2. Random-walk / Neumann path expansion --------------------------------
D8 = np.diag(np.diag(A8))
Dinv8 = np.diag(1.0 / np.diag(A8))
B8 = np.eye(N8) - Dinv8 @ A8
rho = np.max(np.abs(np.linalg.eigvals(B8)))
rho_theory = np.cos(np.pi * h8)  # (cos(pi h)+cos(pi h))/2
ok("rho(B) == (cos(pi h)+cos(pi h))/2 == cos(pi/9)", np.isclose(rho, rho_theory, rtol=1e-12))

Kmax = 400
errs = np.empty(Kmax + 1)
Bk = np.eye(N8)
acc = np.eye(N8)
Snorm = np.linalg.norm(S8)
errs[0] = np.linalg.norm(acc @ Dinv8 - S8) / Snorm
for k in range(1, Kmax + 1):
    Bk = Bk @ B8
    acc += Bk
    errs[k] = np.linalg.norm(acc @ Dinv8 - S8) / Snorm
fit_rate = (errs[300] / errs[100]) ** (1.0 / 200)
ok("Neumann series (sum B^k) D^{-1} -> A^{-1}  (rel err < 1e-9 at K=400)", errs[-1] < 1e-9)
ok("Neumann convergence geometric at rate rho(B)  (fitted rate within 0.2%)",
   abs(fit_rate - rho) / rho < 2e-3)
info(f"rho(B)={rho:.6f}=cos(pi/9), fitted decay rate (K=100..300)={fit_rate:.6f}, "
     f"rel err at K=400: {errs[-1]:.2e}")
partA["neumann"] = {
    "rho_B": rho,
    "cos_pi_over_9": rho_theory,
    "fitted_rate_K100_300": fit_rate,
    "rel_err_K": {str(k): errs[k] for k in [0, 1, 2, 5, 10, 25, 50, 100, 200, 300, 400]},
}

fig, ax = plt.subplots(figsize=(6.4, 4.6))
K = np.arange(Kmax + 1)
ax.semilogy(K, errs, lw=1.6, label=r"$\|(\sum_{k \leq K} B^k)D^{-1} - A^{-1}\|_F / \|A^{-1}\|_F$")
ax.semilogy(K, errs[0] * rho ** K, "k--", lw=1.0, label=r"$\rho(B)^K = \cos(\pi/9)^K$ reference")
ax.set_xlabel("K (path length / # terms)")
ax.set_ylabel("relative error")
ax.set_title("Green's function as a sum over lattice random-walk paths\n"
             r"$A^{-1} = (\sum_k B^k) D^{-1}$, geometric at $\rho(B)=\cos(\pi h)$")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR / "grid8_neumann.png", dpi=150)
plt.close(fig)

# ---- 3. Two-sided regression matrix B ---------------------------------------
stencil_ok = True
for k in range(N8):
    i, j = divmod(k, n8)
    nbrs = [(i + di) * n8 + (j + dj)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= i + di < n8 and 0 <= j + dj < n8]
    row = B8[k].copy()
    if not np.allclose(row[nbrs], 0.25, atol=1e-14):
        stencil_ok = False
    row[nbrs] = 0.0
    if not np.allclose(row, 0.0, atol=1e-14):
        stencil_ok = False
ok("B rows: exactly 1/4 on the stencil neighbors (4 for interior nodes), 0 elsewhere", stencil_ok)
i_test = 3 * n8 + 4  # interior node, direct conditional-variance check
rest = np.arange(N8) != i_test
cvar = S8[i_test, i_test] - S8[i_test, rest] @ np.linalg.solve(S8[np.ix_(rest, rest)],
                                                               S8[rest, i_test])
ok("conditional variance == 1/A_ii == h^2/4",
   np.isclose(cvar, 1.0 / A8[i_test, i_test]) and np.isclose(1.0 / A8[i_test, i_test], h8**2 / 4))
ok("notebook identity A == (I - B) D", np.allclose(A8, (np.eye(N8) - B8) @ D8))
partA["two_sided"] = {"stencil_coefficient": 0.25, "conditional_variance": cvar,
                      "h2_over_4": h8**2 / 4}

fig, ax = plt.subplots(figsize=(6.6, 5.6))
im = ax.imshow(B8, cmap="RdBu_r", vmin=-0.25, vmax=0.25)
add_block_grid(ax, N8, n8, color="k", lw=0.5, alpha=0.35)
ax.set_title("two-sided regression matrix $B = I - D^{-1}A$\n"
             r"$\pm1$ bands (E/W neighbors, gaps at block boundaries) and $\pm8$ bands (N/S)")
ax.set_xlabel("node index (k = i*8 + j)")
ax.set_ylabel("node index")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(FIGDIR / "grid8_B.png", dpi=150)
plt.close(fig)

# ---- 4. One-sided regressions: L = chol(A) ----------------------------------
# Whitening z = L^T u: z_i = L_ii u_i + sum_{j>i} L_ji u_j, i.e. u_i regressed
# on its SUCCESSORS with coefficients -L[j,i]/L[i,i] (j > i), innovation sd
# 1/L_ii. This is phiR2L; phiL2R regresses on predecessors (mchol of Sigma).
L8 = np.linalg.cholesky(A8)
nz = np.abs(L8) > 1e-12
offs = np.subtract.outer(np.arange(N8), np.arange(N8))  # i - j
bandwidth = int(offs[nz].max())
patternA = np.tril(A8) != 0.0
nnz_in_pattern = int((nz & patternA).sum())
nnz_fill = int((nz & ~patternA).sum())
band_slots = int(((offs >= 1) & (offs <= n8)).sum())
nnz_pattern_strict = int((nz & patternA & (offs >= 1)).sum())
ok(f"chol(A) bandwidth exactly n=8 (max nonzero offset {bandwidth})", bandwidth == n8)
info(f"L nonzeros: {nnz_in_pattern} on A's lower pattern (incl. {N8} diagonal), {nnz_fill} fill-in; "
     f"strictly below the diagonal: {nnz_pattern_strict} pattern + {nnz_fill} fill = "
     f"{nnz_pattern_strict + nnz_fill} of {band_slots} strict-band slots — "
     "fill occupies the whole interior band")

# wavefront profile: mean |coefficient| vs lateral distance within the wavefront.
# For node k=(r,c), successors k+d (d=1..8) are the elimination wavefront:
# (r,c+1..7) at lateral distance d, then (r+1, c-(8-d)) at lateral distance
# 8-d; lateral offset 0 = the neighbor (r+1,c) directly in the next grid row
# (flat offset +8). Averaged over interior rows r=1..6, all columns (lateral
# distance 7 is only reachable from the edge columns c=0,7).
lat_sum = np.zeros(n8)
lat_cnt = np.zeros(n8, dtype=int)
off_sum = np.zeros(n8)  # by flat offset d=1..8
off_cnt = np.zeros(n8, dtype=int)
for r in range(1, 7):
    for c in range(0, n8):
        k = r * n8 + c
        for d in range(1, n8 + 1):
            coef = -L8[k + d, k] / L8[k, k]
            lat = d if c + d <= n8 - 1 else n8 - d
            lat_sum[lat] += abs(coef)
            lat_cnt[lat] += 1
            off_sum[d - 1] += abs(coef)
            off_cnt[d - 1] += 1
wavefront = lat_sum / lat_cnt
by_offset = off_sum / off_cnt
ok("wavefront profile: mean |coefficient| decays monotonically with lateral distance 0..7",
   bool(np.all(np.diff(wavefront) < 0)))
info("wavefront profile (lateral 0..7): " + ", ".join(f"{v:.4f}" for v in wavefront))
info("by flat offset d=1..8:          " + ", ".join(f"{v:.4f}" for v in by_offset))

P8 = np.eye(N8)[::-1]
C_S = np.linalg.cholesky(S8)
ok("reversal identity chol(A^{-1}) == P L^{-T} P",
   np.allclose(C_S, P8 @ np.linalg.inv(L8).T @ P8))
ok("180-degree rotation is an automorphism: P A P == A", np.allclose(P8 @ A8 @ P8, A8))

# phiL2R by direct sequential regression on predecessors (Pourahmadi mchol)
Phi = np.zeros((N8, N8))
for i in range(1, N8):
    Phi[i, :i] = np.linalg.solve(S8[:i, :i], S8[:i, i])
T = np.eye(N8) - Phi
Dm = T @ S8 @ T.T
ok("(I - phiL2R) Sigma (I - phiL2R)' diagonal (modified Cholesky of covariance)",
   np.allclose(Dm, np.diag(np.diag(Dm)), atol=1e-10))
W = np.zeros((N8, N8))  # phiR2L: successor coefficients from chol(A)
for i in range(N8 - 1):
    W[i, i + 1:] = -L8[i + 1:, i] / L8[i, i]
ok("mirror image: phiL2R == P phiR2L P (L2R and R2L coefficient sets are 180-degree rotations)",
   np.allclose(Phi, P8 @ W @ P8))
partA["cholesky"] = {
    "bandwidth": bandwidth, "nnz_on_A_pattern": nnz_in_pattern, "nnz_fill": nnz_fill,
    "nnz_on_A_pattern_strict_below_diag": nnz_pattern_strict,
    "strict_band_slots": band_slots,
    "wavefront_profile_lateral_0_7": wavefront,
    "wavefront_profile_by_flat_offset_1_8": by_offset,
}

fig, ax = plt.subplots(figsize=(6.8, 5.8))
logL = np.log10(np.abs(L8) + 1e-17)
im = ax.imshow(logL, cmap="magma", vmin=-12, vmax=logL.max())
pi, pj = np.nonzero(patternA)
ax.scatter(pj, pi, s=6, facecolors="none", edgecolors="cyan", linewidths=0.5,
           label="IC(0) pattern = tril(A): diag + W (k-1) + N (k-8) neighbors")
add_block_grid(ax, N8, n8, color="w", lw=0.4, alpha=0.4)
ax.set_title(r"$\log_{10}|L|$, $L=\mathrm{chol}(A)$: band fills in;"
             "\ncyan = the only entries IC(0) keeps")
ax.set_xlabel("column")
ax.set_ylabel("row")
ax.legend(loc="upper right", fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(FIGDIR / "grid8_cholL.png", dpi=150)
plt.close(fig)

# ---- 5. IC(0) at n=8 and the Vecchia cross-check -----------------------------
Lic8 = ic0(A8)
M8 = Lic8 @ Lic8.T
ok("IC(0) reproduces A exactly on its own pattern ((L L^T)_ij == A_ij on pattern)",
   np.allclose(M8[np.abs(A8) > 0], A8[np.abs(A8) > 0]))
Lnorm = np.linalg.norm(L8)
rel_kept = np.linalg.norm((Lic8 - L8)[patternA]) / Lnorm
dropped = np.linalg.norm(L8[~patternA]) / Lnorm
wA = np.linalg.eigvalsh(A8)
kappa_A8 = wA[-1] / wA[0]
wg = sla.eigh(A8, M8, eigvals_only=True)
kappa_ic8 = wg[-1] / wg[0]
info(f"||L_ic - L||_F/||L||_F on kept pattern = {rel_kept:.4f}; "
     f"dropped mass ||L off-pattern||_F/||L||_F = {dropped:.4f}")
info(f"kappa(A) = {kappa_A8:.2f}  ->  kappa(M_ic^-1 A) = {kappa_ic8:.2f}  (n=8)")
ok("IC(0) reduces the condition number at n=8", kappa_ic8 < kappa_A8)

# Vecchia cross-check. IC(0)'s column k encodes the truncated regression of
# u_k on its successor stencil {k+1 (E), k+8 (S -- next grid row down; rows
# are drawn north-at-top as in the figures)} only; the covariance-side
# (Vecchia) coefficients regress u_k on the same set from exact Sigma
# submatrices. Because P A P = A (180-degree rotation), this is the same
# comparison as the task's 'regress node on W,N predecessor neighbors'
# phrasing -- verified below as its own PASS.
max_coef_dev = 0.0
max_sd_dev = 0.0
dev_list = []
coef_mags = []
for k in range(N8):
    i, j = divmod(k, n8)
    nb = []
    if j + 1 < n8:
        nb.append(k + 1)
    if i + 1 < n8:
        nb.append(k + n8)
    if nb:
        w_vec = np.linalg.solve(S8[np.ix_(nb, nb)], S8[nb, k])
        w_ic = -Lic8[nb, k] / Lic8[k, k]
        dev_list.append(np.max(np.abs(w_vec - w_ic)))
        coef_mags.append(np.mean(np.abs(w_vec)))
        max_coef_dev = max(max_coef_dev, dev_list[-1])
        cv = S8[k, k] - S8[k, nb] @ w_vec
    else:
        cv = S8[k, k]
    max_sd_dev = max(max_sd_dev, abs(np.sqrt(cv) - 1.0 / Lic8[k, k]))
mean_coef_dev = float(np.mean(dev_list))
mean_vecchia_coef = float(np.mean(coef_mags))
# same comparison in the task's W,S-predecessor phrasing, via the rotation:
kt = 3 * n8 + 4
nb_pred = [kt - 1, kt - n8]  # W and N neighbors (flat k-1, k-8)
w_pred = np.linalg.solve(S8[np.ix_(nb_pred, nb_pred)], S8[nb_pred, kt])
kr = N8 - 1 - kt
nb_succ = [kr + 1, kr + n8]
w_succ = np.linalg.solve(S8[np.ix_(nb_succ, nb_succ)], S8[nb_succ, kr])
ok("W,N-predecessor truncated regression == rotated successor regression (P A P = A)",
   np.allclose(np.sort(w_pred), np.sort(w_succ)))
ok(f"Vecchia vs IC(0): same order but measurably different on the grid, unlike the 1-D chain "
   f"(max |dev| = {max_coef_dev:.2e} on coefficients of typical size {mean_vecchia_coef:.2f})",
   1e-6 < max_coef_dev < 0.15 and max_sd_dev < 0.01)
info(f"|Vecchia - IC(0)| coefficient deviation: max = {max_coef_dev:.3e} (at interior nodes), "
     f"mean = {mean_coef_dev:.3e}, i.e. ~{100*max_coef_dev/0.37:.0f}% of a typical interior "
     f"coefficient (~0.37); max innovation-sd deviation |sqrt(cond var) - 1/L_ii| = {max_sd_dev:.3e}")
partA["ic0"] = {
    "rel_err_on_kept_pattern": rel_kept, "dropped_mass": dropped,
    "kappa_A": kappa_A8, "kappa_Minv_A": kappa_ic8,
    "vecchia_max_coef_dev": max_coef_dev, "vecchia_mean_coef_dev": mean_coef_dev,
    "vecchia_mean_coef_magnitude": mean_vecchia_coef,
    "vecchia_max_innovation_sd_dev": max_sd_dev,
}

# ---- 6. Multiscale regressors ------------------------------------------------
ones = np.ones(N8)
cov_ubar = S8 @ ones / N8            # Cov(u_i, ubar)
var_ubar = ones @ S8 @ ones / N8**2  # Var(ubar)
beta = cov_ubar / var_ubar
beta_g = beta.reshape(n8, n8)
centers = [(3, 3), (3, 4), (4, 3), (4, 4)]
corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
ok("global-mean loadings beta form the bridge shape: max at the 4 center nodes, min at corners",
   np.isclose(beta_g[3, 3], beta_g.max())
   and all(np.isclose(beta_g[c], beta_g.max()) for c in centers)
   and all(np.isclose(beta_g[c], beta_g.min()) for c in corners))
S_res1 = S8 - np.outer(cov_ubar, cov_ubar) / var_ubar
ve1 = 1.0 - np.trace(S_res1) / np.trace(S8)
Z2 = block_average_matrix(n8, 2)  # 16 coarse regressors
cross = S8 @ Z2
S_res2 = S8 - cross @ np.linalg.solve(Z2.T @ S8 @ Z2, cross.T)
ve16 = 1.0 - np.trace(S_res2) / np.trace(S8)
ok(f"variance explained jumps: global mean {ve1:.3f} -> 2x2 block averages {ve16:.3f}",
   ve16 > ve1 > 0)
info(f"fraction of total variance explained: single global mean = {ve1:.4f}, "
     f"16 2x2-block averages = {ve16:.4f}")

d_before = np.sqrt(np.diag(S8))
corr_before = (S8[kc] / (d_before[kc] * d_before)).reshape(n8, n8)
d_after = np.sqrt(np.diag(S_res2))
corr_after = (S_res2[kc] / (d_after[kc] * d_after)).reshape(n8, n8)
ii, jj = np.meshgrid(np.arange(n8), np.arange(n8), indexing="ij")
far = np.maximum(np.abs(ii - 3), np.abs(jj - 3)) >= 3
far_before = np.mean(np.abs(corr_before[far]))
far_after = np.mean(np.abs(corr_after[far]))
ok(f"residual correlation length collapses after 2x2-coarse conditioning "
   f"(mean |corr| at Chebyshev dist >= 3: {far_before:.3f} -> {far_after:.3f})",
   far_after < 0.25 * far_before)
partA["multiscale"] = {
    "var_explained_global_mean": ve1, "var_explained_2x2_blocks": ve16,
    "beta_max": beta_g.max(), "beta_min": beta_g.min(),
    "beta_center_over_edge_mid": beta_g[3, 3] / beta_g[0, 3],
    "mean_abs_corr_far_before": far_before, "mean_abs_corr_far_after": far_after,
    "center_node": [3, 3],
}

fig, ax = plt.subplots(figsize=(5.6, 4.8))
im = ax.imshow(beta_g, cmap="viridis")
ax.set_title(r"$\beta_i = \mathrm{Cov}(u_i,\bar u)/\mathrm{Var}(\bar u)$:"
             "\nloading of each node on the global mean (bridge shape)")
ax.set_xlabel("grid col j")
ax.set_ylabel("grid row i")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(FIGDIR / "grid8_coarse_beta.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
for ax, C, ttl in [
    (axes[0], corr_before, "corr(u(3,3), u(.)) under $\\Sigma = A^{-1}$"),
    (axes[1], corr_after, "same, residual field after conditioning\non the 16 2x2 block averages"),
]:
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.plot(3, 3, "k+", ms=10, mew=2)
    ax.set_title(ttl)
    ax.set_xlabel("grid col j")
    ax.set_ylabel("grid row i")
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("coarse averages absorb the long-range correlations", y=1.0)
fig.tight_layout()
fig.savefig(FIGDIR / "grid8_residual_cov.png", dpi=150)
plt.close(fig)

RESULTS["part_a"] = partA
print(f"  [time] Part A done in {time.time()-t0:.1f}s")

# ===========================================================================
# PART B -- SOLVER STUDY at n = 32
# ===========================================================================
t0 = time.time()
print("== PART B: solver study at n=32 (N=1024) ==")
n = 32
N = n * n
h = 1.0 / (n + 1)
A = poisson_2d(n)
Ad = A.toarray()
partB = {"n": n, "N": N, "h": h}

# ---- 7. hot/cold-rod problem -------------------------------------------------
f = np.zeros((n, n))
rod_hot = [(i, 4) for i in range(3, 9)]     # 6 nodes near NW corner
rod_cold = [(i, 27) for i in range(23, 29)]  # 6 nodes near SE corner
for i, j in rod_hot:
    f[i, j] = 1.0
for i, j in rod_cold:
    f[i, j] = -1.0
b = f.ravel()
ustar = spla.spsolve(A.tocsc(), b)
U = ustar.reshape(n, n)
imax = np.unravel_index(U.argmax(), U.shape)
imin = np.unravel_index(U.argmin(), U.shape)
ok("solution: global max on the hot rod, global min on the cold rod, correct signs",
   imax in rod_hot and imin in rod_cold
   and all(U[p] > 0 for p in rod_hot) and all(U[p] < 0 for p in rod_cold))
partB["rods"] = {"hot": rod_hot, "cold": rod_cold, "u_max": U.max(), "u_min": U.min()}

fig, ax = plt.subplots(figsize=(6.2, 5.4))
vm = np.abs(U).max()
im = ax.imshow(U, cmap="RdBu_r", vmin=-vm, vmax=vm)
for pts, c in [(rod_hot, "darkred"), (rod_cold, "darkblue")]:
    (i0, j0) = pts[0]
    ax.add_patch(plt.Rectangle((j0 - 0.5, i0 - 0.5), 1, len(pts), fill=False,
                               edgecolor=c, lw=1.8))
ax.set_title("hot/cold-rod solution $u^*$ (n=32): +1 rod near NW, -1 rod near SE")
ax.set_xlabel("grid col j")
ax.set_ylabel("grid row i")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(FIGDIR / "twolevel_solution.png", dpi=150)
plt.close(fig)

# ---- 8. preconditioners --------------------------------------------------------
Lic = ic0(Ad)


def M_ic(r):
    y = sla.solve_triangular(Lic, r, lower=True)
    return sla.solve_triangular(Lic.T, y, lower=False)


# (c) coarse-only global mean: Z = ones/sqrt(N); M^{-1} = Z (Z^T A Z)^{-1} Z^T + theta I.
# The theta*I term keeps M^{-1} SPD (the rank-one projection alone is singular);
# theta = 1/mean(diag(A)) = h^2/4 makes it a Jacobi-scaled identity.
z1 = np.ones(N) / np.sqrt(N)
sZ = z1 @ (A @ z1)
theta = 1.0 / np.mean(A.diagonal())


def M_coarse_only(r):
    return z1 * ((z1 @ r) / sZ) + theta * r


# (d,e) two-level additive with Z_b = 4x4 block averages (64 coarse dofs).
# Coarse correction Z (Z^T A Z)^{-1} Z^T is invariant to column scaling of Z,
# so 'average' vs 'indicator' scaling changes nothing; Galerkin Ac = Z^T A Z.
Zb = block_average_matrix(n, 4)
Ac = Zb.T @ (A @ Zb)
Ac_cho = sla.cho_factor(Ac)
inv_diag = 1.0 / A.diagonal()


def coarse_corr(r):
    return Zb @ sla.cho_solve(Ac_cho, Zb.T @ r)


def M_twolevel_jac(r):
    return inv_diag * r + coarse_corr(r)


def M_twolevel_ic(r):
    return M_ic(r) + coarse_corr(r)


methods = [
    ("none", None),
    ("ic0", M_ic),
    ("coarse_only", M_coarse_only),
    ("twolevel_jacobi", M_twolevel_jac),
    ("twolevel_ic0", M_twolevel_ic),
]

# dense M^{-1} matrices for kappa(M^{-1} A)
Licinv = sla.solve_triangular(Lic, np.eye(N), lower=True)
Minv_ic_d = Licinv.T @ Licinv
CC = Zb @ np.linalg.solve(Ac, Zb.T)
Minv_dense = {
    "none": np.eye(N),
    "ic0": Minv_ic_d,
    "coarse_only": np.outer(z1, z1) / sZ + theta * np.eye(N),
    "twolevel_jacobi": np.diag(inv_diag) + CC,
    "twolevel_ic0": Minv_ic_d + CC,
}

partB["methods"] = {}
sol = {}
err_curves = {}  # per-iteration rel l2 error ||x_k - u*||/||u*||, k=0..iters (Part C reuses)
ustar_norm = np.linalg.norm(ustar)
for name, M in methods:
    xh = []
    x, res = pcg(A, b, M=M, tol=1e-10, maxiter=2000, x_hist=xh)
    err_curves[name] = np.array([np.linalg.norm(xk - ustar) for xk in xh]) / ustar_norm
    kap, lmin, lmax = kappa_from_Minv(Ad, Minv_dense[name])
    relerr = np.linalg.norm(x - ustar) / np.linalg.norm(ustar)
    sol[name] = (x, res)
    partB["methods"][name] = {
        "iterations": len(res) - 1, "final_relres": res[-1],
        "converged": res[-1] <= 1e-10, "kappa_MinvA": kap,
        "lam_min": lmin, "lam_max": lmax,
        "rel_err_vs_spsolve": relerr, "res_hist": res,
    }
    info(f"{name:16s} iters={len(res)-1:4d}  kappa(M^-1 A)={kap:9.2f}  "
         f"relres={res[-1]:.2e}  |x-u*|/|u*|={relerr:.2e}")

it = {k: v["iterations"] for k, v in partB["methods"].items()}
kap = {k: v["kappa_MinvA"] for k, v in partB["methods"].items()}
ok("all five PCG runs converged to relres <= 1e-10 within 2000 iterations",
   all(v["converged"] for v in partB["methods"].values()))
ok("PCG solutions match spsolve (rel err < 1e-6)",
   all(v["rel_err_vs_spsolve"] < 1e-6 for v in partB["methods"].values()))
# Coarse-only degenerates exactly to plain CG on this RHS: b is odd under the
# 180-degree rotation (P b = -b, the two rods map onto each other with sign
# flip) and P A P = A, so parity propagates through the iteration -- every
# residual stays odd (inductively: A preserves parity, and on an odd residual
# the even z1 contributes nothing, so M^{-1} acts as theta*I). Odd vectors are
# orthogonal to the even z1 (1^T b = 0 is the k=0 case), so z1^T r_k = 0
# throughout and M^{-1} r = theta r -- a scalar multiple of the identity, to
# which PCG is invariant. Note mean-zero b alone would NOT suffice: 1 is not
# an eigenvector of A (boundary rows have nonzero row sums), so 1^T A b != 0
# for generic mean-zero b; the parity argument is what closes the induction.
# kappa(M^{-1}A) improves 2.4x by deflating the (even) lowest mode, but that
# mode is never excited: kappa is a worst-case-RHS bound.
ok(f"coarse-only helps not at all here: iterations {it['none']} -> {it['coarse_only']} "
   f"(odd b, even z1: parity keeps z1^T r_k = 0, so M^{{-1}} degenerates to theta*I "
   "= scaled plain CG)",
   it["coarse_only"] == it["none"] and abs(b.sum()) < 1e-12)
info(f"coarse-only kappa(M^-1 A) = {kap['coarse_only']:.1f} vs {kap['none']:.1f} for none: the "
     "2.4x deflation of the lowest (even) mode is unexploitable on this odd RHS")
ok("iteration ordering: none >= coarse_only >= twolevel_jacobi >= ic0 >= twolevel_ic0",
   it["none"] >= it["coarse_only"] >= it["twolevel_jacobi"] >= it["ic0"] >= it["twolevel_ic0"])
info(f"note twolevel_jacobi needs MORE iterations than ic0 ({it['twolevel_jacobi']} vs "
     f"{it['ic0']}) despite smaller kappa ({kap['twolevel_jacobi']:.1f} vs {kap['ic0']:.1f}): "
     "spectrum clustering beats raw kappa")
ok("kappa ordering matches: adding the coarse level improves both Jacobi and IC(0)",
   kap["twolevel_jacobi"] < kap["none"] and kap["twolevel_ic0"] < kap["ic0"])

fig, ax = plt.subplots(figsize=(7.2, 5.0))
labels = {
    "none": "CG (no preconditioner)",
    "ic0": "IC(0)",
    "coarse_only": "coarse-only (global mean + $\\theta I$)",
    "twolevel_jacobi": "two-level: Jacobi + 4x4 coarse",
    "twolevel_ic0": "two-level: IC(0) + 4x4 coarse",
}
for name, _ in methods:
    res = sol[name][1]
    ls = "--" if name == "coarse_only" else "-"  # dashed: sits exactly on the CG curve
    ax.semilogy(res, ls, lw=1.6, label=f"{labels[name]}  [{len(res)-1} its]")
ax.axhline(1e-10, color="gray", lw=0.8, ls=":")
ax.set_xlabel("iteration")
ax.set_ylabel(r"relative residual $\|r_k\|/\|b\|$")
ax.set_title("hot/cold-rod problem, n=32: PCG convergence")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR / "twolevel_convergence.png", dpi=150)
plt.close(fig)

# ---- 9. error-field pedagogy ---------------------------------------------------
# v_min: lowest eigenvector of A is the analytic sin(pi x) sin(pi y) mode.
xg = np.arange(1, n + 1) * h
v = np.outer(np.sin(np.pi * xg), np.sin(np.pi * xg)).ravel()
v /= np.linalg.norm(v)
lam_min_A = 8 * np.sin(np.pi * h / 2) ** 2 / h**2
ok("v_min = sin(pi x) sin(pi y) is A's lowest eigenvector (residual < 1e-10)",
   np.linalg.norm(A @ v - lam_min_A * v) / lam_min_A < 1e-10)

# Symmetry caveat discovered here (and worth the report): b is exactly ODD
# under the 180-degree rotation P (P b = -b) while v_min is EVEN, so every
# method whose preconditioner commutes with P (plain CG, coarse-only,
# Jacobi+coarse: the 4x4 block space is P-invariant) keeps e_k odd and hence
# EXACTLY orthogonal to v_min for all k. Only IC(0) -- whose lexicographic
# elimination ordering is not rotation-invariant -- leaks error into the even
# subspace, and its lingering mode IS v_min. Smoothness of the other methods'
# errors is therefore measured by the amplitude (norm) fraction
# ||Vlow^T e||/||e|| in the invariant lowest-15-eigenmode subspace of A. The
# cut at 15 is degeneracy-safe: it completes the degenerate (3,4)/(4,3) pair
# at 0-based ranks 13-14 and excludes the next pair (1,5)/(5,1), ranks 15-16,
# whole (w[14] < w[15] = w[16]).
Pn = np.eye(N)[::-1]
ok("RHS is odd under the 180-degree rotation: P b == -b (rods map onto each other)",
   np.allclose(Pn @ b, -b))
wA32, VA32 = np.linalg.eigh(Ad)
n_low = 15
Vlow = VA32[:, :n_low]
ok("dense lam_min(A) matches analytic 8 sin^2(pi h/2)/h^2",
   np.isclose(wA32[0], lam_min_A, rtol=1e-10))

snap_methods = ["none", "ic0", "twolevel_jacobi"]
snap_ks = [0, 5, 15]
snaps = {}
partB["error_fields"] = {}
for name in snap_methods:
    M = dict(methods)[name]
    snaps[name] = {}
    partB["error_fields"][name] = {}
    for k in snap_ks:
        if k == 0:
            xk = np.zeros(N)
        else:
            xk, _ = pcg(A, b, M=M, tol=0.0, maxiter=k)  # tol=0: run exactly k its
        e = ustar - xk
        snaps[name][k] = e
        partB["error_fields"][name][str(k)] = {
            "err_norm": np.linalg.norm(e),
            "err_norm_rel": np.linalg.norm(e) / np.linalg.norm(ustar),
            "smooth_frac_vmin": abs(v @ e) / np.linalg.norm(e),
            "smooth_frac_low15": np.linalg.norm(Vlow.T @ e) / np.linalg.norm(e),
        }
ef = partB["error_fields"]
info("smooth-mode content |<e_k, v_min>|/||e_k||:    " + "; ".join(
    f"{m}: " + ", ".join(f"k={k}: {ef[m][str(k)]['smooth_frac_vmin']:.3f}" for k in snap_ks)
    for m in snap_methods))
info("lowest-15-eigenmode amplitude fraction of e_k: " + "; ".join(
    f"{m}: " + ", ".join(f"k={k}: {ef[m][str(k)]['smooth_frac_low15']:.3f}" for k in snap_ks)
    for m in snap_methods))
info("relative error ||e_k||/||u*||:                 " + "; ".join(
    f"{m}: " + ", ".join(f"k={k}: {ef[m][str(k)]['err_norm_rel']:.2e}" for k in snap_ks)
    for m in snap_methods))
ok("symmetry-preserving methods have exactly zero v_min content in e_k (odd RHS, even v_min)",
   all(ef[m][str(k)]["smooth_frac_vmin"] < 1e-6
       for m in ["none", "twolevel_jacobi"] for k in [5, 15]))
ok("plain CG error still large and smooth at k=15 "
   f"(rel err {ef['none']['15']['err_norm_rel']:.2f}, "
   f"low-mode fraction {ef['none']['15']['smooth_frac_low15']:.2f})",
   ef["none"]["15"]["err_norm_rel"] > 0.1 and ef["none"]["15"]["smooth_frac_low15"] > 0.9)
ok("IC(0) kills rough error fast but a smooth global mode lingers: its k=15 error is "
   f"{100*ef['ic0']['15']['smooth_frac_vmin']:.0f}% the v_min mode "
   f"(rel err {ef['ic0']['15']['err_norm_rel']:.1e})",
   ef["ic0"]["15"]["smooth_frac_vmin"] > 0.5
   and ef["ic0"]["15"]["err_norm_rel"] < 1e-3 < ef["none"]["15"]["err_norm_rel"])
ok("two-level kills rough and smooth together: smallest error at k=5 "
   f"({ef['twolevel_jacobi']['5']['err_norm_rel']:.2e} vs IC(0) "
   f"{ef['ic0']['5']['err_norm_rel']:.2e} vs CG {ef['none']['5']['err_norm_rel']:.2e}), "
   "zero v_min content throughout",
   ef["twolevel_jacobi"]["5"]["err_norm_rel"] < ef["ic0"]["5"]["err_norm_rel"]
   < ef["none"]["5"]["err_norm_rel"]
   and ef["twolevel_jacobi"]["15"]["err_norm_rel"] < 1e-2
   and ef["twolevel_jacobi"]["15"]["smooth_frac_vmin"] < 1e-6)
info("honest caveat: by k=15 IC(0)'s error NORM has caught up "
     f"({ef['ic0']['15']['err_norm_rel']:.1e} < {ef['twolevel_jacobi']['15']['err_norm_rel']:.1e} "
     "two-level Jacobi) -- but 91% of what remains is the smooth mode it cannot kill; "
     "two-level IC(0) beats both (32 vs 39/48 iterations)")

fig, axes = plt.subplots(3, 3, figsize=(11.0, 10.4), layout="constrained")
row_titles = {"none": "plain CG", "ic0": "IC(0)-PCG", "twolevel_jacobi": "two-level (Jacobi+coarse)"}
for ri, name in enumerate(snap_methods):
    vm = max(np.abs(snaps[name][k]).max() for k in snap_ks)
    for ci, k in enumerate(snap_ks):
        ax = axes[ri, ci]
        im = ax.imshow(snaps[name][k].reshape(n, n), cmap="RdBu_r", vmin=-vm, vmax=vm)
        d = ef[name][str(k)]
        ax.set_title(f"k={k}   |e|/|u*|={d['err_norm_rel']:.1e}\n"
                     f"low-mode {d['smooth_frac_low15']:.2f}, v_min {d['smooth_frac_vmin']:.2f}",
                     fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if ci == 0:
            ax.set_ylabel(row_titles[name], fontsize=10)
    fig.colorbar(im, ax=axes[ri, :].tolist(), fraction=0.02, pad=0.02)
fig.suptitle("error fields $e_k = u^* - x_k$ (shared symmetric scale per row):\n"
             "plain CG stays large+smooth; IC(0) converges fast but leaves the $v_{min}$ mode;\n"
             "two-level suppresses rough and smooth together", fontsize=11)
fig.savefig(FIGDIR / "twolevel_error_fields.png", dpi=150)
plt.close(fig)

partB["lam_min_A"] = lam_min_A
partB["theta_coarse_only"] = theta
RESULTS["part_b"] = partB
print(f"  [time] Part B done in {time.time()-t0:.1f}s")

# ===========================================================================
# PART C -- COST MODELS: flops and span for the five Part-B methods
# ===========================================================================
# APPENDED SECTION: reports 11/13 cite the 36 PASS lines above by position,
# so everything new lives here, after all Part-A/B output.
#
# FLOP convention (report 14 SS4, PASS 30): 1 MAC = 2 flops.
#   * A-matvec = 2 nnz(A).
#   * PCG vector work per iteration, counted off python/pcg.py's loop: six
#     2N ops (dot p.Ap; axpy x; axpy r; norm r; dot r.z; axpy p) = 12N.
#     alpha/beta scalar divides are O(1) and dropped.
#   * IC(0) apply = two sparse triangular solves = 2 x 2 nnz(L), nnz(L)
#     counted from the actual factor.
#   * Coarse correction, counted from coarse_corr()'s implementation:
#     restriction Zb^T r = 64 x (16 mults + 15 adds); cho_solve = two dense
#     64x64 triangular solves, sum_i (2i+1) = m^2 flops each; prolongation
#     Zb c = one mult per fine node (each row of Zb has a single nonzero).
#   * coarse-only apply, from M_coarse_only(): dot (2N) + scalar divide (1)
#     + scale (N) + scale (N) + add (N) = 5N + 1.
#   * additive two-level = smoother flops + coarse flops + one vector add.
# Setup (included in totals, checked in PASS 40): pcg()'s pre-loop work =
# bnorm (2N) + z0 = M(b) (one M-apply) + rz0 dot (2N) = 4N + M-apply flops;
# NOTE this is 2N MORE than report 14 SS4's setup term ("one apply and one
# dot" = 2N + apply), which omits the bnorm = ||b|| reduction that pcg.py
# actually performs -- so 14 SS4's plain-CG total sits exactly 2N = 2,048
# flops below this convention; per-iteration conventions are identical.
# span = span(M) + 10 (the bnorm reduction runs in parallel). One-time
# factorizations (ic0(A), forming Zb^T A Zb, cho_factor) are EXCLUDED:
# amortized offline, identical for every solve with that preconditioner.
#
# SPAN convention (work-span / Blelloch, unbounded processors): depths in
# cm_depths below, one-line justification each. Per-iteration span is the
# pcg.py critical path: matvec -> dot(p,Ap) -> alpha divide -> r-axpy ->
# M(r) -> dot(r,z) -> beta divide -> p-axpy = matvec + M-apply +
# 2 reductions + 4 elementwise (x-axpy and the norm stop-test hang off the
# path in parallel).
#
# Triangular-solve span from the dependency DAG: in the lower solve
# L y = r on the IC(0) pattern, row k has strictly-lower nonzeros only at
# k-1 (W neighbor, absent at grid col 0) and k-n (N neighbor, absent in
# grid row 0), so y_(r,c) waits on y_(r,c-1) and y_(r-1,c). By induction
# both predecessors of (r,c) sit on level r+c-1, so level(r,c) = r+c: the
# anti-diagonal hyperplanes r+c = 0..2(n-1) -- exactly 2n-1 = 63 sequential
# wavefronts, each of depth 3 (two MACs + one divide). PASS 37 computes
# this longest path by explicit levelization of the actual factor instead
# of assuming it. The L^T solve is the 180-degree-rotated DAG (level
# 2(n-1)-(r+c)), again 63 wavefronts, and it consumes the L-solve's output:
# the pair is sequential, span = 2 x 63 x 3 = 378.
t0 = time.time()
print("== PART C: work/span cost models for the five Part-B methods ==")

cm_nnzA = int(A.nnz)
cm_nnzL = int((np.abs(Lic) > 0).sum())
cm_nnzZb = int((Zb != 0).sum())
cm_mc = Zb.shape[1]  # 64 coarse dofs

# --- PASS 37: explicit topological levelization of the L solve's DAG ------
cm_strict = np.tril(np.abs(Lic) > 0, -1)  # strictly-lower nonzeros = edges j->k
cm_level = np.zeros(N, dtype=int)
for cm_k in range(N):  # rows of a lower-triangular L are already topological
    cm_preds = np.nonzero(cm_strict[cm_k])[0]
    if cm_preds.size:
        cm_level[cm_k] = cm_level[cm_preds].max() + 1
cm_nwave = int(cm_level.max()) + 1
ok(f"IC(0) triangular-solve dependency DAG levelizes into exactly 2n-1 = {2*n-1} wavefronts "
   f"(longest path over the factor's {int(cm_strict.sum())} strictly-lower nonzeros: "
   f"{cm_nwave} levels, computed not assumed)",
   cm_nwave == 2 * n - 1)
info(f"levelization: level(r,c) == r+c for all nodes: "
     f"{bool((cm_level == (np.arange(N)//n + np.arange(N)%n)).all())}; "
     f"largest wavefront {int(np.bincount(cm_level).max())} nodes (the main anti-diagonal)")

# --- flop model ------------------------------------------------------------
cm_matvec_fl = 2 * cm_nnzA                    # 9984
cm_vec_fl = 12 * N                            # six 2N ops in the pcg loop
cm_ic_fl = 2 * (2 * cm_nnzL)                  # two tri solves x 2 nnz(L)
cm_restrict_fl = cm_mc * (16 + 15)            # 16 mults + 15 adds per coarse dof
cm_csolve_fl = 2 * cm_mc * cm_mc              # cho_solve: two dense tri solves, m^2 each
cm_prolong_fl = N                             # one mult per fine node
cm_cc_fl = cm_restrict_fl + cm_csolve_fl + cm_prolong_fl
cm_M_fl = {
    "none": 0,
    "ic0": cm_ic_fl,
    "coarse_only": 5 * N + 1,
    "twolevel_jacobi": N + cm_cc_fl + N,      # Jacobi scale + coarse + combine add
    "twolevel_ic0": cm_ic_fl + cm_cc_fl + N,  # IC(0) + coarse + combine add
}

# --- span model ------------------------------------------------------------
cm_depths = {
    "elementwise": (1, "independent per component (scale/divide/add)"),
    "axpy": (1, "fused multiply-add, independent per component"),
    "stencil_matvec": (4, "ceil(log2 5)+1: 5 stencil products in parallel + depth-3 sum tree"),
    "reduction": (10, "dot/norm over N=1024: ceil(log2 N) = 10 binary tree"),
    "dense_inverse_apply_64": (7, "ceil(log2 64)+1: per coarse row, products in parallel + 6-deep sum tree"),
    "restriction_4x4": (5, "ceil(log2 16)+1: per coarse dof, scale + 4-deep sum tree over its 16 fine nodes"),
    "prolongation": (1, "each fine node scales its single block value"),
    "tri_solve_wavefront": (189, "63 hyperplane wavefronts (level(r,c)=r+c, PASS 37) x depth 3 (two MACs + divide)"),
    "ic0_apply": (378, "L then L^T solves sequentially dependent: 2 x 189"),
    "coarse_correction": (13, "restriction 5 -> dense inverse-apply 7 -> prolongation 1, sequential"),
    "coarse_only_apply": (13, "dot 10 -> divide 1 -> scale 1 on one branch, theta*r in parallel, + combine add 1"),
    "additive_combine": (1, "smoother and coarse branches independent: max(spans) + one add"),
    "pcg_iteration_overhead": (28, "matvec 4 + dot 10 + alpha-div 1 + r-axpy 1 + dot 10 + beta-div 1 + p-axpy 1; x-axpy and stop test off-path"),
    "setup": ("M-apply span + 10", "z0 = M(b) then rz0 dot (10); bnorm reduction in parallel"),
}
cm_M_sp = {
    "none": 0,
    "ic0": cm_depths["ic0_apply"][0],
    "coarse_only": cm_depths["coarse_only_apply"][0],
    "twolevel_jacobi": max(1, cm_depths["coarse_correction"][0]) + 1,
    "twolevel_ic0": max(cm_depths["ic0_apply"][0], cm_depths["coarse_correction"][0]) + 1,
}

cm_names = [m for m, _ in methods]
cm = {}
for cm_m in cm_names:
    cm_iters = it[cm_m]
    cm_pif = cm_matvec_fl + cm_vec_fl + cm_M_fl[cm_m]
    cm_sf = 4 * N + cm_M_fl[cm_m]
    cm_pis = cm_depths["pcg_iteration_overhead"][0] + cm_M_sp[cm_m]
    cm_ss = cm_M_sp[cm_m] + cm_depths["reduction"][0]
    cm[cm_m] = {
        "iters": cm_iters,
        "per_iter_flops": cm_pif, "setup_flops": cm_sf,
        "total_flops": cm_iters * cm_pif + cm_sf,
        "per_iter_span": cm_pis, "setup_span": cm_ss,
        "total_span": cm_iters * cm_pis + cm_ss,
        "final_rel_err": float(err_curves[cm_m][-1]),
    }
cm_rank_f = sorted(cm_names, key=lambda m: cm[m]["total_flops"])
cm_rank_s = sorted(cm_names, key=lambda m: cm[m]["total_span"])

info("cost table (model): method / per-iter flops / total flops / per-iter span / total span / iters")
for cm_m in cm_names:
    c = cm[cm_m]
    info(f"  {cm_m:16s} {c['per_iter_flops']:7,d} fl/it  {c['total_flops']:10,d} fl total  "
         f"{c['per_iter_span']:4d} span/it  {c['total_span']:7,d} span total  {c['iters']:3d} its")

# --- PASS 38-40: rankings and consistency ----------------------------------
ok(f"flops ranking (cheapest first): {' < '.join(cm_rank_f)} -- the two IC(0)-based methods "
   f"are the two cheapest in total flops ({cm['ic0']['total_flops']:,} and "
   f"{cm['twolevel_ic0']['total_flops']:,} vs {cm['none']['total_flops']:,} plain CG): "
   "the 378-deep solves are only 2 x 2 nnz(L) flops and buy the biggest iteration cuts",
   set(cm_rank_f[:2]) == {"ic0", "twolevel_ic0"} and cm_rank_f[0] == "ic0")
ok(f"span ranking FLIPS: {' < '.join(cm_rank_s)} -- two-level Jacobi ({cm['twolevel_jacobi']['total_span']:,}) "
   f"and plain CG ({cm['none']['total_span']:,}) are cheapest in span while the IC(0)-based methods "
   f"are the two most expensive ({cm['twolevel_ic0']['total_span']:,} and {cm['ic0']['total_span']:,}): "
   "378-deep sequential triangular solves dwarf the depth-14 two-level apply; the flops winner "
   "ic0 is the span loser",
   cm_rank_s[0] == "twolevel_jacobi" and cm_rank_s[1] == "none"
   and set(cm_rank_s[-2:]) == {"ic0", "twolevel_ic0"} and cm_rank_s[-1] == cm_rank_f[0])
ok("cost-model consistency: total == iters x per-iter + setup for flops and span, all five methods",
   all(c["total_flops"] == c["iters"] * c["per_iter_flops"] + c["setup_flops"]
       and c["total_span"] == c["iters"] * c["per_iter_span"] + c["setup_span"]
       for c in cm.values())
   and all(err_curves[m].shape[0] == cm[m]["iters"] + 1 for m in cm_names))

RESULTS["cost_models"] = {
    "flop_convention": "1 MAC = 2 flops; A-matvec = 2 nnz(A); pcg loop vector work = six 2N ops "
                       "= 12N; IC(0) apply = 2 tri solves x 2 nnz(L); coarse correction from "
                       "coarse_corr(): restriction 64x(16 mults+15 adds) + cho_solve two dense "
                       "64x64 tri solves (m^2 flops each) + prolongation N mults; coarse-only "
                       "apply 5N+1; additive two-level adds one vector add; setup = 4N + one "
                       "M-apply (pcg pre-loop: bnorm 2N + z0 = M(b) + rz0 dot 2N -- 2N more "
                       "than report 14 SS4's setup of one apply + one dot, which omits the "
                       "bnorm reduction pcg.py performs; per-iteration conventions identical); "
                       "one-time factorizations excluded (amortized)",
    "span_convention": "work-span (Blelloch) depth, unbounded processors; per-iteration span = "
                       "stencil matvec + M-apply + 2 reductions + 4 elementwise (pcg critical "
                       "path); setup span = M-apply + 10; additive two-level = max(branch "
                       "spans) + 1",
    "constants": {
        "N": N, "n": n,
        "nnz_A": cm_nnzA, "matvec_flops": cm_matvec_fl,
        "nnz_L_ic0": cm_nnzL, "ic0_apply_flops": cm_ic_fl,
        "nnz_Zb": cm_nnzZb, "coarse_dim": cm_mc,
        "restriction_flops": cm_restrict_fl, "coarse_solve_flops": cm_csolve_fl,
        "prolongation_flops": cm_prolong_fl, "coarse_correction_flops": cm_cc_fl,
        "pcg_vector_flops_per_iter": cm_vec_fl, "pcg_setup_vector_flops": 4 * N,
        "ic0_wavefronts_computed": cm_nwave,
        "ic0_dag_edges": int(cm_strict.sum()),
    },
    "depths": {k: {"depth": v[0], "why": v[1]} for k, v in cm_depths.items()},
    "methods": cm,
    "ranking_by_flops": cm_rank_f,
    "ranking_by_span": cm_rank_s,
}

# --- figures: rel l2 error vs cumulative flops / span -----------------------
for cm_fname, cm_key, cm_xlab, cm_scale, cm_ttl in [
    ("twolevel_error_vs_flops.png", "flops", "cumulative flops (model, millions)", 1e6,
     "hot/cold-rod, n=32: error vs modeled WORK\n(1 MAC = 2 flops; IC(0)-based methods cheapest)"),
    ("twolevel_error_vs_span.png", "span", "cumulative span (model, critical-path depth)", 1.0,
     "same runs vs modeled SPAN (unbounded processors)\n"
     "ranking flips: IC(0)'s 378-deep triangular solves dominate"),
]:
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for cm_m in cm_names:
        c = cm[cm_m]
        cum = (c[f"setup_{cm_key}"]
               + c[f"per_iter_{cm_key}"] * np.arange(c["iters"] + 1)) / cm_scale
        ls = "--" if cm_m == "coarse_only" else "-"
        (ln,) = ax.semilogy(cum, err_curves[cm_m], ls, lw=1.6,
                            label=f"{labels[cm_m]}  [{c['iters']} its, "
                                  f"{c[f'total_{cm_key}']:,} {cm_key}]")
        ax.semilogy(cum[-1], err_curves[cm_m][-1], "o", ms=5, color=ln.get_color())
    ax.axhline(1e-10, color="gray", lw=0.8, ls=":")
    ax.text(0.01, 1.3e-10, "rel err 1e-10", color="gray", fontsize=7,
            ha="left", transform=ax.get_yaxis_transform())
    ax.set_xlabel(cm_xlab)
    ax.set_ylabel(r"relative error $\Vert x_k - u^*\Vert / \Vert u^*\Vert$")
    ax.set_title(cm_ttl, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / cm_fname, dpi=150)
    plt.close(fig)
print(f"  [time] Part C done in {time.time()-t0:.1f}s")

with open(RESDIR / "grid_multiscale.json", "w") as fjson:
    json.dump(jsonable(RESULTS), fjson, indent=2)
print(f"saved results/grid_multiscale.json; {N_FAIL} FAIL line(s)")
