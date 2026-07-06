"""Fluctuation-dissipation toy experiment: equilibrium fluctuations of the
discrete Dirichlet Laplacian know the Green's function, and regressions fit
to those fluctuations recover the precision matrix -- up to and including a
working PCG preconditioner learned from thermal samples alone.

Physical setup (k_B T = 1 throughout): the Gibbs measure of the discrete
Dirichlet energy (1/2) u^T A u is N(0, A^{-1}) (reports/09). FDT here is the
Gaussian identity

    deterministic response to a unit point kick,  A^{-1} e_j
        ==  spontaneous-fluctuation covariance column  Sigma[:, j] / k_B T.

Part A (1-D chain, n = 32, A = laplacian_1d(32)/h^2):
  A1  three samplers of N(0, A^{-1}): exact coloring u = L^{-T} z;
      overdamped Langevin du = -A u dt + sqrt(2 k_B T) dW (Euler-Maruyama);
      Gibbs sampler = Gauss-Seidel sweep + N(0, k_B T h^2/2) noise per site.
  A2  kick response vs covariance column from each sampler.
  A3  regressions fit to the samples recover A: two-sided (notebook
      A = (I-B).D2 form), sequential left-to-right (Pourahmadi modified
      Cholesky; bridge drifts to the pinned RIGHT wall, coefficient
      (n+1-i)/(n+2-i), 1-based), sequential right-to-left (recovers chol(A),
      coefficient i/(i+1); reversal identity chol(Sigma) = P L^{-T} P
      verified on ESTIMATED quantities).
  A4  whitening held-out samples with the exact and the fitted factors; the
      'hidden heaters': two-sided residuals u_i - (u_{i-1}+u_{i+1})/2 equal
      (h^2/2) * (A u), the hidden thermal forcing (covariance A: variance
      h^2/2 per site, neighbor correlation exactly -1/2, zero beyond).

Part B (2-D grid, A = poisson_2d):
  B5  n = 16: response-vs-covariance for the center node (Green's-function
      bump) as side-by-side heatmaps.
  B6  n = 32 (N = 1024): Vecchia / truncated sequential regression fit from
      the SAMPLED covariance alone (lexicographic order, each node regressed
      on its at-most-2 previously-ordered stencil neighbors W and S), giving
      an SPD preconditioner M = (I-Phi)^T D^{-2} (I-Phi) applied by two
      sparse triangular solves; raced in PCG (b = grf_rhs(32), tol 1e-10)
      against none / Jacobi / the exact-covariance Vecchia ceiling.
  B7  bookkeeping for the report: Sigma_hat Frobenius errors vs N_s and the
      1-D Langevin critical-slowing-down number (tau of the slowest mode).

Euler-Maruyama bias note (A1b): the EM chain u+ = (I - dt A) u + sqrt(2 dt) z
has EXACT stationary covariance solving S = (I-dtA)S(I-dtA)^T + 2 dt I,
i.e. S_EM = (A - (dt/2) A^2)^{-1}  --  per-mode variance 1/(lam(1 - dt*lam/2)),
an O(dt) inflation. At dt = 0.5/lam_max the Frobenius-norm bias is ~1e-3
(below sampling noise), but on the stiffest mode the inflation is exactly
1/(1 - dt*lam_max/2) = 4/3, which we measure directly.

Seeds (np.random.default_rng): coloring-1D 1, Langevin 2, Gibbs 3, holdout 4,
coloring-2D-16 5, coloring-2D-32 6; grf_rhs uses its default seed 42.
Deterministic identities use tight tolerances (<= 1e-8 relative); statistical
checks state their sampling tolerance, derived from the exact iid-Gaussian
covariance-error formula  E||Sigma_hat-Sigma||_F^2 = (tr(Sigma)^2+||Sigma||_F^2)/N
with a factor 3 (iid samplers) or 5 (autocorrelated samplers, thinned to ~1
autocorrelation time so residual correlation inflates errors ~1.5-2x).

Outputs: results/fdt.json, figures/fdt_*.png (dpi 150).
Run from the repo root:  uv run python python/experiments/fdt_fluctuations.py
Runtime ~1-2 min. Expected: all PASS.
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
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcg import pcg
from poisson import grf_rhs, laplacian_1d, poisson_2d
from preconditioners import jacobi

T0 = time.time()
ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

KBT = 1.0
SEEDS = {"color_1d": 1, "langevin": 2, "gibbs": 3, "holdout": 4,
         "color_2d_16": 5, "color_2d_32": 6, "grf_rhs": 42}

res = {"meta": {"kBT": KBT, "seeds": SEEDS}, "checks": [], "figures": []}


def ok(name, cond, detail=""):
    line = f"{'PASS' if cond else 'FAIL'}: {name}"
    if detail:
        line += f"   [{detail}]"
    print(line)
    res["checks"].append({"name": name, "pass": bool(cond), "detail": detail})


def relF(X, Y):
    return float(np.linalg.norm(X - Y) / np.linalg.norm(Y))


def wishart_pred(S, N):
    """Exact E||Sigma_hat - Sigma||_F / ||Sigma||_F for N iid Gaussian samples
    (uncentered): Var(Sigma_hat_ij) = (S_ii S_jj + S_ij^2)/N."""
    return float(np.sqrt((np.trace(S) ** 2 + (S * S).sum()) / N) / np.linalg.norm(S))


def col_pred(S, j, N):
    """Predicted relative l2 error of column j of Sigma_hat (N iid samples)."""
    num = (S[j, j] * np.trace(S) + (S[:, j] ** 2).sum()) / N
    return float(np.sqrt(num) / np.linalg.norm(S[:, j]))


def emp_corr(Z):
    C = Z @ Z.T / Z.shape[1]
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def maxoff(C):
    M = C.copy()
    np.fill_diagonal(M, 0.0)
    return float(np.abs(M).max())


def to_py(o):
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ======================================================================
# PART A -- 1-D chain, n = 32
# ======================================================================
print("=" * 72)
print("PART A: 1-D chain, n = 32, A = laplacian_1d(32)/h^2, k_B T = 1")
print("=" * 72)

n = 32
h = 1.0 / (n + 1)
Ad = (laplacian_1d(n) / h**2).toarray()
x = np.arange(1, n + 1) * h
Sigma = np.linalg.inv(Ad)

G_bridge = np.minimum.outer(x, x) - np.outer(x, x)
ok("1D: A^{-1} == h*(min(s,t)-st)  (Brownian bridge, n=32)",
   relF(Sigma, h * G_bridge) < 1e-10, f"relF={relF(Sigma, h*G_bridge):.2e}")
ok("1D: conditional variance 1/A_ii == h^2/2",
   bool(np.allclose(1.0 / np.diag(Ad), h * h / 2)), f"h^2/2={h*h/2:.4e}")

evals, evecs = np.linalg.eigh(Ad)
lam_min, lam_max = float(evals[0]), float(evals[-1])
kappa_1d = lam_max / lam_min
v_slow, v_fast = evecs[:, 0], evecs[:, -1]
res["meta"].update({"n_1d": n, "h_1d": h, "lam_min_1d": lam_min,
                    "lam_max_1d": lam_max, "kappa_1d": kappa_1d})

# ---------------------------------------------------------------- A1(a)
print("\n--- A1(a): exact coloring  u = L^{-T} z ---")
Lch = np.linalg.cholesky(Ad)                    # A = L L^T
rng_c = np.random.default_rng(SEEDS["color_1d"])
N_MAX = 100_000
Uc = sla.solve_triangular(Lch.T, rng_c.standard_normal((n, N_MAX)),
                          lower=False, check_finite=False)
Ns_list = [100, 1000, 10_000, 100_000]
errs_color, preds_color = [], []
for N in Ns_list:
    Sh = Uc[:, :N] @ Uc[:, :N].T / N
    errs_color.append(relF(Sh, Sigma))
    preds_color.append(wishart_pred(Sigma, N))
S_color = Uc @ Uc.T / N_MAX

slope = float(np.polyfit(np.log(Ns_list), np.log(errs_color), 1)[0])
ok("coloring: ||Sigma_hat-Sigma||_F/||Sigma||_F ~ N^{-1/2} (loglog slope)",
   -0.65 < slope < -0.35, f"slope={slope:.3f}")
ratio = errs_color[-1] / preds_color[-1]
ok("coloring: error at N=1e5 within [0.4,2.5]x exact iid-Wishart prediction",
   0.4 < ratio < 2.5,
   f"err={errs_color[-1]:.4e}, pred={preds_color[-1]:.4e}, ratio={ratio:.2f}")

# ---------------------------------------------------------------- A1(b)
print("\n--- A1(b): overdamped Langevin (Euler-Maruyama) ---")
# du = -A u dt + sqrt(2 k_B T) dW;  EM: u+ = (I - dt A) u + sqrt(2 dt kBT) z.
# Stable for dt < 2/lam_max; we use dt = 0.5/lam_max.
# O(dt) EM bias: exact EM stationary covariance is S_EM = (A - (dt/2)A^2)^{-1}
# (per-mode 1/(lam(1-dt*lam/2))), verified below via its Lyapunov equation,
# i.e. NOT A^{-1}: stiffest-mode variance is inflated by 1/(1-dt*lam_max/2)=4/3.
dt = 0.5 / lam_max
tau_pred_steps = -1.0 / np.log1p(-dt * lam_min)      # ~ 1/(lam_min dt) = 2*kappa
thin_L = int(np.ceil(tau_pred_steps))
C_L = 256
rounds_L = 79                                        # N_lang = 79*256 = 20224
burn_L = 10 * thin_L
rng_L = np.random.default_rng(SEEDS["langevin"])
U = np.zeros((n, C_L))
amp = np.sqrt(2.0 * dt * KBT)
for _ in range(burn_L):
    U += -dt * (Ad @ U) + amp * rng_L.standard_normal((n, C_L))
ACF_LEN = 20_000
proj = np.empty((ACF_LEN, C_L))
snaps = []
for t in range(rounds_L * thin_L):
    U += -dt * (Ad @ U) + amp * rng_L.standard_normal((n, C_L))
    if t < ACF_LEN:
        proj[t] = v_slow @ U
    if (t + 1) % thin_L == 0:
        snaps.append(U.copy())
UL = np.concatenate(snaps, axis=1)
N_lang = UL.shape[1]
S_lang = UL @ UL.T / N_lang

err_lang = relF(S_lang, Sigma)
tol_lang = 5.0 * wishart_pred(Sigma, N_lang)
ok(f"Langevin: rel Frobenius error of Sigma_hat (N={N_lang}, thin={thin_L} steps)",
   err_lang < tol_lang, f"err={err_lang:.4e} < tol {tol_lang:.4e} (5x iid pred)")

# EM stationary covariance: S_EM = (A - (dt/2) A^2)^{-1}; Lyapunov fixed point.
S_EM = np.linalg.inv(Ad - 0.5 * dt * (Ad @ Ad))
lyap = (np.eye(n) - dt * Ad) @ S_EM @ (np.eye(n) - dt * Ad).T + 2 * dt * KBT * np.eye(n)
ok("Langevin: S_EM=(A-(dt/2)A^2)^{-1} solves EM Lyapunov fixed point",
   relF(lyap, S_EM) < 1e-10, f"relF={relF(lyap, S_EM):.2e}")
bias_frob = relF(S_EM, Sigma)
err_vs_EM = relF(S_lang, S_EM)
print(f"      predicted EM bias ||S_EM-Sigma||_F/||Sigma||_F = {bias_frob:.3e}"
      f" (below sampling noise {err_lang:.3e}); err vs S_EM = {err_vs_EM:.3e}")
# Measured bias on the stiffest mode (decorrelates in ~2 steps -> iid at thin).
var_fast = float(np.mean((v_fast @ UL) ** 2))
em_fast = 1.0 / (lam_max * (1.0 - dt * lam_max / 2.0))   # = (4/3)/lam_max
exact_fast = 1.0 / lam_max
ok("Langevin EM bias, stiffest mode: measured variance == EM prediction "
   "1/(lam(1-dt*lam/2)) = (4/3)/lam_max  (tol 5%)",
   abs(var_fast / em_fast - 1.0) < 0.05,
   f"measured={var_fast:.4e}, EM pred={em_fast:.4e}, exact 1/lam={exact_fast:.4e}")
ok("Langevin EM bias is real: stiffest-mode variance inflated >15% over 1/lam_max"
   " (theory: +33.3%)",
   var_fast / exact_fast - 1.0 > 0.15,
   f"measured inflation = {100*(var_fast/exact_fast-1):.1f}%")

# Autocorrelation time of the slowest mode (item 7: critical slowing down).
lags = np.arange(25, 2501, 25)
den = float(np.mean(proj * proj))
acf = np.array([np.mean(proj[:-k] * proj[k:]) for k in lags]) / den
mask = acf > 0.2
tau_meas = float(-1.0 / np.polyfit(lags[mask], np.log(acf[mask]), 1)[0])
ok("Langevin: slowest-mode autocorrelation time == 1/(lam_min dt) = 2*kappa steps"
   " (tol 25%)",
   abs(tau_meas / tau_pred_steps - 1.0) < 0.25,
   f"measured={tau_meas:.0f} steps, predicted={tau_pred_steps:.0f} steps "
   f"(= 2*kappa = {2*kappa_1d:.0f}); in time units {tau_meas*dt:.4f} vs "
   f"1/lam_min={1/lam_min:.4f}")

res["partA"] = {"samplers": {
    "coloring": {"N_list": Ns_list, "rel_frob_err": errs_color,
                 "wishart_pred": preds_color, "loglog_slope": slope},
    "langevin": {"N": N_lang, "dt": dt, "dt_x_lam_max": dt * lam_max,
                 "chains": C_L, "thin_steps": thin_L, "burn_steps": burn_L,
                 "rel_frob_err": err_lang, "rel_frob_err_vs_S_EM": err_vs_EM,
                 "predicted_EM_bias_frob": bias_frob,
                 "stiff_mode_var_measured": var_fast,
                 "stiff_mode_var_EM_pred": em_fast,
                 "stiff_mode_var_exact": exact_fast,
                 "tau_steps_measured": tau_meas,
                 "tau_steps_predicted": tau_pred_steps,
                 "tau_time_measured": tau_meas * dt,
                 "tau_time_predicted": 1.0 / lam_min},
}}

# ---------------------------------------------------------------- A1(c)
print("\n--- A1(c): Gibbs sampler = Gauss-Seidel sweep + N(0, h^2/2) noise ---")
rho_gs = np.cos(np.pi * h) ** 2                 # GS/Gibbs per-sweep rate
tau_gibbs = -1.0 / np.log(rho_gs)               # ~ 1/(pi h)^2 sweeps
thin_G = int(np.ceil(tau_gibbs))
C_G = 256
rounds_G = 78                                   # N_gibbs = 78*256 = 19968
burn_G = 15 * thin_G
rng_G = np.random.default_rng(SEEDS["gibbs"])
u = np.zeros((n, C_G))
sig_G = np.sqrt(KBT * h * h / 2.0)              # sd of full-conditional noise
snaps_G = []
for s in range(burn_G + rounds_G * thin_G):
    z = rng_G.standard_normal((n, C_G))
    for i in range(n):                          # sequential = Gauss-Seidel scan
        left = u[i - 1] if i > 0 else 0.0
        right = u[i + 1] if i < n - 1 else 0.0
        u[i] = 0.5 * (left + right) + sig_G * z[i]
    if s >= burn_G and (s - burn_G + 1) % thin_G == 0:
        snaps_G.append(u.copy())
UG = np.concatenate(snaps_G, axis=1)
N_gibbs = UG.shape[1]
S_gibbs = UG @ UG.T / N_gibbs
err_gibbs = relF(S_gibbs, Sigma)
tol_gibbs = 5.0 * wishart_pred(Sigma, N_gibbs)
ok(f"Gibbs: rel Frobenius error of Sigma_hat (N={N_gibbs}, thin={thin_G} sweeps)",
   err_gibbs < tol_gibbs, f"err={err_gibbs:.4e} < tol {tol_gibbs:.4e} (5x iid pred)")
res["partA"]["samplers"]["gibbs"] = {
    "N": N_gibbs, "chains": C_G, "thin_sweeps": thin_G, "burn_sweeps": burn_G,
    "rho_gauss_seidel": float(rho_gs), "tau_sweeps": float(tau_gibbs),
    "rel_frob_err": err_gibbs}

# Figure: covariance convergence (coloring) + the two dynamic samplers.
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(Ns_list, errs_color, "o-", label="exact coloring (measured)")
ax.loglog(Ns_list, preds_color, "k--", label="iid Wishart prediction")
ref = errs_color[0] * (np.array(Ns_list) / Ns_list[0]) ** -0.5
ax.loglog(Ns_list, ref, ":", color="gray", label=r"$N^{-1/2}$ reference")
ax.loglog([N_lang], [err_lang], "s", color="tab:red",
          label=f"Langevin (N={N_lang})")
ax.loglog([N_gibbs], [err_gibbs], "^", color="tab:green",
          label=f"Gibbs (N={N_gibbs})")
ax.set_xlabel("number of snapshots $N_s$")
ax.set_ylabel(r"$\|\hat\Sigma-A^{-1}\|_F\,/\,\|A^{-1}\|_F$")
ax.set_title("Empirical covariance $\\to A^{-1}$ at the Monte Carlo rate (1D, n=32)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fdt_covariance_convergence.png", dpi=150)
plt.close(fig)
res["figures"].append("figures/fdt_covariance_convergence.png")

# ---------------------------------------------------------------- A2
print("\n--- A2: FDT -- kick response vs spontaneous-jitter covariance ---")
jc = n // 2                                     # 0-based center node, x=0.515
e_j = np.zeros(n)
e_j[jc] = 1.0
u_resp = np.linalg.solve(Ad, e_j)               # deterministic response to kick
fdt_cols = {}
for name, Sh, NN, fac in [("coloring", S_color, N_MAX, 3.0),
                          ("langevin", S_lang, N_lang, 5.0),
                          ("gibbs", S_gibbs, N_gibbs, 5.0)]:
    col = Sh[:, jc] / KBT
    err = float(np.linalg.norm(col - u_resp) / np.linalg.norm(u_resp))
    tol = fac * col_pred(Sigma, jc, NN)
    ok(f"FDT 1D ({name}): A^{{-1}}e_j == Sigma_hat[:,j]/kBT, rel l2 err < "
       f"{fac:.0f}x sampling pred", err < tol, f"err={err:.4e}, tol={tol:.4e}")
    fdt_cols[name] = {"rel_l2_err": err, "tol": tol}
res["partA"]["fdt_response"] = {"j_center_0based": jc, "x_j": float(x[jc]),
                                "columns": fdt_cols}

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, u_resp, "k-", lw=2, label=r"kick response $A^{-1}e_j$ (deterministic)")
ax.plot(x, S_color[:, jc] / KBT, "o", ms=5, alpha=0.8,
        label=f"coloring cov column (N={N_MAX})")
ax.plot(x, S_lang[:, jc] / KBT, "s", ms=5, alpha=0.8,
        label=f"Langevin cov column (N={N_lang})")
ax.plot(x, S_gibbs[:, jc] / KBT, "^", ms=5, alpha=0.8,
        label=f"Gibbs cov column (N={N_gibbs})")
ax.axvline(x[jc], color="gray", ls=":", lw=1)
ax.set_xlabel("x")
ax.set_ylabel("displacement / covariance")
ax.set_title("Fluctuation-dissipation: point-kick response = jitter covariance "
             f"($k_BT=1$, node $j={jc}$)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fdt_response_vs_covariance.png", dpi=150)
plt.close(fig)
res["figures"].append("figures/fdt_response_vs_covariance.png")

# ---------------------------------------------------------------- A3
print("\n--- A3(a): two-sided regressions from data -> A = (I-B).D2 ---")
S_hat = S_color                                  # N = 1e5, uncentered
A_hat = np.linalg.inv(S_hat)
B_hat = np.eye(n) - (1.0 / np.diag(A_hat))[:, None] * A_hat   # I - diag(Ah)^-1 Ah
nb_dev = max(abs(B_hat[i, i + s] - 0.5) for i in range(n)
             for s in (-1, 1) if 0 <= i + s < n)
Bo = B_hat.copy()
np.fill_diagonal(Bo, 0.0)
for i in range(n):
    for s_ in (-1, 1):
        if 0 <= i + s_ < n:
            Bo[i, i + s_] = 0.0
nonnb = float(np.abs(Bo).max())
ok("two-sided: B_hat rows -> 1/2 on neighbors (tol 0.02, N=1e5)",
   nb_dev < 0.02, f"max|B_hat[i,i+-1]-0.5|={nb_dev:.4f}")
ok("two-sided: B_hat -> 0 off the stencil (tol 0.02)",
   nonnb < 0.02, f"max non-neighbor |B_hat|={nonnb:.4f}")

X_all = Uc.T                                     # samples as rows (N x n)
honest_nodes = [5, 16, 26]
for i in honest_nodes:
    idx = [k for k in range(n) if k != i]
    w_ne = np.linalg.solve(S_hat[np.ix_(idx, idx)], S_hat[idx, i])
    w_ls = np.linalg.lstsq(X_all[:, idx], X_all[:, i], rcond=None)[0]
    rd = float(np.abs(w_ne - w_ls).max() / np.abs(w_ne).max())
    r_vec = X_all[:, i] - X_all[:, idx] @ w_ls
    rv = float(np.mean(r_vec**2))
    ok(f"two-sided node {i}: honest lstsq on samples == Sigma_hat normal equations",
       rd < 1e-8, f"max coeff rel diff={rd:.2e}")
    ok(f"two-sided node {i}: residual variance == 1/A_hat_ii",
       abs(rv * A_hat[i, i] - 1.0) < 1e-8,
       f"rv={rv:.4e}, 1/A_hat_ii={1/A_hat[i,i]:.4e}")

A_rec_2s = (np.eye(n) - B_hat) @ np.diag(np.diag(A_hat))  # diag(1/resid var)
err_2s = relF(A_rec_2s, Ad)
pred_inv = float(np.sqrt(np.trace(Ad) ** 2 + (Ad * Ad).sum())
                 / (np.sqrt(N_MAX) * np.linalg.norm(Ad)))
ok("two-sided: A_rec = (I-B_hat) diag(1/resid var) recovers A "
   "(tol 3x first-order pred)",
   err_2s < 3 * pred_inv, f"relF={err_2s:.4e}, pred={pred_inv:.4e}")
res["partA"]["regressions"] = {"two_sided": {
    "max_neighbor_coeff_dev_from_half": float(nb_dev),
    "max_nonneighbor_coeff": nonnb, "A_rec_rel_err": err_2s,
    "first_order_pred": pred_inv, "honest_nodes": honest_nodes}}


def seq_l2r(S):
    """Regress u_i on predecessors u_0..u_{i-1} (Pourahmadi phiL2R) from a
    covariance matrix alone. Returns strictly-lower Phi, residual variances."""
    m = len(S)
    Phi = np.zeros((m, m))
    d2 = np.empty(m)
    d2[0] = S[0, 0]
    for i in range(1, m):
        w = np.linalg.solve(S[:i, :i], S[:i, i])
        Phi[i, :i] = w
        d2[i] = S[i, i] - S[i, :i] @ w
    return Phi, d2


def seq_r2l(S):
    """Regress u_i on successors u_{i+1}..u_{n-1} (phiR2L). Returns strictly-
    upper Phi_u, residual variances."""
    m = len(S)
    Phi = np.zeros((m, m))
    d2 = np.empty(m)
    d2[-1] = S[-1, -1]
    for i in range(m - 1):
        w = np.linalg.solve(S[i + 1:, i + 1:], S[i + 1:, i])
        Phi[i, i + 1:] = w
        d2[i] = S[i, i] - S[i, i + 1:] @ w
    return Phi, d2


print("\n--- A3(b): sequential left-to-right (Pourahmadi / phiL2R) ---")
# Closed form, derive-and-check on the EXACT Sigma: the bridge conditioned on
# its past is a diffusion pinned to the RIGHT wall: E[u_i|u_{i-1}] =
# u_{i-1} (1-x_i)/(1-x_{i-1}) = u_{i-1} (n+1-i)/(n+2-i)  (1-based i).
Phi_ex, d2_ex = seq_l2r(Sigma)
i0 = np.arange(1, n)                                   # 0-based row index
closed_l2r = (n - i0) / (n + 1.0 - i0)                 # (n+1-i)/(n+2-i), 1-based
dev_cl = float(np.abs(Phi_ex[i0, i0 - 1] - closed_l2r).max())
ok("L2R closed form (exact Sigma): coeff(u_i ~ u_{i-1}) == (n+1-i)/(n+2-i)",
   dev_cl < 1e-9, f"max dev={dev_cl:.2e}")
Phi_far = Phi_ex.copy()
Phi_far[i0, i0 - 1] = 0.0
ok("L2R Markov (exact Sigma): only the immediate predecessor matters",
   float(np.abs(Phi_far).max()) < 1e-9,
   f"max non-immediate coeff={np.abs(Phi_far).max():.2e}")

errs_l2r = []
Phi_hat = d2_hat = None
Phi_1k = None
for N in Ns_list:
    ShN = Uc[:, :N] @ Uc[:, :N].T / N
    Ph, dd = seq_l2r(ShN)
    A_rec = (np.eye(n) - Ph).T @ np.diag(1.0 / dd) @ (np.eye(n) - Ph)
    errs_l2r.append(relF(A_rec, Ad))
    if N == 1000:
        Phi_1k = Ph.copy()
    if N == N_MAX:
        Phi_hat, d2_hat = Ph, dd
        ok("L2R assembly identity on SAMPLES: (I-Phi)^T D^-2 (I-Phi) == "
           "inv(Sigma_hat)", relF(A_rec, A_hat) < 1e-8,
           f"relF={relF(A_rec, A_hat):.2e}")
coeff_l2r_hat = Phi_hat[i0, i0 - 1]
dev_hat = float(np.abs(coeff_l2r_hat - closed_l2r).max())
ok("L2R from samples (N=1e5): immediate-predecessor coeffs match closed form "
   "(tol 0.02)", dev_hat < 0.02, f"max dev={dev_hat:.4f}")
mono = all(errs_l2r[k + 1] < errs_l2r[k] for k in range(len(errs_l2r) - 1))
ok("L2R: ||A_rec - A||/||A|| decreases with N_s and ends < 3x pred",
   mono and errs_l2r[-1] < 3 * pred_inv,
   "errs=" + ", ".join(f"{N}:{e:.3e}" for N, e in zip(Ns_list, errs_l2r)))
res["partA"]["regressions"]["l2r"] = {
    "N_list": Ns_list, "A_rec_rel_err": errs_l2r,
    "closed_form_coeffs": closed_l2r, "estimated_coeffs_1e5": coeff_l2r_hat,
    "max_coeff_dev_1e5": dev_hat}

print("\n--- A3(c): sequential right-to-left -> chol(A) ---")
i1 = np.arange(0, n - 1)                               # 0-based
closed_r2l = (i1 + 1.0) / (i1 + 2.0)                   # i/(i+1), 1-based
dev_chol = float(np.abs(-Lch[i1 + 1, i1] / Lch[i1, i1] - closed_r2l).max())
ok("R2L closed form (exact chol(A)): -L[i+1,i]/L[i,i] == i/(i+1)",
   dev_chol < 1e-10, f"max dev={dev_chol:.2e}")
Phi_u_hat, d2r_hat = seq_r2l(S_hat)
L_hat = (np.eye(n) - Phi_u_hat).T @ np.diag(1.0 / np.sqrt(d2r_hat))
ok("R2L from SAMPLES: regressions assemble to chol(inv(Sigma_hat)) exactly",
   relF(L_hat, np.linalg.cholesky(A_hat)) < 1e-8,
   f"relF={relF(L_hat, np.linalg.cholesky(A_hat)):.2e}")
# Reversal identity (report 09 / verify_statistical_identities.py check 6):
# chol of the covariance READ IN REVERSED VARIABLE ORDER is the inverse
# transpose of chol(precision):  chol(P Sigma P) == P L^{-T} P.
P = np.eye(n)[::-1]
rev = relF(np.linalg.cholesky(P @ S_hat @ P), P @ np.linalg.inv(L_hat).T @ P)
ok("reversal identity on ESTIMATED quantities: chol(P Sigma_hat P) == P L_hat^{-T} P",
   rev < 1e-8, f"relF={rev:.2e}")
coeff_r2l_hat = Phi_u_hat[i1, i1 + 1]
dev_r2l = float(np.abs(coeff_r2l_hat - closed_r2l).max())
ok("R2L from samples (N=1e5): immediate-successor coeffs match i/(i+1) "
   "(tol 0.02)", dev_r2l < 0.02, f"max dev={dev_r2l:.4f}")
res["partA"]["regressions"]["r2l"] = {
    "closed_form_coeffs": closed_r2l, "estimated_coeffs_1e5": coeff_r2l_hat,
    "max_coeff_dev_1e5": dev_r2l, "reversal_identity_relF": rev}

fig, ax = plt.subplots(figsize=(8, 5))
nodes1 = i0 + 1                                        # 1-based node of L2R row
ax.plot(nodes1, closed_l2r, "-", color="tab:blue",
        label=r"L2R closed form $(n+1-i)/(n+2-i)$ (drift to right wall)")
ax.plot(nodes1, coeff_l2r_hat, "o", ms=5, color="tab:blue",
        label="L2R fitted, $N_s=10^5$")
ax.plot(nodes1, Phi_1k[i0, i0 - 1], "o", ms=4, mfc="none", color="tab:blue",
        alpha=0.5, label="L2R fitted, $N_s=10^3$")
ax.plot(i1 + 1, closed_r2l, "-", color="tab:red",
        label=r"R2L closed form $i/(i+1)$ (drift to left wall)")
ax.plot(i1 + 1, coeff_r2l_hat, "s", ms=5, color="tab:red",
        label="R2L fitted, $N_s=10^5$")
ax.set_xlabel("node $i$ (1-based)")
ax.set_ylabel("regression coefficient on the adjacent node")
ax.set_title("Sequential regressions on samples find the Cholesky coefficients")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fdt_regression_coeffs.png", dpi=150)
plt.close(fig)
res["figures"].append("figures/fdt_regression_coeffs.png")

# ---------------------------------------------------------------- A4
print("\n--- A4: whitening held-out samples; the hidden heaters ---")
rng_h = np.random.default_rng(SEEDS["holdout"])
N_H = 20_000
H = sla.solve_triangular(Lch.T, rng_h.standard_normal((n, N_H)),
                         lower=False, check_finite=False)
C_raw = emp_corr(H)
W_exact = Lch.T
W_l2r = np.diag(1.0 / np.sqrt(d2_hat)) @ (np.eye(n) - Phi_hat)
W_r2l = np.diag(1.0 / np.sqrt(d2r_hat)) @ (np.eye(n) - Phi_u_hat)  # == L_hat^T
C_ex = emp_corr(W_exact @ H)
C_l2 = emp_corr(W_l2r @ H)
C_r2 = emp_corr(W_r2l @ H)
mo = {"raw": maxoff(C_raw), "exact_LT": maxoff(C_ex),
      "fitted_L2R": maxoff(C_l2), "fitted_R2L": maxoff(C_r2)}
print(f"      max |off-diag corr|: raw={mo['raw']:.3f}, exact L^T={mo['exact_LT']:.4f}, "
      f"fitted L2R={mo['fitted_L2R']:.4f}, fitted R2L={mo['fitted_R2L']:.4f}")
ok("whitening: exact L^T decorrelates held-out samples (max off-diag < 0.05; "
   "noise floor ~0.025 at N=2e4)", mo["exact_LT"] < 0.05, f"{mo['exact_LT']:.4f}")
ok("whitening: fitted L2R factor decorrelates held-out samples (tol 0.06)",
   mo["fitted_L2R"] < 0.06, f"{mo['fitted_L2R']:.4f}")
ok("whitening: fitted R2L factor decorrelates held-out samples (tol 0.06)",
   mo["fitted_R2L"] < 0.06, f"{mo['fitted_R2L']:.4f}")

# Hidden heaters: two-sided residuals = (h^2/2) * (A u) = (h^2/2) * hidden forcing.
R_phys = H.copy()
R_phys[1:] -= 0.5 * H[:-1]
R_phys[:-1] -= 0.5 * H[1:]
F_hidden = Ad @ H
ok("hidden heaters: u_i - (u_{i-1}+u_{i+1})/2 == (h^2/2)*(A u) per sample "
   "(deterministic)", relF(R_phys, (h * h / 2) * F_hidden) < 1e-12,
   f"relF={relF(R_phys, (h*h/2)*F_hidden):.2e}")
vr = (R_phys**2).mean(axis=1)
dev_vr = float(np.abs(vr / (KBT * h * h / 2) - 1.0).max())
ok("hidden heaters: Var(residual_i) == kBT h^2/2 at every site (tol 6%)",
   dev_vr < 0.06, f"max site dev={100*dev_vr:.1f}%, h^2/2={h*h/2:.4e}")
C_f = emp_corr(F_hidden)
nb_corr = float(np.mean([C_f[i, i + 1] for i in range(n - 1)]))
far = C_f.copy()
np.fill_diagonal(far, 0.0)
for i in range(n - 1):
    far[i, i + 1] = far[i + 1, i] = 0.0
far_max = float(np.abs(far).max())
ok("hidden forcing f = A u: neighbor correlation == -1/2 (Cov(Au) = A; tol 0.03)",
   abs(nb_corr + 0.5) < 0.03, f"mean neighbor corr={nb_corr:.4f}")
ok("hidden forcing: correlation beyond one lattice step == 0 (tol 0.05)",
   far_max < 0.05, f"max |corr| beyond neighbors={far_max:.4f}")
res["partA"]["whitening"] = {
    "N_holdout": N_H, "max_offdiag_corr": mo,
    "residual_var_max_site_dev": dev_vr, "residual_var_target": h * h / 2,
    "hidden_forcing_neighbor_corr": nb_corr,
    "hidden_forcing_max_far_corr": far_max}

fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
panels = [(C_raw, f"raw samples\nmax|off|={mo['raw']:.2f}"),
          (C_ex, "after exact $L^T$\nmax|off|=" + f"{mo['exact_LT']:.3f}"),
          (C_l2, "after fitted L2R $\\hat D^{-1}(I-\\hat\\Phi)$\nmax|off|="
           + f"{mo['fitted_L2R']:.3f}"),
          (C_r2, "after fitted R2L $\\hat L^T$\nmax|off|="
           + f"{mo['fitted_R2L']:.3f}")]
for axp, (Cm, title) in zip(axes, panels):
    im = axp.imshow(Cm, vmin=-1, vmax=1, cmap="RdBu_r")
    axp.set_title(title, fontsize=10)
    axp.set_xticks([])
    axp.set_yticks([])
fig.colorbar(im, ax=axes, shrink=0.8, label="empirical correlation")
fig.suptitle("Whitening held-out equilibrium samples (1D, n=32, $N_s$=2e4): "
             "residuals are the hidden heaters, $r=(h^2/2)\\,Au$", fontsize=11)
fig.savefig(FIG / "fdt_whitening_residuals.png", dpi=150, bbox_inches="tight")
plt.close(fig)
res["figures"].append("figures/fdt_whitening_residuals.png")

# ======================================================================
# PART B -- 2-D grid
# ======================================================================
print("\n" + "=" * 72)
print("PART B: 2-D grid -- FDT + a preconditioner learned from fluctuations")
print("=" * 72)

# ---------------------------------------------------------------- B5
print("\n--- B5: n=16 (N=256) response vs covariance heatmaps ---")
n2 = 16
h2 = 1.0 / (n2 + 1)
A16 = poisson_2d(n2)
Ad16 = A16.toarray()
ok("2D: conditional variance 1/A_ii == h^2/4 (n=16)",
   bool(np.allclose(1.0 / A16.diagonal(), h2 * h2 / 4)), f"h^2/4={h2*h2/4:.4e}")
L16 = np.linalg.cholesky(Ad16)
Sigma16 = sla.cho_solve((L16, True), np.eye(n2 * n2))
rng16 = np.random.default_rng(SEEDS["color_2d_16"])
N16 = 100_000
S16 = np.zeros((n2 * n2, n2 * n2))
for _ in range(2):
    Zc = rng16.standard_normal((n2 * n2, N16 // 2))
    Uc16 = sla.solve_triangular(L16.T, Zc, lower=False, check_finite=False)
    S16 += Uc16 @ Uc16.T
S16 /= N16
kc = (n2 // 2) * n2 + n2 // 2                    # center node (8,8) -> k=136
e_c = np.zeros(n2 * n2)
e_c[kc] = 1.0
resp16 = np.linalg.solve(Ad16, e_c)
col16 = S16[:, kc] / KBT
err16 = float(np.linalg.norm(col16 - resp16) / np.linalg.norm(resp16))
tol16 = 3.0 * col_pred(Sigma16, kc, N16)
ok("FDT 2D (n=16): center-node response == sampled covariance column "
   "(rel l2 err < 3x pred)", err16 < tol16, f"err={err16:.4e}, tol={tol16:.4e}")
res["partB"] = {"n16": {"N_samples": N16, "center_node": kc,
                        "rel_l2_err": err16, "tol": tol16,
                        "sigma_rel_frob_err": relF(S16, Sigma16)}}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
vmax = float(resp16.max())
for axp, field, title in [
        (axes[0], resp16, "deterministic kick response $A^{-1}e_c$"),
        (axes[1], col16, f"sampled covariance column ($N_s$={N16:,})")]:
    im = axp.imshow(field.reshape(n2, n2), vmin=0, vmax=vmax, cmap="viridis")
    axp.set_title(title, fontsize=10)
    axp.set_xticks([])
    axp.set_yticks([])
    fig.colorbar(im, ax=axp, shrink=0.85)
fig.suptitle("FDT on the 2-D grid (n=16): the Green's-function bump appears in "
             f"the thermal jitter (rel err {100*err16:.1f}%)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fdt_response_vs_covariance_2d.png", dpi=150)
plt.close(fig)
res["figures"].append("figures/fdt_response_vs_covariance_2d.png")

# ---------------------------------------------------------------- B6
print("\n--- B6: n=32 (N=1024) Vecchia preconditioner from fluctuation data ---")
n3 = 32
N3 = n3 * n3
A32 = poisson_2d(n3)
Ad32 = A32.toarray()
L32 = np.linalg.cholesky(Ad32)
Sigma32 = sla.cho_solve((L32, True), np.eye(N3))

rng32 = np.random.default_rng(SEEDS["color_2d_32"])
chunks = [2000, 8000, 10000, 10000, 10000, 10000]     # cumulative 2k,10k,...,50k
snap_at = {2000, 10000, 50000}
S_acc = np.zeros((N3, N3))
count = 0
S_snap = {}
for c in chunks:
    Zc = rng32.standard_normal((N3, c))
    Uc32 = sla.solve_triangular(L32.T, Zc, lower=False, check_finite=False)
    S_acc += Uc32 @ Uc32.T
    count += c
    if count in snap_at:
        S_snap[count] = S_acc / count
sigma32_errs = {N: relF(S_snap[N], Sigma32) for N in sorted(S_snap)}
sigma32_pred = {N: wishart_pred(Sigma32, N) for N in sorted(S_snap)}
for N in sorted(S_snap):
    print(f"      Sigma_hat (n=32 grid) rel Frob err at N_s={N}: "
          f"{sigma32_errs[N]:.4e}  (iid pred {sigma32_pred[N]:.4e})")
ratio_25 = sigma32_errs[2000] / sigma32_errs[50000]
ok("2D n=32: Sigma_hat error falls ~sqrt(25)=5x from N_s=2k to 50k "
   "(ratio in [3,8])", 3.0 < ratio_25 < 8.0, f"ratio={ratio_25:.2f}")


def vecchia_from_cov(S, nside, full=False):
    """Truncated sequential regression (Vecchia) read off a covariance matrix.

    Lexicographic order k = i*nside + j; node k is regressed on its
    previously-ordered stencil neighbors: W = k-1 (if j>0) and S = k-nside
    (if i>0). Coefficients and residual variances come from 2x2 (or 1x1)
    normal equations that touch ONLY covariance entries. With ``full=True``
    every predecessor is used (sanity path: recovers the exact precision).

    Returns (G, d2): G = I - Phi unit-lower-triangular sparse CSR,
    d2 = residual variances, so that M = G^T diag(1/d2) G is SPD and
    M^{-1} r costs two sparse triangular solves.
    """
    Np = nside * nside
    rows, cols, vals = [], [], []
    d2 = np.empty(Np)
    coef_w, coef_s = [], []
    for k in range(Np):
        i, j = divmod(k, nside)
        if full:
            par = list(range(k))
        else:
            par = []
            if j > 0:
                par.append(k - 1)       # W neighbor
            if i > 0:
                par.append(k - nside)   # S neighbor
        if par:
            cc = np.linalg.solve(S[np.ix_(par, par)], S[par, k])
            d2[k] = S[k, k] - S[par, k] @ cc
            rows += [k] * len(par)
            cols += list(par)
            vals += list(-cc)
            if not full and len(par) == 2:
                coef_w.append(cc[0])
                coef_s.append(cc[1])
        else:
            d2[k] = S[k, k]
    G = sp.csr_matrix(
        (vals + [1.0] * Np, (rows + list(range(Np)), cols + list(range(Np)))),
        shape=(Np, Np))
    return G, d2, np.array(coef_w), np.array(coef_s)


def vecchia_prec(G, d2):
    GT = G.T.tocsr()

    def M(r):
        y = spla.spsolve_triangular(GT, np.asarray(r, dtype=np.float64),
                                    lower=False)
        y *= d2
        return spla.spsolve_triangular(G, y, lower=True)
    return M


# Sanity: full-predecessor regression on the exact covariance == A (4x4 grid).
A4x = poisson_2d(4).toarray()
S4x = np.linalg.inv(A4x)
G4, d24, _, _ = vecchia_from_cov(S4x, 4, full=True)
M4 = G4.toarray().T @ np.diag(1.0 / d24) @ G4.toarray()
ok("Vecchia assembly sanity: full-predecessor fit on exact cov == A (4x4 grid)",
   relF(M4, A4x) < 1e-8, f"relF={relF(M4, A4x):.2e}")

fits = {"exact": vecchia_from_cov(Sigma32, n3)}
for N in sorted(S_snap):
    fits[f"N{N}"] = vecchia_from_cov(S_snap[N], n3)
for name, (G, d2, cw, cs) in fits.items():
    ok(f"Vecchia fit '{name}': all residual variances positive (M SPD)",
       bool((d2 > 0).all()), f"min d^2={d2.min():.3e}")
cw_ex, cs_ex = fits["exact"][2], fits["exact"][3]
cw_50, cs_50 = fits["N50000"][2], fits["N50000"][3]
coef_dev = float(max(np.abs(cw_50 - cw_ex).max(), np.abs(cs_50 - cs_ex).max()))
ok("Vecchia: sampled coefficients (N_s=50k) -> exact-covariance coefficients "
   "(tol 0.05)", coef_dev < 0.05,
   f"max dev={coef_dev:.4f}; exact interior means W={cw_ex.mean():.4f}, "
   f"S={cs_ex.mean():.4f}")

b = grf_rhs(n3)                                       # canonical RHS, seed 42
runs = {}
runs["none"] = pcg(A32, b, M=None, tol=1e-10, maxiter=2000)
runs["jacobi"] = pcg(A32, b, M=jacobi(A32), tol=1e-10, maxiter=2000)
for name, (G, d2, _, _) in fits.items():
    label = "vecchia_exact" if name == "exact" else f"vecchia_{name}"
    runs[label] = pcg(A32, b, M=vecchia_prec(G, d2), tol=1e-10, maxiter=2000)

iters = {k: len(v[1]) - 1 for k, v in runs.items()}
final = {k: float(v[1][-1]) for k, v in runs.items()}
true_res = {k: float(np.linalg.norm(b - A32 @ v[0]) / np.linalg.norm(b))
            for k, v in runs.items()}
for k in runs:
    print(f"      PCG[{k:16s}]  iterations={iters[k]:4d}  "
          f"final relres={final[k]:.2e}  true relres={true_res[k]:.2e}")
ok("PCG: every preconditioner converged to 1e-10 (true residual < 1e-9)",
   all(final[k] <= 1e-10 and true_res[k] < 1e-9 for k in runs),
   f"max true relres={max(true_res.values()):.2e}")
ok("PCG: Jacobi == plain CG iterations (constant diagonal -> scalar M)",
   abs(iters["none"] - iters["jacobi"]) <= 1,
   f"none={iters['none']}, jacobi={iters['jacobi']}")
ok("PCG: exact-covariance Vecchia beats plain CG",
   iters["vecchia_exact"] < iters["none"],
   f"{iters['vecchia_exact']} vs {iters['none']}")
ok("PCG: sampled Vecchia improves with N_s and approaches the sample-free "
   "ceiling (iters_50k <= iters_2k+1 and <= ceiling+10)",
   iters["vecchia_N50000"] <= iters["vecchia_N2000"] + 1
   and iters["vecchia_N50000"] <= iters["vecchia_exact"] + 10,
   f"2k={iters['vecchia_N2000']}, 10k={iters['vecchia_N10000']}, "
   f"50k={iters['vecchia_N50000']}, exact={iters['vecchia_exact']}")

# Exact condition numbers of the preconditioned systems (dense generalized eig).
kappaA = float(np.linalg.eigvalsh(Ad32)[-1] / np.linalg.eigvalsh(Ad32)[0])
kappas = {"none": kappaA, "jacobi": kappaA}
for name, (G, d2, _, _) in fits.items():
    Gd = G.toarray()
    Md = Gd.T @ np.diag(1.0 / d2) @ Gd
    w = sla.eigh(Ad32, Md, eigvals_only=True)
    kappas["vecchia_exact" if name == "exact" else f"vecchia_{name}"] = \
        float(w[-1] / w[0])
print("      kappa(M^{-1}A): " +
      ", ".join(f"{k}={v:.1f}" for k, v in kappas.items()))
ok("kappa: exact-covariance Vecchia cuts kappa(A) by > 2x",
   kappas["vecchia_exact"] < kappaA / 2,
   f"kappa(A)={kappaA:.1f} -> {kappas['vecchia_exact']:.1f}")

res["partB"]["n32"] = {
    "N_samples_list": sorted(S_snap), "sigma_rel_frob_err": sigma32_errs,
    "sigma_wishart_pred": sigma32_pred, "err_ratio_2k_over_50k": float(ratio_25),
    "vecchia_interior_coeff_means_exact": {"W": float(cw_ex.mean()),
                                           "S": float(cs_ex.mean())},
    "vecchia_coeff_max_dev_50k_vs_exact": coef_dev,
    "vecchia_min_d2": {name: float(f[1].min()) for name, f in fits.items()},
    "pcg_iterations": iters, "pcg_final_relres": final,
    "pcg_true_relres": true_res, "kappa": kappas,
    "rhs": "grf_rhs(32), seed 42", "tol": 1e-10}

fig, ax = plt.subplots(figsize=(8, 5.5))
styles = {"none": ("k-", None), "jacobi": ("k--", None),
          "vecchia_N2000": ("-", "tab:orange"),
          "vecchia_N10000": ("-", "tab:green"),
          "vecchia_N50000": ("-", "tab:blue"),
          "vecchia_exact": ("-", "tab:red")}
labels = {"none": "plain CG", "jacobi": "Jacobi",
          "vecchia_N2000": "Vecchia, $N_s$=2k samples",
          "vecchia_N10000": "Vecchia, $N_s$=10k samples",
          "vecchia_N50000": "Vecchia, $N_s$=50k samples",
          "vecchia_exact": "Vecchia, exact covariance (ceiling)"}
for k in ["none", "jacobi", "vecchia_N2000", "vecchia_N10000",
          "vecchia_N50000", "vecchia_exact"]:
    st, col = styles[k]
    kw = {} if col is None else {"color": col}
    ax.semilogy(runs[k][1], st, label=f"{labels[k]} ({iters[k]} its)", **kw)
ax.axhline(1e-10, color="gray", ls=":", lw=1)
ax.set_xlabel("PCG iteration")
ax.set_ylabel(r"relative residual $\|r_k\|/\|b\|$")
ax.set_title("A preconditioner learned from thermal fluctuations "
             "(2-D Poisson, $N$=1024, b = GRF)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fdt_learned_preconditioner.png", dpi=150)
plt.close(fig)
res["figures"].append("figures/fdt_learned_preconditioner.png")

# ---------------------------------------------------------------- wrap up
res["meta"]["elapsed_sec"] = round(time.time() - T0, 1)
npass = sum(c["pass"] for c in res["checks"])
ntot = len(res["checks"])
res["meta"]["checks_passed"] = f"{npass}/{ntot}"
with open(RES / "fdt.json", "w") as f:
    json.dump(to_py(res), f, indent=2)

print("\n" + "=" * 72)
print(f"SUMMARY: {npass}/{ntot} checks passed; elapsed {res['meta']['elapsed_sec']}s")
print("figures: " + ", ".join(res["figures"]))
print("results: results/fdt.json")
