"""Numerical verification of every claim in reports/eigenmodes_explainer.html.

The page's system is the 1-D Dirichlet chain A = laplacian_1d(n)/h^2 with
n = 64, h = 1/65 -- the same operator as the bridge/green-tents explainers,
now read in its eigenbasis.  Five sections:

A. MODES     -- sine eigenvectors v_k(i) = sin(ik*pi/(n+1)) and eigenvalues
                lam_k = 4 sin^2(k*pi*h/2)/h^2 verified against eigh; heat
                half-lives ln2/lam_k via expm (ratio = kappa); optimally
                damped Richardson/GD run from a known error mix, iterates
                DST-projected, per-mode coefficient == (1 - alpha*lam_k)^m.
B. ROTATION  -- V^T A V diagonal (energy = N independent parabolas); one-pass
                DST-I solve == spsolve.
C. DEFLATION -- M_p^{-1} = V_p Lam_p^{-1} V_p^T + alpha_p (I - V_p V_p^T),
                alpha_p = 2/(lam_{p+1}+lam_n), p in {0,1,2,4,8,16}: spectrum
                = {1 x p} U {alpha_p lam_k}, kappa_eff = lam_n/lam_{p+1}
                ~ kappa/(p+1)^2, measured GD iterations to 1e-10 vs p.
D. SGD       -- the honest quadratic caricature: constant-step GD with
                additive isotropic gradient noise, with and without the p=4
                spectral head solved exactly each step; noise floors from the
                exact stationary-variance formula; iterations to 2x floor.
E. LEDGER    -- none / spectral p=4,8 / coarse-average deflation p=4,8 (same
                formula, block-average Z; principal angles vs the true
                bottom-p eigenspace) / Nystrom rank 4,8 (python/nystrom.py) /
                block-Jacobi 2 subdomains / IC(0), all as preconditioners for
                optimally damped Richardson: iterations to 1e-10, kappa_eff,
                directions left wrong (eigenvalues of omega*M^{-1}A farther
                than 10% from 1), plus measured PCG iterations for the
                clustering-vs-range contrast of report 13.

Run from the repo root:
    uv run python python/experiments/eigenmodes_checks.py

Expected output: all PASS; writes results/eigenmodes.json with every number
quoted in the report.  Deterministic, runtime well under 30 s.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.fft
import scipy.sparse.linalg as spla
from scipy.linalg import expm, subspace_angles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nystrom import NystromPreconditioner
from pcg import pcg
from poisson import laplacian_1d
from preconditioners import ic0

np.set_printoptions(precision=6, suppress=True)
results = {"checks": []}


def ok(name, cond, err=None):
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {name}" + (f"   (max err {err:.3g})" if err is not None else ""))
    results["checks"].append({"name": name, "pass": bool(cond),
                              **({"max_err": float(err)} if err is not None else {})})


# ---- the chain and its exact modes ------------------------------------------
n = 64
h = 1.0 / (n + 1)
Asp = (laplacian_1d(n) / h**2).tocsr()
Ad = Asp.toarray()
k = np.arange(1, n + 1)
lam = 4.0 * np.sin(k * np.pi * h / 2.0) ** 2 / h**2          # ascending
i_idx = np.arange(1, n + 1)
V = np.sin(np.outer(i_idx, k) * np.pi * h)                    # v_k(i), unnormalized
Vh = V * np.sqrt(2.0 * h)                                     # orthonormal columns
kappa = lam[-1] / lam[0]
alpha = 2.0 / (lam[0] + lam[-1])
rho_plain = (kappa - 1.0) / (kappa + 1.0)

# =============================================================================
# A. MODES
# =============================================================================
# A1: the sine/eigenvalue formulas, against the matrix itself and against eigh
err = np.abs(Ad @ V - V * lam).max() / lam[-1]
ok("A1a sine formula: A v_k == lam_k v_k, lam_k = 4 sin^2(k pi h/2)/h^2",
   err < 1e-13, err)
w_eigh = np.linalg.eigvalsh(Ad)
err = np.abs(w_eigh - lam).max() / lam[-1]
ok("A1b eigh(A) reproduces the closed-form eigenvalues", err < 1e-13, err)
err = np.abs(Vh.T @ Vh - np.eye(n)).max()
ok("A1c normalized sines are orthonormal: (2h)^{1/2} V has V^T V = I",
   err < 1e-12, err)

# A2: heat decay exp(-lam_k t) via expm (which knows nothing of our formulas)
t_half = math.log(2.0) / lam                                  # per-mode half-life
E1 = expm(-Ad * t_half[0])
c1 = Vh[:, 0] @ (E1 @ Vh[:, 0])
cn_at_t1 = Vh[:, -1] @ (E1 @ Vh[:, -1])
En = expm(-Ad * t_half[-1])
cn = Vh[:, -1] @ (En @ Vh[:, -1])
err = max(abs(c1 - 0.5), abs(cn - 0.5))
ok("A2a heat semigroup: mode k has half-life ln2/lam_k (k=1 and k=64, via expm)",
   err < 1e-10, err)
ok("A2b half-life ratio t_half(1)/t_half(64) == kappa; mode 64 is numerically "
   "dead (< 1e-12; exact value exp(-kappa ln 2) underflows) at mode 1's half-life",
   abs(t_half[0] / t_half[-1] - kappa) < 1e-9 * kappa and abs(cn_at_t1) < 1e-12,
   abs(t_half[0] / t_half[-1] - kappa) / kappa)

# A3: optimally damped Richardson/GD, DST-projected: coef_k(m) = (1-alpha lam_k)^m
# Known error mix: e_0 = sum_k vhat_k  (unit coefficient in every mode).
e0 = Vh @ np.ones(n)
e0_norm = np.linalg.norm(e0)                                  # = 8 = sqrt(n)
x_star = e0.copy()                                            # solve A x = b from x0 = 0
b = Ad @ x_star
dst_scale = 1.0 / (2.0 * math.sqrt((n + 1) / 2.0))            # DST-I -> Vh^T coords
err = np.abs(scipy.fft.dst(e0, type=1) * dst_scale - Vh.T @ e0).max()
ok("A3a DST-I with scale 1/sqrt(2(n+1)) == projection onto normalized sine modes",
   err < 1e-12, err)

m_gd = 400
x = np.zeros(n)
coefs = np.empty((m_gd + 1, n))
coefs[0] = scipy.fft.dst(x_star - x, type=1) * dst_scale
for m in range(1, m_gd + 1):
    x = x + alpha * (b - Ad @ x)
    coefs[m] = scipy.fft.dst(x_star - x, type=1) * dst_scale
pred = (1.0 - alpha * lam)[None, :] ** np.arange(m_gd + 1)[:, None]
err = np.abs(coefs - pred).max()
ok("A3b GD per-mode contraction measured == (1 - alpha lam_k)^m, all 64 modes, "
   "400 iterations", err < 1e-9, err)
per_mode_factor = 1.0 - alpha * lam
k_fastest = int(np.argmin(np.abs(per_mode_factor))) + 1
# lam_k + lam_{65-k} = 4/h^2 (sin^2 + cos^2), so modes k and 65-k contract at
# exactly the same speed under the optimal step: slowest pair {1,64}, fastest
# pair {32,33}.
pair_dev = np.abs(np.abs(per_mode_factor) - np.abs(per_mode_factor[::-1])).max()
ok("A3c mode pairing: |1-alpha lam_k| == |1-alpha lam_{65-k}| for every k "
   "(lam_k + lam_{65-k} = 4/h^2); slowest pair {1,64} at the rate, fastest "
   "pair {32,33}",
   pair_dev < 1e-12
   and abs(abs(per_mode_factor[0]) - rho_plain) < 1e-12
   and abs(per_mode_factor[0]) - np.abs(per_mode_factor[1:-1]).max() > 0
   and k_fastest in (32, 33),
   pair_dev)

# =============================================================================
# B. ROTATION
# =============================================================================
D = Vh.T @ Ad @ Vh
offdiag = np.abs(D - np.diag(np.diag(D))).max() / lam[-1]
ok("B1 V^T A V is diagonal (energy = 64 independent parabolas)",
   offdiag < 1e-13, offdiag)
err = np.abs(np.diag(D) - lam).max() / lam[-1]
ok("B2 the diagonal is lam_k (per-parabola stiffness)", err < 1e-13, err)

rng_b = np.random.default_rng(7)
b_test = rng_b.standard_normal(n)
x_dst = scipy.fft.dst(scipy.fft.dst(b_test, type=1) / lam, type=1) / (2.0 * (n + 1))
x_direct = spla.spsolve(Asp, b_test)
err = np.linalg.norm(x_dst - x_direct) / np.linalg.norm(x_direct)
ok("B3 one-pass DST-I solve (transform, divide by lam_k, transform back) "
   "== spsolve", err < 1e-12, err)

# =============================================================================
# shared runners
# =============================================================================
def richardson(minv_apply, omega, e_start, tol=1e-10, cap=25000, log_every=0):
    """Damped preconditioned Richardson on the error: e <- e - omega Minv A e.
    Returns (iterations to ||e||/||e0|| <= tol, subsampled log10 rel-err curve)."""
    e = e_start.copy()
    nrm0 = np.linalg.norm(e)
    curve = []
    for m in range(cap + 1):
        rel = np.linalg.norm(e) / nrm0
        if log_every and m % log_every == 0:
            curve.append([m, rel])
        if rel <= tol:
            return m, curve
        e = e - omega * minv_apply(Ad @ e)
    return cap + 1, curve


def spectrum_of(minv_dense):
    mu = np.linalg.eigvals(minv_dense @ Ad)
    assert np.abs(mu.imag).max() < 1e-6 * np.abs(mu.real).max()
    return np.sort(mu.real)


def ledger_entry(name, minv_dense, tol_iters_cap=25000):
    mu = spectrum_of(minv_dense)
    omega = 2.0 / (mu[0] + mu[-1])
    keff = mu[-1] / mu[0]
    scaled = omega * mu
    wrong = int(np.sum(np.abs(scaled - 1.0) > 0.1))
    wrong_below = int(np.sum(scaled < 0.9))
    wrong_above = int(np.sum(scaled > 1.1))
    iters, _ = richardson(lambda r: minv_dense @ r, omega, e0, cap=tol_iters_cap)
    _, res_hist = pcg(Ad, b, M=lambda r: minv_dense @ r, tol=1e-10, maxiter=500)
    pcg_iters = len(res_hist) - 1
    # which modes are left wrong: Rayleigh diagnostic in the sine basis
    q = np.einsum("ij,ij->j", Vh, omega * (minv_dense @ (Ad @ Vh)))
    wrong_modes = np.nonzero(np.abs(q - 1.0) > 0.1)[0] + 1
    entry = {
        "name": name,
        "kappa_eff": float(keff),
        "omega": float(omega),
        "mu_min": float(mu[0]), "mu_max": float(mu[-1]),
        "wrong": wrong, "wrong_below": wrong_below, "wrong_above": wrong_above,
        "iters_richardson": int(iters),
        "iters_pcg": int(pcg_iters),
        "rate": float((keff - 1.0) / (keff + 1.0)),
        "rayleigh_wrong_modes_first": int(wrong_modes[0]) if len(wrong_modes) else 0,
        "rayleigh_wrong_modes_last": int(wrong_modes[-1]) if len(wrong_modes) else 0,
        "rayleigh_wrong_count": int(len(wrong_modes)),
        "rayleigh_wrong_low_half": int(np.sum(wrong_modes <= n // 2)),
        "rayleigh_wrong_high_half": int(np.sum(wrong_modes > n // 2)),
    }
    return entry, mu, scaled


# =============================================================================
# C. PARTIAL PROJECTION (spectral deflation)
# =============================================================================
p_list = [0, 1, 2, 4, 8, 16]
defl = {}
spec_minv = {}
for p in p_list:
    alpha_p = 2.0 / (lam[p] + lam[-1])                        # lam[p] = lam_{p+1}
    Vp = Vh[:, :p]
    Minv = alpha_p * (np.eye(n) - Vp @ Vp.T)
    if p:
        Minv += Vp @ np.diag(1.0 / lam[:p]) @ Vp.T
    spec_minv[p] = Minv
    mu = spectrum_of(Minv)
    ones = np.sum(np.abs(mu - 1.0) < 1e-9)
    tail_expected = np.sort(alpha_p * lam[p:])
    tail_measured = mu[np.abs(mu - 1.0) >= 1e-9] if p else mu
    # p of the tail values can themselves be ~1 only accidentally; verify by
    # matching the full multiset instead:
    full_expected = np.sort(np.concatenate([np.ones(p), alpha_p * lam[p:]]))
    err_spec = np.abs(mu - full_expected).max()
    keff = mu[-1] / mu[0]
    keff_exact = lam[-1] / lam[p]
    iters, curve = richardson(lambda r, M=Minv: M @ r, 1.0, e0, log_every=0)
    # asymptotic tail slope == log rho_p: fit the last 50 iterations of a run
    # of length max(iters, 500) -- long runs are asymptotic at the 1e-10 gate,
    # short runs need the extension to shed sub-dominant modes
    e = e0.copy()
    hist = []
    for m in range(max(iters, 500)):
        e = e - Minv @ (Ad @ e)
        hist.append(np.linalg.norm(e))
    rho_p = (keff_exact - 1.0) / (keff_exact + 1.0)
    slope = (math.log(hist[-1]) - math.log(hist[-51])) / 50.0
    defl[p] = {
        "p": p,
        "alpha_p": float(alpha_p),
        "lam_p_plus_1": float(lam[p]),
        "kappa_eff": float(keff_exact),
        "kappa_eff_measured": float(keff),
        "rate": float(rho_p),
        "iters": int(iters),
        "eigs_at_1": int(ones),
        "spectrum_max_dev": float(err_spec),
        "ratio_vs_kappa_over_p1sq": float(keff_exact / (kappa / (p + 1) ** 2)),
        "tail_slope_measured": float(slope),
    }

err = max(d["spectrum_max_dev"] for d in defl.values())
ok("C1 deflated spectrum == {1 (x p)} U {alpha_p lam_k, k>p} for all six p",
   err < 1e-9, err)
ok("C2 every M_p^{-1} is already optimally damped: p eigenvalues pinned at 1, "
   "at least p for each p>0",
   all(defl[p]["eigs_at_1"] >= p for p in p_list))
err = max(abs(defl[p]["kappa_eff_measured"] - defl[p]["kappa_eff"])
          / defl[p]["kappa_eff"] for p in p_list)
ok("C3 kappa_eff measured == lam_n/lam_{p+1} for all p", err < 1e-9, err)
ratios = [defl[p]["ratio_vs_kappa_over_p1sq"] for p in p_list]
ok("C4 the (p+1)^2 payoff: kappa_eff/(kappa/(p+1)^2) in [1.0, 1.06] "
   "for p <= 16 (sublinearity of sin)",
   all(1.0 - 1e-12 <= r <= 1.06 for r in ratios), max(ratios) - 1.0)
err = max(abs(defl[p]["tail_slope_measured"] - math.log(defl[p]["rate"]))
          / abs(math.log(defl[p]["rate"])) for p in p_list)
ok("C5 measured GD tail slope == log((kappa_eff-1)/(kappa_eff+1)) "
   "for all p (rel err)", err < 1e-3, err)
it = [defl[p]["iters"] for p in p_list]
ok("C6 iterations to 1e-10 drop monotonically with p: "
   + " > ".join(str(v) for v in it), all(a > b for a, b in zip(it, it[1:])))

# =============================================================================
# D. SGD caricature (fixed seed, sigma known, constant step)
# =============================================================================
sigma = 1.0
seed_sgd = 0
cap_plain, cap_head = 12000, 3000
p_head = 4
Minv4 = spec_minv[p_head]
alpha4 = 2.0 / (lam[p_head] + lam[-1])

def stationary_floor(step_per_mode):
    """sqrt(E||e_inf||^2) for e_{m+1,k} = (1-s_k lam_k) e_{m,k} - s_k sigma xi."""
    g = 1.0 - step_per_mode * lam
    var = step_per_mode**2 * sigma**2 / (1.0 - g**2)
    return math.sqrt(var.sum()), var

floor_plain, var_plain = stationary_floor(np.full(n, alpha))
step_head = np.concatenate([1.0 / lam[:p_head], np.full(n - p_head, alpha4)])
floor_head, var_head = stationary_floor(step_head)

def run_sgd(minv, cap, floor, seed):
    rng = np.random.default_rng(seed)
    e = e0.copy()
    hit = None
    traj = []
    tail_sq = []
    for m in range(cap + 1):
        nrm = np.linalg.norm(e)
        traj.append(nrm)
        if hit is None and nrm <= 2.0 * floor:
            hit = m
        if m > cap - 2000:
            tail_sq.append(nrm**2)
        e = e - minv @ (Ad @ e + sigma * rng.standard_normal(n))
    return hit, traj, float(np.mean(tail_sq))

hit_plain, traj_plain, ms_plain = run_sgd(alpha * np.eye(n), cap_plain,
                                          floor_plain, seed_sgd)
hit_head, traj_head, ms_head = run_sgd(Minv4, cap_head, floor_head, seed_sgd)

ok(f"D1 plain SGD reaches 2x its noise floor in {hit_plain} iterations; "
   f"with the p=4 head solved exactly, {hit_head} -- transient collapses "
   f"{hit_plain / hit_head:.0f}x",
   hit_plain is not None and hit_head is not None
   and hit_plain > 20 * hit_head)
ok("D2 measured stationary mean-square error within [0.6, 1.6]x the exact "
   "per-mode formula, both runs",
   0.6 < ms_plain / floor_plain**2 < 1.6 and 0.6 < ms_head / floor_head**2 < 1.6,
   max(abs(ms_plain / floor_plain**2 - 1), abs(ms_head / floor_head**2 - 1)))
ok("D3 the price of the exact head: its noise floor is higher "
   f"({floor_head:.4f} vs {floor_plain:.4f}) -- Newton amplifies noise in "
   "flat directions; head-mode stationary variance sigma^2/lam_k^2",
   floor_head > 10 * floor_plain, floor_head / floor_plain)

# =============================================================================
# E. THE LEDGER
# =============================================================================
ledger = {}

# (1) none
ledger["none"], mu_none, _ = ledger_entry("none (optimally damped GD)", np.eye(n))
ok("E1 none: kappa_eff == kappa == lam_n/lam_1",
   abs(ledger["none"]["kappa_eff"] - kappa) / kappa < 1e-12,
   abs(ledger["none"]["kappa_eff"] - kappa) / kappa)

# (2) spectral deflation p = 4, 8
for p in (4, 8):
    ledger[f"spectral_p{p}"], _, _ = ledger_entry(f"spectral deflation p={p}",
                                                  spec_minv[p])
ok("E2 spectral deflation ledger iterations match section C runs (p=4, p=8)",
   ledger["spectral_p4"]["iters_richardson"] == defl[4]["iters"]
   and ledger["spectral_p8"]["iters_richardson"] == defl[8]["iters"])

# (3) coarse-average deflation, same formula, block-average Z
angles_deg = {}
for p in (4, 8):
    bs = n // p
    Z = np.zeros((n, p))
    for j in range(p):
        Z[j * bs:(j + 1) * bs, j] = 1.0 / math.sqrt(bs)       # orthonormal
    alpha_p = 2.0 / (lam[p] + lam[-1])
    W = Z.T @ Ad @ Z
    Minv = Z @ np.linalg.inv(W) @ Z.T + alpha_p * (np.eye(n) - Z @ Z.T)
    ledger[f"coarse_p{p}"], _, _ = ledger_entry(f"coarse averages p={p}", Minv)
    ang = np.degrees(subspace_angles(Z, Vh[:, :p]))
    angles_deg[p] = sorted(float(a) for a in ang)
ok("E3a coarse-average space vs true bottom-p eigenspace: largest principal "
   f"angle {angles_deg[4][-1]:.2f} deg (p=4), {angles_deg[8][-1]:.2f} deg (p=8) "
   "-- cheap imitations, never orthogonal (< 45 deg)",
   angles_deg[4][-1] < 45.0 and angles_deg[8][-1] < 45.0)
ok("E3b coarse deflation sits between spectral and none: "
   f"{ledger['spectral_p4']['iters_richardson']} < "
   f"{ledger['coarse_p4']['iters_richardson']} < "
   f"{ledger['none']['iters_richardson']} (p=4, same for p=8)",
   ledger["spectral_p4"]["iters_richardson"] < ledger["coarse_p4"]["iters_richardson"]
   < ledger["none"]["iters_richardson"]
   and ledger["spectral_p8"]["iters_richardson"] < ledger["coarse_p8"]["iters_richardson"]
   < ledger["none"]["iters_richardson"])

# (4) Nystrom rank 4, 8 (deflates the TOP -- 07's lesson relocated to 1D)
ny_top = {}
for p in (4, 8):
    ny = NystromPreconditioner(Asp, rank=p, mu=0.0, seed=0)
    Minv = np.column_stack([ny.apply(col) for col in np.eye(n)])
    ledger[f"nystrom_p{p}"], _, _ = ledger_entry(f"Nystrom rank {p}", Minv)
    ny_top[p] = {"lams": [float(v) for v in ny.lams],
                 "true_top": [float(v) for v in lam[::-1][:p]],
                 "lam_ell": float(ny.lam_ell)}
ok("E4a Nystrom eigenvalue estimates undershoot the TRUE TOP eigenvalues "
   "(Lemma 2.1) -- it deflates the top, the harmless end",
   all(ny_top[p]["lams"][j] <= ny_top[p]["true_top"][j] + 1e-6
       for p in (4, 8) for j in range(p)))
ok("E4b Nystrom leaves the BOTTOM wrong: kappa_eff(p=8) still > 0.9 kappa, "
   f"iterations {ledger['nystrom_p8']['iters_richardson']} vs "
   f"{ledger['none']['iters_richardson']} plain -- barely improves",
   ledger["nystrom_p8"]["kappa_eff"] > 0.9 * kappa
   and ledger["nystrom_p8"]["iters_richardson"] > 0.9 * ledger["none"]["iters_richardson"])
ok("E4c Nystrom's wrong directions include the lowest mode k=1 "
   "(Rayleigh diagnostic), unlike spectral deflation whose k=1..p are exact",
   ledger["nystrom_p8"]["rayleigh_wrong_modes_first"] == 1
   and ledger["spectral_p8"]["rayleigh_wrong_modes_first"] > 8)

# (5) block-Jacobi, 2 subdomains (exact half-solves)
Mbj = np.zeros_like(Ad)
Mbj[:n // 2, :n // 2] = Ad[:n // 2, :n // 2]
Mbj[n // 2:, n // 2:] = Ad[n // 2:, n // 2:]
Minv_bj = np.linalg.inv(Mbj)
ledger["block_jacobi_2"], mu_bj, _ = ledger_entry("block-Jacobi, 2 subdomains",
                                                  Minv_bj)
outliers = mu_bj[np.abs(mu_bj - 1.0) > 1e-8]
ok("E5a block-Jacobi(2): exactly 2 eigenvalues of M^{-1}A differ from 1 "
   "(the rank-2 symmetric interface coupling of the single separator bond)",
   len(outliers) == 2)
err = max(abs(outliers[0] - (1.0 - 32.0 / 33.0)), abs(outliers[1] - (1.0 + 32.0 / 33.0)))
ok("E5b the outliers are exactly 1 +/- 32/33, so kappa_eff = 65 = n+1",
   err < 1e-9 and abs(ledger["block_jacobi_2"]["kappa_eff"] - 65.0) < 1e-6, err)
ok("E5c clustering vs range (report 13): CG pays per CLUSTER -- PCG in "
   f"{ledger['block_jacobi_2']['iters_pcg']} iterations (<= 4); Richardson pays "
   f"for the RANGE -- {ledger['block_jacobi_2']['iters_richardson']} iterations "
   "(>= 500) at rate 32/33",
   ledger["block_jacobi_2"]["iters_pcg"] <= 4
   and ledger["block_jacobi_2"]["iters_richardson"] >= 500)

# (6) IC(0): exact on the chain (no fill-in on a tree)
L = ic0(Ad)
err = np.abs(L @ L.T - Ad).max() / np.abs(Ad).max()
ok("E6a IC(0) is EXACT on the chain: L L^T == A on the tridiagonal pattern "
   "(zero fill-in on a tree/path)", err < 1e-14, err)
Minv_ic = np.linalg.inv(L @ L.T)
ledger["ic0"], _, _ = ledger_entry("IC(0)", Minv_ic)
ok("E6b IC(0): zero directions left wrong, Richardson converges in 1 iteration",
   ledger["ic0"]["wrong"] == 0 and ledger["ic0"]["iters_richardson"] == 1)

npass = sum(c["pass"] for c in results["checks"])
print(f"\n{npass}/{len(results['checks'])} PASS")

# ---- every number the report quotes -----------------------------------------
gd_fig_modes = [1, 8, 33, 64]
gd_fig = {str(kk): {
    "factor": float(abs(per_mode_factor[kk - 1])),
    "signed_factor": float(per_mode_factor[kk - 1]),
    "measured": [[m, float(abs(coefs[m, kk - 1]))] for m in range(0, m_gd + 1, 5)],
} for kk in gd_fig_modes}

sub_plain = [[m, float(traj_plain[m] / e0_norm)] for m in range(0, cap_plain + 1, 20)]
sub_head = [[m, float(traj_head[m] / e0_norm)] for m in range(0, cap_head + 1, 5)]

results["quoted"] = {
    "n": n, "h": h, "n_plus_1": n + 1, "two_n_plus_2": 2 * (n + 1),
    "h2_inv": (n + 1) ** 2, "four_over_h2": 4.0 * (n + 1) ** 2,
    "half_n": n // 2, "half_n_plus_1": n // 2 + 1,
    "tol": 1e-10, "e0_slow_prefactor": math.sqrt(2.0) / 8.0,
    "iters_drop_factor_p16": defl[0]["iters"] / defl[16]["iters"],
    "coarse_vs_spectral_p4": ledger["coarse_p4"]["iters_richardson"]
                             / ledger["spectral_p4"]["iters_richardson"],
    "coarse_vs_spectral_p8": ledger["coarse_p8"]["iters_richardson"]
                             / ledger["spectral_p8"]["iters_richardson"],
    "bj_richardson_over_cg": ledger["block_jacobi_2"]["iters_richardson"]
                             / ledger["block_jacobi_2"]["iters_pcg"],
    "lam_1": float(lam[0]), "lam_2": float(lam[1]), "lam_3": float(lam[2]),
    "lam_4": float(lam[3]), "lam_5": float(lam[4]), "lam_9": float(lam[8]),
    "lam_17": float(lam[16]), "lam_33": float(lam[32]), "lam_n": float(lam[-1]),
    "pi_sq": math.pi**2,
    "kappa": float(kappa),
    "alpha": float(alpha),
    "rate_plain": float(rho_plain),
    "t_half_1": float(t_half[0]), "t_half_n": float(t_half[-1]),
    "t_half_ratio": float(t_half[0] / t_half[-1]),
    "e0_norm": float(e0_norm),
    "gd_check": {"iters_run": m_gd, "max_dev": float(np.abs(coefs - pred).max()),
                 "modes_shown": gd_fig_modes, "per_mode": gd_fig,
                 "k_slowest_pair": [1, n], "k_fastest": k_fastest},
    "dst_solve_relerr": float(np.linalg.norm(x_dst - x_direct)
                              / np.linalg.norm(x_direct)),
    "deflation": {str(p): defl[p] for p in p_list},
    "sgd": {
        "sigma": sigma, "seed": seed_sgd, "p_head": p_head,
        "step_plain": float(alpha), "step_tail": float(alpha4),
        "floor_plain": float(floor_plain), "floor_head": float(floor_head),
        "floor_ratio": float(floor_head / floor_plain),
        "hit_2x_plain": int(hit_plain), "hit_2x_head": int(hit_head),
        "collapse_factor": float(hit_plain / hit_head),
        "ms_over_floor_plain": float(ms_plain / floor_plain**2),
        "ms_over_floor_head": float(ms_head / floor_head**2),
        "ms_dev_plain_pct": float(abs(ms_plain / floor_plain**2 - 1.0) * 100.0),
        "ms_dev_head_pct": float(abs(ms_head / floor_head**2 - 1.0) * 100.0),
        "head_var_share": float(var_head[:p_head].sum() / var_head.sum()),
        "traj_plain": sub_plain, "traj_head": sub_head,
        "cap_plain": cap_plain, "cap_head": cap_head,
        "floor_plain_rel": float(floor_plain / e0_norm),
        "floor_head_rel": float(floor_head / e0_norm),
        "notes": "Quadratic caricature only: constant-step GD with additive "
                 "isotropic gradient noise on the chain quadratic. It shows why "
                 "curvature-aware preconditioning shortens the low-curvature "
                 "transient in stochastic optimization, and that the exact head "
                 "solve trades a higher noise floor for a shorter transient. "
                 "No broader ML claims.",
    },
    "ledger": ledger,
    "principal_angles_deg": {str(p): angles_deg[p] for p in (4, 8)},
    "nystrom_eigs": ny_top,
    "block_jacobi": {"outlier_lo": float(outliers[0]), "outlier_hi": float(outliers[1]),
                     "ratio_32_33": 32.0 / 33.0, "kappa_eff": 65.0,
                     "rate": 32.0 / 33.0},
    "checks_total": len(results["checks"]),
    "checks_passed": npass,
}

out = Path(__file__).resolve().parents[2] / "results" / "eigenmodes.json"
out.write_text(json.dumps(results, indent=2))
print(f"wrote {out}")
