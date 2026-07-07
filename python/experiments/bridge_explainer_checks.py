"""Numerical verification of every number displayed by reports/bridge_explainer.html.

The interactive rebuild of the bridge explainer adds four labs (tent lab,
paths lab, pin-the-walk lab, assassin lab) plus a three-domain analogy
section.  This script backs every number those additions display:

  tent lab      -- nodal exactness: the columns of Sigma/h land exactly on the
                   continuum tent G(x,y) = min(x,y) - xy at n = 5 (exact
                   Fraction arithmetic AND the same Thomas-algorithm solve the
                   page runs in JS) and at n = 31; the FIG2 fractions 1/24,
                   1/54, 1/216, 5/216; slopes 1-y / -y and peak y(1-y).
  paths lab     -- Var B_t = t(1-t) exactly on the discrete walk; +-2 sigma
                   POINTWISE coverage 2*Phi(2)-1 = 0.95450 measured by seeded
                   Monte Carlo (B = W - tW_1, W = scaled cumulative sums);
                   whole-path coverage measured for honesty.
  pin lab       -- Cov(W_t,W_1) = t = t Var(W_1), so tW_1 is the least-squares
                   regression of the walk on its endpoint; the matrix identity
                   P M P^T == min(t_i,t_j) - t_i t_j with P = I - t e_m^T,
                   exact in Fractions (m = 8) and to machine eps (m = 64);
                   the four bilinearity terms min(s,t), -ts, -st, +st.
  three domains -- resistance R_eff(y) = y(1-y) and the 1-y / y current
                   divider; taut-string force balance: shear V = u' is
                   piecewise constant (1-y, -y), jumps by -1 at the load,
                   reactions sum to the load; Maxwell reciprocity.
  assassin lab  -- block-Jacobi on the n = 64 chain, p = 2/4/8:
                   rank(A - M) = 2(p-1); exactly 2(p-1) eigenvalue outliers
                   plus a cluster at 1; kappa(M^{-1}A) = 112.15 at p = 4
                   (the page's old static figure said "112"); PCG converges in
                   2(p-1)+1 iterations (7 at p = 4); plain CG needs all 64;
                   preconditioned steepest descent stalls ~1e-4 after 400.
                   Full error curves + spectra are exported for embedding.

Run from the repo root:
    uv run python python/experiments/bridge_explainer_checks.py

Expected output: all PASS; writes results/bridge_explainer.json with every
quoted number and the curve data embedded in the page.
"""
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.linalg import cholesky, eigvalsh, inv, matrix_rank, norm, solve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcg import pcg
from poisson import laplacian_1d

results = {"checks": []}


def ok(name, cond, err=None):
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {name}" + (f"   (max err {err:.3g})" if err is not None else ""))
    results["checks"].append({"name": name, "pass": bool(cond),
                              **({"max_err": float(err)} if err is not None else {})})


def frac_inv(M):
    """Exact inverse of a list-of-lists Fraction matrix by Gauss-Jordan."""
    m = len(M)
    aug = [row[:] + [Fraction(int(i == j)) for j in range(m)] for i, row in enumerate(M)]
    for c in range(m):
        piv = next(r for r in range(c, m) if aug[r][c] != 0)
        aug[c], aug[piv] = aug[piv], aug[c]
        pv = aug[c][c]
        aug[c] = [v / pv for v in aug[c]]
        for r in range(m):
            if r != c and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[c])]
    return [row[m:] for row in aug]


def thomas(n, h, j):
    """Solve (tridiag(-1,2,-1)/h^2) u = e_j / h -- the exact algorithm the
    page's tent-lab JS runs (Thomas forward sweep / back substitution)."""
    a = np.full(n, -1.0) / h**2   # sub-diagonal
    bd = np.full(n, 2.0) / h**2   # diagonal
    c = np.full(n, -1.0) / h**2   # super-diagonal
    d = np.zeros(n)
    d[j] = 1.0 / h
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[0] = c[0] / bd[0]
    dp[0] = d[0] / bd[0]
    for i in range(1, n):
        mden = bd[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / mden
        dp[i] = (d[i] - a[i] * dp[i - 1]) / mden
    u = np.zeros(n)
    u[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        u[i] = dp[i] - cp[i] * u[i + 1]
    return u


# ============================ 1 · TENT LAB ==================================
# 1a. n = 5 exact: Sigma = A^{-1} in Fraction arithmetic equals h x_i (1-x_j)
n5 = 5
h5 = Fraction(1, 6)
A5 = [[Fraction(0)] * n5 for _ in range(n5)]
for i in range(n5):
    A5[i][i] = 2 / h5**2                      # 72
    if i > 0:
        A5[i][i - 1] = -1 / h5**2             # -36
    if i < n5 - 1:
        A5[i][i + 1] = -1 / h5**2
S5 = frac_inv(A5)
x5 = [Fraction(i, 6) for i in range(1, 6)]
exact5 = all(S5[i][j] == h5 * x5[min(i, j)] * (1 - x5[max(i, j)])
             for i in range(n5) for j in range(n5))
ok("n=5 exact: Sigma_ij == h x_i(1-x_j), all 25 fractions (Gauss-Jordan in Q)", exact5)

# the four fractions the kept FIG2 caption quotes
quoted = (S5[2][2] == Fraction(1, 24) and S5[1][3] == Fraction(1, 54)
          and S5[0][4] == Fraction(1, 216) and S5[0][0] == Fraction(5, 216))
ok("FIG2 fractions: Sigma_33=1/24, Sigma_24=1/54, Sigma_15=1/216, Sigma_11=5/216", quoted)

# 1b + 1c. dots-on-tent by the page's own Thomas solve, n = 5 and n = 31
dots_gap = {}
for n in (5, 31):
    h = 1.0 / (n + 1)
    x = np.arange(1, n + 1) / (n + 1)
    gap = 0.0
    for j in range(n):
        u = thomas(n, h, j)                   # == column j of Sigma / h
        tent = np.minimum(x, x[j]) - x * x[j]  # G(x_i, x_j)
        gap = max(gap, np.abs(u - tent).max())
    dots_gap[n] = gap
    ok(f"tent lab, n={n}: Thomas-solve dots land on min(x,y)-xy (all {n} sources)",
       gap < 1e-14, gap)

# 1d. slopes and peak read out by the lab: 1-y, -y, y(1-y) from the solve
n, h = 31, 1.0 / 32
x = np.arange(1, n + 1) / 32
err = 0.0
for j in (4, 9, 15, 24):                      # a spread of slider positions
    y = x[j]
    u = np.concatenate(([0.0], thomas(n, h, j), [0.0]))
    dl, dr = (u[1] - u[0]) / h, (u[j + 2] - u[j + 1]) / h
    err = max(err, abs(dl - (1 - y)), abs(dr - (-y)), abs(u[j + 1] - y * (1 - y)))
ok("tent lab readouts: slopes 1-y / -y, peak y(1-y) at 4 slider stops", err < 1e-12, err)

# ============================ 2 · PIN LAB ===================================
# 2a. regression fact: on the discrete walk, Cov(W_t, W_1) = min(t,1) = t
#     = t Var(W_1), so the least-squares predictor of W_t from W_1 is t W_1.
m = 64
t = np.arange(1, m + 1) / m                   # t_m = 1: the endpoint IS a node
M = np.minimum.outer(t, t)                    # Cov(W_ti, W_tj) = min(ti,tj)
beta = M[:, -1] / M[-1, -1]                   # regression coefficients on W_1
err = np.abs(beta - t).max()
ok("pin lab: Cov(W_t,W_1)/Var(W_1) == t -- the dashed line IS the regression",
   err < 1e-15, err)

# 2b. residual covariance, exact: P M P^T == min(t_i,t_j) - t_i t_j,
#     P = I - t e_m^T, in Fraction arithmetic at m = 8
mf = 8
tf = [Fraction(i, mf) for i in range(1, mf + 1)]
Mf = [[min(a, b) for b in tf] for a in tf]
Pf = [[Fraction(int(i == j)) - (tf[i] if j == mf - 1 else 0)
       for j in range(mf)] for i in range(mf)]
PM = [[sum(Pf[i][k] * Mf[k][j] for k in range(mf)) for j in range(mf)] for i in range(mf)]
PMP = [[sum(PM[i][k] * Pf[j][k] for k in range(mf)) for j in range(mf)] for i in range(mf)]
exact_pin = all(PMP[i][j] == min(tf[i], tf[j]) - tf[i] * tf[j]
                for i in range(mf) for j in range(mf))
ok("pin lab, exact: P min P^T == min - t_i t_j, P = I - t e_m^T (m=8, in Q)", exact_pin)

# 2c. same identity in float64 at m = 64
P = np.eye(m) - np.outer(t, np.eye(m)[-1])
B = P @ M @ P.T
err = np.abs(B - (np.minimum.outer(t, t) - np.outer(t, t))).max()
ok("pin lab, m=64: P M P^T == min - t_i t_j to machine eps", err < 1e-14, err)

# 2d. the four bilinearity terms, one by one, at (s,t) = (1/4, 5/8)
s_, t_ = 0.25, 0.625
i_, j_ = int(s_ * m) - 1, int(t_ * m) - 1
terms = {
    "cov_WsWt": M[i_, j_],                    # min(s,t)
    "minus_t_cov_WsW1": -t_ * M[i_, -1],      # -t s
    "minus_s_cov_W1Wt": -s_ * M[-1, j_],      # -s t
    "plus_st_varW1": s_ * t_ * M[-1, -1],     # +s t
}
err = max(abs(terms["cov_WsWt"] - min(s_, t_)), abs(terms["minus_t_cov_WsW1"] + t_ * s_),
          abs(terms["minus_s_cov_W1Wt"] + s_ * t_), abs(terms["plus_st_varW1"] - s_ * t_),
          abs(sum(terms.values()) - (min(s_, t_) - s_ * t_)))
ok("bilinearity: four terms = min(s,t), -ts, -st, +st; sum = min(s,t)-st", err < 1e-15, err)

# ============================ 3 · PATHS LAB =================================
# 3a. discrete bridge variance is EXACTLY t(1-t): Var(B_ti) = diag of P M P^T
err = np.abs(np.diag(B) - t * (1 - t)).max()
ok("paths lab: Var(B_t) == t(1-t) exactly -- the gold parabola", err < 1e-14, err)

# 3b. and the same parabola is the diagonal of Sigma/h on the n=31 chain
n, h = 31, 1.0 / 32
xg = np.arange(1, n + 1) / 32
Sg = inv((laplacian_1d(n) / h**2).toarray())
err = np.abs(np.diag(Sg) / h - xg * (1 - xg)).max()
ok("paths lab: diag(Sigma)/h == x(1-x) -- variance IS the tent-peak envelope",
   err < 1e-14, err)

# 3c. +-2 sigma POINTWISE coverage by seeded Monte Carlo.
#     Bridge built exactly as in the page's JS: W = sqrt(dt)*cumsum(N(0,1)),
#     B = W - t W_1.  Coverage per (path, location) should be 2 Phi(2) - 1.
rng = np.random.default_rng(20260706)
m = 64
tmc = np.arange(1, m + 1) / m
sig = np.sqrt(tmc[:-1] * (1 - tmc[:-1]))      # interior sd; endpoint excluded
N = 200_000
inside = 0
whole = 0
pts = 0
for _ in range(20):                           # 20 chunks of 10k paths
    g = rng.standard_normal((N // 20, m))
    W = np.cumsum(g, axis=1) * np.sqrt(1.0 / m)
    Bb = W - np.outer(W[:, -1], tmc)          # B_ti; B at t_m = 1 is 0 exactly
    Bi = Bb[:, :-1]
    okmask = np.abs(Bi) <= 2 * sig
    inside += okmask.sum()
    whole += okmask.all(axis=1).sum()
    pts += Bi.size
cov_point = inside / pts
cov_path = whole / N
target = 2 * (0.5 * (1 + math.erf(2 / math.sqrt(2)))) - 1   # 0.954500
ok(f"paths lab: pointwise +-2sigma coverage {cov_point:.4f} vs 2*Phi(2)-1 = {target:.4f} "
   f"(200k seeded paths)", abs(cov_point - target) < 2e-3, abs(cov_point - target))
print(f"      (whole-path coverage: only {cov_path:.3f} of paths stay inside the "
      f"band everywhere -- the band is pointwise, not a path envelope)")

# ============================ 4 · THREE DOMAINS =============================
# 4a. resistance: unit current at y, both ends grounded; peak voltage =
#     R_eff(y) = y(1-y) = y || (1-y); near-wall current split 1-y / y.
err = 0.0
for j in range(n):
    y = xg[j]
    u = np.concatenate(([0.0], thomas(n, h, j), [0.0]))
    Reff = u[j + 1]
    i_left, i_right = u[1] / h, u[-2] / h     # currents into the two grounds
    par = 1.0 / (1.0 / y + 1.0 / (1 - y))
    err = max(err, abs(Reff - y * (1 - y)), abs(Reff - par),
              abs(i_left - (1 - y)), abs(i_right - y), abs(i_left + i_right - 1))
ok("resistance: R_eff(y) == y(1-y) == y||(1-y); divider splits 1-y / y (all 31 y)",
   err < 1e-12, err)

# 4b. taut string at unit tension, unit transverse point load at y:
#     deflection = tent; internal shear V = u' is piecewise CONSTANT,
#     equal to 1-y then -y, and JUMPS by -1 across the load (force balance);
#     support reactions (1-y) + y = 1 carry the load.
j = 9
y = xg[j]
u = np.concatenate(([0.0], thomas(n, h, j), [0.0]))
V = np.diff(u) / h                            # shear diagram (32 segments)
VL, VR = V[:j + 1], V[j + 1:]
err = max(np.abs(VL - (1 - y)).max(), np.abs(VR - (-y)).max())
ok("string: shear V = u' piecewise constant, 1-y then -y (the shear diagram "
   "IS the flux diagram)", err < 1e-12, err)
jump = VR[0] - VL[0]
react_sum = VL[0] + (-VR[-1])                 # R_left + R_right
err = max(abs(jump + 1), abs(react_sum - 1))
ok("string force balance: shear jumps by -1 at the load; reactions (1-y)+y = 1",
   err < 1e-12, err)

# 4c. Maxwell reciprocity: deflection at x from a load at y == deflection at
#     y from a load at x (influence-line symmetry) -- Sigma is symmetric.
err = np.abs(Sg - Sg.T).max()
ok("Maxwell reciprocity: G(x,y) == G(y,x) (influence lines are symmetric)",
   err < 1e-14, err)

# ============================ 5 · ASSASSIN LAB ==============================
n64 = 64
h64 = 1.0 / 65
A = (laplacian_1d(n64) / h64**2).toarray()
b = np.random.RandomState(0).randn(n64)       # canonical seeded right-hand side
xstar = solve(A, b)
xnorm = norm(xstar)


def block_jacobi(A, p):
    Mp = np.zeros_like(A)
    bs = A.shape[0] // p
    for k in range(p):
        sl = slice(k * bs, (k + 1) * bs)
        Mp[sl, sl] = A[sl, sl]
    return Mp


lab = {}
kappas = {}
for p in (2, 4, 8):
    Mp = block_jacobi(A, p)
    Minv = inv(Mp)
    L = cholesky(Mp)
    S = solve(L, solve(L, A).T).T             # L^{-1} A L^{-T}, similar to M^{-1}A
    ev = eigvalsh((S + S.T) / 2)
    kappas[p] = ev[-1] / ev[0]
    n_out = int(np.sum(np.abs(ev - 1) > 1e-8))
    rank_E = int(matrix_rank(A - Mp))
    ok(f"assassin p={p}: rank(A-M) == 2(p-1) == {2*(p-1)} and spectrum = cluster "
       f"at 1 + {n_out} outliers", rank_E == 2 * (p - 1) and n_out == 2 * (p - 1))

    # PCG iteration count at tol 1e-10 (the page's claim: 2(p-1)+1).
    # In exact arithmetic CG terminates at the number of DISTINCT eigenvalues,
    # 2(p-1)+1; in float64 the plunge lands exactly there but can miss a tight
    # tolerance gate by a hair (at p=8 it hits 1.08e-10 at iter 15, so the
    # counter reads 16).  We check both facts separately and quote both.
    kthy = 2 * (p - 1) + 1
    Mfun = lambda r: Minv @ r
    _, rh = pcg(A, b, M=Mfun, tol=1e-10, maxiter=200)
    it_pcg = len(rh) - 1
    res_at_thy = rh[kthy]
    ok(f"assassin p={p}: PCG residual plunges at iteration {kthy} == 2(p-1)+1 "
       f"(relres there {res_at_thy:.2e}; counter at tol 1e-10 reads {it_pcg})",
       res_at_thy < 5e-9 and it_pcg <= kthy + 1)
    if p in (2, 4):
        ok(f"assassin p={p}: iteration counter == {kthy} exactly at tol 1e-10",
           it_pcg == kthy)

    # PCG relative-ERROR curve for the race plot (run past the tol floor)
    xh = []
    _, _ = pcg(A, b, M=Mfun, tol=1e-15, maxiter=40, x_hist=xh)
    cg_err = [max(float(norm(xk - xstar) / xnorm), 1e-17) for xk in xh]

    # preconditioned steepest descent, 400 iterations, same M, same b
    xk = np.zeros(n64)
    r = b.copy()
    sd_err = [1.0]
    for _ in range(400):
        z = Minv @ r
        a = (r @ z) / (z @ (A @ z))
        xk = xk + a * z
        r = r - a * (A @ z)
        sd_err.append(float(norm(xk - xstar) / xnorm))
    lab[p] = {"eig": [float(v) for v in ev], "kappa": float(kappas[p]),
              "outliers": n_out, "rank_E": rank_E, "pcg_iters": it_pcg,
              "k_theory": kthy, "relres_at_theory": float(res_at_thy),
              "cg_err": cg_err, "sd_err": sd_err,
              "sd_final": sd_err[-1]}

ok(f"assassin p=4: kappa(M^-1 A) = {kappas[4]:.2f} (page's static figure said "
   f"112)", abs(kappas[4] - 112.15) < 0.1)

# plain CG: needs all 64 iterations, error curve for the race plot
xh = []
_, rh = pcg(A, b, M=None, tol=1e-10, maxiter=200, x_hist=xh)
it_plain = len(rh) - 1
ok(f"assassin: plain CG needs {it_plain} == n == 64 iterations", it_plain == 64)
plain_err = [max(float(norm(xk - xstar) / xnorm), 1e-17) for xk in xh]

# SD stall: same order as the old figure's 3.5e-4 (its b came from the
# external make_figs.py and is not reproducible; we quote OUR measured value)
sd4 = lab[4]["sd_final"]
ok(f"assassin p=4: SD relative error after 400 iters = {sd4:.2e} -- stalls at "
   f"the 1e-4 scale (old figure quoted 3.5e-4 with an unknown rhs)",
   1e-5 < sd4 < 1e-3)

npass = sum(c["pass"] for c in results["checks"])
ntot = len(results["checks"])
print(f"\n{npass}/{ntot} PASS")

# ---- every number the page displays + the embedded curve data --------------
results["quoted"] = {
    "checks_total": ntot, "checks_passed": npass,
    "tent": {"n5_dots_gap": dots_gap[5], "n31_dots_gap": dots_gap[31],
             "fractions": {"center": "1/24", "skew": "1/54",
                           "corner": "1/216", "first": "5/216"}},
    "paths": {"pointwise_coverage": cov_point, "target_2phi2_minus_1": target,
              "whole_path_coverage": cov_path, "n_paths_mc": N, "m_steps": m},
    "pin": {"m_exact": mf, "m_float": 64},
    "domains": {"y_example": float(y), "slope_left": float(1 - y),
                "slope_right": float(-y), "Reff": float(y * (1 - y))},
    "assassin": {
        "n": n64, "b": "RandomState(0).randn(64)",
        "kappa": {p: float(kappas[p]) for p in (2, 4, 8)},
        "pcg_iters": {p: lab[p]["pcg_iters"] for p in (2, 4, 8)},
        "plain_cg_iters": it_plain,
        "sd_final": {p: lab[p]["sd_final"] for p in (2, 4, 8)},
        "old_figure_quotes": {"kappa_p4": 112, "sd_after_400": 3.5e-4,
                              "note": "make_figs.py external; rhs unknown; "
                                      "measured values quoted in the page"},
    },
}
results["embed"] = {"p": {p: lab[p] for p in (2, 4, 8)}, "plain_err": plain_err}

out = Path(__file__).resolve().parents[2] / "results" / "bridge_explainer.json"
out.write_text(json.dumps(results, indent=1))
print(f"wrote {out}")
