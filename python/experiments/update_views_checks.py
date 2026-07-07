"""Numerical verification for the update-views explainer (one update, four languages).

The system is the suite's 1-D Dirichlet chain A = laplacian_1d(n)/h^2 with
n = 32, h = 1/33 (constant diagonal 2/h^2: grounded walls, cf. green-tents
SS4, "Free versus grounded").  The page unifies four readings of one
stationary update and this script
machine-checks every number it will quote:

1. SPLITTINGS  -- for Richardson(alpha_opt) / Jacobi / GS / SOR(omega_opt) /
                  SSOR(omega_opt) / IC-rank-T (pivoted partial Cholesky +
                  diagonal residual, T in {4,8,16,31}) / two-level (additive
                  Jacobi + 8 block-average coarse): the component/operational
                  loop == x_{k+1} = M^{-1}(N x_k + b) == x + M^{-1}(b - A x),
                  with M, N = M - A built explicitly.
2. EXACT RATES -- rho(T_J) = cos(pi h), rho(T_GS) = cos^2(pi h),
                  omega_opt = 2/(1+sin pi h) == 2/(1+sqrt(1-mu^2)),
                  rho(T_SOR(omega_opt)) = omega_opt - 1; the SOR omega-curve
                  on a grid in (0,2) against the consistently-ordered
                  quadratic-root formula, kink at omega_opt; measured
                  asymptotic slopes (renormalized power iteration) match each
                  rho to ~1%; iterations-to-1e-8 on the green-tents rhs
                  (unit heat in at node 8, out at node 24, scaled 1/h).
3. WHITENED GD -- (V3) for the SPD M's (Jacobi, SSOR, IC-rank-T, two-level):
                  M = C^T C, plain step-1 GD on C^{-T}A C^{-1} z = C^{-T} b
                  mapped back by C^{-1} coincides with M-preconditioned
                  Richardson at every iterate (report 15 SS8's move for
                  stationary methods).  Negative: M_GS is nonsymmetric and GS
                  == exact cyclic coordinate descent (each micro-step zeroes
                  dJ/dx_i); SOR micro-step = omega times the CD step.
4. SHAWE-TAYLOR -- Algorithm 5.12 verbatim (R recursion, d bookkeeping,
                  nu_j = sqrt(d_j), pivot on max d) run on the bridge
                  covariance K = Sigma = A^{-1} AND on A itself: full run
                  reproduces chol up to the pivot permutation, d == Schur
                  diagonal at every step, pivot sequences recorded
                  (near-bisection on Sigma, odd/red-black sweep on A);
                  rank-T factor + diagonal residual as the factor-analysis
                  surrogate M (09 SS6): kappa(M^{-1}A) and iterations vs T;
                  duality: (pivoted and unpivoted) Gram-Schmidt on the
                  grounded B's columns == (pivoted and partial) Cholesky on
                  A = B^T B (09 SS4.3, Shawe-Taylor SS5.2).
5. ELECTRICAL  -- A == B^T B with the grounded (n+1) x n difference matrix
                  (+-1/h); Jacobi = neighbor-voltage average + h^2/2 source
                  lift; GD on (1/2)||Bx - y||^2 with B^T y = b == Richardson
                  on the normal equations (user's Trefethen margin note);
                  Jacobi == diagonal Newton on the same least-squares
                  objective (Hessian diag(B^T B) = D).
6. TWO-LEVEL   -- additive Jacobi + 8 block averages: spectrum, kappa_eff,
                  iterations; deflation reading (principal angles / eigen
                  overlaps vs the bottom-8 sine modes, eigenmodes_explainer
                  convention).

Run from the repo root:
    uv run python python/experiments/update_views_checks.py

Expected output: all PASS; writes results/update_views.json with every number
and curve the report quotes (plus n = 8 exact small matrices).  Deterministic,
runtime well under 60 s.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, eigh, subspace_angles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poisson import laplacian_1d
from preconditioners import ic0  # noqa: E402  (chain: IC(0) == exact chol)

np.set_printoptions(precision=6, suppress=True, linewidth=120)
results = {"checks": []}


def ok(name, cond, err=None):
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {name}" + (f"   (max err {err:.3g})" if err is not None else ""))
    results["checks"].append({"name": name, "pass": bool(cond),
                              **({"max_err": float(err)} if err is not None else {})})


# ---- the chain --------------------------------------------------------------
n = 32
h = 1.0 / (n + 1)
Ad = (laplacian_1d(n) / h**2).toarray()
Dv = np.diag(Ad).copy()                       # constant 2/h^2
Dm = np.diag(Dv)
Lstr = np.tril(Ad, -1)
Ustr = np.triu(Ad, 1)
kk = np.arange(1, n + 1)
lam = 4.0 * np.sin(kk * np.pi * h / 2.0) ** 2 / h**2
ii = np.arange(1, n + 1)
Vh = np.sin(np.outer(ii, kk) * np.pi * h) * math.sqrt(2.0 * h)   # orthonormal modes
kappa = lam[-1] / lam[0]

mu_J = math.cos(math.pi * h)                                     # rho(T_Jacobi)
rho_gs_exact = mu_J**2                                           # rho(T_GS)
omega_opt = 2.0 / (1.0 + math.sin(math.pi * h))
omega_opt_trefethen = 2.0 / (1.0 + math.sqrt(1.0 - mu_J**2))
rho_sor_exact = omega_opt - 1.0
alpha_opt = 2.0 / (lam[0] + lam[-1])                             # == h^2/2 on the chain

# fixed rhs, green-tents convention: unit heat in at node 8, out at node 24
b = np.zeros(n)
b[8 - 1] = 1.0 / h
b[24 - 1] = -1.0 / h
x_star = np.linalg.solve(Ad, b)
rng = np.random.default_rng(0)
x0_rand = rng.standard_normal(n)

# =============================================================================
# component / operational loops (V1: the per-entry formulas)
# =============================================================================
def sweep_richardson(x, rhs, a=alpha_opt):
    xn = np.empty(n)
    for i in range(n):
        left = x[i - 1] if i > 0 else 0.0
        right = x[i + 1] if i < n - 1 else 0.0
        xn[i] = x[i] + a * (rhs[i] - (2.0 * x[i] - left - right) / h**2)
    return xn


def sweep_jacobi(x, rhs):
    xn = np.empty(n)
    for i in range(n):
        left = x[i - 1] if i > 0 else 0.0
        right = x[i + 1] if i < n - 1 else 0.0
        xn[i] = (left + right + h * h * rhs[i]) / 2.0        # stale neighbors
    return xn


def sweep_gs(x, rhs):
    x = x.copy()
    for i in range(n):
        left = x[i - 1] if i > 0 else 0.0                    # fresh left neighbor
        right = x[i + 1] if i < n - 1 else 0.0
        x[i] = (left + right + h * h * rhs[i]) / 2.0
    return x


def sweep_sor_forward(x, rhs, w):
    x = x.copy()
    for i in range(n):
        left = x[i - 1] if i > 0 else 0.0
        right = x[i + 1] if i < n - 1 else 0.0
        x[i] = (1.0 - w) * x[i] + w * (left + right + h * h * rhs[i]) / 2.0
    return x


def sweep_sor_backward(x, rhs, w):
    x = x.copy()
    for i in reversed(range(n)):
        left = x[i - 1] if i > 0 else 0.0
        right = x[i + 1] if i < n - 1 else 0.0
        x[i] = (1.0 - w) * x[i] + w * (left + right + h * h * rhs[i]) / 2.0
    return x


def sweep_ssor(x, rhs, w=omega_opt):
    return sweep_sor_backward(sweep_sor_forward(x, rhs, w), rhs, w)


# =============================================================================
# Shawe-Taylor & Cristianini Algorithm 5.12 (verbatim: R recursion, d array,
# nu_j = sqrt(d_j), pivot [a, I(j+1)] = max(d)) -- and its Gram-Schmidt dual
# =============================================================================
def alg512(K, T):
    """Pivoted partial Cholesky; returns R (T x N) with K ~= R^T R, pivots,
    nu_j list, and the d array after each of the T steps."""
    N = K.shape[0]
    d = np.array(np.diag(K), dtype=float)
    R = np.zeros((T, N))
    perm, nus, d_hist = [], [], []
    for j in range(T):
        i_j = int(np.argmax(d))
        nu = math.sqrt(max(d[i_j], 0.0))
        perm.append(i_j)
        nus.append(nu)
        for i in range(N):                                    # the book's inner loop
            R[j, i] = (K[i_j, i] - R[:j, i_j] @ R[:j, i]) / nu
        d = d - R[j, :] ** 2
        d_hist.append(d.copy())
    return R, perm, nus, d_hist


def pivoted_mgs(B, T):
    """Pivoted modified Gram-Schmidt on B's columns (pivot on largest residual
    column norm^2 == the same d array).  Returns R (T x N), pivots, d history."""
    Bres = B.copy()
    N = B.shape[1]
    R = np.zeros((T, N))
    perm, d_hist = [], []
    for j in range(T):
        d = np.einsum("ij,ij->j", Bres, Bres)
        i_j = int(np.argmax(d))
        perm.append(i_j)
        q = Bres[:, i_j] / math.sqrt(d[i_j])
        R[j, :] = q @ Bres
        Bres = Bres - np.outer(q, R[j, :])
        d_hist.append(np.einsum("ij,ij->j", Bres, Bres))
    return R, perm, d_hist


def mgs_natural(B):
    """Unpivoted modified Gram-Schmidt, natural column order: B = Q R."""
    Bres = B.copy()
    N = B.shape[1]
    R = np.zeros((N, N))
    Q = np.zeros_like(B)
    for j in range(N):
        R[j, j] = np.linalg.norm(Bres[:, j])
        Q[:, j] = Bres[:, j] / R[j, j]
        R[j, j + 1:] = Q[:, j] @ Bres[:, j + 1:]
        Bres[:, j + 1:] -= np.outer(Q[:, j], R[j, j + 1:])
    return Q, R


# =============================================================================
# method registry: name -> explicit M (V2's preconditioner) + operational loop
# =============================================================================
def ic_rank_T(T):
    """Factor-analysis surrogate M = R_T^T R_T + diag(Schur residual) from
    pivoted partial Cholesky on A (09 SS6), optimally damped: M_eff = M0/omega."""
    R_T, perm_T, nus_T, d_hist_T = alg512(Ad, T)
    dT = d_hist_T[-1].copy()
    dT[perm_T] = 0.0                                          # exact zeros at pivots
    M0 = R_T.T @ R_T + np.diag(dT)
    M0 = (M0 + M0.T) / 2.0
    mu = eigh(Ad, M0, eigvals_only=True)                      # spectrum of M0^{-1}A
    om = 2.0 / (mu[0] + mu[-1])
    # operational solve of M0 z = r: the LDL^T form M0 = Lhat blkdiag(A_PP,
    # diag(s)) Lhat^T, Lhat = [[I,0],[A_QP A_PP^{-1}, I]] -- the truncated-
    # regression / innovations path (report 15), not a generic dense solve.
    P = list(perm_T)
    Q = [i for i in range(n) if i not in P]
    App = Ad[np.ix_(P, P)]
    Cb = np.linalg.solve(App, Ad[np.ix_(P, Q)]).T             # |Q| x T regression coefs
    s = dT[Q]

    def m0_solve(r):
        rP, rQ = r[P], r[Q]
        wQ = rQ - Cb @ rP                                     # forward (innovations)
        vP = np.linalg.solve(App, rP)
        vQ = wQ / s                                           # diagonal
        zQ = vQ
        zP = vP - Cb.T @ vQ                                   # backward
        z = np.empty(n)
        z[P], z[Q] = zP, zQ
        return z

    # explicit LDL^T reconstruction check matrix (in pivot-permuted order)
    idx = P + Q
    Lh = np.eye(n)
    Lh[T:, :T] = Cb
    Dh = np.zeros((n, n))
    Dh[:T, :T] = App
    Dh[T:, T:] = np.diag(s)
    M0_perm_rebuilt = Lh @ Dh @ Lh.T
    ldl_dev = np.abs(M0[np.ix_(idx, idx)] - M0_perm_rebuilt).max() / np.abs(Ad).max()

    return {
        "M": M0 / om, "M0": M0, "omega": om, "mu": mu,
        "loop": lambda x, rhs: x + om * m0_solve(rhs - Ad @ x),
        "pivots_1based": [p + 1 for p in perm_T],
        "kappa_eff": float(mu[-1] / mu[0]),
        "rho_ref": float((mu[-1] / mu[0] - 1.0) / (mu[-1] / mu[0] + 1.0)),
        "ldl_dev": float(ldl_dev), "spd": True,
    }


def blockavg_1d(n, nb):
    """1-D analogue of preconditioners.block_average_matrix: column j puts
    1/bs on its bs consecutive nodes (bs = n/nb)."""
    bs = n // nb
    Z = np.zeros((n, nb))
    for j in range(nb):
        Z[j * bs:(j + 1) * bs, j] = 1.0 / bs
    return Z


Z8 = blockavg_1d(n, 8)
Ac = Z8.T @ Ad @ Z8
Ac_inv = np.linalg.inv(Ac)
M0inv_2l = Z8 @ Ac_inv @ Z8.T + np.diag(1.0 / Dv)
mu_2l = np.sort(np.linalg.eigvals(M0inv_2l @ Ad).real)
assert np.abs(np.linalg.eigvals(M0inv_2l @ Ad).imag).max() < 1e-8 * mu_2l[-1]
omega_2l = 2.0 / (mu_2l[0] + mu_2l[-1])
Minv_2l = omega_2l * M0inv_2l
M_2l = np.linalg.inv(Minv_2l)
M_2l = (M_2l + M_2l.T) / 2.0


def loop_two_level(x, rhs):
    r = rhs - Ad @ x
    yc = np.linalg.solve(Ac, Z8.T @ r)                        # coarse solve, 8x8
    return x + omega_2l * (Z8 @ yc + r / Dv)                  # + Jacobi smoothing, damped


M_sor = Dm / omega_opt + Lstr
M_ssor = (omega_opt / (2.0 - omega_opt)) * (Dm / omega_opt + Lstr) @ np.diag(1.0 / Dv) \
         @ (Dm / omega_opt + Ustr)
M_ssor = (M_ssor + M_ssor.T) / 2.0

ic = {T: ic_rank_T(T) for T in (4, 8, 16, 31)}

methods = {
    "richardson": {"M": np.eye(n) / alpha_opt, "loop": sweep_richardson, "spd": True,
                   "param": ("alpha", alpha_opt), "rho_ref": mu_J},
    "jacobi": {"M": Dm.copy(), "loop": sweep_jacobi, "spd": True,
               "param": ("-", None), "rho_ref": mu_J},
    "gs": {"M": Dm + Lstr, "loop": sweep_gs, "spd": False,
           "param": ("-", None), "rho_ref": rho_gs_exact},
    "sor": {"M": M_sor, "loop": lambda x, r: sweep_sor_forward(x, r, omega_opt),
            "spd": False, "param": ("omega", omega_opt), "rho_ref": rho_sor_exact},
    "ssor": {"M": M_ssor, "loop": sweep_ssor, "spd": True,
             "param": ("omega", omega_opt), "rho_ref": None},   # set from eig below
    "ic_T4": {**ic[4], "param": ("T", 4)},
    "ic_T8": {**ic[8], "param": ("T", 8)},
    "ic_T16": {**ic[16], "param": ("T", 16)},
    "ic_T31": {**ic[31], "param": ("T", 31)},
    "two_level": {"M": M_2l, "loop": loop_two_level, "spd": True,
                  "param": ("omega", omega_2l),
                  "rho_ref": float((mu_2l[-1] / mu_2l[0] - 1.0)
                                   / (mu_2l[-1] / mu_2l[0] + 1.0))},
}
rho_ssor = float(np.abs(np.linalg.eigvals(
    np.eye(n) - np.linalg.solve(M_ssor, Ad))).max())
methods["ssor"]["rho_ref"] = rho_ssor

# =============================================================================
# 1. SPLITTINGS: loop == M^{-1}(N x + b) == x + M^{-1}(b - A x)
# =============================================================================
scale_x = max(1.0, np.abs(x_star).max())
for name, m in methods.items():
    M = m["M"]
    N = M - Ad
    xa = x0_rand.copy(); xb = x0_rand.copy(); xc = x0_rand.copy()
    dev = 0.0
    for _ in range(5):
        xa = m["loop"](xa, b)
        xb = np.linalg.solve(M, N @ xb + b)
        xc = xc + np.linalg.solve(M, b - Ad @ xc)
        dev = max(dev, np.abs(xa - xb).max(), np.abs(xa - xc).max(),
                  np.abs(xb - xc).max())
    dev /= max(scale_x, np.abs(xb).max())
    m["splitting_dev"] = float(dev)
    ok(f"1.{name}: component loop == M^-1(Nx+b) == x + M^-1(b-Ax), 5 iters",
       dev < 1e-11, dev)

err = np.abs(methods["richardson"]["M"] - Dm).max() / Dv[0]
ok("1.rich==jac: Richardson(alpha_opt) has M = I/alpha = 2/h^2 I == D, i.e. "
   "alpha_opt = h^2/2 (the suite's Jacobi-no-op: constant diagonal)",
   err < 1e-14 and abs(alpha_opt - h * h / 2.0) < 1e-14 * alpha_opt, err)
err = max(ic[T]["ldl_dev"] for T in ic)
ok("1.ic-ldl: M0 == Lhat blkdiag(A_PP, diag(schur)) Lhat^T (innovations form, "
   "all four T)", err < 1e-12, err)

# =============================================================================
# 2. EXACT RATES
# =============================================================================
T_J = np.eye(n) - np.linalg.solve(Dm, Ad)
rho_J_eig = np.abs(np.linalg.eigvals(T_J)).max()
err = abs(rho_J_eig - mu_J)
ok("2.jacobi: rho(T_J) == cos(pi h) == mu", err < 1e-12, err)

T_GS = np.eye(n) - np.linalg.solve(Dm + Lstr, Ad)
rho_GS_eig = np.abs(np.linalg.eigvals(T_GS)).max()
err = abs(rho_GS_eig - rho_gs_exact)
ok("2.gs: rho(T_GS) == cos^2(pi h) == mu^2", err < 1e-10, err)

err = abs(omega_opt - omega_opt_trefethen)
ok("2.omega: 2/(1+sin pi h) == 2/(1+sqrt(1-mu^2)) (Trefethen-notes form)",
   err < 1e-15, err)


def sor_T(w):
    return np.eye(n) - np.linalg.solve(Dm / w + Lstr, Ad)


rho_sor_eig = np.abs(np.linalg.eigvals(sor_T(omega_opt))).max()
err = abs(rho_sor_eig - rho_sor_exact)
ok("2.sor: rho(T_SOR(omega_opt)) == omega_opt - 1 == (1-sin pi h)/(1+sin pi h) "
   "(defective pair at the kink: eig tol 1e-6)", err < 1e-6, err)
err = abs(rho_sor_exact - (1 - math.sin(math.pi * h)) / (1 + math.sin(math.pi * h)))
ok("2.sor-form: omega_opt - 1 == (1-sin pi h)/(1+sin pi h)", err < 1e-15, err)


def rho_sor_formula(w):
    """max |z| over roots of (z + w - 1)^2 = z w^2 mu_k^2, mu_k = cos(k pi h)."""
    best = 0.0
    for m_k in np.cos(kk * np.pi * h):
        bq = 2.0 * (w - 1.0) - (w * m_k) ** 2
        cq = (w - 1.0) ** 2
        disc = bq * bq - 4.0 * cq
        if disc >= 0:
            r = max(abs((-bq + math.sqrt(disc)) / 2.0), abs((-bq - math.sqrt(disc)) / 2.0))
        else:
            r = math.sqrt(cq)                                 # complex pair, |z| = sqrt(cq)
        best = max(best, r)
    return best


omega_grid = np.concatenate([np.linspace(0.05, 1.95, 191), [omega_opt]])
omega_grid.sort()
sor_curve = []
dev_grid = 0.0
for w in omega_grid:
    r_eig = float(np.abs(np.linalg.eigvals(sor_T(w))).max())
    r_form = float(rho_sor_formula(w))
    dev_grid = max(dev_grid, abs(r_eig - r_form))
    sor_curve.append([float(w), r_eig, r_form])
ok("2.sor-curve: rho(T_SOR(omega)) eigvals == quadratic-root formula on 192-pt "
   "grid in (0,2)", dev_grid < 1e-6, dev_grid)
above = [abs(r_eig - (w - 1.0)) for w, r_eig, _ in sor_curve if w >= omega_opt]
ok("2.sor-above: rho == omega - 1 for all grid omega >= omega_opt",
   max(above) < 1e-6, max(above))
w_fine = np.linspace(omega_opt - 0.05, omega_opt + 0.05, 2001)
r_fine = np.array([rho_sor_formula(w) for w in w_fine])
w_argmin = w_fine[int(np.argmin(r_fine))]
ok("2.sor-kink: argmin of the omega-curve == omega_opt (within 5e-5 fine grid)",
   abs(w_argmin - omega_opt) < 5e-5, abs(w_argmin - omega_opt))

ssor_curve = [[float(w), float(np.abs(np.linalg.eigvals(
    np.eye(n) - np.linalg.solve(
        (w / (2 - w)) * (Dm / w + Lstr) @ np.diag(1 / Dv) @ (Dm / w + Ustr), Ad)
)).max())] for w in np.linspace(0.05, 1.95, 96)]

# measured asymptotic slopes: renormalized power iteration on T = I - M^{-1}A
def measure_rho(M, m_total=6000, m_burn=3000):
    Tm = np.eye(n) - np.linalg.solve(M, Ad)
    v = np.random.default_rng(1).standard_normal(n)
    v /= np.linalg.norm(v)
    acc = 0.0
    for m in range(1, m_total + 1):
        v = Tm @ v
        nrm = np.linalg.norm(v)
        if nrm < 1e-300:
            return 0.0
        v /= nrm
        if m > m_burn:
            acc += math.log(nrm)
    return math.exp(acc / (m_total - m_burn))


slope_dev = {}
for name, m in methods.items():
    rho_ref = m["rho_ref"]
    if rho_ref < 1e-8:                                        # exact method (ic_T31)
        e1 = (np.eye(n) - np.linalg.solve(m["M"], Ad)) @ x0_rand
        m["rho_measured"] = 0.0
        ok(f"2.slope.{name}: exact preconditioner, error dead after ONE sweep",
           np.linalg.norm(e1) < 1e-10 * np.linalg.norm(x0_rand),
           np.linalg.norm(e1) / np.linalg.norm(x0_rand))
        continue
    r_meas = measure_rho(m["M"])
    m["rho_measured"] = float(r_meas)
    rel = abs(r_meas - rho_ref) / rho_ref
    slope_dev[name] = rel
ok("2.slopes: measured asymptotic rate == rho for every method, rel err < 1% "
   f"({', '.join(f'{k} {v:.2e}' for k, v in slope_dev.items())})",
   max(slope_dev.values()) < 0.01, max(slope_dev.values()))

# iterations to 1e-8 on the fixed rhs, from x0 = 0
profile_frames = [0, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60]
trajectories = {}
for name, m in methods.items():
    M = m["M"]
    lu = cho_factor((M + M.T) / 2.0) if m["spd"] else None
    x = np.zeros(n)
    nrm0 = np.linalg.norm(x_star)
    errs = [1.0]
    profiles = {0: x.copy()}
    it = None
    cap = 30000
    for k in range(1, cap + 1):
        if m["spd"]:
            x = x + cho_solve(lu, b - Ad @ x)
        else:
            x = x + np.linalg.solve(M, b - Ad @ x)
        errs.append(float(np.linalg.norm(x - x_star) / nrm0))
        if k in profile_frames:
            profiles[k] = x.copy()
        if errs[-1] <= 1e-8:
            it = k
            break
    m["iters_1e8"] = it
    keep = np.unique(np.round(np.linspace(0, len(errs) - 1, 150)).astype(int))
    trajectories[name] = {
        "err_curve": [[int(kk_), errs[kk_]] for kk_ in keep],
        "profile_iters": [f for f in profile_frames if f in profiles],
        "profiles": [[float(v) for v in profiles[f]] for f in profile_frames
                     if f in profiles],
    }
ok("2.iters: every method reaches 1e-8 (caps not hit); Richardson == Jacobi "
   f"iteration-for-iteration ({methods['richardson']['iters_1e8']} each)",
   all(m["iters_1e8"] is not None for m in methods.values())
   and methods["richardson"]["iters_1e8"] == methods["jacobi"]["iters_1e8"])
ok("2.gs-half: GS iterations ~ half of Jacobi (rho_GS = rho_J^2): "
   f"{methods['gs']['iters_1e8']} vs {methods['jacobi']['iters_1e8']}",
   abs(methods["gs"]["iters_1e8"] * 2 - methods["jacobi"]["iters_1e8"])
   < 0.05 * methods["jacobi"]["iters_1e8"])

# =============================================================================
# 3. WHITENED GD (V3): SPD methods only; GS/SOR = exact cyclic coordinate descent
# =============================================================================
whiten_dev = {}
for name in ("jacobi", "ssor", "ic_T4", "ic_T8", "ic_T16", "ic_T31", "two_level"):
    M = (methods[name]["M"] + methods[name]["M"].T) / 2.0
    Lc = np.linalg.cholesky(M)
    C = Lc.T                                                  # M = C^T C
    Ci = np.linalg.inv(C)
    At = Ci.T @ Ad @ Ci
    bt = Ci.T @ b
    lu = cho_factor(M)
    x = x0_rand.copy()
    z = C @ x0_rand
    dev = 0.0
    for _ in range(60):
        x = x + cho_solve(lu, b - Ad @ x)                     # M-preconditioned Richardson
        z = z - (At @ z - bt)                                 # plain GD, step 1, whitened
        dev = max(dev, np.linalg.norm(Ci @ z - x) / np.linalg.norm(x_star))
    whiten_dev[name] = float(dev)
    ok(f"3.whiten.{name}: step-1 GD on C^-T A C^-1 mapped back by C^-1 == "
       "M-preconditioned Richardson, 60 iterates", dev < 1e-11, dev)

CJ = np.linalg.cholesky(Dm).T
c_scalar = CJ[0, 0]
err = np.abs(CJ - c_scalar * np.eye(n)).max()
ok("3.jacobi-coords: C_Jacobi = sqrt(2)/h I is a GLOBAL rescale (constant "
   "diagonal, unit conditional variances) -- the suite's Jacobi-no-op",
   err < 1e-12 and abs(c_scalar - math.sqrt(2.0) / h) < 1e-9, err)

K_ssor = math.sqrt(omega_opt / (2.0 - omega_opt)) * np.diag(1.0 / np.sqrt(Dv)) \
         @ (Dm / omega_opt + Ustr)
err = np.abs(K_ssor.T @ K_ssor - M_ssor).max() / np.abs(M_ssor).max()
err2 = np.abs(np.linalg.cholesky(M_ssor) - K_ssor.T).max() / np.abs(K_ssor).max()
ok("3.ssor-coords: closed-form sweep-whitener K = sqrt(w/(2-w)) D^-1/2 (D/w + U) "
   "has K^T K == M_SSOR and K^T == chol(M_SSOR) (uniqueness)",
   err < 1e-12 and err2 < 1e-10, max(err, err2))

asym_gs = np.abs(methods["gs"]["M"] - methods["gs"]["M"].T).max()
asym_sor = np.abs(M_sor - M_sor.T).max()
ok("3.nonsym: M_GS and M_SOR are NONSYMMETRIC (max |M - M^T| = 1/h^2 = 1089): "
   "no whitening reading exists for the sweeps",
   abs(asym_gs - 1.0 / h**2) < 1e-9 and abs(asym_sor - 1.0 / h**2) < 1e-9,
   abs(asym_gs - 1.0 / h**2))

# GS == exact cyclic coordinate descent on J(x) = 1/2 x^T A x - b^T x
x = x0_rand.copy()
dev_cd, dev_grad = 0.0, 0.0
res_scale = np.abs(Ad @ x0_rand - b).max()
for i in range(n):
    r_i = b[i] - Ad[i] @ x
    cd_target = x[i] + r_i / Ad[i, i]                         # exact line min along e_i
    left = x[i - 1] if i > 0 else 0.0
    right = x[i + 1] if i < n - 1 else 0.0
    gs_value = (left + right + h * h * b[i]) / 2.0
    dev_cd = max(dev_cd, abs(gs_value - cd_target) / scale_x)
    x[i] = gs_value
    dev_grad = max(dev_grad, abs(Ad[i] @ x - b[i]) / res_scale)
ok("3.gs-cd: each GS micro-step == exact coordinate minimization of J "
   "(x_i <- x_i + r_i/A_ii)", dev_cd < 1e-13, dev_cd)
ok("3.gs-grad: dJ/dx_i == 0 immediately after each micro-step (Gibbs without "
   "noise, report 09 SS3)", dev_grad < 1e-13, dev_grad)
err = np.abs(x - sweep_gs(x0_rand, b)).max() / scale_x
ok("3.gs-sweep: the full micro-step sequence == one GS sweep", err < 1e-14, err)

x = x0_rand.copy()
dev_sor = 0.0
for i in range(n):
    r_i = b[i] - Ad[i] @ x
    over_target = x[i] + omega_opt * r_i / Ad[i, i]           # omega x the CD step
    left = x[i - 1] if i > 0 else 0.0
    right = x[i + 1] if i < n - 1 else 0.0
    sor_value = (1 - omega_opt) * x[i] + omega_opt * (left + right + h * h * b[i]) / 2.0
    dev_sor = max(dev_sor, abs(sor_value - over_target) / scale_x)
    x[i] = sor_value
ok("3.sor-cd: each SOR micro-step == relax omega times PAST the coordinate "
   "minimum (x_i <- x_i + omega r_i/A_ii)", dev_sor < 1e-13, dev_sor)

# =============================================================================
# 4. SHAWE-TAYLOR: Alg 5.12 on Sigma = A^{-1} and on A; the GS/Cholesky duality
# =============================================================================
Sigma = np.linalg.inv(Ad)
Sigma = (Sigma + Sigma.T) / 2.0
xg = ii * h
Sig_closed = h * np.minimum.outer(xg, xg) * (1.0 - np.maximum.outer(xg, xg))
err = np.abs(Sigma - Sig_closed).max() / Sigma.max()
ok("4.bridge: Sigma = A^-1 == h * x_min (1 - x_max) -- the Brownian-bridge "
   "covariance, scaled by h", err < 1e-10, err)


def schur_diag_dev(K, perm, d_hist):
    dev = 0.0
    for j in range(1, len(perm) + 1):
        P = perm[:j]
        S = K - K[:, P] @ np.linalg.solve(K[np.ix_(P, P)], K[P, :])
        dev = max(dev, np.abs(np.diag(S) - d_hist[j - 1]).max())
    return dev / np.abs(np.diag(K)).max()


for tag, K in (("Sigma", Sigma), ("A", Ad)):
    R, perm, nus, d_hist = alg512(K, n)
    err = np.abs(K - R.T @ R).max() / np.abs(K).max()
    ok(f"4.{tag}-full: T=n run reproduces K == R^T R exactly", err < 1e-10, err)
    Rp = R[:, perm]
    err = np.abs(np.tril(Rp, -1)).max() / np.abs(Rp).max()
    ok(f"4.{tag}-tri: R[:, pivots] is upper triangular", err < 1e-10, err)
    PKP = K[np.ix_(perm, perm)]
    err = max(np.abs(PKP - Rp.T @ Rp).max() / np.abs(K).max(),
              np.abs(Rp - cholesky(PKP, lower=False)).max() / np.abs(Rp).max())
    ok(f"4.{tag}-chol: P K P^T == Rp^T Rp and Rp == chol(P K P^T) "
       "(pivoted chol up to permutation)", err < 1e-9, err)
    err = schur_diag_dev(K, perm, d_hist)
    ok(f"4.{tag}-schur: d array after every step == diag of the Schur "
       "complement (unexplained variance bookkeeping)", err < 1e-9, err)
    if tag == "Sigma":
        piv_sigma = [p + 1 for p in perm]
        nus_sigma = [float(v) for v in nus]
        unexplained_sigma = [float(max(d, 0.0)) for d in
                             [Sigma.trace()] + [dh.sum() for dh in d_hist]]
    else:
        piv_A = [p + 1 for p in perm]

ok(f"4.pivots-Sigma: bridge-covariance pivot sequence starts at the midpoint "
   f"and bisects: {piv_sigma[:7]} (recorded in full in the JSON)",
   piv_sigma[0] in (16, 17) and set(piv_sigma[1:3]) <= {8, 9, 24, 25})
odd_front = all(p % 2 == 1 for p in piv_A[:16])
ok(f"4.pivots-A: on A the pivots sweep the ODD nodes first (red-black order "
   f"discovered by greedy variance): {piv_A[:8]}...", odd_front)

# duality: Gram-Schmidt on the grounded B's columns == Cholesky on A
B = np.zeros((n + 1, n))
for e in range(n + 1):
    if e < n:
        B[e, e] = 1.0 / h
    if e >= 1:
        B[e, e - 1] = -1.0 / h

Qb, Rgs = mgs_natural(B)
L_chol = cholesky(Ad, lower=True)
err = np.abs(Rgs - L_chol.T).max() / np.abs(L_chol).max()
ok("4.duality: unpivoted Gram-Schmidt on B's columns gives R == chol(A)^T "
   "(09 SS4.3 / Shawe-Taylor SS5.2)", err < 1e-9, err)
err = np.abs(ic0(Ad) - L_chol).max() / np.abs(L_chol).max()
ok("4.ic0: IC(0) == exact chol(A) on the chain (no fill-in on a path; "
   "report 15's truncated regression is exact here)", err < 1e-12, err)
T_cut = 8
err = np.abs(Rgs[:T_cut, :] - L_chol.T[:T_cut, :]).max() / np.abs(L_chol).max()
ok(f"4.partial: first T={T_cut} Gram-Schmidt rows == first {T_cut} rows of "
   "chol(A)^T (partial GS in edge space == partial Cholesky on A)",
   err < 1e-9, err)
Rg_p, perm_gp, dh_gp = pivoted_mgs(B, 16)
Rc_p, perm_cp, nus_cp, dh_cp = alg512(Ad, 16)
err = np.abs(Rg_p - Rc_p).max() / np.abs(Rc_p).max()
ok("4.pivoted-duality: PIVOTED Gram-Schmidt on B (pivot on residual column "
   "norm^2) == Alg 5.12 on A: same 16 pivots, same R rows",
   perm_gp == perm_cp and err < 1e-8, err)

ic_table = {str(T): {
    "kappa_eff": ic[T]["kappa_eff"],
    "omega": float(ic[T]["omega"]),
    "rho": ic[T]["rho_ref"],
    "iters_1e8": methods[f"ic_T{T}"]["iters_1e8"],
    "pivots_1based": ic[T]["pivots_1based"],
} for T in (4, 8, 16, 31)}
kap_seq = [ic[T]["kappa_eff"] for T in (4, 8, 16, 31)]
ok("4.ic-kappa: factor-analysis surrogate kappa(M^-1 A) falls monotonically "
   f"with T: {' > '.join(f'{v:.3g}' for v in kap_seq)}; T=31 gives M == A "
   "exactly (kappa 1, one step)",
   all(a > b for a, b in zip(kap_seq, kap_seq[1:]))
   and abs(kap_seq[-1] - 1.0) < 1e-9
   and methods["ic_T31"]["iters_1e8"] == 1)
err = np.abs(ic[31]["M0"] - Ad).max() / np.abs(Ad).max()
ok("4.ic-T31: rank-31 pivoted factor + 1x1 Schur residual reconstructs A "
   "EXACTLY (full rank on the chain -> one-step convergence)", err < 1e-10, err)

# =============================================================================
# 5. ELECTRICAL (V4): A = B^T B, voltage sweeps, the margin-note claims
# =============================================================================
btb_dev = np.abs(B.T @ B - Ad).max() / np.abs(Ad).max()
ok("5.gram: A == B^T B with the grounded (n+1) x n difference matrix "
   "(entries +-1/h, walls as grounded rails)", btb_dev < 1e-14, btb_dev)

xt = x0_rand.copy()
lhs = sweep_jacobi(xt, b)
xp = np.concatenate([[0.0], xt, [0.0]])
rhs_formula = (xp[:-2] + xp[2:]) / 2.0 + (h * h / 2.0) * b
rhs_matrix = (b - (Lstr + Ustr) @ xt) / Dv
err = max(np.abs(lhs - rhs_formula).max(), np.abs(lhs - rhs_matrix).max()) / scale_x
ok("5.jacobi-volts: Jacobi update == neighbor-voltage average + h^2/2 source "
   "lift == D^-1(b - (L+U)x)", err < 1e-14, err)

y_flux = B @ x_star                                           # B^T y = A x* = b
err = np.abs(B.T @ y_flux - b).max() / np.abs(b).max()
ok("5.flux: y = B x* satisfies B^T y == b (edge currents of the true solution)",
   err < 1e-12, err)
xg_ = x0_rand.copy(); xr_ = x0_rand.copy()
dev = 0.0
for _ in range(60):
    xg_ = xg_ - alpha_opt * (B.T @ (B @ xg_ - y_flux))        # GD on 1/2||Bx-y||^2
    xr_ = xr_ + alpha_opt * (b - Ad @ xr_)                    # Richardson, normal eqs
    dev = max(dev, np.abs(xg_ - xr_).max() / scale_x)
ok("5.gd-ls: plain GD on 1/2||Bx - y||^2 (grad = B^T(Bx-y) = Ax - b) == "
   "Richardson on the normal equations, 60 iterates", dev < 1e-11, dev)

xd_ = x0_rand.copy(); xj_ = x0_rand.copy()
dev = 0.0
Hdiag = np.einsum("ij,ij->j", B, B)                           # diag(B^T B) == D
for _ in range(20):
    xd_ = xd_ - (B.T @ (B @ xd_ - y_flux)) / Hdiag            # diagonal Newton on LS
    xj_ = sweep_jacobi(xj_, b)
    dev = max(dev, np.abs(xd_ - xj_).max() / scale_x)
diag_dev = np.abs(Hdiag - Dv).max() / Dv[0]
ok("5.diag-newton: Jacobi == diagonal-Newton on the least-squares objective "
   "(Hessian diag(B^T B) = 2/h^2 = D), 20 iterates",
   dev < 1e-12 and diag_dev < 1e-14, max(dev, diag_dev))

# =============================================================================
# 6. TWO-LEVEL: spectrum, deflation reading
# =============================================================================
kappa_2l = float(mu_2l[-1] / mu_2l[0])
ok(f"6.spectrum: additive Jacobi + 8 block averages: mu in "
   f"[{mu_2l[0]:.4f}, {mu_2l[-1]:.4f}], kappa_eff = {kappa_2l:.3f} "
   f"(vs kappa = {kappa:.1f} plain: {kappa / kappa_2l:.0f}x), "
   f"iterations {methods['two_level']['iters_1e8']} vs "
   f"{methods['jacobi']['iters_1e8']} Jacobi",
   kappa_2l < kappa / 25.0
   and methods["two_level"]["iters_1e8"] * 20 < methods["jacobi"]["iters_1e8"])

Qz = np.linalg.qr(Z8)[0]
angles = np.degrees(subspace_angles(Qz, Vh[:, :8]))
overlaps = [float(np.linalg.norm(Qz.T @ Vh[:, k_]) ** 2) for k_ in range(16)]
ok(f"6.deflation: coarse space vs bottom-8 sine modes: largest principal angle "
   f"{max(angles):.2f} deg < 45 (eigenmodes_explainer convention); mode-1 "
   f"overlap {overlaps[0]:.3f}", max(angles) < 45.0)
Pi = Z8 @ Ac_inv @ Z8.T @ Ad                                  # coarse correction operator
err = max(np.abs(Pi @ Pi - Pi).max(),
          np.abs(Ad @ Pi - Pi.T @ Ad).max() / np.abs(Ad).max(),
          np.abs(Pi @ Z8 - Z8).max())
ok("6.projector: the coarse correction Z Ac^-1 Z^T A is the A-ORTHOGONAL "
   "projection onto range(Z): Pi^2 == Pi, A Pi == Pi^T A, Pi Z == Z (coarse "
   "residuals solved exactly in one shot -- the deflation reading)",
   err < 1e-10, err)
# honest energy bookkeeping: Euclidean overlap with the bottom modes is high
# (angles above), but the A-energy overlap ||Pi v_k||_A^2 / ||v_k||_A^2 is
# mediocre for EVERY mode -- piecewise-constant blocks store energy in their
# jumps.  The kappa win comes from the compound spectrum, not from nailing
# the bottom modes exactly; recorded, not over-claimed.
q_rayleigh = [float(Vh[:, k_] @ (Minv_2l @ (Ad @ Vh[:, k_]))) for k_ in range(n)]
energy_overlap = [float((Vh[:, k_] @ (Ad @ (Pi @ Vh[:, k_]))) / lam[k_])
                  for k_ in range(n)]
ok("6.energy: A-energy overlap of the coarse projection with mode 1 is only "
   f"{energy_overlap[0]:.3f} (Euclidean overlap {overlaps[0]:.3f}): piecewise-"
   "constant coarse functions are energy-rough; the win is spectral "
   f"(kappa {kappa:.0f} -> {kappa_2l:.1f}), not modal",
   0.05 < energy_overlap[0] < 0.9 and overlaps[0] > 0.95)

# =============================================================================
# n = 8 exact small matrices for the page
# =============================================================================
n8 = 8
h8 = 1.0 / (n8 + 1)
A8 = (laplacian_1d(n8) / h8**2).toarray()
B8 = np.zeros((n8 + 1, n8))
for e in range(n8 + 1):
    if e < n8:
        B8[e, e] = 1.0 / h8
    if e >= 1:
        B8[e, e - 1] = -1.0 / h8
Sigma8 = np.linalg.inv(A8)
Sigma8 = (Sigma8 + Sigma8.T) / 2.0
R8_sig, perm8_sig, nus8_sig, _ = alg512(Sigma8, n8)
R8_A, perm8_A, _, _ = alg512(A8, n8)
L8 = cholesky(A8, lower=True)
_, Rgs8 = mgs_natural(B8)
err = max(np.abs(B8.T @ B8 - A8).max(),
          np.abs(Rgs8 - L8.T).max() / np.abs(L8).max())
ok("n8: A8 == B8^T B8 (entries 162/-81, +-9) and GS-on-B8 == chol(A8)^T",
   err < 1e-10, err)

npass = sum(c["pass"] for c in results["checks"])
print(f"\n{npass}/{len(results['checks'])} PASS")

# =============================================================================
# every number / curve the report quotes
# =============================================================================
method_table = {}
for name, m in methods.items():
    kappa_eff = None
    if name in ("richardson", "jacobi"):
        kappa_eff = float(kappa)
    elif name.startswith("ic_"):
        kappa_eff = ic[int(name[4:])]["kappa_eff"]
    elif name == "two_level":
        kappa_eff = kappa_2l
    method_table[name] = {
        "param_name": m["param"][0],
        "param_value": (float(m["param"][1]) if m["param"][1] is not None else None),
        "M_symmetric": bool(m["spd"]),
        "rho": float(m["rho_ref"]),                           # per-sweep unexplained fraction (report 12)
        "rho_measured": float(m.get("rho_measured", 0.0)),
        "iters_1e8": int(m["iters_1e8"]),
        "iters_predicted": (int(math.ceil(math.log(1e-8) / math.log(m["rho_ref"])))
                            if m["rho_ref"] > 1e-8 else 1),
        "kappa_eff": kappa_eff,
        "splitting_dev": m["splitting_dev"],
    }

results["quoted"] = {
    "n": n, "h": h, "n_plus_1": n + 1, "h2_inv": (n + 1) ** 2,
    "diag_A": 2.0 * (n + 1) ** 2, "half_h2": h * h / 2.0,
    "mu": mu_J, "mu_formula": "cos(pi h)",
    "rho_gs": rho_gs_exact, "rho_sor_opt": rho_sor_exact,
    "rho_ssor_opt": rho_ssor,
    "omega_opt": omega_opt, "omega_opt_trefethen_form": omega_opt_trefethen,
    "sin_pi_h": math.sin(math.pi * h),
    "alpha_opt": alpha_opt,
    "alpha_opt_is_half_h2": bool(abs(alpha_opt - h * h / 2) < 1e-18),
    "lam_1": float(lam[0]), "lam_n": float(lam[-1]), "kappa": float(kappa),
    "rhs": {"in_node": 8, "out_node": 24, "scale": 1.0 / h,
            "convention": "green-tents unit heat, +1/h at node 8, -1/h at node 24"},
    "x_star": [float(v) for v in x_star],
    "b": [float(v) for v in b],
    "methods": method_table,
    "sor_curve": sor_curve,                                   # [omega, rho_eig, rho_formula]
    "ssor_curve": ssor_curve,
    "trajectories": trajectories,
    "whitening": {
        "max_dev": whiten_dev,
        "jacobi_scalar": float(c_scalar),
        "jacobi_scalar_formula": "sqrt(2)/h",
        "gs_asym_maxabs": float(asym_gs),
        "sor_asym_maxabs": float(asym_sor),
        "gs_cd_microstep_dev": float(dev_cd),
        "gs_grad_after_update_dev": float(dev_grad),
        "sor_overrelax_microstep_dev": float(dev_sor),
    },
    "shawe_taylor": {
        "pivots_sigma_1based": piv_sigma,
        "pivots_A_1based": piv_A,
        "nu_sigma": nus_sigma,
        "unexplained_trace_sigma": unexplained_sigma,         # trace(Schur) after 0..n pivots
        "sigma_diag": [float(v) for v in np.diag(Sigma)],
        "ic_preconditioner": ic_table,
    },
    "electrical": {
        "BtB_max_rel_dev": float(btb_dev),
        "B_shape": [n + 1, n],
        "B_entry": 1.0 / h,
        "y_construction": "y = B x* (edge currents of the true solution)",
        "diag_BtB_vs_D_rel_dev": float(diag_dev),
    },
    "two_level": {
        "n_coarse": 8, "block_size": 4,
        "mu_min": float(mu_2l[0]), "mu_max": float(mu_2l[-1]),
        "omega": float(omega_2l), "kappa_eff": kappa_2l,
        "rho": methods["two_level"]["rho_ref"],
        "iters_1e8": int(methods["two_level"]["iters_1e8"]),
        "principal_angles_deg": [float(a) for a in angles],
        "mode_overlaps": overlaps,
        "energy_overlaps": energy_overlap,
        "rayleigh_q": q_rayleigh,
    },
    "small_n8": {
        "h": h8,
        "A": [[int(round(v)) for v in row] for row in A8],
        "B": [[int(round(v)) for v in row] for row in B8],
        "chol_A_lower": [[float(v) for v in row] for row in L8],
        "gs_on_B_R": [[float(v) for v in row] for row in Rgs8],
        "Sigma": [[float(v) for v in row] for row in Sigma8],
        "pivots_sigma_1based": [p + 1 for p in perm8_sig],
        "pivots_A_1based": [p + 1 for p in perm8_A],
        "R_sigma": [[float(v) for v in row] for row in R8_sig],
        "nu_sigma": [float(v) for v in nus8_sig],
    },
    "checks_total": len(results["checks"]),
    "checks_passed": npass,
}

out = Path(__file__).resolve().parents[2] / "results" / "update_views.json"
out.write_text(json.dumps(results, indent=2))
size = os.path.getsize(out)
print(f"wrote {out} ({size/1024:.0f} KB)")

# ---- console tables ----------------------------------------------------------
print("\nrates / iterations (rhs: +1/h at node 8, -1/h at node 24; tol 1e-8):")
print(f"{'method':<12}{'param':<22}{'rho':>10}{'rho_meas':>10}{'iters':>8}{'kappa_eff':>11}")
for name, t in method_table.items():
    pv = f"{t['param_name']}={t['param_value']:.6g}" if t["param_value"] is not None else "-"
    ke = f"{t['kappa_eff']:.4g}" if t["kappa_eff"] else "-"
    print(f"{name:<12}{pv:<22}{t['rho']:>10.6f}{t['rho_measured']:>10.6f}"
          f"{t['iters_1e8']:>8}{ke:>11}")
print(f"\npivots on Sigma (bridge covariance): {piv_sigma}")
print(f"pivots on A (stiffness):             {piv_A}")
