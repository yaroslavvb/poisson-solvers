"""Numerical verification of every claim in the geometric-multigrid stepper
(the forthcoming multigrid explainer page).

Arena: 2-D Dirichlet diffusion -div(grad u) = f on the unit square,
A = poisson_2d(n) with n = 31 interior points per side (h = 1/32), levels
31 -> 15 -> 7 -> 3 -> 1 (standard coarsening h -> 2h, all odd; at 1x1 the
"solve" is a single division by A = 16).  f = the suite's hot/cold-rod pair
(reports 11/13's n = 32 problem) mapped to n = 31 by relative position:
hot rod +1 at column 4, rows 3..8; cold rod -1 at column 26, rows 22..27
(0-based interior indices; exact point-symmetric image of the n = 32 rods).

Components, each a separate testable function:
  smoother     damped Jacobi, omega = 4/5 (Trottenberg's high-frequency-
               optimal damping, report 12), nu1 = nu2 = 2 sweeps;
               per-mode factor mu_kl = 1 - omega (sin^2(k pi h/2)
               + sin^2(l pi h/2)), smoothing factor 3/5 over the
               high-frequency quadrant -- verified via the exact 2-D symbol
               AND by fitting measured rough-mode decay.
  restriction  full weighting, the 1/16 [1 2 1; 2 4 2; 1 2 1] stencil
               (kron of 1-D [1/4, 1/2, 1/4] rows).
  prolongation bilinear interpolation; classical duality P == 4 R^T in 2-D
               (2^d for the d-dim operator-scaling convention A = stencil/h^2:
               R's rows average with weight sum 1 so that R A P rediscretizes
               at scale H^2 = 4 h^2; the transpose picks up the factor 4).
  coarse ops   DISCRETIZATION coarsening (poisson_2d at each level) in the
               stepper; Galerkin R A P computed alongside and the difference
               reported honestly (it is the classical 9-point stencil).
  V-cycle      recursive, x_{m+1} = x_m + M^{-1}(b - A x_m); one cycle = one
               application of M^{-1} (verified as a linearity identity).

Punchline numbers: empirical V(2,2)-cycle contraction rho_V at n = 15/31/63
(mesh-independent; exact spectral radii cross-checked by dense/Arnoldi
eigenvalues of the error propagator), against damped-Jacobi rho -> 1; plus
V-cycle-preconditioned CG iterations for the suite-culture comparison
(report 04/08 conventions), and the O(N) work count in report 11 Part C's
flop convention (1 MAC = 2 flops, A-matvec = 2 nnz(A)).

Every field / spectrum / rate the stepper's panels S1-S9 need goes to
results/multigrid.json (floats at 6 significant digits).

Run from the repo root:
    uv run python python/experiments/multigrid_checks.py

Expected output: all PASS; writes results/multigrid.json (< 4 MB); < 90 s.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.fft
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcg import pcg
from poisson import poisson_2d

t_start = time.time()
np.set_printoptions(precision=6, suppress=True, linewidth=140)
results = {"checks": []}


def ok(name, cond, err=None):
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {name}" + (f"   (max err {err:.3g})" if err is not None else ""))
    results["checks"].append({"name": name, "pass": bool(cond),
                              **({"max_err": float(err)} if err is not None else {})})


def info(msg):
    print("  " + msg)


def r6(x):
    """Round floats to 6 significant digits, recursively, for JSON compactness."""
    if isinstance(x, (float, np.floating)):
        return float(f"{float(x):.6g}")
    if isinstance(x, (list, tuple)):
        return [r6(v) for v in x]
    return x


def field(a):
    """2-D numpy array -> nested lists at 6 sig digits (row-major, row = x/slow axis)."""
    return r6(np.asarray(a, dtype=float).tolist())


nrm = np.linalg.norm

OMEGA = 4.0 / 5.0          # report 12's high-frequency-optimal damping
NU1 = NU2 = 2              # default pre/post smoothing sweeps
TOL = 1e-10


# ===========================================================================
# components (each separately testable)
# ===========================================================================
def restrict_1d(n):
    """1-D full weighting (n-1)/2 x n: row j = [1/4, 1/2, 1/4] at fine
    nodes 2j, 2j+1, 2j+2 (0-based; coarse node j sits at fine node 2j+1)."""
    nc = (n - 1) // 2
    R = sp.lil_matrix((nc, n))
    for j in range(nc):
        f = 2 * j + 1
        R[j, f - 1] = 0.25
        R[j, f] = 0.5
        R[j, f + 1] = 0.25
    return R.tocsr()


def prolong_1d(n):
    """1-D linear interpolation n x (n-1)/2: column j = [1/2, 1, 1/2] at fine
    nodes 2j, 2j+1, 2j+2; Dirichlet walls absorb the missing half-weights."""
    nc = (n - 1) // 2
    P = sp.lil_matrix((n, nc))
    for j in range(nc):
        f = 2 * j + 1
        P[f - 1, j] = 0.5
        P[f, j] = 1.0
        P[f + 1, j] = 0.5
    return P.tocsr()


def build_levels(nf):
    """Level list nf -> ... -> 1 with rediscretized A = poisson_2d(n) at every
    level, 2-D full-weighting R and bilinear P between consecutive levels."""
    levels = []
    n = nf
    while True:
        A = poisson_2d(n)
        levels.append({"n": n, "N": n * n, "h": 1.0 / (n + 1),
                       "A": A, "invd": 1.0 / A.diagonal()})
        if n == 1:
            break
        n = (n - 1) // 2
    for lev in range(len(levels) - 1):
        n = levels[lev]["n"]
        levels[lev]["R"] = sp.kron(restrict_1d(n), restrict_1d(n)).tocsr()
        levels[lev]["P"] = sp.kron(prolong_1d(n), prolong_1d(n)).tocsr()
    return levels


def sweep(lv, u, f):
    """One damped-Jacobi sweep, matrix form: u + omega D^{-1} (f - A u)."""
    return u + OMEGA * lv["invd"] * (f - lv["A"] @ u)


def sweep_stencil(U, F, h):
    """The same sweep as an explicit 5-point stencil loop on the 2-D field
    (zero Dirichlet padding), for the matrix-form == stencil-loop identity."""
    n = U.shape[0]
    Up = np.zeros((n + 2, n + 2))
    Up[1:-1, 1:-1] = U
    out = np.empty_like(U)
    for i in range(n):
        for j in range(n):
            Au = (4.0 * Up[i + 1, j + 1] - Up[i, j + 1] - Up[i + 2, j + 1]
                  - Up[i + 1, j] - Up[i + 1, j + 2]) / h**2
            out[i, j] = U[i, j] + OMEGA * (h**2 / 4.0) * (F[i, j] - Au)
    return out


def vcycle(levels, lev, u, f, nu1=NU1, nu2=NU2, rec=None):
    """Recursive V(nu1,nu2)-cycle on A_lev u = f. Returns the new iterate.
    At n = 1 the exact solve is a division by A = 4/h^2 = 16."""
    lv = levels[lev]
    if lv["n"] == 1:
        u = f / lv["A"][0, 0]
        if rec is not None:
            rec.add("coarsest solve (division by 16)", lev, u, f, "bottom")
        return u
    for _ in range(nu1):
        u = sweep(lv, u, f)
    if rec is not None:
        rec.add("pre-smooth", lev, u, f, "down")
    r = f - lv["A"] @ u
    fc = lv["R"] @ r
    if rec is not None:
        rec.add_restrict(lev, r, fc)
    ec = vcycle(levels, lev + 1, np.zeros(levels[lev + 1]["N"]), fc, nu1, nu2, rec)
    corr = lv["P"] @ ec
    u = u + corr
    if rec is not None:
        rec.add("coarse correction added", lev, u, f, "up", corr=corr)
    for _ in range(nu2):
        u = sweep(lv, u, f)
    if rec is not None:
        rec.add("post-smooth", lev, u, f, "up")
    return u


class Recorder:
    """Stage-by-stage V-cycle trace: per event the level, iterate, residual
    norm, level-subproblem error norm (vs the level's exact solve), and the
    level error field -- everything panel S8's animation needs."""

    def __init__(self, levels):
        self.levels = levels
        self.events = []

    def _exact(self, lev, f):
        lv = self.levels[lev]
        if lv["n"] == 1:
            return f / lv["A"][0, 0]
        return spla.spsolve(lv["A"].tocsc(), f)

    def add(self, stage, lev, u, f, direction, corr=None):
        lv = self.levels[lev]
        ustar = self._exact(lev, f)
        ev = {"stage": stage, "level": lev, "n": lv["n"], "dir": direction,
              "u": u.copy(), "e": ustar - u,
              "resid_norm": float(nrm(f - lv["A"] @ u)),
              "level_err_norm": float(nrm(ustar - u))}
        if corr is not None:
            ev["corr"] = corr.copy()
        self.events.append(ev)

    def add_restrict(self, lev, r_fine, f_coarse):
        self.events.append({"stage": "restrict residual", "level": lev,
                            "n": self.levels[lev]["n"], "dir": "down",
                            "r_fine": r_fine.copy(), "f_coarse": f_coarse.copy(),
                            "resid_norm": float(nrm(r_fine)),
                            "coarse_rhs_norm": float(nrm(f_coarse))})


def rod_rhs(n):
    """The suite's hot/cold-rod pair (n = 32: hot +1 at col 4 rows 3-8, cold
    -1 at col 27 rows 23-28) mapped to size n by relative position: each
    defining index i maps to floor((i+1)(n+1)/33 + 1/2) - 1, rod rows filled
    contiguously.  Returns (b, rod index metadata)."""
    def m(i):
        return int(math.floor((i + 1) * (n + 1) / 33.0 + 0.5)) - 1
    F = np.zeros((n, n))
    hot = {"col": m(4), "rows": [m(3), m(8)]}
    cold = {"col": m(27), "rows": [m(23), m(28)]}
    F[hot["rows"][0]:hot["rows"][1] + 1, hot["col"]] = 1.0
    F[cold["rows"][0]:cold["rows"][1] + 1, cold["col"]] = -1.0
    return F.ravel(), {"hot": hot, "cold": cold}


def dst2(E):
    """2-D DST-I coefficients of a field in the ORTHONORMAL sine basis:
    C = Vh^T E Vh with Vh[i,k] = sqrt(2h) sin((i+1)(k+1) pi h)."""
    n = E.shape[0]
    s = 1.0 / (2.0 * math.sqrt((n + 1) / 2.0))
    return scipy.fft.dstn(E, type=1) * s * s


# ===========================================================================
# 0. arena: n = 31, the rod pair, levels 31 -> 15 -> 7 -> 3 -> 1
# ===========================================================================
print("== 0. arena: n = 31, A = poisson_2d(31), hot/cold-rod rhs, 5 levels ==")
n = 31
N = n * n
h = 1.0 / (n + 1)
levels = build_levels(n)
A = levels[0]["A"]
b, rods = rod_rhs(n)
xstar = spla.spsolve(A.tocsc(), b)

ok(f"levels are 31 -> 15 -> 7 -> 3 -> 1 (all odd, standard h -> 2h) and the "
   f"1x1 coarsest operator is exactly [[16]] = 4/h^2 at h = 1/2",
   [lv["n"] for lv in levels] == [31, 15, 7, 3, 1]
   and levels[-1]["A"].shape == (1, 1) and levels[-1]["A"][0, 0] == 16.0)

Frod = b.reshape(n, n)
ok(f"rod rhs at n = 31: hot +1 col {rods['hot']['col']} rows "
   f"{rods['hot']['rows'][0]}-{rods['hot']['rows'][1]}, cold -1 col "
   f"{rods['cold']['col']} rows {rods['cold']['rows'][0]}-"
   f"{rods['cold']['rows'][1]} (0-based) -- the exact point-symmetric image "
   f"of report 11's n = 32 rods: b + rot180(b) == 0, and so is the solution",
   rods["hot"] == {"col": 4, "rows": [3, 8]}
   and rods["cold"] == {"col": 26, "rows": [22, 27]}
   and np.abs(Frod + Frod[::-1, ::-1]).max() == 0.0
   and np.abs(xstar.reshape(n, n) + xstar.reshape(n, n)[::-1, ::-1]).max()
   < 1e-12 * np.abs(xstar).max())

# orthonormal sine modes and 2-D eigenvalues (for spectra and mu surface)
kk = np.arange(1, n + 1)
sin2 = np.sin(kk * np.pi * h / 2.0) ** 2                     # sin^2(k pi h / 2)
lam1 = 4.0 * sin2 / h**2
Vh = np.sin(np.outer(np.arange(1, n + 1), kk) * np.pi * h) * math.sqrt(2.0 * h)
mu = 1.0 - OMEGA * (sin2[:, None] + sin2[None, :])           # per-mode factor
lam2 = lam1[:, None] + lam1[None, :]
err = np.abs(mu.ravel() - (1.0 - OMEGA * (h**2 / 4.0) * lam2.ravel())).max()
ok("per-mode smoothing factor identity: 1 - omega(sin^2 + sin^2) == "
   "1 - omega h^2 lam_kl / 4 == report 12's 1 - omega lambda_k in D = 4/h^2 "
   "scaling", err < 1e-15, err)

rng_fld = np.random.default_rng(3)
E_test = rng_fld.standard_normal((n, n))
err = np.abs(dst2(E_test) - Vh.T @ E_test @ Vh).max()
ok("2-D DST-I with scale 1/(2(n+1)) == projection onto orthonormal sine "
   "modes (the spectra used by panels S1/S7)", err < 1e-12, err)

# high-frequency mask: modes NOT representable on the 15-grid
HF = (np.maximum.outer(kk, kk) >= (n + 1) // 2)              # max(k,l) >= 16
ok(f"high/low frequency split: exactly 15^2 = 225 coarse-representable modes "
   f"(max(k,l) <= 15) = the number of coarse unknowns; {int(HF.sum())} rough",
   int((~HF).sum()) == 225 and int(HF.sum()) == N - 225)

# ===========================================================================
# 1. smoother identities: matrix form == stencil loop
# ===========================================================================
print("== 1. the smoother: damped Jacobi omega = 4/5 ==")
U0 = rng_fld.standard_normal((n, n))
Fq = rng_fld.standard_normal((n, n))
u_m = U0.ravel().copy()
U_s = U0.copy()
for _ in range(3):
    u_m = sweep(levels[0], u_m, Fq.ravel())
    U_s = sweep_stencil(U_s, Fq, h)
err = np.abs(u_m.reshape(n, n) - U_s).max() / np.abs(U_s).max()
ok("smoother identity: matrix form u + omega D^{-1}(f - A u) == explicit "
   "5-point stencil loop, 3 sweeps, random u0/f", err < 1e-13, err)

# per-mode contraction measured == mu_kl^m for all 961 modes
E0m = Vh @ np.ones((n, n)) @ Vh.T                            # all coefs = 1
e = E0m.ravel().copy()
m_sw = 25
dev = 0.0
for m in range(1, m_sw + 1):
    e = sweep(levels[0], e, np.zeros(N))                     # homogeneous: e <- S e
    dev = max(dev, np.abs(dst2(e.reshape(n, n)) - mu**m).max())
ok(f"per-mode contraction measured == mu_kl^m for all {N} modes, "
   f"{m_sw} sweeps (DST-projected)", dev < 1e-8, dev)

# ===========================================================================
# 2. smoothing factor: 3/5 over the high-frequency quadrant, omega* = 4/5
# ===========================================================================
print("== 2. smoothing factor == 3/5 at omega = 4/5 (exact 2-D symbol) ==")
# continuous symbol: sup over HF quadrant [0,pi]^2 \ [0,pi/2)^2 of
# |1 - omega s|, s = sin^2(t1/2) + sin^2(t2/2) in [1/2, 2]:
# max(|1 - omega/2|, |1 - 2 omega|) = 3/5 at omega = 4/5, attained at
# (pi/2, 0) and (pi, pi).
tt = np.linspace(0.0, np.pi, 1201)
S1g, S2g = np.meshgrid(np.sin(tt / 2) ** 2, np.sin(tt / 2) ** 2, indexing="ij")
hf_mask = (np.maximum.outer(tt, tt) >= np.pi / 2 - 1e-15)
sym = np.abs(1.0 - OMEGA * (S1g + S2g))[hf_mask]
mu_end_lo = abs(1.0 - OMEGA * 0.5)                           # theta = (pi/2, 0)
mu_end_hi = abs(1.0 - OMEGA * 2.0)                           # theta = (pi, pi)
ok("smoothing factor over the HF quadrant == 3/5 EXACTLY at omega = 4/5: "
   "sup |1 - omega s| = max(|1 - omega/2|, |1 - 2 omega|) = 3/5, attained at "
   "both ends theta = (pi/2, 0) and (pi, pi); 1201^2 symbol grid never exceeds",
   abs(mu_end_lo - 0.6) < 1e-15 and abs(mu_end_hi - 0.6) < 1e-15
   and sym.max() <= 0.6 + 1e-12, abs(sym.max() - 0.6))

w_grid = np.linspace(0.01, 1.2, 4001)
mu_of_w = np.maximum(np.abs(1.0 - w_grid / 2.0), np.abs(1.0 - 2.0 * w_grid))
w_star = w_grid[int(np.argmin(mu_of_w))]
ok(f"omega = 4/5 is the HF-optimal damping (Trottenberg, report 12): "
   f"argmin over omega of the HF sup is {w_star:.4f} with value "
   f"{mu_of_w.min():.4f}; exact solution of |1 - omega/2| = |1 - 2 omega| "
   f"is omega = 4/5, value 3/5",
   abs(w_star - 0.8) < 5e-4 and abs(mu_of_w.min() - 0.6) < 5e-4)

# discrete smoothing factor at n = 31 and its empirical fit
mu_disc = float(np.abs(mu[HF]).max())
arg = np.unravel_index(int(np.abs(np.where(HF, mu, 0.0)).argmax()), mu.shape)
mode_disc = (int(arg[0] + 1), int(arg[1] + 1))
e = (Vh @ np.where(HF, 1.0, 0.0) @ Vh.T).ravel()             # rough modes only
sup_hist = []
for m in range(1, 26):
    e = sweep(levels[0], e, np.zeros(N))
    sup_hist.append(float(np.abs(dst2(e.reshape(n, n)))[HF].max()))
fit = (sup_hist[24] / sup_hist[9]) ** (1.0 / 15.0)
ok(f"discrete smoothing factor at n = 31: max |mu| over rough modes = "
   f"{mu_disc:.6f} at mode {mode_disc}, within 0.4% of 3/5 (from below -- "
   f"the discrete grid sits strictly inside the HF quadrant); empirical fit "
   f"from rough-only decay, sweeps 10 -> 25: {fit:.6f}",
   abs(fit - mu_disc) < 1e-6 and 0.0 < 0.6 - mu_disc < 0.004,
   abs(fit - mu_disc))

# ===========================================================================
# 3. S1: why relaxation stalls -- 0/5/20/100 sweeps on the real error
# ===========================================================================
print("== 3. S1: relaxation stalls (0/5/20/100 sweeps, field + spectrum) ==")
rng0 = np.random.default_rng(0)
x0 = rng0.standard_normal(N)
x0 *= nrm(xstar) / nrm(x0)          # 50/50 smooth (rod solution) / rough (noise)
e0 = xstar - x0
SWEEPS = [0, 5, 20, 100]
s1_fields, s1_spectra, s1_norms = {}, {}, {}
e = e0.copy()
hf_frac = {}
lf11 = {}
s1_curve = [float(nrm(e0))]
for m in range(0, 101):
    if m > 0:
        e = sweep(levels[0], e, np.zeros(N))
        s1_curve.append(float(nrm(e)))
    if m in SWEEPS:
        C = dst2(e.reshape(n, n))
        s1_fields[m] = e.reshape(n, n).copy()
        s1_spectra[m] = np.abs(C)
        s1_norms[m] = float(nrm(e))
        hf_frac[m] = float((C[HF] ** 2).sum() / (C ** 2).sum())
        lf11[m] = float(C[0, 0])

mu11 = float(mu[0, 0])
err = abs(lf11[100] / lf11[0] - mu11 ** 100) / abs(mu11 ** 100)
ok(f"S1 smooth corner FROZEN: mode (1,1) keeps {lf11[100]/lf11[0]:.4f} of its "
   f"amplitude after 100 sweeps == mu_11^100 = {mu11**100:.4f} "
   f"(mu_11 = {mu11:.6f})", err < 1e-6, err)
hf_amp_ratio = float(s1_spectra[100][HF].max() / s1_spectra[0][HF].max())
ok(f"S1 rough corner DEAD: max rough-mode amplitude after 100 sweeps is "
   f"{hf_amp_ratio:.2e} of its start (rounding floor; exact factor "
   f"mu_disc^100 ~ 4e-23); rough ENERGY fraction {hf_frac[0]:.3f} -> "
   f"{hf_frac[100]:.2e}", hf_amp_ratio < 1e-12 and hf_frac[100] < 1e-24)
tail_rate = (s1_curve[100] / s1_curve[90]) ** (1.0 / 10.0)
mu12, mu22 = float(mu[0, 1]), float(mu[1, 1])
ok(f"S1 stall: ||e|| drops only {s1_norms[0]/s1_norms[100]:.2f}x in 100 "
   f"sweeps ({s1_norms[0]:.4g} -> {s1_norms[100]:.4g}, under one order of "
   f"magnitude); per-sweep ratio over sweeps 90-100 is {tail_rate:.6f} -- "
   f"inside the smooth-corner band [mu_22, mu_11] = [{mu22:.4f}, {mu11:.4f}] "
   f"and closest to mu_12 = {mu12:.4f}: the rod solution is point-"
   f"antisymmetric so its slowest mode is (1,2) (report 13's parity note); "
   f"only the noise part of x0 populates (1,1)",
   s1_norms[0] / s1_norms[100] < 10.0
   and mu22 - 1e-9 <= tail_rate <= mu11 + 1e-12
   and abs(tail_rate - mu12) < 0.002)

# ===========================================================================
# 4. duality: bilinear P == 4 R^T (and R preserves constants)
# ===========================================================================
print("== 4. transfer duality: P == 4 R^T ==")
R2, P2 = levels[0]["R"], levels[0]["P"]
err = np.abs((P2 - 4.0 * R2.T)).max() if (P2 - 4.0 * R2.T).nnz else 0.0
ok("bilinear prolongation == 4 x full-weighting^T EXACTLY (2-D; the constant "
   "is 2^d: R rows sum to 1 -- an average -- while P columns sum to 4 -- a "
   "partition of unity spread over ~4 fine nodes; in the A = stencil/h^2 "
   "scaling this is what makes R A P a 2h-scale operator)", err == 0.0, err)
row_sums = np.asarray(R2.sum(axis=1)).ravel()
col_sums = np.asarray(P2.sum(axis=0)).ravel()
ok("full weighting preserves constants: every R row sums to 1 (all 225 rows, "
   "including boundary-adjacent ones); every P column sums to 4",
   np.abs(row_sums - 1.0).max() == 0.0 and np.abs(col_sums - 4.0).max() == 0.0)
err = np.abs(sp.kron(restrict_1d(n), restrict_1d(n)) - R2).max()
stencil = np.outer([0.25, 0.5, 0.25], [0.25, 0.5, 0.25]) * 16.0
ok("2-D full weighting == kron(1-D, 1-D): the 1/16 [1 2 1; 2 4 2; 1 2 1] "
   "stencil (16 x outer([1/4,1/2,1/4]) == [[1,2,1],[2,4,2],[1,2,1]])",
   err == 0.0 and np.array_equal(stencil, [[1, 2, 1], [2, 4, 2], [1, 2, 1]]))

# ===========================================================================
# 5. S3: smooth error is coarse-representable
# ===========================================================================
print("== 5. S3: decimate smoothed error to 2h and interpolate back ==")
Pd = P2.toarray()


def roundtrip_inject(e_flat):
    Ef = e_flat.reshape(n, n)
    return P2 @ Ef[1::2, 1::2].ravel()


def rep_errs(e_flat):
    rt = roundtrip_inject(e_flat)
    inj = float(nrm(e_flat - rt) / nrm(e_flat))
    fw = float(nrm(e_flat - P2 @ (R2 @ e_flat)) / nrm(e_flat))
    copt, *_ = np.linalg.lstsq(Pd, e_flat, rcond=None)
    opt = float(nrm(e_flat - Pd @ copt) / nrm(e_flat))
    return {"inject_interp": inj, "fullweight_interp": fw, "best_in_range_P": opt}


s3 = {str(m): rep_errs(s1_fields[m].ravel()) for m in SWEEPS}
e_rough = (Vh @ np.where(HF, 1.0, 0.0) @ Vh.T).ravel()
e_rough /= nrm(e_rough)
s3["rough"] = rep_errs(e_rough)
info("representation error (inject->interp): " +
     ", ".join(f"m={m}: {s3[str(m)]['inject_interp']:.4f}" for m in SWEEPS)
     + f", rough field: {s3['rough']['inject_interp']:.4f}")
ok(f"S3 smoothed error IS coarse-representable: decimate-then-interpolate "
   f"loses {s3['5']['inject_interp']*100:.1f}% of the 5-sweep error and "
   f"{s3['20']['inject_interp']*100:.1f}% of the 20-sweep error (rel L2)",
   s3["5"]["inject_interp"] < 0.10 and s3["20"]["inject_interp"] < 0.05)
ok(f"S3 a rough field is NOT: same round trip loses "
   f"{s3['rough']['inject_interp']*100:.0f}% of a pure rough-mode field "
   f"(>= 15x the smoothed error's loss); best-in-range(P) errors "
   f"{s3['rough']['best_in_range_P']:.3f} vs {s3['5']['best_in_range_P']:.3f}",
   s3["rough"]["inject_interp"] > 0.5
   and s3["rough"]["inject_interp"] > 15 * s3["5"]["inject_interp"])
ok("S3 representation error falls monotonically with smoothing sweeps "
   + " > ".join(f"{s3[str(m)]['inject_interp']:.4f}" for m in SWEEPS),
   all(s3[str(a)]["inject_interp"] > s3[str(bb)]["inject_interp"]
       for a, bb in zip(SWEEPS, SWEEPS[1:])))
e5_rt = roundtrip_inject(s1_fields[5].ravel())

# ===========================================================================
# 6. one recorded V-cycle: S4 restriction, S5 recursion, S6 correction,
#    S7 post-smoothing, S8 the animated trace
# ===========================================================================
print("== 6. one recorded V(2,2)-cycle from x0 (panels S4-S8) ==")
rec = Recorder(levels)
rec.add("init", 0, x0, b, "down")
x1 = vcycle(levels, 0, x0.copy(), b, rec=rec)
ev = rec.events
stage_names = [f"L{e_['level']}:{e_['stage']}" for e_ in ev]
info(" -> ".join(stage_names))

ok("S8 trace has the full V shape: init, then (pre-smooth, restrict) at "
   "levels 31/15/7/3, division at 1, then (correct, post-smooth) at 3/7/15/31 "
   "-- 18 stages",
   len(ev) == 18
   and [e_["n"] for e_ in ev] == [31, 31, 31, 15, 15, 7, 7, 3, 3, 1,
                                  3, 3, 7, 7, 15, 15, 31, 31])

# every stage's field is finite and its recorded norms match the fields
dev = 0.0
for e_ in ev:
    if "u" in e_:
        lv = levels[e_["level"]]
        dev = max(dev, abs(nrm(e_["e"]) - e_["level_err_norm"]),
                  float(np.abs(e_["u"]).max() * 0 + (not np.isfinite(e_["u"]).all())))
    else:
        dev = max(dev, abs(nrm(e_["r_fine"]) - e_["resid_norm"]),
                  abs(nrm(e_["f_coarse"]) - e_["coarse_rhs_norm"]))
ok("every exported stage field is finite and consistent with its recorded "
   "norms (all 18 stages recomputed)", dev < 1e-9, dev)

# S4: restriction halves the grid, keeps the residual's shape
ev_restrict = [e_ for e_ in ev if e_["stage"] == "restrict residual"]
rft = ev_restrict[0]
ok(f"S4 full weighting: fine 31x31 residual (norm {rft['resid_norm']:.4g}) -> "
   f"coarse 15x15 rhs (norm {rft['coarse_rhs_norm']:.4g}); coarse rhs == "
   f"R r exactly, quarter the unknowns at every level "
   f"(225/961, 49/225, 9/49, 1/9)",
   np.abs(levels[0]["R"] @ rft["r_fine"] - rft["f_coarse"]).max() == 0.0
   and [e_["n"] for e_ in ev_restrict] == [31, 15, 7, 3])

# S5: the coarsest solve is a division
bot = next(e_ for e_ in ev if e_["dir"] == "bottom")
ok(f"S5 recursion bottoms out at 1 unknown: the exact solve is f/16 "
   f"(f = {bot['u'][0]*16:.6g} -> u = {bot['u'][0]:.6g}); residual and level "
   f"error are exactly 0 there",
   bot["resid_norm"] == 0.0 and bot["level_err_norm"] < 1e-15)

# S6: the correction drop at the fine level
pre0 = next(e_ for e_ in ev if e_["stage"] == "pre-smooth" and e_["level"] == 0)
cor0 = next(e_ for e_ in ev if e_["stage"] == "coarse correction added"
            and e_["level"] == 0)
post0 = next(e_ for e_ in ev if e_["stage"] == "post-smooth" and e_["level"] == 0)
drop = cor0["level_err_norm"] / pre0["level_err_norm"]
ok(f"S6 the big drop: adding the interpolated coarse solution cuts the fine "
   f"error {1/drop:.1f}x in one shot (||e|| {pre0['level_err_norm']:.4g} -> "
   f"{cor0['level_err_norm']:.4g})", drop < 0.5)

# S7: post-smoothing kills the interpolation ripple
C_cor = dst2(cor0["e"].reshape(n, n))
C_post = dst2(post0["e"].reshape(n, n))
hf_cor = float((C_cor[HF] ** 2).sum() / (C_cor ** 2).sum())
hf_post = float((C_post[HF] ** 2).sum() / (C_post ** 2).sum())
ok(f"S7 interpolation ripple killed: rough-mode energy fraction of the error "
   f"{hf_cor:.3f} after correction -> {hf_post:.4f} after 2 post-sweeps "
   f"(rough amplitudes cut ~mu_disc^2 = {mu_disc**2:.3f} per mode)",
   hf_post < 0.35 * hf_cor and hf_cor > 0.05)

# the cycle as M^{-1}: linearity identity
Mb = vcycle(levels, 0, np.zeros(N), b - A @ x0)
err = nrm(x1 - (x0 + Mb)) / nrm(x1)
ok("one V-cycle == one application of M^{-1}: vcycle(x0, b) == "
   "x0 + vcycle(0, b - A x0) to machine precision (linearity of every stage)",
   err < 1e-12, err)

# ===========================================================================
# 7. Galerkin R A P vs rediscretization
# ===========================================================================
print("== 7. Galerkin R A P vs rediscretized poisson_2d(15) ==")
Ag = (R2 @ A @ P2).toarray()
Ar = poisson_2d(15).toarray()
nc = 15
sym_err = np.abs(Ag - Ag.T).max() / np.abs(Ag).max()
wg = np.linalg.eigvalsh(0.5 * (Ag + Ag.T))
ok("Galerkin operator R A P is symmetric (R = P^T/4) and positive definite",
   sym_err < 1e-13 and wg[0] > 0.0, sym_err)

# interior stencil: translation-invariant 9-point
Hc = 2.0 * h
cen = (nc // 2) * nc + nc // 2
sten = np.zeros((3, 3))
row = Ag[cen]
for di in (-1, 0, 1):
    for dj in (-1, 0, 1):
        sten[di + 1, dj + 1] = row[cen + di * nc + dj]
sten_h2 = sten * Hc**2
dev = 0.0
for ci in range(2, nc - 2):
    for cj in range(2, nc - 2):
        c2 = ci * nc + cj
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                dev = max(dev, abs(Ag[c2, c2 + di * nc + dj] - sten[di + 1, dj + 1]))
expected_sten = np.array([[-0.25, -0.5, -0.25], [-0.5, 3.0, -0.5],
                          [-0.25, -0.5, -0.25]]) / Hc**2 * Hc**2
ok(f"Galerkin interior stencil is translation-invariant and 9-point: "
   f"H^2 * stencil == [[-1/4,-1/2,-1/4],[-1/2,3,-1/2],[-1/4,-1/2,-1/4]] "
   f"(vs rediscretization's 5-point [[0,-1,0],[-1,4,-1],[0,-1,0]])",
   dev < 1e-12 and np.abs(sten_h2 - expected_sten).max() < 1e-12, dev)

fro_rel = float(nrm(Ag - Ar) / nrm(Ar))
max_rel = float(np.abs(Ag - Ar).max() / np.abs(Ar).max())
info(f"Galerkin vs rediscretization: rel Frobenius diff {fro_rel:.4f}, "
     f"max entry diff {max_rel:.4f} of max|A_2h| -- NOT the same operator")
ok(f"Galerkin != rediscretization, quantified honestly: rel Frobenius "
   f"difference {fro_rel:.3f} (9-point vs 5-point), but both are O(h^2) "
   f"discretizations -- they agree on smooth modes: relative difference of "
   f"Rayleigh quotients on the 3 lowest coarse sine modes < 2%",
   0.05 < fro_rel < 0.5
   and all(abs((v @ (Ag @ v)) / (v @ (Ar @ v)) - 1.0) < 0.02
           for v in [np.outer(np.sin((np.arange(1, nc + 1)) * ki * np.pi / (nc + 1)),
                              np.sin((np.arange(1, nc + 1)) * li * np.pi / (nc + 1))).ravel()
                     for ki, li in [(1, 1), (1, 2), (2, 2)]]))

# two-grid spectral radii at n = 31, nu = 1+1: Galerkin ties report 12's 0.357
Sd = np.eye(N) - OMEGA * (levels[0]["invd"][:, None] * A.toarray())
Pi_g = np.eye(N) - Pd @ np.linalg.solve(Ag, (R2 @ A).toarray())
rho_tg_gal = float(np.abs(np.linalg.eigvals(Sd @ Pi_g @ Sd)).max())
Pi_r = np.eye(N) - Pd @ np.linalg.solve(Ar, (R2 @ A).toarray())
rho_tg_red = float(np.abs(np.linalg.eigvals(Sd @ Pi_r @ Sd)).max())
ok(f"two-grid rho at n = 31 (nu = 1+1, Galerkin coarse solve) = "
   f"{rho_tg_gal:.4f} == report 12's mesh-independent house number 0.357 "
   f"(same error propagator: full-weighting/bilinear Galerkin correction is "
   f"the same A-orthogonal projection as report 12's bilinear coarse space)",
   abs(rho_tg_gal - 0.357) < 0.01)
ok(f"two-grid rho with REDISCRETIZED coarse operator = {rho_tg_red:.4f} -- "
   f"within 15% of Galerkin's; rediscretization is what the stepper runs",
   abs(rho_tg_red - rho_tg_gal) < 0.15 * rho_tg_gal)
ok(f"WHY 0.357: both two-grid radii equal the DISCRETE SMOOTHING FACTOR "
   f"SQUARED, mu_disc^2 = {mu_disc**2:.7f} (= 0.598074^2; one pre- + one "
   f"post-sweep on the worst rough mode, which every coarse correction "
   f"leaves untouched) -- Galerkin and rediscretized agree to machine "
   f"precision because the binding mode is pure high-frequency",
   abs(rho_tg_gal - mu_disc**2) < 1e-6
   and abs(rho_tg_red - mu_disc**2) < 1e-6,
   max(abs(rho_tg_gal - mu_disc**2), abs(rho_tg_red - mu_disc**2)))

# ===========================================================================
# 8. S9: V-cycle convergence at n = 15 / 31 / 63 -- mesh independence
# ===========================================================================
print("== 8. S9: rho_V at n = 15/31/63 (fit + exact), Jacobi contrast ==")
NCYC = 30
FIT_FLOOR = 1e-12
s9 = {}
all_levels = {}
for nn in (15, 31, 63):
    lvls = levels if nn == 31 else build_levels(nn)
    all_levels[nn] = lvls
    An = lvls[0]["A"]
    bn, rods_n = rod_rhs(nn)
    xs = spla.spsolve(An.tocsc(), bn)
    rngn = np.random.default_rng(0)
    xn = rngn.standard_normal(nn * nn)
    xn *= nrm(xs) / nrm(xn)
    e0n = nrm(xs - xn)
    rels = [1.0]
    for _ in range(NCYC):
        xn = vcycle(lvls, 0, xn, bn)
        rels.append(float(nrm(xs - xn) / e0n))
    idx = [i for i, v in enumerate(rels) if v > FIT_FLOOR]
    a_fit, b_fit = 2, idx[-1]                      # skip 2-cycle transient
    rho_fit = (rels[b_fit] / rels[a_fit]) ** (1.0 / (b_fit - a_fit))
    s9[nn] = {"n": nn, "levels": [lv["n"] for lv in lvls],
              "rel_err_per_cycle": r6(rels),
              "fit_window_cycles": [a_fit, b_fit],
              "rho_V_fit": float(rho_fit),
              "rods": rods_n}

# exact spectral radii of the V(2,2) error propagator E = I - M^{-1} A
for nn in (15, 31):
    lvls = all_levels[nn]
    NN = nn * nn
    Ev = np.empty((NN, NN))
    zf = np.zeros(NN)
    for i in range(NN):
        ei = np.zeros(NN)
        ei[i] = 1.0
        Ev[:, i] = vcycle(lvls, 0, ei, zf)
    s9[nn]["rho_V_exact"] = float(np.abs(np.linalg.eigvals(Ev)).max())
    s9[nn]["rho_source"] = "dense eigvals(E_V)"
try:
    lvls = all_levels[63]
    NN = 63 * 63
    zf = np.zeros(NN)
    opv = spla.LinearOperator((NN, NN),
                              matvec=lambda v: vcycle(lvls, 0, v.copy(), zf))
    vals = spla.eigs(opv, k=4, which="LM", ncv=32, tol=1e-9,
                     return_eigenvectors=False)
    s9[63]["rho_V_exact"] = float(np.abs(vals).max())
    s9[63]["rho_source"] = "sparse eigs(E_V, LM)"
except Exception as ex:                                       # pragma: no cover
    s9[63]["rho_V_exact"] = None
    s9[63]["rho_source"] = f"eigs failed: {ex}"

for nn in (15, 31, 63):
    d = s9[nn]
    info(f"n={nn:3d}: rho_V fit {d['rho_V_fit']:.4f} (cycles "
         f"{d['fit_window_cycles'][0]}..{d['fit_window_cycles'][1]}), "
         f"exact {d['rho_V_exact']:.4f} [{d['rho_source']}]")

ok("V(2,2)-cycle fitted rates agree with the exact spectral radius of "
   "E = I - M^{-1}A at every size (fit within [0.8, 1.05] x rho_exact: the "
   "fit is a lower-bound-ish transient average of |eigenvalues| <= rho)",
   all(0.8 * s9[nn]["rho_V_exact"] <= s9[nn]["rho_V_fit"]
       <= 1.05 * s9[nn]["rho_V_exact"] for nn in (15, 31, 63)))
rhos = [s9[nn]["rho_V_exact"] for nn in (15, 31, 63)]
ok(f"MESH INDEPENDENCE: rho_V = {rhos[0]:.4f}, {rhos[1]:.4f}, {rhos[2]:.4f} "
   f"at n = 15/31/63 -- spread {max(rhos)-min(rhos):.4f} (< 0.01), all "
   f"bounded away from 1 (< 0.25): the same "
   f"{math.ceil(math.log(1e-10)/math.log(max(rhos)))} cycles reach 1e-10 at "
   f"every size, while Jacobi's rho crosses 0.999",
   max(rhos) - min(rhos) < 0.01 and max(rhos) < 0.25)

# damped-Jacobi contrast: rho_J = 1 - 2 omega sin^2(pi h/2) -> 1
jac = {}
for nn in (15, 31, 63):
    hh = 1.0 / (nn + 1)
    rho_j = 1.0 - 2.0 * OMEGA * math.sin(math.pi * hh / 2.0) ** 2
    Ssp = sp.identity(nn * nn, format="csr") \
        - OMEGA * sp.diags(all_levels[nn][0]["invd"]) @ all_levels[nn][0]["A"]
    hi = spla.eigsh(Ssp, k=1, which="LA", return_eigenvectors=False, tol=1e-12)
    lo = spla.eigsh(Ssp, k=1, which="SA", return_eigenvectors=False, tol=1e-12)
    rho_meas = float(max(abs(hi[0]), abs(lo[0])))
    jac[nn] = {"rho_exact": float(rho_j), "rho_eigsh": rho_meas,
               "iters_to_1e-10": int(math.ceil(math.log(1e-10) / math.log(rho_j))),
               "sweeps_equal_one_vcycle_drop":
                   float(math.log(s9[nn]["rho_V_exact"]) / math.log(rho_j))}
err = max(abs(jac[nn]["rho_eigsh"] - jac[nn]["rho_exact"]) for nn in (15, 31, 63))
ok("damped-Jacobi spectral radius == 1 - 2 omega sin^2(pi h/2) at all three "
   "sizes (eigsh on the symmetric S confirms the smooth-end formula)",
   err < 1e-9, err)
q1 = (1 - jac[15]["rho_exact"]) / (1 - jac[31]["rho_exact"])
q2 = (1 - jac[31]["rho_exact"]) / (1 - jac[63]["rho_exact"])
ok(f"Jacobi rho -> 1 like 1 - O(h^2): gap ratios {q1:.3f}, {q2:.3f} == 4 "
   f"(h halving) within 3%; rho_J(63) = {jac[63]['rho_exact']:.6f} needs "
   f"{jac[63]['iters_to_1e-10']:,} sweeps to 1e-10 where the V-cycle needs "
   f"{math.ceil(math.log(1e-10)/math.log(s9[63]['rho_V_exact']))} cycles",
   abs(q1 - 4) < 0.12 and abs(q2 - 4) < 0.12
   and jac[63]["rho_exact"] > 0.999)

# ===========================================================================
# 9. V-cycle-preconditioned CG (suite-culture comparison)
# ===========================================================================
print("== 9. V-cycle-preconditioned CG vs plain CG ==")
M15 = np.empty((225, 225))
z15 = np.zeros(225)
for i in range(225):
    ei = np.zeros(225)
    ei[i] = 1.0
    M15[:, i] = vcycle(all_levels[15], 0, z15.copy(), ei)
sym_err = np.abs(M15 - M15.T).max() / np.abs(M15).max()
wM = np.linalg.eigvalsh(0.5 * (M15 + M15.T))
dots = []
for nn in (31, 63):
    rngd = np.random.default_rng(11)
    r1 = rngd.standard_normal(nn * nn)
    r2 = rngd.standard_normal(nn * nn)
    zz = np.zeros(nn * nn)
    Mr1 = vcycle(all_levels[nn], 0, zz.copy(), r1)
    Mr2 = vcycle(all_levels[nn], 0, zz.copy(), r2)
    dots.append(abs(r2 @ Mr1 - r1 @ Mr2) / (nrm(Mr1) * nrm(r2)))
ok(f"the V(2,2)-cycle from zero guess is a valid SPD preconditioner: dense "
   f"M^{{-1}} at n=15 symmetric (rel dev {sym_err:.2e}) with lam_min = "
   f"{wM[0]:.3e} > 0; adjointness dot-tests at n=31/63 pass "
   f"(equal pre/post sweeps of the symmetric Jacobi smoother + P = 4 R^T)",
   sym_err < 1e-10 and wM[0] > 0 and max(dots) < 1e-10, max(dots))

pcg_tbl = {}
for nn in (15, 31, 63):
    An = all_levels[nn][0]["A"]
    bn, _ = rod_rhs(nn)
    zz = np.zeros(nn * nn)
    _, hist_v = pcg(An, bn, M=lambda r, L=all_levels[nn]: vcycle(L, 0, np.zeros(len(r)), r),
                    tol=TOL, maxiter=500)
    _, hist_p = pcg(An, bn, M=None, tol=TOL, maxiter=5000)
    pcg_tbl[nn] = {"vpcg_iters": len(hist_v) - 1, "cg_iters": len(hist_p) - 1,
                   "vpcg_res_hist": r6([float(v) for v in hist_v])}
    info(f"n={nn:3d}: V-PCG {pcg_tbl[nn]['vpcg_iters']} iters, plain CG "
         f"{pcg_tbl[nn]['cg_iters']} iters (rel resid 1e-10)")
vp = [pcg_tbl[nn]["vpcg_iters"] for nn in (15, 31, 63)]
cgs = [pcg_tbl[nn]["cg_iters"] for nn in (15, 31, 63)]
ok(f"V-PCG is mesh-independent: {vp[0]}/{vp[1]}/{vp[2]} iterations at "
   f"n = 15/31/63 (spread <= 2) while plain CG grows ~2x per refinement: "
   f"{cgs[0]}/{cgs[1]}/{cgs[2]}",
   max(vp) - min(vp) <= 2 and max(vp) <= 12
   and cgs[1] > 1.6 * cgs[0] and cgs[2] > 1.6 * cgs[1])

# ===========================================================================
# 10. work count: O(N) per cycle, 4/3 geometric overhead
# ===========================================================================
print("== 10. work count (report 11 Part C flop convention: 1 MAC = 2 flops) ==")


def cycle_flops(lvls, nu1=NU1, nu2=NU2):
    """Per-V-cycle flops. Convention (report 11 Part C / report 14 SS4):
    1 MAC = 2 flops; sparse matvec = 2 nnz; per level with n > 1:
    (nu1+nu2) sweeps x (matvec + f-Au [N] + axpy update [2N])
    + residual (matvec + N) + restriction (2 nnz(R)) + prolong-add
    (2 nnz(P) + N); coarsest solve = 1 divide."""
    total = 0
    per_level = []
    for lv in lvls:
        if lv["n"] == 1:
            per_level.append(1)
            total += 1
            break
        fl = ((nu1 + nu2) * (2 * lv["A"].nnz + 3 * lv["N"])
              + (2 * lv["A"].nnz + lv["N"])
              + 2 * lv["R"].nnz + (2 * lv["P"].nnz + lv["N"]))
        per_level.append(int(fl))
        total += fl
    return int(total), per_level


work = {}
for nn in (15, 31, 63):
    lvls = all_levels[nn]
    tot, per_level = cycle_flops(lvls)
    Ns = [lv["N"] for lv in lvls]
    NNf = Ns[0]
    matvec_fl = 2 * lvls[0]["A"].nnz
    work[nn] = {"flops_per_cycle": tot, "per_level_flops": per_level,
                "level_unknowns": Ns,
                "sum_N_over_Nfine": float(sum(Ns) / NNf),
                "flops_per_unknown": float(tot / NNf),
                "flops_in_fine_matvec_units": float(tot / matvec_fl),
                "fine_level_flops": per_level[0],
                "overhead_vs_fine_level": float(tot / per_level[0])}
    info(f"n={nn:3d}: {tot:,} flops/cycle = {tot/NNf:.1f} per unknown = "
         f"{tot/matvec_fl:.2f} fine matvecs; sum N_l/N = {sum(Ns)/NNf:.4f}")

ok(f"geometric-series overhead: sum of level unknowns / N_fine = "
   f"{work[15]['sum_N_over_Nfine']:.4f} / {work[31]['sum_N_over_Nfine']:.4f} "
   f"/ {work[63]['sum_N_over_Nfine']:.4f} at n = 15/31/63 -- increasing "
   f"toward and bounded by 4/3 (2-D quartering)",
   work[15]["sum_N_over_Nfine"] < work[31]["sum_N_over_Nfine"]
   < work[63]["sum_N_over_Nfine"] < 4.0 / 3.0)
ok(f"O(N) per cycle: flops/unknown = {work[15]['flops_per_unknown']:.1f} / "
   f"{work[31]['flops_per_unknown']:.1f} / {work[63]['flops_per_unknown']:.1f} "
   f"at n = 15/31/63 (a 10% total rise across an 18x size range, settling: "
   f"within 4% for 63-vs-31); whole-cycle "
   f"overhead vs fine-level-only work {work[63]['overhead_vs_fine_level']:.3f}"
   f" < 4/3",
   abs(work[63]["flops_per_unknown"] / work[31]["flops_per_unknown"] - 1) < 0.04
   and abs(work[31]["flops_per_unknown"] / work[15]["flops_per_unknown"] - 1) < 0.10
   and abs(work[63]["flops_per_unknown"] / work[15]["flops_per_unknown"] - 1) < 0.105
   and work[63]["overhead_vs_fine_level"] < 4.0 / 3.0)
solve_cycles = math.ceil(math.log(1e-10) / math.log(s9[63]["rho_V_exact"]))
info(f"full solve to 1e-10 at n=63: {solve_cycles} cycles x "
     f"{work[63]['flops_in_fine_matvec_units']:.1f} matvec-units = "
     f"{solve_cycles * work[63]['flops_per_cycle']:,} flops, O(N) end to end")

npass = sum(c["pass"] for c in results["checks"])
print(f"\n{npass}/{len(results['checks'])} PASS   "
      f"({time.time() - t_start:.1f} s)")

# ===========================================================================
# JSON export: every field / spectrum / rate the stepper needs
# ===========================================================================
results["meta"] = r6({
    "n": n, "h": h, "N": N, "omega": OMEGA, "nu1": NU1, "nu2": NU2,
    "levels": [lv["n"] for lv in levels],
    "coarsest_A": 16.0, "coarsest_h": 0.5,
    "rods_n31": rods, "rod_map_rule":
        "n=32 rod index i -> floor((i+1)(n+1)/33 + 1/2) - 1, rows filled",
    "x0": "standard_normal(seed 0), scaled to ||x*|| (half smooth, half rough)",
    "smoother": "damped Jacobi, S = I - omega D^{-1} A, D = 4/h^2 I",
    "restriction": "full weighting 1/16 [1 2 1; 2 4 2; 1 2 1]",
    "prolongation": "bilinear; P == 4 R^T exactly",
    "coarse_operators": "rediscretized poisson_2d at every level "
                        "(Galerkin R A P quantified in 'galerkin')",
    "smoothing_factor_exact": 0.6, "smoothing_factor_discrete_n31": mu_disc,
    "smoothing_factor_mode": list(mode_disc),
    "tol": TOL,
    "spectra_convention": "abs of orthonormal 2-D DST-I coefficients, "
                          "31x31, mode (k,l) at index [k-1][l-1]",
})

results["s1_stall"] = {
    "sweeps": SWEEPS,
    "fields": {str(m): field(s1_fields[m]) for m in SWEEPS},
    "spectra_abs": {str(m): field(s1_spectra[m]) for m in SWEEPS},
    "err_norm": {str(m): r6(s1_norms[m]) for m in SWEEPS},
    "err_norm_curve_per_sweep": r6(s1_curve),
    "hf_energy_fraction": {str(m): r6(hf_frac[m]) for m in SWEEPS},
    "mode11_coef": {str(m): r6(lf11[m]) for m in SWEEPS},
    "mode11_factor": r6(mu11), "mode11_after100_pred": r6(mu11 ** 100),
}
results["s2_smoothing"] = {
    "mu_surface": field(mu),
    "mu_formula": "1 - omega (sin^2(k pi h/2) + sin^2(l pi h/2))",
    "hf_mask_rule": "max(k,l) >= 16",
    "smoothing_factor_exact": 0.6,
    "smoothing_factor_discrete": r6(mu_disc),
    "arg_mode": list(mode_disc),
    "empirical_fit": r6(float(fit)),
    "fit_window_sweeps": [10, 25],
    "rough_sup_per_sweep": r6(sup_hist),
    "omega_scan": {"omega": r6([float(w) for w in w_grid[::100]]),
                   "hf_sup": r6([float(v) for v in mu_of_w[::100]])},
}
results["s3_representable"] = {
    "rep_errors": {k: r6(v) for k, v in s3.items()},
    "field_smoothed5": field(s1_fields[5]),
    "field_smoothed5_roundtrip": field(e5_rt.reshape(n, n)),
    "field_smoothed5_diff": field(s1_fields[5] - e5_rt.reshape(n, n)),
    "field_rough": field(e_rough.reshape(n, n)),
    "field_rough_roundtrip": field(roundtrip_inject(e_rough).reshape(n, n)),
    "field_rough_diff": field((e_rough - roundtrip_inject(e_rough)).reshape(n, n)),
    "note": "decimate = injection at coarse nodes (fine [1::2,1::2]); "
            "interpolate = bilinear P; errors are rel L2",
}

trace = []
fine_err = None
for e_ in ev:
    entry = {"stage": e_["stage"], "level": int(e_["level"]), "n": int(e_["n"]),
             "dir": e_["dir"], "resid_norm": r6(e_["resid_norm"])}
    if "u" in e_:
        entry["level_err_norm"] = r6(e_["level_err_norm"])
        entry["field_error"] = field(e_["e"].reshape(e_["n"], e_["n"]))
        if e_["level"] == 0:
            fine_err = float(nrm(xstar - e_["u"]))
            entry["fine_err_norm"] = r6(fine_err)
    else:
        entry["coarse_rhs_norm"] = r6(e_["coarse_rhs_norm"])
        entry["field_r_fine"] = field(e_["r_fine"].reshape(e_["n"], e_["n"]))
        ncn = levels[e_["level"] + 1]["n"]
        entry["field_f_coarse"] = field(e_["f_coarse"].reshape(ncn, ncn))
    if "corr" in e_:
        entry["field_correction"] = field(e_["corr"].reshape(e_["n"], e_["n"]))
    trace.append(entry)
results["s8_vcycle_trace"] = {
    "stages": trace,
    "note": "one V(2,2)-cycle at n=31 from x0; field_error = level-exact "
            "solution minus current iterate; fine_err_norm only at level-0 "
            "stages (coarse work does not change the fine iterate)",
    "fine_err_start": r6(float(nrm(xstar - x0))),
    "fine_err_end": r6(float(nrm(xstar - x1))),
    "cycle_contraction": r6(float(nrm(xstar - x1) / nrm(xstar - x0))),
}
results["s7_postsmooth"] = {
    "field_after_correction": field(cor0["e"].reshape(n, n)),
    "field_after_postsmooth": field(post0["e"].reshape(n, n)),
    "spectrum_abs_after_correction": field(np.abs(C_cor)),
    "spectrum_abs_after_postsmooth": field(np.abs(C_post)),
    "hf_fraction_before": r6(hf_cor), "hf_fraction_after": r6(hf_post),
}
results["galerkin"] = r6({
    "rel_frobenius_diff": fro_rel, "max_entry_rel_diff": max_rel,
    "galerkin_stencil_times_H2": [[float(v) for v in rw] for rw in sten_h2],
    "redisc_stencil_times_H2": [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0],
                                [0.0, -1.0, 0.0]],
    "rho_twogrid_nu1_galerkin_n31": rho_tg_gal,
    "rho_twogrid_nu1_rediscretized_n31": rho_tg_red,
    "report12_house_number": 0.357,
})
results["s9_convergence"] = {
    str(nn): {k: (r6(v) if not isinstance(v, dict) else v)
              for k, v in s9[nn].items()} for nn in (15, 31, 63)}
results["s9_jacobi_contrast"] = {str(nn): r6(jac[nn]) for nn in (15, 31, 63)}
results["s9_pcg"] = {str(nn): pcg_tbl[nn] for nn in (15, 31, 63)}
results["work"] = {str(nn): r6(work[nn]) for nn in (15, 31, 63)}
results["work"]["convention"] = (
    "report 11 Part C / report 14 SS4: 1 MAC = 2 flops, A-matvec = 2 nnz(A); "
    "sweep = matvec + N (f-Au) + 2N (axpy); residual = matvec + N; "
    "restriction 2 nnz(R); prolong-add 2 nnz(P) + N; coarsest = 1 divide")
results["checks_total"] = len(results["checks"])
results["checks_passed"] = int(npass)

out = Path(__file__).resolve().parents[2] / "results" / "multigrid.json"
out.write_text(json.dumps(results, indent=1))
size_mb = out.stat().st_size / 1e6
print(f"wrote {out} ({size_mb:.2f} MB)")
