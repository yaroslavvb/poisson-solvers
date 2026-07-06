"""Experiments for report 13 -- "preconditioning as decoupling".

The through-line: the difficulty of minimizing J(u) = 1/2 u'Au - b'u coordinate
by coordinate is exactly the COUPLING A_ij = d2J/du_i du_j (= the precision
cross-terms of the GMRF u ~ N(0, A^{-1}), reports 09-12).  A preconditioner is
a change of variables that removes coupling; this script measures four rungs:

  A. what coupling is: diagonal A one-steps under Jacobi-GD; the DST-I
     eigenbasis (PCA/KL whitening, 09 SS4.3) decouples the real A exactly.
  B. separability: A = H + V is a Kronecker SUM of two commuting halves,
     each half = 32 independent tridiagonal chains; the single-shift
     ADI/Kronecker preconditioner M^{-1} = 2 sigma (V+sI)^{-1}(H+sI)^{-1}
     has closed-form spectrum and kappa(M^{-1}A) = O(sqrt(kappa(A))).
     Bonus: the Green's function A^{-1} is SEMISEPARABLE (rank-1 triangles
     in 1-D, numerically low-rank far blocks in 2-D -- the H-matrix fact).
  C. space: a grid-column separator makes left/right EXACTLY conditionally
     independent (GMRF Markov property); the two-subdomain block-Jacobi
     preconditioner "optimizes each half pretending the other is frozen"
     and its residual coupling lives on the interface (extreme eigenvectors
     of M^{-1}A localize at cols 15/16).
  D. GD vs CG on coupled problems: rank-1 coupling (a 1-factor covariance
     model) is free for CG (2 iterations) and fatal for GD; clustered
     spectra are CG's currency (cf. 06 NPO clustering, 07 Nystrom failure);
     CG iters ~ c sqrt(kappa) across the ladder; CG's directions are
     A-orthogonal = it builds its own decoupled coordinates on the fly
     (09 SS5 sequential regression).

Solvers implemented HERE (not in pcg.py):
  * gd(...)    -- steepest descent with exact line search along z = M^{-1}r,
                  alpha = (r.z)/(z.Az); plain GD when M is None.
  * cg_err(...)-- a small re-implementation of the Hestenes-Stiefel PCG loop
                  of pcg.pcg (bitwise-identical residual history, verified in
                  section 9) that ADDITIONALLY tracks error histories
                  e_k = x_k - x* in l2 and A-norm; x* from spsolve (or exact
                  eigen-solve where available).  We re-implemented rather
                  than callback-instrument pcg so the error tracking cannot
                  perturb the reference iteration.
  Both track ||e_k||_A via the identity e'Ae = -r'e (r = -Ae), so the error
  histories cost no extra matvecs.

Run from the repo root:
    uv run python python/experiments/decoupling.py

Outputs: PASS/FAIL lines, results/decoupling.json, and figures/
  decoupling_error_curves.png, decoupling_interface_mode.png,
  decoupling_gd_vs_cg.png, decoupling_adi_spectrum.png,
  decoupling_semiseparable.png                       (all dpi=150).
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
from preconditioners import block_average_matrix, ic0

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"
RESDIR = ROOT / "results"
np.set_printoptions(precision=4, suppress=True)

T_START = time.time()
SEEDS = {"diag": 1, "x0": 2, "crosspartial_u": 3, "tridiag_rhs": 4,
         "adi_probe": 5, "rank1_v": 6, "rank1_b": 7, "cluster_q": 8,
         "cluster_widths": 9, "cluster_b": 10}
RESULTS = {"meta": {"seeds": SEEDS, "tol": 1e-10}, "checks": [], "figures": [],
           "deviations": []}
N_FAIL = 0


def ok(name, cond):
    global N_FAIL
    cond = bool(cond)
    if not cond:
        N_FAIL += 1
    print(f"{'PASS' if cond else 'FAIL'}: {name}")
    RESULTS["checks"].append({"name": name, "pass": cond})
    return cond


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
# solvers with ERROR tracking (see module docstring)
# ---------------------------------------------------------------------------
def gd(A, b, xstar, M=None, x0=None, tol=1e-10, maxiter=20000):
    """Steepest descent, exact line search along z = M^{-1} r.

    alpha = (r.z)/(z.Az); plain GD when M is None.  Returns
    (x, rel2, relA, iters): rel2[k] = ||x_k - x*||_2 / ||x_0 - x*||_2,
    relA the same in the A-norm (via e'Ae = -r'e; no extra matvec).
    Stops when rel2 <= tol.
    """
    matvec = A if callable(A) else (lambda p: A @ p)
    Mfun = M if M is not None else (lambda rr: rr)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    e = x - xstar
    n0_2 = np.linalg.norm(e)
    r = b - matvec(x)
    n0_A = np.sqrt(max(-(r @ e), 0.0))
    rel2, relA = [1.0], [1.0]
    for _ in range(maxiter):
        z = Mfun(r)
        Az = matvec(z)
        rz = r @ z
        zAz = z @ Az
        if zAz <= 0.0 or rz == 0.0:
            break
        alpha = rz / zAz
        x = x + alpha * z
        r = r - alpha * Az
        e = x - xstar
        rel2.append(np.linalg.norm(e) / n0_2)
        relA.append(np.sqrt(max(-(r @ e), 0.0)) / n0_A)
        if rel2[-1] <= tol:
            break
    return x, rel2, relA, len(rel2) - 1


def cg_err(A, b, xstar, M=None, tol=1e-10, maxiter=2000, stop="err", record_dirs=0):
    """PCG with the identical Hestenes-Stiefel recurrence as pcg.pcg
    (x0 = 0; verified below to reproduce its residual history bitwise),
    plus error histories rel2/relA and optional recording of the first
    `record_dirs` search directions p_k.  stop='err' halts on
    rel l2 error <= tol; stop='res' mimics pcg's residual criterion."""
    matvec = A if callable(A) else (lambda p: A @ p)
    Mfun = M if M is not None else (lambda rr: rr)
    b = np.asarray(b, dtype=np.float64)
    bnorm = np.linalg.norm(b)
    res_hist = [1.0]
    x = np.zeros_like(b)
    r = b.copy()
    z = Mfun(r)
    p = z.copy()
    rz = r @ z
    n0_2 = np.linalg.norm(xstar)          # e0 = -x*
    n0_A = np.sqrt(max(b @ xstar, 0.0))   # e0' A e0 = x*' A x* = b'x*
    rel2, relA, P = [1.0], [1.0], []
    for _ in range(maxiter):
        if record_dirs and len(P) < record_dirs:
            P.append(p.copy())
        Ap = matvec(p)
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        relres = np.linalg.norm(r) / bnorm
        res_hist.append(relres)
        e = x - xstar
        rel2.append(np.linalg.norm(e) / n0_2)
        relA.append(np.sqrt(max(-(r @ e), 0.0)) / n0_A)
        z = Mfun(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
        if ((stop == "err" and rel2[-1] <= tol)
                or (stop == "Aerr" and relA[-1] <= tol)
                or (stop == "res" and relres <= tol)):
            break
    dirs = np.array(P).T if record_dirs else None
    return x, res_hist, rel2, relA, dirs


def kappa_from_Minv(Ad, Minv):
    """Spectrum of M^{-1}A for SPD M^{-1}: eig(M^{-1}A) = eig(R'AR), R = chol(M^{-1})."""
    R = np.linalg.cholesky((Minv + Minv.T) / 2.0)
    w = np.linalg.eigvalsh(R.T @ Ad @ R)
    return w[-1] / w[0], w


def iters_to(rel, tol=1e-10):
    idx = np.nonzero(np.asarray(rel) <= tol)[0]
    return int(idx[0]) if idx.size else None


def tail_rate(hist, m):
    """Geometric-mean per-iteration contraction over the last m steps."""
    h = np.asarray(hist, dtype=np.float64)
    m = min(m, len(h) - 1)
    return float((h[-1] / h[-1 - m]) ** (1.0 / m))


# ===========================================================================
# shared objects: n = 32 2-D Poisson, hot/cold-rod problem of report 11
# ===========================================================================
n = 32
N = n * n
h = 1.0 / (n + 1)
A = poisson_2d(n)
Ad = A.toarray()
d1 = laplacian_1d(n)
eye = sp.identity(n, format="csr")

# 1-D and 2-D eigenvalues (DST-I, verified in report 09's checker)
k1 = np.arange(1, n + 1)
lam1 = 4.0 * np.sin(k1 * np.pi * h / 2.0) ** 2 / h**2
lam2 = np.add.outer(lam1, lam1).ravel()           # kron-sum eigenvalues
kappa_A = lam1[-1] / lam1[0]                       # = lam2.max()/lam2.min()

# hot/cold-rod RHS, exactly as grid_regressions_multiscale.py part B
f = np.zeros((n, n))
rod_hot = [(i, 4) for i in range(3, 9)]
rod_cold = [(i, 27) for i in range(23, 29)]
for i, j in rod_hot:
    f[i, j] = 1.0
for i, j in rod_cold:
    f[i, j] = -1.0
b_rod = f.ravel()
xstar = spla.spsolve(A.tocsc(), b_rod)

RESULTS["meta"]["n"] = n
RESULTS["meta"]["kappa_A"] = float(kappa_A)
RESULTS["meta"]["sqrt_kappa_A"] = float(np.sqrt(kappa_A))
ok(f"kappa(A) at n=32 equals the house number 440.69 (measured {kappa_A:.2f})",
   abs(kappa_A - 440.69) < 0.01)

# ===========================================================================
# PART A -- WHAT COUPLING IS
# ===========================================================================
t0 = time.time()
print("== PART A: what coupling is ==")
partA = {}

# ---- 1a. decoupled limit: diagonal A, Jacobi-GD one-steps ------------------
Nd = 400
rng = np.random.default_rng(SEEDS["diag"])
dvals = rng.uniform(0.5, 10.0, Nd)
b_d = rng.standard_normal(Nd)
x0_d = np.random.default_rng(SEEDS["x0"]).standard_normal(Nd)
xstar_d = b_d / dvals
_, rel2_d, _, _ = gd(np.diag(dvals), b_d, xstar_d,
                     M=lambda r: r / dvals, x0=x0_d, maxiter=5)
partA["diag_onestep_relerr"] = rel2_d[1]
ok(f"diagonal A: Jacobi-GD converges in ONE step from random x0 "
   f"(rel err after 1 step = {rel2_d[1]:.2e})", rel2_d[1] < 1e-12)
# each coordinate is an independent parabola J_i = a_i u_i^2/2 - b_i u_i;
# z = D^{-1}r = x* - x and alpha = 1: the exact per-coordinate minimizer.

# ---- 1b. on the real A the cross-partial IS the coupling -------------------
# J(u) = 1/2 u'Au - b'u  =>  d2J/du_i du_j = A_ij.  J is quadratic, so the
# second-order finite difference is EXACT for any step (we use step 1.0).
rngu = np.random.default_rng(SEEDS["crosspartial_u"])
u_pt = rngu.standard_normal(N)
J = lambda u: 0.5 * u @ (A @ u) - b_rod @ u  # noqa: E731
pairs = [(5, 6), (5, 5 + n), (5, 200), (100, 101), (500, 531)]
dev_cp = 0.0
for (ii, jj) in pairs:
    ei = np.zeros(N); ei[ii] = 1.0
    ej = np.zeros(N); ej[jj] = 1.0
    d2 = J(u_pt + ei + ej) - J(u_pt + ei) - J(u_pt + ej) + J(u_pt)
    dev_cp = max(dev_cp, abs(d2 - Ad[ii, jj]))
partA["cross_partial_max_dev"] = dev_cp
ok(f"d2J/du_i du_j == A_ij (finite differences on 5 pairs, max dev = "
   f"{dev_cp:.2e} on entries of size 4/h^2 = {4/h**2:.0f})", dev_cp < 1e-6)

# ---- 1c. Jacobi-GD on the real A does NOT one-step -------------------------
# diag(A) is the constant 4/h^2, so Jacobi-GD and plain GD are the SAME
# iteration (z = r/d rescales the direction; exact line search undoes it).
dconst = A.diagonal()
inv_diag = 1.0 / dconst
M_jac = lambda r: r * inv_diag  # noqa: E731
_, rel2_gd, relA_gd, it_gd = gd(A, b_rod, xstar, M=None, maxiter=20000)
_, rel2_jgd, relA_jgd, it_jgd = gd(A, b_rod, xstar, M=M_jac, maxiter=20000)
m200 = min(200, len(rel2_gd) - 1, len(rel2_jgd) - 1)
traj_dev = float(np.max(np.abs(np.asarray(rel2_gd[:m200]) - np.asarray(rel2_jgd[:m200]))
                        / np.asarray(rel2_gd[:m200])))
ok(f"diag(A) constant => Jacobi-GD IS plain GD (iters {it_jgd} vs {it_gd}, "
   f"max rel traj dev over first {m200} its = {traj_dev:.1e})",
   dconst.max() == dconst.min() and abs(it_jgd - it_gd) <= 2 and traj_dev < 1e-6)

rate_jgd = tail_rate(relA_jgd, 500)
rate_formula = (kappa_A - 1.0) / (kappa_A + 1.0)
cos_pih = np.cos(np.pi * h)
# Subtlety (echoes report 11's coarse-only note): the rod rhs is ODD under the
# 180-degree rotation, so all EVEN eigenmodes (k1+k2 even) of the error are
# exactly unexcited -- GD runs at the effective kappa over excited modes only.
# kappa is a worst-case-rhs bound; a generic (GRF) rhs excites everything.
V1e = np.sqrt(2.0 * h) * np.sin(np.outer(np.arange(1, n + 1),
                                         np.arange(1, n + 1)) * np.pi * h)
coeff_rod = np.kron(V1e, V1e).T @ xstar
excited = np.abs(coeff_rod) > 1e-10 * np.abs(coeff_rod).max()
lam_exc_min, lam_exc_max = lam2[excited].min(), lam2[excited].max()
kappa_eff = lam_exc_max / lam_exc_min
rate_eff = (kappa_eff - 1.0) / (kappa_eff + 1.0)
ok(f"rod rhs parity: lowest excited mode is (1,2) with lam = "
   f"{lam_exc_min:.3f} = lam_1+lam_2 = {lam1[0]+lam1[1]:.3f}; effective "
   f"kappa = {kappa_eff:.2f} (not {kappa_A:.2f})",
   abs(lam_exc_min - (lam1[0] + lam1[1])) < 1e-8 * lam_exc_min)
ok(f"Jacobi-GD does NOT one-step: {it_jgd} iterations to 1e-10 on the rod "
   f"problem; asymptotic A-norm rate {rate_jgd:.6f} = (kappa_eff-1)/"
   f"(kappa_eff+1) = {rate_eff:.6f} at the parity-effective kappa",
   it_jgd > 1000 and abs(rate_jgd - rate_eff) < 1e-3)

b_grf_a = grf_rhs(n)                      # generic rhs: all modes excited
xstar_grf = spla.spsolve(A.tocsc(), b_grf_a)
_, rel2_ggrf, relA_ggrf, it_ggrf = gd(A, b_grf_a, xstar_grf, M=M_jac,
                                      maxiter=20000)
rate_ggrf = tail_rate(relA_ggrf, 500)
ok(f"on a generic (GRF) rhs Jacobi-GD attains the full-kappa rate: measured "
   f"{rate_ggrf:.6f} vs (kappa-1)/(kappa+1) = {rate_formula:.6f} "
   f"({it_ggrf} iterations to 1e-10)",
   abs(rate_ggrf - rate_formula) < 1e-3)
ok(f"(kappa-1)/(kappa+1) == cos(pi h) == rho(Jacobi Richardson) of report 12 "
   f"({rate_formula:.6f} vs {cos_pih:.6f})", abs(rate_formula - cos_pih) < 1e-12)
partA["jacobi_gd"] = {
    "iters_to_1e-10_rod": it_jgd, "rate_measured_rod": rate_jgd,
    "kappa_eff_rod": float(kappa_eff), "rate_formula_eff": float(rate_eff),
    "lam_min_excited": float(lam_exc_min),
    "iters_to_1e-10_grf": it_ggrf, "rate_measured_grf": rate_ggrf,
    "rate_formula": rate_formula, "cos_pi_h": float(cos_pih)}

# ---- 2. exact decoupling by frequency: the DST-I eigenbasis ----------------
V1 = np.sqrt(2.0 * h) * np.sin(np.outer(np.arange(1, n + 1),
                                        np.arange(1, n + 1)) * np.pi * h)
V2 = np.kron(V1, V1)                                # 2-D sine products
orth_dev = float(np.max(np.abs(V2.T @ V2 - np.eye(N))))
T = V2.T @ Ad @ V2
off = T - np.diag(np.diag(T))
off_rel = float(np.max(np.abs(off)) / np.max(np.diag(T)))
diag_dev = float(np.max(np.abs(np.diag(T) - lam2)) / lam2.max())
x_dst = V2 @ ((V2.T @ b_rod) / lam2)
relerr_dst = float(np.linalg.norm(x_dst - xstar) / np.linalg.norm(xstar))
partA["dst"] = {"orth_dev": orth_dev, "offdiag_rel": off_rel,
                "diag_dev_rel": diag_dev, "onepass_relerr": relerr_dst}
ok(f"2-D DST-I basis is orthonormal (max |V'V - I| = {orth_dev:.1e}) and "
   f"diagonalizes A (max offdiag of V'AV = {off_rel:.1e} rel to diag; "
   f"diag == lam_i+lam_j to {diag_dev:.1e})",
   orth_dev < 1e-12 and off_rel < 1e-10 and diag_dev < 1e-12)
ok(f"in the eigenbasis the problem is {N} independent scalar parabolas: "
   f"one-pass solve x = V(V'b/lam) matches spsolve to {relerr_dst:.1e}",
   relerr_dst < 1e-10)
RESULTS["part_a"] = partA
info(f"Part A done in {time.time()-t0:.1f}s")

# ===========================================================================
# PART B -- SEPARABLE: THE KRONECKER/ADI STORY
# ===========================================================================
t0 = time.time()
print("== PART B: separability -- Kronecker sum, ADI, semiseparable inverse ==")
partB = {}

# ---- 3. A = H + V, commuting halves ----------------------------------------
H = (sp.kron(eye, d1) / h**2).tocsr()   # couples within grid ROWS (j-index)
Vv = (sp.kron(d1, eye) / h**2).tocsr()  # couples within grid COLUMNS (i-index)
split_dev = float(np.max(np.abs((H + Vv - A).toarray())))
comm_dev = float(np.max(np.abs((H @ Vv - Vv @ H).toarray())))
w_dense = np.linalg.eigvalsh(Ad)
eig_dev = float(np.max(np.abs(w_dense - np.sort(lam2))) / lam2.max())
partB["split"] = {"split_dev": split_dev, "commutator_fro_max": comm_dev,
                  "eigsum_dev_rel": eig_dev}
ok(f"A == H + V exactly (max |H+V-A| = {split_dev:.1e})", split_dev == 0.0)
ok(f"H and V COMMUTE exactly: max |HV - VH| = {comm_dev:.1e} (Kronecker "
   f"algebra: HV = VH = kron(d1,d1)/h^4)", comm_dev == 0.0)
ok(f"Kronecker SUM spectrum: eig(A) == {{lam_i + lam_j}} "
   f"(max rel dev {eig_dev:.1e})", eig_dev < 1e-12)

# ---- 4. each half = 32 independent tridiagonal chains ----------------------
Hb = H.toarray().reshape(n, n, n, n)     # [i, j, i', j']
Vb = Vv.toarray().reshape(n, n, n, n)
massH = np.abs(Hb).sum(axis=(1, 3))      # coupling mass between grid rows i,i'
massV = np.abs(Vb).sum(axis=(0, 2))      # coupling mass between grid cols j,j'
offH = float(np.max(np.abs(massH - np.diag(np.diag(massH)))))
offV = float(np.max(np.abs(massV - np.diag(np.diag(massV)))))
d1h = (d1 / h**2).toarray()
blk_dev = max(float(np.max(np.abs(Hb[i, :, i, :] - d1h))) for i in range(n))
ok(f"H is block-diagonal: zero coupling between different grid rows "
   f"(max = {offH:.1e}); all 32 blocks == d1/h^2 (max dev {blk_dev:.1e})",
   offH == 0.0 and blk_dev == 0.0)
ok(f"V couples only within grid columns (max cross-column mass = {offV:.1e})",
   offV == 0.0)

sigma = float(np.sqrt(lam1[0] * lam1[-1]))     # geometric-mean shift
ab = np.zeros((2, n))                          # banded (d1/h^2 + sigma I)
ab[0] = 2.0 / h**2 + sigma
ab[1, :-1] = -1.0 / h**2


def solve_rows(Y):
    """(H + sigma I)^{-1} Y: 32 INDEPENDENT tridiagonal solves, one per grid
    row (same matrix, so one banded factorization serves all rows). O(N)."""
    return sla.solveh_banded(ab, Y.T, lower=True).T


def solve_cols(Y):
    """(V + sigma I)^{-1} Y: 32 independent tridiagonal solves, one per grid
    column. O(N)."""
    return sla.solveh_banded(ab, Y, lower=True)


rngt = np.random.default_rng(SEEDS["tridiag_rhs"])
y_t = rngt.standard_normal(N)
xh_tri = solve_rows(y_t.reshape(n, n)).ravel()
xh_sp = spla.spsolve((H + sigma * sp.identity(N)).tocsc(), y_t)
devH = float(np.linalg.norm(xh_tri - xh_sp) / np.linalg.norm(xh_sp))
xv_tri = solve_cols(y_t.reshape(n, n)).ravel()
xv_sp = spla.spsolve((Vv + sigma * sp.identity(N)).tocsc(), y_t)
devV = float(np.linalg.norm(xv_tri - xv_sp) / np.linalg.norm(xv_sp))
partB["tridiag_solve_dev"] = {"H": devH, "V": devV}
ok(f"(H + sigma I)x=y via 32 separate tridiagonal solves matches spsolve "
   f"(rel dev {devH:.1e}); same for V by columns ({devV:.1e})",
   devH < 1e-12 and devV < 1e-12)

# ---- 5. the ADI / Kronecker-product preconditioner --------------------------
# M^{-1} = 2 sigma (V + sigma I)^{-1} (H + sigma I)^{-1}, sigma = sqrt(l1*ln).
# COST: one M^{-1} apply = 32 row-solves + 32 column-solves = 64 tridiagonal
# solves = O(N) flops -- same order as one sparse matvec with A.
M_adi = lambda r: 2.0 * sigma * solve_cols(solve_rows(r.reshape(n, n))).ravel()  # noqa: E731

Hd_s = H.toarray() + sigma * np.eye(N)
Vd_s = Vv.toarray() + sigma * np.eye(N)
Minv_adi = 2.0 * sigma * np.linalg.solve(Vd_s, np.linalg.solve(Hd_s, np.eye(N)))
asym = float(np.max(np.abs(Minv_adi - Minv_adi.T)) / np.max(np.abs(Minv_adi)))
w_minv = np.linalg.eigvalsh((Minv_adi + Minv_adi.T) / 2.0)
r_probe = np.random.default_rng(SEEDS["adi_probe"]).standard_normal(N)
apply_dev = float(np.linalg.norm(M_adi(r_probe) - Minv_adi @ r_probe)
                  / np.linalg.norm(Minv_adi @ r_probe))
ok(f"ADI M is SPD: commuting SPD factors => M^{{-1}} symmetric "
   f"(asym {asym:.1e}), eigenvalues in [{w_minv[0]:.2e}, {w_minv[-1]:.2e}] > 0; "
   f"tridiagonal apply == dense M^{{-1}}r ({apply_dev:.1e})",
   asym < 1e-10 and w_minv[0] > 0 and apply_dev < 1e-10)

Fgrid = 2.0 * sigma * np.add.outer(lam1, lam1) / np.multiply.outer(lam1 + sigma,
                                                                   lam1 + sigma)
fvals = np.sort(Fgrid.ravel())
kap_adi, w_adi = kappa_from_Minv(Ad, Minv_adi)
spec_dev = float(np.max(np.abs(w_adi - fvals)) / fvals.max())
ok(f"spectrum of M^{{-1}}A == closed form f(lam_i,lam_j) = "
   f"2s(li+lj)/((li+s)(lj+s)) (max dev {spec_dev:.1e})", spec_dev < 1e-10)

imin = np.unravel_index(np.argmin(Fgrid), Fgrid.shape)
imax = np.unravel_index(np.argmax(Fgrid), Fgrid.shape)
f_11 = Fgrid[0, 0]
f_nn = Fgrid[-1, -1]
balance_dev = abs(f_11 - f_nn) / f_11
ok(f"geometric-mean shift balances the corners: f(l1,l1) == f(ln,ln) = "
   f"{f_11:.6f} (rel dev {balance_dev:.1e}); min of f at pure corners "
   f"{tuple(int(q+1) for q in imin)}, max at mixed corner "
   f"{tuple(int(q+1) for q in imax)} = {Fgrid[imax]:.6f}",
   balance_dev < 1e-12 and set((imin, (int(imax[0]), int(imax[1]))))
   <= {(0, 0), (n - 1, n - 1), (0, n - 1), (n - 1, 0)})
sqrt_eff = kap_adi / np.sqrt(kappa_A)
ok(f"the square-root effect: kappa(M^-1 A) = {kap_adi:.3f} = "
   f"{sqrt_eff:.3f} * sqrt(kappa(A)) = {sqrt_eff:.3f} * {np.sqrt(kappa_A):.3f}",
   kap_adi < np.sqrt(kappa_A))
partB["adi"] = {"sigma": sigma, "lam1d_min": float(lam1[0]),
                "lam1d_max": float(lam1[-1]), "kappa_MinvA": float(kap_adi),
                "kappa_A": float(kappa_A), "sqrt_kappa_A": float(np.sqrt(kappa_A)),
                "ratio_to_sqrt": float(sqrt_eff),
                "f_min": float(fvals[0]), "f_max": float(fvals[-1]),
                "f_argmin_1based": [int(q) + 1 for q in imin],
                "f_argmax_1based": [int(q) + 1 for q in imax],
                "spec_dev": spec_dev, "corner_balance_dev": float(balance_dev)}

# CG and GD with the ADI preconditioner (error curves used in section 9)
_, res_adi, rel2_cg_adi, relA_cg_adi, _ = cg_err(A, b_rod, xstar, M=M_adi)
_, rel2_gd_adi, relA_gd_adi, it_gd_adi = gd(A, b_rod, xstar, M=M_adi, maxiter=5000)
partB["adi"]["cg_iters_err"] = iters_to(rel2_cg_adi)
partB["adi"]["gd_iters_err"] = iters_to(rel2_gd_adi)
info(f"ADI-CG: {iters_to(rel2_cg_adi)} its to rel err 1e-10; "
     f"ADI-GD: {iters_to(rel2_gd_adi)} its "
     f"(worst-case prediction ln(1e10)/ln((k+1)/(k-1)) = "
     f"{np.log(1e10)/-np.log((kap_adi-1)/(kap_adi+1)):.0f})")

# ---- 6. semiseparability of the inverse -------------------------------------
A1d = (d1 / h**2).toarray()
S1 = np.linalg.inv(A1d)
xg = np.arange(1, n + 1) * h
G1 = np.outer(h * xg, 1.0 - xg)          # rank-1: u_i v_j = h x_i (1 - x_j)
iu = np.triu_indices(n)
tri_dev = float(np.max(np.abs(S1[iu] - G1[iu])) / S1.max())
rank_S1 = int(np.linalg.matrix_rank(S1))
ok(f"1-D Green's function: (A^-1)_ij == h x_i (1-x_j) for i<=j "
   f"(max rel dev {tri_dev:.1e}) -- the upper triangle of A^-1 IS a rank-1 "
   f"matrix (Brownian-bridge covariance), while A^-1 itself has full rank "
   f"{rank_S1}", tri_dev < 1e-10 and rank_S1 == n)

S2 = np.linalg.inv(Ad)
Bfar = S2[: 8 * n, 24 * n:]              # grid rows 0-7  vs  rows 24-31
sv_far = sla.svdvals(Bfar)
numrank_far = int(np.sum(sv_far > 1e-8 * sv_far[0]))
sv_S2_min = float(np.linalg.eigvalsh(S2)[0])
ok(f"2-D H-matrix fact: far off-diagonal block of A^-1 (rows 0-7 x rows "
   f"24-31, 256x256) has numerical rank {numrank_far} at 1e-8 "
   f"(sv1 = {sv_far[0]:.2e}), while A^-1 is dense (all entries > 0: min "
   f"{S2.min():.2e}) and full-rank (lam_min = {sv_S2_min:.2e} > 0)",
   numrank_far <= 40 and S2.min() > 0 and sv_S2_min > 0)
partB["semiseparable"] = {
    "tri_dev_rel": tri_dev, "rank_S1": rank_S1,
    "far_block_rows": [0, 7], "far_block_cols_rows": [24, 31],
    "far_block_sv": sv_far[:30], "far_block_numrank_1e-8": numrank_far,
    "S2_min_entry": float(S2.min()), "S2_lam_min": sv_S2_min}
RESULTS["part_b"] = partB
info(f"Part B done in {time.time()-t0:.1f}s")

# ===========================================================================
# PART C -- SPACE: TWO SUBDOMAINS AND CONDITIONAL INDEPENDENCE
# ===========================================================================
t0 = time.time()
print("== PART C: two subdomains, Markov property, interface coupling ==")
partC = {}
cols_idx = lambda cols: (np.arange(n)[:, None] * n + np.asarray(cols)[None, :]).ravel()  # noqa: E731

# ---- 7. GMRF Markov property across a column separator ---------------------
iL = cols_idx(range(0, 15))       # cols 0..14
iI = cols_idx([15, 16])           # separator: cols 15..16
iR = cols_idx(range(17, n))       # cols 17..31
A_LR_sep = A[np.ix_(iL, iR)]
S_LR = S2[np.ix_(iL, iR)]
S_LI = S2[np.ix_(iL, iI)]
S_II = S2[np.ix_(iI, iI)]
S_IR = S2[np.ix_(iI, iR)]
cond_cov = S_LR - S_LI @ np.linalg.solve(S_II, S_IR)
marg_max = float(np.max(np.abs(S_LR)))
cond_max = float(np.max(np.abs(cond_cov)))
partC["markov"] = {"A_LR_nnz": int(A_LR_sep.nnz), "marginal_max": marg_max,
                   "conditional_max": cond_max,
                   "conditional_rel": cond_max / marg_max}
ok(f"no precision edges cross the separator: nnz(A_LR) = {A_LR_sep.nnz}",
   A_LR_sep.nnz == 0)
ok(f"GMRF Markov property: Cov(u_L, u_R | u_I) == 0 exactly "
   f"(max |cond cov| = {cond_max:.1e} vs marginal max |Sigma_LR| = "
   f"{marg_max:.1e}; ratio {cond_max/marg_max:.1e})",
   cond_max / marg_max < 1e-10)

# ---- 8. two-subdomain block-Jacobi ------------------------------------------
iL8 = cols_idx(range(0, 16))      # cols 0..15  (a 32x16 Poisson block)
iR8 = cols_idx(range(16, n))      # cols 16..31
A_LL = A[np.ix_(iL8, iL8)].tocsc()
A_RR = A[np.ix_(iR8, iR8)].tocsc()
lu_L, lu_R = spla.splu(A_LL), spla.splu(A_RR)


def M_bj(r):
    """Optimize each half pretending the other is frozen: one independent
    32x16 Poisson solve per subdomain."""
    z = np.empty_like(r)
    z[iL8] = lu_L.solve(r[iL8])
    z[iR8] = lu_R.solve(r[iR8])
    return z


Minv_bj = np.zeros((N, N))
Minv_bj[np.ix_(iL8, iL8)] = np.linalg.inv(A_LL.toarray())
Minv_bj[np.ix_(iR8, iR8)] = np.linalg.inv(A_RR.toarray())
kap_bj, w_bj = kappa_from_Minv(Ad, Minv_bj)
n_unit = int(np.sum(np.abs(w_bj - 1.0) < 1e-8))
A_LR8 = A[np.ix_(iL8, iR8)]
rank_LR8 = int(np.linalg.matrix_rank(A_LR8.toarray()))
pair_dev = float(np.max(np.abs(w_bj + w_bj[::-1] - 2.0)))
ok(f"residual coupling is the interface: A_LR has nnz = {A_LR8.nnz} = one "
   f"column of edges, rank = {rank_LR8}; M^-1 A has eigenvalue 1 with "
   f"multiplicity {n_unit} = N - 2*rank = {N - 2*rank_LR8}, rest split "
   f"1 +/- mu (max pairing dev {pair_dev:.1e})",
   A_LR8.nnz == 32 and rank_LR8 == 32 and n_unit == N - 64 and pair_dev < 1e-8)
partC["block_jacobi"] = {"kappa_MinvA": float(kap_bj),
                         "lam_min": float(w_bj[0]), "lam_max": float(w_bj[-1]),
                         "n_unit_eigs": n_unit, "A_LR_nnz": int(A_LR8.nnz),
                         "A_LR_rank": rank_LR8, "eig_pairing_dev": pair_dev,
                         "spectrum_low10": w_bj[:10], "spectrum_high10": w_bj[-10:]}
info(f"block-Jacobi(2): kappa(M^-1 A) = {kap_bj:.2f}, spectrum in "
     f"[{w_bj[0]:.4f}, {w_bj[-1]:.4f}]")

# extreme eigenvectors localize at the interface (generalized eig A v = lam M v,
# M = blockdiag(A_LL, A_RR) in the original ordering)
Md_bj = Ad.copy()
Md_bj[np.ix_(iL8, iR8)] = 0.0
Md_bj[np.ix_(iR8, iL8)] = 0.0
wg, Vg = sla.eigh(Ad, Md_bj)
v_min = Vg[:, 0].reshape(n, n)
v_max = Vg[:, -1].reshape(n, n)
if v_min[:, 15].sum() < 0:
    v_min = -v_min
E_min = (v_min**2).sum(axis=0)
E_max = (v_max**2).sum(axis=0)
peak_min, peak_max = int(np.argmax(E_min)), int(np.argmax(E_max))
frac_iface = float((E_min[14:18].sum()) / E_min.sum())
ok(f"extreme eigenvectors LOCALIZE AT THE INTERFACE: column-energy peak of "
   f"the lam_min = {wg[0]:.4f} eigenvector at col {peak_min}, of lam_max = "
   f"{wg[-1]:.4f} at col {peak_max} (cols 14-17 carry {frac_iface:.0%} of "
   f"the lam_min mode's energy)",
   peak_min in (15, 16) and peak_max in (15, 16))
partC["block_jacobi"]["interface_mode"] = {
    "lam_min": float(wg[0]), "lam_max": float(wg[-1]),
    "col_energy_peak_min": peak_min, "col_energy_peak_max": peak_max,
    "energy_frac_cols14_17": frac_iface, "col_energy_min": E_min / E_min.sum()}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
vm = np.abs(v_min).max()
im = axes[0].imshow(v_min, cmap="RdBu_r", vmin=-vm, vmax=vm)
axes[0].axvline(15.5, color="k", lw=0.8, ls="--")
axes[0].set_title(f"eigenvector of $\\lambda_{{\\min}}(M^{{-1}}A)$ = {wg[0]:.3f}\n"
                  "(block-Jacobi, cols 0-15 | 16-31)")
axes[0].set_xlabel("grid col $j$")
axes[0].set_ylabel("grid row $i$")
fig.colorbar(im, ax=axes[0], fraction=0.046)
axes[1].plot(E_min / E_min.sum(), "o-", label=f"$\\lambda_{{\\min}}$ mode")
axes[1].plot(E_max / E_max.sum(), "s--", ms=4, label=f"$\\lambda_{{\\max}}$ mode")
axes[1].axvspan(14.5, 16.5, color="0.85", label="interface cols 15/16")
axes[1].set_xlabel("grid col $j$")
axes[1].set_ylabel("column energy fraction $\\sum_i v_{ij}^2$")
axes[1].set_title("the residual coupling lives on the interface")
axes[1].legend()
fig.tight_layout()
fig.savefig(FIGDIR / "decoupling_interface_mode.png", dpi=150)
plt.close(fig)
RESULTS["figures"].append("decoupling_interface_mode.png")

# CG + GD with block-Jacobi(2)
_, res_bj, rel2_cg_bj, relA_cg_bj, _ = cg_err(A, b_rod, xstar, M=M_bj)
_, rel2_gd_bj, relA_gd_bj, it_gd_bj = gd(A, b_rod, xstar, M=M_bj, maxiter=20000)
partC["block_jacobi"]["cg_iters_err"] = iters_to(rel2_cg_bj)
partC["block_jacobi"]["gd_iters_err"] = iters_to(rel2_gd_bj)
info(f"blockJacobi2-CG: {iters_to(rel2_cg_bj)} its; blockJacobi2-GD: "
     f"{iters_to(rel2_gd_bj)} its to rel err 1e-10")
RESULTS["part_c"] = partC
info(f"Part C done in {time.time()-t0:.1f}s")

# ===========================================================================
# 9. THE LADDER on the hot/cold-rod problem (error curves + table)
# ===========================================================================
t0 = time.time()
print("== 9. ladder of decouplers on the hot/cold-rod problem ==")

# two-level additive IC(0) + 4x4 block averages, exactly as report 11
Lic = ic0(Ad)
M_ic = lambda r: sla.solve_triangular(  # noqa: E731
    Lic.T, sla.solve_triangular(Lic, r, lower=True), lower=False)
Zb = block_average_matrix(n, 4)
Ac = Zb.T @ (A @ Zb)
Ac_cho = sla.cho_factor(Ac)
M_tl = lambda r: M_ic(r) + Zb @ sla.cho_solve(Ac_cho, Zb.T @ r)  # noqa: E731
Licinv = sla.solve_triangular(Lic, np.eye(N), lower=True)
Minv_tl = Licinv.T @ Licinv + Zb @ np.linalg.solve(Ac, Zb.T)
kap_tl, w_tl = kappa_from_Minv(Ad, Minv_tl)

# plain CG (also: bitwise cross-check of cg_err against pcg.pcg)
x_ref, res_ref = pcg(A, b_rod, M=None, tol=1e-10, maxiter=2000)
_, res_mine, rel2_cg, relA_cg, _ = cg_err(A, b_rod, xstar, M=None,
                                          stop="res", maxiter=2000)
mlen = min(len(res_ref), len(res_mine))
bit_dev = float(np.max(np.abs(np.asarray(res_ref[:mlen]) - np.asarray(res_mine[:mlen]))))
ok(f"cg_err reproduces pcg.pcg's residual history (max dev = {bit_dev:.1e} "
   f"over {mlen} entries -- same recurrence, error tracking is passive)",
   bit_dev <= 1e-13)
# rerun plain CG with the error stopping rule for the ladder table
_, _, rel2_cg, relA_cg, _ = cg_err(A, b_rod, xstar, M=None, maxiter=2000)
_, _, rel2_cg_tl, relA_cg_tl, _ = cg_err(A, b_rod, xstar, M=M_tl)

LADDER = {
    "gd": rel2_gd, "jacobi_gd": rel2_jgd, "cg": rel2_cg,
    "adi_cg": rel2_cg_adi, "blockjacobi2_cg": rel2_cg_bj,
    "twolevel_ic0_cg": rel2_cg_tl, "adi_gd": rel2_gd_adi,
    "blockjacobi2_gd": rel2_gd_bj,
}
iters_table = {k: iters_to(v) for k, v in LADDER.items()}
iters_table["dst_direct"] = 1
ok("ladder ordering (iterations to rel l2 err 1e-10): GD >= CG >= ADI-CG >= "
   "blockJacobi2-CG-ish >= twolevel-CG >= DST(1): " +
   ", ".join(f"{k}={v}" for k, v in iters_table.items()),
   iters_table["gd"] > iters_table["cg"] > iters_table["adi_cg"]
   >= iters_table["twolevel_ic0_cg"] > 1
   and iters_table["cg"] > iters_table["blockjacobi2_cg"])

fig, ax = plt.subplots(figsize=(8.4, 5.6))
styles = {
    "gd": ("plain GD", "tab:red", "-", 1.6),
    "jacobi_gd": ("Jacobi-GD (identical: const diag)", "darkorange", "--", 1.2),
    "adi_gd": ("ADI-GD", "tab:pink", ":", 1.2),
    "blockjacobi2_gd": ("blockJacobi2-GD", "tab:brown", ":", 1.2),
    "cg": ("CG", "tab:blue", "-", 1.6),
    "adi_cg": ("ADI-CG", "tab:green", "-", 1.6),
    "blockjacobi2_cg": ("blockJacobi2-CG", "tab:purple", "-", 1.6),
    "twolevel_ic0_cg": ("twolevel IC(0)+4x4-CG", "tab:cyan", "-", 1.6),
}
for k, (lab, c, ls, lw) in styles.items():
    hcurve = LADDER[k]
    ax.loglog(np.arange(1, len(hcurve)), hcurve[1:], ls, color=c, lw=lw,
              label=f"{lab}  [{iters_table[k]}]")
ax.loglog([1], [relerr_dst], "k*", ms=14,
          label=f"DST direct (1 pass)  [{relerr_dst:.0e}]")
ax.axhline(1e-10, color="0.7", lw=0.8, ls=":")
ax.set_xlabel("iteration")
ax.set_ylabel(r"relative error $\|x_k - x^*\|_2 / \|x^*\|_2$")
ax.set_title("the decoupling ladder on the hot/cold-rod problem (n=32)\n"
             "[iterations to $10^{-10}$] in legend")
ax.legend(fontsize=8, loc="lower left")
ax.set_ylim(1e-13, 3)
fig.tight_layout()
fig.savefig(FIGDIR / "decoupling_error_curves.png", dpi=150)
plt.close(fig)
RESULTS["figures"].append("decoupling_error_curves.png")

RESULTS["ladder"] = {
    "iters_to_1e-10_err": iters_table,
    "dst_direct_relerr": relerr_dst,
    "curves": {k: (v if len(v) < 400 else list(np.asarray(v)[::10]))
               for k, v in LADDER.items()},
    "curves_note": "curves longer than 400 entries stored subsampled 1:10",
    "kappa": {"none": float(kappa_A), "adi": float(kap_adi),
              "blockjacobi2": float(kap_bj), "twolevel_ic0": float(kap_tl)},
}
info(f"ladder done in {time.time()-t0:.1f}s")

# ===========================================================================
# PART D -- GD vs CG ON COUPLED PROBLEMS
# ===========================================================================
t0 = time.time()
print("== PART D: GD vs CG on coupled problems ==")
partD = {}

# ---- 10. rank-1 coupling: A = I + rho v v', dim 400 -------------------------
# Statistical reading: a 1-factor covariance model -- everyone coupled through
# one common factor.  TWO distinct eigenvalues {1, 1+rho} => CG needs exactly
# 2 iterations.  For GD the 2-eigenvalue dynamics is exactly solvable: with
# m = kappa * (v-component of e0)^2 / (perp-component of e0)^2, the per-step
# A-norm^2 contraction is CONSTANT,
#     f(m) = (kappa-1)^2 m / ((1 + kappa^2 m)(1 + m)),
# because the mixing ratio obeys the exact involution m -> 1/(kappa^2 m)
# (w' = -1/(kappa^2 w) for the component ratio w) and f is invariant under it.
# The textbook rate (kappa-1)/(kappa+1) = sqrt(f(1/kappa)) is the WORST case
# over m, attained iff m = 1/kappa; the rhs b = v + w_perp realizes it exactly
# (then x* = w_perp + v/kappa and m0 = 1/kappa).  We machine-check both.
dim = 400
rngv = np.random.default_rng(SEEDS["rank1_v"])
v1f = rngv.standard_normal(dim)
v1f /= np.linalg.norm(v1f)
b_rand = np.random.default_rng(SEEDS["rank1_b"]).standard_normal(dim)
w_perp = b_rand - (v1f @ b_rand) * v1f
w_perp /= np.linalg.norm(w_perp)
b_worst = v1f + w_perp

RHOS = [1e2, 1e4, 1e6]
GD_CAP = 120_000
partD["rank1"] = {"dim": dim, "rhos": RHOS, "gd_cap": GD_CAP, "cases": {}}
rank1_curves = {}
for rho in RHOS:
    kap = 1.0 + rho
    Aop = (lambda rho_: lambda p: p + rho_ * (v1f @ p) * v1f)(rho)
    Adense = np.eye(dim) + rho * np.outer(v1f, v1f)
    case = {}
    for tag, bb in [("random_b", b_rand), ("worst_b", b_worst)]:
        xst = np.linalg.solve(Adense, bb)
        _, res_c, rel2_c, relA_c, _ = cg_err(Aop, bb, xst, maxiter=25)
        # exact 2-eigenvalue GD theory for this rhs:
        beta0 = v1f @ xst                        # e0 = -x*
        a0sq = xst @ xst - beta0**2
        m0 = kap * beta0**2 / a0sq
        f_pred = (kap - 1.0) ** 2 * m0 / ((1.0 + kap**2 * m0) * (1.0 + m0))
        rate_pred = float(np.sqrt(f_pred))       # per-step A-norm rate
        rate_wc = (kap - 1.0) / (kap + 1.0)
        _, rel2_g, relA_g, it_g = gd(Aop, bb, xst, maxiter=GD_CAP)
        rate_meas = tail_rate(relA_g, min(200, len(relA_g) - 2))
        conv = rel2_g[-1] <= 1e-10
        it_rep = it_g if conv else None
        it_extrap = (it_g if conv else
                     int(np.ceil(np.log(1e-10) / np.log(rate_meas))))
        case[tag] = {
            "cg_relerr_after_2": rel2_c[2], "cg_relres_after_2": res_c[2],
            "m0": float(m0), "rate_predicted_f(m0)": rate_pred,
            "rate_worst_case": float(rate_wc), "rate_measured": rate_meas,
            "gd_converged": bool(conv), "gd_iters_measured": it_rep,
            "gd_iters_to_1e-10": it_extrap,
            "gd_final_relerr": rel2_g[-1]}
        rank1_curves[(rho, tag)] = rel2_g
    ok(f"rho={rho:.0e}: CG converges in EXACTLY 2 iterations (rel err after 2 "
       f"= {case['random_b']['cg_relerr_after_2']:.1e} random b, "
       f"{case['worst_b']['cg_relerr_after_2']:.1e} worst b)",
       case["random_b"]["cg_relerr_after_2"] < 1e-7
       and case["worst_b"]["cg_relerr_after_2"] < 1e-7)
    logrel = lambda r_, p_: abs(np.log(r_) - np.log(p_)) / abs(np.log(p_))  # noqa: E731
    ok(f"rho={rho:.0e}: GD(worst b) rate {case['worst_b']['rate_measured']:.8f}"
       f" == (kappa-1)/(kappa+1) = {case['worst_b']['rate_worst_case']:.8f}",
       logrel(case["worst_b"]["rate_measured"],
              case["worst_b"]["rate_worst_case"]) < 0.02)
    ok(f"rho={rho:.0e}: GD(random b) rate {case['random_b']['rate_measured']:.8f}"
       f" == closed form f(m0)^(1/2) = {case['random_b']['rate_predicted_f(m0)']:.8f}"
       f" (worst case NOT attained for generic b)",
       logrel(case["random_b"]["rate_measured"],
              case["random_b"]["rate_predicted_f(m0)"]) < 0.02)
    partD["rank1"]["cases"][f"{rho:.0e}"] = case

its_w = [partD["rank1"]["cases"][f"{r:.0e}"]["worst_b"]["gd_iters_to_1e-10"]
         for r in RHOS]
ok(f"GD iteration count to 1e-10 GROWS with rho: {its_w} for rho = 1e2/1e4/1e6 "
   f"(worst-case rhs; CG stays at 2)", its_w[0] < its_w[1] < its_w[2])

# ---- 11. cluster spectrum ----------------------------------------------------
# 3 tight clusters; clustering -- not range -- is CG's currency
# (ties to report 06's NPO clustering and report 07's Nystrom failure).
rngq = np.random.default_rng(SEEDS["cluster_q"])
Q, _ = np.linalg.qr(rngq.standard_normal((dim, dim)))
centers = np.repeat([1.0, 1e3, 1e6], [300, 80, 20])
upert = np.random.default_rng(SEEDS["cluster_widths"]).uniform(-0.5, 0.5, dim)
b_cl = np.random.default_rng(SEEDS["cluster_b"]).standard_normal(dim)
WIDTHS = [1e-3, 1e-2, 1e-1]
partD["clusters"] = {"dim": dim, "sizes": [300, 80, 20],
                     "centers": [1.0, 1e3, 1e6], "widths": WIDTHS, "cases": {}}
cl_curves = {}
for wdt in WIDTHS:
    lam_cl = centers * (1.0 + wdt * upert)
    A_cl = (Q * lam_cl) @ Q.T
    A_cl = (A_cl + A_cl.T) / 2.0
    xst_cl = Q @ ((Q.T @ b_cl) / lam_cl)         # exact eigen-solve
    kap_cl = float(lam_cl.max() / lam_cl.min())
    _, res_cl, rel2_cl, relA_cl, _ = cg_err(A_cl, b_cl, xst_cl, maxiter=100)
    itc = iters_to(rel2_cl)
    partD["clusters"]["cases"][f"{wdt:.0e}"] = {
        "kappa": kap_cl, "cg_iters_err": itc,
        "cg_relerr_first_12": rel2_cl[: min(13, len(rel2_cl))],
        "cg_relAerr_first_12": relA_cl[: min(13, len(relA_cl))],
        "cg_relerr_at_3": rel2_cl[3] if len(rel2_cl) > 3 else None,
        "cg_relAerr_at_3": relA_cl[3] if len(relA_cl) > 3 else None}
    cl_curves[wdt] = rel2_cl
    if wdt == WIDTHS[0]:
        GD_CAP_CL = 30_000
        _, rel2_gcl, relA_gcl, _ = gd(A_cl, b_cl, xst_cl, maxiter=GD_CAP_CL)
        rate_gcl = tail_rate(relA_gcl, 5000)
        rate_wc_cl = (kap_cl - 1.0) / (kap_cl + 1.0)
        proj = int(np.ceil(np.log(1e-10) / np.log(rate_gcl)))
        partD["clusters"]["gd_w1e-3"] = {
            "cap": GD_CAP_CL, "relerr_at_cap": rel2_gcl[-1],
            "rate_measured_tail": rate_gcl, "rate_worst_case": float(rate_wc_cl),
            "projected_iters_to_1e-10": proj}
        ok(f"clusters w=1e-3: GD hopeless -- after {GD_CAP_CL} its rel err "
           f"still {rel2_gcl[-1]:.1e}; tail rate {rate_gcl:.8f} vs "
           f"(kappa-1)/(kappa+1) = {rate_wc_cl:.8f} (kappa = {kap_cl:.3e}), "
           f"projected ~{proj:.1e} its to 1e-10",
           rel2_gcl[-1] > 1e-4 and proj > 1_000_000)
cnt = [partD["clusters"]["cases"][f"{w_:.0e}"]["cg_iters_err"] for w_ in WIDTHS]
e3 = partD["clusters"]["cases"]["1e-03"]["cg_relerr_first_12"]
a3 = partD["clusters"]["cases"]["1e-03"]["cg_relAerr_first_12"]
dropmax = max(e3[k] / e3[k + 1] for k in range(len(e3) - 1))
# note: with nonzero widths "3 clusters => 3 iterations" is idealized; CG
# spends its first sweeps on the two heavy clusters (l2 error, dominated by
# the lambda~1 block, barely moves: e3[3] below), then plunges once the Ritz
# values cover all three -- the count and drops are reported as measured.
ok(f"clusters: CG needs {cnt[0]} its at width 1e-3 (l2 err after 3 its "
   f"{e3[3]:.1e}, A-norm err {a3[3]:.1e}; largest single-sweep drop "
   f"{dropmax:.0f}x), and the count grows only gently with width: "
   f"{dict(zip(['1e-3','1e-2','1e-1'], cnt))} while kappa stays ~1e6 -- "
   f"clustering, not range, is CG's currency",
   cnt[0] <= 25 and cnt[0] <= cnt[1] <= cnt[2] <= 120)
partD["clusters"]["counts_by_width"] = dict(zip(["1e-3", "1e-2", "1e-1"], cnt))
partD["clusters"]["largest_single_sweep_drop_w1e-3"] = float(dropmax)

# figure: rank-1 sweep + cluster demo
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
cols = {1e2: "tab:blue", 1e4: "tab:green", 1e6: "tab:red"}
for rho in RHOS:
    cur = rank1_curves[(rho, "worst_b")]
    axes[0].semilogy(cur[: min(len(cur), 20000)], color=cols[rho], lw=1.3,
                     label=f"GD, $\\rho$={rho:.0e} (worst b)")
    cur2 = rank1_curves[(rho, "random_b")]
    axes[0].semilogy(cur2, color=cols[rho], lw=1.0, ls="--",
                     label=f"GD, $\\rho$={rho:.0e} (random b)")
    c2 = partD["rank1"]["cases"][f"{rho:.0e}"]["worst_b"]["cg_relerr_after_2"]
    axes[0].semilogy([2], [max(c2, 1e-14)], "*", color=cols[rho], ms=13,
                     zorder=6, markeredgecolor="k", markeredgewidth=0.4)
axes[0].text(2.7, 1.5e-12, "CG: 2 iterations\n(any $\\rho$, either rhs)",
             fontsize=8, va="bottom")
axes[0].set_xscale("log")
axes[0].set_ylim(1e-15, 3)
axes[0].set_xlabel("iteration")
axes[0].set_ylabel(r"$\|x_k - x^*\|/\|x^*\|$")
axes[0].set_title(r"rank-1 coupling $A = I + \rho vv^T$ (1-factor model):"
                  "\nfree for CG, fatal for GD")
axes[0].legend(fontsize=7, loc="lower center", ncol=2, framealpha=0.95)
for wdt, c in zip(WIDTHS, ["tab:blue", "tab:green", "tab:red"]):
    axes[1].semilogy(cl_curves[wdt], "o-", color=c, ms=3, lw=1.2,
                     label=f"CG, width {wdt:.0e} "
                           f"[{partD['clusters']['cases'][f'{wdt:.0e}']['cg_iters_err']}]")
gcl = partD["clusters"]["gd_w1e-3"]
axes[1].semilogy(np.arange(0, len(rel2_gcl), 20), rel2_gcl[::20], "-",
                 color="0.4", lw=1.2,
                 label=f"GD, width 1e-3 (proj. {gcl['projected_iters_to_1e-10']:.0e} its)")
axes[1].set_xscale("log")
axes[1].set_ylim(1e-13, 3)
axes[1].set_xlabel("iteration")
axes[1].set_ylabel(r"$\|x_k - x^*\|/\|x^*\|$")
axes[1].set_title("3-cluster spectrum {1, 1e3, 1e6}, $\\kappa \\approx 10^6$:"
                  "\nclustering, not range, is CG's currency")
axes[1].legend(fontsize=8, loc="center right")
fig.tight_layout()
fig.savefig(FIGDIR / "decoupling_gd_vs_cg.png", dpi=150)
plt.close(fig)
RESULTS["figures"].append("decoupling_gd_vs_cg.png")

# ---- 12. CG iters ~ c sqrt(kappa) across the ladder --------------------------
# Residual-criterion counts from pcg (comparable to report 08's 116 anchor).
b_grf = grf_rhs(n)
_, res_grf = pcg(A, b_grf, M=None, tol=1e-10, maxiter=2000)
ok(f"anchor: unpreconditioned PCG on the report-03 GRF rhs takes "
   f"{len(res_grf)-1} iterations (report 08: 116)", len(res_grf) - 1 == 116)

# Chebyshev bound: ||e_k||_A <= eps ||e0||_A once k >= sqrt(kappa) ln(2/eps)/2.
CHEB = 0.5 * np.log(2.0 / 1e-10)
err_key = {"none": "cg", "adi": "adi_cg", "blockjacobi2": "blockjacobi2_cg",
           "twolevel_ic0": "twolevel_ic0_cg"}
n_at_1 = {"none": 0,     # unpreconditioned: no engineered cluster at 1
          "adi": int(np.sum(np.abs(w_adi - 1.0) < 1e-8)),
          "blockjacobi2": int(np.sum(np.abs(w_bj - 1.0) < 1e-8)),
          "twolevel_ic0": int(np.sum(np.abs(w_tl - 1.0) < 1e-8))}
fit_methods = {}
for name, M_, kap_ in [("none", None, kappa_A), ("adi", M_adi, kap_adi),
                       ("blockjacobi2", M_bj, kap_bj),
                       ("twolevel_ic0", M_tl, kap_tl)]:
    _, res_ = pcg(A, b_rod, M=M_, tol=1e-10, maxiter=2000)
    it_ = len(res_) - 1
    _, _, _, relA_, _ = cg_err(A, b_rod, xstar, M=M_, maxiter=2000, stop="Aerr")
    itA_ = iters_to(relA_)
    bound_ = float(np.sqrt(kap_) * CHEB)
    fit_methods[name] = {"iters_relres": it_, "iters_Anorm_err": itA_,
                         "iters_err": iters_to(LADDER[err_key[name]]),
                         "kappa": float(kap_), "sqrt_kappa": float(np.sqrt(kap_)),
                         "c": it_ / float(np.sqrt(kap_)),
                         "chebyshev_bound": bound_,
                         "bound_utilization": itA_ / bound_,
                         "n_eigs_at_1": n_at_1[name]}
its_v = np.array([fit_methods[m]["iters_relres"] for m in fit_methods])
itsA_v = np.array([fit_methods[m]["iters_Anorm_err"] for m in fit_methods])
sqk_v = np.array([fit_methods[m]["sqrt_kappa"] for m in fit_methods])
bnd_v = np.array([fit_methods[m]["chebyshev_bound"] for m in fit_methods])
c_fit = float((its_v * sqk_v).sum() / (sqk_v**2).sum())   # LS through origin
rel_pred_err = np.abs(c_fit * sqk_v - its_v) / its_v
order_match = bool(np.array_equal(np.argsort(its_v), np.argsort(sqk_v)))
partD["sqrt_kappa_fit"] = {
    "methods": fit_methods, "c_fit": c_fit, "cheb_ln_factor": float(CHEB),
    "pred_rel_err": dict(zip(fit_methods, rel_pred_err)),
    "c_range": [float(min(its_v / sqk_v)), float(max(its_v / sqk_v))],
    "ordering_match": order_match}
ok("sqrt(kappa) BOUNDS every ladder method (A-norm iters <= "
   "sqrt(kappa) ln(2/tol)/2): " +
   ", ".join(f"{m}: {fit_methods[m]['iters_Anorm_err']} <= "
             f"{fit_methods[m]['chebyshev_bound']:.0f}" for m in fit_methods),
   bool(np.all(itsA_v <= np.ceil(bnd_v))))
ok(f"...but sqrt(kappa) alone does NOT predict the fine ordering "
   f"(ordering_match = {order_match}): c = iters/sqrt(kappa) spans "
   f"[{min(its_v/sqk_v):.2f}, {max(its_v/sqk_v):.2f}]; ADI (flat spectrum, "
   f"{fit_methods['adi']['n_eigs_at_1']} eigenvalues at 1) uses "
   f"{100*fit_methods['adi']['bound_utilization']:.0f}% of its bound while "
   f"blockJacobi2 ({fit_methods['blockjacobi2']['n_eigs_at_1']}/1024 "
   f"eigenvalues exactly at 1) beats its larger kappa "
   f"({fit_methods['blockjacobi2']['kappa']:.1f} vs "
   f"{fit_methods['adi']['kappa']:.1f}) with "
   f"{fit_methods['blockjacobi2']['iters_relres']} vs "
   f"{fit_methods['adi']['iters_relres']} its -- spectral shape (clustering, "
   f"section 11) decides within the bound",
   fit_methods["blockjacobi2"]["iters_relres"]
   < fit_methods["adi"]["iters_relres"]
   and fit_methods["blockjacobi2"]["kappa"] > fit_methods["adi"]["kappa"]
   and fit_methods["adi"]["bound_utilization"] > 0.5)
info(f"single-c fit (LS through origin): c = {c_fit:.2f}, max prediction "
     f"error {100*float(rel_pred_err.max()):.0f}% -- reported for "
     "completeness; the bound + clustering story above is the honest one")

# ---- 13. CG as an on-the-fly decoupler: A-orthogonal directions --------------
_, _, _, _, P30 = cg_err(A, b_rod, xstar, M=None, maxiter=200, record_dirs=30)
G30 = P30.T @ (A @ P30)
dnorm = np.sqrt(np.diag(G30))
Gn = np.abs(G30) / np.outer(dnorm, dnorm)
np.fill_diagonal(Gn, 0.0)
conj30 = float(Gn.max())
conj10 = float(Gn[:10, :10].max())
partD["conjugacy"] = {"max_normalized_pAp_first10": conj10,
                      "max_normalized_pAp_first30": conj30}
ok(f"CG's search directions are A-orthogonal (sequential Gram-Schmidt in the "
   f"A-inner product = 09 SS5's sequential regression): max |p_i'Ap_j| / "
   f"(|p_i|_A |p_j|_A) = {conj10:.1e} over the first 10, {conj30:.1e} over "
   f"the first 30 directions (finite-precision loss grows with depth)",
   conj30 < 1e-6)
RESULTS["part_d"] = partD
info(f"Part D done in {time.time()-t0:.1f}s")

# ===========================================================================
# remaining figures
# ===========================================================================
fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.semilogy(np.sort(w_dense) / w_dense.max(), lw=1.5,
            label=f"eig(A)/$\\lambda_{{max}}$   ($\\kappa$ = {kappa_A:.1f})")
ax.semilogy(w_adi / w_adi.max(), lw=1.5,
            label=f"eig($M^{{-1}}A$)/max, ADI   ($\\kappa$ = {kap_adi:.2f})")
ax.set_xlabel("eigenvalue index (sorted)")
ax.set_ylabel("eigenvalue / max (log)")
ax.set_title("ADI compresses the spectrum: "
             f"$\\kappa$ {kappa_A:.0f} $\\to$ {kap_adi:.1f} "
             f"$\\approx$ {kap_adi/np.sqrt(kappa_A):.2f}$\\sqrt{{\\kappa(A)}}$")
ax.legend()
fig.tight_layout()
fig.savefig(FIGDIR / "decoupling_adi_spectrum.png", dpi=150)
plt.close(fig)
RESULTS["figures"].append("decoupling_adi_spectrum.png")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
im0 = axes[0].imshow(S1, cmap="viridis")
axes[0].set_title("1-D $A^{-1}$ = Brownian-bridge covariance\n"
                  "$h\\,(\\min(s,t) - st)$")
fig.colorbar(im0, ax=axes[0], fraction=0.046)
dev1 = np.abs(S1 - G1)
with np.errstate(divide="ignore"):
    im1 = axes[1].imshow(np.log10(dev1 + 1e-20), cmap="magma")
axes[1].set_title("$\\log_{10}|A^{-1} - uv^T|$, $u_i = h x_i$, $v_j = 1-x_j$:\n"
                  "upper triangle is EXACTLY rank-1 (semiseparable)")
fig.colorbar(im1, ax=axes[1], fraction=0.046)
axes[2].semilogy(np.arange(1, 31), sv_far[:30] / sv_far[0], "o-", ms=4)
axes[2].axhline(1e-8, color="0.6", ls=":", label="1e-8 cutoff")
axes[2].axvline(numrank_far, color="tab:red", ls="--",
                label=f"numerical rank {numrank_far}")
axes[2].set_xlabel("singular value index")
axes[2].set_ylabel("$\\sigma_k/\\sigma_1$")
axes[2].set_title("2-D $A^{-1}$: far block (rows 0-7 vs 24-31,\n"
                  "256x256) is numerically low-rank (H-matrix)")
axes[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "decoupling_semiseparable.png", dpi=150)
plt.close(fig)
RESULTS["figures"].append("decoupling_semiseparable.png")

# ===========================================================================
# deviations, JSON, summary
# ===========================================================================
RESULTS["deviations"] = [
    "GD runs that would exceed the runtime budget (rank-1 rho=1e6 and the "
    f"cluster problem, both kappa ~ 1e6) were capped ({GD_CAP} / 30000 "
    "iterations); their iterations-to-1e-10 are extrapolated from the "
    "measured tail rate (for the rank-1 case the per-step rate is exactly "
    "constant, so the extrapolation is exact up to the last digit of the rate).",
    "Rank-1 refinement: for a RANDOM rhs the (kappa-1)/(kappa+1) GD rate is "
    "provably not attained -- the per-step A-norm contraction is the exact "
    "closed form sqrt(f(m0)), f(m) = (k-1)^2 m/((1+k^2 m)(1+m)) (derivation "
    "in comments, machine-verified). The worst-case rhs b = v + w_perp, "
    "which attains (kappa-1)/(kappa+1) exactly, was added; both are reported.",
    "Ladder iteration counts are ERROR-based (||x_k - x*||/||x*|| <= 1e-10); "
    "section 12's sqrt(kappa) fit uses pcg's residual-based counts to stay "
    "comparable with report 08's 116-iteration anchor. Both are in the JSON.",
    "Error histories come from cg_err, a re-implementation of pcg.pcg's "
    "recurrence (verified to reproduce its residual history to machine "
    "precision in section 9) rather than a callback, so the reference "
    "iteration is untouched.",
    "Section 11: at width 1e-3 CG needs ~"
    f"{partD['clusters']['cases']['1e-03']['cg_iters_err']} iterations, not "
    "literally 3: three clusters of nonzero width are not three points; the "
    "count and per-sweep drops are reported as measured.",
    "The hot/cold-rod rhs excites no even eigenmodes (odd parity), so GD "
    "rate checks on the rod problem use the parity-effective kappa "
    f"({RESULTS['part_a']['jacobi_gd']['kappa_eff_rod']:.2f}); the "
    "full-kappa rate 0.995472 = cos(pi h) is verified on the GRF rhs.",
    "Section 12 reframed after measurement: sqrt(kappa) is verified as the "
    "Chebyshev BOUND on iterations, but it does not predict the fine "
    "ordering -- blockJacobi2 (960 eigenvalues exactly at 1) beats ADI "
    "despite a 1.8x larger kappa. A single-c fit is reported for "
    "completeness with its (poor) max prediction error.",
]
RESULTS["meta"]["runtime_seconds"] = round(time.time() - T_START, 1)
RESULTS["meta"]["n_checks"] = len(RESULTS["checks"])
RESULTS["meta"]["n_fail"] = N_FAIL

with open(RESDIR / "decoupling.json", "w") as fh:
    json.dump(jsonable(RESULTS), fh, indent=1)

print(f"== summary: {len(RESULTS['checks'])} checks, {N_FAIL} FAIL; "
      f"runtime {RESULTS['meta']['runtime_seconds']}s ==")
print("figures:", ", ".join(RESULTS["figures"]))
print("results: results/decoupling.json")
