"""Every preconditioner is an autoregressive predictor; every solver is Richardson.

THE unified experiment (reports 09/10/11 notation): h = 1/(n+1), A =
Kronecker-sum Laplacian / h^2 (poisson_2d, N = n^2, lexicographic k = i*n+j),
B = I - diag(A)^{-1} A the two-sided regression matrix = Jacobi iteration
matrix with rho(B) = cos(pi h), Sigma = A^{-1}, phiL2R/phiR2L the sequential
regressions, reversal identity chol(A^{-1}) = P L^{-T} P.

Every method below is run as the SAME stationary preconditioned Richardson
("residual") iteration

    x_{k+1} = x_k + C r_k,      r_k = b - A x_k,

with C a genuinely linear operator (NO conjugate gradients anywhere), and for
each method the exact convergence factor rho = spectral radius of the
error-propagation matrix E = I - C A (dense at N=1024) is compared with the
measured asymptotic slope of ||e_k||, e_k = x* - x_k.

The predictor ladder:
  1. optimally damped Richardson  C = alpha* I, alpha* = 2/(lam_min+lam_max)
  2. Jacobi C = D^{-1}: the two-sided predictor applied synchronously (stale
     neighbors); pretty identity alpha_Jacobi = h^2/4 = alpha* exactly, so
     undamped Jacobi IS optimal Richardson and rho(B) = cos(pi h) = (k-1)/(k+1)
  3. Gauss-Seidel C = tril(A)^{-1}: same weights applied sequentially with
     fresh values (noise-free systematic-scan Gibbs sweep); rho = cos^2(pi h)
  4. PERFECT AR predictor: Phi (strictly lower prediction weights) and
     innovation variances d2 built two independent ways; M = (I-Phi)^T
     diag(1/d2) (I-Phi) == A; Richardson with C = M^{-1} converges in ONE step
  5. truncated AR = IC(0) (reused from preconditioners.ic0, reports 09/10/11)
  6. Galerkin coarse-only, Z = (a) 4x4 block averages (64 dofs) and
     (b) bilinear interpolation from the 16x16 coarse grid: one-shot
     A-orthogonal projection, then STALLS at the plateau
  7. two-level multiplicative (two-grid): damped Jacobi (omega = 4/5) 1 pre +
     1 post around the Galerkin coarse correction, both coarse spaces
  8. two-level additive (blocks space), C = theta (omega D^{-1} + Z Ac^{-1} Z^T)
  9. mesh (in)dependence n in {16, 32, 64}
 10. weights anatomy of the perfect predictor at the central node (16,16),
     innovation variances, and the Schur-complement (discrete DtN) identity.

Coarse-grid choice (justification): for n = 32, n+1 = 33 is odd so perfect
2h-nesting is impossible. We take the standard vertex-centered coarsening
n_c = n/2 = 16: coarse node j at fine node 2j (1-based), uniform spacing
h_c = 2h, prolongation = 1-D linear interpolation stencil [1/2, 1, 1/2]
(tensorized to bilinear), and Galerkin A_c = Z^T A Z absorbs the one-sided
distance-h gap at the far wall. 16x16 (not 15x15) keeps n_c = n/2 uniform
across the n in {16,32,64} mesh study and leaves every fine node within h of
coarse support.

omega = 4/5 (justification): classical optimal damping for the 2-D 5-point
Laplacian smoother -- it minimizes the high-frequency smoothing factor of
damped Jacobi at mu = 3/5 (Briggs-Henson-McCormick, "A Multigrid Tutorial",
2nd ed., sec. 2; Trottenberg et al., "Multigrid", sec. 2.1).

Run from the repo root:
    uv run python python/experiments/richardson_ar.py

PASS/FAIL style follows verify_statistical_identities.py. Figures ->
figures/ (dpi=150), numbers -> results/richardson_ar.json.
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

from poisson import poisson_2d, grf_rhs
from preconditioners import block_average_matrix, ic0

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"
RESDIR = ROOT / "results"
np.set_printoptions(precision=4, suppress=True)

RESULTS = {}
PASS_LINES = []
N_FAIL = 0
T_START = time.time()


def ok(name, cond):
    global N_FAIL
    if not cond:
        N_FAIL += 1
    line = f"{'PASS' if cond else 'FAIL'}: {name}"
    print(line)
    PASS_LINES.append(line)
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
# shared machinery
# ---------------------------------------------------------------------------
TOL = 1e-10
MAXITER = 20000


def run_richardson(A, b, apply_C, xstar, tol=TOL, maxiter=MAXITER):
    """Stationary preconditioned Richardson x <- x + C (b - A x), x0 = 0.

    Tracks the ERROR e_k = x* - x_k in both the 2-norm and the A-norm,
    relative to e_0 = x*. Returns (x, hist_l2, hist_A, it_converged)."""
    x = np.zeros_like(b)
    e0 = xstar - x
    e0_l2 = np.linalg.norm(e0)
    e0_A = np.sqrt(e0 @ (A @ e0))
    hist_l2 = [1.0]
    hist_A = [1.0]
    it_conv = None
    for k in range(1, maxiter + 1):
        r = b - A @ x
        x = x + apply_C(r)
        e = xstar - x
        rel = np.linalg.norm(e) / e0_l2
        hist_l2.append(rel)
        hist_A.append(np.sqrt(max(e @ (A @ e), 0.0)) / e0_A)
        if rel <= tol:
            it_conv = k
            break
    return x, np.array(hist_l2), np.array(hist_A), it_conv


def tail_slope(hist, wmax=100):
    """Asymptotic geometric slope from the tail of an error history."""
    h = np.asarray(hist)
    k_end = len(h) - 1
    w = max(3, min(wmax, k_end // 2))
    return (h[k_end] / h[k_end - w]) ** (1.0 / w)


def rho_dense(E):
    return float(np.max(np.abs(np.linalg.eigvals(E))))


def seeded_slope(A, apply_C, E_dense, nsteps=30, wtail=10):
    """Asymptotic slope measured by iterating the ACTUAL operator (via apply_C)
    on the dominant eigenvector of the dense error-propagation matrix E.

    A generic x0 = 0 start can terminate before the dominant mode wins (the
    two-grid transient); seeding with the dominant mode measures the true
    asymptotic factor of the implemented iteration in a few steps."""
    wv, Vv = np.linalg.eig(E_dense)
    v = np.real(Vv[:, np.argmax(np.abs(wv))])
    v = v / np.linalg.norm(v)
    e = v.copy()
    norms = [1.0]
    for _ in range(nsteps):
        e = e - apply_C(A @ e)
        norms.append(np.linalg.norm(e))
    norms = np.array(norms)
    return float((norms[-1] / norms[-1 - wtail]) ** (1.0 / wtail))


def interp_1d(n):
    """1-D linear interpolation n x (n/2): coarse node j at fine node 2j+1
    (0-based), stencil [1/2, 1, 1/2], Dirichlet walls absorb the missing
    half-weights at the domain ends."""
    nc = n // 2
    P = np.zeros((n, nc))
    for j in range(nc):
        f = 2 * j + 1
        P[f, j] = 1.0
        P[f - 1, j] += 0.5
        if f + 1 < n:
            P[f + 1, j] += 0.5
    return P


def make_tri_solver(mat_csc):
    """splu of an (already triangular) sparse matrix with natural ordering:
    the LU factorization is the matrix itself; .solve is an exact sparse
    triangular solve (trans='T' gives the transposed solve)."""
    return spla.splu(mat_csc, permc_spec="NATURAL", diag_pivot_thresh=0.0)


def make_coarse(A, Z):
    """Galerkin coarse operator machinery: returns (apply_coarse, Ac)."""
    if sp.issparse(Z):
        Ac = np.asarray((Z.T @ (A @ Z)).todense())
    else:
        Ac = Z.T @ (A @ Z)
    cho = sla.cho_factor(Ac)

    def apply_coarse(r):
        return Z @ sla.cho_solve(cho, Z.T @ r)

    return apply_coarse, Ac, cho


def make_twogrid(A, Z, omega):
    """One multiplicative two-grid cycle applied to r (as error correction):
    pre-smooth (damped Jacobi, nu=1), Galerkin coarse correction, post-smooth.
    Error propagation E = S (I - Z Ac^{-1} Z^T A) S, S = I - omega D^{-1} A."""
    inv_diag = 1.0 / A.diagonal()
    apply_coarse, Ac, cho = make_coarse(A, Z)

    def apply_C(r):
        e = omega * inv_diag * r                # pre-smooth from zero guess
        e = e + apply_coarse(r - A @ e)         # coarse correction
        e = e + omega * inv_diag * (r - A @ e)  # post-smooth
        return e

    return apply_C, Ac, cho


# ===========================================================================
# 0. canonical problem
# ===========================================================================
print("== 0. canonical problem: n=32, A=poisson_2d(32), b=grf_rhs(32) (seed 42) ==")
n = 32
N = n * n
h = 1.0 / (n + 1)
A = poisson_2d(n)
Ad = A.toarray()
b = grf_rhs(n)  # alpha=2, tau=3, seed=42 defaults -- the canonical RHS
xstar = spla.spsolve(A.tocsc(), b)
e0 = xstar.copy()  # x0 = 0
e0_l2 = np.linalg.norm(e0)
e0_A = np.sqrt(e0 @ (A @ e0))

wA = np.linalg.eigvalsh(Ad)
lam_min_exact = 8 * np.sin(np.pi * h / 2) ** 2 / h**2
lam_max_exact = 8 * np.cos(np.pi * h / 2) ** 2 / h**2
kappa = wA[-1] / wA[0]
ok("dense eigh(A) matches analytic lam_min = 8 sin^2(pi h/2)/h^2, lam_max = 8 cos^2(pi h/2)/h^2",
   np.isclose(wA[0], lam_min_exact, rtol=1e-11) and np.isclose(wA[-1], lam_max_exact, rtol=1e-11))
ok(f"kappa(A) = {kappa:.2f} matches report 11's 440.69 at n=32", abs(kappa - 440.69) < 0.01)
ok("lam_min + lam_max == 8/h^2 exactly (Kronecker-sum spectrum pairs up)",
   np.isclose(lam_min_exact + lam_max_exact, 8 / h**2, rtol=1e-13))
info(f"h=1/33, N={N}, lam_min={wA[0]:.4f}, lam_max={wA[-1]:.4f}, "
     f"kappa={kappa:.2f}, ||e0||={e0_l2:.4f}, ||e0||_A={e0_A:.4f}")
RESULTS["meta"] = {
    "n": n, "N": N, "h": h, "rhs": "grf_rhs(32), alpha=2, tau=3, seed=42",
    "tol_rel_l2": TOL, "maxiter": MAXITER,
    "lam_min": wA[0], "lam_max": wA[-1], "kappa_A": kappa,
    "e0_l2": e0_l2, "e0_A": e0_A,
}

LADDER = {}      # method -> record dict
HIST = {}        # method -> (hist_l2, hist_A)

# ===========================================================================
# 1. no-predictor baseline: optimally damped Richardson
# ===========================================================================
print("== 1. optimally damped Richardson: C = alpha* I ==")
alpha_star = 2.0 / (lam_min_exact + lam_max_exact)   # = 2/(8/h^2) = h^2/4
rho_rich_theory = (kappa - 1) / (kappa + 1)
rho_rich = float(np.max(np.abs(1.0 - alpha_star * wA)))  # E = I - alpha A symmetric
x_r, hl, ha, it_r = run_richardson(A, b, lambda r: alpha_star * r, xstar)
sl = tail_slope(hl)
HIST["richardson_opt"] = (hl, ha)
ok(f"Richardson(alpha*) converged to rel l2 err <= 1e-10 in {it_r} iterations", it_r is not None)
ok(f"rho exact {rho_rich:.6f} == (kappa-1)/(kappa+1) = {rho_rich_theory:.6f}",
   np.isclose(rho_rich, rho_rich_theory, rtol=1e-12))
ok(f"measured asymptotic slope {sl:.6f} matches rho {rho_rich:.6f} to 1%",
   abs(sl - rho_rich) / rho_rich < 0.01)
LADDER["richardson_opt"] = {"rho_exact": rho_rich, "slope_measured": float(sl),
                            "iters_to_tol": it_r, "alpha_star": alpha_star}

# ===========================================================================
# 2. Jacobi: the perfect two-sided predictor applied synchronously
# ===========================================================================
print("== 2. Jacobi: C = D^{-1}, one synchronous sweep of conditional means ==")
inv_diag = 1.0 / A.diagonal()
Bmat = np.eye(N) - inv_diag[:, None] * Ad          # B = I - D^{-1} A
rho_B = float(np.max(np.abs(np.linalg.eigvalsh(Bmat))))  # B symmetric (const diag)
x_j, hl, ha, it_j = run_richardson(A, b, lambda r: inv_diag * r, xstar)
sl_j = tail_slope(hl)
HIST["jacobi"] = (hl, ha)
E_jac = np.eye(N) - inv_diag[:, None] * Ad
ok("Jacobi error-propagation matrix E = I - D^{-1}A is exactly B (two-sided regression matrix)",
   np.array_equal(E_jac, Bmat))
ok(f"pretty identity: alpha_Jacobi = 1/A_ii = h^2/4 = {h*h/4:.6e} == alpha* = 2/(lam_min+lam_max)",
   np.allclose(inv_diag, alpha_star, rtol=1e-13) and np.isclose(h * h / 4, alpha_star, rtol=1e-13))
ok(f"three numbers coincide: rho(B) = {rho_B:.8f}, cos(pi h) = {np.cos(np.pi*h):.8f}, "
   f"(kappa-1)/(kappa+1) = {rho_rich_theory:.8f}",
   np.isclose(rho_B, np.cos(np.pi * h), rtol=1e-10)
   and np.isclose(rho_B, rho_rich_theory, rtol=1e-10))
ok("undamped Jacobi IS optimally damped Richardson: identical error histories (rtol 1e-10)",
   it_j == it_r and np.allclose(hl, HIST["richardson_opt"][0], rtol=1e-10))
ok(f"measured slope {sl_j:.6f} matches rho(B) {rho_B:.6f} to 1%",
   abs(sl_j - rho_B) / rho_B < 0.01)
LADDER["jacobi"] = {"rho_exact": rho_B, "slope_measured": float(sl_j), "iters_to_tol": it_j,
                    "cos_pi_h": float(np.cos(np.pi * h))}

# ===========================================================================
# 3. Gauss-Seidel: same weights, sequential, fresh values
# ===========================================================================
print("== 3. Gauss-Seidel: C = tril(A)^{-1}, systematic-scan Gibbs sweep ==")
tril_lu = make_tri_solver(sp.tril(A, format="csc"))
apply_gs = lambda r: tril_lu.solve(r)  # noqa: E731
# one GS sweep == sequential conditional-mean updates (noise-free Gibbs scan)
rng = np.random.default_rng(0)
x_test = rng.standard_normal(N)
x_sweep = x_test.copy()
for i in range(N):
    row = Ad[i]
    x_sweep[i] = (b[i] - row[:i] @ x_sweep[:i] - row[i + 1:] @ x_test[i + 1:]) / row[i]
x_gs_form = x_test + apply_gs(b - A @ x_test)
ok("one GS sweep == node-by-node replacement by conditional mean given fresh W/N + stale E/S "
   "neighbors (systematic-scan Gibbs, noise-free)",
   np.allclose(x_sweep, x_gs_form, rtol=1e-12, atol=1e-14))
E_gs = np.eye(N) - sla.solve_triangular(np.tril(Ad), Ad, lower=True)
rho_gs = rho_dense(E_gs)
x_g, hl, ha, it_g = run_richardson(A, b, apply_gs, xstar)
sl_g = tail_slope(hl)
HIST["gauss_seidel"] = (hl, ha)
ok(f"rho_GS = {rho_gs:.8f} == cos^2(pi h) = {np.cos(np.pi*h)**2:.8f} (Young: consistently ordered)",
   np.isclose(rho_gs, np.cos(np.pi * h) ** 2, rtol=1e-6))
ok(f"GS converged in {it_g} iterations; measured slope {sl_g:.6f} matches rho to 1%",
   it_g is not None and abs(sl_g - rho_gs) / rho_gs < 0.01)
LADDER["gauss_seidel"] = {"rho_exact": rho_gs, "slope_measured": float(sl_g),
                          "iters_to_tol": it_g, "cos2_pi_h": float(np.cos(np.pi * h) ** 2)}

# ===========================================================================
# 4. THE PERFECT AR PREDICTOR
# ===========================================================================
print("== 4. perfect AR predictor: Phi, d2, M = (I-Phi)^T diag(1/d2) (I-Phi) ==")
t0 = time.time()
P_rev = np.eye(N)[::-1]
ok("180-degree rotation is an automorphism: P A P == A (reversal machinery applies)",
   np.allclose(P_rev @ Ad @ P_rev, Ad))

# route (i): from L = chol(A) via the phiR2L / reversal machinery.
# Whitening z = L^T u regresses u_i on SUCCESSORS with coefficients
# -L[j,i]/L[i,i] (phiR2L, W matrix); the mirror identity phiL2R = P phiR2L P
# (valid because P A P = A) turns them into predecessor weights, and the
# innovation variances reverse with them: d2[i] = 1/L[N-1-i,N-1-i]^2.
L = np.linalg.cholesky(Ad)
Lnorm = L / np.diag(L)[None, :]
W_r2l = np.eye(N) - Lnorm.T                       # strictly upper successor weights
Phi = W_r2l[::-1, ::-1].copy()                    # P W P: strictly lower
d2 = (1.0 / np.diag(L) ** 2)[::-1].copy()

# route (ii): directly from the modified Cholesky (LDL^T) of Sigma = A^{-1}:
# Sigma = L_S diag(d2) L_S^T with L_S unit lower  =>  (I-Phi) = L_S^{-1}.
Sigma = np.linalg.inv(Ad)
C_S = np.linalg.cholesky(Sigma)
dgS = np.diag(C_S)
L_S = C_S / dgS[None, :]
d2_ii = dgS**2
T_ii = sla.solve_triangular(L_S, np.eye(N), lower=True, unit_diagonal=True)
Phi_ii = np.eye(N) - T_ii
dev_phi = float(np.max(np.abs(Phi - Phi_ii)))
dev_d2 = float(np.max(np.abs(d2 - d2_ii) / d2))
ok(f"two constructions of Phi agree: chol(A)+reversal vs modified-chol(Sigma), "
   f"max |dev| = {dev_phi:.2e}", dev_phi < 1e-8)
ok(f"two constructions of d2 agree (max rel dev = {dev_d2:.2e})", dev_d2 < 1e-8)

# explicit least-squares spot checks on Sigma submatrices at 3 nodes
k_center = 16 * n + 16          # interior node (16,16)
k_edge = 16                     # edge node (0,16): only same-row predecessors
sample_checks = {}
for label, kk in [("interior_(16,16)", k_center), ("edge_(0,16)", k_edge),
                  ("last_node_(31,31)", N - 1)]:
    w_ls = np.linalg.solve(Sigma[:kk, :kk], Sigma[:kk, kk])
    d2_ls = Sigma[kk, kk] - Sigma[kk, :kk] @ w_ls
    dev_w = float(np.max(np.abs(Phi[kk, :kk] - w_ls)))
    dev_v = float(abs(d2[kk] - d2_ls) / d2[kk])
    sample_checks[label] = {"max_weight_dev": dev_w, "rel_d2_dev": dev_v,
                            "d2": float(d2[kk])}
    ok(f"explicit LS regression at {label}: weights match Phi row to ~1e-10 "
       f"(max dev {dev_w:.2e}), d2 matches (rel dev {dev_v:.2e})",
       dev_w < 1e-9 and dev_v < 1e-9)
# the last node's predecessors are ALL other nodes -> its one-sided regression
# IS the two-sided stencil regression: weights 1/4 on its 2 neighbors, d2 = h^2/4
w_last = Phi[N - 1, :]
nb_last = [N - 2, N - 1 - n]
mask = np.ones(N, bool)
mask[nb_last] = False
mask[N - 1] = False
ok("last node (31,31): one-sided == full conditional: weights exactly 1/4 on its 2 stencil "
   f"neighbors (max |other| = {np.max(np.abs(w_last[mask])):.1e}), d2 = h^2/4 "
   f"(rel dev {abs(d2[N-1] - h*h/4)/(h*h/4):.1e})",
   np.allclose(w_last[nb_last], 0.25, atol=1e-10)
   and np.max(np.abs(w_last[mask])) < 1e-10
   and np.isclose(d2[N - 1], h * h / 4, rtol=1e-10))

T_perf = np.eye(N) - Phi                       # unit lower triangular
M_perf = T_perf.T @ (T_perf / d2[:, None])     # (I-Phi)^T diag(1/d2) (I-Phi)
relF = np.linalg.norm(M_perf - Ad) / np.linalg.norm(Ad)
ok(f"perfect predictor's normal equations ARE the operator: ||M - A||_F/||A||_F = {relF:.2e} "
   "< 1e-10", relF < 1e-10)


def apply_perfect(r):
    """C = M^{-1} via two unit-triangular solves: M^{-1} = T^{-1} D2 T^{-T}."""
    y = sla.solve_triangular(T_perf.T, r, lower=False, unit_diagonal=True)
    return sla.solve_triangular(T_perf, d2 * y, lower=True, unit_diagonal=True)


x_p, hl, ha, it_p = run_richardson(A, b, apply_perfect, xstar)
HIST["perfect_ar"] = (hl, ha)
Minv_perf = sla.solve_triangular(T_perf, (d2[:, None] * sla.solve_triangular(
    T_perf, np.eye(N), lower=True, unit_diagonal=True).T), lower=True, unit_diagonal=True)
E_perf = np.eye(N) - Minv_perf @ Ad
rho_perf = rho_dense(E_perf)
ok(f"perfect predictor == perfect preconditioner: ||e_1||/||e_0|| = {hl[1]:.2e} < 1e-12 "
   "in ONE Richardson iteration", it_p == 1 and hl[1] < 1e-12)
info(f"rho(I - M^{{-1}}A) = {rho_perf:.2e} (zero up to roundoff); "
     f"section time {time.time()-t0:.1f}s")
LADDER["perfect_ar"] = {"rho_exact": rho_perf, "slope_measured": None, "iters_to_tol": it_p,
                        "one_step_rel_err_l2": float(hl[1]),
                        "one_step_rel_err_A": float(ha[1]),
                        "M_vs_A_relF": relF, "phi_construction_dev": dev_phi,
                        "d2_construction_dev": dev_d2, "sample_node_checks": sample_checks}

# ===========================================================================
# 5. truncated AR = IC(0)
# ===========================================================================
print("== 5. truncated AR: IC(0) (Vecchia on the stencil pattern, report 11) ==")
Lic = ic0(Ad)                                  # reused, not re-derived
Lic_sp = sp.csc_matrix(Lic)
lu_ic = make_tri_solver(Lic_sp)
apply_ic = lambda r: lu_ic.solve(lu_ic.solve(r, trans="N"), trans="T")  # noqa: E731
Licinv = sla.solve_triangular(Lic, np.eye(N), lower=True)
G_ic = Licinv @ Ad @ Licinv.T                  # similar to M^{-1}A, symmetric
w_ic = np.linalg.eigvalsh(G_ic)
rho_ic = float(max(abs(1.0 - w_ic[0]), abs(1.0 - w_ic[-1])))
x_i, hl, ha, it_i = run_richardson(A, b, apply_ic, xstar)
sl_i = tail_slope(hl)
HIST["ic0"] = (hl, ha)
ratios = hl[1:] / hl[:-1]
transition_iter = int(np.argmax(ratios >= 0.99 * sl_i))
early_l2 = float((hl[10] / hl[0]) ** 0.1)
early_A = float((ha[10] / ha[0]) ** 0.1)
exp_ratio = float(np.log(early_l2) / np.log(sl_i))
ok(f"IC(0) Richardson converged in {it_i} iterations; rho exact = {rho_ic:.6f} "
   f"(spec(M^-1 A) = [{w_ic[0]:.4f}, {w_ic[-1]:.4f}])", it_i is not None and rho_ic < 1)
ok(f"measured terminal slope {sl_i:.6f} matches rho {rho_ic:.6f} to 1%",
   abs(sl_i - rho_ic) / rho_ic < 0.01)
ok(f"fast initial rough-error kill then slow geometric tail: first-10-iteration slope "
   f"{early_l2:.4f} (l2) / {early_A:.4f} (A-norm) vs terminal {sl_i:.4f} = {exp_ratio:.1f}x "
   f"faster in rate exponent; slope reaches 99% of terminal at iteration {transition_iter}",
   exp_ratio > 2.0 and transition_iter > 3)
info(f"kappa(M_ic^-1 A) = {w_ic[-1]/w_ic[0]:.2f}; slow tail governed by the smoothest mode: "
     f"1 - lam_min(M^-1 A) = {1-w_ic[0]:.6f}")
# residual-vs-error disparity (report section 1's vignette, measured): the
# surviving smooth error is under-reported by the residual, increasingly so
# as the error smooths (worst case lam_max/lam_min in amplitude).
x_re = np.zeros(N)
resid_err = {}
for k in range(1, 41):
    x_re = x_re + apply_ic(b - A @ x_re)
    if k in (20, 40):
        resid_err[k] = {
            "rel_err_l2": float(np.linalg.norm(xstar - x_re) / e0_l2),
            "rel_resid_l2": float(np.linalg.norm(b - A @ x_re) / np.linalg.norm(b)),
        }
ratio20 = resid_err[20]["rel_err_l2"] / resid_err[20]["rel_resid_l2"]
ratio40 = resid_err[40]["rel_err_l2"] / resid_err[40]["rel_resid_l2"]
ok(f"IC(0)'s residual under-reports its smooth surviving error, increasingly: rel err "
   f"{resid_err[20]['rel_err_l2']:.4f} vs rel resid {resid_err[20]['rel_resid_l2']:.4f} "
   f"({ratio20:.2f}x) at iter 20 -> {ratio40:.2f}x at iter 40 "
   f"(worst case lam_max/lam_min = {kappa:.0f}x in amplitude)",
   ratio20 > 1.5 and ratio40 > 1.2 * ratio20)
LADDER["ic0"] = {"rho_exact": rho_ic, "slope_measured": float(sl_i), "iters_to_tol": it_i,
                 "resid_vs_err": resid_err,
                 "resid_underreport_ratio_iter20": float(ratio20),
                 "resid_underreport_ratio_iter40": float(ratio40),
                 "lam_min_MinvA": float(w_ic[0]), "lam_max_MinvA": float(w_ic[-1]),
                 "kappa_MinvA": float(w_ic[-1] / w_ic[0]),
                 "transition_iter_99pct_terminal": transition_iter,
                 "terminal_slope": float(sl_i), "first_ratio": float(ratios[0]),
                 "early_slope_l2_first10": early_l2, "early_slope_A_first10": early_A,
                 "early_vs_terminal_rate_exponent": exp_ratio}

# ===========================================================================
# 6. Galerkin coarse-only: one-shot projection, then stall
# ===========================================================================
print("== 6. Galerkin coarse-only: (a) 4x4 block averages, (b) bilinear 16x16 ==")
Z_blk = block_average_matrix(n, 4)             # 1024 x 64, reused from report 11
P1 = interp_1d(n)
Z_bil = np.kron(P1, P1)                        # 1024 x 256, bilinear from 16x16
COARSE_MAXIT = 15  # deviation from maxiter=20000: iteration is exactly stationary
                   # after step 1 (verified below); running 20000 stalled steps
                   # adds nothing but time.
for tag, Z in [("coarse_only_blocks", Z_blk), ("coarse_only_bilinear", Z_bil)]:
    apply_co, Ac_co, _ = make_coarse(A, Z)
    x_c, hl, ha, it_c = run_richardson(A, b, apply_co, xstar, maxiter=COARSE_MAXIT)
    HIST[tag] = (hl, ha)
    E_co = np.eye(N) - Z @ np.linalg.solve(Ac_co, Z.T @ Ad)
    rho_co = rho_dense(E_co)
    # A-orthogonal projection: idempotent, e_1 A-orthogonal to range(Z), Pythagoras
    idem = np.max(np.abs(E_co @ E_co - E_co))
    e1 = e0 - apply_co(A @ e0)
    orth = np.max(np.abs(Z.T @ (A @ e1))) / np.max(np.abs(Z.T @ (A @ e0)))
    pyth = abs((e0 @ (A @ e0)) - (e1 @ (A @ e1)) - ((e0 - e1) @ (A @ (e0 - e1))))
    pyth_rel = pyth / (e0 @ (A @ e0))
    nc = Z.shape[1]
    ok(f"{tag} ({nc} dofs): E is an A-orthogonal projector (idempotent {idem:.1e}, "
       f"Z^T A e_1 = 0 to {orth:.1e}, A-Pythagoras to {pyth_rel:.1e}), rho(E) = {rho_co:.6f} = 1",
       idem < 1e-9 and orth < 1e-10 and pyth_rel < 1e-12 and abs(rho_co - 1) < 1e-8)
    stall = abs(hl[-1] - hl[1]) / hl[1]
    ok(f"{tag}: eliminates the coarse component in ONE step then STALLS "
       f"(plateau rel l2 = {hl[1]:.4f}, A-norm = {ha[1]:.4f}; drift over "
       f"{COARSE_MAXIT-1} further its = {stall:.1e})", stall < 1e-10)
    LADDER[tag] = {"rho_exact": rho_co, "slope_measured": None, "iters_to_tol": None,
                   "coarse_dofs": nc, "plateau_rel_l2": float(hl[1]),
                   "plateau_rel_A": float(ha[1]), "plateau_iter": 1,
                   "idempotency_dev": float(idem), "A_pythagoras_rel_dev": float(pyth_rel)}
pa = LADDER["coarse_only_blocks"]["plateau_rel_A"]
pb = LADDER["coarse_only_bilinear"]["plateau_rel_A"]
ok(f"bilinear coarse space captures much more of e_0 than block averages: A-norm plateau "
   f"{pb:.4f} < {pa:.4f} (fraction of ||e_0||_A outside the coarse space)", pb < 0.6 * pa)

# ===========================================================================
# 7. two-level multiplicative (two-grid cycles)
# ===========================================================================
print("== 7. two-level multiplicative: omega=4/5 damped Jacobi, nu=1 pre + 1 post ==")
OMEGA = 0.8
Sd = np.eye(N) - OMEGA * inv_diag[:, None] * Ad
for tag, Z in [("twogrid_mult_blocks", Z_blk), ("twogrid_mult_bilinear", Z_bil)]:
    apply_tg, Ac_tg, cho_tg = make_twogrid(A, Z, OMEGA)
    Pi = np.eye(N) - Z @ np.linalg.solve(Ac_tg, Z.T @ Ad)
    E_tg = Sd @ Pi @ Sd
    rho_tg = rho_dense(E_tg)
    # implementation consistency: the cycle's error propagation IS E = S Pi S
    v = rng.standard_normal(N)
    dev_op = np.linalg.norm((v - apply_tg(Ad @ v)) - E_tg @ v) / np.linalg.norm(v)
    x_t, hl, ha, it_t = run_richardson(A, b, apply_tg, xstar)
    sl_t = tail_slope(hl)
    HIST[tag] = (hl, ha)
    ok(f"{tag}: cycle operator == S(I - Z Ac^-1 Z^T A)S on a random vector "
       f"(rel dev {dev_op:.1e})", dev_op < 1e-12)
    ok(f"{tag}: converged in {it_t} iterations, rho exact = {rho_tg:.4f} < 1",
       it_t is not None and rho_tg < 1)
    # rho ~ 0.36 means 1e-10 is reached in ~17 its, before the dominant mode
    # fully separates from the cluster below it (last generic ratios still
    # rising); the dominant-mode-seeded slope measures the true asymptote of
    # the implemented operator.
    sl_seed = seeded_slope(A, apply_tg, E_tg)
    ok(f"{tag}: measured slope matches rho {rho_tg:.4f} to 1% -- generic-x0 tail slope "
       f"{sl_t:.4f} ({100*abs(sl_t-rho_tg)/rho_tg:.1f}% off, transient-limited), "
       f"dominant-mode-seeded slope {sl_seed:.6f}",
       abs(sl_t - rho_tg) / rho_tg < 0.01 or abs(sl_seed - rho_tg) / rho_tg < 0.005)
    LADDER[tag] = {"rho_exact": rho_tg, "slope_measured": float(sl_t),
                   "slope_measured_seeded": float(sl_seed), "iters_to_tol": it_t,
                   "omega": OMEGA, "nu_pre": 1, "nu_post": 1, "coarse_dofs": Z.shape[1]}

# ===========================================================================
# 8. two-level additive (Jacobi + coarse, damped)
# ===========================================================================
print("== 8. two-level additive: C = theta (omega D^{-1} + Z_blk Ac^{-1} Z_blk^T) ==")
apply_co_blk, Ac_blk, cho_blk = make_coarse(A, Z_blk)
C_add0 = OMEGA * np.diag(inv_diag) + Z_blk @ np.linalg.solve(Ac_blk, Z_blk.T)
R_add = np.linalg.cholesky(C_add0)
mu = np.linalg.eigvalsh(R_add.T @ Ad @ R_add)   # spec(C_add0 A), real positive
theta = 2.0 / (mu[0] + mu[-1])                  # optimal damping: rho = (kap-1)/(kap+1)
rho_add = float((mu[-1] - mu[0]) / (mu[-1] + mu[0]))
apply_add = lambda r: theta * (OMEGA * inv_diag * r + apply_co_blk(r))  # noqa: E731
x_a, hl, ha, it_a = run_richardson(A, b, apply_add, xstar)
sl_a = tail_slope(hl)
HIST["additive_blocks"] = (hl, ha)
rho_add_dense = rho_dense(np.eye(N) - theta * C_add0 @ Ad)
ok(f"additive: theta = 2/(mu_min+mu_max) = {theta:.4f} (documented choice: optimal damping of "
   f"the SPD additive operator, spec = [{mu[0]:.4f}, {mu[-1]:.4f}]) gives rho = {rho_add:.4f} < 1",
   rho_add < 1 and np.isclose(rho_add, rho_add_dense, rtol=1e-8))
ok(f"additive: converged in {it_a} iterations; measured slope {sl_a:.4f} matches rho to 1%",
   it_a is not None and abs(sl_a - rho_add) / rho_add < 0.01)
LADDER["additive_blocks"] = {"rho_exact": rho_add, "slope_measured": float(sl_a),
                             "iters_to_tol": it_a, "theta": float(theta), "omega": OMEGA,
                             "mu_min": float(mu[0]), "mu_max": float(mu[-1])}

# ===========================================================================
# 9. mesh (in)dependence: n in {16, 32, 64}
# ===========================================================================
print("== 9. mesh (in)dependence: Jacobi / GS / IC(0) / two-grid-bilinear ==")
t0 = time.time()
MESH = {}


def build_mesh_case(nn):
    """Build the 4 mesh-study methods at size nn; exact rho dense if nn<=32,
    else sparse eigsh/eigs + measured slope."""
    NN = nn * nn
    hh = 1.0 / (nn + 1)
    An = poisson_2d(nn)
    bn = grf_rhs(nn)
    xs = spla.spsolve(An.tocsc(), bn)
    invd = 1.0 / An.diagonal()
    out = {"h": hh, "N": NN}

    # --- operators -------------------------------------------------------
    lu_t = make_tri_solver(sp.tril(An, format="csc"))
    Adn = An.toarray()
    Licn = ic0(Adn)
    lu_i = make_tri_solver(sp.csc_matrix(Licn))
    Zb = np.kron(interp_1d(nn), interp_1d(nn))
    apply_tgn, Ac_n, cho_n = make_twogrid(An, Zb, OMEGA)
    appliers = {
        "jacobi": lambda r: invd * r,
        "gauss_seidel": lambda r: lu_t.solve(r),
        "ic0": lambda r: lu_i.solve(lu_i.solve(r, trans="N"), trans="T"),
        "twogrid_mult_bilinear": apply_tgn,
    }

    # --- exact rho -------------------------------------------------------
    rho = {"jacobi": float(np.cos(np.pi * hh)),
           "gauss_seidel": float(np.cos(np.pi * hh) ** 2)}
    rho_src = {"jacobi": "analytic cos(pi h)", "gauss_seidel": "analytic cos^2(pi h)"}
    if nn <= 32:
        E = np.eye(NN) - sla.solve_triangular(np.tril(Adn), Adn, lower=True)
        rho_gs_d = rho_dense(E)
        assert np.isclose(rho_gs_d, rho["gauss_seidel"], rtol=1e-6)
        Licinv_n = sla.solve_triangular(Licn, np.eye(NN), lower=True)
        wg = np.linalg.eigvalsh(Licinv_n @ Adn @ Licinv_n.T)
        rho["ic0"] = float(max(abs(1 - wg[0]), abs(1 - wg[-1])))
        rho_src["ic0"] = "dense eigvalsh(L^-1 A L^-T)"
        Pi_n = np.eye(NN) - Zb @ np.linalg.solve(Ac_n, Zb.T @ Adn)
        Sd_n = np.eye(NN) - OMEGA * invd[:, None] * Adn
        rho["twogrid_mult_bilinear"] = rho_dense(Sd_n @ Pi_n @ Sd_n)
        rho_src["twogrid_mult_bilinear"] = "dense eigvals(S Pi S)"
    else:
        # sparse/iterative where practical (n=64)
        try:
            Bn = sp.identity(NN) - sp.diags(invd) @ An
            v = spla.eigsh(Bn, k=1, which="LA", return_eigenvectors=False, tol=1e-10)
            rho["jacobi"] = float(v[0])
            rho_src["jacobi"] = "sparse eigsh(B, LA)"
        except Exception as ex:  # pragma: no cover
            rho_src["jacobi"] += f" (eigsh failed: {ex})"
        try:
            op = spla.LinearOperator((NN, NN), matvec=lambda v: v - lu_t.solve(An @ v))
            vals = spla.eigs(op, k=4, which="LM", ncv=64, tol=1e-8,
                             return_eigenvectors=False)
            rho["gauss_seidel"] = float(np.max(np.abs(vals)))
            rho_src["gauss_seidel"] = "sparse eigs(E_GS, LM)"
        except Exception as ex:  # pragma: no cover
            rho_src["gauss_seidel"] += f" (eigs failed: {ex})"
        try:
            def g_ic(v):
                return lu_i.solve(An @ lu_i.solve(v, trans="T"), trans="N")
            opg = spla.LinearOperator((NN, NN), matvec=g_ic)
            lo = spla.eigsh(opg, k=1, which="SA", return_eigenvectors=False,
                            tol=1e-8, ncv=64, maxiter=20000)
            hi = spla.eigsh(opg, k=1, which="LA", return_eigenvectors=False,
                            tol=1e-8, ncv=64, maxiter=20000)
            rho["ic0"] = float(max(abs(1 - lo[0]), abs(1 - hi[0])))
            rho_src["ic0"] = "sparse eigsh(L^-1 A L^-T, SA/LA)"
        except Exception as ex:  # pragma: no cover
            rho["ic0"] = None
            rho_src["ic0"] = f"eigsh failed: {ex}"
        try:
            opt = spla.LinearOperator((NN, NN),
                                      matvec=lambda v: v - apply_tgn(An @ v))
            vals = spla.eigs(opt, k=4, which="LM", ncv=48, tol=1e-9,
                             return_eigenvectors=False)
            rho["twogrid_mult_bilinear"] = float(np.max(np.abs(vals)))
            rho_src["twogrid_mult_bilinear"] = "sparse eigs(E_TG, LM)"
        except Exception as ex:  # pragma: no cover
            rho["twogrid_mult_bilinear"] = None
            rho_src["twogrid_mult_bilinear"] = f"eigs failed: {ex}"

    # --- measured slopes ---------------------------------------------------
    for m, ap in appliers.items():
        _, hln, _, itn = run_richardson(An, bn, ap, xs)
        out[m] = {"rho_exact": rho.get(m), "rho_source": rho_src.get(m),
                  "slope_measured": float(tail_slope(hln)), "iters_to_tol": itn}
    return out


for nn in [16, 64]:
    MESH[nn] = build_mesh_case(nn)
MESH[32] = {"h": h, "N": N}
for m in ["jacobi", "gauss_seidel", "ic0", "twogrid_mult_bilinear"]:
    MESH[32][m] = {"rho_exact": LADDER[m]["rho_exact"],
                   "rho_source": "dense (ladder)",
                   "slope_measured": LADDER[m]["slope_measured"],
                   "iters_to_tol": LADDER[m]["iters_to_tol"]}

print("  n      method                    rho_exact   slope_meas   iters")
for nn in [16, 32, 64]:
    for m in ["jacobi", "gauss_seidel", "ic0", "twogrid_mult_bilinear"]:
        d = MESH[nn][m]
        r = "   n/a  " if d["rho_exact"] is None else f"{d['rho_exact']:.6f}"
        print(f"  {nn:3d}   {m:24s} {r}   {d['slope_measured']:.6f}   "
          f"{d['iters_to_tol']}")

hs = {nn: 1.0 / (nn + 1) for nn in [16, 32, 64]}
for m, tol_scale in [("jacobi", 0.05), ("gauss_seidel", 0.05), ("ic0", 0.30)]:
    r16 = MESH[16][m]["rho_exact"] or MESH[16][m]["slope_measured"]
    r32 = MESH[32][m]["rho_exact"]
    r64 = MESH[64][m]["rho_exact"] or MESH[64][m]["slope_measured"]
    q1 = (1 - r16) / (1 - r32)
    q2 = (1 - r32) / (1 - r64)
    p1 = (hs[16] / hs[32]) ** 2
    p2 = (hs[32] / hs[64]) ** 2
    ok(f"{m}: 1-rho scales like h^2: ratios {q1:.2f}, {q2:.2f} vs h^2-ratios "
       f"{p1:.2f}, {p2:.2f} (within {int(100*tol_scale)}%)",
       abs(q1 / p1 - 1) < tol_scale and abs(q2 / p2 - 1) < tol_scale)
tg_rhos = [MESH[nn]["twogrid_mult_bilinear"]["rho_exact"]
           or MESH[nn]["twogrid_mult_bilinear"]["slope_measured"] for nn in [16, 32, 64]]
ok(f"two-grid stays essentially mesh-independent: rho = "
   f"{tg_rhos[0]:.4f}, {tg_rhos[1]:.4f}, {tg_rhos[2]:.4f} across n = 16/32/64 "
   f"(spread {max(tg_rhos)-min(tg_rhos):.4f}, bounded away from 1)",
   max(tg_rhos) - min(tg_rhos) < 0.05 and max(tg_rhos) < 0.5)
info(f"mesh study time {time.time()-t0:.1f}s")
RESULTS["mesh_dependence"] = MESH

# ===========================================================================
# 10. weights anatomy of the perfect predictor
# ===========================================================================
print("== 10. weights anatomy: perfect-AR row of the central node (16,16) ==")
kc = k_center                       # 528
wrow = Phi[kc, :kc]
# compass convention of report 11: row i increases SOUTHWARD from the top, so
# the stencil predecessors of (16,16) are W = k-1 = (16,15) and the
# previous-row neighbor k-32 = (15,16) (= N under report 11's compass; the
# task's 'S' under a south-at-top drawing -- same node either way).
w_W = float(wrow[kc - 1])
w_S = float(wrow[kc - n])
wavefront_idx = np.arange(kc - n, kc)          # last 32 predecessors
tail_mask = np.ones(n, bool)
tail_mask[[0, n - 1]] = False                  # exclude k-32 and k-1
w_tail_max = float(np.max(np.abs(wrow[wavefront_idx][tail_mask])))
w_sum = float(np.sum(wrow))
# lateral profile along the wavefront: same-row branch (16, 16-lat) and
# previous-row branch (15, 16+lat)
lat_same = np.arange(1, 17)                    # d = 1..16 -> (16, 15..0)
w_same = np.abs(wrow[kc - lat_same])
lat_prev = np.arange(0, 16)                    # (15, 16..31) = k-32..k-17
w_prev = np.abs(wrow[kc - n + lat_prev])
fit_range = (lat_prev >= 2) & (lat_prev <= 12)
cfit = np.polyfit(lat_prev[fit_range], np.log(w_prev[fit_range]), 1)
decay_rate = float(np.exp(cfit[0]))            # |w| ~ C * rate^lat
fit_dev = float(np.max(np.abs(cfit[0] * lat_prev[fit_range] + cfit[1]
                              - np.log(w_prev[fit_range]))))
off_wave_max = float(np.max(np.abs(wrow[:kc - n])))
ok(f"perfect-AR weights are supported EXACTLY on the last-n wavefront: max |weight| on the "
   f"{kc-n} pre-wavefront predecessors = {off_wave_max:.1e} (the wavefront separates node k "
   "from all earlier nodes -- GMRF global Markov property = bandwidth-n of chol(A))",
   off_wave_max < 1e-12)
ok(f"W/previous-row stencil dominance: w_W = {w_W:.4f} ({w_W/w_tail_max:.1f}x), "
   f"w_S(prev-row) = {w_S:.4f} ({w_S/w_tail_max:.1f}x) vs largest wavefront-tail weight "
   f"{w_tail_max:.4f}", w_W > 2.5 * w_tail_max and w_S > 2.5 * w_tail_max)
ok(f"wavefront tail decays geometrically along the previous row: fitted rate "
   f"{decay_rate:.4f}/step over lateral 2..12 (max log-dev {fit_dev:.2f})",
   0.05 < decay_rate < 0.95 and fit_dev < 0.5)
info(f"sum of all {kc} weights = {w_sum:.6f} (two-sided interior row sums to 1); "
     f"w_W - w_S = {w_W-w_S:.4f}: lexicographic ordering breaks the W/prev-row tie -- "
     "the same-row W neighbor is the most recently 'observed' node and takes more weight")
info("no closed form in 2D (contrast 1D's coefficient (n+1-i)/(n+2-i) on the single "
     "predecessor neighbor); the wavefront row is a row of a Schur-complement "
     "(discrete Dirichlet-to-Neumann / transfer) operator -- verified below")

# innovation variances d2 across the grid
d2_grid = d2.reshape(n, n)
d2_c = float(d2[kc])
ratio_d2 = d2_c / (h * h / 4)
ok(f"one-sided innovation variance > two-sided conditional variance: d2(16,16) = {d2_c:.6e} "
   f"vs h^2/4 = {h*h/4:.6e}, ratio = {ratio_d2:.4f} (conditions on less)",
   ratio_d2 > 1.0)
info(f"d2 range on grid: [{d2.min():.4e}, {d2.max():.4e}]; first-row max "
     f"{d2_grid[0].max():.4e} (few predecessors) vs deep-interior ~{d2_c:.4e}")

# --- Schur-complement / DtN verification -----------------------------------
# D-scaling, defined precisely: from any Schur-complement row s (a precision
# row), prediction weights = -s_offdiag / s_diag and s_diag = 1/d2 (innovation
# precision).
# (b) direct: eliminate the SUCCESSORS k+1..N-1; row k of the result is the
#     marginal precision row of node k given predecessors only.
Q_blk = Ad[:kc + 1, :kc + 1] - Ad[:kc + 1, kc + 1:] @ np.linalg.solve(
    Ad[kc + 1:, kc + 1:], Ad[kc + 1:, :kc + 1])
w_schur_b = -Q_blk[kc, :kc] / Q_blk[kc, kc]
dev_b = float(np.max(np.abs(w_schur_b - wrow)))
dev_b_d2 = float(abs(1.0 / Q_blk[kc, kc] - d2_c) / d2_c)
ok(f"Schur (eliminate successors of k): effective row at k reproduces the prediction weights "
   f"(max dev {dev_b:.2e}) and 1/S_kk == d2 (rel dev {dev_b_d2:.2e})",
   dev_b < 1e-9 and dev_b_d2 < 1e-9)
# (a) literal: eliminate nodes 0..k-1 exactly; first row of the trailing Schur
#     complement = L[k,k] * L[k:,k]^T (trailing Cholesky), i.e. the phiR2L
#     successor weights of node k -- which by the reversal identity are the
#     prediction weights of the mirror node N-1-k.
S_tr = Ad[kc:, kc:] - Ad[kc:, :kc] @ np.linalg.solve(Ad[:kc, :kc], Ad[:kc, kc:])
w_succ_schur = -S_tr[0, 1:] / S_tr[0, 0]
w_succ_L = -L[kc + 1:, kc] / L[kc, kc]
dev_a1 = float(np.max(np.abs(w_succ_schur - w_succ_L)))
km = N - 1 - kc                                 # mirror node (15,15)
w_mirror = Phi[km, :km][::-1]                   # predecessor weights, reversed
dev_a2 = float(np.max(np.abs(w_succ_schur[:km] - w_mirror)))
dev_a3 = float(abs(1.0 / S_tr[0, 0] - d2[km]) / d2[km])
ok(f"Schur (eliminate 0..k-1, the task's literal form): row at k == chol(A) successor "
   f"weights (dev {dev_a1:.2e}) == reversed prediction weights of the mirror node "
   f"(15,15) (dev {dev_a2:.2e}); 1/S_00 == d2(mirror) (rel dev {dev_a3:.2e})",
   dev_a1 < 1e-9 and dev_a2 < 1e-9 and dev_a3 < 1e-9)

RESULTS["weights_anatomy"] = {
    "node": [16, 16], "flat_index": kc,
    "w_W": w_W, "w_S_prev_row": w_S, "largest_wavefront_tail_weight": w_tail_max,
    "fitted_tail_decay_rate_per_step": decay_rate, "sum_of_all_weights": w_sum,
    "wavefront_same_row_abs": w_same, "wavefront_prev_row_abs": w_prev,
    "d2_center": d2_c, "two_sided_cond_var_h2_over_4": h * h / 4,
    "d2_over_h2_over_4": ratio_d2,
    "d2_grid_min": float(d2.min()), "d2_grid_max": float(d2.max()),
    "d2_row16_profile": d2_grid[16].tolist(),
    "schur_direct_max_dev": dev_b, "schur_direct_d2_rel_dev": dev_b_d2,
    "schur_eliminate_predecessors_dev_vs_cholA": dev_a1,
    "schur_eliminate_predecessors_dev_vs_mirror_phi": dev_a2,
    "max_weight_off_wavefront": off_wave_max,
    "phi_row_center": wrow,
    "d2_full": d2,
}

# ===========================================================================
# sanity cross-method checks
# ===========================================================================
print("== sanity: cross-method claims ==")
ok(f"GS beats Jacobi ~2x in rate exponent: log(rho_GS)/log(rho_J) = "
   f"{np.log(rho_gs)/np.log(rho_B):.4f}",
   abs(np.log(rho_gs) / np.log(rho_B) - 2.0) < 0.01)
conv_methods = ["richardson_opt", "jacobi", "gauss_seidel", "ic0",
                "twogrid_mult_blocks", "twogrid_mult_bilinear", "additive_blocks"]
ok("all rho<1 methods actually converge to rel l2 err <= 1e-10 within 20000 iterations",
   all(LADDER[m]["iters_to_tol"] is not None for m in conv_methods))
rho_tg_bil = LADDER["twogrid_mult_bilinear"]["rho_exact"]
ok(f"two-level multiplicative bilinear is the best stationary method: rho = {rho_tg_bil:.4f} "
   "< every other (non-perfect) rho",
   all(rho_tg_bil < LADDER[m]["rho_exact"] for m in conv_methods
       if m != "twogrid_mult_bilinear"))

# ===========================================================================
# figures
# ===========================================================================
print("== figures ==")
LABELS = {
    "richardson_opt": "Richardson $\\alpha^*$",
    "jacobi": "Jacobi (= Richardson $\\alpha^*$)",
    "gauss_seidel": "Gauss-Seidel",
    "perfect_ar": "perfect AR ($M=A$)",
    "ic0": "IC(0) = truncated AR",
    "coarse_only_blocks": "coarse-only 4x4 blocks (64)",
    "coarse_only_bilinear": "coarse-only bilinear 16x16 (256)",
    "twogrid_mult_blocks": "two-grid mult. (blocks)",
    "twogrid_mult_bilinear": "two-grid mult. (bilinear)",
    "additive_blocks": "two-level additive (blocks)",
}
ORDER = ["richardson_opt", "jacobi", "gauss_seidel", "perfect_ar", "ic0",
         "coarse_only_blocks", "coarse_only_bilinear",
         "twogrid_mult_blocks", "twogrid_mult_bilinear", "additive_blocks"]
STYLE = {"jacobi": ":", "coarse_only_blocks": "--", "coarse_only_bilinear": "--"}

for which, fname, ylab in [(0, "richardson_error_convergence.png",
                            r"$\|e_k\|_2 / \|e_0\|_2$"),
                           (1, "richardson_error_Anorm.png",
                            r"$\|e_k\|_A / \|e_0\|_A$")]:
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for m in ORDER:
        hist = HIST[m][which]
        rho_m = LADDER[m]["rho_exact"]
        if m == "perfect_ar":
            lab = f"{LABELS[m]}  [$\\rho\\approx 0$, 1 it]"
            ax.semilogy(np.arange(len(hist)), np.maximum(hist, 1e-16), "o-",
                        lw=1.6, ms=5, label=lab)
            continue
        if m.startswith("coarse_only"):
            lab = (f"{LABELS[m]}  [$\\rho=1$, plateau "
                   f"{LADDER[m]['plateau_rel_l2' if which == 0 else 'plateau_rel_A']:.2f}]")
            xs_plot = np.arange(len(hist))
            ax.semilogy([0, 1000], [hist[1], hist[1]], STYLE[m], lw=1.4, label=lab)
            continue
        it = LADDER[m]["iters_to_tol"]
        lab = f"{LABELS[m]}  [$\\rho$={rho_m:.4f}, {it} its]"
        ax.semilogy(np.arange(len(hist)), hist, STYLE.get(m, "-"), lw=1.5, label=lab)
    ax.axhline(1e-10, color="gray", lw=0.8, ls=":")
    ax.set_xlim(0, 1000)
    ax.set_ylim(1e-14, 3)
    ax.set_xlabel("iteration k")
    ax.set_ylabel(ylab)
    ax.set_title("every preconditioner as an AR predictor, every solver as stationary "
                 "Richardson\n$x_{k+1} = x_k + C\\,(b - Ax_k)$,  n=32, b=grf_rhs(32)"
                 + ("  (A-norm)" if which else ""))
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / fname, dpi=150)
    plt.close(fig)

# mesh dependence figure
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
ns = [16, 32, 64]
mcol = {"jacobi": "tab:blue", "gauss_seidel": "tab:orange", "ic0": "tab:green",
        "twogrid_mult_bilinear": "tab:red"}
for m in mcol:
    rr = [MESH[nn][m]["rho_exact"] if MESH[nn][m]["rho_exact"] is not None
          else MESH[nn][m]["slope_measured"] for nn in ns]
    axes[0].plot(ns, rr, "o-", color=mcol[m], label=m)
    axes[1].loglog(ns, [1 - r for r in rr], "o-", color=mcol[m], label=m)
nref = np.array(ns, float)
axes[1].loglog(ns, 0.5 * (1 - MESH[16]["jacobi"]["rho_exact"]) * (17.0 / (nref + 1)) ** 2,
               "k--", lw=1.0, label=r"$\propto h^2$")
axes[0].set_xlabel("n")
axes[0].set_ylabel(r"$\rho(E)$")
axes[0].set_title(r"$\rho \to 1$ for one-level methods; two-grid stays flat")
axes[0].set_xticks(ns)
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
axes[1].set_xlabel("n")
axes[1].set_ylabel(r"$1 - \rho$")
axes[1].set_title(r"$1-\rho = O(h^2)$ for Jacobi / GS / IC(0)")
axes[1].set_xticks(ns)
axes[1].set_xticklabels([str(v) for v in ns])
axes[1].minorticks_off()
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIGDIR / "richardson_mesh_dependence.png", dpi=150)
plt.close(fig)

# weights anatomy figure
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
wgrid = np.full(N, np.nan)
wgrid[:kc] = np.abs(wrow)
wgrid = wgrid.reshape(n, n)
logw = np.log10(wgrid + 1e-17)
cmap = plt.get_cmap("magma").copy()
cmap.set_bad("0.82")
im = axes[0].imshow(logw, cmap=cmap, vmin=-12, vmax=0)
axes[0].plot(16, 16, "c+", ms=12, mew=2)
axes[0].set_title("$\\log_{10}|$perfect-AR weights$|$ of node (16,16) on its 528 predecessors\n"
                  "(grey = not a predecessor; black = EXACT zero -- the wavefront separates)")
axes[0].set_xlabel("grid col j")
axes[0].set_ylabel("grid row i")
fig.colorbar(im, ax=axes[0], fraction=0.046)
axes[1].semilogy(lat_same, w_same, "o-", ms=4, label="same row (16, 16-lat)  [W branch]")
axes[1].semilogy(lat_prev, w_prev, "s-", ms=4, label="previous row (15, 16+lat)")
lat_fit = np.linspace(0, 15, 50)
axes[1].semilogy(lat_fit, np.exp(cfit[1]) * decay_rate**lat_fit, "k--", lw=1.0,
                 label=f"geometric fit: rate {decay_rate:.3f}/step")
axes[1].set_xlabel("lateral distance along the wavefront")
axes[1].set_ylabel("|weight|")
axes[1].set_title("wavefront profile: stencil W/S dominate,\ngeometric tail "
                  f"(w_W={w_W:.3f}, w_S={w_S:.3f}, max tail={w_tail_max:.3f})")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3, which="both")
im = axes[2].imshow(d2_grid, cmap="viridis")
axes[2].set_title("innovation variances $d^2$ (one-sided)\n"
                  f"interior {d2_c:.3e} = {ratio_d2:.3f} x two-sided $h^2/4$")
axes[2].set_xlabel("grid col j")
axes[2].set_ylabel("grid row i")
fig.colorbar(im, ax=axes[2], fraction=0.046)
fig.tight_layout()
fig.savefig(FIGDIR / "richardson_ar_weights.png", dpi=150)
plt.close(fig)

# ===========================================================================
# JSON
# ===========================================================================
rates_table = {}
for m in ORDER:
    d = LADDER[m]
    rates_table[m] = {
        "rho_exact": d["rho_exact"],
        "slope_measured": d["slope_measured"],
        "iters_to_1e-10_or_plateau": (d["iters_to_tol"] if d["iters_to_tol"] is not None
                                      else {"plateau_rel_l2": d.get("plateau_rel_l2"),
                                            "plateau_rel_A": d.get("plateau_rel_A"),
                                            "plateau_iter": d.get("plateau_iter")}),
    }
RESULTS["ladder"] = LADDER
RESULTS["richardson_rates_table"] = rates_table
RESULTS["histories"] = {m: {"rel_err_l2": HIST[m][0], "rel_err_A": HIST[m][1]}
                        for m in ORDER}
RESULTS["pass_fail"] = PASS_LINES
RESULTS["figures"] = ["richardson_error_convergence.png", "richardson_error_Anorm.png",
                      "richardson_mesh_dependence.png", "richardson_ar_weights.png"]
with open(RESDIR / "richardson_ar.json", "w") as fj:
    json.dump(jsonable(RESULTS), fj, indent=2)
print(f"saved results/richardson_ar.json; {N_FAIL} FAIL line(s); "
      f"total time {time.time()-T_START:.1f}s")
