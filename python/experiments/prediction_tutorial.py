"""Machine-checked companion for the step-by-step prediction tutorial (report 15).

Every matrix the tutorial displays is PRINTED here (exact fractions where the
input is rational, via the fractions module) and every identity is PASS-checked
in the style of verify_statistical_identities.py.

Notation follows the companion notebook
~/git0/newton/whitening_inverse_transposed.nb:
  B            hollow two-sided regression matrix, B = I - diag(A)^{-1} A
               (row convention: row i predicts u_i; the notebook's getB
               stacks the coefficient vectors as COLUMNS -- the transpose)
  D2           diagonal of inverse residual variances (= diag(A))
  precisionMat == (I - B).D2
  mchol        modified Cholesky Sigma = C D^2 C^T (C unit lower)
  phiL2R       regress coordinate i on its predecessors 1..i-1
  phiR2L       regress coordinate i on its successors i+1..n

Steps (the report mirrors these numbers):
  1  the chain, exact: n=5, h=1/6, A and Sigma = A^{-1} as exact fractions
  2  two-sided prediction: B, D2, A == (I-B).D2, worked row E[u_3|rest]
  3  one-sided prediction: phiL2R (coeffs 4/5,3/4,2/3,1/2), phiR2L
     (1/2,2/3,3/4,4/5), A == (I-Phi)^T diag(1/d^2) (I-Phi) exact, reversal
     identity chol(Sigma) = P L^{-T} P, mchol, whitening z = (I-Phi)u
  4  perfect prediction = direct solve: M^{-1} assembled from the regressions
     equals Sigma exactly; one Richardson step lands on x* for b = e_3
  5  approximate prediction by TRUNCATION (2D, n=4, N=16): exact phiL2R
     wavefront row vs IC(0) kept pattern, dropped l2 mass, M_ic == L_ic L_ic^T,
     rho / kappa vs exact-regression Vecchia (report 11's comparison)
  6  approximate prediction by SCALE: chain global-mean loadings beta exact
     (bridge-shaped), variance explained; 2D 2x2 block averages + two-level
     M^{-1} = D^{-1} + Z Ac^{-1} Z^T
  7  residual correction: Richardson x <- x + M^{-1}(b - Ax) for jacobi /
     IC(0) / two-level; measured tail rate == rho(I - M^{-1}A) (report 12's
     per-sweep unexplained fraction)
  8  CG interaction: CG vs PCG(IC0) vs PCG(two-level); PCG trajectory ==
     whitened-CG-mapped-back trajectory (report 13 SS5.4 made executable);
     CG direction A-orthogonality

Run from the repo root:
    uv run python python/experiments/prediction_tutorial.py

Figures -> figures/tutorial15_*.png (dpi 150); numbers ->
results/prediction_tutorial.json.
"""

import json
import sys
from fractions import Fraction as Fr
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcg import pcg
from poisson import poisson_2d
from preconditioners import block_average_matrix, ic0

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"
RESDIR = ROOT / "results"
np.set_printoptions(precision=4, suppress=True, linewidth=150)

RESULTS = {}
N_PASS = 0
N_FAIL = 0


def ok(name, cond):
    global N_PASS, N_FAIL
    if cond:
        N_PASS += 1
    else:
        N_FAIL += 1
    print(f"{'PASS' if cond else 'FAIL'}: {name}")
    return bool(cond)


def info(msg):
    print(f"  [info] {msg}")


# ---------------------------------------------------------------------------
# exact Fraction linear algebra (small matrices as lists of lists of Fraction)
# ---------------------------------------------------------------------------
def feye(n):
    return [[Fr(1) if i == j else Fr(0) for j in range(n)] for i in range(n)]


def fT(M):
    return [list(row) for row in zip(*M)]


def fmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]


def fsub(A, B):
    return [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def finv(M):
    """Exact Gauss-Jordan inverse over the rationals."""
    n = len(M)
    aug = [[Fr(M[i][j]) for j in range(n)]
           + [Fr(1) if j == i else Fr(0) for j in range(n)] for i in range(n)]
    for c in range(n):
        p = next(r for r in range(c, n) if aug[r][c] != 0)
        aug[c], aug[p] = aug[p], aug[c]
        piv = aug[c][c]
        aug[c] = [v / piv for v in aug[c]]
        for r in range(n):
            if r != c and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[c])]
    return [row[n:] for row in aug]


def fdiag(vals):
    n = len(vals)
    return [[vals[i] if i == j else Fr(0) for j in range(n)] for i in range(n)]


def ffloat(M):
    return np.array([[float(x) for x in row] for row in M])


def fprint(name, M):
    """Aligned exact-fraction matrix print."""
    s = [[str(x) for x in row] for row in M]
    w = [max(len(s[i][j]) for i in range(len(s))) for j in range(len(s[0]))]
    print(f"{name} =")
    for row in s:
        print("    [ " + "   ".join(v.rjust(wj) for v, wj in zip(row, w)) + " ]")


def fjson(M):
    if isinstance(M, list):
        return [fjson(v) for v in M]
    if isinstance(M, Fr):
        return str(M)
    return M


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, Fr):
        return str(x)
    return x


# ===========================================================================
# STEP 1 -- the chain, exact
# ===========================================================================
print("== STEP 1: the chain, exact: n=5, h=1/6, A = tridiag(-1,2,-1)/h^2 ==")
n = 5
h = Fr(1, 6)
x = [Fr(i, 6) for i in range(1, n + 1)]          # physical nodes i/6
A = [[Fr(0)] * n for _ in range(n)]
for i in range(n):
    A[i][i] = Fr(2) / h**2                        # 72
    if i > 0:
        A[i][i - 1] = Fr(-1) / h**2               # -36
    if i < n - 1:
        A[i][i + 1] = Fr(-1) / h**2
fprint("A  (h^-2 = 36, so 2/h^2 = 72, -1/h^2 = -36)", A)

# Sigma from the bridge formula (report 09): Sigma_ij = h x_i (1 - x_j), i <= j
S = [[h * x[min(i, j)] * (1 - x[max(i, j)]) for j in range(n)] for i in range(n)]
fprint("Sigma = A^{-1}  (bridge formula Sigma_ij = h x_i (1-x_j), i<=j)", S)

S_gj = finv(A)                                    # independent exact route
I5 = feye(n)
ok("Sigma_ij = h x_i (1-x_j) (i<=j) equals the exact Gauss-Jordan inverse of A "
   "entry-by-entry (Fraction arithmetic, zero tolerance)", S == S_gj)
ok("A . Sigma == I exactly in Fractions (every entry a Fraction identity)",
   fmul(A, S) == I5)
ok("exactness: every printed entry is an exact rational (no floats anywhere)",
   all(isinstance(v, Fr) for row in A + S for v in row))
RESULTS["step1"] = {"n": n, "h": str(h), "A": fjson(A), "Sigma": fjson(S)}

# ===========================================================================
# STEP 2 -- two-sided prediction (notebook B, D2)
# ===========================================================================
print("\n== STEP 2: two-sided prediction: B = I - diag(A)^{-1} A, D2 = diag(A) ==")
Dinv = fdiag([1 / A[i][i] for i in range(n)])
B = fsub(I5, fmul(Dinv, A))
fprint("B  (row i = regression coefficients of u_i on all the others)", B)
D2 = fdiag([A[i][i] for i in range(n)])
fprint("D2 (inverse residual variances = diag(A))", D2)
ok("B is hollow (zero diagonal) with exactly 1/2 on both neighbors",
   all(B[i][i] == 0 for i in range(n))
   and all(B[i][j] == Fr(1, 2) for i in range(n) for j in (i - 1, i + 1)
           if 0 <= j < n))
ok("notebook identity precisionMat == (I - B).D2 == A exactly in Fractions",
   fmul(fsub(I5, B), D2) == A)
# Convention note (report 10 SS3.1): the notebook's getB stacks the
# per-coordinate coefficient vectors as COLUMNS (transpose of the row
# convention used here); for a row-convention B the general identity is
# D2.(I - B) == A, and the notebook's order (I - B).D2 == A holds on this
# chain only because diag(A) = 72 I makes B symmetric.
Bt = [[B[j][i] for j in range(n)] for i in range(n)]
ok("convention check: B == B^T here (constant diagonal 72 I), so "
   "(I - B).D2 == D2.(I - B) == A; in general the notebook's getB is the "
   "transpose (column) convention and only D2.(I - B) == A",
   B == Bt and fmul(D2, fsub(I5, B)) == A)
# worked row i = 3 (1-based; index 2)
i3 = 2
ok("worked row: E[u_3 | rest] = (u_2 + u_4)/2 (B row 3 = [0, 1/2, 0, 1/2, 0])",
   B[i3] == [Fr(0), Fr(1, 2), Fr(0), Fr(1, 2), Fr(0)])
ok("Var(u_3 | rest) = 1/A_33 = h^2/2 = 1/72 exactly",
   1 / A[i3][i3] == h**2 / 2 == Fr(1, 72))
# independent Schur-complement route: delete row/col 3 of Sigma
idx = [k for k in range(n) if k != i3]
Soo = [[S[a][b] for b in idx] for a in idx]
Sio = [S[i3][b] for b in idx]
w_full = fmul([Sio], finv(Soo))[0]                # regression of u_3 on rest
cvar = S[i3][i3] - sum(a * b for a, b in zip(w_full, Sio))
ok("Schur route agrees: Sigma_3,rest . Sigma_rest,rest^{-1} = (1/2, 1/2) on "
   "neighbors and conditional variance = 1/72, all exact",
   w_full == [Fr(0), Fr(1, 2), Fr(1, 2), Fr(0)] and cvar == Fr(1, 72))
RESULTS["step2"] = {"B": fjson(B), "D2_diag": fjson([D2[i][i] for i in range(n)]),
                    "row3_coeffs": fjson(B[i3]), "row3_cond_var": str(cvar)}

# ===========================================================================
# STEP 3 -- one-sided prediction (phiL2R / phiR2L, mchol)
# ===========================================================================
print("\n== STEP 3: one-sided prediction: phiL2R, phiR2L, mchol, whitening ==")
# phiL2R: exact sequential regression of u_i on predecessors u_1..u_{i-1}
PhiL = [[Fr(0)] * n for _ in range(n)]
d2 = [Fr(0)] * n
d2[0] = S[0][0]
for i in range(1, n):
    Spp = [[S[a][b] for b in range(i)] for a in range(i)]
    Spi = [S[a][i] for a in range(i)]
    w = fmul([Spi], finv(Spp))[0]
    PhiL[i][:i] = w
    d2[i] = S[i][i] - sum(a * b for a, b in zip(w, Spi))
fprint("Phi_L2R (row i = exact regression of u_i on u_1..u_{i-1})", PhiL)
fprint("d^2 (innovation variances, exact)", [d2])
ok("phiL2R coefficients: only the immediate predecessor enters (Markov) and "
   "equals (n+1-i)/(n+2-i): 4/5, 3/4, 2/3, 1/2 for i = 2..5 EXACTLY",
   all(PhiL[i][j] == 0 for i in range(n) for j in range(n)
       if j != i - 1) and [PhiL[i][i - 1] for i in range(1, n)]
   == [Fr(4, 5), Fr(3, 4), Fr(2, 3), Fr(1, 2)])
ok("innovation variances d_i^2 = h^2 (n+1-i)/(n+2-i) exactly: "
   "5/216, 1/45, 1/48, 1/54, 1/72",
   d2 == [h**2 * Fr(n + 1 - i, n + 2 - i) for i in range(1, n + 1)]
   and d2 == [Fr(5, 216), Fr(1, 45), Fr(1, 48), Fr(1, 54), Fr(1, 72)])
info("same ratio twice: for i >= 2, d_i^2 = h^2 * phi_i -- the innovation "
     "variance carries the regression coefficient")
TL = fsub(I5, PhiL)
ok("A == (I - phiL2R)^T diag(1/d^2) (I - phiL2R) exactly in Fractions",
   fmul(fmul(fT(TL), fdiag([1 / v for v in d2])), TL) == A)
ok("whitening: z = (I - phiL2R) u has diagonal covariance: "
   "(I-Phi) Sigma (I-Phi)^T == diag(d^2) exactly",
   fmul(fmul(TL, S), fT(TL)) == fdiag(d2))

# phiR2L: exact sequential regression of u_i on successors u_{i+1}..u_n
PhiR = [[Fr(0)] * n for _ in range(n)]
dR2 = [Fr(0)] * n
dR2[n - 1] = S[n - 1][n - 1]
for i in range(n - 2, -1, -1):
    Sss = [[S[a][b] for b in range(i + 1, n)] for a in range(i + 1, n)]
    Ssi = [S[a][i] for a in range(i + 1, n)]
    w = fmul([Ssi], finv(Sss))[0]
    PhiR[i][i + 1:] = w
    dR2[i] = S[i][i] - sum(a * b for a, b in zip(w, Ssi))
fprint("Phi_R2L (row i = exact regression of u_i on u_{i+1}..u_n)", PhiR)
fprint("dR^2 (innovation variances, exact)", [dR2])
ok("phiR2L coefficients: only the immediate successor, i/(i+1): "
   "1/2, 2/3, 3/4, 4/5 for i = 1..4 EXACTLY",
   all(PhiR[i][j] == 0 for i in range(n) for j in range(n) if j != i + 1)
   and [PhiR[i][i + 1] for i in range(n - 1)]
   == [Fr(1, 2), Fr(2, 3), Fr(3, 4), Fr(4, 5)])
TR = fsub(I5, PhiR)
ok("A == (I - phiR2L)^T diag(1/dR^2) (I - phiR2L) exactly in Fractions",
   fmul(fmul(fT(TR), fdiag([1 / v for v in dR2])), TR) == A)

# reversal identity + mchol (numeric, rtol ~ 1e-14)
Af = ffloat(A)
Sf = ffloat(S)
L = np.linalg.cholesky(Af)
P = np.eye(n)[::-1]
ok("reversal identity chol(Sigma) == P L^{-T} P with L = chol(A) (numeric, "
   "max dev %.1e)" % np.max(np.abs(np.linalg.cholesky(Sf)
                                   - P @ np.linalg.inv(L).T @ P)),
   np.allclose(np.linalg.cholesky(Sf), P @ np.linalg.inv(L).T @ P,
               rtol=1e-12, atol=1e-15))
ok("chol(A) encodes phiR2L: -L[i+1,i]/L[i,i] == i/(i+1) (successor "
   "coefficients live in the columns of L)",
   np.allclose([-L[i + 1, i] / L[i, i] for i in range(n - 1)],
               [i / (i + 1) for i in range(1, n)], rtol=1e-14))
# mchol of Sigma: Sigma = C D^2 C^T, C unit lower == (I - phiL2R)^{-1}
CS = np.linalg.cholesky(Sf)
dg = np.diag(CS)
C_unit = CS / dg[None, :]
ok("mchol(Sigma) = C D^2 C^T matches: C == (I - phiL2R)^{-1} and "
   "D^2 == diag(d^2) (max devs %.1e, %.1e)"
   % (np.max(np.abs(C_unit - ffloat(finv(TL)))),
      np.max(np.abs(dg**2 - ffloat([d2])[0]))),
   np.allclose(C_unit, ffloat(finv(TL)), rtol=1e-12, atol=1e-15)
   and np.allclose(dg**2, ffloat([d2])[0], rtol=1e-13))
RESULTS["step3"] = {"phiL2R": fjson(PhiL), "d2": fjson(d2),
                    "phiR2L": fjson(PhiR), "dR2": fjson(dR2)}

# ===========================================================================
# STEP 4 -- perfect prediction = direct solve
# ===========================================================================
print("\n== STEP 4: perfect prediction = direct solve ==")
# M^{-1} assembled FROM the regressions (no matrix inversion of A anywhere):
# (I-Phi)^{-1} is the exact inverse of a unit lower bidiagonal (back-substitution)
Minv = fmul(fmul(finv(TL), fdiag(d2)), fT(finv(TL)))
ok("M^{-1} = (I-Phi)^{-1} diag(d^2) (I-Phi)^{-T} == Sigma exactly in Fractions "
   "(assembled from regression coefficients, A never inverted)", Minv == S)
b4 = [Fr(0), Fr(0), Fr(1), Fr(0), Fr(0)]          # b = e_3
x_step = fmul(Minv, [[v] for v in b4])            # x1 = x0 + M^{-1}(b - A x0), x0=0
x1 = [r[0] for r in x_step]
fprint("x* = M^{-1} e_3 (one Richardson step from x0 = 0)", [x1])
ok("one Richardson step lands on the exact solution: x1 == column 3 of Sigma "
   "== (1/72, 1/36, 1/24, 1/36, 1/72) exactly",
   x1 == [S[i][2] for i in range(n)]
   and x1 == [Fr(1, 72), Fr(1, 36), Fr(1, 24), Fr(1, 36), Fr(1, 72)])
ok("check: A x1 == e_3 exactly (it IS the solve)",
   [r[0] for r in fmul(A, [[v] for v in x1])] == b4)
RESULTS["step4"] = {"b": fjson(b4), "x_star": fjson(x1)}

# ===========================================================================
# STEP 5 -- approximate prediction by TRUNCATION (2D)
# ===========================================================================
print("\n== STEP 5: truncation in 2D: A2 = poisson_2d(4), N = 16, h = 1/5 ==")
n2 = 4
N2 = n2 * n2
h2 = 1.0 / (n2 + 1)
A2 = poisson_2d(n2)
A2d = A2.toarray()
S2 = np.linalg.inv(A2d)
wA2 = np.linalg.eigvalsh(A2d)
kappa_A2 = wA2[-1] / wA2[0]
ok(f"plain kappa(A2) = {kappa_A2:.4f} == cot^2(pi h/2) = "
   f"{1/np.tan(np.pi*h2/2)**2:.4f} (2-D Dirichlet spectrum)",
   np.isclose(kappa_A2, 1 / np.tan(np.pi * h2 / 2) ** 2, rtol=1e-10))

k5 = 10                                            # node (2,2) 0-based: late interior
w_exact = np.linalg.solve(S2[:k5, :k5], S2[:k5, k5])
d2_exact = S2[k5, k5] - S2[k5, :k5] @ w_exact
print(f"  exact phiL2R row of node k=10 = grid (2,2) [0-based], predecessors k=0..9:")
for kk in range(k5):
    gi, gj = divmod(kk, n2)
    tag = {k5 - 1: "  <- W neighbor (2,1), KEPT by IC(0)",
           k5 - n2: "  <- prev-row neighbor (1,2), KEPT by IC(0)"}.get(kk, "")
    print(f"    k={kk:2d} (i,j)=({gi},{gj}): {w_exact[kk]:8.4f}{tag}")
print(f"    innovation variance d^2 = {d2_exact:.6f}  (h^2/4 = {h2*h2/4:.6f})")
off_wave = float(np.max(np.abs(w_exact[:k5 - n2])))
ok(f"wavefront support: exact one-sided weights vanish off the last-n "
   f"predecessors (max |weight| before the wavefront = {off_wave:.1e})",
   off_wave < 1e-12)

# IC(0) truncation: coefficients from ic0's ACTUAL factor.  ic0 eliminates
# first-to-last, so its column k holds the truncated regression of u_k on its
# stencil SUCCESSORS (report 11).  The mirror identity P A P = A converts the
# task's W/prev-row PREDECESSOR phrasing at node k to ic0's column at the
# mirror node N-1-k.
Lic = ic0(A2d)
kr = N2 - 1 - k5                                   # mirror node 5 = (1,1)
w_ic_W = -Lic[kr + 1, kr] / Lic[kr, kr]            # -> coefficient on k-1 (W)
w_ic_S = -Lic[kr + n2, kr] / Lic[kr, kr]           # -> coefficient on k-n (prev row)
w_succ_mirror = np.linalg.solve(S2[kr + 1:, kr + 1:], S2[kr + 1:, kr])
ok("mirror identity (P A P = A): exact predecessor row of node 10 == reversed "
   "exact successor row of node 5 (max dev %.1e)"
   % np.max(np.abs(w_exact - w_succ_mirror[::-1])),
   np.allclose(w_exact, w_succ_mirror[::-1], atol=1e-12))
print(f"  IC(0) kept coefficients (from ic0's factor, column {kr}): "
      f"W = {w_ic_W:.4f}, prev-row = {w_ic_S:.4f}")
print(f"  exact coefficients at the kept positions:  "
      f"W = {w_exact[k5-1]:.4f}, prev-row = {w_exact[k5-n2]:.4f}")
dropped_idx = [kk for kk in range(k5) if kk not in (k5 - 1, k5 - n2)]
drop_mass = float(np.linalg.norm(w_exact[dropped_idx]))
drop_rel = drop_mass / float(np.linalg.norm(w_exact))
print(f"  dropped coefficients l2 mass = {drop_mass:.4f} "
      f"({100*drop_rel:.1f}% of the full row's l2 norm {np.linalg.norm(w_exact):.4f})")
ok("the truncation bites: dropped l2 mass is real (> 0.05) but subdominant "
   f"(< the kept W coefficient {w_exact[k5-1]:.4f})",
   0.05 < drop_mass < w_exact[k5 - 1])

# assemble M_ic from ic0's factor in its OWN elimination direction
# (successor / phiR2L convention): L_ic L_ic^T = (I-W~)^T diag(1/d~^2) (I-W~)
Lnorm = Lic / np.diag(Lic)[None, :]
W_tilde = np.eye(N2) - Lnorm.T                     # strictly upper: successor weights
d2_tilde = 1.0 / np.diag(Lic) ** 2
T_tilde = np.eye(N2) - W_tilde
M_ic = T_tilde.T @ (T_tilde / d2_tilde[:, None])
ok("assembled M_ic = (I - Phi~)^T diag(1/d~^2) (I - Phi~) == L_ic L_ic^T "
   "(max dev %.1e)" % np.max(np.abs(M_ic - Lic @ Lic.T)),
   np.allclose(M_ic, Lic @ Lic.T, rtol=1e-13, atol=1e-12))
Prev = np.eye(N2)[::-1]
info(f"direction-dependence of the truncated recurrence: P M_ic P != M_ic "
     f"(max dev {np.max(np.abs(Prev@M_ic@Prev - M_ic)):.2e}) even though "
     f"P A P == A -- exact regression is reversal-equivariant, IC(0) is not")

# exact-regression Vecchia on the SAME pattern (coefficients from Sigma2)
Phi_v = np.zeros((N2, N2))
d2_v = np.zeros(N2)
dev_coef = []
for kk in range(N2):
    gi, gj = divmod(kk, n2)
    nb = []
    if gi > 0:
        nb.append(kk - n2)
    if gj > 0:
        nb.append(kk - 1)
    if nb:
        wv = np.linalg.solve(S2[np.ix_(nb, nb)], S2[nb, kk])
        Phi_v[kk, nb] = wv
        d2_v[kk] = S2[kk, kk] - S2[kk, nb] @ wv
        krr = N2 - 1 - kk                          # ic0 coefficients via mirror
        wic = np.array([-Lic[krr + (kk - m), krr] / Lic[krr, krr] for m in nb])
        dev_coef.append(float(np.max(np.abs(wv - wic))))
    else:
        d2_v[kk] = S2[kk, kk]
T_v = np.eye(N2) - Phi_v
M_vec = T_v.T @ (T_v / d2_v[:, None])
max_dev_iv = max(dev_coef)
w_vec_10 = Phi_v[k5, [k5 - 1, k5 - n2]]
print(f"  Vecchia (exact-Sigma, same kept pattern) at node 10: "
      f"W = {w_vec_10[0]:.4f}, prev-row = {w_vec_10[1]:.4f}")
ok(f"IC(0) vs Vecchia: same order but measurably different on the grid, max "
   f"coefficient dev = {max_dev_iv:.3e} (report 11 measured 8.3e-2 at n=8; "
   f"same phenomenon at n=4)", 1e-6 < max_dev_iv < 0.15)

def spec_precond(M, Amat):
    """Generalized eigenvalues of M^{-1} A via the SPD square root of M^{-1}."""
    R = np.linalg.cholesky(np.linalg.inv(M))
    return np.linalg.eigvalsh(R.T @ Amat @ R)

w_ic_spec = spec_precond(M_ic, A2d)
w_v_spec = spec_precond(M_vec, A2d)
rho_ic = float(max(abs(1 - w_ic_spec[0]), abs(1 - w_ic_spec[-1])))
kap_ic = float(w_ic_spec[-1] / w_ic_spec[0])
rho_v = float(max(abs(1 - w_v_spec[0]), abs(1 - w_v_spec[-1])))
kap_v = float(w_v_spec[-1] / w_v_spec[0])
print(f"  IC(0):   spec(M^-1 A) = [{w_ic_spec[0]:.4f}, {w_ic_spec[-1]:.4f}], "
      f"rho(I - M^-1 A) = {rho_ic:.4f}, kappa = {kap_ic:.4f}")
print(f"  Vecchia: spec(M^-1 A) = [{w_v_spec[0]:.4f}, {w_v_spec[-1]:.4f}], "
      f"rho(I - M^-1 A) = {rho_v:.4f}, kappa = {kap_v:.4f}")
ok(f"both truncated predictors precondition: kappa {kap_ic:.3f} (IC0) and "
   f"{kap_v:.3f} (Vecchia) << kappa(A2) = {kappa_A2:.3f}",
   kap_ic < 0.25 * kappa_A2 and kap_v < 0.25 * kappa_A2)
ok(f"Vecchia (KL-optimal on the pattern, Schafer et al. 2021) beats IC(0) "
   f"here: rho {rho_v:.4f} < {rho_ic:.4f}, kappa {kap_v:.4f} < {kap_ic:.4f}",
   rho_v < rho_ic and kap_v < kap_ic)
RESULTS["step5"] = {
    "n": n2, "N": N2, "h": h2, "kappa_A2": float(kappa_A2),
    "node": {"flat": k5, "grid": [2, 2]},
    "phiL2R_row_exact": w_exact.tolist(), "d2_exact": float(d2_exact),
    "ic0_kept": {"W": float(w_ic_W), "prev_row": float(w_ic_S)},
    "vecchia_kept": {"W": float(w_vec_10[0]), "prev_row": float(w_vec_10[1])},
    "dropped_l2_mass": drop_mass, "dropped_l2_rel": drop_rel,
    "ic_vs_vecchia_max_coef_dev": max_dev_iv,
    "rho_ic0": rho_ic, "kappa_ic0": kap_ic,
    "rho_vecchia": rho_v, "kappa_vecchia": kap_v,
    "report11_n8_dev": 8.3e-2,
}

# ===========================================================================
# STEP 6 -- approximate prediction by SCALE
# ===========================================================================
print("\n== STEP 6: prediction by scale: global mean (chain, exact), blocks (2D) ==")
ones5 = [[Fr(1)] for _ in range(n)]
cov_ubar = [sum(S[i][j] for j in range(n)) / n for i in range(n)]   # Cov(u_i, ubar)
var_ubar = sum(sum(row) for row in S) / n**2                        # Var(ubar)
beta = [c / var_ubar for c in cov_ubar]
fprint("beta_i = Cov(u_i, ubar)/Var(ubar) (exact)", [beta])
ok("beta = (5/7, 8/7, 9/7, 8/7, 5/7) exactly: bridge-shaped (max at center, "
   "symmetric), and sum(beta) = n = 5",
   beta == [Fr(5, 7), Fr(8, 7), Fr(9, 7), Fr(8, 7), Fr(5, 7)]
   and sum(beta) == n)
ok("Var(ubar) = 7/360 and Cov(u_i, ubar) = i(6-i)/360 exactly",
   var_ubar == Fr(7, 360)
   and cov_ubar == [Fr((i + 1) * (6 - (i + 1)), 360) for i in range(n)])
S_res = [[S[i][j] - cov_ubar[i] * cov_ubar[j] / var_ubar for j in range(n)]
         for i in range(n)]
fprint("residual covariance Sigma - beta Cov(u, ubar)^T (exact)", S_res)
tr_S = sum(S[i][i] for i in range(n))
tr_res = sum(S_res[i][i] for i in range(n))
ve_chain = 1 - tr_res / tr_S
ok(f"variance explained by the single global-mean regressor = 1 - "
   f"tr(res)/tr(Sigma) = {ve_chain} = {float(ve_chain):.4f} exactly (111/175)",
   ve_chain == Fr(111, 175))
ok("residual covariance is singular in the ubar direction: rows of the "
   "residual sum to 0 exactly (the mean is fully predicted)",
   all(sum(row) == 0 for row in S_res))

# 2D N=16 with 2x2 block averages
Z2 = block_average_matrix(n2, 2)                   # 16 x 4
ones16 = np.ones(N2)
cb = S2 @ ones16 / N2
vb = ones16 @ S2 @ ones16 / N2**2
ve_glob2d = 1 - np.trace(S2 - np.outer(cb, cb) / vb) / np.trace(S2)
cross = S2 @ Z2
S2_res = S2 - cross @ np.linalg.solve(Z2.T @ S2 @ Z2, cross.T)
ve_blk2d = 1 - np.trace(S2_res) / np.trace(S2)
print(f"  2D variance explained: global mean = {ve_glob2d:.4f}, "
      f"2x2 block averages (4 regressors) = {ve_blk2d:.4f}")
ok(f"variance explained jumps with scale detail: {ve_glob2d:.3f} -> "
   f"{ve_blk2d:.3f} (report 11 at n=8: 0.152 -> 0.567; same jump, smaller grid)",
   ve_blk2d > ve_glob2d > 0)

# two-level additive preconditioner M^{-1} = D^{-1} + Z Ac^{-1} Z^T
inv_diag2 = 1.0 / A2d.diagonal()
Ac = Z2.T @ A2d @ Z2                               # Galerkin coarse operator
C0 = np.diag(inv_diag2) + Z2 @ np.linalg.solve(Ac, Z2.T)
mu = spec_precond(np.linalg.inv(C0), A2d)          # spec(C0 A)
kap_2l = float(mu[-1] / mu[0])
theta = 2.0 / (mu[0] + mu[-1])                     # optimal damping
rho_2l = float((mu[-1] - mu[0]) / (mu[-1] + mu[0]))
print(f"  two-level: spec(C0 A) = [{mu[0]:.4f}, {mu[-1]:.4f}], kappa = "
      f"{kap_2l:.4f}; damped theta = {theta:.4f} -> rho = {rho_2l:.4f}")
ok(f"two-level M^{{-1}} = D^{{-1}} + Z Ac^{{-1}} Z^T preconditions: "
   f"kappa(C0 A) = {kap_2l:.3f} < kappa(A2) = {kappa_A2:.3f}, and with optimal "
   f"damping theta = {theta:.4f} the Richardson factor rho = {rho_2l:.4f} < 1",
   kap_2l < kappa_A2 and rho_2l < 1)
RESULTS["step6"] = {
    "beta_chain": fjson(beta), "var_ubar": str(var_ubar),
    "cov_ubar": fjson(cov_ubar), "residual_cov_chain": fjson(S_res),
    "ve_chain_exact": str(ve_chain), "ve_chain_float": float(ve_chain),
    "ve_2d_global": float(ve_glob2d), "ve_2d_2x2blocks": float(ve_blk2d),
    "twolevel": {"mu_min": float(mu[0]), "mu_max": float(mu[-1]),
                 "kappa": kap_2l, "theta": float(theta), "rho": rho_2l},
}

# ===========================================================================
# STEP 7 -- residual correction (Richardson)
# ===========================================================================
print("\n== STEP 7: residual correction x <- x + M^{-1}(b - A x), 2D N=16 ==")
b2 = np.zeros(N2)
b2[0] = 1.0                                        # hot corner (1,1) 1-based
b2[5] = -1.0                                       # cold (2,2) 1-based
xstar2 = np.linalg.solve(A2d, b2)
info("b = +1 at corner node (1,1) [flat 0], -1 at node (2,2) [flat 5] "
     "(1-based grid labels; a hot/cold source pair on the 4x4 grid)")

Licinv = np.linalg.inv(Lic)
apply_j = lambda r: inv_diag2 * r                              # noqa: E731
apply_ic = lambda r: Licinv.T @ (Licinv @ r)                   # noqa: E731
apply_2l = lambda r: theta * (C0 @ r)                          # noqa: E731
rho_jac = float(np.cos(np.pi * h2))
methods = [("jacobi", apply_j, rho_jac),
           ("ic0", apply_ic, rho_ic),
           ("two-level", apply_2l, rho_2l)]

def run_rich(applyC, maxit=400, tol=1e-12):
    xk = np.zeros(N2)
    e0n = np.linalg.norm(xstar2)
    hist = [1.0]
    for _ in range(maxit):
        xk = xk + applyC(b2 - A2d @ xk)
        hist.append(np.linalg.norm(xstar2 - xk) / e0n)
        if hist[-1] <= tol:
            break
    return np.array(hist)

HIST7 = {}
rows7 = {}
print("  per-iteration relative l2 error (first 8 iterations):")
print("    iter   jacobi      ic0         two-level")
for name, ap, _ in methods:
    HIST7[name] = run_rich(ap)
for k in range(1, 9):
    print(f"    {k:3d}   {HIST7['jacobi'][k]:.4e}  {HIST7['ic0'][k]:.4e}  "
          f"{HIST7['two-level'][k]:.4e}")
for name, ap, rho in methods:
    hh = HIST7[name]
    kend = len(hh) - 1
    w = min(20, kend // 2)
    slope = float((hh[kend] / hh[kend - w]) ** (1.0 / w))
    iters10 = int(np.argmax(hh <= 1e-10)) if np.any(hh <= 1e-10) else None
    rows7[name] = {"rho": float(rho), "tail_slope": slope,
                   "iters_to_1e-10": iters10,
                   "first8_rel_err": hh[1:9].tolist()}
    ok(f"{name}: measured tail rate {slope:.6f} == rho(I - M^-1 A) = {rho:.6f} "
       f"to 1% (report 12's per-sweep unexplained fraction, verified); "
       f"iters to 1e-10 = {iters10}", abs(slope - rho) / rho < 0.01)
ok("rate ladder matches prediction quality: rho_ic0 < rho_two-level < "
   f"rho_jacobi ({rho_ic:.3f} < {rho_2l:.3f} < {rho_jac:.3f})",
   rho_ic < rho_2l < rho_jac)
RESULTS["step7"] = {"b_hot": 0, "b_cold": 5, "methods": rows7}

# ===========================================================================
# STEP 8 -- CG interaction
# ===========================================================================
print("\n== STEP 8: CG vs PCG, and PCG == CG in whitened coordinates ==")
TOL8 = 1e-10
xh_cg = []
x_cg, rh_cg = pcg(A2, b2, None, tol=TOL8, x_hist=xh_cg)
xh_ic = []
x_pic, rh_ic = pcg(A2, b2, apply_ic, tol=TOL8, x_hist=xh_ic)
xh_2l = []
x_p2l, rh_2l = pcg(A2, b2, lambda r: C0 @ r, tol=TOL8, x_hist=xh_2l)
it_cg, it_ic, it_2l = len(rh_cg) - 1, len(rh_ic) - 1, len(rh_2l) - 1
print(f"  iterations to rel resid 1e-10: CG = {it_cg}, PCG(IC0) = {it_ic}, "
      f"PCG(two-level) = {it_2l}")
ok(f"PCG(two-level) converges in fewer iterations than CG ({it_2l} < {it_cg})",
   it_2l < it_cg)
info(f"PCG(IC0) = {it_ic} vs CG = {it_cg}: at N = 16 plain CG already "
     "finite-terminates on the few distinct eigenvalues of A; the kappa "
     "payoff of IC(0) shows at scale (reports 08/11), not on a 4x4 grid")

# THE claim made executable: PCG(M) trajectory == plain CG run on the
# split-preconditioned (whitened) system, mapped back through the factor.
def whitened_traj(Ahat, bhat, back):
    xh = []
    _, rh = pcg(Ahat, bhat, None, tol=TOL8, x_hist=xh)
    return [back @ xw for xw in xh], rh

nx2 = np.linalg.norm(xstar2)
# IC0: M = L L^T -> whitened system L^{-1} A L^{-T}, map back x = L^{-T} xhat
xw_ic, rhw_ic = whitened_traj(Licinv @ A2d @ Licinv.T, Licinv @ b2, Licinv.T)
m_ic = min(len(xh_ic), len(xw_ic))
dev_ic = max(np.linalg.norm(xh_ic[k] - xw_ic[k]) for k in range(m_ic)) / nx2
ok(f"PCG(IC0) trajectory == CG on L_ic^-1 A L_ic^-T mapped back through "
   f"L_ic^-T: max rel iterate dev over all {m_ic} iterates = {dev_ic:.1e} "
   f"(<= 1e-10), same iteration count ({len(rh_ic)-1} == {len(rhw_ic)-1})",
   dev_ic < 1e-10 and len(rh_ic) == len(rhw_ic))
# two-level: M^{-1} = C0 = R R^T -> whitened system R^T A R, map back x = R xhat
R0 = np.linalg.cholesky(C0)
xw_2l, rhw_2l = whitened_traj(R0.T @ A2d @ R0, R0.T @ b2, R0)
m_2l = min(len(xh_2l), len(xw_2l))
dev_2l = max(np.linalg.norm(xh_2l[k] - xw_2l[k]) for k in range(m_2l)) / nx2
ok(f"PCG(two-level) trajectory == CG on R^T A R mapped back through R "
   f"(M^-1 = C0 = R R^T): max rel iterate dev = {dev_2l:.1e} (<= 1e-10), "
   f"same iteration count ({len(rh_2l)-1} == {len(rhw_2l)-1})",
   dev_2l < 1e-10 and len(rh_2l) == len(rhw_2l))

# CG direction A-orthogonality: increments x_{k+1}-x_k are parallel to p_k
D_inc = [xh_cg[k + 1] - xh_cg[k] for k in range(len(xh_cg) - 1)]
D_inc = [d for d in D_inc if np.linalg.norm(d) > 1e-12 * nx2]
Gm = np.array([[di @ (A2d @ dj) / np.sqrt((di @ (A2d @ di)) * (dj @ (A2d @ dj)))
                for dj in D_inc] for di in D_inc])
offd = float(np.max(np.abs(Gm - np.diag(np.diag(Gm)))))
ok(f"CG search directions are A-orthogonal: max normalized |p_i^T A p_j|, "
   f"i != j, over {len(D_inc)} directions = {offd:.1e} (< 1e-8)", offd < 1e-8)
RESULTS["step8"] = {
    "iters": {"cg": it_cg, "pcg_ic0": it_ic, "pcg_twolevel": it_2l},
    "whitened_traj_dev": {"ic0": float(dev_ic), "twolevel": float(dev_2l)},
    "cg_A_orth_max_offdiag": offd,
    "res_hist": {"cg": rh_cg, "pcg_ic0": rh_ic, "pcg_twolevel": rh_2l},
}

# ===========================================================================
# figures
# ===========================================================================
print("\n== figures ==")

def annotate_frac(ax, M, Mf, fs=9):
    for i in range(len(M)):
        for j in range(len(M[0])):
            ax.text(j, i, str(M[i][j]), ha="center", va="center", fontsize=fs,
                    color="w" if abs(Mf[i, j]) > 0.55 * np.max(np.abs(Mf))
                    else "k")

fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))
panels = [("A = tridiag(-1,2,-1)/h^2  (h = 1/6)", A, 10),
          ("B = I - diag(A)^{-1} A  (two-sided predictor)", B, 10),
          ("Phi_L2R  (one-sided predictor: regress on predecessors)", PhiL, 10),
          ("Sigma = A^{-1}  (bridge covariance h x_i(1-x_j))", S, 9)]
for ax, (title, M, fs) in zip(axes.ravel(), panels):
    Mf = ffloat(M)
    ax.imshow(Mf, cmap="RdBu_r",
              vmin=-np.max(np.abs(Mf)), vmax=np.max(np.abs(Mf)))
    annotate_frac(ax, M, Mf, fs)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(n), [f"{j+1}" for j in range(n)])
    ax.set_yticks(range(n), [f"{i+1}" for i in range(n)])
fig.suptitle("Step 1-3, exact: the n = 5 chain (every entry an exact fraction)",
             fontsize=12)
fig.tight_layout()
fig.savefig(FIGDIR / "tutorial15_matrices.png", dpi=150)
plt.close(fig)
print("  wrote figures/tutorial15_matrices.png")

fig, ax = plt.subplots(figsize=(8.6, 4.6))
kk = np.arange(k5)
ax.bar(kk - 0.2, w_exact, width=0.4, color="#3465a4",
       label="exact one-sided regression (all 10 predecessors)")
kept = np.zeros(k5)
kept[k5 - 1] = w_ic_W
kept[k5 - n2] = w_ic_S
ax.bar(kk + 0.2, kept, width=0.4, color="#cc0000",
       label="IC(0) kept pattern (W + prev-row only, ic0 factor coefficients)")
ax.plot([k5 - 1, k5 - n2], w_vec_10, "k_", markersize=18, mew=2,
        label="Vecchia: exact regression on the kept pattern")
ax.axvspan(k5 - n2 - 0.5, k5 - 0.5, alpha=0.12, color="gray")
ax.text(k5 - n2 + 1.0, 0.28, "wavefront\n(last n = 4\npredecessors)",
        fontsize=8, ha="center")
ax.set_xticks(kk, [f"{m}\n({m//n2},{m%n2})" for m in kk], fontsize=8)
ax.set_xlabel("predecessor: flat index (grid row, col)")
ax.set_ylabel("prediction weight of node 10 = (2,2)")
ax.set_title(f"Step 5: truncating the one-sided predictor, 4x4 grid "
             f"(dropped l2 mass {drop_mass:.3f} = {100*drop_rel:.0f}% of the row)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "tutorial15_truncation.png", dpi=150)
plt.close(fig)
print("  wrote figures/tutorial15_truncation.png")

fig, ax = plt.subplots(figsize=(7.6, 5.2))
colors = {"jacobi": "#3465a4", "ic0": "#cc0000", "two-level": "#4e9a06"}
for name, _, rho in methods:
    hh = HIST7[name]
    ax.semilogy(range(len(hh)), hh, color=colors[name],
                label=f"{name}: measured (rho = {rho:.3f})")
    ks = np.arange(0, min(len(hh) + 10, 140))
    ax.semilogy(ks, hh[1] * rho ** (ks - 1.0), "--", color=colors[name],
                lw=0.9, alpha=0.6)
ax.set_ylim(1e-13, 2)
ax.set_xlabel("Richardson iteration  x <- x + M$^{-1}$(b - Ax)")
ax.set_ylabel("relative l2 error")
ax.set_title("Step 7: residual correction, 4x4 grid, hot/cold pair\n"
             "dashed = rho$^k$: the per-sweep unexplained fraction")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIGDIR / "tutorial15_residual_correction.png", dpi=150)
plt.close(fig)
print("  wrote figures/tutorial15_residual_correction.png")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))
for tag, xh, xw, col in [("PCG(IC0)", xh_ic, xw_ic, "#cc0000"),
                         ("PCG(two-level)", xh_2l, xw_2l, "#4e9a06")]:
    errs_p = [np.linalg.norm(v - xstar2) / nx2 for v in xh]
    errs_w = [np.linalg.norm(v - xstar2) / nx2 for v in xw]
    axL.semilogy(errs_p, "o", color=col, ms=7, mfc="none",
                 label=f"{tag}: pcg() iterates")
    axL.semilogy(errs_w, "-", color=col, lw=1.2,
                 label=f"{tag}: whitened CG mapped back")
axL.set_xlabel("iteration")
axL.set_ylabel("relative l2 error")
axL.set_title("trajectories coincide: PCG = CG in whitened coordinates")
axL.legend(fontsize=8)
for tag, xh, xw, col in [("PCG(IC0)", xh_ic, xw_ic, "#cc0000"),
                         ("PCG(two-level)", xh_2l, xw_2l, "#4e9a06")]:
    m = min(len(xh), len(xw))
    devs = [np.linalg.norm(xh[k] - xw[k]) / nx2 for k in range(m)]
    axR.semilogy(range(m), np.maximum(devs, 1e-18), "s-", color=col, ms=4,
                 label=tag)
axR.axhline(1e-10, color="k", ls=":", lw=1, label="1e-10 claim threshold")
axR.set_xlabel("iteration")
axR.set_ylabel(r"per-iterate deviation / $\Vert x^*\Vert$")
axR.set_title("iterate-by-iterate deviation (roundoff only)")
axR.legend(fontsize=8)
fig.suptitle("Step 8: the 'CG in the predictor's whitened coordinates' claim, "
             "executable", fontsize=11)
fig.tight_layout()
fig.savefig(FIGDIR / "tutorial15_whitened_cg.png", dpi=150)
plt.close(fig)
print("  wrote figures/tutorial15_whitened_cg.png")

# ===========================================================================
# save + summary
# ===========================================================================
RESDIR.mkdir(exist_ok=True)
with open(RESDIR / "prediction_tutorial.json", "w") as f:
    json.dump(jsonable(RESULTS), f, indent=1)
print(f"\nwrote results/prediction_tutorial.json")
print(f"SUMMARY: {N_PASS} PASS, {N_FAIL} FAIL")
if N_FAIL:
    sys.exit(1)
