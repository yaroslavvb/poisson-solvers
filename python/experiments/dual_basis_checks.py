"""Numerical verification of every claim in the dual-basis explainer.

The arena is the Dirichlet chain A = laplacian_1d(n)/h^2, n = 32, h = 1/33,
with the grounded incidence matrix B_inc ((n+1) x n, one row per edge
wall-1, 1-2, ..., n-wall, entries +1/h at the TAIL and -1/h at the HEAD,
Vishnoi's sign convention from mathematica/electrical_networks.wls).

NOTATION (the suite's B collision, resolved here once):
  B_inc = grounded incidence matrix (edge measurements),  A = B_inc^T B_inc
  B_reg = I - diag(A)^{-1} A = two-sided regression / Jacobi iteration matrix
          (reports 11/15's "B")

Part I  -- dual bases: A = Gram of sparse edge MEASUREMENTS, A^{-1} = Gram of
           dense edge RESPONSES X = B_inc A^{-1} = pinv(B_inc^T); rows of X are
           unit-dipole response fields; biorthogonality X^T B_inc = I; the
           cross-Gram X B_inc^T = Pi = I - 11^T/(n+1) (grounded: NOT the free
           Pi, which for a tree is the identity); leverage diag = n/(n+1) =
           conductance x effective resistance in the ground-as-a-node CYCLE;
           Maxwell 1864 unit-load assembly with random conductances;
           the continuum closed form X_{e,i} = h(x_i - 1[x_i >= x_e]);
           exact-fraction n = 8 pieces; the free path P8 gauge story.
Part II -- imperfect predictors, two families:
           (AR, whitener side) stationary AR(1) coefficient c approximating the
           exact nonstationary phi_i = (n+1-i)/(n+2-i): c-sweep, wall failure
           at c = 1 (rank-one, spectrum {1 x 31, 33}), quantized-phi flavor;
           (MA, colorer side) truncated Neumann M_k^{-1} = D^{-1} sum B_reg^j
           and clipped dipole responses M_w^{-1} = X_w^T X_w: k/w sweeps,
           SPD checks, kappa-vs-w scaling (the bridge no finite range can
           model), and the flops/span cost ledger in report 11 SS5.2's
           accounting (1 MAC = 2 flops, matvec = 2 nnz, vector op = 2N).

Run from the repo root:
    uv run python python/experiments/dual_basis_checks.py

Expected output: all PASS; writes results/dual_basis.json with every number,
curve, and matrix the report page needs.  Fully deterministic, < 60 s.
"""
import json
import time
from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poisson import laplacian_1d
from pcg import pcg

t0 = time.time()
np.set_printoptions(precision=6, suppress=True, linewidth=120)
results = {"checks": []}


def ok(name, cond, err=None):
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {name}" + (f"   (max err {err:.3g})" if err is not None else ""))
    results["checks"].append({"name": name, "pass": bool(cond),
                              **({"max_err": float(err)} if err is not None else {})})


# =============================================================================
# Part 0 -- the arena
# =============================================================================
n = 32
h = 1.0 / (n + 1)                       # 1/33
A = (laplacian_1d(n) / h**2).toarray()  # precision / stiffness matrix
S = np.linalg.inv(A)                    # covariance Sigma = A^{-1}
x = np.arange(1, n + 1) * h             # interior nodes x_i = i/33

# grounded incidence B_inc: (n+1) x n; edge e = 1..n+1 runs node (e-1) -> node e,
# nodes 0 and n+1 are the grounded walls (columns dropped);
# +1/h at the tail, -1/h at the head (Vishnoi convention).
B_inc = np.zeros((n + 1, n))
for e in range(1, n + 2):               # 1-based edge index
    if e - 1 >= 1:
        B_inc[e - 1, e - 2] = +1.0 / h  # tail node e-1
    if e <= n:
        B_inc[e - 1, e - 1] = -1.0 / h  # head node e

# =============================================================================
# Part I.A -- dual bases on the n = 32 chain
# =============================================================================
print("== I.A: dual bases (n = 32 grounded chain) ==")

# A1. precision = Gram of the sparse edge measurements
err = np.abs(B_inc.T @ B_inc - A).max() / np.abs(A).max()
ok("A == B_inc^T B_inc: precision = Gram of sparse edge measurements "
   "(2 nonzeros per interior row, 1 per wall row)", err < 1e-14, err)

# A2. the response matrix X: dense dual of the sparse measurement rows
X = B_inc @ S                            # (n+1) x n
err = np.abs(X - np.linalg.pinv(B_inc.T)).max()
ok("X = B_inc A^{-1} == pinv(B_inc^T): responses are the pseudoinverse dual "
   "of the measurements", err < 1e-12, err)

# A3. covariance = Gram of the dense edge responses
err = np.abs(X.T @ X - S).max() / np.abs(S).max()
ok("A^{-1} == X^T X: covariance = Gram of dense edge responses", err < 1e-12, err)

# A4. row e of X = global response to the unit dipole across edge e
err = max(np.abs(X[e] - np.linalg.solve(A, B_inc[e])).max() for e in range(n + 1))
ok("every row e of X == solve(A, b_e), b_e = e-th measurement row "
   "(unit dipole injected across edge e; all 33 rows)", err < 1e-12, err)

# A5. biorthogonality: measurements and responses are reciprocal coordinates
err = np.abs(X.T @ B_inc - np.eye(n)).max()
ok("biorthogonality X^T B_inc == I_32 (dual bases)", err < 1e-12, err)

# A6-A9. the cross-Gram Pi = X B_inc^T on edge space
Pi = X @ B_inc.T                         # (n+1) x (n+1)
err = np.abs(Pi - Pi.T).max()
ok("Pi = X B_inc^T symmetric", err < 1e-12, err)
err = np.abs(Pi @ Pi - Pi).max()
ok("Pi idempotent (Pi^2 == Pi): an orthogonal projection on edge space",
   err < 1e-12, err)
evPi = np.sort(sla.eigvalsh(Pi))
cond = (np.abs(evPi[0]) < 1e-12 and np.abs(evPi[1:] - 1).max() < 1e-12)
ok("spec(Pi) == {0 (x1), 1 (x32)}: rank n = 32, corank 1", cond,
   max(abs(evPi[0]), np.abs(evPi[1:] - 1).max()))
J = np.ones((n + 1, n + 1)) / (n + 1)
err = np.abs(Pi - (np.eye(n + 1) - J)).max()
ok("GROUNDED resolution: Pi == I_33 - 11^T/33 exactly; null vector = the "
   "constant current (a uniform loop through ground)", err < 1e-12, err)

# A10. leverage scores: constant diagonal n/(n+1)
err = np.abs(np.diag(Pi) - n / (n + 1)).max()
ok("diag(Pi) == 32/33 for every edge: the leverage score of each "
   "edge-datapoint; sum = 32 = rank (grounded Foster)", err < 1e-12, err)

# A11. physical reading: grounding = shorting both walls into ONE ground node,
# turning the path into a CYCLE on n+1 vertices; Pi_ee = c_e * Reff(cycle edge).
m = n + 1                                # cycle: nodes 1..n plus ground
L_cyc = np.zeros((m, m))
edges_cyc = [(m - 1, 0)] + [(i, i + 1) for i in range(n - 1)] + [(n - 1, m - 1)]
for (u, v) in edges_cyc:
    c_e = 1.0 / h**2
    L_cyc[u, u] += c_e; L_cyc[v, v] += c_e
    L_cyc[u, v] -= c_e; L_cyc[v, u] -= c_e
Lp_cyc = np.linalg.pinv(L_cyc)
errs = []
for (u, v) in edges_cyc:
    d = np.zeros(m); d[u] = 1; d[v] = -1
    reff = d @ Lp_cyc @ d
    errs.append(abs((1.0 / h**2) * reff - n / (n + 1)))
err = max(errs)
ok("diag(Pi) == conductance x effective resistance in the ground-as-a-node "
   "CYCLE network (all 33 edges)", err < 1e-10, err)

# A12. NOT the free-graph Pi: the free path (walls as real vertices) is a TREE,
# its cut space is all of edge space, so Pi_free == identity.
mF = n + 2                               # 34 vertices: wall, 1..32, wall
B_full = np.zeros((n + 1, mF))
for e in range(n + 1):                   # edge e+1: vertex e -> vertex e+1
    B_full[e, e] = +1.0 / h
    B_full[e, e + 1] = -1.0 / h
L_full = B_full.T @ B_full
Pi_free = B_full @ np.linalg.pinv(L_full) @ B_full.T
err = np.abs(Pi_free - np.eye(n + 1)).max()
ok("free-path contrast: B_full L_full^+ B_full^T == I_33 (tree => no cycles "
   "to project out); grounding removes exactly rank one", err < 1e-10, err)

# A13. continuum identity: the dipole row is a tilted step
# X_{e,i} = h (x_i - 1[x_i >= x_e]),   x_e = e h = head node of edge e.
X_closed = np.empty_like(X)
for e in range(1, n + 2):
    for i in range(1, n + 1):
        X_closed[e - 1, i - 1] = h * (x[i - 1] - (1.0 if i >= e else 0.0))
err = np.abs(X - X_closed).max()
ok("continuum closed form: X_{e,i} == h (x_i - 1[x_i >= x_e]) for all 33x32 "
   "entries (tail-head orientation; the guessed h(1_{x>s} - x) is its negative)",
   err < 1e-12, err)
# ... and it is literally the difference of adjacent Green tents
Spad = np.hstack([np.zeros((n, 1)), S, np.zeros((n, 1))])   # wall columns = 0
X_tents = (Spad[:, 0:n + 1] - Spad[:, 1:n + 2]).T / h
err = np.abs(X - X_tents).max()
ok("dipole row e == (tent at s = x_{e-1} minus tent at s = x_e)/h "
   "(adjacent columns of A^{-1}, walls padded 0)", err < 1e-12, err)

# A14-A15. Maxwell 1864 / Mohr / Castigliano unit-load assembly, weighted chain
rng = np.random.default_rng(42)
c_w = rng.uniform(0.5, 2.0, n + 1)       # random positive conductances
B0 = np.sign(B_inc) * 1.0                # pure +-1 graph incidence
A_w = B0.T @ np.diag(c_w) @ B0
S_w = np.linalg.inv(A_w)
V = S_w                                  # column i = potentials, unit inject at i
I_all = np.diag(c_w) @ B0 @ V            # I_all[e, i] = current on edge e, inject i
S_assembled = I_all.T @ np.diag(1.0 / c_w) @ I_all
err = np.abs(S_assembled - S_w).max() / np.abs(S_w).max()
ok("unit-load assembly (Maxwell 1864): (A^{-1})_ij == sum_e r_e I_e^(i) I_e^(j), "
   "random conductances U(0.5,2), seed 42, all 32x32 entries", err < 1e-11, err)
err = np.abs(S_w - S_w.T).max()
ok("Maxwell-Betti reciprocity == symmetry of the response Gram", err < 1e-12, err)

# =============================================================================
# Part I.B -- exact rational pieces at n = 8 (printable matrices)
# =============================================================================
print("\n== I.B: exact fractions at n = 8 ==")
n8 = 8
h8 = Fraction(1, 9)
A8 = [[Fraction(81) * (2 if i == j else (-1 if abs(i - j) == 1 else 0))
       for j in range(n8)] for i in range(n8)]
S8 = [[h8 * Fraction(min(i + 1, j + 1), 9) * (1 - Fraction(max(i + 1, j + 1), 9))
       for j in range(n8)] for i in range(n8)]
B8 = [[Fraction(0)] * n8 for _ in range(n8 + 1)]
for e in range(1, n8 + 2):
    if e - 1 >= 1:
        B8[e - 1][e - 2] = Fraction(9)
    if e <= n8:
        B8[e - 1][e - 1] = Fraction(-9)


def fmat(P, Q):
    return [[sum(P[i][k] * Q[k][j] for k in range(len(Q)))
             for j in range(len(Q[0]))] for i in range(len(P))]


I8 = [[Fraction(int(i == j)) for j in range(n8)] for i in range(n8)]
ok("n=8: A Sigma == I entry-by-entry in Fraction arithmetic, zero tolerance",
   fmat(A8, S8) == I8)

X8 = fmat(B8, S8)                        # exact X = B_inc A^{-1}
X8_closed = [[Fraction((i) - 9 * (1 if i >= e else 0), 81)
              for i in range(1, n8 + 1)] for e in range(1, n8 + 2)]
ok("n=8: X = B_inc A^{-1} exact == closed form (i - 9*1[i>=e])/81, "
   "all 9x8 entries in Fractions", X8 == X8_closed)
X8T = [list(r) for r in zip(*X8)]
ok("n=8: X^T B_inc == I_8 EXACTLY in Fractions (biorthogonality, zero tolerance)",
   fmat(X8T, B8) == I8)

# worked unit-load entry: (A^{-1})_{3,6} for unit conductances c_e = 1/h^2 = 81
c8 = Fraction(81); r8 = Fraction(1, 81)
v3 = [S8[i][2] for i in range(n8)]       # unit injection at node 3
v6 = [S8[i][5] for i in range(n8)]
def currents(v):
    vp = [Fraction(0)] + v + [Fraction(0)]
    return [c8 * (vp[e - 1] - vp[e]) for e in range(1, n8 + 2)]
I3, I6 = currents(v3), currents(v6)
assembled_36 = sum(r8 * I3[e] * I6[e] for e in range(n8 + 1))
ok("n=8 worked entry: sum_e r_e I_e^(3) I_e^(6) == (A^{-1})_{3,6} == 1/81 exactly "
   "(currents -2/3,+1/3 and -1/3,+2/3; 9 edges)",
   assembled_36 == S8[2][5] == Fraction(1, 81))

# =============================================================================
# Part I.C -- the free path P8 (gauge-fixed version, ties to Vishnoi wls)
# =============================================================================
print("\n== I.C: free path P8 ==")
nv, ne = 8, 7
Bf = np.zeros((ne, nv))
for e in range(ne):
    Bf[e, e] = 1.0; Bf[e, e + 1] = -1.0  # +1 tail / -1 head
Lf = Bf.T @ Bf
Lfp = np.linalg.pinv(Lf)
Xf = Bf @ Lfp
err = np.abs(Xf - np.linalg.pinv(Bf.T)).max()
ok("P8: X_free = B L^+ == pinv(B^T)", err < 1e-12, err)
err = np.abs(Xf.T @ Xf - Lfp).max()
ok("P8: X_free^T X_free == L^+ (covariance Gram, free version)", err < 1e-12, err)
err = max(np.abs(Xf[e] - Lfp @ Bf[e]).max() for e in range(ne))
err = max(err, np.abs(Xf.sum(axis=1)).max())
ok("P8: rows of X_free are gauge-fixed dipole fields: row e == L^+ b_e, "
   "each summing to 0 (perp to the constant gauge mode)", err < 1e-12, err)
Pif = Bf @ Lfp @ Bf.T
err = np.abs(Pif - np.eye(ne)).max()
ok("P8: Pi_free == I_7; diag = effective resistances = 1 per tree edge, "
   "Foster sum = 7 = vertices - 1", err < 1e-12, err)
err = np.abs(Xf.T @ Bf - (np.eye(nv) - np.ones((nv, nv)) / nv)).max()
ok("P8 honest note: X_free^T B == I - 11^T/8 (the CENTERING projector, not I: "
   "biorthogonality only modulo the gauge)", err < 1e-12, err)

# =============================================================================
# Part II -- imperfect predictors.  Shared machinery.
# =============================================================================
sqrtA = None
def eig_MinvA_from_Minv(Minv):
    """Spectrum of M^{-1}A via the similar SPD matrix A^{1/2} M^{-1} A^{1/2}."""
    global sqrtA
    if sqrtA is None:
        w, Q = np.linalg.eigh(A)
        sqrtA = Q @ np.diag(np.sqrt(w)) @ Q.T
    return np.sort(sla.eigvalsh(sqrtA @ Minv @ sqrtA))


def stats(lam):
    lmin, lmax = float(lam[0]), float(lam[-1])
    kappa = lmax / lmin
    return {"lam_min": lmin, "lam_max": lmax, "kappa": kappa,
            "rho_richardson_opt": (kappa - 1) / (kappa + 1),
            "rho_richardson_plain": float(np.abs(1 - lam).max())}


rng_b = np.random.default_rng(2026)
b_rhs = rng_b.standard_normal(n)         # one fixed RHS for every PCG run
def pcg_iters(Mfun):
    _, hist = pcg(A, b_rhs, M=Mfun, tol=1e-10, maxiter=5000)
    return len(hist) - 1, hist[-1]


nnz_A = 3 * n - 2                        # 94
FL = {"vec": 2 * n, "matvecA": 2 * nnz_A}
def flops_total(apply_flops, iters):
    """Report 11 SS5.2 convention: 1 MAC = 2 flops; per iteration 12N vector
    flops + one A-matvec (2 nnz A) + one M-apply; setup 4N + one M-apply."""
    per_iter = 12 * n + FL["matvecA"] + apply_flops
    return per_iter, 4 * n + apply_flops + iters * per_iter


# =============================================================================
# Part II.A -- AR side: stationary whitener c vs exact phi_i = (n+1-i)/(n+2-i)
# =============================================================================
print("\n== II.A: AR side (imperfect whitener) ==")

phi = np.array([(n + 1 - i) / (n + 2 - i) for i in range(1, n + 1)])  # i=1..n
phi_reg = np.array([S[i, i - 1] / S[i - 1, i - 1] for i in range(1, n)])
err = np.abs(phi[1:] - phi_reg).max()
d2 = h**2 * phi                          # d_i^2 = h^2 phi_i, valid i = 1..n
d2_reg = np.array([S[0, 0]] + [S[i, i] - phi[i] * S[i, i - 1] for i in range(1, n)])
err = max(err, np.abs(d2 - d2_reg).max() / d2.min())
ok("exact L2R predictor: phi_i == (n+1-i)/(n+2-i) == Sigma_{i,i-1}/Sigma_{i-1,i-1} "
   "and d_i^2 == h^2 phi_i (formula extends to i=1: d_1^2 = Sigma_11)",
   err < 1e-10, err)

Phi = np.zeros((n, n))
for i in range(1, n):
    Phi[i, i - 1] = phi[i]
W_exact = np.eye(n) - Phi
M_exact = W_exact.T @ np.diag(1.0 / d2) @ W_exact
err = np.abs(M_exact - A).max() / np.abs(A).max()
it_exact, _ = pcg_iters(lambda r, W=W_exact, d=d2:
                        sla.solve_triangular(W, d * sla.solve_triangular(
                            W.T, r, lower=False), lower=True))
ok(f"perfect whitener: (I-Phi)^T D^{{-2}} (I-Phi) == A and PCG converges in "
   f"{it_exact} iteration", err < 1e-12 and it_exact == 1, err)

Ssub = np.zeros((n, n))                  # the shift: row i regresses on i-1
for i in range(1, n):
    Ssub[i, i - 1] = 1.0

def M_stationary(c, variant):
    """M_c = (I - c S)^T D^{-1} (I - c S).
    variant 'const':   D = d_c^2 I with d_c^2 = h^2 (1+c^2)/2, chosen so the
                       interior diagonal of M_c matches diag(A) = 2/h^2.
                       Any scalar works: kappa/PCG/rho_opt are invariant.
    variant 'matched': D_ii = Var_Sigma(u_i - c u_{i-1}), the TRUE residual
                       variance of the imperfect coefficient under Sigma
                       (node 1 predicts from the wall value 0)."""
    W = np.eye(n) - c * Ssub
    if variant == "const":
        D = np.full(n, h**2 * (1 + c**2) / 2.0)
    else:
        D = np.array([S[0, 0]] + [S[i, i] - 2 * c * S[i, i - 1]
                                  + c**2 * S[i - 1, i - 1] for i in range(1, n)])
    return W, D, W.T @ np.diag(1.0 / D) @ W


c_grid = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
ar_rows = []
for c in c_grid:
    row = {"c": c}
    for variant in ("const", "matched"):
        W, D, M = M_stationary(c, variant)
        lam = np.sort(sla.eigh(A, M, eigvals_only=True))   # eig(M^{-1}A)
        st = stats(lam)
        iters, _ = pcg_iters(lambda r, W=W, D=D:
                             sla.solve_triangular(W, D * sla.solve_triangular(
                                 W.T, r, lower=False), lower=True))
        st["pcg_iters"] = iters
        st["min_eig_M"] = float(np.sort(sla.eigvalsh(M))[0])
        st["spectrum"] = lam.tolist()
        row[variant] = st
    ar_rows.append(row)

# scalar-D invariance (documented, then machine-checked at c = 0.8)
W, D, M = M_stationary(0.8, "const")
lam1 = np.sort(sla.eigh(A, M, eigvals_only=True))
lam2 = np.sort(sla.eigh(A, 7.0 * M, eigvals_only=True)) * 7.0
err = np.abs(lam1 / lam1[0] - lam2 / lam2[0]).max()
ok("const-D honesty: the scalar d_c^2 drops out of kappa/PCG/rho_opt "
   "(checked: spectrum shape invariant under D -> 7D at c = 0.8)", err < 1e-10, err)

# c = 1 wall failure: random-walk prior, rank-one defect at the right wall
W1, D1, M1 = M_stationary(1.0, "const")  # d^2 = h^2 at c = 1
err = np.abs(M1 - (A - np.outer(np.eye(n)[-1], np.eye(n)[-1]) / h**2)).max() \
    / np.abs(A).max()
lam1 = np.sort(sla.eigh(A, M1, eigvals_only=True))
spec_err = max(np.abs(lam1[:-1] - 1).max(), abs(lam1[-1] - (n + 1)))
it_c1 = next(r for r in ar_rows if r["c"] == 1.0)["const"]["pcg_iters"]
rho_plain_c1 = next(r for r in ar_rows if r["c"] == 1.0)["const"]["rho_richardson_plain"]
ok(f"c=1 (random-walk prior) fails at the wall by RANK ONE: M_1 == A - e_n e_n^T/h^2 "
   f"(the whitener never learns the right wall), spec(M^{{-1}}A) == {{1 x 31, 33}}, "
   f"PCG shrugs it off in {it_c1} iterations, plain Richardson DIVERGES "
   f"(rho = {rho_plain_c1:.0f})",
   err < 1e-12 and spec_err < 1e-9 and it_c1 == 2 and rho_plain_c1 > 30,
   max(err, spec_err))

# fine c-grid for the slider / optimum
c_fine = np.linspace(0.0, 1.0, 201)
kfine = {"const": [], "matched": []}
for c in c_fine:
    for variant in ("const", "matched"):
        _, _, M = M_stationary(c, variant)
        lam = np.sort(sla.eigh(A, M, eigvals_only=True))
        kfine[variant].append(float(lam[-1] / lam[0]))
opt = {v: {"c_opt": float(c_fine[int(np.argmin(kfine[v]))]),
           "kappa_opt": float(min(kfine[v]))} for v in kfine}
ok(f"stationary c-sweep computed: optimum c* = {opt['const']['c_opt']:.3f} "
   f"(kappa = {opt['const']['kappa_opt']:.2f}) const-D; "
   f"c* = {opt['matched']['c_opt']:.3f} (kappa = {opt['matched']['kappa_opt']:.2f}) "
   f"matched-D; both beat Jacobi/c=0 (kappa = {ar_rows[0]['const']['kappa']:.1f})",
   opt["const"]["kappa_opt"] < ar_rows[0]["const"]["kappa"])

# second imperfection flavor: quantize the exact phi_i to bits b
quant_rows = []
for bits in (1, 2, 3, 4, 6, 8):
    phi_q = np.round(phi * 2**bits) / 2**bits
    phi_q[0] = 0.0                       # node 1 has no predecessor coefficient
    Dq = np.array([S[0, 0]] + [S[i, i] - 2 * phi_q[i] * S[i, i - 1]
                               + phi_q[i]**2 * S[i - 1, i - 1] for i in range(1, n)])
    Wq = np.eye(n)
    for i in range(1, n):
        Wq[i, i - 1] = -phi_q[i]
    Mq = Wq.T @ np.diag(1.0 / Dq) @ Wq
    lam = np.sort(sla.eigh(A, Mq, eigvals_only=True))
    st = stats(lam)
    iters, _ = pcg_iters(lambda r, W=Wq, D=Dq:
                         sla.solve_triangular(W, D * sla.solve_triangular(
                             W.T, r, lower=False), lower=True))
    st["pcg_iters"] = iters
    quant_rows.append({"bits": bits, **st})
ok(f"quantized-phi flavor: rounding the exact coefficients to b bits "
   f"(matched variances); kappa falls {quant_rows[0]['kappa']:.2f} -> "
   f"{quant_rows[-1]['kappa']:.6f} from 1 to 8 bits, monotone",
   all(quant_rows[i]["kappa"] >= quant_rows[i + 1]["kappa"] - 1e-9
       for i in range(len(quant_rows) - 1)))

# AR cost model (report 11 convention)
nnz_W = 2 * n - 1                        # unit diagonal + n-1 subdiagonal
ar_apply_flops = 2 * (2 * nnz_W) + 2 * n     # two triangular solves + diag scale
ar_span = 2 * (n - 1) + 1                # two sequential substitution chains

# =============================================================================
# Part II.B -- MA side: truncated Neumann walks and clipped dipole responses
# =============================================================================
print("\n== II.B: MA side (imperfect colorer) ==")

Dj = np.diag(A).copy()                   # = 2/h^2, constant
B_reg = np.eye(n) - A / Dj[0]            # I - D^{-1}A, scalar D
off = np.abs(B_reg - sp.diags([0.5 * np.ones(n - 1), 0.5 * np.ones(n - 1)],
                              [-1, 1]).toarray()).max()
rho_B = float(np.abs(np.linalg.eigvalsh(B_reg)).max())
err = max(off, abs(rho_B - np.cos(np.pi / (n + 1))))
ok("B_reg = I - diag(A)^{-1}A == tridiag(1/2, 0, 1/2) (the half-step walk "
   "matrix; scalar D makes 11 SS2's two orderings equal), "
   "rho(B_reg) == cos(pi/33) = 0.995472", err < 1e-12, err)

# Neumann partial sums -> A^{-1} at the walk rate (spectral, exact)
mu = np.linalg.eigvalsh(B_reg)
Ks = np.arange(0, 201)
neu_err = []
for K in Ks:
    lam_MK = (h**2 / 2) * (1 - mu**(K + 1)) / (1 - mu)
    lam_S = (h**2 / 2) / (1 - mu)
    neu_err.append(float(np.abs(lam_MK - lam_S).max() / lam_S.max()))
slope = np.polyfit(Ks[50:], np.log(neu_err[50:]), 1)[0]
err = abs(slope - np.log(rho_B))
ok(f"Neumann/walk series: partial sums D^{{-1}} sum B_reg^j -> A^{{-1}}, "
   f"fitted rate e^{{{slope:.6f}}} == rho(B_reg) = {rho_B:.6f} "
   f"(long walks carry the far field)", err < 1e-4, err)

nnz_Breg = 2 * (n - 1)
k_grid = [0, 1, 2, 3, 4, 8, 16, 32, 64]
Breg_sp = sp.csr_matrix(B_reg)
ma_k_rows = []
min_eigs_k = []
for k in k_grid:
    # dense M_k^{-1} for spectra / bandwidth; Horner callable for PCG
    Mk_inv = np.zeros((n, n)); P = np.eye(n)
    for j in range(k + 1):
        Mk_inv += P
        P = B_reg @ P
    Mk_inv *= h**2 / 2
    me = float(np.sort(sla.eigvalsh(Mk_inv))[0])
    min_eigs_k.append(me)
    lam = eig_MinvA_from_Minv(Mk_inv)
    st = stats(lam)

    def horner(r, k=k):
        z = r.copy()
        for _ in range(k):
            z = r + Breg_sp @ z
        return (h**2 / 2) * z
    iters, _ = pcg_iters(horner)
    bw = int(max(np.abs(i - j) for i, j in zip(*np.nonzero(np.abs(Mk_inv) > 1e-14))))
    apply_fl = 2 * n + k * (2 * nnz_Breg + 2 * n)   # D^{-1}r + k (matvec + axpy)
    per_iter, total = flops_total(apply_fl, iters)
    ma_k_rows.append({"k": k, "min_eig_Minv": me, **st, "pcg_iters": iters,
                      "bandwidth": bw, "nnz_Minv": int((np.abs(Mk_inv) > 1e-14).sum()),
                      "flops_apply": apply_fl, "flops_per_iter": per_iter,
                      "flops_total": total, "span_apply": 3 * k + 1,
                      "spectrum": lam.tolist()})
ok(f"truncated Neumann SPD for EVERY k tested including odd "
   f"(min eig over k-grid = {min(min_eigs_k):.3e} > 0): rho(B_reg) < 1 makes "
   f"1 - mu^(k+1) > 0 for all k -- the even-k-only safety rule is not needed "
   f"on this chain", min(min_eigs_k) > 0)
err = max(abs(r["bandwidth"] - min(r["k"], n - 1)) for r in ma_k_rows)
ok("bandwidth(M_k^{-1}) == min(k, n-1): the walk length IS the range of the "
   "covariance model", err == 0)

# clipped dipole responses: keep X_{e,i} only for e-1-w <= i <= e+w
w_grid = [0, 1, 2, 4, 8, 16, 24, 31]
ma_w_rows = []
clipped_examples = {}
for w in w_grid:
    Xw = np.zeros_like(X)
    for e in range(1, n + 2):
        lo, hi = max(1, e - 1 - w), min(n, e + w)
        Xw[e - 1, lo - 1:hi] = X[e - 1, lo - 1:hi]
    Mw_inv = Xw.T @ Xw
    me = float(np.sort(sla.eigvalsh(Mw_inv))[0])
    lam = eig_MinvA_from_Minv(Mw_inv)
    st = stats(lam)
    iters, _ = pcg_iters(lambda r, Xw=Xw: Xw.T @ (Xw @ r))
    nz = np.nonzero(np.abs(Mw_inv) > 1e-16)
    bw = int(max(np.abs(i - j) for i, j in zip(*nz)))
    nnz_Xw = int((np.abs(Xw) > 0).sum())
    apply_fl = 2 * (2 * nnz_Xw)                     # two sparse matvecs
    per_iter, total = flops_total(apply_fl, iters)
    ma_w_rows.append({"w": w, "min_eig_Minv": me, **st, "pcg_iters": iters,
                      "bandwidth": bw, "nnz_Xw": nnz_Xw,
                      "nnz_Minv": int((np.abs(Mw_inv) > 1e-16).sum()),
                      "flops_apply": apply_fl, "flops_per_iter": per_iter,
                      "flops_total": total,
                      "span_apply": 2 * (1 + int(np.ceil(np.log2(2 * w + 2)))),
                      "spectrum": lam.tolist()})
    if w in (2, 4, 8):
        clipped_examples[str(w)] = {"edge_1": Xw[0].tolist(),
                                    "edge_17": Xw[16].tolist(),
                                    "edge_33": Xw[32].tolist()}
ok(f"clipped responses M_w^{{-1}} = X_w^T X_w SPD by construction and "
   f"nonsingular at every w (min eig at w=0: {ma_w_rows[0]['min_eig_Minv']:.3e} > 0)",
   min(r["min_eig_Minv"] for r in ma_w_rows) > 0)
r31 = next(r for r in ma_w_rows if r["w"] == 31)
ok(f"w = 31 (full window) recovers the exact colorer: kappa = "
   f"{r31['kappa']:.12f}, PCG converges in {r31['pcg_iters']} iteration",
   abs(r31["kappa"] - 1) < 1e-9 and r31["pcg_iters"] == 1)
err = max(abs(r["bandwidth"] - min(2 * r["w"] + 1, n - 1)) for r in ma_w_rows)
ok("bandwidth(M_w^{-1}) == min(2w+1, n-1): window w => covariance range 2w+1",
   err == 0)

# the theorem-shaped observation: kappa(w) ~ (n/w)^2 -- no finite range models
# the bridge (report 14's separator, seen from the covariance side)
wfit = [r for r in ma_w_rows if 1 <= r["w"] <= 16]
sl, ic = np.polyfit(np.log([r["w"] for r in wfit]),
                    np.log([r["kappa"] for r in wfit]), 1)
ok(f"finite range cannot model the bridge: kappa(M_w^{{-1}}A) ~ w^({sl:.2f}) "
   f"over w in [1,16] (slope approx -2: halve the clipping, quadruple the "
   f"conditioning debt)", -2.6 < sl < -1.4)

npass = sum(c["pass"] for c in results["checks"])
print(f"\n{npass}/{len(results['checks'])} PASS   ({time.time()-t0:.1f} s)")

# =============================================================================
# tables to stdout
# =============================================================================
print("\n--- AR c-sweep (stationary whitener; const-D | matched-D) ---")
print(f"{'c':>5} | {'lmin':>8} {'lmax':>8} {'kappa':>9} {'rho_opt':>8} "
      f"{'rho_plain':>9} {'its':>4} | {'kappa_m':>9} {'its_m':>5}")
for r in ar_rows:
    a, b_ = r["const"], r["matched"]
    print(f"{r['c']:>5.2f} | {a['lam_min']:>8.4f} {a['lam_max']:>8.3f} "
          f"{a['kappa']:>9.3f} {a['rho_richardson_opt']:>8.5f} "
          f"{a['rho_richardson_plain']:>9.4f} {a['pcg_iters']:>4d} | "
          f"{b_['kappa']:>9.3f} {b_['pcg_iters']:>5d}")

print("\n--- MA k-sweep (truncated Neumann walks) ---")
print(f"{'k':>3} | {'bw':>3} {'minEig':>9} {'lmin':>8} {'lmax':>7} {'kappa':>8} "
      f"{'rho_opt':>8} {'its':>4} {'fl/apply':>8} {'fl total':>9}")
for r in ma_k_rows:
    print(f"{r['k']:>3d} | {r['bandwidth']:>3d} {r['min_eig_Minv']:>9.2e} "
          f"{r['lam_min']:>8.5f} {r['lam_max']:>7.4f} {r['kappa']:>8.2f} "
          f"{r['rho_richardson_opt']:>8.5f} {r['pcg_iters']:>4d} "
          f"{r['flops_apply']:>8d} {r['flops_total']:>9d}")

print("\n--- MA w-sweep (clipped dipole responses) ---")
print(f"{'w':>3} | {'bw':>3} {'nnzXw':>5} {'minEig':>9} {'lmin':>8} {'lmax':>7} "
      f"{'kappa':>9} {'rho_opt':>8} {'its':>4} {'fl/apply':>8} {'fl total':>9}")
for r in ma_w_rows:
    print(f"{r['w']:>3d} | {r['bandwidth']:>3d} {r['nnz_Xw']:>5d} "
          f"{r['min_eig_Minv']:>9.2e} {r['lam_min']:>8.5f} {r['lam_max']:>7.4f} "
          f"{r['kappa']:>9.3f} {r['rho_richardson_opt']:>8.5f} "
          f"{r['pcg_iters']:>4d} {r['flops_apply']:>8d} {r['flops_total']:>9d}")

print("\n--- quantized-phi flavor ---")
for r in quant_rows:
    print(f"bits={r['bits']}: kappa={r['kappa']:.6f}  iters={r['pcg_iters']}")

# =============================================================================
# JSON export -- every number / curve / matrix the page needs
# =============================================================================
frac = lambda M: [[str(v) for v in row] for row in M]
results["arena"] = {
    "n": n, "h": h, "nnz_A": nnz_A, "kappa_A": float(
        np.linalg.eigvalsh(A)[-1] / np.linalg.eigvalsh(A)[0]),
    "notation": "B_inc = grounded incidence ((n+1) x n, +1/h tail, -1/h head, "
                "Vishnoi); B_reg = I - diag(A)^{-1}A (reports 11/15's B)",
}
results["dual_basis_explorer"] = {
    "x_nodes": x.tolist(),
    "B_inc_rows": B_inc.tolist(),                   # 33 measurement rows
    "X_rows": X.tolist(),                           # 33 response rows
    "pi_diag": np.diag(Pi).tolist(),                # leverage / eff. resistance
    "pi_diag_value": n / (n + 1),
    "Pi_closed_form": "I_33 - (1/33) * ones(33,33)",
    "continuum_formula": "X[e,i] = h * (x_i - 1[x_i >= x_e]),  x_e = e*h "
                         "(tail-head orientation; negate for head-tail)",
    "continuum_note": "row e = (tent at x_{e-1} - tent at x_e)/h; equals h*x_i "
                      "left of edge e and h*(x_i - 1) right of it",
}
results["unit_load"] = {
    "conductances_seed42": c_w.tolist(),
    "max_rel_err_assembly": float(np.abs(S_assembled - S_w).max() / np.abs(S_w).max()),
    "worked_n8_entry": {
        "i": 3, "j": 6, "value": "1/81",
        "r_e": "1/81", "I3": [str(v) for v in I3], "I6": [str(v) for v in I6],
        "terms_r_I3_I6": [str(r8 * I3[e] * I6[e]) for e in range(n8 + 1)],
    },
}
results["printable_n8"] = {
    "h": "1/9", "X_closed_form": "X[e,i] = (i - 9*1[i>=e]) / 81",
    "X_rows_exact": frac(X8),
    "row_examples": {"edge_1": frac([X8[0]])[0], "edge_5": frac([X8[4]])[0],
                     "edge_9": frac([X8[8]])[0]},
    "XT_Binc_is_identity_exact": True,
    "A_Sigma_is_identity_exact": True,
}
results["free_path_P8"] = {
    "B": Bf.tolist(), "L_pinv": Lfp.tolist(), "X_free": Xf.tolist(),
    "Pi_free": "I_7 exactly (tree)", "XtB": "I - 11^T/8 (centering, gauge)",
    "reff_per_edge": np.diag(Pif).tolist(), "foster_sum": float(np.trace(Pif)),
}
results["ar_slider"] = {
    "exact_phi": phi.tolist(), "exact_d2": d2.tolist(),
    "phi_formula": "phi_i = (n+1-i)/(n+2-i),  d_i^2 = h^2 phi_i (i = 1..n)",
    "D_variants": {
        "const": "d_c^2 = h^2 (1+c^2)/2 (matches interior diag of A; any "
                 "scalar gives identical kappa/PCG/rho_opt)",
        "matched": "D_ii = Sigma_ii - 2c Sigma_{i,i-1} + c^2 Sigma_{i-1,i-1} "
                   "(true residual variance of the imperfect coefficient)"},
    "c_grid": ar_rows,
    "c_fine": c_fine.tolist(),
    "kappa_fine": kfine,
    "optimum": opt,
    "c1_story": {"identity": "M_1 = A - e_n e_n^T / h^2 (const-D, d^2 = h^2)",
                 "spectrum": "{1 x 31, n+1 = 33}", "pcg_iters": it_c1,
                 "rho_plain": rho_plain_c1,
                 "moral": "rank-one wall defect: fatal for plain Richardson, "
                          "two iterations for PCG"},
    "quantized": quant_rows,
    "cost": {"flops_apply": ar_apply_flops, "span_apply": ar_span,
             "flops_per_iter": flops_total(ar_apply_flops, 1)[0],
             "model": "two unit-diagonal bidiagonal solves at 2*nnz(W)=126 "
                      "flops each + diagonal scale 2n; span = two sequential "
                      "length-n substitution chains = 2(n-1)+1 = 63"},
}
results["ma_sliders"] = {
    "B_reg": "tridiag(1/2, 0, 1/2), rho = cos(pi/33) = %.6f" % rho_B,
    "neumann_convergence": {"K": Ks.tolist(), "rel_err": neu_err,
                            "fitted_log_rate": float(slope),
                            "log_rho": float(np.log(rho_B))},
    "k_grid": ma_k_rows,
    "w_grid": ma_w_rows,
    "window_def": "keep X[e,i] for e-1-w <= i <= e+w (w nodes beyond each "
                  "endpoint of edge e); w=0 keeps only the edge's endpoints",
    "clipped_examples": clipped_examples,
    "full_row_edge_17": X[16].tolist(),
    "kappa_vs_w_fit": {"slope": float(sl), "intercept": float(ic),
                       "range": "w in [1, 16]",
                       "law": "kappa ~ w^(%.2f): finite-range covariance "
                              "cannot capture the bridge (report 14's "
                              "separator, from the covariance side)" % sl},
    "cost_models": {
        "convention": "report 11 SS5.2 / 14 SS4 PASS 30: 1 MAC = 2 flops, "
                      "matvec = 2 nnz, vector op = 2N; PCG per iteration = "
                      "12N + one A-matvec (2*94) + one M-apply; setup 4N + "
                      "one M-apply; total = setup + iters * per-iter",
        "neumann": "apply = 2n (D^{-1}r) + k*(2*nnz(B_reg) + 2n) = 64 + 188k; "
                   "span = 3k+1 rounds (each walk step embarrassingly parallel)",
        "clipped": "apply = 2 * 2*nnz(X_w) (two sparse matvecs, X_w then "
                   "X_w^T); span = 2*(1 + ceil(log2(2w+2))) -- constant-depth, "
                   "embarrassingly parallel",
        "ar": "apply = 316 flops but span = 63: the update_views lesson -- "
              "triangular solves are flop-cheap and depth-expensive"},
}
results["summary"] = {"checks_total": len(results["checks"]),
                      "checks_passed": npass,
                      "runtime_s": round(time.time() - t0, 2),
                      "pcg_rhs": "standard normal, rng seed 2026, tol 1e-10"}

out = Path(__file__).resolve().parents[2] / "results" / "dual_basis.json"
out.write_text(json.dumps(results, indent=1))
print(f"\nwrote {out} ({out.stat().st_size/1024:.0f} KB)")
