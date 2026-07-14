"""Numerical + exact-rational verification of every claim in the grid
dual-basis explainer -- the 2-D sibling of dual_basis_explainer.html:
its 1-D measurement / response bases go to the FREE 7x7 grid graph.

Arena: 7x7 grid graph, n = 49 vertices, m = 84 edges, unit conductances.
Vertices row-major v = 7r + c (r, c in 0..6), widget coordinates (x, y) = (c, r).
ORIENTATION (fixed once, documented): every edge points from its LOWER vertex
index to its HIGHER one (DirectedGraph[..., "Acyclic"]-style lex order, the
mathematica/electrical_networks.wls convention); the edge list is sorted lex
by (tail, head), so each vertex's rightward edge (v, v+1) precedes its
downward edge (v, v+7).  B in R^{84 x 49} has +1 at the TAIL and -1 at the
HEAD of each edge (Vishnoi's sign convention).

Cast (one matrix per visualization mode):
  L        = B^T B      free vertex Laplacian (49 x 49), kernel = constants
  L^+                   pseudoinverse: free-grid Green function, mean-zero
                        gauge (grounding_explainer's pinv row, now 2-D)
  X        = B L^+       == pinv(B^T): the response matrix; rows = gauge-fixed
                        dipole fields (dual_basis' centered steps, now 2-D)
  L_edge   = B B^T       the EDGE LAPLACIAN -- the answer to "does B B^T have
                        a name?"  It is the down part of the first
                        combinatorial / Hodge Laplacian L_1 = B B^T (+ face /
                        curl terms, absent here because no 2-cells are
                        included).  Sign rule, line-graph identity, and the
                        shared nonzero spectrum with L are all checked below.
  Pi       = B L^+ B^T   Vishnoi's projection onto the cut space (Ch. 4);
                        diag = effective resistances = leverage scores =
                        spanning-tree marginals (Thm 4.5, exact Matrix-Tree)
Plus the random-walk sum: ground g, G_g = (L with row/col g deleted)^{-1} ==
sum_k (D^{-1} W)^k D^{-1} restricted off g, with the exact absorbing-chain
visit-count identity G_g[i,j] = E[# visits to j from i before hitting g]/d_j
verified in Fraction arithmetic (no Monte Carlo).

Checks 1-10 mirror the page's ten click-modes; every matrix / field / curve
the widgets need goes to results/grid_viz.json (floats at 6 sig digits).

Run from the repo root:
    uv run python python/experiments/grid_viz_checks.py

Expected output: all PASS; < 60 s; JSON < 4 MB.
"""
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

t0 = time.time()
np.set_printoptions(precision=6, suppress=True, linewidth=140)
results = {"checks": []}


def ok(name, cond, err=None):
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {name}" + (f"   (max err {err:.3g})" if err is not None else ""))
    results["checks"].append({"name": name, "pass": bool(cond),
                              **({"max_err": float(err)} if err is not None else {})})


def r6(x):
    """Round floats to 6 significant digits, recursively, for JSON compactness."""
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, (list, tuple)):
        return [r6(v) for v in x]
    return x


# =============================================================================
# Part 0 -- the arena: 7x7 free grid, lex orientation, Vishnoi B
# =============================================================================
print("== 0: arena (7x7 free grid, n = 49, m = 84) ==")
R = C = 7
n = R * C                                    # 49
coords = [[c, r] for r in range(R) for c in range(C)]   # v = 7r + c -> (x, y)
edges = []
for v in range(n):
    r, c = divmod(v, C)
    if c < C - 1:
        edges.append((v, v + 1))             # rightward
    if r < R - 1:
        edges.append((v, v + C))             # downward
edges.sort()                                 # lex by (tail, head); tail < head
m = len(edges)                               # 84
eidx = {e: i for i, e in enumerate(edges)}

B = np.zeros((m, n))
for i, (u, v) in enumerate(edges):
    B[i, u] = +1.0                           # tail
    B[i, v] = -1.0                           # head

L = B.T @ B
deg = [int(x) for x in np.diag(L)]           # Python ints (Fraction-safe)
deg_np = np.array(deg)
Lp = np.linalg.pinv(L)
X = np.linalg.pinv(B.T)                      # computed INDEPENDENTLY of Lp
L_edge = B @ B.T
L_edge_p = np.linalg.pinv(L_edge)
Pi = B @ Lp @ B.T
J = np.ones((n, n)) / n

ok(f"arena: m == 84 edges, degree histogram {{2: 4 corners, 3: 20 boundary, "
   f"4: 25 interior}}, sum deg == 2m == 168",
   m == 84 and deg.count(2) == 4 and deg.count(3) == 20
   and deg.count(4) == 25 and sum(deg) == 168)

# exact integer Laplacian + Matrix-Tree cross-check against OEIS A007341(7)
L_int = [[0] * n for _ in range(n)]
for (u, v) in edges:
    L_int[u][u] += 1
    L_int[v][v] += 1
    L_int[u][v] -= 1
    L_int[v][u] -= 1


def bareiss_det(M):
    """Exact determinant of an integer matrix (fraction-free Bareiss)."""
    M = [row[:] for row in M]
    N_ = len(M)
    sign, prev = 1, 1
    for k in range(N_ - 1):
        if M[k][k] == 0:
            for i2 in range(k + 1, N_):
                if M[i2][k] != 0:
                    M[k], M[i2] = M[i2], M[k]
                    sign = -sign
                    break
            else:
                return 0
        pk, Mk = M[k][k], M[k]
        for i2 in range(k + 1, N_):
            Mi = M[i2]
            mik = Mi[k]
            for j2 in range(k + 1, N_):
                Mi[j2] = (Mi[j2] * pk - mik * Mk[j2]) // prev
        prev = pk
    return sign * M[N_ - 1][N_ - 1]


def minor0(Mat):
    return [row[1:] for row in Mat[1:]]


N_G = bareiss_det(minor0(L_int))             # spanning trees of the 7x7 grid
ok(f"Matrix-Tree sanity: #spanning trees == {N_G} == OEIS A007341(7) "
   f"(exact integer Bareiss determinant of the 48x48 minor)",
   N_G == 19872369301840986112)

# =============================================================================
# 1 -- rows of B: the measurements (click an edge -> vertex spikes)
# =============================================================================
print("\n== 1: rows of B (edge measurements) ==")
struct_ok = all(
    B[i, u] == 1.0 and B[i, v] == -1.0 and np.count_nonzero(B[i]) == 2
    and u < v
    for i, (u, v) in enumerate(edges))
ok("1. every row of B is a +1/-1 dipole spike pair: +1 at the tail, -1 at the "
   "head, tail index < head index (lex orientation), exactly 2 nonzeros, "
   "row sums 0 (all 84 rows)", struct_ok and np.abs(B.sum(axis=1)).max() == 0.0)

# =============================================================================
# 2 -- rows of X = pinv(B^T): gauge-fixed dipole response fields
# =============================================================================
print("\n== 2: rows of X (dipole response fields) ==")
err = np.abs(X - B @ Lp).max()
ok("2a. X == pinv(B^T) == B L^+ (response matrix; pinv computed independently "
   "of L^+ -- the 1-D version is dual_basis_explainer's X)", err < 1e-12, err)
err = max(np.abs(X[e] - Lp @ B[e]).max() for e in range(m))
ok("2b. every row e of X == L^+ b_e: the potential field of a unit dipole "
   "(+1 tail, -1 head) across edge e, in the mean-zero gauge (all 84 rows)",
   err < 1e-12, err)
err = np.abs(X.sum(axis=1)).max()
ok("2c. every response row is BALANCED: row sums == 0 (free graph -> fields "
   "live perpendicular to the constant gauge mode)", err < 1e-12, err)
err = np.abs(X.T @ X - Lp).max()
ok("2d. X^T X == L^+ (covariance = Gram of the dense responses, free version)",
   err < 1e-12, err)
err = np.abs(X.T @ B - (np.eye(n) - J)).max()
ok("2e. X^T B == I - 11^T/49: biorthogonality modulo the gauge (the centering "
   "projector, exactly dual_basis' free-path P8 note, now 2-D)", err < 1e-12, err)

# =============================================================================
# 3 -- columns of B: the incident-edge stencil (click a vertex -> edge arrows)
# =============================================================================
print("\n== 3: columns of B (vertex stencils) ==")
col_ok = True
for v in range(n):
    nz = np.nonzero(B[:, v])[0]
    if len(nz) != deg[v]:
        col_ok = False
    for i in nz:
        u_, w_ = edges[i]
        want = 1.0 if u_ == v else -1.0
        if B[i, v] != want or v not in (u_, w_):
            col_ok = False
ok("3. every column v of B hits exactly deg(v) edges: +1 where v is the tail, "
   "-1 where v is the head (the incident-edge stencil; all 49 columns)", col_ok)

# =============================================================================
# 4 -- columns of X: internal current pattern from a unit injection
# =============================================================================
print("\n== 4: columns of X (unit-injection current patterns) ==")
err = max(np.abs(X[:, v] - B @ Lp[:, v]).max() for v in range(n))
ok("4a. column v of X == B (L^+ e_v): Ohm's law currents of the Green "
   "potential field -- the unit-load / Maxwell picture on edges (all 49 cols)",
   err < 1e-12, err)
err = np.abs(B.T @ X - (np.eye(n) - J)).max()
ok("4b. divergence B^T X == I - 11^T/49: injection +1 at v is balanced by the "
   "gauge's uniform extraction -1/49 everywhere (Kirchhoff on the free graph)",
   err < 1e-12, err)

# =============================================================================
# 5 -- vertex Gram L = B^T B: the stencil row
# =============================================================================
print("\n== 5: vertex Gram L = B^T B ==")
L_expected = np.array(L_int, dtype=float)
err = np.abs(L - L_expected).max()
ok("5. L == B^T B has the stencil structure exactly: L[v,v] == deg(v) "
   "(2/3/4), L[v,w] == -1 iff v ~ w, 0 otherwise (conditional / regression "
   "reading of row v)", err == 0.0, err)

# =============================================================================
# 6 -- EDGE Gram L_edge = B B^T: the edge Laplacian and its sign rule
# =============================================================================
print("\n== 6: edge Gram L_edge = B B^T (the edge Laplacian) ==")
# sign rule, derived: L_edge[e,f] = sum_v s_e(v) s_f(v) with s = +1 tail / -1
# head.  Diagonal: (+1)^2 + (-1)^2 = 2.  Off-diagonal, one shared vertex v:
#   tail of both, or head of both  -> +1   (edges DIVERGE from or CONVERGE at v)
#   tail of one and head of other  -> -1   (edges form a DIRECTED PATH through v)
L_edge_expected = np.zeros((m, m))
sign_counts = {+1: 0, -1: 0}
for i, (u1, v1) in enumerate(edges):
    L_edge_expected[i, i] = 2.0
    for j2, (u2, v2) in enumerate(edges):
        if j2 == i:
            continue
        shared = {u1, v1} & {u2, v2}
        if len(shared) == 1:
            w_ = shared.pop()
            s1 = +1 if w_ == u1 else -1
            s2 = +1 if w_ == u2 else -1
            L_edge_expected[i, j2] = s1 * s2
            if j2 > i:
                sign_counts[s1 * s2] += 1
err = np.abs(L_edge - L_edge_expected).max()
ok(f"6a. sign rule of the EDGE LAPLACIAN verified structurally: diag == 2 "
   f"(unit conductances), off-diag +1 when edges converge/diverge at the "
   f"shared vertex, -1 when they form a directed path through it "
   f"({sign_counts[1]} pairs +1, {sign_counts[-1]} pairs -1, "
   f"{sign_counts[1] + sign_counts[-1]} == sum_v C(deg v, 2) == 214 adjacent "
   f"pairs)", err == 0.0 and sign_counts[1] + sign_counts[-1] == 214, err)

absB = np.abs(B)
A_lg = (L_edge_expected != 0).astype(float) - np.eye(m)  # line-graph adjacency
err = np.abs(absB @ absB.T - (2 * np.eye(m) + A_lg)).max()
ok("6b. line-graph identity: |B| |B|^T == 2I + A(line graph) -- unsigned "
   "version; 'edge Laplacian' is legible as 2I + signed line-graph adjacency",
   err == 0.0, err)

evL = np.sort(np.linalg.eigvalsh(L))
evLe = np.sort(np.linalg.eigvalsh(L_edge))
tolz = 1e-9
nzL = evL[np.abs(evL) > tolz]
nzLe = evLe[np.abs(evLe) > tolz]
zL = int((np.abs(evL) <= tolz).sum())
zLe = int((np.abs(evLe) <= tolz).sum())
err = np.abs(nzL - nzLe).max()
ok(f"6c. nonzero spectra of L and L_edge COINCIDE with multiplicity "
   f"(48 shared eigenvalues, max gap {err:.2e}); L has {zL} zero (constants), "
   f"L_edge has {zLe} zeros == m - n + 1 == 36 == dim(cycle space) -- why "
   f"'edge Laplacian' deserves the name",
   err < 1e-10 and zL == 1 and zLe == 36 and len(nzL) == 48, err)

# closed-form spectrum: P7 x P7 Cartesian product
path_ev = np.array([2 - 2 * np.cos(np.pi * k / 7) for k in range(7)])
grid_ev = np.sort((path_ev[:, None] + path_ev[None, :]).ravel())
err = np.abs(evL - grid_ev).max()
ok("6d. spectrum closed form: eig(L) == {4 sin^2(i pi/14) + 4 sin^2(j pi/14)} "
   "(P7 [] P7 Cartesian product), lambda_2 = 0.198062, lambda_max = 7.60388",
   err < 1e-10, err)

# =============================================================================
# 7 -- L^+: the free-grid Green function in the mean-zero gauge
# =============================================================================
print("\n== 7: L^+ (free Green function, mean-zero gauge) ==")
err = max(np.abs(Lp - Lp.T).max(), np.abs(Lp @ np.ones(n)).max())
ok("7a. L^+ symmetric with L^+ 1 == 0: every Green column lives in the "
   "mean-zero gauge (grounding_explainer's pinv row, now a 2-D field)",
   err < 1e-12, err)
err = np.abs(L @ Lp - (np.eye(n) - J)).max()
ok("7b. L L^+ == I - 11^T/49: column v of L^+ is the potential field of "
   "injection +1 at v, extraction 1/49 everywhere (the free-graph Green "
   "function identity)", err < 1e-12, err)

# =============================================================================
# 8 -- L_edge^+ == X X^T: dipole-to-dipole transfer
# =============================================================================
print("\n== 8: L_edge^+ == X X^T ==")
err = np.abs(L_edge_p - X @ X.T).max()
ok("8a. L_edge^+ == X X^T: the pseudoinverse of the edge Laplacian is the "
   "GRAM OF THE RESPONSES -- entry (e,f) = <dipole field e, dipole field f> = "
   "dipole-to-dipole transfer in the resistance metric", err < 1e-11, err)
err = np.abs(L_edge @ L_edge_p - Pi).max()
ok("8b. L_edge L_edge^+ == Pi: inverting the edge Laplacian recovers exactly "
   "Vishnoi's projection (modes 6, 8, 9 are one story)", err < 1e-11, err)

# =============================================================================
# 9 -- Pi = B L^+ B^T = X B^T: projection, leverage, spanning trees, cycles
# =============================================================================
print("\n== 9: Pi = B L^+ B^T (cut-space projection) ==")
err = np.abs(Pi - X @ B.T).max()
ok("9a. Pi == B L^+ B^T == X B^T (cross-Gram of measurements and responses)",
   err < 1e-12, err)
err = max(np.abs(Pi - Pi.T).max(), np.abs(Pi @ Pi - Pi).max())
ok("9b. Pi symmetric idempotent (Pi^2 == Pi): orthogonal projection on edge "
   "space", err < 1e-11, err)
evPi = np.sort(np.linalg.eigvalsh(Pi))
rank_pi = int((evPi > 0.5).sum())
err = max(np.abs(evPi[:m - 48]).max(), np.abs(evPi[m - 48:] - 1).max())
ok(f"9c. spec(Pi) == {{0 x 36, 1 x 48}}: rank == n - 1 == 48, corank == "
   f"m - n + 1 == 36 (cut space + cycle space = edge space)",
   rank_pi == 48 and err < 1e-10, err)

reff = np.array([Lp[u, u] + Lp[v, v] - 2 * Lp[u, v] for (u, v) in edges])
err = np.abs(np.diag(Pi) - reff).max()
ok("9d. diag(Pi) == effective resistances b_e^T L^+ b_e == leverage scores "
   "of the 84 edge rows", err < 1e-12, err)
err = abs(np.trace(Pi) - 48)
ok("9e. Foster's theorem: sum_e Reff(e) == tr Pi == n - 1 == 48", err < 1e-10, err)

# spanning-tree marginals by exact Matrix-Tree deletion (Thm 4.5)
tree_marg = []
for (u, v) in edges:
    Ld = [row[:] for row in L_int]
    Ld[u][u] -= 1
    Ld[v][v] -= 1
    Ld[u][v] += 1
    Ld[v][u] += 1
    N_del = bareiss_det(minor0(Ld))          # trees of G - e
    tree_marg.append(Fraction(N_G - N_del, N_G))
err = max(abs(float(p) - reff[i]) for i, p in enumerate(tree_marg))
ok("9f. Thm 4.5 by EXACT Matrix-Tree deletion: P[e in T] == "
   "(N(G) - N(G-e))/N(G) == Reff(e) for all 84 edges (integer determinants, "
   "zero-tolerance fractions vs float Reff)", err < 1e-10, err)
ok("9g. exact marginals sum: sum_e P[e in T] == 48 EXACTLY in Fractions "
   "(every spanning tree has 48 edges)", sum(tree_marg) == Fraction(48))

# cycle annihilation: the square face at (3,3), then all 36 faces
faces = []
for fr in range(R - 1):
    for fc in range(C - 1):
        a = 7 * fr + fc
        faces.append({
            "corner": a,
            "edges": [eidx[(a, a + 1)], eidx[(a + 1, a + 8)],
                      eidx[(a + 7, a + 8)], eidx[(a, a + 7)]],
            "signs": [1, 1, -1, -1],         # traversal a -> a+1 -> a+8 -> a+7 -> a
        })
Cyc = np.zeros((len(faces), m))
for k, f in enumerate(faces):
    for ei, s in zip(f["edges"], f["signs"]):
        Cyc[k, ei] = s
face_demo = next(f for f in faces if f["corner"] == 24)   # face at (3,3)
c_demo = Cyc[[f["corner"] for f in faces].index(24)]
err = max(np.abs(B.T @ c_demo).max(), np.abs(Pi @ c_demo).max())
ok("9h. explicit face cycle killed by Pi: the square 24->25->32->31->24 has "
   "B^T c == 0 (divergence-free) and Pi c == 0, (I - Pi) c == c -- a cycle "
   "current is invisible to Pi (the page's mini-visualization)",
   err < 1e-12 and np.abs((np.eye(m) - Pi) @ c_demo - c_demo).max() < 1e-12, err)
err = np.abs(Pi @ Cyc.T).max()
rank_cyc = int(np.linalg.matrix_rank(Cyc))
ok(f"9i. ALL 36 face cycles are killed by Pi and span ker(Pi): "
   f"rank(faces) == 36 == corank(Pi) (I - Pi projects onto the cycle space)",
   err < 1e-11 and rank_cyc == 36, err)

# =============================================================================
# 10 -- random-walk sum: ground g, Neumann expansion, visit counts
# =============================================================================
print("\n== 10: random-walk sum (ground g = 0, source s = 24) ==")
g, s = 0, 24                                  # ground corner (0,0); source center (3,3)
keep = [v for v in range(n) if v != g]
s_idx = keep.index(s)
W_int = [[-L_int[i][j2] if i != j2 else 0 for j2 in range(n)] for i in range(n)]
Wk = np.array([[W_int[i][j2] for j2 in keep] for i in keep], dtype=float)
dk = deg_np[keep].astype(float)
Q = Wk / dk[:, None]                          # Q[i,j] = W_ij / d_i (walk off g)
Lg = np.array([[L_int[i][j2] for j2 in keep] for i in keep], dtype=float)
Gg = np.linalg.inv(Lg)

Dm12 = 1.0 / np.sqrt(dk)
rho = float(np.abs(np.linalg.eigvalsh(Dm12[:, None] * Wk * Dm12[None, :])).max())
tau_relax = -1.0 / np.log(rho)

# full-matrix Neumann sum: G_g == sum_k Q^k D^{-1}
T = np.diag(1.0 / dk)
S_full = T.copy()
k_used = 0
for k in range(1, 30001):
    T = Q @ T
    S_full += T
    k_used = k
    if np.abs(T).max() < 1e-17:
        break
err = np.abs(S_full - Gg).max() / np.abs(Gg).max()
ok(f"10a. Neumann / path expansion: G_g == sum_k (D^-1 W)^k D^-1 restricted "
   f"off g, summed to K = {k_used} (rel err {err:.1e}); convergence "
   f"guaranteed by rho == {rho:.6f} < 1", err < 1e-12, err)

# column-of-source partial sums: the walk sum FILLING IN the Green function
K_LIST = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
K_max = 3000
t_vec = np.zeros(len(keep))
t_vec[s_idx] = 1.0 / dk[s_idx]
S_vec = t_vec.copy()
fields, err_curve = {}, np.zeros(K_max + 1)
exact_col = Gg[:, s_idx]
err_curve[0] = np.abs(S_vec - exact_col).max()
if 0 in K_LIST:
    fields[0] = S_vec.copy()
for k in range(1, K_max + 1):
    t_vec = Q @ t_vec
    S_vec += t_vec
    err_curve[k] = np.abs(S_vec - exact_col).max()
    if k in K_LIST:
        fields[k] = S_vec.copy()

sel = [k for k in range(0, K_max + 1, 2) if 1e-11 < err_curve[k] < 1e-4]
slope = np.polyfit(sel, np.log(err_curve[sel]), 1)[0]
rho_meas = float(np.exp(slope))
err = abs(rho_meas - rho) / rho
ok(f"10b. measured decay rate of the partial-sum error == rho: fitted "
   f"{rho_meas:.6f} vs rho(D^-1 W|off g) = {rho:.6f} (even-K fit, bipartite "
   f"grid; tau_relax = -1/ln rho = {tau_relax:.1f} steps)", err < 5e-3, err)

# exact absorbing-chain visit counts (Fractions, no Monte Carlo)
nk = len(keep)


def frac_solve(Aint, Bint):
    """Exact N with Aint N = Bint (Gauss-Jordan over Fractions)."""
    M = [[Fraction(Aint[i][j2]) for j2 in range(nk)]
         + [Fraction(Bint[i][j2]) for j2 in range(nk)] for i in range(nk)]
    for k in range(nk):
        p = next(i for i in range(k, nk) if M[i][k] != 0)
        M[k], M[p] = M[p], M[k]
        piv = M[k][k]
        M[k] = [v / piv for v in M[k]]
        for i in range(nk):
            if i != k and M[i][k] != 0:
                f_ = M[i][k]
                M[i] = [a - f_ * b for a, b in zip(M[i], M[k])]
    return [row[nk:] for row in M]


Lg_int = [[L_int[i][j2] for j2 in keep] for i in keep]
D_int = [[deg[keep[i]] if i == j2 else 0 for j2 in range(nk)] for i in range(nk)]
N_exact = frac_solve(Lg_int, D_int)          # N = L_g^{-1} D = fundamental matrix

Wk_int = [[W_int[keep[i]][keep[j2]] for j2 in range(nk)] for i in range(nk)]
neigh = [[j2 for j2 in range(nk) if Wk_int[i][j2]] for i in range(nk)]
rec_ok = all(
    N_exact[i][j2] == (1 if i == j2 else 0)
    + Fraction(sum(N_exact[k2][j2] for k2 in neigh[i]), deg[keep[i]])
    for i in range(nk) for j2 in range(nk))
ok("10c. absorbing-chain recurrence EXACT in Fractions: visits(i->j) == "
   "delta_ij + sum_k P(i->k) visits(k->j) for all 48x48 pairs (N = (I-Q)^{-1} "
   "= fundamental matrix of the chain absorbed at g; start counts as a visit)",
   rec_ok)
err = max(abs(float(N_exact[i][j2]) / deg[keep[j2]] - Gg[i, j2])
          for i in range(nk) for j2 in range(nk))
ok("10d. visit-count identity: G_g[i,j] == E[# visits to j starting at i "
   "before absorption at g] / d_j, exact fractions vs float inverse "
   "(all 48x48 entries)", err < 1e-10, err)
rev_ok = all(deg[keep[i]] * N_exact[i][j2] == deg[keep[j2]] * N_exact[j2][i]
             for i in range(nk) for j2 in range(i + 1, nk))
ok("10e. reversibility bonus: d_i N[i,j] == d_j N[j,i] EXACTLY (symmetry of "
   "G_g seen through visit counts)", rev_ok)

# h_i := E[steps i -> g] is the row sum of the fundamental matrix.  Two real
# legs: (1) h must satisfy the first-step recurrence h_i = 1 + (1/d_i)
# sum_{j ~ i} h_j (with h_g = 0) EXACTLY -- the unique solution of an
# invertible system, so this pins h independently of how it was computed;
# (2) the source entry must equal the literal fraction quoted on the page,
# so any drift in the computation fails the check.
h_row = [sum(N_exact[i][j2] for j2 in range(nk)) for i in range(nk)]
first_step_ok = all(                          # neighbors == g contribute h_g = 0
    h_row[i] == 1 + Fraction(sum(h_row[j2] for j2 in neigh[i]), deg[keep[i]])
    for i in range(nk))
hit_time = h_row[s_idx]
ok(f"10f. exact expected hitting time center -> corner: E[steps 24 -> 0] == "
   f"{hit_time} == {float(hit_time):.6f} (row sum of the fundamental matrix; "
   f"satisfies the first-step recurrence h_i = 1 + (1/d_i) sum_(j~i) h_j "
   f"exactly at all 48 vertices, and equals the pinned literal 3083931/16211)",
   first_step_ok and hit_time == Fraction(3083931, 16211))

npass = sum(c["pass"] for c in results["checks"])
print(f"\n{npass}/{len(results['checks'])} PASS   ({time.time() - t0:.1f} s)")

# =============================================================================
# key numbers to stdout
# =============================================================================
print(f"\nspanning trees N(G)      = {N_G}")
print(f"Reff / diag(Pi) range    = [{reff.min():.6f}, {reff.max():.6f}], "
      f"mean {reff.mean():.6f}, Foster sum {reff.sum():.6f}")
print(f"spectra: lambda_2(L)     = {evL[1]:.6f}, lambda_max = {evL[-1]:.6f}, "
      f"48 shared nonzeros, L_edge zeros = {zLe}")
print(f"walk: rho                = {rho:.6f}, tau_relax = {tau_relax:.1f}, "
      f"measured rate = {rho_meas:.6f}")
print(f"hitting time E[24 -> 0]  = {float(hit_time):.4f} steps "
      f"({hit_time.numerator}/{hit_time.denominator})")

# =============================================================================
# JSON export -- every matrix / field / curve the ten widgets need
# =============================================================================
def emb49(vec48):
    """Embed a keep-indexed field into all 49 vertices, 0 at the ground."""
    out = [0.0] * n
    for i, v in enumerate(keep):
        out[v] = float(vec48[i])
    return out


results["arena"] = {
    "graph": "7x7 free grid, n = 49 vertices, m = 84 edges, unit conductances",
    "vertex_order": "row-major v = 7r + c, r, c in 0..6; widget (x, y) = (c, r)",
    "orientation": "every edge tail -> head with tail index < head index "
                   "(DirectedGraph-Acyclic lex order, electrical_networks.wls "
                   "convention); edge list sorted lex by (tail, head)",
    "sign_convention": "Vishnoi: B[e, tail] = +1, B[e, head] = -1",
    "spanning_trees": str(N_G),
    "siblings": {"1d_dual_basis": "dual_basis_explainer.html",
                 "grounding": "grounding_explainer.html",
                 "green_tents": "green_tents_explainer.html"},
}
results["geometry"] = {
    "n": n, "m": m,
    "vertex_xy": coords,
    "edges": [[u, v] for (u, v) in edges],
    "degrees": deg,
}
results["matrices"] = {
    "B": [[int(x) for x in row] for row in B],
    "X": r6([[float(x) for x in row] for row in X]),
    "L": [[int(x) for x in row] for row in L],
    "L_edge": [[int(x) for x in row] for row in L_edge],
    "L_pinv": r6([[float(x) for x in row] for row in Lp]),
    "L_edge_pinv": r6([[float(x) for x in row] for row in L_edge_p]),
    "Pi": r6([[float(x) for x in row] for row in Pi]),
}
results["spectra"] = {
    "L_eigs": r6(evL.tolist()),
    "L_edge_eigs": r6(evLe.tolist()),
    "shared_nonzeros": 48,
    "L_zero_multiplicity": 1,
    "L_edge_zero_multiplicity": zLe,
    "cycle_space_dim": m - n + 1,
    "closed_form": "eig(L) = {4 sin^2(i pi/14) + 4 sin^2(j pi/14), i,j = 0..6} "
                   "(P7 x P7 Cartesian product)",
}
results["modes"] = {
    "1_B_row": {
        "click": "edge", "render": "vertex field", "data": "matrices.B row e",
        "caption": "the measurement: +1 at the tail, -1 at the head"},
    "2_X_row": {
        "click": "edge", "render": "vertex field", "data": "matrices.X row e",
        "caption": "gauge-fixed dipole response field; balanced (sums to 0); "
                   "the 2-D sibling of dual_basis' centered steps"},
    "3_B_col": {
        "click": "vertex", "render": "edge arrows", "data": "matrices.B column v",
        "caption": "incident-edge stencil: +1 on out-edges (v is tail), "
                   "-1 on in-edges (v is head)"},
    "4_X_col": {
        "click": "vertex", "render": "edge arrows", "data": "matrices.X column v",
        "caption": "internal currents from unit injection at v, balanced by "
                   "the gauge's uniform -1/49 extraction (Maxwell unit load)"},
    "5_L_row": {
        "click": "vertex", "render": "vertex field", "data": "matrices.L row v",
        "caption": "vertex Gram: degree at v, -1 at neighbors (stencil / "
                   "conditional reading)"},
    "6_L_edge_row": {
        "click": "edge", "render": "edge field", "data": "matrices.L_edge row e",
        "caption": "EDGE LAPLACIAN row: 2 on e, +1 on edges converging/"
                   "diverging at a shared vertex, -1 on directed-path "
                   "neighbors; |B||B|^T = 2I + A(line graph)",
        "sign_rule": "L_edge[e,f] = s_e(v) s_f(v) at the shared vertex v; "
                     "+1 tail-tail or head-head, -1 tail-head",
        "adjacent_pairs": {"plus": sign_counts[1], "minus": sign_counts[-1]}},
    "7_L_pinv_col": {
        "click": "vertex", "render": "vertex field", "data": "matrices.L_pinv column v",
        "caption": "free-grid Green function in the mean-zero gauge "
                   "(grounding_explainer's pinv row, now 2-D)"},
    "8_L_edge_pinv_row": {
        "click": "edge", "render": "edge field", "data": "matrices.L_edge_pinv row e",
        "caption": "L_edge^+ == X X^T: dipole-to-dipole transfer in the "
                   "resistance metric (Gram of the responses)"},
    "9_Pi_row": {
        "click": "edge", "render": "edge field", "data": "matrices.Pi row e",
        "caption": "currents through every edge when a unit dipole drives "
                   "edge e; diag = Reff = leverage = P[e in spanning tree]",
        "diag_reff": r6(reff.tolist()),
        "tree_marginals_exact": [str(p) for p in tree_marg],
        "foster_sum": 48,
        "cycle_demo": {
            "face_corner": 24,
            "vertices": [24, 25, 32, 31],
            "edge_indices": face_demo["edges"],
            "signs": face_demo["signs"],
            "vector": [int(x) for x in c_demo],
            "Pi_c_maxabs": float(np.abs(Pi @ c_demo).max()),
            "caption": "a face-cycle current: B^T c = 0, Pi c = 0, "
                       "(I - Pi) c = c -- invisible to Pi"},
        "all_faces": faces},
    "10_walk_sum": {
        "click": "vertex (source); fields baked for default s = 24, g = 0",
        "render": "vertex field animation over K",
        "ground": g, "source": s,
        "rho": r6(rho), "tau_relax": r6(tau_relax),
        "rho_measured": r6(rho_meas),
        "K_list": K_LIST,
        "partial_fields": {str(k): r6(emb49(fields[k])) for k in K_LIST},
        "exact_field": r6(emb49(exact_col)),
        "err_curve": {
            "K": list(range(0, 100)) + list(range(100, K_max + 1, 20)),
            "max_abs_err": r6([float(err_curve[k]) for k in
                               list(range(0, 100)) + list(range(100, K_max + 1, 20))])},
        "visit_count_note": "G_g[i,j] = E[# visits to j of the simple random "
                            "walk started at i, before absorption at g, "
                            "counting the start when i == j] / d_j "
                            "(verified exactly in Fractions)",
        "exact_examples": {
            "G_g[s,s]": str(Fraction(N_exact[s_idx][s_idx], deg[s])),
            "N[s,s]_visits": str(N_exact[s_idx][s_idx]),
            "hitting_time_24_to_0": str(hit_time)}},
}
results["summary"] = {
    "checks_total": len(results["checks"]),
    "checks_passed": npass,
    "runtime_s": round(time.time() - t0, 2),
    "reff_range": r6([float(reff.min()), float(reff.max())]),
    "rho": r6(rho),
    "tau_relax": r6(tau_relax),
    "hitting_time_center_to_corner": r6(float(hit_time)),
    "lambda_2": r6(float(evL[1])),
    "lambda_max": r6(float(evL[-1])),
}

out = Path(__file__).resolve().parents[2] / "results" / "grid_viz.json"
out.write_text(json.dumps(results, indent=1))
size_mb = out.stat().st_size / 1e6
assert size_mb < 4.0, f"JSON too large: {size_mb:.2f} MB"
print(f"\nwrote {out} ({out.stat().st_size / 1024:.0f} KB)")
