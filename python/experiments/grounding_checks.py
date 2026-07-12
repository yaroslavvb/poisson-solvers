"""Numerical + exact-rational verification of every claim in the grounding /
boundary-conditions tutorial (the free-vs-grounded seed of
reports/green_tents_explainer.html SS4, expanded).

Arena: the n = 6 free path graph, L = B^T B with B the (Vishnoi-sign) edge-
vertex incidence matrix -- exactly the user's Mathematica setup

    B = -Transpose@IncidenceMatrix@DirectedGraph[g, "Acyclic"]   (rows e_i - e_{i+1})
    L = Transpose[B].B                                           (diag {1,2,2,2,2,1})

plus n = 32 for continuum-limit curves.  Two Mathematica predicates are ported
verbatim (1-based Part -> 0-based slice):

    GroundedInverse[l_, k_] : delete row/col k, invert, re-embed with a zero
                              row/col at k
    TriangularRankQ[m_, r_] : AllTrue over j of MatrixRank[m[[1;;j, j;;n]]] <= r
                              -- every ALIGNED corner block (rows 1..j, cols
                              j..n) has rank <= r.  This is the semiseparable-
                              rank certificate; its max over j equals report
                              14's off-diagonal-block (Markov separator) rank
                              on the chain, which we also verify.

Claims verified (each its own PASS line): the user's two findings
(GroundedInverse[L,1] triangular-rank-1, PseudoInverse[L] triangular-rank-2);
triangular-rank-1 for EVERY grounding k = 1..6; grounded-at-1 == the integer
matrix min(i,j) - 1 exactly (Brownian-motion kernel); interior-k block split
(Markov separator -> block-diagonal inverse, each block a motion kernel run
away from k); Dirichlet-both == the suite's tridiag(-1,2,-1) == bridge
covariance min(i,j)(7-max(i,j))/7; the Robin closed form
(L + kappa e_1 e_1^T)^{-1} = min(i,j) - 1 + 1/kappa (derived affine u,v pair),
its kappa -> 0 / kappa -> inf limits, rank 1 for every kappa; Gantmacher-Krein
single-pair sampling on random SPD tridiagonals; THE GAUGE IDENTITY
L^+ = C . GroundedInverse(L,k) . C for every k (C = I - 11^T/6, exact
rationals, proved via the four Moore-Penrose axioms); pinv triangular rank
EXACTLY 2 (equality, not bound) with the explicit two-pair decomposition
showing double centering is what adds the +1 rank; R_eff(i,j) = |i-j|
invariant across all 6 groundings and the pinv; Sturm-Liouville u_i v_j factor
extraction per mode against the predicted families (bridge x(1-x)-family,
motion min-family, Robin affine).

Rank tolerances: exact ranks at n = 6 use fractions.Fraction Gaussian
elimination (no tolerance at all).  Float ranks (n = 32, random samples) use
numpy.linalg.matrix_rank's default SVD cut  tol = sigma_max * max(m,n) * eps
(eps = 2.22e-16), i.e. singular values below ~1e-14 * sigma_max are zero.

Run from the repo root:
    uv run python python/experiments/grounding_checks.py

Expected output: 24 lines, all PASS; writes results/grounding.json with every
number, matrix, u/v pair, rank certificate, and n = 32 widget curve the report
page needs.  Runtime well under 30 s.
"""
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.linalg import inv, matrix_rank, pinv

t0 = time.time()
np.set_printoptions(precision=6, suppress=True, linewidth=150)
results = {"checks": []}


def ok(name, cond, err=None):
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {name}" + (f"   (max err {err:.3g})" if err is not None else ""))
    results["checks"].append({"name": name, "pass": bool(cond),
                              **({"max_err": float(err)} if err is not None else {})})


# ---------------------------------------------------------------------------
# exact rational linear algebra (n = 6 certificates carry NO float tolerance)
# ---------------------------------------------------------------------------
def fmat(rows):
    return [[Fraction(x) for x in row] for row in rows]


def fmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]


def finv(M):
    n = len(M)
    A = [[Fraction(x) for x in row] + [Fraction(int(i == j)) for j in range(n)]
         for i, row in enumerate(M)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [row[n:] for row in A]


def frank(M):
    """Exact rank by Fraction Gaussian elimination (empty block -> 0)."""
    if not M or not M[0]:
        return 0
    A = [[Fraction(x) for x in row] for row in M]
    rows, cols, r = len(A), len(A[0]), 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        pv = A[r][c]
        A[r] = [x / pv for x in A[r]]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        r += 1
        if r == rows:
            break
    return r


def fstr(M):
    return [[str(x) for x in row] for row in M]


def ffloat(M):
    return [[float(x) for x in row] for row in M]


# ---------------------------------------------------------------------------
# faithful ports of the user's Mathematica functions
# ---------------------------------------------------------------------------
def grounded_inverse_exact(L, k):
    """GroundedInverse[l_, k_] with 1-based k, exact Fractions: delete row/col
    k, invert, re-embed with a zero row/col at position k."""
    n = len(L)
    idx = [i for i in range(n) if i != k - 1]
    sub = [[L[i][j] for j in idx] for i in idx]
    si = finv(sub)
    G = [[Fraction(0)] * n for _ in range(n)]
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            G[i][j] = si[a][b]
    return G


def grounded_inverse_float(L, k):
    n = L.shape[0]
    idx = [i for i in range(n) if i != k - 1]
    G = np.zeros((n, n))
    G[np.ix_(idx, idx)] = inv(L[np.ix_(idx, idx)])
    return G


def corner_ranks(M, rank_fn):
    """rank of m[[1;;j, j;;n]] for j = 1..n -- TriangularRankQ's blocks."""
    n = len(M)
    return [rank_fn([row[j - 1:] for row in M[:j]]) for j in range(1, n + 1)]


def triangular_rank_q(M, r, rank_fn=frank):
    """TriangularRankQ[m_, r_] := AllTrue[Range[n], MatrixRank[m[[1;;#, #;;n]]] <= r &]"""
    return all(rk <= r for rk in corner_ranks(M, rank_fn))


def offdiag_ranks(M, rank_fn):
    """report 14's certificate: rank of rows 1..j x cols j+1..n (strict split)."""
    n = len(M)
    return [rank_fn([row[j:] for row in M[:j]]) for j in range(1, n)]


def np_rank(block):
    B = np.asarray(block, dtype=float)
    if B.size == 0:
        return 0
    return int(matrix_rank(B))   # default tol = sigma_max * max(m,n) * eps


# ---------------------------------------------------------------------------
# the arena: n = 6 free path, L = B^T B  (Vishnoi sign: row e = e_i - e_{i+1})
# ---------------------------------------------------------------------------
n = 6
B = np.zeros((n - 1, n))
for e in range(n - 1):
    B[e, e], B[e, e + 1] = 1.0, -1.0          # -Transpose@IncidenceMatrix, acyclic
L = B.T @ B
Lf = fmat(L.astype(int).tolist())              # exact copy
ones = np.ones(n)

# 1. Neumann operator itself: degree diagonal {1,2,2,2,2,1} (the degree-1
#    corners ARE the insulation), zero row sums, kernel = constants, rank 5
want = np.diag([1., 2, 2, 2, 2, 1]) + np.diag(-ones[:5], 1) + np.diag(-ones[:5], -1)
cond = (np.array_equal(L, want) and np.abs(L @ ones).max() == 0.0
        and matrix_rank(L) == n - 1
        and np.abs(L @ (3.7 * ones)).max() == 0.0)
ok("Neumann/free: L = B^T B has diag {1,2,2,2,2,1}, zero row sums, kernel = constants, rank 5", cond)

# 2. solvability: L x = f needs 1^T f = 0 (integral of f = 0); unbalanced
#    injection is inconsistent, balanced is consistent (up to a constant)
e1 = np.eye(n)[0]
x_bad = np.linalg.lstsq(L, e1, rcond=None)[0]
r_bad = np.linalg.norm(L @ x_bad - e1)          # residual == projection onto
want_bad = 1.0 / np.sqrt(6)                     # constants: (1/6)*ones, |.| = 1/sqrt(6)
f_bal = e1 - np.eye(n)[1]
x_bal = np.linalg.lstsq(L, f_bal, rcond=None)[0]
r_bal = np.abs(L @ x_bal - f_bal).max()
r_shift = np.abs(L @ (x_bal + 2.5) - f_bal).max()
ok("solvability: L x = e_1 inconsistent (best residual = 1/sqrt(6), the constant component); "
   "L x = e_1 - e_2 solvable, +const still solves",
   abs(r_bad - want_bad) < 1e-12 and r_bal < 1e-12 and r_shift < 1e-12,
   max(abs(r_bad - want_bad), r_bal, r_shift))

# exact rational pseudoinverse: P = C . GroundedInverse(L,1) . C proved below;
# float pinv for cross-checks
Lp = pinv(L)

# 3. pinv = the mean-zero gauge: L+ 1 = 0 = 1^T L+, and for balanced f the
#    voltages v = L+ f satisfy L v = f with sum(v) = 0
v = Lp @ f_bal
err = max(np.abs(Lp @ ones).max(), np.abs(ones @ Lp).max(),
          np.abs(L @ v - f_bal).max(), abs(v.sum()))
ok("pinv gauge: L+ 1 = 0, 1^T L+ = 0; balanced f -> L(L+ f) = f and sum(L+ f) = 0",
   err < 1e-12, err)

# ---- exact grounded inverses for all k, and the exact rational pinv --------
Gk = {k: grounded_inverse_exact(Lf, k) for k in range(1, n + 1)}
J6 = Fraction(1, 6)
C = [[Fraction(int(i == j)) - J6 for j in range(n)] for i in range(n)]   # I - 11^T/6
P = fmul(fmul(C, Gk[1]), C)                    # candidate exact pinv

# 4. USER FINDING 1: GroundedInverse[L, 1] is triangular-rank-1 (exact certificate)
cert1 = corner_ranks(Gk[1], frank)
ok("USER FINDING: TriangularRankOneQ[GroundedInverse[L,1]] -- corner ranks "
   f"{cert1}, all <= 1 (exact Fractions)", triangular_rank_q(Gk[1], 1))

# 5. USER FINDING 2: PseudoInverse[L] is triangular-rank-2 (exact certificate)
certP = corner_ranks(P, frank)
ok(f"USER FINDING: TriangularRankQ[PseudoInverse[L], 2] -- corner ranks {certP}, "
   "all <= 2 (exact Fractions)", triangular_rank_q(P, 2))

# 6. ... and EXACTLY 2, not just <= 2: rank-1 certificate fails, some corner
#    block has exact rank 2
ranks2 = [j + 1 for j, r in enumerate(certP) if r == 2]
ok(f"pinv triangular rank EXACTLY 2: max corner rank = {max(certP)}, rank-2 blocks at j = {ranks2}, "
   "TriangularRankQ[.,1] = False", max(certP) == 2 and not triangular_rank_q(P, 1))

# 7. every grounding is triangular-rank-1 (exact, all k = 1..6)
all_certs = {k: corner_ranks(Gk[k], frank) for k in range(1, n + 1)}
ok("TriangularRankOneQ[GroundedInverse[L,k]] for ALL k = 1..6 (exact certificates "
   + str([max(c) for c in all_certs.values()]) + ")",
   all(triangular_rank_q(Gk[k], 1) for k in range(1, n + 1)))

# 8. grounded at 1 == the INTEGER matrix min(i,j) - 1, exactly (Brownian motion:
#    scaled continuum min(x_i, x_j); one pin = motion, two pins = bridge)
minm = [[Fraction(min(i, j) - 1) for j in range(1, n + 1)] for i in range(1, n + 1)]
ok("GroundedInverse[L,1] == [min(i,j) - 1] EXACTLY (integer Brownian-motion kernel)",
   Gk[1] == minm)

# 9. interior k splits the chain (Markov separator): zero cross-blocks, and
#    each diagonal block is a motion kernel run away from k:
#    i,j < k -> k - max(i,j);   i,j > k -> min(i,j) - k
def split_ok(k):
    G = Gk[k]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            g = G[i - 1][j - 1]
            if i == k or j == k or (i < k < j) or (j < k < i):
                want = 0
            elif i < k and j < k:
                want = k - max(i, j)
            else:
                want = min(i, j) - k
            if g != want:
                return False
    return True

ok("interior grounding k = 2..5: exact block split, cross-blocks == 0 (Markov separator), "
   "blocks == [k - max(i,j)] and [min(i,j) - k]", all(split_ok(k) for k in range(2, n)))

# 10. Dirichlet both ends: append wall nodes 0,7 to the path, ground BOTH ->
#     delete end rows/cols of the free 8-Laplacian == bump corners 1 -> 2 ==
#     tridiag(-1,2,-1), the suite's A (poisson.laplacian_1d)
m8 = 8
L8 = np.diag([1.] + [2.] * (m8 - 2) + [1.]) + np.diag(-np.ones(m8 - 1), 1) + np.diag(-np.ones(m8 - 1), -1)
A6 = L8[1:-1, 1:-1]
bump = L + np.outer(e1, e1) + np.outer(np.eye(n)[5], np.eye(n)[5])
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poisson import laplacian_1d
ok("Dirichlet both: free 8-path Laplacian minus wall rows/cols == L + e1e1^T + e6e6^T "
   "== tridiag(-1,2,-1) == suite laplacian_1d(6)",
   np.array_equal(A6, bump) and np.array_equal(A6, laplacian_1d(6).toarray()))

# 11. its inverse is the BRIDGE covariance min(i,j)(7-max(i,j))/7 exactly
#     == (1/h)(min(x,y) - xy) with x_i = i/7, h = 1/7; triangular rank 1
A6f = fmat(A6.astype(int).tolist())
Bri = finv(A6f)
bridge_want = [[Fraction(min(i, j) * (7 - max(i, j)), 7) for j in range(1, 7)] for i in range(1, 7)]
certB = corner_ranks(Bri, frank)
ok("bridge: inv(tridiag) == [min(i,j)(7-max(i,j))/7] EXACTLY == (1/h)(min(x,y)-xy), h = 1/7; "
   f"triangular rank 1 (corner ranks {certB})",
   Bri == bridge_want and triangular_rank_q(Bri, 1))

# 12. ROBIN closed form, exact: (L + kappa e1 e1^T)^{-1} == min(i,j) - 1 + 1/kappa.
#     Derivation: columns are discrete-harmonic -> affine per side; left BC row
#     (1+kappa)u_1 - u_2 = 0 gives u_i = i - 1 + 1/kappa; free right end gives
#     v_j = 1; unit-resistor Wronskian normalizer = 1.  Green matrix u_i v_j.
robin_exact_kappas = [Fraction(1, 100), Fraction(1, 10), Fraction(1), Fraction(10), Fraction(100)]
robin_ok, robin_rank1 = True, True
robin_exact = {}
for kap in robin_exact_kappas:
    M = [[Lf[i][j] + (kap if i == j == 0 else 0) for j in range(n)] for i in range(n)]
    G = finv(M)
    wantG = [[Fraction(min(i, j) - 1) + 1 / kap for j in range(1, n + 1)] for i in range(1, n + 1)]
    robin_ok &= (G == wantG)
    robin_rank1 &= triangular_rank_q(G, 1)
    robin_exact[str(kap)] = G
ok("ROBIN closed form EXACT: inv(L + kappa e1e1^T) == [min(i,j) - 1 + 1/kappa] for kappa in "
   "{1/100, 1/10, 1, 10, 100}; u_i = i - 1 + 1/kappa, v_j = 1", robin_ok)
ok("ROBIN rank: TriangularRankOneQ holds for every kappa > 0 (exact at 5 kappas)", robin_rank1)

# 13. Robin float sweep, 13 log-spaced kappas 1e-6 .. 1e6, incl. both limits:
#     kappa -> 0: kappa*G -> 11^T (blow-up along constants, the Neumann kernel);
#     kappa -> inf: G -> GroundedInverse(L,1) with error exactly 1/kappa.
kgrid = np.logspace(-6, 6, 13)
sweep_err, sweep_rank1 = 0.0, True
minm_np = np.array([[min(i, j) - 1. for j in range(1, n + 1)] for i in range(1, n + 1)])
for kap in kgrid:
    G = inv(L + kap * np.outer(e1, e1))
    wantG = minm_np + 1.0 / kap
    sweep_err = max(sweep_err, np.abs((G - wantG) / np.abs(wantG).max()).max())
    sweep_rank1 &= all(r <= 1 for r in corner_ranks(G.tolist(), np_rank))
ok("ROBIN float sweep: closed form holds on kappa = 1e-6..1e6 (13 pts, rel err) "
   "and SVD triangular rank 1 at every kappa", sweep_err < 1e-8 and sweep_rank1, sweep_err)
G_small = inv(L + 1e-6 * np.outer(e1, e1))
G_big = inv(L + 1e6 * np.outer(e1, e1))
err_lim = max(np.abs(1e-6 * G_small - np.ones((n, n))).max(),
              np.abs(G_big - minm_np).max() / 1e-6 - 1.0)
ok("ROBIN limits: kappa*G -> 11^T as kappa -> 0 (Neumann blow-up along constants); "
   "G -> grounded-at-1 as kappa -> inf, error == 1/kappa", err_lim < 1e-5, err_lim)

# 14. Gantmacher-Krein sampling: the inverse of ANY irreducible SPD tridiagonal
#     is a single-pair (Green) matrix u_i v_j on the upper triangle ->
#     triangular rank 1.  20 random M-matrix samples at n = 6 and n = 32.
rng = np.random.default_rng(0)
gk_ok = True
for size in (6, 32):
    for _ in range(10):
        off = -rng.uniform(0.5, 2.0, size - 1)
        d = np.zeros(size)
        d[:-1] += -off
        d[1:] += -off
        d += rng.uniform(0.1, 1.0, size)       # strict diagonal dominance -> SPD
        T = np.diag(d) + np.diag(off, 1) + np.diag(off, -1)
        gk_ok &= all(r <= 1 for r in corner_ranks(inv(T).tolist(), np_rank))
ok("Gantmacher-Krein: 20 random irreducible SPD tridiagonals (n = 6, 32) -> "
   "inverse has SVD triangular rank 1", gk_ok)

# 15. THE GAUGE IDENTITY: L+ == C . GroundedInverse(L,k) . C for EVERY k,
#     C = I - 11^T/6, in exact rational arithmetic.  Proof of pinv-hood: P
#     satisfies all four Moore-Penrose axioms exactly.
mp = (fmul(fmul(Lf, P), Lf) == Lf and fmul(fmul(P, Lf), P) == P
      and fmul(Lf, P) == [list(r) for r in zip(*fmul(Lf, P))]
      and fmul(P, Lf) == [list(r) for r in zip(*fmul(P, Lf))])
same = all(fmul(fmul(C, Gk[k]), C) == P for k in range(1, n + 1))
float_agree = np.abs(np.array(ffloat(P)) - Lp).max()
ok("GAUGE IDENTITY: C.GroundedInverse(L,k).C IDENTICAL for all k = 1..6 (exact) and satisfies "
   "all four Moore-Penrose axioms exactly => == L+; numpy pinv agrees",
   mp and same and float_agree < 1e-12, float_agree)

# 16. centering is what adds the +1 rank: exact two-pair decomposition of the
#     pinv upper triangle.  With N = min(i,j) - 1, r_i = rowmean_i(N),
#     m = grandmean(N):  L+_{ij} (i<=j) = [i - 1 - r_i + m] * 1 + 1 * [-r_j]
#     -- pair 1 is the motion solution (left BC), pair 2 is the centering.
rvec = [sum(minm[i][j] for j in range(n)) / Fraction(n) for i in range(n)]
mbar = sum(rvec) / Fraction(n)
u1 = [Fraction(i) - rvec[i] + mbar for i in range(n)]        # i-1-r_i+m, i 1-based -> Fraction(i0)
v1 = [Fraction(1)] * n
u2 = [Fraction(1)] * n
v2 = [-rvec[j] for j in range(n)]
two_pair = all(P[i][j] == u1[i] * v1[j] + u2[i] * v2[j]
               for i in range(n) for j in range(n) if i <= j)
ok("+1 rank from centering: L+ upper triangle == u1 v1^T + u2 v2^T exactly, "
   "u1_i = (i-1) - rowmean_i + grandmean (motion pair), v2_j = -rowmean_j (centering pair)",
   two_pair)

# 17. R_eff invariance: R_eff(i,j) = G_ii + G_jj - 2 G_ij is THE SAME for all
#     6 groundings and the pinv, and equals |i - j| exactly (unit-resistor path)
def reff(G):
    return [[G[i][i] + G[j][j] - 2 * G[i][j] for j in range(n)] for i in range(n)]

R_want = [[Fraction(abs(i - j)) for j in range(n)] for i in range(n)]
R_all = [reff(Gk[k]) for k in range(1, n + 1)] + [reff(P)]
ok("R_eff gauge-invariance: identical from GroundedInverse(L,k) for every k AND from pinv, "
   "== |i - j| EXACTLY (full 6x6, exact rationals)", all(R == R_want for R in R_all))

# 18. Sturm-Liouville single-pair factors, verified exactly per mode
#     (u satisfies the LEFT BC, v the RIGHT BC; interior grounding is
#     reducible -> one pair PER BLOCK, not one global pair)
def upper_match(G, u, v):
    return all(G[i][j] == u[i] * v[j] for i in range(n) for j in range(n) if i <= j)

uv_ok = (
    upper_match(Bri, [Fraction(i) for i in range(1, 7)],
                [Fraction(7 - j, 7) for j in range(1, 7)])          # bridge: x(1-x) family
    and upper_match(Gk[1], [Fraction(i - 1) for i in range(1, 7)],
                    [Fraction(1)] * 6)                              # motion: min family
    and upper_match(Gk[6], [Fraction(1)] * 6,
                    [Fraction(6 - j) for j in range(1, 7)])         # mirrored motion
    and upper_match(robin_exact["1"], [Fraction(i) for i in range(1, 7)],
                    [Fraction(1)] * 6)                              # Robin kappa=1: u = i-1+1/kappa = i
)
# interior k = 3: pair per block -- left (1, k-j), right (i-k, 1), zero cross
k3 = 3
G3 = Gk[k3]
blk = all(G3[i - 1][j - 1] == (Fraction(k3 - j) if j < k3 else 0)
          for i in range(1, k3) for j in range(i, 7)) and \
      all(G3[i - 1][j - 1] == (Fraction(i - k3) if i > k3 else 0)
          for i in range(k3, 7) for j in range(i, 7))
# numeric extraction demo (u from last column, v from first row, rescaled)
Gb = np.array(ffloat(Bri))
u_num = Gb[:, -1].copy()
v_num = Gb[0, :] / u_num[0]
ext_err = max(abs(Gb[i, j] - u_num[i] * v_num[j]) for i in range(6) for j in range(i, 6))
ok("Sturm-Liouville factors: bridge u=i, v=(7-j)/7; motion u=i-1, v=1; mirrored u=1, v=6-j; "
   "Robin(kappa=1) u=i, v=1 -- all EXACT; interior k=3 one pair PER block; numeric extraction matches",
   uv_ok and blk and ext_err < 1e-12, ext_err)

# 19. certificate equivalence: max aligned-corner rank == max strict
#     off-diagonal (report 14 separator) rank on every mode inverse
cert_pairs_ok = True
for M in [P, Bri, robin_exact["1"]] + [Gk[k] for k in range(1, 7)]:
    cr, od = corner_ranks(M, frank), offdiag_ranks(M, frank)
    cert_pairs_ok &= (max(cr) == max(od))
ok("semiseparable == separator rank: max corner-block rank equals max off-diagonal-block "
   "rank (report 14) on pinv, bridge, Robin, and all 6 groundings", cert_pairs_ok)

# ---------------------------------------------------------------------------
# n = 32: continuum-limit arena for the widget
# ---------------------------------------------------------------------------
N = 32
B32 = np.zeros((N - 1, N))
for e in range(N - 1):
    B32[e, e], B32[e, e + 1] = 1.0, -1.0
L32 = B32.T @ B32
Lp32 = pinv(L32)
C32 = np.eye(N) - np.ones((N, N)) / N

# 20. everything again at n = 32 (SVD ranks): grounding rank 1 for all 32 k,
#     block split, min-kernels, pinv rank exactly 2, gauge identity, R_eff = |i-j|
G32 = {k: grounded_inverse_float(L32, k) for k in range(1, N + 1)}
ok32 = True
for k in range(1, N + 1):
    G = G32[k]
    want = np.zeros((N, N))
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if i == k or j == k or (i < k < j) or (j < k < i):
                pass
            elif i < k and j < k:
                want[i - 1, j - 1] = k - max(i, j)
            else:
                want[i - 1, j - 1] = min(i, j) - k
    ok32 &= np.abs(G - want).max() < 1e-8
    ok32 &= all(r <= 1 for r in corner_ranks(G.tolist(), np_rank))
cr32 = corner_ranks(Lp32.tolist(), np_rank)
gauge32 = max(np.abs(C32 @ G32[k] @ C32 - Lp32).max() for k in (1, 16, 32))
idx = np.arange(1, N + 1)
R32 = np.add.outer(np.diag(Lp32), np.diag(Lp32)) - 2 * Lp32
reff32 = np.abs(R32 - np.abs(np.subtract.outer(idx, idx))).max()
ok("n = 32: grounded inverse == exact min-kernels + block split for ALL k = 1..32, SVD rank 1; "
   f"pinv triangular rank exactly 2 (max corner rank {max(cr32)}); gauge identity; R_eff == |i-j|",
   ok32 and max(cr32) == 2 and gauge32 < 1e-10 and reff32 < 1e-9,
   max(gauge32, reff32))

# 21. n = 32 continuum kernels: Dirichlet-both == (1/h)(min(x,y) - xy) with
#     x_i = i/33, h = 1/33 (bridge); grounded-at-1 == (1/h) min(x0_i, x0_j)
#     with x0_i = (i-1)/33 (Brownian MOTION)
A32 = L32 + np.diag(np.eye(N)[0]) + np.diag(np.eye(N)[-1])   # corner bump 1 -> 2
S32 = inv(A32)
h = 1.0 / (N + 1)
x = idx * h
xm = np.minimum.outer(x, x)
err_b = np.abs(S32 - (xm - np.outer(x, x)) / h).max()
x0 = (idx - 1) * h
err_m = np.abs(G32[1] - np.minimum.outer(x0, x0) / h).max()
ok("n = 32 continuum: bridge inverse == (1/h)(min(x,y) - xy), motion inverse == (1/h) min(x,y) "
   "(h = 1/33)", err_b < 1e-9 and err_m < 1e-9, max(err_b, err_m))

# 22. n = 32 Robin widget grid verified against the closed form
kappas32 = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
minm32 = np.minimum.outer(idx, idx) - 1.0
err_r32, rank_r32 = 0.0, True
R32inv = {}
for kap in kappas32:
    G = inv(L32 + kap * np.outer(np.eye(N)[0], np.eye(N)[0]))
    R32inv[kap] = G
    wantG = minm32 + 1.0 / kap
    err_r32 = max(err_r32, np.abs((G - wantG) / np.abs(wantG).max()).max())
    rank_r32 &= all(r <= 1 for r in corner_ranks(G.tolist(), np_rank))
ok("n = 32 ROBIN grid: inv(L + kappa e1e1^T) == [min(i,j) - 1 + 1/kappa] on the 9-kappa widget "
   "grid, SVD rank 1 each", err_r32 < 1e-9 and rank_r32, err_r32)

npass = sum(c["pass"] for c in results["checks"])
print(f"\n{npass}/{len(results['checks'])} PASS")

# ---------------------------------------------------------------------------
# widget / report data export
# ---------------------------------------------------------------------------
jsel6, jsel32 = 4, 16      # 1-based selected Green columns


def col6(M, j=jsel6):
    return [float(M[i][j - 1]) for i in range(6)]


def rnd(A, d=6):
    return [[round(float(v), d) for v in row] for row in np.asarray(A).tolist()]


modes6 = {}
modes6["dirichlet_both"] = {
    "operator": "tridiag(-1,2,-1) = L + e1e1^T + e6e6^T (append walls, delete)",
    "inverse_exact": fstr(Bri), "inverse_float": ffloat(Bri), "column": col6(Bri),
    "u": fstr([[Fraction(i) for i in range(1, 7)]])[0],
    "v": fstr([[Fraction(7 - j, 7) for j in range(1, 7)]])[0],
    "uv_family": "bridge: u ~ x, v ~ (1-x); G ~ x(1-s)",
    "corner_ranks": certB, "triangular_rank": max(certB),
}
for k in range(1, 7):
    left_u = ["1"] * (k - 1)
    left_v = [str(k - j) for j in range(1, k)]
    right_u = [str(i - k) for i in range(k + 1, 7)]
    right_v = ["1"] * (6 - k)
    modes6[f"ground_{k}"] = {
        "operator": f"delete row/col {k} of L, invert, re-embed (GroundedInverse[L,{k}])",
        "inverse_exact": fstr(Gk[k]), "inverse_float": ffloat(Gk[k]), "column": col6(Gk[k]),
        "uv_per_block": {"left": {"u": left_u, "v": left_v},
                         "right": {"u": right_u, "v": right_v}},
        "uv_family": "motion: min-family run away from k; one pair per irreducible block",
        "corner_ranks": all_certs[k], "triangular_rank": max(all_certs[k]),
    }
modes6["pinv"] = {
    "operator": "Moore-Penrose pseudoinverse of L (mean-zero gauge)",
    "inverse_exact": fstr(P), "inverse_float": ffloat(P), "column": col6(P),
    "uv_pairs": [{"u": [str(x) for x in u1], "v": [str(x) for x in v1]},
                 {"u": [str(x) for x in u2], "v": [str(x) for x in v2]}],
    "uv_family": "TWO pairs: motion pair + centering pair (C = I - 11^T/6 adds +1 rank)",
    "corner_ranks": certP, "triangular_rank": max(certP),
}
modes6["robin"] = {
    "operator": "L + kappa e1 e1^T (partial grounding through resistor 1/kappa)",
    "closed_form": "inverse_ij = min(i,j) - 1 + 1/kappa; u_i = i - 1 + 1/kappa, v_j = 1",
    "kappas_exact": [str(k) for k in robin_exact_kappas],
    "inverse_exact": {str(k): fstr(robin_exact[str(k)]) for k in robin_exact_kappas},
    "kappa_grid": kgrid.tolist(),
    "columns": {f"{kap:g}": [min(i, jsel6) - 1 + 1.0 / kap for i in range(1, 7)] for kap in kgrid},
    "corner_ranks": corner_ranks(robin_exact["1"], frank),
    "triangular_rank": 1,
}

modes32 = {
    "dirichlet_both": {"heatmap": rnd(S32), "column": [float(S32[i, jsel32 - 1]) for i in range(N)],
                       "u": (idx * h).tolist(), "v": (1 - idx * h).tolist(),
                       "corner_ranks": corner_ranks(S32.tolist(), np_rank)},
    "pinv": {"heatmap": rnd(Lp32), "column": [float(Lp32[i, jsel32 - 1]) for i in range(N)],
             "corner_ranks": cr32},
    "ground_k": {str(k): {"heatmap": [[int(round(v)) for v in row] for row in G32[k].tolist()],
                          "column": [float(G32[k][i, jsel32 - 1]) for i in range(N)],
                          "corner_ranks": corner_ranks(G32[k].tolist(), np_rank)}
                 for k in range(1, N + 1)},
    "robin": {"kappas": kappas32,
              "heatmaps": {f"{kap:g}": rnd(R32inv[kap], 4) for kap in (0.1, 1.0, 10.0)},
              "heatmap_rule": "heatmap(kappa) = ground_1-style [min(i,j)-1] + 1/kappa everywhere",
              "columns": {f"{kap:g}": [float(R32inv[kap][i, jsel32 - 1]) for i in range(N)]
                          for kap in kappas32},
              "corner_ranks": corner_ranks(R32inv[1.0].tolist(), np_rank)},
}

results["n6"] = {
    "L": L.astype(int).tolist(),
    "B_rows": B.astype(int).tolist(),
    "selected_column": jsel6,
    "modes": modes6,
    "R_eff_exact": [[int(x) for x in row] for row in
                    [[abs(i - j) for j in range(6)] for i in range(6)]],
    "gauge": {
        "C": "I - 11^T/6 (exact rationals)",
        "statement": "L+ == C . GroundedInverse(L,k) . C for EVERY k = 1..6; proved exactly via "
                     "the four Moore-Penrose axioms; all groundings coincide after double centering",
        "pinv_exact": fstr(P),
    },
}
results["n32"] = {
    "selected_column": jsel32, "h": h,
    "x_bridge": x.tolist(), "x_motion": x0.tolist(),
    "modes": modes32,
    "reff_max_err_vs_absij": float(reff32),
}
results["meta"] = {
    "rank_tolerance": "n=6: exact Fraction Gaussian elimination (no tolerance); floats: "
                      "numpy matrix_rank default tol = sigma_max * max(m,n) * eps (2.22e-16)",
    "conventions": "1-based node labels to mirror Mathematica; unit resistors; "
                   "bridge continuum x_i = i/(n+1); motion continuum x_i = (i-1)/(n+1)",
    "checks_total": len(results["checks"]), "checks_passed": npass,
    "runtime_s": round(time.time() - t0, 3),
}

out = Path(__file__).resolve().parents[2] / "results" / "grounding.json"
out.write_text(json.dumps(results, separators=(",", ":")))
print(f"wrote {out} ({out.stat().st_size:,} bytes) in {time.time() - t0:.2f} s")
