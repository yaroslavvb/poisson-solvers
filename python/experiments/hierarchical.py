"""Experiments for report 14 -- H-matrices / HODLR: where the low rank in
A^{-1} comes from, and what a hierarchical approximate inverse buys as a
preconditioner.

The through-line continues reports 09/13: the Green's function A^{-1} is the
covariance of the GMRF u ~ N(0, A^{-1}); conditional independence given a
separator (zero precision entries across it, report 09) is an algebraic RANK
statement about covariance blocks:

    Sigma_LR = Sigma_LI Sigma_II^{-1} Sigma_IR   =>   rank(Sigma_LR) <= |I|.

PART A -- WHERE THE LOW RANK COMES FROM (the separator theorem):
  1. Machine-check the identity on the 2-D grid (cols 0..14 | 15..16 | 17..31,
     exactly the split of report 13 SS3) => rank(Sigma_LR) <= |I| = 64.
     Sharper 1-D statement: a contiguous split of the chain has a SINGLE-NODE
     separator, so every off-diagonal block of A1^{-1} has rank EXACTLY 1 --
     report 13's semiseparable identity (A^{-1})_ij = h x_i (1-x_j), i<=j,
     re-derived as "the separator has one node".
  2. HODLR partition (lexicographic bisection, leaf 64) of the 2-D A^{-1} at
     N=1024: per-level numerical ranks at 1e-6 / 1e-10 vs the a-priori bound
     32 (one grid row = the separator between adjacent row bands).  FINDING:
     for these WEAK-admissibility (adjacent) blocks the bound is TIGHT -- the
     singular values decay slowly ("logarithmic teeth") down to index 32 and
     then fall off a machine-precision cliff; the report-13 far block
     (strong admissibility, rows 0-7 x 24-31) has numerical rank 11 at 1e-8.
  3. 1-D vs 2-D contrast at N=1024: same partition, every 1-D block rank 1.
     Separator size 1 vs 32 = why 1-D is semiseparable and 2-D is "low-rank
     with logarithmic teeth".

PART B -- HODLR COMPRESSION AND THE PRECONDITIONER:
  4. M_r ~ A^{-1} by truncated SVD of the off-diagonal blocks of the exact
     dense inverse (ranks r in {1,2,4,8,16}, dense 64x64 leaves): spectral
     error, storage/compression, apply flops, kappa(M_r A), PCG iterations
     on the hot/cold-rod problem (error-based convention of decoupling.py)
     vs plain CG (73 its) and vs the block-Jacobi(2) baseline (kappa 19.29,
     12 its, report 13 SS4), and exact Richardson rho(I - M_r A).  TWO
     HONEST FINDINGS: (i) all M_r are SPD and PCG thrives, but UNDAMPED
     Richardson diverges for r <= 8 (lam_max(M_r A) > 2); optimally damped
     Richardson (omega = 2/(lmin+lmax)) converges for every r at
     rho = (kappa-1)/(kappa+1).  (ii) block-Jacobi(2) wins the raw iteration
     race (12 its) despite kappa 19.29, because 960 of its 1024 eigenvalues
     are EXACTLY 1 (report 13's clustering-beats-kappa lesson); kappa-wise
     the HODLR family passes it from r=4, iteration-wise only at r=16.
  5. 1-D punchline at n=1024: the rank-1 HODLR inverse IS the exact inverse
     to machine precision -> Richardson converges in ONE iteration.  In 1-D
     the hierarchical structure is not an approximation.
  7. results/hodlr_viz_data.json for the interactive page (ranks 2 and 8,
     exact viz data contract; fidelity of the JSON apply rule machine-checked).
  8. figures/anim14_hodlr_pcg.gif -- GD vs CG vs PCG+M_2 vs PCG+M_8 on the
     temperature field (static fallback for the interactive page).

HONEST COST NOTE (6): every M_r here is built from the exact dense inverse --
O(N^3) offline work, purely pedagogical.  Real hierarchical arithmetic builds
equivalent operators in near-linear time:
  * W. Hackbusch, "A sparse matrix arithmetic based on H-matrices. Part I:
    Introduction to H-matrices", Computing 62(2):89-108, 1999.
  * L. Grasedyck & W. Hackbusch, "Construction and arithmetics of
    H-matrices", Computing 70(4):295-334, 2003 (formatted H-arithmetic, H-LU).
  * P.-G. Martinsson & V. Rokhlin, "A fast direct solver for boundary
    integral equations in two dimensions", J. Comput. Phys. 205(1):1-23, 2005
    (recursive skeletonization; HSS-type direct solvers).
  * K. L. Ho & L. Ying, "Hierarchical interpolative factorization for
    elliptic operators: differential equations", Comm. Pure Appl. Math.
    69(8):1415-1451, 2016 (near-O(N) factorization of exactly our A).

Run from the repo root:
    uv run python python/experiments/hierarchical.py

Outputs: PASS/FAIL lines, results/hierarchical.json (every number the report
quotes), results/hodlr_viz_data.json (viz data contract), and figures/
  hierarchical_sval_decay.png, hierarchical_precond_sweep.png,
  anim14_hodlr_pcg.gif (+ anim14_hodlr_pcg_frames.png)      (PNGs dpi=150).
"""
import io
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcg import pcg
from poisson import laplacian_1d, poisson_2d

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"
RESDIR = ROOT / "results"
np.set_printoptions(precision=4, suppress=True)

T_START = time.time()
SEEDS = {"viz_check": 141, "rhs_1d": 142}
RESULTS = {"meta": {"seeds": SEEDS, "tol": 1e-10, "leaf": 64,
                    "ranks": [1, 2, 4, 8, 16]},
           "checks": [], "figures": [], "deviations": []}
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


def deviation(msg):
    print(f"  [deviation] {msg}")
    RESULTS["deviations"].append(msg)


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


def numrank(s, tol):
    """Numerical rank: # singular values > tol * s_1."""
    return int(np.sum(s > tol * s[0]))


def iters_to(rel, tol=1e-10):
    idx = np.nonzero(np.asarray(rel) <= tol)[0]
    return int(idx[0]) if idx.size else None


# ---------------------------------------------------------------------------
# solvers (error-based convention of decoupling.py: rel l2 error vs spsolve)
# ---------------------------------------------------------------------------
def cg_err(A, b, xstar, M=None, tol=1e-10, maxiter=2000, stop="err"):
    """PCG with the identical Hestenes-Stiefel recurrence as pcg.pcg
    (x0 = 0; bitwise cross-checked below), plus error histories rel2/relA
    (relA via e'Ae = -r'e, no extra matvec).  stop='err' halts on rel l2
    error <= tol; stop='res' mimics pcg's residual criterion."""
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
    n0_A = np.sqrt(max(b @ xstar, 0.0))   # e0' A e0 = b'x*
    rel2, relA = [1.0], [1.0]
    for _ in range(maxiter):
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
    return x, res_hist, rel2, relA


def gd_record(A, b, kmax):
    """Plain steepest descent (exact line search), returns x_0..x_kmax."""
    x = np.zeros_like(b)
    r = b.copy()
    xs = [x.copy()]
    for _ in range(kmax):
        Ar = A @ r
        alpha = (r @ r) / (r @ Ar)
        x = x + alpha * r
        r = r - alpha * Ar
        xs.append(x.copy())
    return xs


def pcg_record(A, b, M, kmax):
    """pcg.pcg recurrence recording x_0..x_kmax (holds x once converged)."""
    Mfun = M if M is not None else (lambda rr: rr)
    x = np.zeros_like(b)
    r = b.copy()
    z = Mfun(r)
    p = z.copy()
    rz = r @ z
    xs = [x.copy()]
    for _ in range(kmax):
        Ap = A @ p
        pAp = p @ Ap
        if pAp <= 0 or rz == 0.0:          # converged to machine precision
            xs.append(x.copy())
            continue
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        z = Mfun(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
        xs.append(x.copy())
    return xs


def kappa_from_Minv(Ad_, Minv):
    """kappa(M^{-1}A) for SPD M^{-1}: eig(M^{-1}A) = eig(R'AR), R = chol."""
    R = np.linalg.cholesky((Minv + Minv.T) / 2.0)
    w = np.linalg.eigvalsh(R.T @ Ad_ @ R)
    return w[-1] / w[0], w


# ---------------------------------------------------------------------------
# HODLR machinery
# ---------------------------------------------------------------------------
LEAF = 64


def hodlr_partition(Ntot, leaf=LEAF):
    """Recursive bisection of [0, Ntot).  Returns (off, leaves): off = UPPER
    off-diagonal blocks only [(r0, m, c0, nc, level)], leaves = [(r0, m)]."""
    off, leaves = [], []

    def rec(lo, hi, depth):
        m = hi - lo
        if m <= leaf:
            leaves.append((lo, m))
            return
        mid = lo + m // 2
        off.append((lo, mid - lo, mid, hi - mid, depth))
        rec(lo, mid, depth + 1)
        rec(mid, hi, depth + 1)

    rec(0, Ntot, 0)
    return off, leaves


def hodlr_svds(S, off):
    """Thin SVD of every upper off-diagonal block of S."""
    return [np.linalg.svd(S[r0:r0 + m, c0:c0 + nc], full_matrices=False)
            for (r0, m, c0, nc, _l) in off]


def build_hodlr(S, off, leaves, svds, r):
    """Dense HODLR approximation M_r of S: exact dense leaves, off-diagonal
    blocks truncated to rank r (U absorbs singular values; the lower block is
    the exact transpose of the upper's truncation, so M_r is EXACTLY
    symmetric).  Returns (M, storage_floats, viz_blocks) with viz_blocks per
    the results/hodlr_viz_data.json contract."""
    M = np.zeros_like(S)
    floats = 0
    viz = []
    for (lo, m) in leaves:
        D = S[lo:lo + m, lo:lo + m]
        M[lo:lo + m, lo:lo + m] = D
        floats += m * m
        viz.append({"r0": lo, "c0": lo, "m": m, "nc": m,
                    "type": "dense", "d": D.ravel()})
    for (r0, m, c0, nc, _l), (U, s, Vt) in zip(off, svds):
        k = min(r, s.size)
        Us = U[:, :k] * s[:k]                 # m x k, absorbs sigma
        V = Vt[:k].T                          # nc x k
        B = Us @ V.T
        M[r0:r0 + m, c0:c0 + nc] = B
        M[c0:c0 + nc, r0:r0 + m] = B.T
        floats += 2 * k * (m + nc)            # upper + mirrored lower factors
        viz.append({"r0": r0, "c0": c0, "m": m, "nc": nc, "type": "lowrank",
                    "k": k, "U": Us.ravel(), "V": V.ravel()})
        # lower block B^T = (V s)(U)^T -- its "U" also absorbs sigma
        viz.append({"r0": c0, "c0": r0, "m": nc, "nc": m, "type": "lowrank",
                    "k": k, "U": (V * s[:k]).ravel(), "V": U[:, :k].ravel()})
    return M, floats, viz


def sig7(x):
    """Round to 7 significant digits (viz data contract)."""
    return float(f"{x:.7g}")


def sig7_list(a):
    return [sig7(v) for v in np.asarray(a, dtype=np.float64).ravel()]


def apply_viz_blocks(blocks, x):
    """Python reimplementation of the JS apply rule from the viz contract:
    y = sum over blocks (dense: y[r0:r0+m] += D @ x[c0:c0+nc];
    lowrank: y[r0:r0+m] += U @ (V^T @ x[c0:c0+nc]))."""
    y = np.zeros_like(x)
    for blk in blocks:
        r0, c0, m, nc = blk["r0"], blk["c0"], blk["m"], blk["nc"]
        if blk["type"] == "dense":
            D = np.asarray(blk["d"]).reshape(m, nc)
            y[r0:r0 + m] += D @ x[c0:c0 + nc]
        else:
            k = blk["k"]
            U = np.asarray(blk["U"]).reshape(m, k)
            V = np.asarray(blk["V"]).reshape(nc, k)
            y[r0:r0 + m] += U @ (V.T @ x[c0:c0 + nc])
    return y


# ===========================================================================
# shared objects: n = 32 2-D Poisson, hot/cold-rod problem (house numbers)
# ===========================================================================
t0 = time.time()
n = 32
N = n * n
h = 1.0 / (n + 1)
A = poisson_2d(n)
Ad = A.toarray()
S2 = np.linalg.inv(Ad)                     # exact dense A^{-1} (pedagogical)
S2 = (S2 + S2.T) / 2.0     # inv() is symmetric only to rounding (~1e-19);
#                            symmetrize so the HODLR M_r below are EXACTLY
#                            symmetric (leaves bitwise symmetric, lower
#                            off-diag blocks assigned as exact transposes)

k1 = np.arange(1, n + 1)
lam1 = 4.0 * np.sin(k1 * np.pi * h / 2.0) ** 2 / h**2
kappa_A = float(lam1[-1] / lam1[0])
wA, VA = np.linalg.eigh(Ad)
Ah = (VA * np.sqrt(wA)) @ VA.T             # A^{1/2}, for eig(M A) = eig(Ah M Ah)
norm_S2 = 1.0 / wA[0]                      # ||A^{-1}||_2

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

RESULTS["meta"].update({"n": n, "N": N, "h": h, "kappa_A": kappa_A})
ok(f"kappa(A) at n=32 equals the house number 440.69 (measured {kappa_A:.2f})",
   abs(kappa_A - 440.69) < 0.01)

# bitwise cross-check of cg_err against pcg.pcg (as decoupling.py section 9)
x_ref, res_ref = pcg(A, b_rod, M=None, tol=1e-10, maxiter=2000)
_, res_mine, _, _ = cg_err(A, b_rod, xstar, M=None, stop="res", maxiter=2000)
mlen = min(len(res_ref), len(res_mine))
bit_dev = float(np.max(np.abs(np.asarray(res_ref[:mlen])
                              - np.asarray(res_mine[:mlen]))))
ok(f"cg_err reproduces pcg.pcg's residual history (max dev {bit_dev:.1e} "
   f"over {mlen} entries)", bit_dev <= 1e-13)

# ===========================================================================
# PART A -- WHERE THE LOW RANK COMES FROM
# ===========================================================================
print("== PART A: the separator theorem -- where the low rank comes from ==")
partA = {}

# ---- 1. the rank bound, proved by computation --------------------------------
cols_idx = lambda cols: (np.arange(n)[:, None] * n + np.asarray(cols)[None, :]).ravel()  # noqa: E731
iL = cols_idx(range(0, 15))       # grid cols 0..14   (|L| = 480)
iI = cols_idx([15, 16])           # separator cols 15..16 (|I| = 64)
iR = cols_idx(range(17, n))       # grid cols 17..31  (|R| = 480)
A_LR_sep = A[np.ix_(iL, iR)]
S_LR = S2[np.ix_(iL, iR)]
S_LI = S2[np.ix_(iL, iI)]
S_II = S2[np.ix_(iI, iI)]
S_IR = S2[np.ix_(iI, iR)]
ident_dev = float(np.max(np.abs(S_LR - S_LI @ np.linalg.solve(S_II, S_IR)))
                  / np.max(np.abs(S_LR)))
sv_LR = np.linalg.svd(S_LR, compute_uv=False)
cliff_LR = float(sv_LR[len(iI)] / sv_LR[0])          # s_65 / s_1
nr_LR_10 = numrank(sv_LR, 1e-10)
ok(f"no precision edges cross the separator: nnz(A_LR) = {A_LR_sep.nnz} "
   "(GMRF Markov property, report 13 SS3)", A_LR_sep.nnz == 0)
ok(f"rank bound identity Sigma_LR == Sigma_LI Sigma_II^-1 Sigma_IR holds to "
   f"machine precision (max rel dev {ident_dev:.1e}) => rank(Sigma_LR) <= "
   f"|I| = {len(iI)}", ident_dev < 1e-10)
ok(f"and the bound bites: s_65/s_1 of the 480x480 Sigma_LR = {cliff_LR:.1e} "
   f"(numerical rank at 1e-10 = {nr_LR_10} <= 64)",
   cliff_LR < 1e-10 and nr_LR_10 <= len(iI))
info(f"in fact rank(Sigma_LR) = {nr_LR_10}: col 15 ALONE separates L from "
     "{col 16} u R, so the MINIMAL separator (32 nodes) gives the tight "
     "bound -- the 2-column |I| = 64 bound is loose by 2x")
partA["separator_2d"] = {
    "split_cols": [[0, 14], [15, 16], [17, 31]], "sep_size": len(iI),
    "A_LR_nnz": int(A_LR_sep.nnz), "identity_rel_dev": ident_dev,
    "sv_LR_first20": sv_LR[:20], "s65_over_s1": cliff_LR,
    "numrank_1e-10": nr_LR_10}

# sharper 1-D statement: single-node separator => rank exactly 1
n64 = 64
h64 = 1.0 / (n64 + 1)
A64 = (laplacian_1d(n64) / h64**2).toarray()
S64 = np.linalg.inv(A64)
B64 = S64[:32, 32:]
sv64 = np.linalg.svd(B64, compute_uv=False)
ratio64 = float(sv64[1] / sv64[0])
x64 = np.arange(1, n64 + 1) * h64
G64 = h64 * np.outer(x64[:32], 1.0 - x64[32:])       # h x_i (1-x_j), i<j
formula_dev64 = float(np.max(np.abs(B64 - G64)) / np.max(np.abs(B64)))
ok(f"1-D (n=64): contiguous split has a SINGLE-node separator => off-diag "
   f"block of A1^-1 is rank 1 EXACTLY (s2/s1 = {ratio64:.1e} < 1e-14)",
   ratio64 < 1e-14)
ok(f"...and the block IS report 13's semiseparable identity h x_i (1-x_j) "
   f"(max rel dev {formula_dev64:.1e})", formula_dev64 < 1e-12)
partA["separator_1d_n64"] = {"s2_over_s1": ratio64, "sv_first5": sv64[:5],
                             "formula_rel_dev": formula_dev64}

# ---- 2. 2-D measured rank tables (HODLR partition of A^{-1}) ------------------
off2d, leaves2d = hodlr_partition(N)
svds2d = hodlr_svds(S2, off2d)
levels = sorted(set(l for (_r0, _m, _c0, _nc, l) in off2d))
SEP_BOUND = 32                    # one grid row separates adjacent row bands
tab2d = {}
print("  2-D rank table: HODLR off-diagonal blocks of A^{-1} (N=1024, "
      "lexicographic bisection, upper blocks; bound = grid-row separator = 32)")
print("  level  size  #blk | numrank@1e-6 max/mean | numrank@1e-10 max/mean |"
      " s33/s1 max   | bound")
for lv in levels:
    svs = [sv for (blk, (_u, sv, _vt)) in zip(off2d, svds2d) if blk[4] == lv]
    size = [blk[1] for blk in off2d if blk[4] == lv][0]
    nr6 = [numrank(s, 1e-6) for s in svs]
    nr10 = [numrank(s, 1e-10) for s in svs]
    cliff = [float(s[SEP_BOUND] / s[0]) for s in svs]
    tab2d[str(lv)] = {
        "block_size": size, "n_blocks_upper": len(svs), "bound": SEP_BOUND,
        "numrank_1e-6_max": max(nr6), "numrank_1e-6_mean": float(np.mean(nr6)),
        "numrank_1e-10_max": max(nr10),
        "numrank_1e-10_mean": float(np.mean(nr10)),
        "s33_over_s1_max": max(cliff),
        "sval_curve_first40": (svs[0][:40] / svs[0][0])}
    print(f"  {lv:>5d}  {size:>4d}  {len(svs):>3d} |"
          f" {max(nr6):>7d} / {np.mean(nr6):5.1f}  |"
          f" {max(nr10):>8d} / {np.mean(nr10):5.1f}  |"
          f" {max(cliff):.2e} | {SEP_BOUND}")
all_cliff = max(v["s33_over_s1_max"] for v in tab2d.values())
ok(f"a-priori bound holds EXACTLY at every level: rank(block) <= 32, "
   f"s33/s1 <= {all_cliff:.1e} (machine-precision cliff at the separator "
   "width)", all_cliff < 1e-12)
ok("weak admissibility is TIGHT: numerical rank at BOTH 1e-6 and 1e-10 "
   "EQUALS the bound 32 for every adjacent block (slow 'logarithmic teeth' "
   "decay above the cliff -- compression comes from the separator bound, "
   "not from spectral decay)",
   all(v["numrank_1e-6_max"] == SEP_BOUND
       and v["numrank_1e-10_max"] == SEP_BOUND for v in tab2d.values()))
deviation("task asked rank tables for block sizes 512/256/128; the level-3 "
          "(size-64) blocks of the same partition are included too for "
          "completeness -- same bound 32, same behavior.")

# report-13 continuity: far (strong-admissibility) block
Bfar = S2[: 8 * n, 24 * n:]
sv_far = np.linalg.svd(Bfar, compute_uv=False)
nr_far = numrank(sv_far, 1e-8)
ok(f"report-13 continuity: FAR 256x256 block (grid rows 0-7 x 24-31) has "
   f"numerical rank {nr_far} at 1e-8 (vs 32 for adjacent blocks: distance "
   "from the separator is what creates decay)", nr_far == 11)
partA["rank_table_2d"] = tab2d
partA["far_block"] = {"numrank_1e-8": nr_far, "sv_first20": sv_far[:20],
                      "rows": [0, 7], "cols_rows": [24, 31]}

# ---- 3. 1-D vs 2-D contrast at N = 1024 ---------------------------------------
n1 = 1024
h1 = 1.0 / (n1 + 1)
A1d = (laplacian_1d(n1) / h1**2).toarray()
S1 = np.linalg.inv(A1d)
xg1 = np.arange(1, n1 + 1) * h1
S1_exact = h1 * np.minimum.outer(xg1, xg1) * (1.0 - np.maximum.outer(xg1, xg1))
formula_dev1k = float(np.max(np.abs(S1 - S1_exact)) / S1.max())
ok(f"1-D n=1024: dense inverse matches the exact semiseparable formula "
   f"h min(x_i,x_j)(1-max(x_i,x_j)) (max rel dev {formula_dev1k:.1e})",
   formula_dev1k < 1e-10)
off1d, leaves1d = hodlr_partition(n1)
svds1d = hodlr_svds(S1, off1d)
tab1d = {}
print("  1-D vs 2-D contrast (same HODLR partition, N=1024):")
print("  level  size | 1-D rank@1e-14 max (s2/s1 max) | 2-D rank@1e-6 max | bound 1-D/2-D")
for lv in levels:
    svs = [sv for (blk, (_u, sv, _vt)) in zip(off1d, svds1d) if blk[4] == lv]
    size = [blk[1] for blk in off1d if blk[4] == lv][0]
    nr14 = [numrank(s, 1e-14) for s in svs]
    rat = [float(s[1] / s[0]) for s in svs]
    tab1d[str(lv)] = {"block_size": size, "n_blocks_upper": len(svs),
                      "bound": 1, "numrank_1e-14_max": max(nr14),
                      "s2_over_s1_max": max(rat)}
    print(f"  {lv:>5d}  {size:>4d} |  {max(nr14):>13d}  ({max(rat):.1e})   |"
          f" {tab2d[str(lv)]['numrank_1e-6_max']:>17d} |   1 / 32")
max_rat_1d = max(v["s2_over_s1_max"] for v in tab1d.values())
ok(f"1-D N=1024: EVERY off-diagonal block ({len(off1d)} upper blocks, sizes "
   f"512..64) has rank exactly 1 at 1e-14 (max s2/s1 = {max_rat_1d:.1e}) -- "
   "separator size 1 vs 32: why 1-D is semiseparable and 2-D is 'low-rank "
   "with logarithmic teeth'", max_rat_1d < 1e-14
   and all(v["numrank_1e-14_max"] == 1 for v in tab1d.values()))
partA["rank_table_1d"] = tab1d
partA["formula_dev_1d_n1024"] = formula_dev1k
RESULTS["part_a"] = partA

# ---- singular-value decay figure ----------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 5.2))
colors = plt.cm.viridis(np.linspace(0.0, 0.8, len(levels)))
for lv, c in zip(levels, colors):
    s = tab2d[str(lv)]["sval_curve_first40"]
    ax.semilogy(np.arange(1, len(s) + 1), s, "-o", ms=3, color=c, lw=1.4,
                label=f"2-D level {lv} (adjacent {tab2d[str(lv)]['block_size']}"
                      f"x{tab2d[str(lv)]['block_size']} block)")
ax.semilogy(np.arange(1, 21), sv_far[:20] / sv_far[0], "s--", ms=4,
            color="tab:red", lw=1.4,
            label="2-D FAR block (rows 0-7 x 24-31, report 13)")
s1d = svds1d[0][1][:40] / svds1d[0][1][0]
ax.semilogy(np.arange(1, 41), np.maximum(s1d, 1e-17), "^:", ms=3,
            color="tab:gray", lw=1.2, label="1-D level 0 (512x512 block)")
ax.axvline(SEP_BOUND, color="k", lw=1.0, ls="--")
ax.annotate("exact-rank cliff at 32\n= separator width (one grid row)",
            xy=(SEP_BOUND, 1e-8), xytext=(20.5, 3e-7), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("far block: rank 11 at $10^{-8}$\n(report 13 SS4)",
            xy=(11, sv_far[10] / sv_far[0]), xytext=(13, 3e-4), fontsize=8,
            color="tab:red", arrowprops=dict(arrowstyle="->", lw=0.8,
                                             color="tab:red"))
for tol, lab in [(1e-6, r"$10^{-6}$"), (1e-10, r"$10^{-10}$")]:
    ax.axhline(tol, color="0.75", lw=0.7, ls=":")
    ax.text(39.5, tol * 1.4, lab, fontsize=7, color="0.4", ha="right")
ax.set_xlabel("singular value index $i$")
ax.set_ylabel(r"$\sigma_i / \sigma_1$")
ax.set_title("HODLR off-diagonal blocks of $A^{-1}$: slow decay, then the\n"
             "separator-rank cliff -- 1-D cliff at 1, 2-D at 32, far blocks decay early")
ax.set_ylim(1e-17, 3)
ax.set_xlim(0.5, 40.5)
ax.legend(fontsize=7.5, loc="lower left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR / "hierarchical_sval_decay.png", dpi=150)
plt.close(fig)
RESULTS["figures"].append("hierarchical_sval_decay.png")
info(f"Part A done in {time.time()-t0:.1f}s")

# ===========================================================================
# PART B -- HODLR COMPRESSION AND THE PRECONDITIONER
# ===========================================================================
t0 = time.time()
print("== PART B: HODLR compression, M_r as a preconditioner ==")
partB = {}

# ---- baselines: plain CG and block-Jacobi(2) (report 13 SS4) -------------------
_, _, rel2_cg, relA_cg = cg_err(A, b_rod, xstar, M=None, maxiter=2000)
it_cg_l2 = iters_to(rel2_cg)
_, _, _, relA_cgA = cg_err(A, b_rod, xstar, M=None, stop="Aerr", maxiter=2000)
it_cg_A = iters_to(relA_cgA)
ok(f"plain CG baseline matches report 13's ladder: {it_cg_l2} its to rel l2 "
   f"err 1e-10 (house number 73); {it_cg_A} its to rel A-norm err 1e-10",
   it_cg_l2 == 73)

iL8 = cols_idx(range(0, 16))
iR8 = cols_idx(range(16, n))
A_LL = A[np.ix_(iL8, iL8)].tocsc()
A_RR = A[np.ix_(iR8, iR8)].tocsc()
lu_L, lu_R = spla.splu(A_LL), spla.splu(A_RR)


def M_bj(r):
    z = np.empty_like(r)
    z[iL8] = lu_L.solve(r[iL8])
    z[iR8] = lu_R.solve(r[iR8])
    return z


Minv_bj = np.zeros((N, N))
Minv_bj[np.ix_(iL8, iL8)] = np.linalg.inv(A_LL.toarray())
Minv_bj[np.ix_(iR8, iR8)] = np.linalg.inv(A_RR.toarray())
kap_bj, w_bj = kappa_from_Minv(Ad, Minv_bj)
n_unit_bj = int(np.sum(np.abs(w_bj - 1.0) < 1e-8))
_, _, rel2_bj, relA_bj = cg_err(A, b_rod, xstar, M=M_bj, maxiter=2000)
it_bj = iters_to(rel2_bj)
ok(f"block-Jacobi(2) baseline reproduces report 13: kappa(M^-1 A) = "
   f"{kap_bj:.2f} (house number 19.29); {it_bj} PCG its to rel err 1e-10 "
   "(ladder house number 12)", abs(kap_bj - 19.29) < 0.02 and it_bj == 12)
partB["baselines"] = {
    "cg_iters_l2": it_cg_l2, "cg_iters_Anorm": it_cg_A,
    "cg_rel2_curve": rel2_cg,
    "blockjacobi2_kappa": float(kap_bj), "blockjacobi2_iters_l2": it_bj,
    "blockjacobi2_n_unit_eigs": n_unit_bj,
    "blockjacobi2_rel2_curve": rel2_bj}

# ---- 4. the rank sweep -----------------------------------------------------------
RANKS = [1, 2, 4, 8, 16]
sweep = {}
Ms = {}
viz_blocks_by_rank = {}
print("  HODLR preconditioner sweep (leaf 64, dense-inverse blocks truncated "
      "to rank r):")
print("   r | rel err_2   | floats  (xN^2)  | apply flops (vs 2N^2) | "
      "min eig M   | kappa(M A) | rho(I-MA) | rho damped | PCG its")
for r in RANKS:
    M, floats, viz = build_hodlr(S2, off2d, leaves2d, svds2d, r)
    Ms[r] = M
    if r in (2, 8):
        viz_blocks_by_rank[r] = viz
    asym = float(np.max(np.abs(M - M.T)))
    relerr2 = float(np.max(np.abs(np.linalg.eigvalsh(M - S2))) / norm_S2)
    minM = float(np.linalg.eigvalsh(M)[0])
    P = Ah @ M @ Ah
    w = np.linalg.eigvalsh((P + P.T) / 2.0)
    spd = w[0] > 0
    kap = float(w[-1] / w[0]) if spd else None
    rho = float(np.max(np.abs(1.0 - w)))
    omega = 2.0 / (w[0] + w[-1])
    rho_damped = float(np.max(np.abs(1.0 - omega * w)))
    Mfun = (lambda MM: lambda rr: MM @ rr)(M)
    _, res_r, rel2_r, relA_r = cg_err(A, b_rod, xstar, M=Mfun, maxiter=2000)
    it_r = iters_to(rel2_r)
    relerr_vs_spsolve = rel2_r[-1]
    flops = 2 * floats
    # Solve-phase flop model, counted off pcg.pcg's actual loop (1 MAC = 2
    # flops): per iteration 1 A-matvec (2 nnz(A)) + 1 M-apply (2 x storage
    # floats: dense block 2mn, low-rank 2k(m+nc)) + six 2N-flop vector ops
    # (p.Ap, x-update, r-update, ||r||, r.z, p-update) = 12N; setup adds one
    # M-apply + one 2N dot (z0 = M b, rz0 = r.z).  Offline construction of
    # M_r is EXCLUDED (see cost_note: pedagogical dense-inverse build).
    per_iter = 2 * A.nnz + 12 * N + flops
    solve_flops = it_r * per_iter + (flops + 2 * N)
    sweep[str(r)] = {
        "rank": r, "asym": asym, "rel_err_2": relerr2,
        "storage_floats": floats, "storage_frac_N2": floats / N**2,
        "compression_factor": N**2 / floats,
        "apply_flops": flops, "dense_apply_flops": 2 * N**2,
        "apply_speedup": (2 * N**2) / flops,
        "min_eig_M": minM, "spd": bool(spd),
        "lam_min_MA": float(w[0]), "lam_max_MA": float(w[-1]),
        "kappa_MA": kap, "rho_richardson": rho,
        "omega_opt": float(omega), "rho_richardson_damped": rho_damped,
        "pcg_iters_l2": it_r, "pcg_final_relerr": float(relerr_vs_spsolve),
        "per_iter_flops": int(per_iter),
        "solve_flops_to_1e10": int(solve_flops),
        "pcg_rel2_curve": rel2_r}
    print(f"  {r:>2d} | {relerr2:.4e} | {floats:>6d} ({floats/N**2:.4f}) | "
          f"{flops:>7d} ({(2*N**2)/flops:4.1f}x) | {minM:.4e} | "
          f"{kap:10.3f} | {rho:9.3e} | {rho_damped:10.4f} | {it_r:>4d}")

sw = [sweep[str(r)] for r in RANKS]
ok("M_r is EXACTLY symmetric by construction (symmetrized S2 leaves + lower "
   f"block = transpose of the upper's truncation; max asym over all r = "
   f"{max(s['asym'] for s in sw):.1e}) -- no extra symmetrization needed",
   max(s["asym"] for s in sw) < 1e-16)
ok(f"all five M_r are SPD (min eig from {sw[0]['min_eig_M']:.2e} at r=1, "
   "all > 0) -- PCG is legitimate at every rank",
   all(s["spd"] and s["min_eig_M"] > 0 for s in sw))
ok("spectral error ||M_r - A^-1||_2/||A^-1||_2 decreases monotonically "
   "with rank: " + ", ".join(f"r={r}: {sweep[str(r)]['rel_err_2']:.1e}"
                             for r in RANKS),
   all(sw[i]["rel_err_2"] > sw[i + 1]["rel_err_2"] for i in range(4)))
ok("kappa(M_r A) decreases monotonically: "
   + ", ".join(f"{sweep[str(r)]['kappa_MA']:.2f}" for r in RANKS)
   + f" (vs kappa(A) = {kappa_A:.2f})",
   all(sw[i]["kappa_MA"] > sw[i + 1]["kappa_MA"] for i in range(4))
   and sw[0]["kappa_MA"] < kappa_A)
ok("PCG iterations decrease monotonically with rank: "
   + ", ".join(f"r={r}: {sweep[str(r)]['pcg_iters_l2']}" for r in RANKS)
   + f" (plain CG {it_cg_l2}, block-Jacobi(2) {it_bj})",
   all(sw[i]["pcg_iters_l2"] >= sw[i + 1]["pcg_iters_l2"] for i in range(4))
   and sw[-1]["pcg_iters_l2"] < it_cg_l2)
ok("all PCG runs converged and match spsolve (max final rel err "
   f"{max(s['pcg_final_relerr'] for s in sw):.1e} <= 1e-10)",
   all(s["pcg_final_relerr"] <= 1e-10 for s in sw))
ok(f"kappa race vs block-Jacobi(2): M_4 already beats it on conditioning "
   f"({sweep['4']['kappa_MA']:.2f} < {kap_bj:.2f}) and M_16 edges it on "
   f"iterations too ({sweep['16']['pcg_iters_l2']} < {it_bj})",
   sweep["4"]["kappa_MA"] < kap_bj
   and sweep["16"]["pcg_iters_l2"] < it_bj)
ok(f"HONEST: block-Jacobi(2) wins the raw iteration race ({it_bj} its at "
   f"kappa {kap_bj:.2f} vs M_8's {sweep['8']['pcg_iters_l2']} at kappa "
   f"{sweep['8']['kappa_MA']:.2f}) because its spectrum is CLUSTERED: "
   f"{n_unit_bj} of {N} eigenvalues of M^-1 A are EXACTLY 1 (= N - 2*32 "
   "interface pairs; report 13's clustering-beats-kappa lesson) -- the "
   "HODLR M_r spread their spectra across [lam_min, lam_max], so kappa is "
   "the honest predictor for them",
   n_unit_bj == N - 64 and it_bj < sweep["8"]["pcg_iters_l2"])
info("iters ~ c*sqrt(kappa) across the HODLR sweep: c = "
     + ", ".join(f"{sweep[str(r)]['pcg_iters_l2']/np.sqrt(sweep[str(r)]['kappa_MA']):.1f}"
                 for r in RANKS)
     + " (order-1 constant, no clustering to exploit)")
info("cost caveat: one block-Jacobi(2) apply = two exact 512-dof subdomain "
     "solves (a direct method in disguise), vs "
     f"{sweep['8']['apply_flops']} flops "
     f"(~{sweep['8']['apply_speedup']:.0f}x below a dense apply) for M_8")
rho_line = ", ".join(f"r={r}: {sweep[str(r)]['rho_richardson']:.3f}"
                     for r in RANKS)
ok(f"HONEST: undamped Richardson rho(I - M_r A) > 1 for r <= 8 ({rho_line}) "
   "-- lam_max(M_r A) > 2 even though kappa is small; only r=16 converges "
   "undamped", all(sweep[str(r)]["rho_richardson"] > 1.0 for r in [1, 2, 4, 8])
   and sweep["16"]["rho_richardson"] < 1.0)
ok("...but optimally damped Richardson (omega = 2/(lmin+lmax)) converges for "
   "EVERY r at rho = (kappa-1)/(kappa+1): "
   + ", ".join(f"{sweep[str(r)]['rho_richardson_damped']:.3f}" for r in RANKS),
   all(s["rho_richardson_damped"] < 1.0 for s in sw))
deviation("undamped Richardson diverges for r in {1,2,4,8} (rho > 1); "
          "reported honestly with the optimally damped variant alongside. "
          "PCG is unaffected (all M_r are SPD).")
deviation("the task hypothesized M_8 would beat block-Jacobi(2) on "
          "iterations; measured block-Jacobi(2) = 12 its (960 eigenvalues "
          "exactly 1 -- clustered spectrum), so only M_16 (11 its) beats it "
          "iteration-wise while kappa favors HODLR from r=4. Reported "
          "honestly with the clustering explanation.")
deviation("storage/flop counts include BOTH the upper and the mirrored lower "
          "off-diagonal factors, matching the viz-JSON apply rule; symmetric "
          "storage would halve the off-diagonal part.")
partB["sweep"] = sweep

# ---- sweep figure ---------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(7.2, 5.0))
ax2 = ax1.twinx()
rr = np.array(RANKS)
its = [sweep[str(r)]["pcg_iters_l2"] for r in RANKS]
kaps = [sweep[str(r)]["kappa_MA"] for r in RANKS]
l1, = ax1.loglog(rr, its, "o-", color="tab:blue", lw=1.8, ms=6,
                 label="PCG iterations to rel err $10^{-10}$")
l2, = ax2.loglog(rr, kaps, "s--", color="tab:red", lw=1.6, ms=5,
                 label=r"$\kappa(M_r A)$")
h1 = ax1.axhline(it_cg_l2, color="tab:blue", lw=0.9, ls=":",
                 label=f"plain CG [{it_cg_l2}]")
h2 = ax1.axhline(it_bj, color="tab:cyan", lw=0.9, ls="-.",
                 label=f"block-Jacobi(2) [{it_bj}]")
h3 = ax2.axhline(kappa_A, color="tab:red", lw=0.8, ls=":", alpha=0.6,
                 label=rf"$\kappa(A)$ = {kappa_A:.0f}")
h4 = ax2.axhline(kap_bj, color="tab:orange", lw=0.8, ls="-.", alpha=0.8,
                 label=rf"$\kappa$ block-Jacobi(2) = {kap_bj:.1f}")
ax1.set_xlabel("HODLR off-diagonal rank $r$")
ax1.set_ylabel("PCG iterations", color="tab:blue")
ax2.set_ylabel(r"$\kappa(M_r A)$", color="tab:red")
ax1.set_xticks(rr, [str(r) for r in RANKS])
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:red")
ax1.set_title("HODLR preconditioner sweep on the hot/cold-rod problem\n"
              "(leaf 64; blocks of the exact dense $A^{-1}$ truncated to rank $r$)")
ax1.legend(handles=[l1, l2, h1, h2, h3, h4], fontsize=8, loc="upper right")
ax1.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIGDIR / "hierarchical_precond_sweep.png", dpi=150)
plt.close(fig)
RESULTS["figures"].append("hierarchical_precond_sweep.png")

# ---- 5. 1-D punchline: the hierarchical structure is NOT an approximation --------
M1d, floats1d, _ = build_hodlr(S1, off1d, leaves1d, svds1d, 1)
w1, V1 = np.linalg.eigh(A1d)
relM1 = float(np.max(np.abs(np.linalg.eigvalsh(M1d - S1))) * w1[0])  # /||S1||
rng1 = np.random.default_rng(SEEDS["rhs_1d"])
b1 = rng1.standard_normal(n1)
x1_star = np.linalg.solve(A1d, b1)
x1_rich = M1d @ b1                       # ONE Richardson step from x0 = 0
rich_err = float(np.linalg.norm(x1_rich - x1_star) / np.linalg.norm(x1_star))
Ah1 = (V1 * np.sqrt(w1)) @ V1.T
P1 = Ah1 @ M1d @ Ah1
rho1d = float(np.max(np.abs(1.0 - np.linalg.eigvalsh((P1 + P1.T) / 2.0))))
ok(f"1-D n=1024: rank-1 HODLR inverse == EXACT inverse "
   f"(||M - A1^-1||_2/||A1^-1||_2 = {relM1:.1e})", relM1 < 1e-12)
ok(f"1-D punchline: Richardson converges in ONE iteration (rel err after 1 "
   f"step = {rich_err:.1e}; rho(I - M A1) = {rho1d:.1e}) -- in 1-D the "
   "hierarchical structure is not an approximation", rich_err < 1e-12)
partB["punchline_1d"] = {
    "n": n1, "rel_err_M_vs_inv": relM1, "storage_floats": floats1d,
    "compression_factor": n1**2 / floats1d,
    "richardson_1iter_rel_err": rich_err, "rho_richardson": rho1d}

# ---- 6. honest cost note ----------------------------------------------------------
partB["cost_note"] = {
    "text": ("All M_r here are assembled from the exact dense A^{-1} "
             "(O(N^3) offline) -- pedagogy, not production. Real H-matrix "
             "arithmetic (H-LU, HSS, recursive skeletonization, hierarchical "
             "interpolative factorization) constructs equivalent operators "
             "in O(N log N)-ish time directly from A."),
    "references": [
        "W. Hackbusch, 'A sparse matrix arithmetic based on H-matrices. "
        "Part I', Computing 62(2):89-108, 1999",
        "L. Grasedyck, W. Hackbusch, 'Construction and arithmetics of "
        "H-matrices', Computing 70(4):295-334, 2003",
        "P.-G. Martinsson, V. Rokhlin, 'A fast direct solver for boundary "
        "integral equations in two dimensions', J. Comput. Phys. "
        "205(1):1-23, 2005",
        "K. L. Ho, L. Ying, 'Hierarchical interpolative factorization for "
        "elliptic operators: differential equations', Comm. Pure Appl. "
        "Math. 69(8):1415-1451, 2016"]}
RESULTS["part_b"] = partB
info(f"Part B core done in {time.time()-t0:.1f}s")

# ---- 7. viz data export (exact contract) -------------------------------------------
t0 = time.time()
viz_obj = {
    "n": n, "h": sig7(h),
    "b": sig7_list(b_rod),
    "x_star": sig7_list(xstar),
    "x_star_norm": sig7(np.linalg.norm(xstar)),
    "hodlr": {}}
for r in (2, 8):
    blocks_out = []
    for blk in viz_blocks_by_rank[r]:
        if blk["type"] == "dense":
            blocks_out.append({"r0": blk["r0"], "c0": blk["c0"],
                               "m": blk["m"], "nc": blk["nc"],
                               "type": "dense", "d": sig7_list(blk["d"])})
        else:
            blocks_out.append({"r0": blk["r0"], "c0": blk["c0"],
                               "m": blk["m"], "nc": blk["nc"],
                               "type": "lowrank", "k": blk["k"],
                               "U": sig7_list(blk["U"]),
                               "V": sig7_list(blk["V"])})
    viz_obj["hodlr"][f"rank{r}"] = blocks_out
viz_path = RESDIR / "hodlr_viz_data.json"
with open(viz_path, "w") as fj:
    json.dump(viz_obj, fj, separators=(",", ":"))
viz_mb = viz_path.stat().st_size / 1024 / 1024
ok(f"results/hodlr_viz_data.json written: {viz_mb:.2f} MB < 5 MB "
   f"({len(viz_obj['hodlr']['rank2'])} + {len(viz_obj['hodlr']['rank8'])} "
   "blocks)", viz_mb < 5.0)

with open(viz_path) as fj:
    viz_reload = json.load(fj)
rngv = np.random.default_rng(SEEDS["viz_check"])
max_dev_apply = 0.0
for _ in range(3):
    xv = rngv.standard_normal(N)
    for r in (2, 8):
        y_json = apply_viz_blocks(viz_reload["hodlr"][f"rank{r}"], xv)
        y_ref = Ms[r] @ xv
        max_dev_apply = max(max_dev_apply,
                            float(np.linalg.norm(y_json - y_ref)
                                  / np.linalg.norm(y_ref)))
ok(f"viz JSON faithful: python reimplementation of the JS apply rule "
   f"reproduces M_r x on 3 random vectors (max rel dev {max_dev_apply:.1e} "
   "< 1e-6, ranks 2 and 8, after 7-significant-digit rounding)",
   max_dev_apply < 1e-6)
RESULTS["viz"] = {"file": "results/hodlr_viz_data.json", "size_mb": viz_mb,
                  "apply_max_rel_dev": max_dev_apply,
                  "n_blocks_rank2": len(viz_obj["hodlr"]["rank2"]),
                  "n_blocks_rank8": len(viz_obj["hodlr"]["rank8"])}
info(f"viz export done in {time.time()-t0:.1f}s")

# ---- 8. GIF: temperature estimate under four methods --------------------------------
t0 = time.time()
KMAX = 24
methods_anim = [
    ("plain GD", gd_record(A, b_rod, KMAX)),
    ("CG", pcg_record(A, b_rod, None, KMAX)),
    ("PCG + HODLR $M_2$", pcg_record(A, b_rod, lambda rr: Ms[2] @ rr, KMAX)),
    ("PCG + HODLR $M_8$", pcg_record(A, b_rod, lambda rr: Ms[8] @ rr, KMAX)),
]
xn = np.linalg.norm(xstar)
vm = float(np.abs(xstar).max())
fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.3))
ims, tts = [], []
for ax, (name, xs) in zip(axes, methods_anim):
    im = ax.imshow(xs[0].reshape(n, n), cmap="coolwarm", vmin=-vm, vmax=vm)
    ims.append(im)
    tts.append(ax.set_title("", fontsize=9))
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(ims[-1], ax=axes.tolist(), fraction=0.02, pad=0.02)
fig.suptitle("temperature estimate $x_k$ on the hot/cold-rod problem "
             "(fixed scale from $x^*$)", fontsize=11)


def update_anim(k):
    for im, tt, (name, xs) in zip(ims, tts, methods_anim):
        im.set_data(xs[k].reshape(n, n))
        err = np.linalg.norm(xs[k] - xstar) / xn
        tt.set_text(f"{name}\nk={k}, rel err {err:.1e}")
    return ims


SIZE_BUDGET = 2.5 * 1024 * 1024


def save_gif(figh, update, frames, fps, path, dpi=90):
    """Save GIF (loops forever); retry at reduced dpi if over 2.5 MB."""
    sz = None
    for d in [dpi, 80, 72, 64, 56]:
        if d > dpi:
            continue
        anim = FuncAnimation(figh, update, frames=frames, blit=False)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=d)
        sz = path.stat().st_size
        if sz < SIZE_BUDGET:
            ok(f"{path.relative_to(ROOT)} ({sz/1024/1024:.2f} MB < 2.5 MB, "
               f"{len(frames)} frames, fps {fps}, dpi {d})", True)
            return sz
    ok(f"{path.relative_to(ROOT)} over budget ({sz/1024/1024:.2f} MB)", False)
    return sz


def save_key_frames(figh, update, keys, path):
    """Render key frames at dpi 150 and hstack into one companion PNG."""
    imgs = []
    for k in keys:
        update(k)
        buf = io.BytesIO()
        figh.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        imgs.append(Image.open(buf).convert("RGB"))
    pad = 12
    wtot = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    htot = max(im.height for im in imgs)
    canvas = Image.new("RGB", (wtot, htot), "white")
    xpos = 0
    for im in imgs:
        canvas.paste(im, (xpos, 0))
        xpos += im.width + pad
    canvas.save(path)
    print(f"  [info] {path.relative_to(ROOT)} "
          f"({path.stat().st_size/1024:.0f} KB, key frames {list(keys)})")


frames = list(range(KMAX + 1)) + [KMAX] * 3      # 25 frames + 3-frame hold
save_key_frames(fig, update_anim, [0, 6, 24],
                FIGDIR / "anim14_hodlr_pcg_frames.png")
gif_sz = save_gif(fig, update_anim, frames, fps=4,
                  path=FIGDIR / "anim14_hodlr_pcg.gif")
plt.close(fig)
RESULTS["figures"] += ["anim14_hodlr_pcg.gif", "anim14_hodlr_pcg_frames.png"]
RESULTS["anim"] = {
    "gif_mb": gif_sz / 1024 / 1024, "frames": len(frames), "kmax": KMAX,
    "rel_err_at_k24": {name: float(np.linalg.norm(xs[KMAX] - xstar) / xn)
                       for name, xs in methods_anim}}
info("rel err at k=24: " + ", ".join(
    f"{name}: {v:.1e}" for name, v in RESULTS["anim"]["rel_err_at_k24"].items()))
info(f"GIF done in {time.time()-t0:.1f}s")

# ---- total solve flops to convergence (report 14 SS4 table column) -----------------
# Plain-CG baseline under the identical counting model (M-apply = 0 flops).
cg_per_iter = 2 * A.nnz + 12 * N
cg_solve_flops = it_cg_l2 * cg_per_iter + 2 * N
partB["flops"] = {
    "convention": "1 MAC = 2 flops; per iter = 2*nnz(A) + 12N + 2*storage; "
                  "setup = one M-apply + 2N; offline M_r construction excluded",
    "nnz_A": int(A.nnz), "vector_flops_per_iter": 12 * N,
    "cg_per_iter_flops": int(cg_per_iter),
    "cg_solve_flops_to_1e10": int(cg_solve_flops)}

# independent recount of the apply cost from the exported viz block lists
recount_ok = True
for r in (2, 8):
    rc = 0
    for blk in viz_blocks_by_rank[r]:
        if blk["type"] == "dense":
            rc += 2 * blk["m"] * blk["nc"]
        else:
            rc += 2 * blk["k"] * (blk["m"] + blk["nc"])
    recount_ok &= (rc == sweep[str(r)]["apply_flops"])
ok("flops column self-consistent: apply cost recounted from the exported "
   "viz block lists (dense 2mn + low-rank 2k(m+nc)) matches 2 x storage "
   "for r = 2 and 8", bool(recount_ok))

tot = [sweep[str(r)]["solve_flops_to_1e10"] for r in RANKS]
mono_dec = all(tot[i] > tot[i + 1] for i in range(len(tot) - 1))
cg_cheapest = cg_solve_flops < min(tot)
ok(f"HONEST flops race at N=1024: total solve flops fall monotonically with "
   f"rank ({', '.join(f'r={r}: {t/1e6:.2f}M' for r, t in zip(RANKS, tot))}) "
   f"— iterations drop faster than the apply grows — yet plain CG "
   f"({cg_solve_flops/1e6:.2f}M) undercuts every HODLR rank: kappa 440.69 "
   f"is too benign to amortize a {sweep['16']['apply_flops']/(2*A.nnz+12*N):.0f}x "
   f"costlier per-iteration apply at this N",
   mono_dec and cg_cheapest)

# ---- trim bulky curves, save results ------------------------------------------------
for key in ["cg_rel2_curve", "blockjacobi2_rel2_curve"]:
    partB["baselines"][key] = partB["baselines"][key][:200]
with open(RESDIR / "hierarchical.json", "w") as fj:
    json.dump(jsonable(RESULTS), fj, indent=2)
res_kb = (RESDIR / "hierarchical.json").stat().st_size / 1024
print(f"saved results/hierarchical.json ({res_kb:.0f} KB) and "
      f"results/hodlr_viz_data.json ({viz_mb:.2f} MB); "
      f"{N_FAIL} FAIL line(s); total {time.time()-T_START:.1f}s")
