"""Numerical verification of every claim in reports/green_tents_explainer.html.

Checks 14 statements about the Green's function ("tent") of the 1-D Dirichlet
chain A = laplacian_1d(n)/h^2 with n = 31, h = 1/32: flank linearity, the
physical slopes (1-s, -s) under the u = A^{-1}(e_j/h) normalization, the
kink jump -1 (= the injected watt), the peak s(1-s) = current-divider
effective resistance to ground, the wall-exit flux split, superposition
(+2 W and -1.5 W sources vs the direct solve), wall bookkeeping
sum f_j(1-s_j) / sum f_j s_j, the lattice normalization (A^{-1}e_j has
slopes scaled by h), the free-vs-grounded Laplacian fact (principal
submatrix of the free path Laplacian == tridiag(-1,2,-1), end row sums 1),
reciprocity, the one-pin Brownian-motion case inv == h*min(x_i,x_j), and
the continuum tent formula G(x,s) = x(1-s) / s(1-x) entry-for-entry.

Run from the repo root:
    uv run python python/experiments/green_tents_checks.py

Expected output: 14 lines, all PASS; writes results/green_tents.json with
every number quoted in the report.
"""
import json
import sys
from pathlib import Path

import numpy as np
from numpy.linalg import inv, solve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poisson import laplacian_1d

np.set_printoptions(precision=6, suppress=True)
results = {"checks": []}
def ok(name, cond, err=None):
    status = "PASS" if cond else "FAIL"
    print(f"{status}: {name}" + (f"   (max err {err:.3g})" if err is not None else ""))
    results["checks"].append({"name": name, "pass": bool(cond),
                              **({"max_err": float(err)} if err is not None else {})})

n = 31; h = 1.0 / (n + 1)                    # 31 interior nodes, h = 1/32
A = (laplacian_1d(n) / h**2).toarray()       # stiffness / precision matrix
S = inv(A)                                   # covariance = h * G on the grid
x = np.arange(1, n + 1) * h                  # interior nodes x_i = i/32

# ---- the star of the show: physical tent at s = 10/32 = 0.3125 -------------
j = 9                                        # 0-based; x[9] = 10/32 = 0.3125
s = x[j]                                     # 0.3125
u = solve(A, np.eye(n)[j] / h)               # ONE WATT at s, walls clamped to 0
up = np.concatenate(([0.0], u, [0.0]))       # pad with the pinned wall values

# 1. flank linearity: all first differences constant on each side of the kink
#    (wall segments included -- the tent goes straight into the heat sinks)
dl = np.diff(up[: j + 2]) / h                # slopes of the j+1 left segments
dr = np.diff(up[j + 1:]) / h                 # slopes of the n+1-j right segments
err = max(np.abs(dl - dl[0]).max(), np.abs(dr - dr[0]).max())
ok("tent flanks are straight lines, wall to kink, on both sides", err < 1e-10, err)

# 2. slope values: left flank 1-s, right flank -s  (Fourier: flux = -u')
err = max(abs(dl[0] - (1 - s)), abs(dr[0] - (-s)))
ok("physical slopes: left = 1-s = 0.6875, right = -s = -0.3125", err < 1e-10, err)

# 3. kink: slope jump = -1 = the injected watt (-u'' = delta_s read pointwise)
kink = dr[0] - dl[0]
ok("slope jump at the source == -1 (the watt)", abs(kink - (-1.0)) < 1e-10,
   abs(kink + 1))

# 4. peak: u(s) = s(1-s) -- thermal resistance from s to ground
peak = u[j]
ok("peak u(s) == s(1-s) == 0.21484375", abs(peak - s * (1 - s)) < 1e-12,
   abs(peak - s * (1 - s)))

# 5. current divider: resistance s to the left wall, 1-s to the right, in
#    parallel: s(1-s)/(s+(1-s)) -- same number as the peak
par = 1.0 / (1.0 / s + 1.0 / (1 - s))
ok("effective resistance to ground == s || (1-s) == the peak", abs(par - peak) < 1e-12,
   abs(par - peak))

# 6. wall-exit split: 1-s watts out the LEFT wall, s out the RIGHT, summing to 1
out_left, out_right = up[1] / h, up[-2] / h  # boundary-segment slopes
err = max(abs(out_left - (1 - s)), abs(out_right - s), abs(out_left + out_right - 1))
ok("wall exits: 0.6875 W left + 0.3125 W right = 1 W", err < 1e-10, err)

# 7. parabolic envelope: peak of EVERY column j' equals s'(1-s')
T = S / h                                    # all physical tents at once
err = np.abs(np.diag(T) - x * (1 - x)).max()
ok("envelope: G(s,s) == s(1-s) for every source position", err < 1e-12, err)

# 8. reciprocity: G(x,s) == G(s,x) (a watt at s warms x as a watt at x warms s)
err = np.abs(T - T.T).max()
ok("reciprocity G(x,s) == G(s,x)", err < 1e-12, err)

# 9. continuum tent formula, entry for entry: G = x(1-s) below the diagonal,
#    s(1-x) above == min(x,s) - x s
G = np.minimum.outer(x, x) - np.outer(x, x)
err = np.abs(T - G).max()
ok("A^{-1}/h == the continuum tent x(1-s)/s(1-x), all 961 entries", err < 1e-12, err)

# 10. superposition: +2 W at s1 = 7/32 and -1.5 W at s2 = 25/32
j1, j2, f1, f2 = 6, 24, 2.0, -1.5
s1, s2 = x[j1], x[j2]                        # 0.21875, 0.78125
b = np.zeros(n); b[j1] = f1 / h; b[j2] = f2 / h
u2 = solve(A, b)                             # direct solve of A u = b
stacked = f1 * T[:, j1] + f2 * T[:, j2]      # two scaled tents, added
err = np.abs(u2 - stacked).max()
ok("superposition: +2W tent - 1.5W tent == direct solve of Au=b", err < 1e-12, err)

# 11. wall bookkeeping for free: out_left = sum f_j (1-s_j), out_right = sum f_j s_j
u2p = np.concatenate(([0.0], u2, [0.0]))
bk_left, bk_right = u2p[1] / h, u2p[-2] / h
want_left = f1 * (1 - s1) + f2 * (1 - s2)    #  1.234375
want_right = f1 * s1 + f2 * s2               # -0.734375
err = max(abs(bk_left - want_left), abs(bk_right - want_right),
          abs(bk_left + bk_right - (f1 + f2)))
ok("wall bookkeeping: 1.234375 W out left, -0.734375 W out right, net 0.5",
   err < 1e-10, err)

# 12. the lattice subtlety: A^{-1} e_j (unit LATTICE source = integral h of
#     physical source) has slopes h(1-s) and -h s -- h times too small
ulat = solve(A, np.eye(n)[j])
ulp = np.concatenate(([0.0], ulat, [0.0]))
err = max(abs(np.diff(ulp)[0] / h - h * (1 - s)), abs(ulat[j] - h * s * (1 - s)))
ok("lattice normalization: A^{-1}e_j slopes = h(1-s), peak = h s(1-s)",
   err < 1e-12, err)

# 13. free vs grounded: the free Laplacian of the 8-vertex path has degree
#     diagonal {1,2,...,2,1} and the constant vector in its kernel; deleting
#     the two pinned end rows/cols yields tridiag(-1,2,-1) with end row sums 1
#     (the wires to ground)
m = 8
L8 = np.diag([1.] + [2.] * (m - 2) + [1.])
L8 += np.diag(-np.ones(m - 1), 1) + np.diag(-np.ones(m - 1), -1)
sub = L8[1:m - 1, 1:m - 1]
rowsums = sub.sum(axis=1)
cond = (np.abs(L8 @ np.ones(m)).max() < 1e-15
        and np.allclose(sub, laplacian_1d(m - 2).toarray())
        and np.allclose(rowsums, [1, 0, 0, 0, 0, 1])
        and np.linalg.matrix_rank(sub) == m - 2)
ok("grounding: free path Laplacian [rows 2..7] == tridiag(-1,2,-1), row sums {1,0,0,0,0,1}",
   cond)

# 14. one pin: diag {2,...,2,1} matrix (right end free) inverts to Brownian
#     MOTION covariance h*min(x_i,x_j)
M1 = laplacian_1d(n).toarray(); M1[-1, -1] = 1.0
S1 = inv(M1 / h**2)
err = np.abs(S1 - h * np.minimum.outer(x, x)).max()
ok("one pin: inverse == h*min(x_i,x_j) (Brownian motion)", err < 1e-12, err)

npass = sum(c["pass"] for c in results["checks"])
print(f"\n{npass}/{len(results['checks'])} PASS")

# ---- every number the report quotes ----------------------------------------
results["quoted"] = {
    "n": n, "h": h, "n_plus_1": n + 1, "n_plus_2": n + 2,
    "max_err_all_checks": max(c.get("max_err", 0.0) for c in results["checks"]),
    "s": s, "one_minus_s": 1 - s,
    "peak": float(peak), "peak_exact": s * (1 - s),
    "slope_left": float(dl[0]), "slope_right": float(dr[0]),
    "kink_jump": float(kink),
    "out_left": float(out_left), "out_right": float(out_right),
    "watts_injected": 1.0,
    "envelope_max": 0.25,
    "superposition": {
        "f1": f1, "s1": s1, "f2": f2, "s2": s2,
        "peak_tent1": f1 * s1 * (1 - s1),          #  0.341796875
        "trough_tent2": f2 * s2 * (1 - s2),        # -0.25634765625
        "u_at_s1": float(u2[j1]),                  #  0.27001953125
        "u_at_s2": float(u2[j2]),                  # -0.16064453125
        "slope_left_segment": float(np.diff(u2p)[0] / h),    # 1.234375
        "slope_mid_segment": float((u2[j2] - u2[j1]) / (s2 - s1)),  # -0.765625
        "slope_right_segment": float(np.diff(u2p)[-1] / h),  # -0.734375
        "out_left": float(bk_left),                #  1.234375
        "out_right": float(bk_right),              # -0.734375
        "net_watts": f1 + f2,                      #  0.5
        "zero_crossing": 4.0 / 7.0,
        "bk_f1_left": f1 * (1 - s1),               #  1.5625
        "bk_f2_left": f2 * (1 - s2),               # -0.328125
        "bk_f1_right": f1 * s1,                    #  0.4375
        "bk_f2_right": f2 * s2,                    # -1.171875
    },
    "lattice": {
        "slope_left": h * (1 - s),                 # 0.021484375
        "peak": h * s * (1 - s),                   # 0.0067138671875
    },
    "grounded": {
        "path_vertices": m, "interior": m - 2,
        "free_diagonal": [1, 2, 2, 2, 2, 2, 2, 1],
        "sub_diagonal": [2, 2, 2, 2, 2, 2],
        "sub_row_sums": [int(r) for r in rowsums],
    },
    "one_pin_peak_var": float(S1[-1, -1]),         # h * x_n = free-end variance
    "checks_total": len(results["checks"]),
    "checks_passed": npass,
}

out = Path(__file__).resolve().parents[2] / "results" / "green_tents.json"
out.write_text(json.dumps(results, indent=2))
print(f"wrote {out}")
