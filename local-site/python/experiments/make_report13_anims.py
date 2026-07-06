"""Animated figures for report 13 -- "preconditioning as decoupling".

Four GIFs (each with a static 3-key-frame companion PNG, same stem +
_frames.png, dpi 150), all real dynamics, constants IDENTICAL to
python/experiments/decoupling.py so numbers match the report:

  anim13_gd_vs_cg.gif    -- 2-D quadratic race, kappa = 25 rotated ellipse:
                            GD zigzag (~30 exact-line-search steps) vs CG
                            done in exactly 2 steps.
  anim13_adi_sweep.gif   -- literal Peaceman-Rachford ADI double-sweeps on
                            the 32x32 hot/cold-rod problem, error field per
                            HALF-sweep: each half-sweep wipes the error
                            structure along one direction (striping flips).
  anim13_interface.gif   -- Richardson with the two-subdomain block-Jacobi
                            M (cols 0..15 | 16..31): interiors die fast,
                            the interface mode lingers; measured tail rate.
  anim13_cg_clusters.gif -- CG on the 3-cluster matrix (dim 400, centers
                            {1, 1e3, 1e6} x {300, 80, 20}, width 1e-3):
                            per-eigencomponent error stems per iteration --
                            each new variance scale costs CG one iteration.

Run from the repo root:
    uv run python python/experiments/make_report13_anims.py

Budget: each GIF < 2.5 MB (figsize <= 7x3.5 in, dpi <= 90, <= 45 frames,
fps 4-8, loop forever). Prints one PASS/size line per file written.
"""
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poisson import laplacian_1d, poisson_2d

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
})

SIZE_BUDGET = 2.5 * 1024 * 1024  # bytes


# ---------------------------------------------------------------------------
# shared machinery
# ---------------------------------------------------------------------------
def save_gif(fig, update, frames, fps, path, dpi=90):
    """Save GIF (loop forever via PillowWriter default loop=0); if over the
    2.5 MB budget, retry at reduced dpi."""
    for d in [dpi, 80, 72, 64, 56]:
        if d > dpi:
            continue
        anim = FuncAnimation(fig, update, frames=frames, blit=False)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=d)
        sz = path.stat().st_size
        if sz < SIZE_BUDGET:
            print(f"PASS: {path.relative_to(ROOT)} "
                  f"({sz/1024/1024:.2f} MB, {len(frames)} frames, "
                  f"fps {fps}, dpi {d})")
            return sz
    print(f"FAIL: {path.relative_to(ROOT)} over budget ({sz/1024/1024:.2f} MB)")
    return sz


def save_key_frames(fig, update, keys, path, labels=None):
    """Render key frames at dpi 150 and hstack into one companion PNG."""
    imgs = []
    for k in keys:
        update(k)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        imgs.append(Image.open(buf).convert("RGB"))
    pad = 12
    wtot = sum(im.width for im in imgs) + pad * (len(imgs) - 1)
    htot = max(im.height for im in imgs)
    canvas = Image.new("RGB", (wtot, htot), "white")
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + pad
    canvas.save(path)
    sz = path.stat().st_size
    print(f"PASS: {path.relative_to(ROOT)} ({sz/1024:.0f} KB, "
          f"key frames {list(keys)}, dpi 150)")
    return sz


def tail_rate(hist, m):
    h = np.asarray(hist, dtype=np.float64)
    m = min(m, len(h) - 1)
    return float((h[-1] / h[-1 - m]) ** (1.0 / m))


# ---------------------------------------------------------------------------
# shared objects, constants IDENTICAL to decoupling.py
# ---------------------------------------------------------------------------
n = 32
N = n * n
h = 1.0 / (n + 1)
A = poisson_2d(n)
d1 = laplacian_1d(n)
eye = sp.identity(n, format="csr")

k1 = np.arange(1, n + 1)
lam1 = 4.0 * np.sin(k1 * np.pi * h / 2.0) ** 2 / h**2

# hot/cold-rod RHS, exactly as decoupling.py / grid_regressions_multiscale.py
f = np.zeros((n, n))
for i, j in [(i, 4) for i in range(3, 9)]:
    f[i, j] = 1.0
for i, j in [(i, 27) for i in range(23, 29)]:
    f[i, j] = -1.0
b_rod = f.ravel()
xstar = spla.spsolve(A.tocsc(), b_rod)

MEASURED = {}

# ===========================================================================
# 1. anim13_gd_vs_cg -- 2-D quadratic race, kappa = 25 rotated ellipse
# ===========================================================================
print("== anim 1: GD zigzag vs CG (kappa = 25) ==")
KAPPA2 = 25.0
theta = np.deg2rad(30.0)
R2 = np.array([[np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]])
A2 = R2 @ np.diag([1.0, KAPPA2]) @ R2.T          # eigvecs: R2[:,0] flat dir
x_min2 = np.zeros(2)                             # b = 0, minimizer at origin
# start: equal energy in both eigendirections (maximal sustained zigzag,
# m0 = 1 is a fixed point pair of the GD mixing-ratio involution)
x0_2 = R2 @ np.array([1.0, 0.2])

GD_STEPS = 30


def gd_path(A_, x0, steps):
    x = x0.copy()
    P = [x.copy()]
    for _ in range(steps):
        r = -(A_ @ x)                            # b = 0
        alpha = (r @ r) / (r @ (A_ @ r))
        x = x + alpha * r
        P.append(x.copy())
    return np.array(P)


def cg_path(A_, x0, steps):
    x = x0.copy()
    r = -(A_ @ x)
    p = r.copy()
    rz = r @ r
    P = [x.copy()]
    for _ in range(steps):
        Ap = A_ @ p
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        P.append(x.copy())
        rz_new = r @ r
        p = r + (rz_new / rz) * p
        rz = rz_new
    return np.array(P)


P_gd = gd_path(A2, x0_2, GD_STEPS)
P_cg = cg_path(A2, x0_2, 2)
e0n = np.linalg.norm(x0_2)
err_gd = np.linalg.norm(P_gd, axis=1) / e0n      # x* = 0
err_cg = np.linalg.norm(P_cg, axis=1) / e0n
MEASURED["anim1"] = {
    "gd_relerr_after_30": float(err_gd[-1]),
    "gd_per_step_rate_tail": tail_rate(err_gd, 10),
    "cg_relerr_after_2": float(err_cg[2]),
}
print(f"  GD rel err after {GD_STEPS}: {err_gd[-1]:.2e} "
      f"(tail rate/step {MEASURED['anim1']['gd_per_step_rate_tail']:.4f}); "
      f"CG rel err after 2: {err_cg[2]:.2e}")

# contours of J(u) = 1/2 u'Au (b = 0)
g = np.linspace(-1.35, 1.35, 240)
GX, GY = np.meshgrid(g, g)
UV = np.stack([GX.ravel(), GY.ravel()])
JJ = 0.5 * np.einsum("ij,jk,ik->i", UV.T, A2, UV.T).reshape(GX.shape)
J0 = 0.5 * x0_2 @ A2 @ x0_2
levels = J0 * np.geomspace(3e-5, 1.4, 10)

fig1, axes1 = plt.subplots(1, 2, figsize=(7.0, 3.4), sharex=True, sharey=True)
art1 = {}
for ax, ttl in zip(axes1, ["gradient descent (exact line search)",
                           "conjugate gradients"]):
    ax.contour(GX, GY, JJ, levels=levels, colors="0.65", linewidths=0.7)
    ax.plot(0, 0, "k+", ms=10, mew=1.6, zorder=5)
    ax.plot(*x0_2, "ko", ms=5, zorder=5)
    ax.set_title(ttl, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.0, 1.35)
    ax.set_xticks([])
    ax.set_yticks([])
art1["gd"], = axes1[0].plot([], [], "-o", color="#D55E00", ms=2.6, lw=1.2)
art1["cg"], = axes1[1].plot([], [], "-o", color="#0072B2", ms=3.4, lw=1.5)
fig1.suptitle(r"minimizing $\frac{1}{2} u^\top A u$, rotated ellipse, "
              r"$\kappa = 25$", fontsize=12)


def update1(k):
    kg = min(k, GD_STEPS)
    kc = min(k, 2)
    art1["gd"].set_data(P_gd[: kg + 1, 0], P_gd[: kg + 1, 1])
    art1["cg"].set_data(P_cg[: kc + 1, 0], P_cg[: kc + 1, 1])
    axes1[0].set_xlabel(f"iteration {kg}: rel error {err_gd[kg]:.1e}")
    axes1[1].set_xlabel(f"iteration {kc}: rel error {err_cg[kc]:.1e}"
                        + ("  (done)" if kc == 2 else ""))


update1(0)
fig1.tight_layout(rect=(0, 0, 1, 0.96))
fig1.subplots_adjust(bottom=0.12)
frames1 = list(range(GD_STEPS + 1)) + [GD_STEPS] * 3
save_key_frames(fig1, update1, [0, 2, 30],
                FIGDIR / "anim13_gd_vs_cg_frames.png")
save_gif(fig1, update1, frames1, fps=6, path=FIGDIR / "anim13_gd_vs_cg.gif")
plt.close(fig1)

# ===========================================================================
# 2. anim13_adi_sweep -- literal Peaceman-Rachford half-sweeps, error field
# ===========================================================================
print("== anim 2: ADI half-sweeps on the hot/cold-rod problem ==")
H = (sp.kron(eye, d1) / h**2).tocsr()   # couples within grid ROWS
Vv = (sp.kron(d1, eye) / h**2).tocsr()  # couples within grid COLUMNS
sigma = float(np.sqrt(lam1[0] * lam1[-1]))     # geometric-mean shift, 207.03
ab = np.zeros((2, n))                          # banded (d1/h^2 + sigma I)
ab[0] = 2.0 / h**2 + sigma
ab[1, :-1] = -1.0 / h**2


def solve_rows(Y):
    """(H + sigma I)^{-1} Y: 32 independent tridiagonal solves, one per row."""
    return sla.solveh_banded(ab, Y.T, lower=True).T


def solve_cols(Y):
    """(V + sigma I)^{-1} Y: 32 independent tridiagonal solves, per column."""
    return sla.solveh_banded(ab, Y, lower=True)


N_DOUBLE = 10                                   # 20 half-sweeps
# Rough seeded start: with x0 = 0 the initial error -x* is already smooth in
# both directions and there is no structure for the half-sweeps to wipe; a
# rough x0 makes the mechanism visible AND excites all modes, so the measured
# double-sweep rate approaches the report's rho = 1 - f_min = 0.826.
rng_adi = np.random.default_rng(5)              # decoupling.py's adi seed
x_adi = rng_adi.standard_normal(N) * 1.5 * np.abs(xstar).max()
adi_fields = [(x_adi - xstar).reshape(n, n).copy()]
adi_titles = ["initial error (rough seeded $x_0$)"]
e0_norm = np.linalg.norm(x_adi - xstar)
adi_relerr = [1.0]
for m in range(1, N_DOUBLE + 1):
    # rows half-step: (H + sigma I) x = (sigma I - V) x + b
    x_adi = solve_rows((sigma * x_adi - Vv @ x_adi + b_rod).reshape(n, n)).ravel()
    adi_fields.append((x_adi - xstar).reshape(n, n).copy())
    adi_titles.append(f"after ROWS half-sweep {m}")
    adi_relerr.append(np.linalg.norm(x_adi - xstar) / e0_norm)
    # columns half-step: (V + sigma I) x = (sigma I - H) x + b
    x_adi = solve_cols((sigma * x_adi - H @ x_adi + b_rod).reshape(n, n)).ravel()
    adi_fields.append((x_adi - xstar).reshape(n, n).copy())
    adi_titles.append(f"after COLUMNS half-sweep {m}")
    adi_relerr.append(np.linalg.norm(x_adi - xstar) / e0_norm)

rate_adi_meas = tail_rate(adi_relerr[::2], 5)   # per DOUBLE sweep
rate_adi_theory = ((sigma - lam1[0]) / (sigma + lam1[0])) ** 2
rows_amp = float(max(adi_relerr[2 * m - 1] / adi_relerr[2 * m - 2]
                     for m in range(1, N_DOUBLE + 1)))
MEASURED["anim2"] = {
    "sigma": sigma,
    "relerr_after_20_half_sweeps": float(adi_relerr[-1]),
    "rate_per_double_sweep_measured": rate_adi_meas,
    "rate_per_double_sweep_theory": float(rate_adi_theory),
    "max_rows_half_step_amplification": rows_amp,
}
print(f"  sigma = {sigma:.3f}; rel err after 20 half-sweeps "
      f"{adi_relerr[-1]:.3e}; per-double-sweep rate measured "
      f"{rate_adi_meas:.4f} vs theory ((s-l1)/(s+l1))^2 = {rate_adi_theory:.4f}"
      f"; max transient rows-half-step amplification {rows_amp:.2f}x "
      f"(PR half-steps are not contractions; the double sweep is)")

vmax2 = np.abs(adi_fields[0]).max()
fig2, ax2 = plt.subplots(figsize=(4.6, 3.5))
im2 = ax2.imshow(adi_fields[0], cmap="coolwarm", vmin=-vmax2, vmax=vmax2)
cb2 = fig2.colorbar(im2, ax=ax2, fraction=0.046)
cb2.set_label("error $x - x^*$", fontsize=10)
ax2.set_xticks([0, 15, 31])
ax2.set_yticks([0, 15, 31])
ax2.set_ylabel("grid row $i$")


def update2(k):
    im2.set_data(adi_fields[k])
    ax2.set_title(f"ADI ($\\sigma = {sigma:.1f}$): {adi_titles[k]}",
                  fontsize=11)
    ax2.set_xlabel(f"grid col $j$   —   rel error {adi_relerr[k]:.2e}")


update2(0)
fig2.tight_layout()
frames2 = list(range(len(adi_fields)))          # 21 frames

# Static 3-frame strip: keep the mechanism frames [0, 1, 2] (the k=1 stripes
# ARE the point), but the k=1 rel-error readout GROWS (x4.10 over the initial
# 1.00) -- a genuine Peaceman-Rachford transient (half-steps are not
# contractions; only the full double sweep is, max amplification printed
# above).  Annotate that frame so the strip is not misread as a bug.
note2 = ax2.text(
    0.5, 0.02, "", transform=ax2.transAxes, ha="center", va="bottom",
    fontsize=7.5, color="0.10", zorder=6,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.55",
              boxstyle="round,pad=0.35", lw=0.6),
)


def update2_key(k):
    update2(k)
    if k % 2 == 1:                              # after a ROWS half-sweep
        amp = adi_relerr[k] / adi_relerr[k - 1]
        note2.set_text(
            f"rel error $\\uparrow\\times${amp:.2f}: genuine PR transient —\n"
            "half-steps are not contractions,\n"
            "only the full double sweep is"
        )
    else:
        note2.set_text("")


save_key_frames(fig2, update2_key, [0, 1, 2],
                FIGDIR / "anim13_adi_sweep_frames.png")
note2.set_text("")                              # keep GIF frames clean
save_gif(fig2, update2, frames2, fps=4, path=FIGDIR / "anim13_adi_sweep.gif")
plt.close(fig2)

# ===========================================================================
# 3. anim13_interface -- Richardson with two-subdomain block-Jacobi M
# ===========================================================================
print("== anim 3: block-Jacobi(2) Richardson, interface mode lingers ==")
cols_idx = lambda cols: (np.arange(n)[:, None] * n
                         + np.asarray(cols)[None, :]).ravel()  # noqa: E731
iL8 = cols_idx(range(0, 16))      # cols 0..15
iR8 = cols_idx(range(16, n))      # cols 16..31
lu_L = spla.splu(A[np.ix_(iL8, iL8)].tocsc())
lu_R = spla.splu(A[np.ix_(iR8, iR8)].tocsc())


def M_bj(r):
    z = np.empty_like(r)
    z[iL8] = lu_L.solve(r[iL8])
    z[iR8] = lu_R.solve(r[iR8])
    return z


N_RICH = 30
x_bj = np.zeros(N)
bj_fields, bj_prof, bj_relerr = [], [], []
e0_norm_bj = np.linalg.norm(x_bj - xstar)
for k in range(N_RICH + 1):
    e = (x_bj - xstar).reshape(n, n)
    bj_fields.append(e.copy())
    bj_prof.append((e**2).sum(axis=0))          # column-wise error energy
    bj_relerr.append(np.linalg.norm(e) / e0_norm_bj)
    if k < N_RICH:
        x_bj = x_bj + M_bj(b_rod - A @ x_bj)    # undamped Richardson

rate_bj_meas = tail_rate(bj_relerr, 10)
# asymptotic rate: continue the same iteration to 200 (transients still die
# out between iterations 30 and ~60; the animation shows the first 30)
x_bj_long = x_bj.copy()
bj_relerr_long = list(bj_relerr)
for k in range(N_RICH, 200):
    x_bj_long = x_bj_long + M_bj(b_rod - A @ x_bj_long)
    bj_relerr_long.append(np.linalg.norm(x_bj_long - xstar) / e0_norm_bj)
rate_bj_asym = tail_rate(bj_relerr_long, 20)
# theory: rho(I - M^{-1}A) from the generalized eigenproblem A v = lam M v
Ad = A.toarray()
Md_bj = Ad.copy()
Md_bj[np.ix_(iL8, iR8)] = 0.0
Md_bj[np.ix_(iR8, iL8)] = 0.0
w_bj = sla.eigh(Ad, Md_bj, eigvals_only=True)
rho_bj_theory = float(np.max(np.abs(1.0 - w_bj)))
MEASURED["anim3"] = {
    "relerr_after_30": float(bj_relerr[-1]),
    "tail_rate_last10_of_30": rate_bj_meas,
    "tail_rate_asymptotic_at_200": rate_bj_asym,
    "rho_theory_spectral": rho_bj_theory,
    "interface_energy_frac_final": float(bj_prof[-1][14:18].sum()
                                         / bj_prof[-1].sum()),
}
print(f"  rel err after {N_RICH}: {bj_relerr[-1]:.3e}; tail rate last 10 of "
      f"30: {rate_bj_meas:.4f}; asymptotic (200 its) {rate_bj_asym:.6f} vs "
      f"rho(I - M^-1 A) = {rho_bj_theory:.6f}; cols 14-17 carry "
      f"{MEASURED['anim3']['interface_energy_frac_final']:.0%} of the error "
      f"energy at iteration 30")

vmax3 = np.abs(bj_fields[0]).max()
prof_all = np.array(bj_prof)
fig3, axes3 = plt.subplots(1, 2, figsize=(7.0, 3.3),
                           gridspec_kw={"width_ratios": [1.0, 1.25]})
im3 = axes3[0].imshow(bj_fields[0], cmap="coolwarm", vmin=-vmax3, vmax=vmax3)
fig3.colorbar(im3, ax=axes3[0], fraction=0.046)
axes3[0].axvline(15.5, color="k", lw=0.8, ls="--")
axes3[0].set_xticks([0, 15, 31])
axes3[0].set_yticks([0, 15, 31])
axes3[0].set_ylabel("grid row $i$")
axes3[0].set_title("error field, cols 0–15 | 16–31", fontsize=10)
axes3[1].axvspan(14.5, 16.5, color="0.88", zorder=0,
                 label="interface cols 15/16")
axes3[1].plot(np.arange(n), prof_all[0], "--", color="0.6", lw=1.0,
              label="iteration 0")
ln3, = axes3[1].semilogy(np.arange(n), prof_all[0], "o-", color="#0072B2",
                         ms=2.6, lw=1.3, label="current iteration")
axes3[1].set_ylim(prof_all[prof_all > 0].min() * 0.4, prof_all.max() * 300)
axes3[1].set_xlabel("grid col $j$")
axes3[1].set_ylabel(r"column error energy $\sum_i e_{ij}^2$")
axes3[1].set_title("interiors die fast; the interface lingers", fontsize=10)
axes3[1].legend(loc="upper right", fontsize=7)


def update3(k):
    im3.set_data(bj_fields[k])
    ln3.set_ydata(np.maximum(prof_all[k], 1e-300))
    axes3[0].set_xlabel(f"iteration {k}")
    if k == N_RICH:
        fig3.suptitle(f"block-Jacobi(2) Richardson — iter {k}: rel error "
                      f"{bj_relerr[k]:.1e};  tail rate {rate_bj_asym:.3f} "
                      f"$= \\rho(I - M^{{-1}}A)$", fontsize=10)
    else:
        fig3.suptitle(f"Richardson $x \\leftarrow x + M^{{-1}}(b - Ax)$, "
                      f"block-Jacobi(2) — iteration {k}: "
                      f"rel error {bj_relerr[k]:.1e}", fontsize=10)


update3(0)
fig3.tight_layout(rect=(0, 0, 1, 0.94))
frames3 = list(range(N_RICH + 1))               # 31 frames
save_key_frames(fig3, update3, [0, 5, 30],
                FIGDIR / "anim13_interface_frames.png")
save_gif(fig3, update3, frames3, fps=6, path=FIGDIR / "anim13_interface.gif")
plt.close(fig3)

# ===========================================================================
# 4. anim13_cg_clusters -- CG error in the eigenbasis, 3-cluster spectrum
# ===========================================================================
print("== anim 4: CG on the 3-cluster spectrum, eigenbasis error stems ==")
# constants IDENTICAL to decoupling.py section 11 (seeds 8/9/10, width 1e-3)
dim = 400
rngq = np.random.default_rng(8)                 # SEEDS["cluster_q"]
Q, _ = np.linalg.qr(rngq.standard_normal((dim, dim)))
centers = np.repeat([1.0, 1e3, 1e6], [300, 80, 20])
upert = np.random.default_rng(9).uniform(-0.5, 0.5, dim)  # SEEDS["cluster_widths"]
b_cl = np.random.default_rng(10).standard_normal(dim)     # SEEDS["cluster_b"]
wdt = 1e-3
lam_cl = centers * (1.0 + wdt * upert)
A_cl = (Q * lam_cl) @ Q.T
A_cl = (A_cl + A_cl.T) / 2.0
xst_cl = Q @ ((Q.T @ b_cl) / lam_cl)            # exact eigen-solve

N_CG = 9
x_c = np.zeros(dim)
r = b_cl.copy()
p = r.copy()
rz = r @ r
coefs = [Q.T @ (x_c - xst_cl)]                  # eigenbasis error components
cl_relerr = [1.0]
xst_norm = np.linalg.norm(xst_cl)
for _ in range(N_CG):
    Ap = A_cl @ p
    alpha = rz / (p @ Ap)
    x_c = x_c + alpha * p
    r = r - alpha * Ap
    coefs.append(Q.T @ (x_c - xst_cl))
    cl_relerr.append(np.linalg.norm(x_c - xst_cl) / xst_norm)
    rz_new = r @ r
    p = r + (rz_new / rz) * p
    rz = rz_new

cl_slices = [slice(0, 300), slice(300, 380), slice(380, 400)]
cl_names = [r"$\lambda \approx 1$ ($\times$300)",
            r"$\lambda \approx 10^3$ ($\times$80)",
            r"$\lambda \approx 10^6$ ($\times$20)"]
cl_colors = ["#3b528b", "#21918c", "#fca50a"]   # viridis-family, cb-friendly
per_cluster = [[float(np.abs(c[s]).max()) for s in cl_slices] for c in coefs]
MEASURED["anim4"] = {
    "relerr_per_iter": [float(v) for v in cl_relerr],
    "max_abs_eigencomponent_per_cluster_per_iter": per_cluster,
}
for k in range(N_CG + 1):
    print(f"  it {k}: rel err {cl_relerr[k]:.3e}; max|comp| per cluster "
          f"(1 / 1e3 / 1e6) = {per_cluster[k][0]:.2e} / "
          f"{per_cluster[k][1]:.2e} / {per_cluster[k][2]:.2e}")

FLOOR = 1e-17
fig4, ax4 = plt.subplots(figsize=(7.0, 3.5))


def update4(k):
    ax4.clear()
    c = np.maximum(np.abs(coefs[k]), FLOOR)
    for s, name, col in zip(cl_slices, cl_names, cl_colors):
        ax4.vlines(lam_cl[s], FLOOR, c[s], color=col, lw=1.0, alpha=0.85,
                   label=name)
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlim(0.3, 4e6)
    ax4.set_ylim(FLOOR, 30)
    ax4.set_xlabel(r"eigenvalue $\lambda_i$ (log)"
                   f"   —   iteration {k}, rel error {cl_relerr[k]:.1e}")
    ax4.set_ylabel(r"error component $|(Q^\top(x_k - x^*))_i|$")
    ax4.set_title("CG error in the eigenbasis — 3 clusters "
                  r"$\{1, 10^3, 10^6\}\times\{300, 80, 20\}$, width $10^{-3}$:"
                  "\neach new variance scale costs CG one iteration",
                  fontsize=11)
    ax4.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig4.tight_layout()


frames4 = list(range(N_CG + 1)) + [N_CG] * 3    # 13 frames, hold the last
save_key_frames(fig4, update4, [0, 4, 9],
                FIGDIR / "anim13_cg_clusters_frames.png")
save_gif(fig4, update4, frames4, fps=4, path=FIGDIR / "anim13_cg_clusters.gif")
plt.close(fig4)

# ===========================================================================
# summary
# ===========================================================================
print("== summary ==")
for name in ["anim13_gd_vs_cg", "anim13_adi_sweep", "anim13_interface",
             "anim13_cg_clusters"]:
    for suff in [".gif", "_frames.png"]:
        pth = FIGDIR / (name + suff)
        okf = pth.exists() and pth.stat().st_size > 0
        print(f"{'OK ' if okf else 'MISSING'}: figures/{name}{suff} "
              f"{pth.stat().st_size/1024:.0f} KB" if okf
              else f"MISSING: figures/{name}{suff}")
import json  # noqa: E402
print("MEASURED:", json.dumps(MEASURED, indent=1))
