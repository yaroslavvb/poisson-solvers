"""Five static schematic diagrams for reports/13-preconditioning-as-decoupling.md.

Figures (all -> figures/, dpi=150 PNG, prefix diagram13_):

  1. diagram13_adi_rows_cols.png   -- SS2.2 "two families of subproblems": rows
     pass vs columns pass on an 8x8 node grid (report experiments use 32x32).
  2. diagram13_chain.png           -- SS3 1-D chain: walls, -1/h^2 springs,
     Markov blanket, and the tridiagonal spy of A.
  3. diagram13_rank1_triangle.png  -- SS3 semiseparable identity, COMPUTED at
     n=16: (A^{-1})_ij = h x_i (1 - x_j) for i <= j.  Crashes if the max
     upper-triangle deviation exceeds 1e-12.
  4. diagram13_subdomains.png      -- SS4.1 space split: L / separator I / R on
     a 16x16 grid; nnz(A_LR) = 0 is machine-checked on poisson_2d(16).
  5. diagram13_energy_contours.png -- SS1/SS5: coupled kappa=25 quadratic with a
     REAL exact-line-search GD zigzag (12 computed steps) vs whitened kappa=1
     (one step to center).

Only figure-generating code lives here; the operators come from poisson.py.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poisson import laplacian_1d, poisson_2d  # noqa: E402

FIGDIR = Path(__file__).resolve().parents[2] / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

# seaborn "muted" qualitative palette -- 8 distinct muted, colorblind-friendly hues
MUTED8 = [
    "#4878d0", "#ee854a", "#6acc64", "#d65f5f",
    "#956cb4", "#8c613c", "#dc7ec0", "#82c6e2",
]
BLUE, ORANGE, RED, GRAY = "#4878d0", "#ee854a", "#d65f5f", "#8a8a8a"


def save(fig, name):
    path = FIGDIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    size = path.stat().st_size
    assert size > 0, f"{name} is empty"
    print(f"PASS {name} written, {size} bytes ({size / 1024:.0f} KB)")
    return path


# ----------------------------------------------------------------------------
# 1. diagram13_adi_rows_cols.png -- the SS2.2 two-families reminder
# ----------------------------------------------------------------------------
def fig_adi_rows_cols():
    m = 8
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    fig.subplots_adjust(wspace=0.28)

    for ax, mode in zip(axes, ("rows", "cols")):
        for k in range(m):  # k indexes the strip (row or column)
            c = MUTED8[k]
            if mode == "rows":
                xs, ys = np.arange(m), np.full(m, m - 1 - k)
            else:
                xs, ys = np.full(m, k), np.arange(m)
            ax.plot(xs, ys, "-", color=c, lw=2.2, zorder=1)
            ax.scatter(xs, ys, s=70, color=c, edgecolor="white",
                       linewidth=0.8, zorder=2)
        ax.set_xlim(-0.7, m - 0.3)
        ax.set_ylim(-0.7, m - 0.3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    axes[0].set_title(
        "Rows pass: $(H + \\sigma I)$ — 8 independent\n1-D chain problems"
    )
    axes[1].set_title(
        "Columns pass: $(V + \\sigma I)$ — 8 independent\n1-D chain problems"
    )

    # two-headed arrow between the panels
    arrow = FancyArrowPatch(
        (0.472, 0.50), (0.528, 0.50), transform=fig.transFigure,
        arrowstyle="<->", mutation_scale=20, lw=1.8, color="0.25",
    )
    fig.add_artist(arrow)
    fig.text(0.5, 0.545, "alternate", ha="center", va="bottom",
             fontsize=12, color="0.25")

    fig.text(
        0.5, 0.035,
        "each colored strip = one tridiagonal solve, $O(n)$;   "
        "one $M^{-1}$ apply = $2n$ solves = $O(N)$",
        ha="center", va="bottom", fontsize=12,
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.13,
                        wspace=0.28)
    save(fig, "diagram13_adi_rows_cols.png")


# ----------------------------------------------------------------------------
# 2. diagram13_chain.png -- the SS3 1-D chain + tridiagonal spy
# ----------------------------------------------------------------------------
def fig_chain():
    m = 8
    h = 1.0 / (m + 1)
    A = (laplacian_1d(m) / h**2).toarray()

    fig = plt.figure(figsize=(10, 6.2))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.05, 1.0],
                           width_ratios=[1, 1.15, 1], hspace=0.32)
    ax = fig.add_subplot(gs[0, :])

    hi = 5           # highlighted interior node (x-position)
    nb = (hi - 1, hi + 1)

    # edges (springs) between all neighbors, walls included
    for x in range(0, m + 1):
        ax.plot([x, x + 1], [0, 0], "-", color="0.45", lw=1.8, zorder=1)
    ax.text(1.5, 0.30, "$-1/h^2$", ha="center", va="bottom", fontsize=12)

    # wall nodes u = 0 (gray, hatched squares)
    for xw in (0, m + 1):
        ax.add_patch(Rectangle((xw - 0.26, -0.26), 0.52, 0.52,
                               facecolor="0.85", edgecolor="0.4",
                               hatch="////", zorder=3))
        ax.text(xw, -0.62, "$u = 0$", ha="center", va="top", fontsize=11)

    # interior nodes
    for x in range(1, m + 1):
        if x == hi:
            fc, ec, lw = ORANGE, "#b35a1f", 2.0
        elif x in nb:
            fc, ec, lw = "#f8d8bf", "#b35a1f", 1.4   # shaded neighbors
        else:
            fc, ec, lw = "white", "0.35", 1.2
        ax.add_patch(Circle((x, 0), 0.22, facecolor=fc, edgecolor=ec,
                            linewidth=lw, zorder=3))
    ax.text(hi, -0.42, "$u_i$", ha="center", va="top", fontsize=12)

    # square bracket over the two shaded neighbors
    bl, br, y0, y1 = nb[0] - 0.35, nb[1] + 0.35, 0.42, 0.60
    ax.plot([bl, bl, br, br], [y0, y1, y1, y0], "-", color="0.2", lw=1.4)
    ax.text((bl + br) / 2, y1 + 0.14,
            "Markov blanket: given these two,\n"
            "$u_i$ is independent of everything else",
            ha="center", va="bottom", fontsize=12)

    ax.set_xlim(-0.9, m + 1.9)
    ax.set_ylim(-1.15, 1.75)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("the 1-D chain: $A_{1\\mathrm{d}} = d_1/h^2$, "
                 "coupling only between neighbors", pad=2)

    # small spy plot of tridiagonal A (dots)
    axs = fig.add_subplot(gs[1, 1])
    axs.spy(A, markersize=9, marker="o", color=BLUE)
    axs.set_xticks(range(m))
    axs.set_yticks(range(m))
    axs.set_xticklabels(range(1, m + 1), fontsize=9)
    axs.set_yticklabels(range(1, m + 1), fontsize=9)
    axs.tick_params(length=0)
    axs.set_title("$A$ tridiagonal: coupling = graph edges", fontsize=11, pad=10)

    # No tight_layout here: the GridSpec above carries an explicit hspace, so
    # matplotlib's tight_layout would treat every axes as incompatible, warn,
    # and change nothing (it returns {} for locally-modified gridspecs).  The
    # layout is fully determined by the GridSpec ratios/hspace already.
    save(fig, "diagram13_chain.png")


# ----------------------------------------------------------------------------
# 3. diagram13_rank1_triangle.png -- semiseparable identity, computed at n=16
# ----------------------------------------------------------------------------
def fig_rank1_triangle():
    n = 16
    h = 1.0 / (n + 1)
    A = (laplacian_1d(n) / h**2).toarray()
    Ainv = np.linalg.inv(A)
    x = np.arange(1, n + 1) * h                     # x_i = i h
    outer = h * np.outer(x, 1.0 - x)                # h * x (1-x)^T

    iu = np.triu(np.ones((n, n), dtype=bool))       # i <= j
    dev = np.abs(Ainv - outer)
    maxdev = dev[iu].max()
    if maxdev > 1e-12:
        raise RuntimeError(
            f"semiseparable identity violated: max upper-tri deviation "
            f"{maxdev:.3e} > 1e-12"
        )
    print(f"PASS rank-1 triangle identity verified: max upper-tri deviation "
          f"= {maxdev:.1e} (<= 1e-12)")

    fig = plt.figure(figsize=(12, 5.2))
    outer_gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.12, 1], wspace=0.32,
                                 left=0.05, right=0.96, top=0.78, bottom=0.08)
    vmax = max(Ainv.max(), outer.max())

    # (a) A^{-1}
    axa = fig.add_subplot(outer_gs[0, 0])
    ima = axa.imshow(Ainv, cmap="viridis", vmin=0, vmax=vmax)
    axa.set_title("(a)  $A_{1\\mathrm{d}}^{-1}$ (computed, $n=16$)")
    axa.set_xticks([0, 7, 15]); axa.set_xticklabels([1, 8, 16], fontsize=9)
    axa.set_yticks([0, 7, 15]); axa.set_yticklabels([1, 8, 16], fontsize=9)
    fig.colorbar(ima, ax=axa, fraction=0.046, pad=0.04)

    # (b) outer product with side/top strips
    gsb = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer_gs[0, 1],
        width_ratios=[1, 11], height_ratios=[1, 11], wspace=0.06, hspace=0.06,
    )
    axt = fig.add_subplot(gsb[0, 1])                # top strip: (1-x_j)
    axl = fig.add_subplot(gsb[1, 0])                # left strip: x_i
    axm = fig.add_subplot(gsb[1, 1])                # full outer product
    axt.imshow((1.0 - x)[None, :], cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axl.imshow(x[:, None], cmap="viridis", aspect="auto", vmin=0, vmax=1)
    axm.imshow(outer, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
    for a in (axt, axl, axm):
        a.set_xticks([]); a.set_yticks([])
    axt.set_title("(b)  outer product $h\\,x\\,(1-x)^\\top$ "
                  "(same color scale as a)", fontsize=12, pad=14)
    axt.text(1.02, 0.5, "$1-x_j$", transform=axt.transAxes,
             ha="left", va="center", fontsize=10)
    axl.text(0.5, -0.02, "$x_i$", transform=axl.transAxes,
             ha="center", va="top", fontsize=10)

    # (c) deviation, upper triangle only
    axc = fig.add_subplot(outer_gs[0, 2])
    masked = np.where(iu, dev, np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.92")
    imc = axc.imshow(masked, cmap=cmap)
    axc.set_title("(c)  $|A^{-1} - h\\,x(1-x)^\\top|$ on $i \\leq j$\n"
                  f"upper-triangle deviation: {maxdev:.1e}")
    axc.set_xticks([0, 7, 15]); axc.set_xticklabels([1, 8, 16], fontsize=9)
    axc.set_yticks([0, 7, 15]); axc.set_yticklabels([1, 8, 16], fontsize=9)
    axc.text(0.28, 0.22, "masked\n($i > j$)", transform=axc.transAxes,
             ha="center", va="center", fontsize=10, color="0.35")
    cb = fig.colorbar(imc, ax=axc, fraction=0.046, pad=0.04)
    cb.set_ticks([0.0, maxdev / 2, maxdev])
    cb.ax.set_yticklabels(["0", f"{maxdev / 2:.0e}", f"{maxdev:.0e}"],
                          fontsize=9)

    fig.suptitle("the upper triangle of $A^{-1}$ is rank-1:  "
                 "$(A^{-1})_{ij} = h\\,x_i\\,(1 - x_j)$ for $i \\leq j$",
                 fontsize=13, y=0.955)
    save(fig, "diagram13_rank1_triangle.png")


# ----------------------------------------------------------------------------
# 4. diagram13_subdomains.png -- the SS4 space split
# ----------------------------------------------------------------------------
def fig_subdomains():
    m = 16
    # machine-check the headline claim on the real operator: L = cols 0-6,
    # I = cols 7-8, R = cols 9-15 (flat index k = i*m + j, j = column)
    A = poisson_2d(m).tocsr()
    cols = np.arange(m * m) % m
    Lidx = np.where(cols <= 6)[0]
    Ridx = np.where(cols >= 9)[0]
    nnz_LR = A[np.ix_(Lidx, Ridx)].nnz
    assert nnz_LR == 0, f"expected nnz(A_LR)=0, got {nnz_LR}"
    print(f"PASS subdomain split verified on poisson_2d(16): nnz(A_LR) = {nnz_LR}")

    fig, ax = plt.subplots(figsize=(8.2, 9.0))

    def node_color(j):
        if j <= 6:
            return BLUE
        if j >= 9:
            return RED
        return GRAY

    # all 5-point stencil edges, thin gray; L-I and I-R edges in black
    for i in range(m):          # i = grid row -> y = m-1-i
        y = m - 1 - i
        for j in range(m - 1):  # horizontal edges (j, j+1)
            black = j in (6, 8)  # blue-gray (6-7) and gray-red (8-9) edges
            ax.plot([j, j + 1], [y, y],
                    color="black" if black else "0.78",
                    lw=1.6 if black else 0.6,
                    zorder=2 if black else 1)
    for j in range(m):          # vertical edges
        for y in range(m - 1):
            ax.plot([j, j], [y, y + 1], color="0.78", lw=0.6, zorder=1)

    # nodes
    for j in range(m):
        ys = np.arange(m)
        marker = "D" if j in (7, 8) else "o"
        ax.scatter(np.full(m, j), ys, s=42 if marker == "D" else 46,
                   marker=marker, color=node_color(j),
                   edgecolor="white", linewidth=0.7, zorder=3)

    # region labels
    ax.text(3.0, m - 0.25, "$L$", ha="center", fontsize=15, color=BLUE)
    ax.text(7.5, m - 0.25, "separator $I$", ha="center", fontsize=13, color="0.35")
    ax.text(12.0, m - 0.25, "$R$", ha="center", fontsize=15, color=RED)

    ax.set_xlim(-0.8, m - 0.2)
    ax.set_ylim(-3.4, m + 0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("the space split: every path from $L$ to $R$ crosses the "
                 "two-column separator $I$", pad=10)

    ax.text(7.5, -1.5,
            "$\\mathrm{nnz}(A_{LR}) = 0$: no direct coupling between $L$ and $R$",
            ha="center", va="center", fontsize=13)
    ax.text(7.5, -2.6,
            "$\\mathrm{Cov}(u_L,\\, u_R\\ \\mathrm{given}\\ u_I) = 0$ exactly "
            "(GMRF Markov property)",
            ha="center", va="center", fontsize=13)

    fig.tight_layout()
    save(fig, "diagram13_subdomains.png")


# ----------------------------------------------------------------------------
# 5. diagram13_energy_contours.png -- coupled zigzag vs whitened one-step
# ----------------------------------------------------------------------------
def gd_exact_line_search(A, x0, nsteps):
    """Real steepest descent with exact line search on J(u) = 1/2 u'Au."""
    xs = [np.asarray(x0, dtype=float)]
    x = xs[0].copy()
    for _ in range(nsteps):
        g = A @ x
        gg = g @ g
        if gg == 0.0:
            break
        alpha = gg / (g @ (A @ g))
        x = x - alpha * g
        xs.append(x.copy())
    return np.array(xs)


def fig_energy_contours():
    kappa = 25.0
    theta = np.deg2rad(30.0)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    Ac = R @ np.diag([1.0, kappa]) @ R.T          # coupled, kappa = 25, rotated
    Aw = np.eye(2)                                # whitened, kappa = 1

    x0 = R @ np.array([2.0, 0.55])                # same start point in both panels

    path_c = gd_exact_line_search(Ac, x0, 12)
    path_w = gd_exact_line_search(Aw, x0, 1)      # one step lands on the center
    assert np.linalg.norm(path_w[-1]) < 1e-14, "whitened GD must one-step"
    errs = np.linalg.norm(path_c, axis=1)
    print(f"PASS energy-contour GD computed: coupled |u_12|/|u_0| = "
          f"{errs[-1] / errs[0]:.2e} after 12 steps; whitened |u_1| = "
          f"{np.linalg.norm(path_w[-1]):.1e} after 1 step")

    grid = np.linspace(-2.6, 2.6, 400)
    X, Y = np.meshgrid(grid, grid)

    def J(A):
        return 0.5 * (A[0, 0] * X**2 + 2 * A[0, 1] * X * Y + A[1, 1] * Y**2)

    J0c = 0.5 * x0 @ (Ac @ x0)
    J0w = 0.5 * x0 @ (Aw @ x0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, A, J0, path, title in (
        (axes[0], Ac, J0c, path_c,
         "coupled: $\\kappa = 25$ — GD zigzags for 12 steps"),
        (axes[1], Aw, J0w, path_w,
         "whitened: $\\kappa = 1$ — one step to the center"),
    ):
        levels = J0 * np.geomspace(3e-3, 1.25, 11)
        ax.contour(X, Y, J(A), levels=levels, cmap="viridis",
                   linewidths=0.9, alpha=0.85)
        ax.plot(path[:, 0], path[:, 1], "o-", color=RED, lw=1.6,
                markersize=4.5, zorder=3)
        ax.plot(*x0, marker="*", color="black", markersize=15, zorder=4)
        ax.annotate("same start $u_0$", x0, xytext=(x0[0] - 0.05, x0[1] + 0.28),
                    ha="right", fontsize=11)
        ax.plot(0, 0, "+", color="black", markersize=11, mew=1.8, zorder=4)
        ax.annotate("$u^\\star$", (0, 0), xytext=(0.14, -0.34), fontsize=12)
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    axes[0].text(0.02, 0.02,
                 "exact line search along $-\\nabla J$;\n"
                 "consecutive steps orthogonal",
                 transform=axes[0].transAxes, fontsize=10, va="bottom")
    axes[1].text(0.02, 0.02,
                 "energy $J(u) = \\frac{1}{2}u^\\top A u - b^\\top u$:\n"
                 "no cross-partials left",
                 transform=axes[1].transAxes, fontsize=10, va="bottom")

    fig.tight_layout()
    save(fig, "diagram13_energy_contours.png")


if __name__ == "__main__":
    fig_adi_rows_cols()
    fig_chain()
    fig_rank1_triangle()
    fig_subdomains()
    fig_energy_contours()
    print("ALL 5 diagrams written to", FIGDIR)
