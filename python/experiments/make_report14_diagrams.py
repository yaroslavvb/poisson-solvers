"""Five static schematic diagrams for the report-14 hierarchical (HODLR) tutorial.

Vocabulary follows reports/13-preconditioning-as-decoupling.md SS3-4: the 1-D
tridiagonal inverse is EXACTLY the rank-1-triangle semiseparable matrix
h x_i (1 - x_j) (Gantmacher-Krein); far 2-D blocks of Sigma = A^{-1} are
numerically low-rank (~11/256 at 1e-8 on the 32x32 grid); conditional
independence across a separator, Cov(L, R | I) = 0, is the GMRF Markov
property, and Sigma_LR = Sigma_LI Sigma_II^{-1} Sigma_IR routes every L-R
dependence through the interface -> rank(Sigma_LR) <= |I|.

Figures (all -> figures/, dpi=150 PNG, prefix diagram14_; #4 is computed,
the rest are drawn):

  1. diagram14_hodlr_partition.png   -- HODLR block structure, 4 levels:
     red dense 64x64 leaves on the diagonal, green low-rank off-diagonal
     blocks (U V^T pictogram), storage O(N log N * r) vs N^2.
  2. diagram14_separator_rank.png    -- WHY off-diagonal blocks are low-rank:
     1-D chain (1-node separator -> rank 1) vs 2-D grid (separator column ->
     rank <= 32, numerically ~11), plus the Sigma_LR = Sigma_LI Sigma_II^{-1}
     Sigma_IR factorization schematic with block dimensions.
  3. diagram14_tree.png              -- the bisection tree 0..1023 down to
     64-leaves; each internal node owns one off-diagonal low-rank block pair.
  4. diagram14_1d_vs_2d.png          -- COMPUTED: log10 |A^{-1}| heatmaps for
     1-D n=64 and 2-D n=8 (N=64) with HODLR partition lines overlaid; the 1-D
     rank-1 triangle identity and exact rank-1 off-diagonal block are
     machine-checked before drawing.
  5. diagram14_nested_dissection.png -- 16x16 grid colored by nested-dissection
     level (center cross, then quadrant crosses): order separators last.

Only figure-generating code lives here; operators come from poisson.py.
Run from the repo root:
    uv run python python/experiments/make_report14_diagrams.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import ListedColormap
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

BLUE, ORANGE, GREEN, RED, GRAY = (
    "#4878d0", "#ee854a", "#6acc64", "#d65f5f", "#8a8a8a",
)
# green shades, darker = bigger block (levels 0..3 of the partition)
GREENS = ["#3e9e3e", "#5cb85c", "#8ed08e", "#c3e6c3"]
LIGHTBLUE, LIGHTGREEN = "#c9d7f0", "#d9f0d9"


def save(fig, name):
    path = FIGDIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    size = path.stat().st_size
    assert size > 0, f"{name} is empty"
    print(f"PASS {name} written, {size} bytes ({size / 1024:.0f} KB)")
    return path


# ----------------------------------------------------------------------------
# 1. diagram14_hodlr_partition.png -- the HODLR block structure (drawn)
# ----------------------------------------------------------------------------
def fig_hodlr_partition():
    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    LEVELS = 4  # diagonal blocks subdivided 4 times -> 16 dense leaves

    def draw(r0, c0, s, level):
        """Rows r0.., cols c0.. (unit-square coords, y flipped by invert)."""
        if level == LEVELS:
            ax.add_patch(Rectangle((c0, r0), s, s, facecolor=RED,
                                   edgecolor="white", linewidth=1.0))
            return
        half = s / 2.0
        for cc, rr in ((c0 + half, r0), (c0, r0 + half)):  # off-diag pair
            ax.add_patch(Rectangle((cc, rr), half, half,
                                   facecolor=GREENS[level],
                                   edgecolor="white", linewidth=1.2))
            if level == 0 and rr < cc:
                continue  # upper-right 512-block carries the UV^T pictogram
            if level <= 1:  # label only blocks big enough for 11pt text
                ax.text(cc + half / 2, rr + half / 2, "rank $r$",
                        ha="center", va="center", fontsize=13 - level,
                        color="white", fontweight="bold")
            elif level == 2:
                ax.text(cc + half / 2, rr + half / 2, "$r$",
                        ha="center", va="center", fontsize=11,
                        color="white", fontweight="bold")
        draw(r0, c0, half, level + 1)
        draw(r0 + half, c0 + half, half, level + 1)

    draw(0.0, 0.0, 1.0, 0)

    # U V^T pictogram inside the big upper-right block: tall thin U along the
    # left edge, short wide V^T along the top edge (r columns / r rows);
    # note y-axis is inverted, so "top" of the block is small y
    ax.add_patch(Rectangle((0.52, 0.10), 0.035, 0.38, facecolor="#1f6e1f",
                           edgecolor="white", linewidth=0.8))
    ax.add_patch(Rectangle((0.585, 0.02), 0.395, 0.035, facecolor="#1f6e1f",
                           edgecolor="white", linewidth=0.8))
    ax.text(0.565, 0.29, "$U$", ha="left", va="center",
            fontsize=12, color="white")
    ax.text(0.78, 0.072, "$V^{\\top}$", ha="center", va="top",
            fontsize=12, color="white")
    ax.text(0.79, 0.40, "rank $r$, stored as $UV^{\\top}$:\n"
            "$2 \\cdot \\frac{N}{2} \\cdot r$ numbers, not $(N/2)^2$",
            ha="center", va="center", fontsize=12, color="white")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()  # matrix orientation: row 0 on top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.3")
    ax.set_title("HODLR partition of $\\Sigma = A^{-1}$ "
                 "(4 levels, recursive bisection)", pad=12)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=RED, edgecolor="white"),
        Rectangle((0, 0), 1, 1, facecolor=GREENS[1], edgecolor="white"),
    ]
    ax.legend(handles,
              ["dense leaf ($64\\times64$)",
               "low-rank block ($UV^{\\top}$, rank $r$)"],
              loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              fontsize=11, handlelength=1.4, handleheight=1.4)
    ax.text(1.05, 0.80, "darker green = bigger block\n(rank stays small)",
            transform=ax.transAxes, ha="left", va="top", fontsize=11,
            color="0.25")

    fig.text(0.5, 0.045,
             "storage: $O(N \\log N \\cdot r)$ numbers instead of $N^2$",
             ha="center", va="bottom", fontsize=13)
    fig.subplots_adjust(left=0.04, right=0.70, top=0.90, bottom=0.12)
    save(fig, "diagram14_hodlr_partition.png")


# ----------------------------------------------------------------------------
# 2. diagram14_separator_rank.png -- WHY off-diagonal blocks are low-rank
# ----------------------------------------------------------------------------
def fig_separator_rank():
    fig = plt.figure(figsize=(12.5, 7.6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 0.85],
                           hspace=0.30, wspace=0.18,
                           left=0.04, right=0.97, top=0.90, bottom=0.05)

    # ---- top left: 1-D chain, one separator node ------------------------
    ax = fig.add_subplot(gs[0, 0])
    m, sep = 9, 4  # 9 interior nodes, node 4 is the separator
    ax.add_patch(Rectangle((-0.45, -0.55), sep - 0.1, 1.1,
                           facecolor=LIGHTBLUE, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((sep + 0.55, -0.55), m - sep - 1 - 0.1, 1.1,
                           facecolor=LIGHTGREEN, edgecolor="none", zorder=0))
    for x in range(m - 1):
        ax.plot([x, x + 1], [0, 0], "-", color="0.45", lw=1.8, zorder=1)
    for x in range(m):
        if x == sep:
            fc, ec, rr = ORANGE, "#b35a1f", 0.30
        elif x < sep:
            fc, ec, rr = BLUE, "#2f4f8f", 0.22
        else:
            fc, ec, rr = GREEN, "#3e7e3e", 0.22
        ax.add_patch(Circle((x, 0), rr, facecolor=fc, edgecolor=ec,
                            linewidth=1.6, zorder=3))
    ax.text((sep - 1) / 2.0, 0.95, "$L$", ha="center", fontsize=14,
            color="#2f4f8f")
    ax.text(sep, 0.95, "$I$ (1 node)", ha="center", fontsize=12,
            color="#b35a1f")
    ax.text(sep + 1 + (m - sep - 2) / 2.0, 0.95, "$R$", ha="center",
            fontsize=14, color="#3e7e3e")
    ax.set_xlim(-0.8, m - 0.2)
    ax.set_ylim(-1.6, 1.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("1-D chain", pad=2)
    ax.text((m - 1) / 2.0, -1.25,
            "every $L$–$R$ dependence factors through 1 node "
            "$\\rightarrow$ rank 1",
            ha="center", va="center", fontsize=12)

    # ---- top right: 2-D grid, one separator column ----------------------
    ax = fig.add_subplot(gs[0, 1])
    g, sc = 16, 7  # 16x16 nodes drawn; column 7 is the separator
    for j in range(g):
        for i in range(g):
            if j == sc:
                c, s = ORANGE, 42
            elif j < sc:
                c, s = BLUE, 26
            else:
                c, s = GREEN, 26
            ax.scatter(j, g - 1 - i, s=s, color=c, edgecolor="white",
                       linewidth=0.5, zorder=2)
    ax.add_patch(Rectangle((sc - 0.45, -0.6), 0.9, g + 0.2, facecolor="none",
                           edgecolor="#b35a1f", linewidth=1.8, zorder=3))
    ax.text((sc - 1) / 2.0, g + 0.1, "$L$", ha="center", fontsize=14,
            color="#2f4f8f")
    ax.text(sc, g + 0.1, "$I$", ha="center", fontsize=14, color="#b35a1f")
    ax.text(sc + 1 + (g - sc - 2) / 2.0, g + 0.1, "$R$", ha="center",
            fontsize=14, color="#3e7e3e")
    ax.set_xlim(-0.8, g - 0.2)
    ax.set_ylim(-2.6, g + 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("2-D grid  (drawn $16\\times16$; report grid $32\\times32$)",
                 pad=2)
    ax.text((g - 1) / 2.0, -1.7,
            "factors through 32 nodes $\\rightarrow$ rank $\\leq$ 32 "
            "(numerically ~11 at $10^{-8}$)",
            ha="center", va="center", fontsize=12)

    # ---- bottom: the factorization schematic ----------------------------
    ax = fig.add_subplot(gs[1, :])
    HL, WI = 2.2, 0.4  # |L| tall, |I| thin
    y0 = 0.0

    def rect(x, w, h, fc, label, dim):
        yb = y0 + (HL - h) / 2.0  # vertically center every factor
        ax.add_patch(Rectangle((x, yb), w, h, facecolor=fc,
                               edgecolor="0.25", linewidth=1.4))
        ax.text(x + w / 2, yb + h + 0.16, label, ha="center", va="bottom",
                fontsize=13)
        ax.text(x + w / 2, y0 - 0.18, dim, ha="center", va="top", fontsize=11)
        return x + w

    x = rect(0.0, HL, HL, "0.88", "$\\Sigma_{LR}$", "$|L| \\times |R|$")
    ax.text(x + 0.45, y0 + HL / 2, "=", ha="center", va="center", fontsize=20)
    x = rect(x + 0.9, WI, HL, LIGHTBLUE, "$\\Sigma_{LI}$",
             "$|L| \\times |I|$")
    ax.text(x + 0.3, y0 + HL / 2, "$\\cdot$", ha="center", va="center",
            fontsize=20)
    x = rect(x + 0.6, WI, WI, ORANGE, "$\\Sigma_{II}^{-1}$",
             "$|I| \\times |I|$")
    ax.text(x + 0.3, y0 + HL / 2, "$\\cdot$", ha="center", va="center",
            fontsize=20)
    x = rect(x + 0.6, HL, WI, LIGHTGREEN, "$\\Sigma_{IR}$",
             "$|I| \\times |R|$")

    arrow = FancyArrowPatch((x + 0.35, y0 + HL / 2), (x + 1.15, y0 + HL / 2),
                            arrowstyle="->", mutation_scale=22, lw=2.0,
                            color="0.25")
    ax.add_patch(arrow)
    ax.text(x + 1.3, y0 + HL / 2,
            "rank$(\\Sigma_{LR}) \\leq |I|$:\nthe whole $L$–$R$ block\n"
            "is squeezed through\nthe interface",
            ha="left", va="center", fontsize=12)

    ax.set_xlim(-0.3, x + 4.6)
    ax.set_ylim(-0.8, HL + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("conditional independence (Cov$(L,R\\,|\\,I)=0$, GMRF "
                 "Markov property) written as a matrix factorization",
                 fontsize=12, pad=4)

    fig.suptitle("why the off-diagonal blocks of $\\Sigma = A^{-1}$ are "
                 "low-rank: dependence is routed through a separator",
                 fontsize=13)
    save(fig, "diagram14_separator_rank.png")


# ----------------------------------------------------------------------------
# 3. diagram14_tree.png -- the bisection tree, 0..1023 down to 64-leaves
# ----------------------------------------------------------------------------
def fig_tree():
    fig, ax = plt.subplots(figsize=(13, 7))
    NLVL = 4  # internal levels 0..3; leaves at level 4 (16 leaves of 64)
    UNIT = 16.0  # leaf i occupies [i, i+1)

    def center(level, k):
        span = UNIT / (2 ** level)
        return k * span + span / 2.0

    ys = {lvl: 4.0 - lvl for lvl in range(NLVL + 1)}
    box_h = 0.42

    # edges first
    for lvl in range(NLVL):
        for k in range(2 ** lvl):
            xp, yp = center(lvl, k), ys[lvl]
            for kk in (2 * k, 2 * k + 1):
                xc, yc = center(lvl + 1, kk), ys[lvl + 1]
                ax.plot([xp, xc], [yp - box_h / 2, yc + box_h / 2],
                        "-", color="0.55", lw=1.2, zorder=1)

    # internal nodes: light-green boxes labeled with their index range
    blk = 512
    for lvl in range(NLVL):
        n_nodes = 2 ** lvl
        span = 1024 // n_nodes
        w = min(2.6, UNIT / n_nodes * 0.82)
        fs = 12 if lvl <= 1 else (11 if lvl == 2 else 9)
        for k in range(n_nodes):
            x = center(lvl, k)
            lo, hi = k * span, (k + 1) * span - 1
            ax.add_patch(Rectangle((x - w / 2, ys[lvl] - box_h / 2), w, box_h,
                                   facecolor=LIGHTGREEN, edgecolor="#3e7e3e",
                                   linewidth=1.3, zorder=2))
            ax.text(x, ys[lvl], f"{lo}–{hi}", ha="center", va="center",
                    fontsize=fs, zorder=3)
        # right-margin annotation: the off-diagonal blocks this level owns
        ax.text(UNIT + 0.35, ys[lvl],
                f"{2 * n_nodes} low-rank blocks, "
                f"${blk}\\times{blk}$",
                ha="left", va="center", fontsize=11, color="#3e7e3e")
        blk //= 2

    # leaves: small red squares
    for k in range(16):
        x = center(NLVL, k)
        ax.add_patch(Rectangle((x - 0.42, ys[NLVL] - box_h / 2), 0.84, box_h,
                               facecolor=RED, edgecolor="#8f3030",
                               linewidth=1.0, zorder=2))
    ax.text(UNIT + 0.35, ys[NLVL], "16 dense diagonal leaves, "
            "$64\\times64$", ha="left", va="center", fontsize=11,
            color="#8f3030")

    # explicit block annotation at the root
    ax.annotate("splitting 0–1023 at 512 creates the block pair\n"
                "$\\Sigma[0{:}512,\\,512{:}1024]$ and its transpose "
                "$\\rightarrow$ stored once, low-rank",
                xy=(center(0, 0) + 1.3, ys[0]), xytext=(11.6, ys[0] + 0.55),
                ha="center", va="center", fontsize=11,
                arrowprops=dict(arrowstyle="->", color="0.35", lw=1.3))

    ax.set_xlim(-0.4, UNIT + 6.6)
    ax.set_ylim(-0.75, 5.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    ax.set_title("recursive bisection of the index set 0..1023 "
                 "($N = 1024$, leaf size 64)", pad=8)
    fig.text(0.5, 0.035,
             "one low-rank block per tree edge pair + dense leaves "
             "= the whole of $\\Sigma$, each entry covered exactly once",
             ha="center", va="bottom", fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.10)
    save(fig, "diagram14_tree.png")


# ----------------------------------------------------------------------------
# 4. diagram14_1d_vs_2d.png -- COMPUTED log10 |A^{-1}| heatmaps + partitions
# ----------------------------------------------------------------------------
def hodlr_overlay(ax, a, b, leaf, lw):
    """White HODLR cross on the diagonal block [a, b) of an imshow axes."""
    if b - a <= leaf:
        return
    m = (a + b) // 2
    ax.plot([a - 0.5, b - 0.5], [m - 0.5, m - 0.5], color="white", lw=lw)
    ax.plot([m - 0.5, m - 0.5], [a - 0.5, b - 0.5], color="white", lw=lw)
    hodlr_overlay(ax, a, m, leaf, max(lw * 0.62, 0.7))
    hodlr_overlay(ax, m, b, leaf, max(lw * 0.62, 0.7))


def fig_1d_vs_2d():
    # 1-D: n = 64 interior nodes, A1 = laplacian_1d(n)/h^2
    n1 = 64
    h1 = 1.0 / (n1 + 1)
    A1 = (laplacian_1d(n1) / h1**2).toarray()
    Ainv1 = np.linalg.inv(A1)

    # machine checks (house facts: report 13 SS3)
    x = np.arange(1, n1 + 1) * h1
    outer = h1 * np.outer(x, 1.0 - x)
    iu = np.triu(np.ones((n1, n1), dtype=bool))
    dev = np.abs(Ainv1 - outer)[iu].max()
    if dev > 1e-12:
        raise RuntimeError(f"1-D semiseparable identity violated: {dev:.3e}")
    print(f"PASS 1-D rank-1 triangle identity (n=64): max upper-tri "
          f"|A^-1_ij - h x_i(1-x_j)| = {dev:.1e} (<= 1e-12)")

    s1d = np.linalg.svd(Ainv1[0:32, 32:64], compute_uv=False)
    ratio = s1d[1] / s1d[0]
    if ratio > 1e-12:
        raise RuntimeError(f"1-D off-diag block not rank 1: s2/s1={ratio:.3e}")
    print(f"PASS 1-D off-diagonal block [0:32,32:64] exactly rank 1: "
          f"s2/s1 = {ratio:.1e} (<= 1e-12)")

    # 2-D: n = 8 per side, N = 64
    n2 = 8
    A2 = poisson_2d(n2).toarray()
    Ainv2 = np.linalg.inv(A2)
    s2d = np.linalg.svd(Ainv2[0:32, 32:64], compute_uv=False)
    nrank = int(np.sum(s2d > 1e-8 * s2d[0]))
    if nrank >= 32:
        raise RuntimeError("2-D off-diag block unexpectedly full rank")
    print(f"PASS 2-D off-diagonal block [0:32,32:64] (N=64) numerical rank "
          f"{nrank}/32 at 1e-8 (strictly < 32)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, M, title in zip(
        axes,
        (Ainv1, Ainv2),
        ("1D: every off-diagonal block exactly rank 1",
         "2D: low rank, thicker teeth near the diagonal"),
    ):
        im = ax.imshow(np.log10(np.abs(M)), cmap="viridis",
                       interpolation="nearest")
        hodlr_overlay(ax, 0, 64, 8, 2.2)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([0, 16, 32, 48, 63])
        ax.set_yticks([0, 16, 32, 48, 63])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("$\\log_{10}|\\Sigma_{ij}|$", fontsize=11)
    axes[0].set_ylabel("$\\Sigma = A^{-1}$,  $A = d_1/h^2$  ($n=64$)")
    axes[1].set_ylabel("$\\Sigma = A^{-1}$,  $A$ = poisson_2d(8)  ($N=64$)")
    fig.suptitle("the solution operator already has the HODLR structure "
                 "(white lines: leaf size 8)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "diagram14_1d_vs_2d.png")


# ----------------------------------------------------------------------------
# 5. diagram14_nested_dissection.png -- the scalable-ordering teaser
# ----------------------------------------------------------------------------
def fig_nested_dissection():
    g = 16
    lvl = np.full((g, g), 2)  # 2 = interior (eliminated first)

    # level-1 separators: cross at the center of each quadrant
    quads = [(0, 6), (8, 15)]  # index ranges left/top and right/bottom
    centers = {(0, 6): 3, (8, 15): 11}
    for rlo, rhi in quads:
        for clo, chi in quads:
            rc, cc = centers[(rlo, rhi)], centers[(clo, chi)]
            lvl[rlo:rhi + 1, cc] = 1
            lvl[rc, clo:chi + 1] = 1
    # level-0 separator: center column + row (overwrites)
    lvl[:, 7] = 0
    lvl[7, :] = 0

    cmap = ListedColormap([RED, ORANGE, LIGHTBLUE])
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ax.pcolormesh(np.arange(g + 1), np.arange(g + 1), lvl[::-1],
                  cmap=cmap, edgecolors="white", linewidth=1.2,
                  vmin=0, vmax=2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.3")
    ax.set_title("nested dissection on a $16\\times16$ grid: "
                 "recursive cross separators", pad=10)

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=LIGHTBLUE, edgecolor="white"),
        Rectangle((0, 0), 1, 1, facecolor=ORANGE, edgecolor="white"),
        Rectangle((0, 0), 1, 1, facecolor=RED, edgecolor="white"),
    ]
    ax.legend(handles,
              ["interiors — eliminated first",
               "level-1 crosses — next",
               "center cross — eliminated last"],
              loc="upper left", bbox_to_anchor=(1.03, 1.0), frameon=False,
              fontsize=11, handlelength=1.4, handleheight=1.4)

    fig.text(0.5, 0.045,
             "order separators last $\\rightarrow$ Cholesky fill is confined "
             "to separator blocks\n+ Schur complements on separators are "
             "again hierarchically low-rank",
             ha="center", va="bottom", fontsize=12)
    fig.subplots_adjust(left=0.05, right=0.62, top=0.92, bottom=0.15)
    save(fig, "diagram14_nested_dissection.png")


if __name__ == "__main__":
    fig_hodlr_partition()
    fig_separator_rank()
    fig_tree()
    fig_1d_vs_2d()
    fig_nested_dissection()
    print("PASS all 5 diagram14 figures written")
