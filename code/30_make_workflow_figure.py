import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


parser = argparse.ArgumentParser(
    description="Generate the publication workflow diagram for DCPI-RWR."
)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "results" / "manuscript_figures",
    help="Directory receiving the PNG and vector PDF outputs.",
)
args = parser.parse_args()

OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8.2,
        "axes.titlesize": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    }
)

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#6B7280",
    "light": "#F6F8FA",
    "ink": "#111827",
}


def box(ax, xy, wh, text, fc="white", ec="#4B5563", lw=1.1, fontsize=8.0,
        weight="normal", radius=0.02, zorder=2):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=COLORS["ink"], fontsize=fontsize, weight=weight,
            linespacing=1.2, zorder=zorder + 1)
    return patch


def arrow(ax, start, end, color="#4B5563", lw=1.2, style="-|>", rad=0.0,
          mutation_scale=10, zorder=4):
    a = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mutation_scale,
        linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=zorder,
    )
    ax.add_patch(a)
    return a


def panel(ax, letter, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.01, 0.01), 0.98, 0.98,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor="white", edgecolor="#9CA3AF", linewidth=1.0,
            linestyle=(0, (3, 2)), zorder=0,
        )
    )
    ax.text(0.035, 0.955, letter, fontsize=12, weight="bold",
            ha="left", va="top", color=COLORS["ink"])
    ax.text(0.09, 0.955, title, fontsize=10.5, weight="bold",
            ha="left", va="top", color=COLORS["ink"])


fig, axs = plt.subplots(2, 2, figsize=(13.5, 8.2))
fig.subplots_adjust(left=0.035, right=0.985, bottom=0.055, top=0.965,
                    wspace=0.055, hspace=0.09)

# a. Inputs and endpoint construction
ax = axs[0, 0]
panel(ax, "a", "Evidence releases and disease-independent network views")

box(ax, (0.06, 0.67), (0.36, 0.18),
    "Open Targets 26.03\nclinical-precedence labels",
    fc="#EAF4FB", ec=COLORS["blue"], weight="bold")
box(ax, (0.58, 0.67), (0.36, 0.18),
    "Open Targets 26.06\nincident disease-specific labels",
    fc="#FFF3E0", ec=COLORS["orange"], weight="bold")
arrow(ax, (0.42, 0.76), (0.58, 0.76), color=COLORS["grey"])
ax.text(0.50, 0.79, "later release", ha="center", va="bottom",
        fontsize=7.2, color=COLORS["ink"])

box(ax, (0.04, 0.22), (0.27, 0.24),
    "STRING\nweighted functional-\nassociation graph",
    fc="#EAF4FB", ec=COLORS["blue"], weight="bold")
box(ax, (0.365, 0.22), (0.27, 0.24),
    "TISCH\nmutual 12-NN cell-type\nsimilarity graph",
    fc="#E9F8F2", ec=COLORS["green"], weight="bold")
box(ax, (0.69, 0.22), (0.27, 0.24),
    "KEGG\nweighted shared-\npathway graph",
    fc="#FBEAF4", ec=COLORS["purple"], weight="bold")

for x, c in [(0.175, COLORS["blue"]), (0.50, COLORS["green"]), (0.825, COLORS["purple"])]:
    for dx, dy in [(-0.04, 0.04), (0.04, 0.04), (-0.02, -0.05), (0.055, -0.04)]:
        ax.add_patch(Circle((x + dx, 0.11 + dy), 0.012, color=c, alpha=0.9, zorder=3))
    for p, q in [((-0.04, 0.04), (0.04, 0.04)), ((-0.04, 0.04), (-0.02, -0.05)),
                 ((0.04, 0.04), (0.055, -0.04)), ((-0.02, -0.05), (0.055, -0.04))]:
        ax.plot([x + p[0], x + q[0]], [0.11 + p[1], 0.11 + q[1]],
                color=c, lw=1.1, zorder=2)

# b. DCPI-RWR core
ax = axs[0, 1]
panel(ax, "b", "Degree-conditioned pathway-integrated random walk")

view_boxes = [
    (0.045, 0.72, COLORS["blue"], "STRING RWR"),
    (0.045, 0.45, COLORS["green"], "Cell-type RWR"),
    (0.045, 0.18, COLORS["purple"], "Pathway RWR"),
]
for x, y, c, label in view_boxes:
    box(ax, (x, y), (0.22, 0.14), label, fc="white", ec=c, weight="bold")

box(ax, (0.36, 0.30), (0.25, 0.31),
    "View-specific degree conditioning\n\n1. log(1 + degree$_v$)\n2. up to 20 quantile bins\n3. within-bin score percentile",
    fc="#F3F4F6", ec=COLORS["grey"], weight="bold", fontsize=7.7)
for y, target_y, c in [(0.79, 0.54, COLORS["blue"]),
                       (0.52, 0.46, COLORS["green"]),
                       (0.25, 0.38, COLORS["purple"])]:
    arrow(ax, (0.265, y), (0.36, target_y), color=c, rad=0.0)

box(ax, (0.36, 0.73), (0.25, 0.12),
    "Raw STRING-RWR rank", fc="#EAF4FB", ec=COLORS["blue"],
    weight="bold", fontsize=7.4)
arrow(ax, (0.265, 0.79), (0.36, 0.79), color=COLORS["blue"])
arrow(ax, (0.61, 0.455), (0.69, 0.725), color=COLORS["grey"], rad=0.05)

box(ax, (0.69, 0.62), (0.25, 0.21),
    "Conditioned-view fusion\nweighted percentile ranks",
    fc="#FFF8E1", ec=COLORS["orange"], weight="bold")
box(ax, (0.69, 0.32), (0.25, 0.18),
    "Blend with raw\nSTRING-RWR rank",
    fc="#EAF4FB", ec=COLORS["blue"], weight="bold")
arrow(ax, (0.815, 0.62), (0.815, 0.50), color=COLORS["grey"])
# The raw STRING-RWR rank bypasses degree conditioning and enters only at
# the final blend, matching the implemented DCPI-RWR equation.
arrow(ax, (0.61, 0.79), (0.69, 0.41), color=COLORS["blue"],
      lw=1.0, style="-|>", rad=-0.22, zorder=1)
box(ax, (0.70, 0.08), (0.23, 0.13),
    "DCPI-RWR score",
    fc="#FDECEC", ec=COLORS["red"], weight="bold", fontsize=8.6)
arrow(ax, (0.815, 0.32), (0.815, 0.21), color=COLORS["red"], lw=1.5)

# c. Model selection and validation
ax = axs[1, 0]
panel(ax, "c", "Locked selection and cross-release multicancer evaluation")

box(ax, (0.05, 0.63), (0.25, 0.22),
    "Historical tuning only\n\n5-fold CV × 3 repeats\nrestart, blend and\nview-weight grid",
    fc="#EAF4FB", ec=COLORS["blue"], weight="bold", fontsize=7.7)

objectives = [
    (0.39, 0.71, COLORS["blue"], "Performance\nmaximise AUPRC"),
    (0.68, 0.71, COLORS["orange"], "Discovery\nmaximise P@100"),
    (0.39, 0.40, COLORS["green"], "Bias-aware\n≥95% AUPRC; minimise\nscore-degree correlation"),
    (0.68, 0.40, COLORS["purple"], "Constrained Top100\n≥95% RWR P@100;\nminimise hub fraction"),
]
for x, y, c, label in objectives:
    box(ax, (x, y), (0.25, 0.17), label, fc="white", ec=c,
        weight="bold", fontsize=7.4)
    arrow(ax, (0.30, 0.74), (x, y + 0.085), color=c, rad=(x - 0.52) * 0.15)

box(ax, (0.09, 0.10), (0.82, 0.16),
    "Six-cancer 26.03→26.06 cross-release label validation\nLUAD • BRCA • CRC • PRAD • melanoma • ovarian cancer",
    fc="#FDECEC", ec=COLORS["red"], weight="bold", fontsize=8.2)
for x, y, _, _ in objectives:
    arrow(ax, (x + 0.125, y), (0.50, 0.26), color=COLORS["grey"], lw=0.9,
          rad=(0.50 - (x + 0.125)) * 0.18)

# d. Readouts and interpretation
ax = axs[1, 1]
panel(ax, "d", "Bias-aware evaluation and LUAD molecular characterization")

box(ax, (0.045, 0.64), (0.27, 0.22),
    "Global discrimination\nROC-AUC and AUPRC\npaired gene bootstrap\nrandom-effects meta-analysis",
    fc="#EAF4FB", ec=COLORS["blue"], weight="bold", fontsize=7.6)
box(ax, (0.365, 0.64), (0.27, 0.22),
    "Fixed-budget discovery\nTop-K incident hits\nTop-5% hub fraction\nmedian STRING degree",
    fc="#FFF8E1", ec=COLORS["orange"], weight="bold", fontsize=7.6)
box(ax, (0.685, 0.64), (0.27, 0.22),
    "Sensitivity analyses\nLOCO transfer\npathway conditional permutation\ndegree-matched null",
    fc="#E9F8F2", ec=COLORS["green"], weight="bold", fontsize=7.4)

for x in [0.18, 0.50, 0.82]:
    arrow(ax, (x, 0.64), (x, 0.53), color=COLORS["grey"])

box(ax, (0.11, 0.27), (0.78, 0.22),
    "Locked LUAD bias-aware Top 20\nTCGA RNA and stage • GSE68465 RNA • CPTAC protein • Cox survival analyses",
    fc="#FBEAF4", ec=COLORS["purple"], weight="bold", fontsize=8.1)

arrow(ax, (0.50, 0.27), (0.50, 0.17), color=COLORS["red"], lw=1.5)
box(ax, (0.18, 0.055), (0.64, 0.11),
    "Output: performance–centrality trade-off and hypothesis-generating candidates",
    fc="#FDECEC", ec=COLORS["red"], weight="bold", fontsize=8.0)

png = OUT / "Figure_1_DCPI_RWR_workflow_final.png"
pdf = OUT / "Figure_1_DCPI_RWR_workflow_final.pdf"
fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.04)
fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print(png)
print(pdf)
