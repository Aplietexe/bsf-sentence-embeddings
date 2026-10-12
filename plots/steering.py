import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from plots.style import DATASETS, GOLD, GRAY, INK, PAPER, RED, SLATE, TEAL, save

recall = pd.read_csv("results/steering.csv").groupby(["dataset", "method"]).recall.mean()
methods = {
    "block": ("Single BSF block", RED),
    "sae feature": ("Single SAE feature", TEAL),
    "full bsf code": ("Full BSF code", RED),
    "full sae code": ("Full SAE code", TEAL),
    "pooled blocks": ("Pooled BSF blocks", RED),
    "pooled k-means": ("Pooled k-means spans", SLATE),
    "difference of means": ("Difference in means", GOLD),
    "random": ("Random direction", GRAY),
}
rows = [0, 1, 2, 3, 4, 5, 6.5, 7.5]

fig, axes = plt.subplots(1, 3, figsize=(10, 5.1), sharey=True)
fig.subplots_adjust(left=0.18, right=0.98, bottom=0.16, top=0.87, wspace=0.18)
for ax, (dataset, title) in zip(axes, DATASETS.items()):
    reference = recall[dataset, "interpolation"]
    ax.axvline(reference, color=INK, lw=1, ls=(0, (4, 4)), alpha=0.65, zorder=1)
    ax.text(reference - 0.025, -0.73, f"Interpolation {reference:.2f}", ha="right", va="center", fontsize=9)
    for row, (method, (label, color)) in zip(rows, methods.items()):
        value = recall[dataset, method]
        ax.plot([0, value], [row, row], color=color, alpha=0.22, lw=1.4)
        ax.scatter(value, row, s=44, color=color, zorder=3)
        ax.text(
            value + 0.06,
            row,
            f"{value:.2f}",
            va="center",
            fontsize=9,
            color=color,
            bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1},
        )
    ax.set(xlim=(-0.025, 1.08), ylim=(8, -1.1), xticks=[0, 0.5, 1])
    ax.set_yticks(rows, [label for label, _ in methods.values()])
    ax.set_title(title, fontsize=13, pad=18)
    ax.tick_params(labelsize=10, pad=7)
    ax.grid(axis="x")
fig.text(0.58, 0.072, "Maximum recall of target neighbors", ha="center")
fig.legend(
    handles=[Line2D([], [], color=INK, lw=1, ls=(0, (4, 4)), label="Interpolation uses the exact target vector")],
    loc="center",
    bbox_to_anchor=(0.58, 0.022),
    fontsize=9,
)
save(fig, "steering")
