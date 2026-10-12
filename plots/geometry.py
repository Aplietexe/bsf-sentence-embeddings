import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.linalg import orthogonal_procrustes

from common import load, split
from plots.style import BOLD, DATASETS, GOLD, GRAY, INK, PAPER, PURPLE, RED, SHORT_NAMES, TEAL, save

MODELS = {"mpnet": "MPNet", "minilm": "MiniLM", "e5": "E5", "bge": "BGE"}
GROUPS = {
    "comp": ("Computing", "#596C98"),
    "rec": ("Recreation", RED),
    "sci": ("Science", TEAL),
    "talk.politics": ("Politics", GOLD),
    "": ("Religion", PURPLE),
}
LABEL_OFFSETS = {
    "comp.graphics": (-12, 18),
    "sci.med": (10, 5),
    "soc.religion.christian": (12, 18),
    "rec.sport.baseball": (10, 10),
    "rec.sport.hockey": (10, -12),
}

y = load("20ng")[1]
train, _ = split(y)
names = np.load("data/20ng.npz", allow_pickle=True)["categories"]
colors = [next(color for prefix, (_, color) in GROUPS.items() if name.startswith(prefix)) for name in names]
layouts = []
for model in MODELS:
    X = load("20ng", model)[0]
    M = np.stack([X[train & (y == c)].mean(0) for c in range(len(names))])
    U, s, _ = np.linalg.svd(M - M.mean(0), full_matrices=False)
    xy = U[:, :2] * s[:2]
    layouts.append(xy @ orthogonal_procrustes(xy, layouts[0])[0] if layouts else xy)

fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.6))
fig.subplots_adjust(left=0.06, right=0.97, top=0.95, bottom=0.10, hspace=0.16, wspace=0.10)
low, high = np.concatenate(layouts).min(0) - 0.15, np.concatenate(layouts).max(0) + 0.15
for ax, xy, title in zip(axes.flat, layouts, MODELS.values()):
    ax.axhline(0, color="#E5DED7", lw=0.65, zorder=0)
    ax.axvline(0, color="#E5DED7", lw=0.65, zorder=0)
    ax.scatter(*xy.T, c=colors, s=40, edgecolor=PAPER, linewidth=0.8, zorder=3)
    for i, name in enumerate(names):
        if name in LABEL_OFFSETS:
            offset = (14, -14) if (title, name) == ("E5", "sci.med") else LABEL_OFFSETS[name]
            ax.annotate(
                SHORT_NAMES[name],
                xy[i],
                xytext=offset,
                textcoords="offset points",
                fontsize=10,
                color=colors[i],
                va="center",
                ha="right" if offset[0] < 0 else "left",
                arrowprops={"arrowstyle": "-", "color": colors[i], "lw": 0.5, "alpha": 0.65},
                zorder=4,
            )
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal", xticks=[], yticks=[])
    ax.set_title(title, loc="left", fontsize=14, pad=4, **BOLD)
fig.legend(
    handles=[
        Line2D([], [], ls="none", marker="o", markersize=7, color=color, label=label)
        for label, color in GROUPS.values()
    ],
    loc="center",
    bbox_to_anchor=(0.5, 0.035),
    ncol=5,
    fontsize=10,
    handletextpad=0.3,
    columnspacing=1.4,
)
save(fig, "category_geometry")

results = pd.read_csv("results/geometry.csv").set_index(["dataset", "model", "quantity"]).value
fig, axes = plt.subplots(1, 3, figsize=(10, 3.7), sharex=True, sharey=True)
fig.subplots_adjust(left=0.13, right=0.96, bottom=0.25, top=0.76, wspace=0.2)
for ax, (dataset, title) in zip(axes, DATASETS.items()):
    for i, model in enumerate(MODELS):
        means = results[dataset, model, "neighbors kept, category means removed"]
        random = results[dataset, model, "neighbors kept, as many random directions removed"]
        ax.scatter(100 * means, i, s=64, color=INK, edgecolor=PAPER, lw=1, zorder=3)
        ax.scatter(100 * random, i, s=65, marker="D", facecolor=PAPER, edgecolor=GRAY, lw=1.5, zorder=3)
        for value, color in [(means, INK), (random, GRAY)]:
            ax.annotate(
                f"{value:.0%}",
                (100 * value, i),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                color=color,
                fontsize=10,
            )
    dims = len(np.load(f"data/{dataset}.npz", allow_pickle=True)["categories"]) - 1
    ax.set(xlim=(-5, 105), ylim=(3.55, -0.6), xticks=[0, 50, 100], xticklabels=["0%", "50%", "100%"])
    ax.set_yticks(range(len(MODELS)), MODELS.values())
    ax.set_title(f"{dims} dimensions removed", fontsize=11, pad=12)
    ax.annotate(title, (0.5, 1), xycoords="axes fraction", xytext=(0, 27.5), textcoords="offset points", ha="center")
    ax.grid(axis="x")
fig.legend(
    handles=[
        Line2D([], [], marker="o", ls="none", color=INK, markersize=7, label="Span of category means"),
        Line2D(
            [],
            [],
            marker="D",
            ls="none",
            markerfacecolor=PAPER,
            markeredgecolor=GRAY,
            markersize=7,
            label="Random subspace",
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.55, 1.012),
    ncol=2,
)
fig.text(0.55, 0.07, "Original top-ten neighbors retained", ha="center")
save(fig, "retrieval_geometry")
