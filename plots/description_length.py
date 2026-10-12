import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plots.style import DATASETS, MUTED, RED, SLATE, TEAL, save

bits = pd.read_csv("results/description_length.csv").groupby(["dataset", "method", "tolerance"]).bits.mean()
sizes = np.array([1, 2, 3, 4, 6])
colors = [TEAL] + [RED] * 4

fig, axes = plt.subplots(1, 3, figsize=(10, 4.8), sharey=True)
fig.subplots_adjust(left=0.09, right=0.98, bottom=0.24, top=0.84, wspace=0.20)
for ax, (dataset, title) in zip(axes, DATASETS.items()):
    for tolerance in (0.5, 0.8):
        values = np.array([bits[dataset, f"block size {b}", tolerance] for b in sizes])
        best = values.argmin()
        ax.plot(sizes, values, color="#B5AFA9", lw=1.4, zorder=2)
        ax.scatter(sizes, values, s=30, color=colors, zorder=3)
        ax.scatter(sizes[best], values[best], s=110, facecolors="none", edgecolors=colors[best], lw=1.2, zorder=4)
        ax.axhline(bits[dataset, "k-means", tolerance], color=SLATE, lw=1, ls=(0, (4, 4)), alpha=0.8, zorder=1)
        if ax is axes[0]:
            ax.text(1, values[0] + 32, f"$R^2={tolerance}$", va="bottom", fontsize=10, color=MUTED)
    ax.set(xlim=(0.7, 6.3), ylim=(0, 760), xticks=sizes, yticks=[0, 200, 400, 600])
    ax.set_title(title, fontsize=13, pad=18)
    ax.tick_params(labelsize=10, pad=7)
    ax.grid(axis="y")
axes[0].set_ylabel("Bits per document", labelpad=12)
fig.text(0.53, 0.125, "Block size  ·  size 1 is the SAE", ha="center", fontsize=10)
fig.legend(
    handles=[
        Line2D([], [], ls="none", color=TEAL, marker="o", label="SAE"),
        Line2D([], [], ls="none", color=RED, marker="o", label="BSF"),
        Line2D([], [], color=SLATE, ls=(0, (4, 4)), label="k-means at the same tolerance"),
    ],
    loc="center",
    bbox_to_anchor=(0.53, 0.045),
    ncol=3,
    fontsize=10,
)
save(fig, "compression")
