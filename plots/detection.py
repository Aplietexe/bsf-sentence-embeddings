import re
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from sentence_transformers import SentenceTransformer

from common import bsf_space, load, split, trained
from plots.style import BOLD, DATASETS, GOLD, LINE, MUTED, PAPER, RED, SHORT_NAMES, SLATE, TEAL, save

results = pd.read_csv("results/detection.csv")

methods = {
    "bsf on/off": ("BSF · activation", "#BC7A89"),
    "bsf coordinates": ("BSF · coordinates", RED),
    "sae feature": ("SAE · single feature", "#537F7E"),
    "sae up to 3 features": ("SAE · selected group", TEAL),
    "k-means coordinates": ("k-means · coordinates", SLATE),
    "random block coordinates": ("Random · coordinates", "#A29D97"),
}
fig, axes = plt.subplots(1, 3, figsize=(10, 4.1), sharey=True)
fig.subplots_adjust(left=0.245, right=0.95, bottom=0.17, top=0.86, wspace=0.14)
for ax, (dataset, title) in zip(axes, DATASETS.items()):
    for i, (method, (label, color)) in enumerate(methods.items()):
        f1 = results[(results.result == "figure 1") & (results.dataset == dataset) & (results.method == method)].f1
        ax.plot([0, f1.mean()], [i, i], color=color, lw=3, alpha=0.15, solid_capstyle="round")
        ax.plot([f1.min(), f1.max()], [i, i], color=color, lw=2, alpha=0.6)
        ax.scatter(f1, i + np.linspace(-0.13, 0.13, len(f1)), s=15, color=color, alpha=0.52, lw=0)
        ax.plot([f1.mean()] * 2, [i - 0.23, i + 0.23], color=color, lw=2)
        ax.text(1.24, i, f"{f1.mean():.2f}", ha="right", va="center", fontsize=10, color=color)
    ax.axhline(3.5, color=LINE, lw=0.7)
    ax.set(xlim=(-0.02, 1.25), ylim=(5.65, -0.65))
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_yticks(range(len(methods)), [label for label, _ in methods.values()])
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(pad=7)
    ax.set_title(title, loc="left", fontsize=12, pad=20)
    ax.set_xlabel("Category F1", fontsize=9, labelpad=8, color=MUTED)
    ax.grid(axis="x")
save(fig, "category_detection")

X, y = load("20ng")
train, test = split(y)
fit, _ = split(y, 0.25, 2718, among=train)
Xb = bsf_space(X, train)
Z = trained("20ng", Xb[train], 24, 3, 2, 0).codes(Xb)
on = np.linalg.norm(Z, axis=2) > 0
block = on[fit].sum(0).argmax()
codes, active_fit, shown = Z[:, block], fit & on[:, block], test & on[:, block]
top = np.argsort(-np.bincount(y[active_fit]), kind="stable")[:3]
mean = codes[active_fit].mean(0)
xy = (codes - mean) @ np.linalg.svd(codes[active_fit] - mean, full_matrices=False)[2][:2].T
xy *= [np.sign(xy[shown & (y == top[0]), 0].mean()), np.sign(xy[shown & (y == top[2]), 1].mean())]
data = np.load("data/20ng.npz", allow_pickle=True)
encoder = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2", revision="e8c3b32edf5434bc2275fc9bab85f82640a19130"
)

fig = plt.figure(figsize=(10, 6.2))
ax = fig.add_axes([0.085, 0.36, 0.595, 0.59])
ax.scatter(*xy[shown & ~np.isin(y, top)].T, s=16, color="#BBB6B1", alpha=0.62, lw=0, zorder=1)
fig.text(0.735, 0.88, "Categories", va="top", **BOLD)
for i, (c, color, marker) in enumerate(zip(top, [RED, TEAL, GOLD], "o^s")):
    docs = np.flatnonzero(shown & (y == c))
    ax.scatter(*xy[docs].T, s=27, color=color, marker=marker, alpha=0.72, edgecolor=PAPER, lw=0.3, zorder=2)
    example = docs[np.argmin(((codes[docs] - codes[active_fit & (y == c)].mean(0)) ** 2).sum(1))]
    ax.scatter(*xy[example], s=155, facecolor=PAPER, edgecolor=color, lw=1.6, zorder=5)
    ax.text(*xy[example], str(i + 1), color=color, ha="center", va="center", fontsize=9, zorder=6)
    name = SHORT_NAMES[data["categories"][c]]
    fig.add_artist(Line2D([0.744], [0.79 - 0.074 * i], marker=marker, color=color, ls="none", markersize=7))
    fig.text(0.763, 0.805 - 0.074 * i, name, va="top")

    text = " ".join(data["texts"][example].split())
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if 30 <= len(s) <= 110] or [text[:100] + " …"]
    excerpt = sentences[np.argmax(encoder.encode(sentences) @ X[train & (y == c)].mean(0))]
    left = 0.06 + 0.305 * i
    fig.text(left, 0.246, f"{i + 1:02d}", color=color, va="top", **BOLD)
    fig.text(left + 0.035, 0.246, name, color=color, va="top", **BOLD)
    fig.text(left, 0.215, data["categories"][c], fontsize=8, color=MUTED, va="top")
    fig.text(left, 0.186, textwrap.fill(f"“{excerpt}”", 33), family="serif", fontsize=12, linespacing=1.35, va="top")
fig.add_artist(Line2D([0.744], [0.568], marker="o", color="#BBB6B1", ls="none", markersize=6))
fig.text(0.763, 0.583, "Other categories", fontsize=10, color=MUTED, va="top")
fig.add_artist(Line2D([0.055, 0.945], [0.275, 0.275], color=LINE, lw=0.7))
ax.axhline(0, color=LINE, lw=0.6, zorder=0)
ax.axvline(0, color=LINE, lw=0.6, zorder=0)
ax.set_aspect("equal", adjustable="datalim")
ax.set_xlabel("First principal component", fontsize=10, labelpad=8, color=MUTED)
ax.set_ylabel("Second principal component", fontsize=10, labelpad=8, color=MUTED)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.tick_params(labelsize=9, pad=5)
ax.spines[["left", "bottom"]].set_visible(True)
save(fig, "documents_in_block")

fig, axes = plt.subplots(1, 3, figsize=(10, 4.3), sharey=True)
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.16, top=0.76, wspace=0.23)
for ax, (dataset, title) in zip(axes, DATASETS.items()):
    f1 = results[(results.result == "figure 3") & (results.dataset == dataset)].groupby("method").f1.mean()
    dense = f1["dense logistic"]
    ax.axhline(dense, color=GOLD, lw=1.3, ls=(0, (3, 3)), zorder=1)
    ax.text(0.5, dense + 0.025, f"{dense:.2f}", color=GOLD, ha="center", fontsize=10)
    for name, other, color, marker in [("bsf", "k-means", RED, "o"), ("k-means", "bsf", SLATE, "D")]:
        for x, pool in enumerate([1, 8]):
            value, gap = f1[f"{name}, pool of {pool}"], f1[f"{other}, pool of {pool}"] - f1[f"{name}, pool of {pool}"]
            ax.annotate(
                f"{value:.2f}",
                (x, value),
                xytext=(11 if x else -11, -12 if 0 < gap < 0.065 else 0),
                textcoords="offset points",
                ha="left" if x else "right",
                va="center",
                color=color,
                fontsize=10,
                bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1},
                zorder=4,
            )
        ax.plot(
            [0, 1],
            [f1[f"{name}, pool of 1"], f1[f"{name}, pool of 8"]],
            color=color,
            lw=1.8,
            marker=marker,
            ms=6,
            mec=PAPER,
            mew=0.7,
            zorder=3,
        )
    ax.set(xlim=(-0.4, 1.4), ylim=(0, 1), yticks=[0, 0.25, 0.5, 0.75, 1])
    ax.set_xticks([0, 1], ["Single", "Pool of 8"])
    ax.tick_params(pad=8)
    ax.set_title(title, fontsize=12, pad=17)
    ax.grid(axis="y")
axes[0].set_ylabel("Category F1", labelpad=10)
fig.legend(
    handles=[
        Line2D([], [], color=RED, marker="o", label="BSF"),
        Line2D([], [], color=SLATE, marker="D", label="k-means"),
        Line2D([], [], color=GOLD, ls=(0, (3, 3)), label="Full-embedding classifier"),
    ],
    loc="upper center",
    bbox_to_anchor=(0.52, 0.98),
    ncol=3,
)
save(fig, "pooling")
