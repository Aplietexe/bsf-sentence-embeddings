import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from common import CIRCLE_STEPS_AND_LR, basis, bsf_space, circles, trained
from plots.style import GOLD, GRAY, INK, PAPER, RED, TEAL, save

results = pd.read_csv("results/circles.csv")
X_train, planes, points_train = circles(12000, 0.0, seed=31415)
X_test, _, points_test = circles(6000, 0.0, seed=92653)
train = np.arange(18000) < 12000
X = bsf_space(np.r_[X_train, X_test], train)
scale = np.sqrt(768 / ((X_train - X_train.mean(0)) ** 2).sum(1).mean())
blocks = results[(results.method == "block") & (results.noise == 0)].astype({"unit": int})
models = [trained("circles0", X[train], 24, 3, 3, seed, *CIRCLE_STEPS_AND_LR["block"][0.0]) for seed in range(5)]
Z_train, Z_test, frames = zip(*[(m.codes(X[train]), m.codes(X[~train]), m.basis()) for m in models])
present_train, present_test = np.linalg.norm(points_train, axis=2) > 0, np.linalg.norm(points_test, axis=2) > 0
theta = np.linspace(0, 2 * np.pi, 300)
fig = plt.figure(figsize=(10, 9.5))


def schematic_input():
    for seed in range(5):
        units = blocks[blocks.seed == seed].sort_values("circle").unit.to_numpy()
        active = np.linalg.norm(Z_test[seed], axis=2) > 0
        for n in np.flatnonzero((active[:, units] & ~present_test).any(1)):
            inside = np.flatnonzero(present_test[n])
            targets = np.einsum("fde,fe->fd", planes[inside], points_test[n, inside])
            outputs = np.stack([frames[seed][units[f]] @ Z_test[seed][n, units[f]] / scale for f in inside])
            errors = np.linalg.norm(outputs - targets, axis=1)
            if errors.min() < 0.15 and errors.max() > 0.5 and active[n, units[inside[errors.argmax()]]]:
                return seed, n, units, active, inside, targets, outputs, errors


seed, n, units, active, inside, targets, outputs, errors = schematic_input()
absent = np.flatnonzero(active[n, units] & ~present_test[n])[0]
outputs = np.stack(
    [
        outputs[errors.argmin()],
        outputs[errors.argmax()],
        frames[seed][units[absent]] @ Z_test[seed][n, units[absent]] / scale,
    ]
)
span = basis(np.concatenate([planes[f] for f in inside] + [outputs.T], axis=1))
rng = np.random.default_rng(17)
views = [span @ np.linalg.qr(rng.normal(size=(span.shape[1], 2)))[0] for _ in range(4096)]
view = max(views, key=lambda v: min(np.linalg.svd(planes[f].T @ v, compute_uv=False)[-1] for f in inside))

ax = fig.add_axes([0.035, 0.765, 0.93, 0.225])
ax.set(xlim=(0, 17.3), ylim=(-1.9, 2.15))
ax.axis("off")


def arrow(start, end, color, lw=1.4):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw))


colors = [RED, TEAL, GOLD]
for i, (f, x, color) in enumerate(zip(inside, [1, 3.1, 5.2], colors)):
    ring = 0.72 * np.c_[np.cos(theta), np.sin(theta)] @ planes[f].T @ view
    point = 0.72 * targets[i] @ view
    ax.plot(x + ring[:, 0], ring[:, 1], color=color, lw=1.25, alpha=0.6)
    arrow((x, 0), (x + point[0], point[1]), color)
    ax.scatter(x + point[0], point[1], s=28, color=color, zorder=3)
    ax.text(x, -0.95, f"$x_{i + 1}$", ha="center", fontsize=13, color=color)
for x, sign in [(2.05, "+"), (4.15, "+"), (6.25, "=")]:
    ax.text(x, 0, sign, ha="center", va="center", fontsize=19)
total = 0.72 * targets.sum(0) @ view
arrow((7.3, 0), (7.3 + total[0], total[1]), INK, 1.6)
ax.scatter(7.3 + total[0], total[1], s=32, color=INK)
ax.text(7.3, -0.95, "Input", ha="center")
ax.annotate("", xy=(9.2, 0), xytext=(8.25, 0), arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.2})
ax.plot([9.2, 9.2], [-1.1, 1.1], color=GRAY, lw=0.8)
ax.text(12.3, 1.85, "BSF reconstructions", ha="center", fontsize=12)
labels = [
    (colors[errors.argmin()], f"Circle {errors.argmin() + 1} · good"),
    (colors[errors.argmax()], f"Circle {errors.argmax() + 1} · poor"),
    ("#77749A", "Circle 4 · absent from input"),
]
for y, output, (color, label) in zip([1.1, 0, -1.1], outputs, labels):
    point = 0.72 * output @ view
    ax.annotate("", xy=(11.1, y), xytext=(9.2, y), arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.8})
    ax.scatter(12, y, s=9, color=GRAY, zorder=3)
    arrow((12, y), (12 + point[0], y + point[1]), color, 1.6)
    ax.scatter(12 + point[0], y + point[1], s=30, color=color)
    ax.text(13.05, y, label, va="center", fontsize=10, color=color)

high = blocks[blocks.seed == 0].sort_values("train_r2").iloc[-1]


def active_rate(Z, present, row):
    return (np.linalg.norm(Z[row.seed][present[:, row.circle], row.unit], axis=1) > 0).mean()


low = blocks[[active_rate(Z_train, present_train, row) >= 0.2 for _, row in blocks.iterrows()]]
low = low.sort_values("train_r2").iloc[0]
examples = [(0.28, high, "High recovery"), (0.74, low, "Low recovery")]
xys = [
    Z_test[r.seed][present_test[:, r.circle], r.unit] @ frames[r.seed][r.unit].T / scale @ planes[r.circle]
    for _, r, _ in examples
]
lim = max(1.25, 1.06 * max(np.abs(xy).max() for xy in xys))
for (center, row, title), xy in zip(examples, xys):
    fig.text(center, 0.725, title, fontsize=13, ha="center")
    fig.text(center, 0.688, f"$R^2={row.r2:.3f}$", fontsize=12, ha="center")
    fig.text(
        center,
        0.343,
        f"Active on {active_rate(Z_test, present_test, row):.0%} of inputs containing the circle",
        ha="center",
        fontsize=10,
    )
    ax = fig.add_axes([center - 0.18, 0.365, 0.36, 0.30])
    ax.plot(np.cos(theta), np.sin(theta), color=INK, lw=1.25, zorder=3)
    ax.scatter(*xy.T, s=8, color=RED, alpha=0.32, edgecolors="none")
    ax.axhline(0, color="#E5DED7", lw=0.65, zorder=0)
    ax.axvline(0, color="#E5DED7", lw=0.65, zorder=0)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xticks=[], yticks=[])
fig.legend(
    handles=[
        Line2D([], [], color=INK, label="True circle"),
        Line2D([], [], ls="none", marker="o", color=RED, markersize=4, label="Reconstructions"),
    ],
    loc="center",
    bbox_to_anchor=(0.51, 0.515),
    fontsize=10,
)

methods = {
    "block": ("BSF block", RED),
    "feature": ("SAE feature", TEAL),
    "group of up to 3 features": ("SAE group", GOLD),
    "random block": ("Random block", GRAY),
    "oracle": ("Known planes", INK),
}
for left, noise in [(0.12, 0.0), (0.60, 0.05)]:
    ax = fig.add_axes([left, 0.085, 0.33, 0.19])
    for method, (_, color) in methods.items():
        rows = results[(results.noise == noise) & (results.method == method)]
        ax.scatter(rows.r2, rows.f1, s=9, color=color, alpha=0.35, edgecolors="none", zorder=3)
        ax.scatter(rows.r2.mean(), rows.f1.mean(), s=55, marker="D", color=color, edgecolors=PAPER, zorder=5)
    ax.set(xlim=(-0.1, 1.07), ylim=(-0.055, 1.075), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
    ax.set_title("No noise" if noise == 0 else r"Noise $\sigma=0.05$", fontsize=11, pad=9)
    ax.set_xlabel("Reconstruction $R^2$", fontsize=10, labelpad=4)
    ax.tick_params(labelsize=9, pad=5)
    ax.grid()
    if noise == 0:
        ax.set_ylabel("Presence F1", fontsize=10, labelpad=8)
fig.legend(
    handles=[
        Line2D([], [], ls="none", marker="o", color=color, markersize=5, label=label)
        for label, color in methods.values()
    ],
    loc="center",
    bbox_to_anchor=(0.52, 0.012),
    ncol=5,
    fontsize=9.5,
    handletextpad=0.4,
    columnspacing=1.3,
)
save(fig, "circle_recovery")
