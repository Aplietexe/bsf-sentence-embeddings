import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from common import BSF, CIRCLE_STEPS_AND_LR, bsf_space, circles, trained


def sums(Z, frames, points, planes, scale):
    K = np.einsum("nub,uvbe,nve->uv", Z, np.einsum("udb,vde->uvbe", frames, frames), Z, optimize=True) / scale**2
    C = np.einsum("nub,ufbe,nfe->uf", Z, np.einsum("udb,fde->ufbe", frames, planes), points, optimize=True) / scale
    return K, C, (points**2).sum((0, 2)), ((points - points.mean(0)) ** 2).sum((0, 2))


def r2(units, f, K, C, norm, spread):
    units = list(units)
    return 1 - (K[np.ix_(units, units)].sum() - 2 * C[units, f].sum() + norm[f]) / spread[f]


def greedy_group(K, C, f, size=3):
    group = []
    while len(group) < size:
        change = np.diag(K) - 2 * C[:, f] + 2 * K[:, group].sum(1)
        change[group] = np.inf
        if change.min() >= 0:
            break
        group.append(change.argmin())
    return group


def f1(pred, true):
    return 2 * (pred & true).sum() / (pred.sum() + true.sum())


rows = []
for noise in (0.0, 0.05):
    X_train, planes, points_train = circles(12000, noise, seed=31415)
    X_test, _, points_test = circles(6000, noise, seed=92653)
    train = np.arange(18000) < 12000
    X = bsf_space(np.r_[X_train, X_test], train)
    scale = np.sqrt(768 / ((X_train - X_train.mean(0)) ** 2).sum(1).mean())
    present = np.linalg.norm(points_test, axis=2) > 0
    record = lambda **row: rows.append(dict(noise=noise, **row))

    ids = np.flatnonzero(present).reshape(-1, 3) % 20
    P = planes[ids].transpose(0, 2, 1, 3).reshape(-1, 768, 6)
    coef = np.linalg.solve(P.transpose(0, 2, 1) @ P, P.transpose(0, 2, 1) @ X_test[..., None]).reshape(-1, 3, 2)
    error = np.zeros(20)
    np.add.at(error, ids, ((coef - np.take_along_axis(points_test, ids[..., None], 1)) ** 2).sum(2))
    spread = ((points_test - points_test.mean(0)) ** 2).sum((0, 2))
    for f in range(20):
        record(seed=0, method="oracle", circle=f, r2=1 - error[f] / spread[f], f1=1.0)

    for seed in range(5):
        models = {
            "block": trained(f"circles{noise:g}", X[train], 24, 3, 3, seed, *CIRCLE_STEPS_AND_LR["block"][noise]),
            "feature": trained(f"circles{noise:g}", X[train], 72, 1, 9, seed, *CIRCLE_STEPS_AND_LR["feature"][noise]),
            "random block": BSF(24, 3, 3, seed + 100),
        }
        for name, model in models.items():
            frames, Z_train, Z_test = model.basis(), model.codes(X[train]), model.codes(X[~train])
            on_train = sums(Z_train, frames, points_train, planes, scale)
            on_test = sums(Z_test, frames, points_test, planes, scale)
            K, C, norm, spread = on_train
            single = 1 - (np.diag(K)[:, None] - 2 * C + norm) / spread
            active = np.linalg.norm(Z_test, axis=2) > 0
            for f, u in enumerate(linear_sum_assignment(-single.T)[1]):
                record(
                    seed=seed,
                    method=name,
                    circle=f,
                    unit=u,
                    train_r2=single[u, f],
                    r2=r2([u], f, *on_test),
                    f1=f1(active[:, u], present[:, f]),
                )
            if name == "feature":
                for f in range(20):
                    group = greedy_group(K, C, f)
                    record(
                        seed=seed,
                        method="group of up to 3 features",
                        circle=f,
                        r2=r2(group, f, *on_test),
                        f1=f1(active[:, group].any(1), present[:, f]),
                    )

            if name == "random block":
                continue
            for f in range(20):
                K, C, norm, spread = sums(Z_test[present[:, f]], frames, points_test[present[:, f]], planes, scale)
                if name == "block":
                    u = np.argmax([f1(active[:, g], present[:, f]) for g in range(24)])
                    record(seed=seed, method="block, Fel et al.", circle=f, r2=r2([u], f, K, C, norm, spread))
                else:
                    pair_error = np.diag(K)[:, None] + np.diag(K) + 2 * K - 2 * C[:, [f]] - 2 * C[:, f] + norm[f]
                    np.fill_diagonal(pair_error, np.inf)
                    record(
                        seed=seed, method="best feature pair, Fel et al.", circle=f, r2=1 - pair_error.min() / spread[f]
                    )

df = pd.DataFrame(rows)
df.to_csv("results/circles.csv", index=False)
print(df.groupby(["method", "noise"], sort=False)[["r2", "f1"]].mean().round(3).to_string())
print(
    "blocks with test R² above 0.95, without noise:",
    ((df.method == "block") & (df.noise == 0) & (df.r2 > 0.95)).sum(),
    "of 100",
)
