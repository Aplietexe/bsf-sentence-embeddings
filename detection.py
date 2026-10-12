import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from common import BLOCKS, BSF, bsf_space, kmeans, load, onoff_f1, split, trained


def balanced_logistic(A, Y, l2):
    A = np.c_[A, np.ones(len(A))]
    weight = np.where(Y, 0.5 / Y.mean(0), 0.5 / (1 - Y.mean(0)))
    penalty = np.r_[np.full(A.shape[1] - 1, l2), 0]
    loss = lambda T: (weight * (np.logaddexp(0, A @ T.T) - Y * (A @ T.T))).sum(0) + 0.5 * (penalty * T**2).sum(1)
    T = np.zeros((Y.shape[1], A.shape[1]))
    for _ in range(40):
        P = expit(A @ T.T)
        grad = (weight * (P - Y)).T @ A + penalty * T
        if (np.abs(grad) < 1e-6).all():
            break
        hess = np.einsum("nc,ni,nj->cij", weight * P * (1 - P), A, A) + np.diag(penalty)
        step = np.linalg.solve(hess, grad[..., None])[..., 0]
        rate = np.ones((len(T), 1))
        while (worse := loss(T - rate * step) > loss(T) + 1e-9).any():
            rate[worse] /= 2
        T -= rate * step
    return T[:, :-1], T[:, -1]


def best_threshold(S, P, split_positives):
    order = np.argsort(-S, 0, kind="stable")
    S, P = np.take_along_axis(S, order, 0), np.take_along_axis(P, order, 0)
    f1 = 2 * P.cumsum(0) / (np.arange(1, len(S) + 1)[:, None] + split_positives)
    f1[:-1][S[:-1] == S[1:]] = 0
    i, cols = f1.argmax(0), np.arange(S.shape[1])
    cut = np.where(i + 1 < len(S), (S[i, cols] + S[np.minimum(i + 1, len(S) - 1), cols]) / 2, -np.inf)
    return f1[i, cols], np.where(f1[i, cols] > 0, cut, np.inf)


def logistic_f1(Z, active, y, fit, val, test):
    C = y.max() + 1
    if not (fit & active).any() or not (val & active).any():
        return np.zeros((2, C))
    mu = Z[fit & active].mean(0)
    scale = max(np.sqrt(((Z[fit & active] - mu) ** 2).mean()), 1e-8)
    features = lambda m: (Z[m & active] - mu) / scale
    onehot = lambda m: y[m & active, None] == np.arange(C)
    Y = onehot(fit)
    trainable = Y.any(0) & ~Y.all(0)
    best = np.zeros((2, C))
    for l2 in (0.1, 1, 10):
        W, b = np.zeros((C, Z.shape[1])), np.zeros(C)
        W[trainable], b[trainable] = balanced_logistic(features(fit), Y[:, trainable], l2)
        score = lambda m: features(m) @ W.T + b
        f_val, cut = best_threshold(score(val), onehot(val), np.bincount(y[val], minlength=C))
        pred = score(test) >= cut
        f_test = 2 * (pred & onehot(test)).sum(0) / (pred.sum(0) + np.bincount(y[test], minlength=C))
        better = f_val > best[0]
        best[:, better] = f_val[better], f_test[better]
    return best


def best_unit(scores):
    scores = np.array(scores)
    return scores[scores[:, 0].argmax(0), 1, np.arange(scores.shape[2])]


def cluster_coordinates(X, train, centers, nearest, active):
    Z = np.zeros((len(X), len(centers), 3))
    for g, center in enumerate(centers):
        members = X[train & (nearest == g)]
        pcs = np.linalg.svd(members - members.mean(0), full_matrices=False)[2][:3]
        Z[active[:, g], g] = (X[active[:, g]] - center) @ pcs.T
    return Z


def onoff(active, y, train, test, pool=1):
    units = np.argsort(-onoff_f1(active, y, train), 0, kind="stable")[:pool]
    pred = active[test][:, units].any(1)
    Y = y[test, None] == np.arange(y.max() + 1)
    tp = (pred & Y).sum(0)
    precision, recall = tp / np.maximum(pred.sum(0), 1), tp / Y.sum(0)
    return precision.mean(), recall.mean(), np.mean(2 * tp / (pred.sum(0) + Y.sum(0)))


rows = []
for dataset, n_blocks in BLOCKS.items():
    X, y = load(dataset)
    train, test = split(y)
    fit, val = split(y, 0.25, 2718, among=train)
    X = bsf_space(X, train)
    Y, C = y[:, None] == np.arange(y.max() + 1), y.max() + 1
    dense = OneVsRestClassifier(LogisticRegression(max_iter=2000), n_jobs=-1).fit(X[train], Y[train])
    pred = dense.decision_function(X[test]) > 0
    rows.append(
        dict(
            dataset=dataset,
            seed=0,
            result="figure 3",
            method="dense logistic",
            f1=np.mean(2 * (pred & Y[test]).sum(0) / (pred.sum(0) + Y[test].sum(0))),
        )
    )
    for seed in range(5):
        record = lambda result, method, f1, **rest: rows.append(
            dict(dataset=dataset, seed=seed, result=result, method=method, f1=np.mean(f1), **rest)
        )
        readout = lambda Z, active: [logistic_f1(Z[:, g], active[:, g], y, fit, val, test) for g in range(Z.shape[1])]
        sizes = {n_blocks, 256, 512} if dataset == "20ng" else {n_blocks, 256}
        Z = {n: trained(dataset, X[train], n, 3, 2, seed).codes(X) for n in sizes}
        active = {n: np.linalg.norm(Z[n], axis=2) > 0 for n in sizes}
        clusters = {n: kmeans(X, train, n, seed) for n in {n_blocks, 256}}

        Zb, on = Z[n_blocks], active[n_blocks]
        record("figure 1", "bsf on/off", best_unit(np.stack([onoff_f1(on, y, val), onoff_f1(on, y, test)], 1)))
        record("figure 1", "bsf coordinates", best_unit(readout(Zb, on)))
        Zs = trained(dataset, X[train], 3 * n_blocks, 1, 6, seed).codes(X)[..., 0]
        sae = np.array(readout(Zs[..., None], Zs != 0))
        record("figure 1", "sae feature", best_unit(sae))
        best = sae[sae[:, 0].argmax(0), :, np.arange(C)]
        for c, top3 in enumerate(np.argsort(-sae[:, 0], 0, kind="stable")[:3].T):
            group = logistic_f1(Zs[:, top3], (Zs[:, top3] != 0).any(1), y, fit, val, test)
            if group[0, c] > best[c, 0]:
                best[c] = group[:, c]
        record("figure 1", "sae up to 3 features", best[:, 1])
        centers, nearest, two_nearest = clusters[n_blocks]
        Zk = cluster_coordinates(X, train, centers, nearest, two_nearest)
        record("figure 1", "k-means coordinates", best_unit(readout(Zk, two_nearest)))
        Zr = BSF(n_blocks, 3, 2, seed + 100).codes(X)
        record("figure 1", "random block coordinates", best_unit(readout(Zr, np.linalg.norm(Zr, axis=2) > 0)))

        for name, units_active in [("bsf", active[256]), ("k-means", clusters[256][2])]:
            for pool in (1, 8):
                record("figure 3", f"{name}, pool of {pool}", onoff(units_active, y, train, test, pool)[2])
        if dataset == "20ng":
            for n in (24, 256, 512):
                precision, recall, f1 = onoff(active[n], y, train, test)
                record("table 1", f"{n} blocks", f1, precision=precision, recall=recall)

df = pd.DataFrame(rows)
df.to_csv("results/detection.csv", index=False)
means = df.groupby(["result", "method", "dataset"], sort=False)[["f1", "precision", "recall"]].mean()
print(means.round(3).to_string())
