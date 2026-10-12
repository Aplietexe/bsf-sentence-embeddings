from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from common import BLOCKS, DEVICE, basis, bsf_space, kmeans, load, onoff_f1, split, trained


def top10(V, queries, exclude):
    sims = torch.as_tensor(queries, device=DEVICE, dtype=torch.float32) @ V.T
    sims[np.arange(len(exclude)), :, exclude] = -torch.inf
    return sims.topk(10).indices.cpu().numpy()


rows = []
for dataset, n_blocks in BLOCKS.items():
    raw, y = load(dataset)
    train, _ = split(y)
    X = bsf_space(raw, train)
    V = torch.as_tensor(raw, device=DEVICE)
    rng = np.random.default_rng(0)
    pairs = [rng.choice(np.flatnonzero(y == rng.integers(y.max() + 1)), 2, replace=False) for _ in range(100)]
    pairs += [
        [rng.choice(np.flatnonzero(y == c)) for c in rng.choice(y.max() + 1, 2, replace=False)] for _ in range(100)
    ]
    source, target = np.array(pairs).T
    docs = np.unique(pairs)
    neighbors = dict(zip(docs, top10(V, raw[docs, None], docs)[:, 0]))
    targets = np.array([neighbors[t] for t in target])
    queries = {"interpolation": [[(1 - a) * raw[s] + a * raw[t] for a in np.linspace(0.1, 0.95, 12)] for s, t in pairs]}

    for seed in range(5):
        centers, nearest, two_nearest = kmeans(X, train, n_blocks, seed)
        spans = [
            np.c_[c, np.linalg.svd(m - m.mean(0), full_matrices=False)[2][:2].T]
            for c, m in zip(centers - X[train].mean(0), (X[train & (nearest == g)] for g in range(n_blocks)))
        ]
        units = {"k-means": (None, np.array(spans), two_nearest)}
        for name, (n, size, k) in {"block": (n_blocks, 3, 2), "sae feature": (3 * n_blocks, 1, 6)}.items():
            model = trained(dataset, X[train], n, size, k, seed)
            Z = model.codes(X)
            units[name] = Z, model.basis(), np.linalg.norm(Z, axis=2) > 0
        ranking = {
            name: np.argsort(-onoff_f1(active, y, train), 0, kind="stable") for name, (_, _, active) in units.items()
        }

        directions = defaultdict(list)
        random = np.random.default_rng(seed + 100)
        for s, t in pairs:
            shift = X[neighbors[t]].mean(0) - X[neighbors[s]].mean(0)
            for name, full in [("block", "full bsf code"), ("sae feature", "full sae code")]:
                Z, frames, _ = units[name]
                g = ranking[name][0, y[t]]
                directions[name].append(frames[g] @ (Z[neighbors[t], g].mean(0) - Z[s, g]))
                directions[full].append(np.einsum("gdb,gb->d", frames, Z[neighbors[t]].mean(0) - Z[s]))
            for name, pooled in [("block", "pooled blocks"), ("k-means", "pooled k-means")]:
                frames = units[name][1][ranking[name][:8, np.unique(y[[s, t]])].ravel()]
                Q = basis(np.concatenate(frames, axis=1))
                directions[pooled].append(Q @ (Q.T @ shift))
            directions["difference of means"].append(shift)
            directions["random"].append(random.standard_normal(768))

        for name, vs in directions.items():
            u = [v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-10 else 0 * v for v in vs]
            queries[name] = [[raw[s] + step * d for step in np.linspace(0.1, 1.4, 12)] for s, d in zip(source, u)]
        for name, q in queries.items():
            hits = (top10(V, np.array(q), source)[..., None] == targets[:, None, None]).any(-1).sum(-1)
            rows.append(dict(dataset=dataset, seed=seed, method=name, recall=hits.max(1).mean() / 10))

df = pd.DataFrame(rows)
df.to_csv("results/steering.csv", index=False)
print(df.pivot_table("recall", "method", "dataset", sort=False).round(3).to_string())
