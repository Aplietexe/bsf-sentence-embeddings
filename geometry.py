from itertools import combinations

import numpy as np
import pandas as pd
import torch

from common import basis, load, split

MODELS = ["mpnet", "minilm", "e5", "bge"]


def centered_means(X, y, train):
    M = np.stack([X[train & (y == c)].mean(0, dtype=np.float64) for c in range(y.max() + 1)])
    return M - M.mean(0)


def neighbors(X, queries):
    X = torch.nn.functional.normalize(torch.from_numpy(X), dim=1)
    S = X[queries] @ X.T
    S[range(len(queries)), queries] = -torch.inf
    return S.topk(10).indices.numpy()


def neighbors_kept(X, Q, queries, before):
    after = neighbors(X - (X @ Q) @ Q.T, queries)
    return np.mean([len(np.intersect1d(a, b)) / 10 for a, b in zip(after, before)])


def explained(X, y, train, test):
    Q = basis(centered_means(X, y, train).T)
    R = X - (X @ Q) @ Q.T
    pcs = lambda Z: np.linalg.svd(Z - Z.mean(0), full_matrices=False)[2][:20].T
    own = [pcs(R[train & (y == c)]) for c in range(y.max() + 1)]
    everyone = pcs(R[train])
    rng = np.random.default_rng(0)
    random = [basis(rng.standard_normal((X.shape[1], 20))) for _ in own]
    fractions = []
    for c in range(len(own)):
        D = R[test & (y == c)] - R[train & (y == c)].mean(0)
        frac = lambda B: np.sum((D @ B) ** 2) / np.sum(D**2)
        fractions.append(
            [
                frac(own[c]),
                frac(everyone),
                np.mean([frac(B) for g, B in enumerate(own) if g != c]),
                np.mean([frac(B) for B in random]),
            ]
        )
    return dict(
        zip(["own category", "all training documents", "another category", "random directions"], np.mean(fractions, 0))
    )


rows = []
record = lambda *row: rows.append(dict(zip(["dataset", "model", "quantity", "value"], row)))
for dataset in ["20ng", "arxiv", "dbpedia"]:
    y = load(dataset)[1]
    train, test = split(y)
    queries = np.random.default_rng(1234).choice(np.flatnonzero(test), 2000, replace=False)
    cosines = {}
    for model in MODELS:
        X = load(dataset, model)[0]
        M = centered_means(X, y, train)
        M_unit = M / np.linalg.norm(M, axis=1, keepdims=True)
        cosines[model] = M_unit @ M_unit.T
        Q = basis(M.T)
        before = neighbors(X, queries)
        record(dataset, model, "neighbors kept, category means removed", neighbors_kept(X, Q, queries, before))
        random = [basis(np.random.default_rng(s).standard_normal(Q.shape)) for s in range(5)]
        record(
            dataset,
            model,
            "neighbors kept, as many random directions removed",
            np.mean([neighbors_kept(X, B, queries, before) for B in random]),
        )
    upper = np.triu_indices(y.max() + 1, 1)
    for a, b in combinations(MODELS, 2):
        record(
            dataset,
            f"{a}, {b}",
            "correlation of category-mean cosines",
            np.corrcoef(cosines[a][upper], cosines[b][upper])[0, 1],
        )
    if dataset == "20ng":
        X = load(dataset)[0].astype(np.float64)
        for source, value in explained(X, y, train, test).items():
            record(dataset, "mpnet", f"variance explained by 20 directions from {source}", value)

df = pd.DataFrame(rows)
df.to_csv("results/geometry.csv", index=False)
print(df.round(4).to_string(index=False))
