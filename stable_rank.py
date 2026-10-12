import numpy as np
import pandas as pd

from common import BSF, bsf_space, circles, load, split, trained


def mean_stable_rank(Z, min_inputs=30):
    ranks = []
    for g in range(Z.shape[1]):
        codes = Z[np.linalg.norm(Z[:, g], axis=1) > 0, g]
        if len(codes) >= min_inputs:
            s2 = np.linalg.svd(codes - codes.mean(0), compute_uv=False) ** 2
            ranks.append(s2.sum() / s2[0])
    return np.mean(ranks)


data = {}
for dataset in ["20ng", "arxiv", "dbpedia"]:
    X, y = load(dataset)
    train, test = split(y)
    data[dataset] = bsf_space(X, train), train, test
train = np.arange(18000) < 12000
X = np.r_[circles(12000, 0.05, seed=31415)[0], circles(6000, 0.05, seed=92653)[0]]
data["circles0.05"] = bsf_space(X, train), train, ~train

rows = []
for name, (X, train, test) in data.items():
    for size in (3, 4, 8, 16):
        config = 768 // size, size, 48 // size
        for seed in range(5):
            rows.append(
                dict(
                    data=name,
                    size=size,
                    seed=seed,
                    stable_rank=mean_stable_rank(trained(name, X[train], *config, seed).codes(X[test])),
                )
            )
            if name == "20ng":
                rows.append(
                    dict(
                        data="20ng random blocks",
                        size=size,
                        seed=seed,
                        stable_rank=mean_stable_rank(BSF(*config, seed + 100).codes(X[test])),
                    )
                )

df = pd.DataFrame(rows)
df.to_csv("results/stable_rank.csv", index=False)
print(df.pivot_table("stable_rank", "size", "data", sort=False).round(2).to_string())
