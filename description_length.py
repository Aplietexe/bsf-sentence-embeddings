from math import comb, log2

import numpy as np
import pandas as pd

from common import bsf_space, kmeans, load, split, trained

PRECISIONS = [2, 3, 4, 5, 6, 8]


def quantize(Z, bits, top):
    width = 2 * top / 2**bits
    return np.where(Z != 0, -top + (np.clip(np.floor((Z + top) / width), 0, 2**bits - 1) + 0.5) * width, 0)


def description_bits(index, coefficients, residual, variance, tolerance, d=768):
    return index + coefficients + max(0, d / 2 * np.log2(residual / ((1 - tolerance) * variance)))


rows = []
for dataset in ["20ng", "arxiv", "dbpedia"]:
    X, y = load(dataset)
    train, test = split(y)
    fit, val = split(y, 0.25, 2718, among=train)
    X = bsf_space(X, fit)
    variance = {"val": X[val].var(0).mean(), "test": X[test].var(0).mean()}
    for seed in range(5):
        centers, nearest, _ = kmeans(X, fit, 256, seed)
        residual = ((X[test] - centers[nearest[test]]) ** 2).mean()
        for tolerance in (0.5, 0.8):
            rows.append(
                dict(
                    dataset=dataset,
                    seed=seed,
                    method="k-means",
                    tolerance=tolerance,
                    r2=1 - residual / variance["test"],
                    bits=description_bits(np.log2(256), 0, residual, variance["test"], tolerance),
                )
            )
        for size in (1, 2, 3, 4, 6):
            n, k = 768 // size, 12 // size
            model = trained(f"{dataset}-fit", X[fit], n, size, k, seed)
            Z, decoder = model.codes(X), model.basis().transpose(0, 2, 1).reshape(-1, 768)
            error = lambda part, codes: ((X[part] - codes.reshape(len(codes), -1) @ decoder) ** 2).mean()
            top = np.abs(Z[fit]).max()
            residual = {
                (name, bits): error(part, quantize(Z[part], bits, top))
                for name, part in [("val", val), ("test", test)]
                for bits in PRECISIONS
            }
            index = log2(comb(n, k))
            for tolerance in (0.5, 0.8):
                cost = lambda b, part: description_bits(index, 12 * b, residual[part, b], variance[part], tolerance)
                bits = min(PRECISIONS, key=lambda b: cost(b, "val"))
                rows.append(
                    dict(
                        dataset=dataset,
                        seed=seed,
                        method=f"block size {size}",
                        tolerance=tolerance,
                        r2=1 - error(test, Z[test]) / variance["test"],
                        index_bits=index,
                        precision=bits,
                        bits=cost(bits, "test"),
                    )
                )

df = pd.DataFrame(rows)
df.to_csv("results/description_length.csv", index=False)
print(df.pivot_table(["bits", "r2", "index_bits"], ["tolerance", "method"], "dataset", sort=False).round(2).to_string())
