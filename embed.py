import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

MODELS = {
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130"),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "e5": ("intfloat/e5-base-v2", "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"),
    "bge": ("BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"),
}
NEWSGROUPS = [
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "alt.atheism",
    "soc.religion.christian",
    "talk.religion.misc",
]


def newsgroups():
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"), categories=NEWSGROUPS)
    rng = np.random.default_rng(0)
    texts, labels = [], []
    for group in NEWSGROUPS:
        docs = [
            t.strip() for t, y in zip(data.data, data.target) if data.target_names[y] == group and len(t.strip()) >= 200
        ]
        for i in rng.choice(len(docs), min(600, len(docs)), replace=False):
            texts.append(docs[i][:1500])
            labels.append(group)
    return texts, labels


def balanced(df):
    rng = np.random.default_rng(0)
    df = df[df.text.str.len() >= 200]
    keep = [
        sub.iloc[np.sort(rng.choice(len(sub), min(250, len(sub)), replace=False))]
        for _, sub in df.groupby("label")
        if len(sub) >= 100
    ]
    df = pd.concat(keep)
    return [t[:1500] for t in df.text], list(df.label)


def arxiv():
    ds = load_dataset("TimSchopf/arxiv_categories", split="train", revision="9d9134623458dc824a96c963722037d914d37638")
    rows = [
        (f"{t.strip()}. {a.strip()}", c[0].split("->")[-1])
        for t, a, c in zip(ds["title"], ds["abstract"], ds["categories"])
        if len(c) == 1 and "->" in c[0]
    ]
    return balanced(pd.DataFrame(rows, columns=["text", "label"]))


def dbpedia():
    df = load_dataset(
        "DeveloperOats/DBPedia_Classes", split="train", revision="4d0aa96069f24063697e4df63b95be78d3f7fb7d"
    ).to_pandas()
    return balanced(pd.DataFrame({"text": df.text.astype(str).str.strip(), "label": df.l3}))


os.makedirs("data", exist_ok=True)
for name, build in [("20ng", newsgroups), ("arxiv", arxiv), ("dbpedia", dbpedia)]:
    texts, labels = build()
    y, categories = pd.factorize(np.array(labels))
    np.savez(f"data/{name}.npz", y=y, categories=categories, texts=np.array(texts, dtype=object))
    for model, (repo, revision) in MODELS.items():
        prefix = "query: " if model == "e5" else ""
        X = SentenceTransformer(repo, revision=revision).encode(
            [prefix + t for t in texts], batch_size=128, normalize_embeddings=True
        )
        np.save(f"data/{name}_{model}.npy", X)
