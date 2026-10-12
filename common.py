import os

import numpy as np
import torch

BLOCKS = {"20ng": 24, "arxiv": 128, "dbpedia": 256}
CIRCLE_STEPS_AND_LR = {
    "block": {0.0: (1500, 1e-3), 0.05: (3000, 3e-4)},
    "feature": {0.0: (3000, 3e-3), 0.05: (3000, 3e-3)},
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def load(dataset, model="mpnet"):
    """Unit-norm embeddings X (N, d) and category indices y (N,)."""
    return np.load(f"data/{dataset}_{model}.npy"), np.load(f"data/{dataset}.npz")["y"]


def split(y, frac=0.2, seed=0, among=None):
    """Masks (kept, held_out) holding out `frac` of each category's documents, among `among` if given."""
    rng = np.random.default_rng(seed)
    among = np.ones(len(y), bool) if among is None else among
    held = np.zeros(len(y), bool)
    for c in range(y.max() + 1):
        idx = np.flatnonzero(among & (y == c))
        held[rng.choice(idx, round(frac * len(idx)), replace=False)] = True
    return among & ~held, held


def bsf_space(X, train):
    """Center on the training mean and scale so training points have mean squared norm d, as BSF inputs expect."""
    X = X - X[train].mean(0)
    return (X * np.sqrt(X.shape[1] / (X[train] ** 2).sum(1).mean())).astype(np.float32)


def circles(n, noise, seed):
    """Each input sums one random point on each of 3 of 20 unit circles lying in random planes of R^768, plus
    isotropic noise. Returns inputs (n, 768), planes (20, 768, 2) and each circle's point (n, 20, 2), zero if absent."""
    rng = np.random.default_rng(0)
    planes = np.stack([np.linalg.qr(rng.standard_normal((768, 2)))[0] for _ in range(20)])
    rng = np.random.default_rng(seed)
    present = np.argsort(rng.random((n, 20)), 1)[:, :3]
    angles = rng.uniform(0, 2 * np.pi, (n, 3))
    points = np.zeros((n, 20, 2))
    np.put_along_axis(points, present[..., None], np.stack([np.cos(angles), np.sin(angles)], -1), 1)
    X = np.einsum("nfe,fde->nd", points, planes) + noise * rng.standard_normal((n, 768))
    return X.astype(np.float32), planes, points


class BSF(torch.nn.Module):
    """Block-sparse featurizer whose blocks are orthonormal frames shared by encoder and decoder, with one gain per
    block and the k blocks of largest norm kept per input (GrassmannianBSF in goodfire-ai/block-sparse-featurizer).
    With blocks of size one it is a TopK sparse autoencoder. Untrained, it gives random blocks."""

    def __init__(self, blocks, size, k, seed, d=768):
        super().__init__()
        torch.manual_seed(seed)
        self.raw = torch.nn.Parameter(torch.linalg.qr(torch.randn(blocks, d, size))[0])
        self.gain = torch.nn.Parameter(torch.ones(blocks))
        self.k = k

    def frames(self):
        """Each block's Q from its QR decomposition, computed via the Cholesky factor of its Gram matrix in float64
        (the same frames up to column signs, and much faster than Householder QR on GPUs)."""
        B = self.raw.double()
        return torch.linalg.solve_triangular(torch.linalg.cholesky(B.mT @ B), B.mT, upper=False).mT.float()

    def encode(self, X, frames=None):
        frames = self.frames() if frames is None else frames
        Z = torch.einsum("nd,gdb->ngb", X, frames) * self.gain[:, None]
        top = Z.norm(dim=2).topk(self.k).indices
        return Z * torch.zeros_like(Z[..., 0]).scatter(1, top, 1)[..., None]

    def forward(self, X):
        frames = self.frames()
        return torch.einsum("ngb,gdb->nd", self.encode(X, frames), frames)

    def fit(self, X, steps=1500, lr=1e-3):
        X = torch.as_tensor(X, device=DEVICE)
        self.to(DEVICE)
        opt = torch.optim.Adam(self.parameters(), lr)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            opt, lr, total_steps=steps, pct_start=0.05, div_factor=25, final_div_factor=1e4
        )
        noise = 0.1 * X.std()
        for _ in range(steps):
            loss = (self(X + noise * torch.randn_like(X)) - X).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            schedule.step()
        return self

    @torch.no_grad()
    def codes(self, X):
        """Codes (n, blocks, size) as a numpy array."""
        return self.to(DEVICE).encode(torch.as_tensor(X, device=DEVICE)).cpu().numpy()

    @torch.no_grad()
    def basis(self):
        """Frames (blocks, d, size) as a numpy array."""
        return self.frames().cpu().numpy()


def trained(name, X, blocks, size, k, seed, steps=1500, lr=1e-3):
    """A BSF trained on X, cached in models/ under `name`, which identifies the training data."""
    model = BSF(blocks, size, k, seed, X.shape[1])
    path = f"models/{name}_{blocks}x{size}_k{k}_s{seed}_{steps}_{lr:g}.pt"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    else:
        torch.save(model.fit(X, steps, lr).state_dict(), path)
    return model.to(DEVICE)


def kmeans(X, train, n, seed):
    """The algorithm of sklearn's KMeans(n, n_init=3), run on the GPU and fit on the training documents. It uses greedy
    k-means++ initialisation and Lloyd iterations until the centers move less than 1e-4 of the variance, and keeps the
    best of 3 runs. A cluster that loses all its documents keeps its center, where sklearn moves it to a far document,
    but no clustering in the post ever empties one. Returns the centers (n, d), each document's nearest center (N,) and
    its two nearest centers as activity (N, n)."""
    Xt = torch.as_tensor(X[train], device=DEVICE, dtype=torch.float64)
    rng = torch.Generator(DEVICE).manual_seed(seed)
    best = None
    for _ in range(3):
        C = Xt[torch.randint(len(Xt), (1,), generator=rng, device=DEVICE)]
        d2 = ((Xt - C) ** 2).sum(1)
        for _ in range(n - 1):
            draws = torch.rand(2 + int(np.log(n)), generator=rng, device=DEVICE, dtype=torch.float64) * d2.sum()
            candidates = torch.searchsorted(d2.cumsum(0), draws).clamp(max=len(Xt) - 1)
            trial = torch.minimum(d2, torch.cdist(Xt[candidates], Xt) ** 2)
            pick = trial.sum(1).argmin()
            C, d2 = torch.cat([C, Xt[candidates[pick], None]]), trial[pick]
        for _ in range(300):
            labels = torch.cdist(Xt, C).argmin(1)
            counts = torch.bincount(labels, minlength=n)[:, None]
            new = torch.where(counts > 0, torch.zeros_like(C).index_add_(0, labels, Xt) / counts.clamp(min=1), C)
            moved, C = ((new - C) ** 2).sum(), new
            if moved <= 1e-4 * Xt.var(0, correction=0).mean():
                break
        inertia = (torch.cdist(Xt, C).min(1).values ** 2).sum()
        if best is None or inertia < best[0]:
            best = inertia, C
    X = torch.as_tensor(X, device=DEVICE, dtype=torch.float64)
    nearest = torch.cdist(X, best[1]).topk(2, largest=False).indices
    active = np.zeros((len(X), n), bool)
    np.put_along_axis(active, nearest.cpu().numpy(), True, 1)
    return best[1].cpu().numpy(), nearest[:, 0].cpu().numpy(), active


def basis(A):
    """Orthonormal basis (d, rank) of the column span of A."""
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > 1e-8 * s[0]]


def onoff_f1(active, y, mask):
    """F1 (units, categories) of predicting each category exactly where each unit is active, on documents in mask."""
    Y = y[mask, None] == np.arange(y.max() + 1)
    tp = active[mask].T.astype(float) @ Y
    return 2 * tp / (active[mask].sum(0)[:, None] + Y.sum(0))
