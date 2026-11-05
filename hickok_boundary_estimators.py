"""Reference-style, independent implementations of Hickok et al. (2026) HP and DU.

Paper: https://arxiv.org/abs/2610.23590
NOT the authors' official implementation.

Neighbors: k distinct observations other than the query point. Exact duplicate
feature vectors contribute a zero directional displacement to DU.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_excluding_self(X: np.ndarray, k: int, *, metric: str = "euclidean") -> np.ndarray:
    """Return (N, k) neighbor indices, excluding the sample's own index."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must have shape (N, D).")
    n = X.shape[0]
    if not (1 <= k < n):
        raise ValueError("Require 1 <= k < N.")
    model = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1).fit(X)
    candidates = model.kneighbors(X, return_distance=False)
    # Remove each query's own index even in the presence of duplicate vectors.
    return np.stack([row[row != i][:k] for i, row in enumerate(candidates)])


def hyperplane_scores(
    X: np.ndarray,
    k: int = 100,
    intrinsic_dim: int | None = None,
    gap_threshold: float = 7.0,
    neighbors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HP scores, inward unit normals and estimated local dimensions.

    PCA is fitted separately in each k-neighborhood using mean-centered points;
    PCA coordinates for the boundary score are relative to the query point.
    When intrinsic_dim=None, uses first eigenvalue ratio > gap_threshold.
    """
    X = np.asarray(X, dtype=np.float64)
    n, D = X.shape
    if neighbors is None:
        neighbors = knn_excluding_self(X, k)
    if neighbors.shape != (n, k):
        raise ValueError("neighbors must have shape (N, k)")
    if intrinsic_dim is not None and not (1 <= intrinsic_dim <= min(k, D)):
        raise ValueError("intrinsic_dim must be between 1 and min(k, D)")
    scores = np.empty(n, dtype=float)
    normals = np.zeros((n, D), dtype=float)
    dimensions = np.empty(n, dtype=np.int32)

    for i in range(n):
        local = X[neighbors[i]]             # k x D
        Y = local - local.mean(axis=0)      # mean center *for PCA*
        # Economy SVD avoids constructing a potentially huge D x D covariance.
        _, singular, Vt = np.linalg.svd(Y, full_matrices=False)
        if intrinsic_dim is not None:
            m = intrinsic_dim
        else:
            eig = singular ** 2
            if len(eig) > 1:
                # Ignore ratios of 0 / 0. A nearly-zero denominator after a
                # nonzero numerator indicates a genuine large spectral gap.
                tiny = np.finfo(float).tiny
                ratios = eig[:-1] / np.maximum(eig[1:], tiny)
                qualifying = np.flatnonzero(ratios > gap_threshold)
                m = int(qualifying[0] + 1) if qualifying.size else len(eig)
            else:
                m = 1
        dimensions[i] = m

        basis = Vt[:m]                       # m x D
        z = (local - X[i]) @ basis.T         # query at origin, k x m
        mean_direction = z.sum(axis=0)
        norm = np.linalg.norm(mean_direction)
        if norm <= 1e-12 * max(np.linalg.norm(z), 1.0):
            # w is undefined if mean displacement vanishes.
            scores[i] = 0.5
            continue
        w = mean_direction / norm
        scores[i] = np.mean((z @ w) >= 0)
        normals[i] = w @ basis              # unit inward normal in ambient space

    return scores, normals, dimensions


def directional_uniformity_scores(
    X: np.ndarray,
    k: int = 100,
    neighbors: np.ndarray | None = None,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """DU scores and inward unit normals per Algorithm 2, Hickok et al.

    V_i = sum_{j in N(i)} (x_j - x_i)/||x_j - x_i||
    s_i = V_i dot sum_{j in N(i)} V_j / k^3.
    Computes V in batches to avoid an N x k x D intermediate array.
    """
    X = np.asarray(X, dtype=np.float64)
    n, D = X.shape
    if neighbors is None:
        neighbors = knn_excluding_self(X, k)
    if neighbors.shape != (n, k):
        raise ValueError("neighbors must have shape (N, k)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    direction_sums = np.empty((n, D), dtype=float)
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        offsets = X[neighbors[lo:hi]] - X[lo:hi, None, :]
        distances = np.linalg.norm(offsets, axis=2, keepdims=True)
        np.divide(offsets, distances, out=offsets, where=distances > 0)
        # Duplicate feature vectors have zero offset and contribute zero.
        direction_sums[lo:hi] = offsets.sum(axis=1)

    scores = np.empty(n, dtype=float)
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        neighbor_sum = direction_sums[neighbors[lo:hi]].sum(axis=1)
        scores[lo:hi] = np.einsum(
            "ij,ij->i", direction_sums[lo:hi], neighbor_sum
        ) / (k ** 3)

    lengths = np.linalg.norm(direction_sums, axis=1, keepdims=True)
    normals = np.divide(
        direction_sums, lengths,
        out=np.zeros_like(direction_sums), where=lengths > 0
    )
    return scores, normals


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(42)
    N = 800
    theta = rng.uniform(0, 2*np.pi, N)
    radius = np.sqrt(rng.uniform(size=N))  # uniform area in unit disk
    X = np.column_stack((radius*np.cos(theta), radius*np.sin(theta)))
    k = 40
    neighbors = knn_excluding_self(X, k)
    hp, hp_normal, hp_dim = hyperplane_scores(X, k, intrinsic_dim=2, neighbors=neighbors)
    du, du_normal = directional_uniformity_scores(X, k, neighbors=neighbors)

    # Empirical ground truth inspired by Section 4: omit ambiguous transition.
    included = (radius > .95) | (radius < .8)
    y = (radius[included] > .95).astype(int)
    print(f"HP ROC-AUC: {roc_auc_score(y, hp[included]):.3f}")
    print(f"DU ROC-AUC: {roc_auc_score(y, du[included]):.3f}")
    print("Sample scores (HP, DU):", hp[:5], du[:5])
