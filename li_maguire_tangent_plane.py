"""Li & Maguire (2011): tangent-hyperplane class-edge detection.

Independent implementation of the edge-pattern pseudocode supplied by the user.
This is NOT verified against the authors' original software, if any.

Reference:
Y. Li and L. P. Maguire, "Selecting Critical Patterns Based on Local
Geometrical and Statistical Information," IEEE TPAMI 33(6), 1189--1201, 2011.
https://doi.org/10.1109/TPAMI.2010.188

Only the tangent-hyperplane *edge* detector is implemented here; the paper's
separate inter-class *border* detector is not included.

Dependencies: numpy, scikit-learn
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def tangent_plane_scores(
    X: np.ndarray,
    k: int = 50,
    gamma: float = 0.1,
    labels: np.ndarray | None = None,
    *,
    neighbors: np.ndarray | None = None,
    batch_size: int = 128,
    n_jobs: int | None = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Li--Maguire class-edge scores and tangent-plane normal vectors.

    Given the k nearest neighbors of x_i **from the same class**, define

        u_ij = (x_j - x_i) / ||x_j - x_i||
        n_i  = sum_j u_ij
        score_i = (1/k) * sum_j 1[dot(u_ij, n_i) >= 0]
        edge_i  = (score_i >= 1 - gamma).

    Parameters
    ----------
    X : array-like, shape (N, D)
        Numeric point cloud; uses Euclidean distance and converts to float64.
    k : int, default=50
        Number of non-self neighbors, drawn from the same class if labeled.
    gamma : float, default=0.1
        Tolerance in the edge rule score >= 1 - gamma. The default is an
        ILLUSTRATIVE choice, not a recommended value from the publication.
    labels : array-like, shape (N,), optional
        Class identifiers. If None, all observations form one class. For
        each class, at least k+1 examples must be available.
    neighbors : int array, shape (N, k), optional
        Precomputed indices of neighbors, excluding self. If supplied, they
        must be same-class when labels is given. Useful for matching kNNs
        across boundary-detection methods in a benchmark.
    batch_size : int
        Number of center points processed at a time to limit memory use.
    n_jobs : int or None
        Passed to scikit-learn's exact kNN search if neighbors is None.

    Returns
    -------
    scores : ndarray, shape (N,)
        Fraction of neighbors whose directions have nonnegative projection
        onto the estimated normal. Values in [0, 1], or NaN if undefined.
    is_edge : ndarray of bool, shape (N,)
        True if score >= 1-gamma; undefined scores are marked False.
    normals : ndarray, shape (N, D)
        Unnormalized sums of unit neighbor directions. These point toward
        the local neighbor mass (approximately inward) when defined.

    Notes
    -----
    * Exact duplicate coordinates are unsupported: a direction of zero
      length cannot be normalized. The function raises ValueError. Deduplicate
      feature vectors first or decide an explicit tie-handling policy.
    * If the normal sum is numerically zero, its direction is undefined;
      the score is set to NaN and the point is not selected. This explicit
      policy covers a degeneracy not specified in the supplied pseudocode.
    * No PCA or intrinsic-dimension estimation is performed: this follows
      the provided Li--Maguire pseudocode, not Hickok et al.'s HP algorithm.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must have shape (N, D), with D >= 1.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values.")
    N, D = X.shape
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not (1 <= k < N):
        raise ValueError("k must be an integer satisfying 1 <= k < N.")
    if not np.isfinite(gamma) or not (0 <= gamma <= 1):
        raise ValueError("gamma must lie in [0, 1].")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    if labels is None:
        labels_array = np.zeros(N, dtype=np.int32)
    else:
        labels_array = np.asarray(labels)
        if labels_array.shape != (N,):
            raise ValueError("labels must have shape (N,).")
        if labels_array.dtype.kind == "f" and not np.isfinite(labels_array).all():
            raise ValueError("labels cannot contain NaN or infinity.")

    scores = np.full(N, np.nan, dtype=np.float64)
    normals = np.zeros((N, D), dtype=np.float64)

    def evaluate(centers: np.ndarray, idx: np.ndarray) -> None:
        # centers: (B,); idx: (B, k) global sample indices.
        vec = X[idx] - X[centers, None, :]   # (B, k, D)
        lengths = np.linalg.norm(vec, axis=2)
        if np.any(lengths == 0):
            raise ValueError(
                "At least one kNN has identical coordinates to its center; "
                "unit direction is undefined. Remove/handle duplicates."
            )
        units = vec / lengths[..., None]
        normal = units.sum(axis=1)           # (B, D); NOT unit-normalized
        normals[centers] = normal
        dots = np.einsum("bkd,bd->bk", units, normal)
        local_scores = np.mean(dots >= 0, axis=1)
        degenerate = np.linalg.norm(normal, axis=1) <= 1e-12 * k
        local_scores[degenerate] = np.nan
        scores[centers] = local_scores

    if neighbors is not None:
        neighbors = np.asarray(neighbors)
        if neighbors.shape != (N, k) or neighbors.dtype.kind not in "iu":
            raise ValueError("neighbors must be an integer array of shape (N, k).")
        if np.any(neighbors < 0) or np.any(neighbors >= N):
            raise ValueError("Neighbor indices must be in [0, N).")
        if np.any(neighbors == np.arange(N)[:, None]):
            raise ValueError("Neighbor lists must exclude each point itself.")
        if np.any(labels_array[neighbors] != labels_array[:, None]):
            raise ValueError("Every neighbor must belong to its center's class.")
        if k > 1 and np.any(np.diff(np.sort(neighbors, axis=1), axis=1) == 0):
            raise ValueError("Each row of neighbors must have k distinct indices.")
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            evaluate(np.arange(start, end), neighbors[start:end])
    else:
        # Independently query neighbors *within each class*.
        for cls in np.unique(labels_array):
            group = np.flatnonzero(labels_array == cls)
            if len(group) <= k:
                raise ValueError(
                    f"Class {cls!r} has {len(group)} observations; need at least k+1={k+1}."
                )
            X_group = X[group]
            model = NearestNeighbors(
                n_neighbors=k + 1, metric="euclidean", n_jobs=n_jobs
            ).fit(X_group)
            for start in range(0, len(group), batch_size):
                end = min(start + batch_size, len(group))
                centers = group[start:end]
                candidate_local = model.kneighbors(
                    X_group[start:end], return_distance=False
                )
                candidate_global = group[candidate_local]
                # Explicitly exclude self: with tied distances it is not
                # guaranteed to appear in the first column.
                keep = candidate_global != centers[:, None]
                if np.any(keep.sum(axis=1) != k):
                    raise ValueError(
                        "Could not find exactly k non-self neighbors; repeated "
                        "coordinates/tied distances may need preprocessing."
                    )
                idx = candidate_global[keep].reshape(len(centers), k)
                evaluate(centers, idx)

    is_edge = np.isfinite(scores) & (scores >= 1.0 - gamma)
    return scores, is_edge, normals


if __name__ == "__main__":
    # Quick sanity-check example: uniformly sampled unit disk.
    rng = np.random.default_rng(42)
    n = 2_000
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    radius = np.sqrt(rng.uniform(size=n))
    X_demo = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))

    scores, is_edge, normals = tangent_plane_scores(
        X_demo, k=50, gamma=0.10, batch_size=256
    )
    print(f"Selected {is_edge.sum()} edge patterns from {n} points.")
    print(f"Mean score near disk boundary (r > .95): {np.nanmean(scores[radius > .95]):.3f}")
    print(f"Mean score in disk interior  (r < .80): {np.nanmean(scores[radius < .80]):.3f}")
