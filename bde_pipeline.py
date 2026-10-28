from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Helpers
# ============================================================

def local_density(neigh: np.ndarray, eps: float = 1e-12) -> float:
    """neigh: (k, d). Returns 1 / sum(||neigh_i||)."""
    norms = np.linalg.norm(neigh, axis=1)
    s = float(np.sum(norms))
    if s <= eps:
        return np.inf
    return 1.0 / s


def sont_dependants(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> bool:
    """True if u and v are numerically linearly dependent."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if np.linalg.norm(u) < eps or np.linalg.norm(v) < eps:
        return True

    if u.shape[0] == 3:
        return np.linalg.norm(np.cross(u, v)) < eps

    return np.linalg.matrix_rank(np.column_stack((u, v))) < 2


def projection_gram_schmidt(
    x: np.ndarray, u: np.ndarray, v: np.ndarray, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project x on span(u,v) using Gram–Schmidt orthonormalization.
    Returns (coords_2d, proj_d). Raises ValueError if u,v are dependent.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if np.linalg.norm(u) < eps or np.linalg.norm(v) < eps:
        raise ValueError("u and v must be non-zero.")

    if np.linalg.matrix_rank(np.column_stack((u, v))) < 2:
        raise ValueError("u and v must be linearly independent.")

    u_tilde = u / np.linalg.norm(u)
    v_ortho = v - np.dot(v, u_tilde) * u_tilde
    if np.linalg.norm(v_ortho) < eps:
        raise ValueError("Vectors colinear after numerical correction.")
    v_tilde = v_ortho / np.linalg.norm(v_ortho)

    a = np.dot(x, u_tilde)
    b = np.dot(x, v_tilde)
    proj = a * u_tilde + b * v_tilde
    return np.array([a, b]), proj


def _initial_neighbor_indices_and_distances(
    data: np.ndarray,
    knn: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return k neighbors per point, excluding self by index only.

    Other rows at distance 0 are valid duplicate points and remain eligible for
    all downstream pipeline computations.
    """
    n_points = data.shape[0]
    if knn >= n_points:
        raise ValueError(f"knn={knn} requires at least {knn + 1} points, found {n_points}.")

    nbrs = NearestNeighbors(n_neighbors=knn + 1, algorithm="auto", metric="euclidean").fit(data)
    distances, indices = nbrs.kneighbors(data)

    selected_indices = np.empty((n_points, knn), dtype=indices.dtype)
    selected_distances = np.empty((n_points, knn), dtype=distances.dtype)

    for row_idx in range(n_points):
        keep = indices[row_idx] != row_idx
        row_indices = indices[row_idx][keep]
        row_distances = distances[row_idx][keep]
        if len(row_indices) < knn:
            raise ValueError(
                f"Expected at least {knn} neighbors after removing self for row {row_idx}, "
                f"found {len(row_indices)}."
            )

        selected_indices[row_idx] = row_indices[:knn]
        selected_distances[row_idx] = row_distances[:knn]

    return selected_distances, selected_indices


# ============================================================
# Your two functions (cleaned but same logic)
# ============================================================

def check_sums(data, denom):
    ref = next(val for val, (a, b) in data if (a, b) == (0, 1)) + 0.0001

    # Construire le dictionnaire {(i,j): valeur} en forçant int/float
    values = {(int(i), int(j)): float(val) for val, (i, j) in data}

    result = []
    # Tous les j présents
    js = {j for (_, j) in values.keys()}

    for j in js:
        if j != 1:      # Exclure la référence (0,1)
            result_j = values[(0, j)] + values[(1, j)]
            result.append(result_j)

    booleen = [1 for val in result if val <= ref] #<= because of tolerance 0.01
    prop = np.count_nonzero(booleen)/float(denom)#2 vecteurs formant le plan

    return prop

def check_pre_concavity(
    angles_int: List[Tuple[float, Tuple[int, int]]],
    denom: int,
    tol: float = 1e-4,
    prop_thresh: float=0.7,
) -> Tuple[float, List[Tuple[float, int, int]]]:
    """
    angles_int: list of (angle_deg, (i,j))
    denom: typically len(vectors)-2 (since you consider j != 0,1)

    Returns:
      prop: proportion of j where values[(0,j)] + values[(1,j)] <= ref + tol
      min_ext_int: list of (min_angle, min_idx, j) with j=v_1 or v_2.
    """
    values = {(int(i), int(j)): float(val) for val, (i, j) in angles_int}

    if (0, 1) not in values:
        raise ValueError("Reference pair (0,1) missing in angles_int.")

    #values[(0, 1)] is the angle between the two basis vectors (0 and 1), it is given in the bde_pipeline before this function

    ref = values[(0, 1)] + tol

    # all possible second indices j
    js = {j for (_, j) in values.keys()}

    result = []
    # only compute sums where both (0,j) and (1,j) exist
    for j in sorted(js):
        if j == 1:
            continue
        if (0, j) in values and (1, j) in values:
            result.append((values[(0, j)] + values[(1, j)], j))

    if denom <= 0:
        raise ValueError(f"Invalid denom={denom}. Expected len(vectors)-2 >= 1.")

    booleen = [1 for val, _ in result if val <= ref]#because of numeric approx of the angle values.
    prop = np.count_nonzero(booleen) / float(denom)

    min_ext_int: List[Tuple[float, int, int]] = []

    # Extreme cases preserved from your original idea
    if prop >= prop_thresh and prop != 1.0:
        for s, j in result:
            if s > ref:
                if (0, j) in values and (1, j) in values:
                    vec_r = [values[(0, j)], values[(1, j)]]
                    min_val = min(vec_r)
                    min_idx = int(np.argmin(vec_r))  # 0 or 1
                    min_ext_int.append((min_val, min_idx, j))

    if prop <= 1 - prop_thresh and prop != 0.0:
        for s, j in result:
            if s <= ref:
                if (0, j) in values and (1, j) in values:
                    vec_r = [values[(0, j)], values[(1, j)]]
                    min_val = min(vec_r)
                    min_idx = int(np.argmin(vec_r))# 0 or 1
                    min_ext_int.append((min_val, min_idx, j))

    return prop, min_ext_int


def check_concavity(
    prop: float,
    prop_thresh: float, #high prop threshold (e.g., boundary score >= 0.7)
    max_angle: float,
    item: List[Tuple[float, int, int]], #min_ext_int from check_pre_concavity function (min_angle, min_idx,j)
    angular_gap: float = 90,
) -> Tuple[bool, Dict]:
    """
    item: list of (value, label, stack_index)
    #adjusted angle and adjusted_prop computed in the gram-schmidt basis.
    #Spliting the list by vector label (0 or 1).
    """
    Index_0 = None
    Index_1 = None
    j_0_max = None
    j_1_max = None
    alpha_0 = []
    j_0 = []
    alpha_1= []
    j_1 = []
    for val, idx, j in item:
        if idx == 0:
            alpha_0.append(val)
            j_0.append(j)
        if idx == 1:
            alpha_1.append(val)
            j_1.append(j)    

    beta_0 = max(alpha_0) if alpha_0 else 0.0
    beta_1 = max(alpha_1) if alpha_1 else 0.0

    

    if prop >= prop_thresh and prop != 1.0:
        if max_angle + beta_0 + beta_1 >= 180:
            omega = max_angle + beta_0 + beta_1
            gap = 360-omega
            return gap >= angular_gap, {"prop_in_gap":0.0,"gap": gap}#hard-coded adj prop because in the gram-scmidt basis.
        if max_angle + beta_0 + beta_1 < 180:
            omega = max_angle + beta_0 + beta_1
            return True, {"prop_in_gap":1.0,"gap": omega}#because the angular gap is the complement of 
        #all gaps, ie., adjusted angles are < 180 degrees.

    if prop <= 1 - prop_thresh and prop != 0.0:
        omega = max_angle - beta_0 - beta_1
        return omega >= angular_gap, {"prop_in_gap":0.0,"gap": omega}

    return False, {}


# ============================================================
# Core pipeline (clean rewrite)
# ============================================================
def pipeline_initial(
    data: np.ndarray,
    knn: int = 32,
    angle_min_deg: float = 1.0,
    angle_max_deg: float = 179.9,
    eps: float = 1e-12,
):

    data = np.asarray(data, dtype=float)
    N, d = data.shape

    distances, indices = _initial_neighbor_indices_and_distances(data, knn)

    prop: List[float] = []
    all_max_angles: List[float] = []
    rho: List[float] = []



    for idx in range(N):
        try:
            vec = data[indices[idx]] - data[idx]   # (knn, d)
            rho_i = local_density(vec, eps=eps)

            # ----------------------------
            # find one dependent pair (optional)
            # ----------------------------
            dep_pairs = []
            for k in range(knn):
                for l in range(k + 1, knn):
                    if sont_dependants(vec[k], vec[l], eps=eps):
                        dep_pairs.append((k, l))

            dep_pair: Optional[Tuple[int, int]] = dep_pairs[0] if dep_pairs else None

            # ----------------------------
            # pairwise angles among vec (store tuples)
            # exclude ONLY that dependent pair (and symmetric)
            # ----------------------------
            angles: List[Tuple[float, Tuple[int, int]]] = []
            for x in range(knn - 1):
                nx = np.linalg.norm(vec[x])
                if nx <= eps:
                    continue
                for y in range(x + 1, knn):
                    if dep_pair is not None and ((x, y) == dep_pair or (y, x) == dep_pair):
                        continue
                    ny = np.linalg.norm(vec[y])
                    if ny <= eps:
                        continue

                    dot = float(np.dot(vec[x], vec[y]))
                    theta = float(np.degrees(np.arccos(np.clip(dot / (nx * ny), -1.0, 1.0))))
                    angles.append((theta, (x, y)))

            valid_angles = [(t, pair) for (t, pair) in angles if angle_min_deg <= t <= angle_max_deg]
            if not valid_angles:
                raise ValueError("No valid pairwise angle found in vec.")

            max_angle, (m, n) = max(valid_angles, key=lambda t: t[0])

            # ----------------------------
            # project remaining vectors onto plane span(vec[m], vec[n])
            # ----------------------------
            proj_list = []
            for z in range(knn):
                if z in (m, n):
                    continue
                try:
                    _, proj_z = projection_gram_schmidt(vec[z], vec[m], vec[n], eps=eps)
                    proj_list.append(proj_z)
                except ValueError:
                    continue

            # stack base plane vectors + projections
            stack_items = [vec[m], vec[n]] + proj_list
            vectors = np.vstack(stack_items)  # (2 + nproj, d)

            # ----------------------------
            # angles among vectors
            # ----------------------------
            labels = list(range(len(vectors)))
            angles_int: List[Tuple[float, Tuple[int, int]]] = []
            for o in range(len(vectors) - 1):
                no = np.linalg.norm(vectors[o])
                if no <= eps:
                    continue
                for p in range(o + 1, len(vectors)):
                    np_ = np.linalg.norm(vectors[p])
                    if np_ <= eps:
                        continue
                    dot = float(np.dot(vectors[o], vectors[p]))
                    theta = float(np.degrees(np.arccos(np.clip(dot / (no * np_), -1.0, 1.0))))
                    angles_int.append((theta, (labels[o], labels[p])))

            denom = knn - 2
            prop_i= check_sums(angles_int, denom=denom)

            prop.append(prop_i)
            all_max_angles.append(float(max_angle))
            rho.append(rho_i)


        except ValueError:
            prop.append(np.nan)
            all_max_angles.append(np.nan)
            rho.append(np.nan)


    # ----------------------------
    # relative density
    # rel_dens[i] = rho[i] / max_{j in knn(i)} rho[j]
    # ----------------------------
    #densities = np.asarray(rho, dtype=float)
    #rel_dens = np.full(N, np.nan, dtype=float)

    #for i in range(N):
    #    neigh = densities[indices[i]]
     #   max_neigh = np.nanmax(neigh)
      #  if np.isfinite(densities[i]) and np.isfinite(max_neigh) and max_neigh > 0:
       #     rel_dens[i] = densities[i] / max_neigh

    return {
        "bos": np.asarray(prop, dtype=float),
        "angles": np.asarray(all_max_angles, dtype=float),
        "indices": indices,
        "distances": distances,
    }

#Usage: results_init = pipeline_initial(data, knn=17)
# results_init["bos"], results_init["angles]

def pipeline_adjusted(
    data: np.ndarray,
    knn: int = 32,
    angle_min_deg: float = 1.0,
    angle_max_deg: float = 179.9,
    prop_thresh: float = 0.7,
    angular_gap: float = 90,
    eps: float = 1e-12,
):

    data = np.asarray(data, dtype=float)
    N, d = data.shape

    distances, indices = _initial_neighbor_indices_and_distances(data, knn)
    init_prop: List[float] = []
    init_angles: List[float] = []
    adjusted_prop: List[float] = []
    adjusted_angles: List[float] = []
    rho: List[float] = []



    for idx in range(N):
        try:
            vec = data[indices[idx]] - data[idx]   # (knn, d)
            rho_i = local_density(vec, eps=eps)

            # ----------------------------
            # find one dependent pair (optional)
            # ----------------------------
            dep_pairs = []
            for k in range(knn):
                for l in range(k + 1, knn):
                    if sont_dependants(vec[k], vec[l], eps=eps):
                        dep_pairs.append((k, l))

            dep_pair: Optional[Tuple[int, int]] = dep_pairs[0] if dep_pairs else None

            # ----------------------------
            # pairwise angles among vec (store tuples)
            # exclude ONLY that dependent pair (and symmetric)
            # ----------------------------
            angles: List[Tuple[float, Tuple[int, int]]] = []
            for x in range(knn - 1):
                nx = np.linalg.norm(vec[x])
                if nx <= eps:
                    continue
                for y in range(x + 1, knn):
                    if dep_pair is not None and ((x, y) == dep_pair or (y, x) == dep_pair):
                        continue
                    ny = np.linalg.norm(vec[y])
                    if ny <= eps:
                        continue

                    dot = float(np.dot(vec[x], vec[y]))
                    theta = float(np.degrees(np.arccos(np.clip(dot / (nx * ny), -1.0, 1.0))))
                    angles.append((theta, (x, y)))

            valid_angles = [(t, pair) for (t, pair) in angles if angle_min_deg <= t <= angle_max_deg]
            if not valid_angles:
                raise ValueError("No valid pairwise angle found in vec.")

            max_angle, (m, n) = max(valid_angles, key=lambda t: t[0])

            # ----------------------------
            # project remaining vectors onto plane span(vec[m], vec[n])
            # ----------------------------
            proj_list = []
            proj_source_indices = []
            for z in range(knn):
                if z in (m, n):
                    continue
                try:
                    _, proj_z = projection_gram_schmidt(vec[z], vec[m], vec[n], eps=eps)
                    proj_list.append(proj_z)
                    proj_source_indices.append(z)
                except ValueError:
                    continue

            # stack base plane vectors + projections
            stack_items = [vec[m], vec[n]] + proj_list
            stack_source_indices = [m, n] + proj_source_indices
            vectors = np.vstack(stack_items)  # (2 + nproj, d)

            # ----------------------------
            # angles among vectors
            # ----------------------------
            labels = list(range(len(vectors)))
            angles_int: List[Tuple[float, Tuple[int, int]]] = []
            for o in range(len(vectors) - 1):
                no = np.linalg.norm(vectors[o])
                if no <= eps:
                    continue
                for p in range(o + 1, len(vectors)):
                    np_ = np.linalg.norm(vectors[p])
                    if np_ <= eps:
                        continue
                    dot = float(np.dot(vectors[o], vectors[p]))
                    theta = float(np.degrees(np.arccos(np.clip(dot / (no * np_), -1.0, 1.0))))
                    angles_int.append((theta, (labels[o], labels[p])))

            denom = knn - 2
            init_prop_i, item_i = check_pre_concavity(angles_int, denom=denom, tol=1e-4)
            adj_prop_i, new_angle = init_prop_i, max_angle #retain initial values if not ok.
            ok, info = check_concavity(init_prop_i, prop_thresh, max_angle, item_i, angular_gap)
            if ok:
                adj_prop_i = info["prop_in_gap"]
                new_angle = info.get("gap")

            
            adjusted_prop.append(adj_prop_i)
            adjusted_angles.append(float(new_angle))
            init_prop.append(init_prop_i)
            init_angles.append(float(max_angle))
            rho.append(rho_i)


        except ValueError:
            init_prop.append(np.nan)
            init_angles.append(np.nan)
            adjusted_prop.append(np.nan)
            adjusted_angles.append(np.nan)
            rho.append(np.nan)


    # ----------------------------
    # relative density
    # rel_dens[i] = rho[i] / max_{j in knn(i)} rho[j]
    # ----------------------------
    #densities = np.asarray(rho, dtype=float)
    #rel_dens = np.full(N, np.nan, dtype=float)

    #for i in range(N):
    #    neigh = densities[indices[i]]
     #   max_neigh = np.nanmax(neigh)
      #  if np.isfinite(densities[i]) and np.isfinite(max_neigh) and max_neigh > 0:
       #     rel_dens[i] = densities[i] / max_neigh

    return {
        "initial_bos": np.asarray(init_prop, dtype=float),
        "initial_angles": np.asarray(init_angles, dtype=float),
        "adjusted_bos": np.asarray(adjusted_prop, dtype=float),
        "adjusted_angles": np.asarray(adjusted_angles, dtype=float),
        "indices": indices,
        "distances": distances,
    }
#Usage:
# results_adjusted = pipeline_adjusted(data, knn=17, prop_thresh=0.7, angular_gap=90)
#results_adjusted["initial_bos"], results_adjusted["initial_angles"], results_adjusted["adjusted_bos"], results_adjusted["adjusted_angles"]


def pipeline_oriented(
    data: np.ndarray,
    knn: int = 32,
    direction_threshold: float = 0.70,
    angle_min_deg: float = 1.0,
    angle_max_deg: float = 179.9,
    eps: float = 1e-12,
):

    data = np.asarray(data, dtype=float)
    N, d = data.shape

    bary = np.mean(data, axis=0)

    distances, indices = _initial_neighbor_indices_and_distances(data, knn)

    all_angles: List[float] = []
    adjusted_angles: List[float] = []
    rho: List[float] = []
    prop: List[float] = []
    raw_prop: List[float] = []

    for idx in range(N):
        rho_i = np.nan
        try:
            vec = data[indices[idx]] - data[idx]
            rho_i = local_density(vec, eps=eps)
            xb = bary - data[idx]
            xb_norm_val = np.linalg.norm(xb)
            if xb_norm_val <= eps:
                raise ValueError("xb is (near) zero; cannot define xb_norm.")
            xb_norm = xb / xb_norm_val

            dep_pairs = []
            for k in range(knn):
                for l in range(k + 1, knn):
                    if sont_dependants(vec[k], vec[l], eps=eps):
                        dep_pairs.append((k, l))

            dep_pair: Optional[Tuple[int, int]] = dep_pairs[0] if dep_pairs else None

            angles: List[Tuple[float, Tuple[int, int]]] = []
            for x in range(knn - 1):
                x_norm_den = np.linalg.norm(vec[x])
                if x_norm_den <= eps:
                    continue
                x_norm = vec[x] / x_norm_den

                for y in range(x + 1, knn):
                    if dep_pair is not None and ((x, y) == dep_pair or (y, x) == dep_pair):
                        continue

                    y_norm_den = np.linalg.norm(vec[y])
                    if y_norm_den <= eps:
                        continue
                    y_norm = vec[y] / y_norm_den

                    normal = x_norm + y_norm
                    normal_norm = np.linalg.norm(normal)
                    if normal_norm <= eps:
                        continue

                    direction = float(np.dot(normal / normal_norm, xb_norm))
                    if direction < direction_threshold:
                        continue

                    dot_xy = float(np.clip(np.dot(x_norm, y_norm), -1.0, 1.0))
                    theta_deg = float(np.degrees(np.arccos(dot_xy)))
                    angles.append((theta_deg, (x, y)))

            valid_angles = [
                (theta, pair) for theta, pair in angles if angle_min_deg <= theta <= angle_max_deg
            ]
            if not valid_angles:
                raise ValueError("No valid (theta, pair) found after gating.")

            max_angle, (m, n) = max(valid_angles, key=lambda t: t[0])
            all_angles.append(float(max_angle))

            proj_list = []
            for z in range(knn):
                if z in (m, n):
                    continue
                try:
                    _, proj_z = projection_gram_schmidt(vec[z], vec[m], vec[n], eps=eps)
                    proj_list.append(proj_z)
                except ValueError:
                    continue

            stack_items = [vec[m], vec[n]] + proj_list
            vectors = np.vstack(stack_items)

            labels = list(range(len(vectors)))
            angles_int: List[Tuple[float, Tuple[int, int]]] = []
            for o in range(len(vectors) - 1):
                no = np.linalg.norm(vectors[o])
                if no <= eps:
                    continue
                for p in range(o + 1, len(vectors)):
                    np_ = np.linalg.norm(vectors[p])
                    if np_ <= eps:
                        continue
                    dot = float(np.dot(vectors[o], vectors[p]))
                    theta_deg = float(np.degrees(np.arccos(np.clip(dot / (no * np_), -1.0, 1.0))))
                    angles_int.append((theta_deg, (labels[o], labels[p])))

            denom = knn - 2
            prop_i = check_sums(angles_int, denom=denom)

            rho.append(rho_i)
            prop.append(prop_i)

        except ValueError:
            all_angles.append(np.nan)
            prop.append(np.nan)
            rho.append(rho_i)

    densities = np.asarray(rho, dtype=float)
    rel_dens = np.full(N, np.nan, dtype=float)

    for i in range(N):
        neigh = densities[indices[i]]
        max_neigh = np.nanmax(neigh)
        if np.isfinite(densities[i]) and np.isfinite(max_neigh) and max_neigh > 0:
            rel_dens[i] = densities[i] / max_neigh

    return {
        "oriented_bos": np.asarray(prop, dtype=float),
        "oriented_angles": np.asarray(all_angles, dtype=float),
        "indices": indices,
        "distances": distances,
    }

#Usage: results_oriented = pipeline_oriented(data, knn=17, direction_threshold=0.7)
#results_oriented["oriented_bos"], results_oriented["oriented_angles"]
