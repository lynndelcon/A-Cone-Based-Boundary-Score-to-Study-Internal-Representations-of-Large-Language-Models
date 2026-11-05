"""Adapters for the three implementation modules in this folder.

Use --adapter my_methods_adapter_TEMPLATE, or copy to my_methods_adapter.py.
Only Hyperplane uses intrinsic_dim. Other methods use ambient coordinates;
Tangent Plane treats the unlabeled cloud as a single class.
"""
from pathlib import Path
import sys

# Support launching boundary_benchmark.py from inside its package directory.
_IMPLEMENTATION_DIR = str(Path(__file__).resolve().parent)
if _IMPLEMENTATION_DIR not in sys.path:
    sys.path.insert(0, _IMPLEMENTATION_DIR)

from bde_pipeline import pipeline_adjusted, pipeline_oriented
from hickok_boundary_estimators import directional_uniformity_scores, hyperplane_scores
from li_maguire_tangent_plane import tangent_plane_scores


def du(X, *, k, intrinsic_dim):
    """Hickok DU scores; higher values indicate stronger boundary evidence."""
    scores, _normals = directional_uniformity_scores(X, k=k)
    return scores


def hyperplane(X, *, k, intrinsic_dim):
    """Hickok HP scores using the benchmark's intrinsic dimension."""
    scores, _normals, _dimensions = hyperplane_scores(
        X, k=k, intrinsic_dim=intrinsic_dim
    )
    return scores


def tangent_plane(X, *, k, intrinsic_dim):
    """Li--Maguire continuous scores; higher values indicate a boundary.

    Default gamma only affects the discarded binary edge decisions. Duplicate
    coordinates retain the implementation's explicit ValueError policy.
    """
    scores, _is_edge, _normals = tangent_plane_scores(X, k=k)
    return scores


def adjusted_angle(X, *, k, intrinsic_dim, prop_thresh=0.7, angular_gap=30):
    """ONE combined BEPS call, preserving initial values as final fallbacks.

    Undefined results remain NaN. The source provides no explicit adjustment
    mask, so the benchmark infers numeric changes from these four arrays.
    """
    result = pipeline_adjusted(X, knn=k, prop_thresh=prop_thresh, angular_gap=angular_gap)
    return {
        'initial_bos': result['initial_prop'],
        'initial_angle_deg': result['initial_angles'],
        'adjusted_bos': result['adjusted_prop'],
        'adjusted_angle_deg': result['adjusted_angles'],
    }


def oriented(X, *, k, intrinsic_dim):
    """Independent oriented BDE: raw BoS and aperture in degrees.

    The user-defined distance estimate is 1 - 2*abs(raw BoS - 0.5).
    It lies in [0, 1]; no geometric scale calibration is applied.
    """
    result = pipeline_oriented(X, knn=k)
    return {
        'scores': result['oriented_prop'],
        'type': 'bos',
        'angle_deg': result['oriented_angles'],
        'distance_estimate': 1 - 2 * abs(result['oriented_prop'] - 0.5),
    }
