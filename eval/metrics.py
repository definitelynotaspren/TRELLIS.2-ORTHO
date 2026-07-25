"""
Evaluation metrics (DEVELOPMENTPLAN.md §5). Build this before changing the
model -- Phase 0's whole point is having a way to tell whether anything
helped. Everything here is numpy/scipy; mesh topology checks use trimesh
lazily (only imported inside `topology_report`) so the rest of the module
stays usable without it.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trellis2.intake.visual_hull import invention_ratio  # noqa: E402  (re-exported for convenience)

__all__ = ['dimension_error', 'chamfer_distance', 'silhouette_iou', 'topology_report', 'invention_ratio']


def dimension_error(measured_extents: np.ndarray, ground_truth_extents: np.ndarray) -> dict:
    """
    Per-axis relative dimension error (§5 target: <5% at Phase 3, <2% at Phase 5).
    Both arguments are `[X, Y, Z]` extents in the same units.
    """
    measured = np.asarray(measured_extents, dtype=np.float64)
    truth = np.asarray(ground_truth_extents, dtype=np.float64)
    if measured.shape != truth.shape:
        raise ValueError(f"Shape mismatch: measured {measured.shape} vs ground truth {truth.shape}")
    with np.errstate(divide='ignore', invalid='ignore'):
        pct = np.abs(measured - truth) / truth * 100
    pct = np.where(truth == 0, np.nan, pct)
    return {
        'per_axis_pct': {ax: float(v) for ax, v in zip('XYZ', pct)},
        'max_pct': float(np.nanmax(pct)) if not np.all(np.isnan(pct)) else None,
        'mean_pct': float(np.nanmean(pct)) if not np.all(np.isnan(pct)) else None,
    }


def chamfer_distance(points_a: np.ndarray, points_b: np.ndarray, normalize_by: Optional[float] = None) -> dict:
    """
    Symmetric nearest-neighbour Chamfer distance between two point clouds
    (§5: normalise by object diagonal for the reported metric). Sample the
    surfaces of both meshes to point clouds and metric-align them before
    calling this -- alignment is out of scope here, this is the distance only.
    """
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3 or b.ndim != 2 or b.shape[1] != 3:
        raise ValueError("points_a / points_b must be (N, 3) arrays")
    d_ab, _ = cKDTree(b).query(a)
    d_ba, _ = cKDTree(a).query(b)
    chamfer = float(d_ab.mean() + d_ba.mean())
    result = {'chamfer': chamfer, 'mean_a_to_b': float(d_ab.mean()), 'mean_b_to_a': float(d_ba.mean())}
    if normalize_by:
        result['chamfer_normalized'] = chamfer / normalize_by
    return result


def silhouette_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Per-view silhouette IoU (§5 target: >0.90 at Phase 3, >0.94 at Phase 5). Primary sheets only."""
    a, b = np.asarray(mask_a, dtype=bool), np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def topology_report(mesh) -> dict:
    """
    Watertightness / genus sanity (§5 target: >95% at Phase 3 and Phase 5).
    `mesh` is a `trimesh.Trimesh`; trimesh is imported lazily here so the rest
    of this module works without it installed.
    """
    watertight = bool(mesh.is_watertight)
    genus = int((2 - mesh.euler_number) / 2) if watertight else None
    return {
        'watertight': watertight,
        'euler_number': int(mesh.euler_number),
        'genus': genus,
        'num_bodies': int(mesh.body_count) if hasattr(mesh, 'body_count') else None,
    }


def aggregate_report(per_item: List[dict]) -> dict:
    """Mean/median/worst-case rollup of a list of per-item metric dicts sharing scalar-valued keys."""
    if not per_item:
        return {}
    keys = set().union(*(d.keys() for d in per_item))
    out: Dict[str, dict] = {}
    for k in keys:
        vals = [d[k] for d in per_item if isinstance(d.get(k), (int, float)) and d.get(k) is not None]
        if not vals:
            continue
        out[k] = {
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
            'worst': float(np.max(vals)),
            'n': len(vals),
        }
    return out
