"""
Engine 3 -- visual hull (DEVELOPMENTPLAN.md §2.7). Voxel-carve against the
primary silhouettes under the assigned ortho cameras. Cheap, no learning: the
hull is the maximal object consistent with the drawings, so anything the
generative model places outside it contradicts the drawings (hard error, §5),
and anything it places inside that the hull doesn't require is invention that
can be reported, not hand-waved.

Confidence follows §2.7's table (High: >=2 views; Medium: 1 view; Low:
interior/occluded everywhere). "Views" is read here as *projection-axis
families* -- front/back both look along Y, left/right along X, top/bottom
along Z -- because a voxel deep inside a silhouette isn't more attested to by
having two views along the *same* axis agree (front and back are required to
roughly agree by the mirror identity in geometry_checks.py; that's a
consistency check, not independent evidence). A voxel is "seen" by a family
when it sits within a small margin of that family's silhouette boundary, i.e.
some view's outline is actually nearby, not just "technically inside."
"""
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .metric_frame import ink_mask, normalize_views
from .slots import DrawingSet, FACE_R, MetricFrame

_AXIS_FAMILIES: Dict[str, Tuple[str, ...]] = {
    'Y': ('front', 'back'),
    'X': ('left', 'right'),
    'Z': ('top', 'bottom'),
}

CONFIDENCE_LOW, CONFIDENCE_MEDIUM, CONFIDENCE_HIGH = 0, 1, 2
CONFIDENCE_NAMES = {CONFIDENCE_LOW: 'low', CONFIDENCE_MEDIUM: 'medium', CONFIDENCE_HIGH: 'high'}


def _world_to_canvas_px(face: str, points: np.ndarray, canvas_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """(N,3) normalised world points -> (row, col) float pixel coords in that face's canvas."""
    cam = points @ FACE_R[face].T
    u, v = cam[:, 0], cam[:, 1]
    col = (u + 0.5) * canvas_size
    row = (0.5 - v) * canvas_size
    return row, col


def carve_visual_hull(
    drawing_set: DrawingSet,
    frame: MetricFrame,
    resolution: int = 64,
    canvas_size: int = 256,
    boundary_margin_frac: float = 0.03,
) -> dict:
    """
    Carve a `resolution`^3 occupancy grid over the normalised [-0.5, 0.5]^3
    cube against every assigned primary sheet, and tag each surviving voxel
    High/Medium/Low per the confidence rule above.

    Returns a dict with `hull_mask` (bool grid), `confidence` (int grid, only
    meaningful where `hull_mask` is True), `hull_volume_frac` (fraction of the
    cube occupied), and `confidence_volume_frac` (fraction of *hull* volume at
    each confidence level).
    """
    canvases = normalize_views(drawing_set, frame, canvas_size=canvas_size)
    masks = {face: ink_mask(img) for face, img in canvases.items()}
    boundary_dist = {face: ndimage.distance_transform_edt(mask) for face, mask in masks.items()}
    margin_px = max(2.0, boundary_margin_frac * canvas_size)

    coords = (np.arange(resolution) + 0.5) / resolution - 0.5
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing='ij')
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    n = points.shape[0]

    survive = np.ones(n, dtype=bool)
    family_constrained = {fam: np.zeros(n, dtype=bool) for fam in _AXIS_FAMILIES}

    for face, mask in masks.items():
        row, col = _world_to_canvas_px(face, points, canvas_size)
        row_i = np.clip(np.round(row).astype(np.int64), 0, canvas_size - 1)
        col_i = np.clip(np.round(col).astype(np.int64), 0, canvas_size - 1)
        is_ink = mask[row_i, col_i]
        survive &= is_ink

        near_boundary = boundary_dist[face][row_i, col_i] <= margin_px
        for fam, faces in _AXIS_FAMILIES.items():
            if face in faces:
                family_constrained[fam] |= (is_ink & near_boundary)

    family_count = np.zeros(n, dtype=np.int64)
    for fam, faces in _AXIS_FAMILIES.items():
        if any(f in masks for f in faces):
            family_count += family_constrained[fam].astype(np.int64)

    confidence = np.where(family_count >= 2, CONFIDENCE_HIGH,
                  np.where(family_count == 1, CONFIDENCE_MEDIUM, CONFIDENCE_LOW))

    hull_mask = survive.reshape(resolution, resolution, resolution)
    confidence_grid = confidence.reshape(resolution, resolution, resolution)

    hull_count = int(hull_mask.sum())
    total = resolution ** 3
    conf_fracs = {}
    for level, name in CONFIDENCE_NAMES.items():
        c = int(np.logical_and(hull_mask.ravel(), confidence == level).sum())
        conf_fracs[name] = (c / hull_count) if hull_count else 0.0

    return {
        'hull_mask': hull_mask,
        'confidence': confidence_grid,
        'resolution': resolution,
        'hull_volume_frac': hull_count / total,
        'confidence_volume_frac': conf_fracs,
        'views_used': sorted(masks.keys()),
    }


def invention_ratio(hull_mask: np.ndarray, generated_mask: np.ndarray) -> dict:
    """
    Compare a generated occupancy grid against the visual hull (§2.7, §5).
    Both grids must be boolean and share the same shape/resolution and frame
    (normalised [-0.5, 0.5]^3 cube).

    - `invention_ratio`: fraction of generated volume *not* required by the
      hull (`1 - hull ∩ gen / gen`) -- geometry the model invented.
    - `hull_violation_frac`: fraction of generated volume that falls *outside*
      the hull entirely -- the model contradicting the drawings, which should
      be ~0 for a well-behaved run (§5, the strongest single signal in the
      eval set: it needs no ground truth and also works on real submissions).
    """
    if hull_mask.shape != generated_mask.shape:
        raise ValueError(f"Shape mismatch: hull {hull_mask.shape} vs generated {generated_mask.shape}")
    gen_vol = int(generated_mask.sum())
    if gen_vol == 0:
        return {'invention_ratio': 0.0, 'hull_violation_frac': 0.0, 'hull_volume': int(hull_mask.sum()), 'generated_volume': 0}
    inside = int(np.logical_and(hull_mask, generated_mask).sum())
    outside = gen_vol - inside
    return {
        'invention_ratio': 1.0 - inside / gen_vol,
        'hull_violation_frac': outside / gen_vol,
        'hull_volume': int(hull_mask.sum()),
        'generated_volume': gen_vol,
    }
