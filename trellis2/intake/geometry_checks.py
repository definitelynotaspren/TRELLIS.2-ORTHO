"""
Engine 1 -- geometry checks (DEVELOPMENTPLAN.md §2.7). Pure computation on ink
masks: no VLM, no learned model, nothing probabilistic. This is the engine the
approval window is entitled to *trust* -- every number here is arithmetic or a
projection-geometry identity, not a guess.

Two checks lean on one identity worth stating explicitly: the orthographic
silhouette (outline) of an opaque solid, projected along a given world axis,
is the same shape whether you view it from the + or - side of that axis --
only the handedness (mirror) of the picture differs, because a shadow doesn't
care which way the light travels. So for each opposite face pair
(front/back, left/right, top/bottom), a correctly-labelled, correctly-drawn
set should show one sheet as close to a mirror image of the other. A low
mirror score is real signal: wrong content, a first-/third-angle mislabel, or
gross drafting inconsistency -- not merely "the object isn't symmetric."
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .metric_frame import ink_bbox_px, ink_mask, reconcile_extents
from .slots import DrawingSet, Face, FACE_AXES, Sheet

# Opposite face pairs and the axis their mirror relationship flips along, per
# the FACE_R camera bases in slots.py: front/back and left/right mirror
# horizontally (their "right" image axis flips sign); top/bottom mirrors
# vertically (their "up" image axis flips sign).
_MIRROR_PAIRS: Tuple[Tuple[Face, Face, str], ...] = (
    ('front', 'back', 'horizontal'),
    ('left', 'right', 'horizontal'),
    ('top', 'bottom', 'vertical'),
)

# Adjacent face pairs that share a world axis, and which profile axis ('row'
# or 'col') of each sheet's ink bbox that shared axis corresponds to -- read
# off FACE_AXES (u=column axis, v=row axis).
_SHARED_AXIS_PAIRS: Tuple[Tuple[Face, Face, str], ...] = (
    ('front', 'left', 'Z'), ('front', 'right', 'Z'), ('back', 'left', 'Z'), ('back', 'right', 'Z'),
    ('front', 'top', 'X'), ('front', 'bottom', 'X'), ('back', 'top', 'X'), ('back', 'bottom', 'X'),
    ('left', 'top', 'Y'), ('left', 'bottom', 'Y'), ('right', 'top', 'Y'), ('right', 'bottom', 'Y'),
)


def _profile_axis(face: Face, axis: str) -> str:
    u_axis, v_axis = FACE_AXES[face]
    if axis == u_axis:
        return 'col'
    if axis == v_axis:
        return 'row'
    raise ValueError(f"Axis {axis!r} is not spanned by face {face!r}")


def _cropped_mask(sheet: Sheet) -> np.ndarray:
    mask = ink_mask(sheet.image)
    x0, y0, x1, y1 = ink_bbox_px(sheet.image)
    return mask[y0:y1, x0:x1]


def _resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.resize((shape[1], shape[0]), Image.NEAREST)
    return np.array(img) > 127


def sheet_report(sheet: Sheet) -> dict:
    """Per-sheet geometry facts: bbox, extents, fill ratio, holes, bilateral symmetry."""
    mask = ink_mask(sheet.image)
    x0, y0, x1, y1 = ink_bbox_px(sheet.image)
    crop = mask[y0:y1, x0:x1]

    filled = ndimage.binary_fill_holes(crop)
    interior_holes = filled & ~crop
    num_holes = int(ndimage.label(interior_holes)[1]) if interior_holes.any() else 0
    fill_ratio = float(crop.mean())
    # Sparse ink with no detected interior holes is ambiguous: it may simply be
    # an outline drawing with no interior detail, or the outline may have a
    # gap that let the border flood-fill leak all the way through. Flagged as
    # a heuristic, not asserted.
    possible_open_contour = bool(fill_ratio < 0.35 and num_holes == 0)

    mirrored = crop[:, ::-1]
    common = _resize_mask(crop, (256, 256)), _resize_mask(mirrored, (256, 256))
    inter = np.logical_and(*common).sum()
    union = np.logical_or(*common).sum()
    symmetry_score = float(inter / union) if union else 0.0

    report = {
        'sheet_id': sheet.id,
        'bbox_px': (x0, y0, x1, y1),
        'extents_px': (x1 - x0, y1 - y0),
        'fill_ratio': fill_ratio,
        'num_interior_holes': num_holes,
        'possible_open_contour': possible_open_contour,
        'bilateral_symmetry_score': symmetry_score,
    }
    if sheet.units_per_px is not None:
        report['extents_world'] = ((x1 - x0) * sheet.units_per_px, (y1 - y0) * sheet.units_per_px)
    return report


def axis_reconciliation_report(drawing_set: DrawingSet) -> dict:
    """
    Per-axis reconciled extent, contributing faces, and disagreement fraction
    (§2.3). Thin wrapper around `metric_frame.reconcile_extents` that also
    reports which faces contributed to each axis, for the approval window's
    slot map (§2.8 §3).
    """
    axes = drawing_set.constrained_axes()
    frame = reconcile_extents(drawing_set)
    out = {}
    for i, ax in enumerate(('X', 'Y', 'Z')):
        out[ax] = {
            'value': float(frame.extents_world[i]),
            'sources': axes[ax],
            'num_measurements': len(axes[ax]),
            'residual': frame.residuals.get(ax),
        }
    return out


def shared_axis_alignment_report(drawing_set: DrawingSet) -> Dict[Tuple[Face, Face], dict]:
    """
    For each pair of assigned adjacent faces that share a world axis, correlate
    their ink occupancy profiles along that axis (row profile for Z, column
    profile for X/Y). Low correlation means the two views disagree about where,
    along the shared axis, the object actually is/isn't present -- a finer,
    per-row/column check than the single-scalar bbox comparison in
    `axis_reconciliation_report`.
    """
    primaries = drawing_set.primary_sheets()
    out = {}
    for face_a, face_b, axis in _SHARED_AXIS_PAIRS:
        if face_a not in primaries or face_b not in primaries:
            continue
        mask_a, mask_b = _cropped_mask(primaries[face_a]), _cropped_mask(primaries[face_b])
        prof_axis_a, prof_axis_b = _profile_axis(face_a, axis), _profile_axis(face_b, axis)

        prof_a = mask_a.any(axis=1) if prof_axis_a == 'row' else mask_a.any(axis=0)
        prof_b = mask_b.any(axis=1) if prof_axis_b == 'row' else mask_b.any(axis=0)

        n = max(len(prof_a), len(prof_b))
        prof_a = np.array(Image.fromarray((prof_a * 255).astype(np.uint8)).resize((1, n), Image.NEAREST)).ravel() > 127
        prof_b = np.array(Image.fromarray((prof_b * 255).astype(np.uint8)).resize((1, n), Image.NEAREST)).ravel() > 127

        inter = np.logical_and(prof_a, prof_b).sum()
        union = np.logical_or(prof_a, prof_b).sum()
        iou = float(inter / union) if union else 0.0
        out[(face_a, face_b)] = {'shared_axis': axis, 'profile_iou': iou}
    return out


def detect_projection_convention(drawing_set: DrawingSet) -> Dict[Tuple[Face, Face], dict]:
    """
    Opposite-face mirror-consistency check (§2.7 Engine 1). For each opposite
    pair present, mirror one sheet's silhouette along the axis its camera basis
    flips and compare against the other via IoU. A low score on an otherwise
    axis-aligned pair (see `shared_axis_alignment_report`) is the catchable
    signature of a first-/third-angle convention mismatch or a mislabelled
    face -- the most common orthographic drafting mistake, and one that
    silently produces a mirrored part if it ships uncaught.
    """
    primaries = drawing_set.primary_sheets()
    out = {}
    for face_a, face_b, mirror_axis in _MIRROR_PAIRS:
        if face_a not in primaries or face_b not in primaries:
            continue
        mask_a = _resize_mask(_cropped_mask(primaries[face_a]), (256, 256))
        mask_b = _resize_mask(_cropped_mask(primaries[face_b]), (256, 256))
        mirrored_b = mask_b[:, ::-1] if mirror_axis == 'horizontal' else mask_b[::-1, :]

        inter = np.logical_and(mask_a, mirrored_b).sum()
        union = np.logical_or(mask_a, mirrored_b).sum()
        mirror_iou = float(inter / union) if union else 0.0

        inter_d = np.logical_and(mask_a, mask_b).sum()
        union_d = np.logical_or(mask_a, mask_b).sum()
        direct_iou = float(inter_d / union_d) if union_d else 0.0

        # A correctly third-angle-consistent opposite pair should mirror-match
        # about as well as (usually better than) they direct-match, per the
        # projection identity in the module docstring. A pair that matches
        # noticeably *better* unmirrored than mirrored is the signature of a
        # first-/third-angle mislabel or a duplicated/misassigned sheet.
        out[(face_a, face_b)] = {
            'mirror_axis': mirror_axis,
            'mirror_iou': mirror_iou,
            'direct_iou': direct_iou,
            'likely_convention_mismatch': bool(direct_iou > 0.7 and direct_iou - mirror_iou > 0.15),
        }
    return out


def run_engine1(drawing_set: DrawingSet) -> dict:
    """Run every Engine 1 check and return one combined report."""
    primaries = drawing_set.primary_sheets()
    axis_reconciliation = None
    if drawing_set.units:
        try:
            axis_reconciliation = axis_reconciliation_report(drawing_set)
        except ValueError:
            # A primary sheet is missing units_per_px -- gate.py surfaces the
            # underlying blocker; the rest of Engine 1 still runs.
            pass
    return {
        'sheets': {face: sheet_report(sheet) for face, sheet in primaries.items()},
        'axis_reconciliation': axis_reconciliation,
        'shared_axis_alignment': shared_axis_alignment_report(drawing_set),
        'projection_convention': detect_projection_convention(drawing_set),
    }
