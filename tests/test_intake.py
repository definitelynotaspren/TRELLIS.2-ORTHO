"""
Unit tests for trellis2.intake -- pure numpy/scipy/Pillow, no torch, no GPU.
Run with: pytest tests/test_intake.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trellis2.intake.slots import DrawingSet, FACE_AXES, FACE_R, FACES, MetricFrame, Sheet, SLOT_INDEX
from trellis2.intake.metric_frame import ink_bbox_px, normalize_views, reconcile_extents
from trellis2.intake.geometry_checks import (
    axis_reconciliation_report, detect_projection_convention, run_engine1, shared_axis_alignment_report,
)
from trellis2.intake.visual_hull import carve_visual_hull, invention_ratio
from trellis2.intake.gate import check_gate, can_approve
from trellis2.intake.report import build_intake_report


CANVAS = 60
MARGIN = 15


def rect_sheet(sheet_id: str, w_px: int, h_px: int, units_per_px: float = 1.0, margin: int = MARGIN) -> Sheet:
    """A sheet whose ink is a single filled rectangle w_px x h_px, on a white canvas."""
    img = Image.new('RGB', (w_px + 2 * margin, h_px + 2 * margin), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([margin, margin, margin + w_px - 1, margin + h_px - 1], fill=(0, 0, 0))
    return Sheet(id=sheet_id, image=img, units_per_px=units_per_px)


def box_drawing_set(x_mm=100.0, y_mm=60.0, z_mm=40.0, px_per_mm=1.0, faces=FACES) -> DrawingSet:
    """A DrawingSet for a rectangular box of the given real-world extents."""
    ds = DrawingSet(units='mm')
    dims = {'X': x_mm, 'Y': y_mm, 'Z': z_mm}
    for face in faces:
        u_axis, v_axis = FACE_AXES[face]
        w_px = round(dims[u_axis] * px_per_mm)
        h_px = round(dims[v_axis] * px_per_mm)
        sheet = rect_sheet(face, w_px, h_px, units_per_px=1.0 / px_per_mm)
        ds.assign(sheet, face)
    return ds


# ---------------------------------------------------------------- slots.py

def test_face_r_are_proper_rotations():
    for face, R in FACE_R.items():
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9, err_msg=face)
        assert np.linalg.det(R) == pytest.approx(1.0), face


def test_slot_index_covers_all_faces_plus_reserved():
    assert set(SLOT_INDEX) == set(FACES) | {'iso', 'free'}
    assert len(set(SLOT_INDEX.values())) == len(SLOT_INDEX)


def test_assign_is_exclusive_across_faces():
    ds = DrawingSet(units='mm')
    sheet = rect_sheet('s1', 10, 10)
    ds.assign(sheet, 'front')
    assert ds.slots['front'].sheets[0].id == 's1'
    ds.assign(sheet, 'top')
    assert ds.slots['front'].sheets == []
    assert ds.slots['top'].sheets[0].id == 's1'


def test_make_primary_reorders_without_duplicating():
    ds = DrawingSet(units='mm')
    a = rect_sheet('a', 10, 10)
    b = Sheet(id='b', image=a.image, role='alternate')
    ds.assign(a, 'front')
    ds.slots['front'].sheets.append(b)
    assert ds.slots['front'].primary.id == 'a'
    ds.slots['front'].make_primary('b')
    assert ds.slots['front'].primary.id == 'b'
    assert sum(1 for s in ds.slots['front'].sheets if s.role == 'primary') == 1


def test_two_primaries_on_one_face_raises():
    ds = DrawingSet(units='mm')
    slot = ds.slots['front']
    slot.sheets = [rect_sheet('a', 10, 10), rect_sheet('b', 10, 10)]
    with pytest.raises(ValueError):
        _ = slot.primary


# ------------------------------------------------------------ metric_frame.py

def test_ink_bbox_px_finds_rectangle():
    sheet = rect_sheet('s', 20, 10, margin=5)
    x0, y0, x1, y1 = ink_bbox_px(sheet.image)
    assert (x1 - x0, y1 - y0) == (20, 10)
    assert (x0, y0) == (5, 5)


def test_reconcile_extents_matches_known_box():
    ds = box_drawing_set(x_mm=100, y_mm=60, z_mm=40, px_per_mm=2.0)
    frame = reconcile_extents(ds)
    np.testing.assert_allclose(frame.extents_world, [100, 60, 40], atol=0.5)
    assert frame.unit_cube_scale == pytest.approx(100, abs=0.5)
    for residual in frame.residuals.values():
        assert residual < 0.02  # a consistent synthetic box should reconcile almost exactly


def test_reconcile_extents_requires_units():
    ds = box_drawing_set()
    ds.units = None
    with pytest.raises(ValueError):
        reconcile_extents(ds)


def test_normalize_views_letterboxes_not_crops():
    ds = box_drawing_set(x_mm=100, y_mm=60, z_mm=40, px_per_mm=1.0)
    frame = reconcile_extents(ds)
    canvases = normalize_views(ds, frame, canvas_size=200)
    assert set(canvases) == set(FACES)
    for img in canvases.values():
        assert img.size == (200, 200)
    # front spans (X, Z) = (100, 40); with L=100 the X extent should fill the
    # canvas width and the Z extent should occupy well under half the height.
    front_ink = np.array(canvases['front'].convert('L')) < 235
    ys, xs = np.nonzero(front_ink)
    assert (xs.max() - xs.min()) > 0.9 * 200
    assert (ys.max() - ys.min()) < 0.5 * 200


# ---------------------------------------------------------- geometry_checks.py

def test_shared_axis_alignment_high_for_consistent_box():
    ds = box_drawing_set()
    report = shared_axis_alignment_report(ds)
    assert len(report) > 0
    for pair, info in report.items():
        assert info['profile_iou'] > 0.9, pair


def test_mirror_consistency_high_for_symmetric_box():
    ds = box_drawing_set()
    report = detect_projection_convention(ds)
    for pair, info in report.items():
        assert info['mirror_iou'] > 0.9, pair
        assert not info['likely_convention_mismatch']


def test_mirror_consistency_flags_mislabelled_side():
    """
    An asymmetric 'L' shape: draw the true right-side view correctly, but
    duplicate it (unflipped) into the 'left' slot -- the classic first-/third-
    angle mislabel this check exists to catch (§2.7 Engine 1).
    """
    def l_shape_sheet(sheet_id):
        img = Image.new('RGB', (60, 60), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([10, 10, 49, 25], fill=(0, 0, 0))
        draw.rectangle([10, 10, 20, 49], fill=(0, 0, 0))
        return Sheet(id=sheet_id, image=img, units_per_px=1.0)

    ds = DrawingSet(units='mm')
    ds.assign(l_shape_sheet('right'), 'right')
    duplicate = l_shape_sheet('left')
    ds.assign(duplicate, 'left')

    report = detect_projection_convention(ds)
    info = report[('left', 'right')]
    assert info['direct_iou'] > 0.95
    assert info['mirror_iou'] < info['direct_iou']
    assert info['likely_convention_mismatch']


# ------------------------------------------------------------- visual_hull.py

def test_carve_visual_hull_matches_box_volume_fraction():
    ds = box_drawing_set(x_mm=100, y_mm=60, z_mm=40, px_per_mm=1.0)
    frame = reconcile_extents(ds)
    hull = carve_visual_hull(ds, frame, resolution=48, canvas_size=192)
    expected_frac = (100 / 100) * (60 / 100) * (40 / 100)  # 0.24
    assert hull['hull_volume_frac'] == pytest.approx(expected_frac, abs=0.03)


def test_carve_visual_hull_confidence_needs_two_axis_families():
    # Only front+back are provided -> only the 'Y' axis family can ever be
    # near a silhouette boundary, so no voxel can reach High confidence
    # (which requires >=2 distinct axis families), by construction.
    ds = box_drawing_set(faces=['front', 'back'])
    frame = reconcile_extents(ds)
    hull = carve_visual_hull(ds, frame, resolution=32, canvas_size=128)
    assert hull['confidence_volume_frac']['high'] == pytest.approx(0.0)

    # With all six faces present, some axis-family overlap is inevitable near
    # the box's edges, so High confidence volume should be strictly positive.
    ds_full = box_drawing_set()
    frame_full = reconcile_extents(ds_full)
    hull_full = carve_visual_hull(ds_full, frame_full, resolution=32, canvas_size=128)
    assert hull_full['confidence_volume_frac']['high'] > 0.0


@pytest.mark.parametrize('shape_fn', [
    lambda x, y, z: (z < 15.0) or (x < 30.0),   # L in the XZ plane (asymmetric front/back view)
    lambda x, y, z: (z < 15.0) or (y < 20.0),   # L in the YZ plane (asymmetric left/right view)
    lambda x, y, z: (x < 30.0) or (y < 20.0),   # L in the XY plane (asymmetric top/bottom view)
])
def test_carve_visual_hull_asymmetric_round_trip(shape_fn):
    """
    Orientation-convention integration test. Symmetric boxes can't catch a
    mirrored face camera; an L-prism can. Build a ground-truth voxel L, render
    its six silhouettes directly from FACE_R (the convention the carver
    assumes), run the full DrawingSet -> reconcile -> carve path, and require
    the carved hull to reproduce the ground truth voxel-exactly. Any flip or
    mirror anywhere in normalize_views / _world_to_canvas_px / FACE_R breaks
    the asymmetry and collapses the IoU.
    """
    res, px_per_mm, margin = 48, 4, 30
    ext = np.array([100.0, 60.0, 40.0])
    L = ext.max()

    centers = (np.arange(res) + 0.5) / res - 0.5
    gx, gy, gz = np.meshgrid(centers, centers, centers, indexing='ij')
    pts_norm = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    pts_mm = pts_norm * L + ext / 2
    inside = np.all((pts_mm >= 0) & (pts_mm <= ext), axis=1)
    occ = np.array([shape_fn(*p) for p in pts_mm]) & inside
    gt = occ.reshape(res, res, res)

    ds = DrawingSet(units='mm')
    pts_centered = pts_mm - ext / 2
    for face in FACES:
        cam = pts_centered[occ] @ FACE_R[face].T
        u, v = cam[:, 0], cam[:, 1]
        w_px = int(np.ceil((u.max() - u.min()) * px_per_mm)) + 2
        h_px = int(np.ceil((v.max() - v.min()) * px_per_mm)) + 2
        img = np.full((h_px + 2 * margin, w_px + 2 * margin), 255, dtype=np.uint8)
        col = ((u - u.min()) * px_per_mm).astype(int) + margin
        row = ((v.max() - v) * px_per_mm).astype(int) + margin
        r_half = max(1, int(px_per_mm * L / res / 2) + 1)
        for c, r in zip(col, row):
            img[max(0, r - r_half):r + r_half, max(0, c - r_half):c + r_half] = 0
        ds.assign(Sheet(id=face, image=Image.fromarray(img).convert('RGB'),
                        units_per_px=1.0 / px_per_mm), face)

    frame = reconcile_extents(ds)
    hull = carve_visual_hull(ds, frame, resolution=res, canvas_size=192)
    carved = hull['hull_mask']
    union = np.logical_or(gt, carved).sum()
    iou = np.logical_and(gt, carved).sum() / union
    assert iou > 0.98
    # the hull must never exclude real geometry
    assert np.logical_and(gt, ~carved).sum() == 0

    # correctly drawn opposite views of an asymmetric object must still
    # mirror-match, and must not trip the convention-mismatch flag
    for pair, info in detect_projection_convention(ds).items():
        assert info['mirror_iou'] > 0.95, pair
        assert not info['likely_convention_mismatch'], pair


def test_invention_ratio_basic():
    hull = np.zeros((4, 4, 4), dtype=bool)
    hull[:2, :2, :2] = True
    gen = np.zeros((4, 4, 4), dtype=bool)
    gen[:2, :2, :2] = True
    gen[2:, 2:, 2:] = True  # half the generated volume is outside the hull
    result = invention_ratio(hull, gen)
    assert result['invention_ratio'] == pytest.approx(0.5)
    assert result['hull_violation_frac'] == pytest.approx(0.5)


# ------------------------------------------------------------------ gate.py

def test_gate_blocks_empty_drawing_set():
    ds = DrawingSet()
    issues = check_gate(ds)
    codes = {i.code for i in issues}
    assert 'units_undeclared' in codes
    assert 'insufficient_faces' in codes


def test_gate_passes_minimal_valid_set():
    ds = DrawingSet(units='mm')
    ds.assign(rect_sheet('front', 100, 40), 'front')
    ds.assign(rect_sheet('left', 60, 40), 'left')
    issues = check_gate(ds, has_anchor=True)
    assert can_approve(issues)


def test_gate_blocks_face_with_sheets_but_no_primary():
    ds = DrawingSet(units='mm')
    ds.assign(rect_sheet('front', 100, 40), 'front')
    ds.assign(rect_sheet('left', 60, 40), 'left')
    ds.slots['left'].sheets[0].role = 'detail'
    issues = check_gate(ds, has_anchor=True)
    assert not can_approve(issues)
    assert any(i.code == 'face_missing_primary' for i in issues)


def test_gate_warns_on_axis_disagreement():
    ds = DrawingSet(units='mm')
    ds.assign(rect_sheet('front', 100, 40), 'front')      # says Z = 40
    ds.assign(rect_sheet('left', 60, 60), 'left')          # says Z = 60 -- disagreement
    issues = check_gate(ds, has_anchor=True)
    codes = {i.code for i in issues}
    assert 'axis_disagreement' in codes


# ------------------------------------------------------------------ report.py

def test_build_intake_report_end_to_end_and_approve():
    ds = box_drawing_set()
    ds.project_note = 'test box'
    report = build_intake_report(ds, hull_resolution=24, canvas_size=96, has_anchor=True)
    assert report.can_approve()
    report.approve()
    assert report.approved
    as_dict = report.to_dict()
    assert as_dict['approved'] is True
    assert as_dict['metric_frame']['units'] == 'mm'


def test_report_approve_raises_when_blocked():
    ds = DrawingSet()
    report = build_intake_report(ds)
    with pytest.raises(ValueError):
        report.approve()
