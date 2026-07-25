import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import aggregate_report, chamfer_distance, dimension_error, silhouette_iou, topology_report


def test_dimension_error_exact_match_is_zero():
    result = dimension_error([100, 60, 40], [100, 60, 40])
    assert result['max_pct'] == pytest.approx(0.0)


def test_dimension_error_reports_per_axis():
    result = dimension_error([104, 60, 40], [100, 60, 40])
    assert result['per_axis_pct']['X'] == pytest.approx(4.0)
    assert result['per_axis_pct']['Y'] == pytest.approx(0.0)
    assert result['max_pct'] == pytest.approx(4.0)


def test_chamfer_distance_identical_clouds_is_zero():
    pts = np.random.RandomState(0).rand(200, 3)
    result = chamfer_distance(pts, pts)
    assert result['chamfer'] == pytest.approx(0.0, abs=1e-9)


def test_chamfer_distance_offset_clouds_is_positive():
    pts_a = np.zeros((50, 3))
    pts_b = np.ones((50, 3))
    result = chamfer_distance(pts_a, pts_b)
    assert result['chamfer'] > 1.0
    normed = chamfer_distance(pts_a, pts_b, normalize_by=result['chamfer'])
    assert normed['chamfer_normalized'] == pytest.approx(1.0)


def test_silhouette_iou_identical_masks_is_one():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    assert silhouette_iou(mask, mask) == pytest.approx(1.0)


def test_silhouette_iou_disjoint_masks_is_zero():
    a = np.zeros((32, 32), dtype=bool)
    a[:16, :16] = True
    b = np.zeros((32, 32), dtype=bool)
    b[16:, 16:] = True
    assert silhouette_iou(a, b) == pytest.approx(0.0)


def test_topology_report_on_a_box():
    trimesh = pytest.importorskip('trimesh')
    mesh = trimesh.creation.box(extents=(1, 2, 3))
    report = topology_report(mesh)
    assert report['watertight'] is True
    assert report['genus'] == 0


def test_aggregate_report_rolls_up_scalars():
    per_item = [{'max_pct': 2.0}, {'max_pct': 4.0}, {'max_pct': 6.0}]
    agg = aggregate_report(per_item)
    assert agg['max_pct']['mean'] == pytest.approx(4.0)
    assert agg['max_pct']['max'] == pytest.approx(6.0)
    assert agg['max_pct']['min'] == pytest.approx(2.0)
    assert agg['max_pct']['n'] == 3
