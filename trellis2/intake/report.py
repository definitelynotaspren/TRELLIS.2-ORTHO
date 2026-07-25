"""
IntakeReport: the single object the approval window renders and, once
approved, the run record freezes (§2.8). Assembles Engine 1 (always), Engine 3
(if units are declared), and Engine 2 (if the caller ran it -- it needs a live
VLM endpoint, see semantic.py) into one JSON-serialisable report.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .gate import GateIssue, can_approve, check_gate
from .geometry_checks import run_engine1
from .metric_frame import reconcile_extents
from .slots import DrawingSet, MetricFrame
from .visual_hull import CONFIDENCE_NAMES, carve_visual_hull


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {(str(k) if not isinstance(k, str) else k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


@dataclass
class IntakeReport:
    slot_map: Dict[str, dict]                  # face -> {sheets: [...], primary: id|None}
    metric_frame: Optional[MetricFrame]
    engine1: dict
    engine2: Optional[dict]
    engine3_summary: Optional[dict]             # carve_visual_hull() output minus the raw voxel grids
    issues: List[GateIssue]
    project_note: str = ''
    approved: bool = False

    @property
    def blockers(self) -> List[GateIssue]:
        return [i for i in self.issues if i.severity == 'blocker']

    @property
    def warnings(self) -> List[GateIssue]:
        return [i for i in self.issues if i.severity == 'warning']

    def can_approve(self) -> bool:
        return can_approve(self.issues)

    def approve(self) -> None:
        """Freeze the metric contract into the run record (§2.8)."""
        if not self.can_approve():
            raise ValueError(
                f"Cannot approve: {len(self.blockers)} blocking issue(s): "
                + "; ".join(i.message for i in self.blockers)
            )
        self.approved = True

    def to_dict(self) -> dict:
        return _jsonable({
            'project_note': self.project_note,
            'slot_map': self.slot_map,
            'metric_frame': None if self.metric_frame is None else {
                'extents_world': self.metric_frame.extents_world,
                'unit_cube_scale': self.metric_frame.unit_cube_scale,
                'residuals': self.metric_frame.residuals,
                'units': self.metric_frame.units,
            },
            'engine1': self.engine1,
            'engine2': self.engine2,
            'engine3_summary': self.engine3_summary,
            'issues': [{'severity': i.severity, 'code': i.code, 'message': i.message} for i in self.issues],
            'approved': self.approved,
        })


def _slot_map(drawing_set: DrawingSet) -> Dict[str, dict]:
    out = {}
    for face, slot in drawing_set.slots.items():
        out[face] = {
            'sheets': [{'id': s.id, 'role': s.role, 'note': s.note} for s in slot.sheets],
            'primary': slot.primary.id if slot.primary else None,
            'note': slot.note,
        }
    return out


def build_intake_report(
    drawing_set: DrawingSet,
    hull_resolution: int = 64,
    canvas_size: int = 256,
    engine2_result: Optional[dict] = None,
    has_anchor: bool = False,
) -> IntakeReport:
    """
    Run Engine 1 (always) and Engine 3 (if units are declared, since carving
    needs a shared metric frame), fold in an already-computed Engine 2 result
    if given, evaluate the gate, and return the assembled report.

    Engine 3 is deliberately summarised, not embedded whole: the raw
    resolution^3 voxel grids belong in the run record / debug artifacts, not
    in the human-facing report.
    """
    engine1 = run_engine1(drawing_set)

    frame: Optional[MetricFrame] = None
    engine3_summary: Optional[dict] = None
    if drawing_set.units is not None:
        try:
            frame = reconcile_extents(drawing_set)
            if frame.unit_cube_scale > 0:
                hull = carve_visual_hull(drawing_set, frame, resolution=hull_resolution, canvas_size=canvas_size)
                engine3_summary = {
                    'hull_volume_frac': hull['hull_volume_frac'],
                    'confidence_volume_frac': hull['confidence_volume_frac'],
                    'views_used': hull['views_used'],
                    'resolution': hull['resolution'],
                }
        except ValueError:
            # No primary sheets yet, or a sheet is missing units_per_px -- gate
            # rules below will surface the underlying blocker/warning.
            pass

    annotation_confidences = None
    if engine2_result is not None:
        annotation_confidences = [a.get('confidence') for a in engine2_result.get('annotations_read', [])]

    issues = check_gate(
        drawing_set,
        has_anchor=has_anchor,
        annotation_confidences=annotation_confidences,
    )

    return IntakeReport(
        slot_map=_slot_map(drawing_set),
        metric_frame=frame,
        engine1=engine1,
        engine2=engine2_result,
        engine3_summary=engine3_summary,
        issues=issues,
        project_note=drawing_set.project_note,
    )
