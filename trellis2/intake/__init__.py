"""
trellis2.intake -- deterministic and semantic intake for hand-drawn
orthographic view sets. See DEVELOPMENTPLAN.md §2, §3.1.

No submodule here imports torch; the whole package is usable standalone (CPU,
numpy/scipy/Pillow only) ahead of and independent of the generation pipeline.
"""
from .slots import (
    Face, FACES, FACE_AXES, FACE_R, MetricFrame, ROLE_POLICY, ROLES, Role,
    Sheet, SLOT_INDEX, DrawingSet, Units, ViewSlot,
)
from .metric_frame import ink_bbox_px, ink_mask, normalize_views, reconcile_extents, sheet_ink_extents_world
from .geometry_checks import (
    axis_reconciliation_report, detect_projection_convention, run_engine1,
    shared_axis_alignment_report, sheet_report,
)
from .visual_hull import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM, carve_visual_hull, invention_ratio
from .gate import GateIssue, blockers, can_approve, check_gate, warnings
from .report import IntakeReport, build_intake_report

__all__ = [
    'Face', 'FACES', 'FACE_AXES', 'FACE_R', 'MetricFrame', 'ROLE_POLICY', 'ROLES', 'Role',
    'Sheet', 'SLOT_INDEX', 'DrawingSet', 'Units', 'ViewSlot',
    'ink_bbox_px', 'ink_mask', 'normalize_views', 'reconcile_extents', 'sheet_ink_extents_world',
    'axis_reconciliation_report', 'detect_projection_convention', 'run_engine1',
    'shared_axis_alignment_report', 'sheet_report',
    'CONFIDENCE_HIGH', 'CONFIDENCE_LOW', 'CONFIDENCE_MEDIUM', 'carve_visual_hull', 'invention_ratio',
    'GateIssue', 'blockers', 'can_approve', 'check_gate', 'warnings',
    'IntakeReport', 'build_intake_report',
]
