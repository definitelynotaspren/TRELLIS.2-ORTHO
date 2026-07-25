"""
Approval-gate rules (DEVELOPMENTPLAN.md §2.8). Blockers stop the run cold;
warnings require acknowledgement but don't. Kept as a standalone rule engine
over plain data (no Gradio/UI dependency) so it's usable headless, per Phase 0's
"ship the metric frame and export scaling headless -- driven by a scripted
slot map, no UI yet."
"""
from dataclasses import dataclass
from typing import List, Literal, Optional

from .geometry_checks import axis_reconciliation_report, detect_projection_convention
from .slots import DrawingSet, FACES, FACE_AXES

Severity = Literal['blocker', 'warning']

# The three axis-pair groups a face belongs to (front/back share one, etc).
# Used to check "fewer than two faces on different axes" (§2.8).
_AXIS_GROUPS = {
    frozenset(('X', 'Z')): ('front', 'back'),
    frozenset(('Y', 'Z')): ('left', 'right'),
    frozenset(('X', 'Y')): ('top', 'bottom'),
}


@dataclass
class GateIssue:
    severity: Severity
    code: str
    message: str


def check_gate(
    drawing_set: DrawingSet,
    disagreement_threshold: float = 0.03,
    has_anchor: bool = False,
    annotation_confidences: Optional[List[str]] = None,
) -> List[GateIssue]:
    """
    Evaluate every blocker and warning rule against `drawing_set`.

    Args:
        disagreement_threshold: axis residual fraction above which a warning fires (§2.8: 3%).
        has_anchor: whether an explicit metric anchor (scale bar / dimension / typed value)
            was supplied. Anchor identification is Engine 2's job (§2.7); passed in here
            rather than recomputed.
        annotation_confidences: confidence strings ('high'/'medium'/'low') for any
            VLM-read annotations (Engine 2), so a low-confidence read can warn.
    """
    issues: List[GateIssue] = []

    # --- Blockers -----------------------------------------------------
    if drawing_set.units is None:
        issues.append(GateIssue('blocker', 'units_undeclared',
                                 "Units have not been declared. There is no default -- "
                                 "a unit mix-up is a 25.4x error that looks perfectly plausible."))

    for face in FACES:
        slot = drawing_set.slots[face]
        if slot.sheets and slot.primary is None:
            issues.append(GateIssue('blocker', 'face_missing_primary',
                                     f"Face '{face}' has sheet(s) assigned but none is marked primary."))

    occupied = set(drawing_set.occupied_faces())
    groups_touched = sum(1 for faces in _AXIS_GROUPS.values() if occupied & set(faces))
    if groups_touched < 2:
        issues.append(GateIssue('blocker', 'insufficient_faces',
                                 "Fewer than two faces on different axes are assigned; "
                                 "the object's shape is underdetermined."))

    if drawing_set.units is not None:
        axes = drawing_set.constrained_axes()
        for ax in ('X', 'Y', 'Z'):
            if not axes[ax]:
                issues.append(GateIssue('blocker', 'axis_unconstrained',
                                         f"The {ax} axis is not constrained by any assigned primary view."))

    # --- Warnings -------------------------------------------------------
    if drawing_set.units is not None:
        try:
            recon = axis_reconciliation_report(drawing_set)
        except ValueError:
            recon = {}
        for ax, info in recon.items():
            if info['residual'] is not None and info['residual'] > disagreement_threshold:
                issues.append(GateIssue('warning', 'axis_disagreement',
                                         f"{ax} axis: views disagree by {info['residual']*100:.1f}% "
                                         f"(sources: {', '.join(info['sources'])})."))
            if info['num_measurements'] == 1:
                issues.append(GateIssue('warning', 'single_view_axis',
                                         f"{ax} axis is constrained by only one view ({info['sources'][0]})."))

        for (face_a, face_b), info in detect_projection_convention(drawing_set).items():
            if info['likely_convention_mismatch']:
                issues.append(GateIssue('warning', 'projection_convention_ambiguous',
                                         f"'{face_a}' and '{face_b}' do not look like consistent "
                                         f"opposite views (mirror IoU {info['mirror_iou']:.2f} vs. "
                                         f"direct IoU {info['direct_iou']:.2f}) -- check for a "
                                         f"first-/third-angle mislabel."))

    if not has_anchor:
        issues.append(GateIssue('warning', 'no_anchor_sheet',
                                 "No metric anchor (scale bar, dimension, or typed value) was supplied; "
                                 "scale is taken entirely from the declared units_per_px."))

    for conf in (annotation_confidences or []):
        if conf == 'low':
            issues.append(GateIssue('warning', 'low_confidence_annotation',
                                     "A read annotation has low confidence; verify against its source crop."))

    return issues


def blockers(issues: List[GateIssue]) -> List[GateIssue]:
    return [i for i in issues if i.severity == 'blocker']


def warnings(issues: List[GateIssue]) -> List[GateIssue]:
    return [i for i in issues if i.severity == 'warning']


def can_approve(issues: List[GateIssue]) -> bool:
    return len(blockers(issues)) == 0
