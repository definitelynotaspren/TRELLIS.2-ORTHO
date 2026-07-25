"""
Unified data model for ortho intake: sheets, view slots, drawing sets, and the
reconciled metric frame. See DEVELOPMENTPLAN.md §2.2.

Nothing in this module depends on torch, so it can be imported and unit-tested
without a GPU or the rest of the TRELLIS2 stack.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from PIL import Image

Face = Literal['front', 'back', 'left', 'right', 'top', 'bottom']
Role = Literal['primary', 'dimensioned', 'detail', 'section', 'alternate']
Units = Literal['mm', 'cm', 'in', 'ft-in']

FACES: Tuple[Face, ...] = ('front', 'back', 'left', 'right', 'top', 'bottom')

# World axes (X, Y horizontal; Z up) each face's image plane spans.
# Two faces per axis pair -> every axis is measured redundantly (§2.3).
FACE_AXES: Dict[Face, Tuple[str, str]] = {
    'front': ('X', 'Z'), 'back': ('X', 'Z'),
    'left': ('Y', 'Z'), 'right': ('Y', 'Z'),
    'top': ('X', 'Y'), 'bottom': ('X', 'Y'),
}

# Fixed conditioning-slot index. 'iso' and 'free' are reserved for a future
# free camera (§2.9, §7.2) and are not used by the current fixed six-face
# protocol, but keep their slots so the embedding table doesn't need to be
# resized later.
SLOT_INDEX: Dict[str, int] = {
    'front': 0, 'back': 1, 'left': 2, 'right': 3, 'top': 4, 'bottom': 5,
    'iso': 6, 'free': 7,
}
NUM_SLOTS = len(SLOT_INDEX)

# World -> camera rotation per face, in a Z-up world (matching upstream's
# Blender/Hammersley-sphere render convention, data_toolkit/render_cond.py).
# Camera space is (right, up, toward-viewer): a point's image coordinates are
# (x_cam, y_cam) and depth-from-camera is -z_cam (larger toward the viewer).
# This is the *physical* camera pose for each named face and is convention-
# invariant; first- vs third-angle drafting only affects where a sheet is
# placed on paper relative to the others, which geometry_checks detects from
# image content, not from this matrix.
def _face_rotation(forward: Tuple[float, float, float],
                    right: Tuple[float, float, float],
                    up: Tuple[float, float, float]) -> np.ndarray:
    back = -np.asarray(forward, dtype=np.float64)
    return np.stack([np.asarray(right, dtype=np.float64),
                      np.asarray(up, dtype=np.float64),
                      back])

FACE_R: Dict[Face, np.ndarray] = {
    'front': _face_rotation(forward=(0, 1, 0), right=(1, 0, 0), up=(0, 0, 1)),
    'back': _face_rotation(forward=(0, -1, 0), right=(-1, 0, 0), up=(0, 0, 1)),
    'right': _face_rotation(forward=(-1, 0, 0), right=(0, 1, 0), up=(0, 0, 1)),
    'left': _face_rotation(forward=(1, 0, 0), right=(0, -1, 0), up=(0, 0, 1)),
    'top': _face_rotation(forward=(0, 0, -1), right=(1, 0, 0), up=(0, 1, 0)),
    'bottom': _face_rotation(forward=(0, 0, 1), right=(1, 0, 0), up=(0, -1, 0)),
}

ROLES: Tuple[Role, ...] = ('primary', 'dimensioned', 'detail', 'section', 'alternate')

# role -> (feeds conditioning tokens, feeds silhouette loss, feeds dimension source). §2.5
ROLE_POLICY: Dict[Role, Tuple[bool, bool, bool]] = {
    'primary': (True, True, True),
    'dimensioned': (False, False, True),
    'detail': (False, False, True),
    'section': (False, False, True),
    'alternate': (False, False, False),
}


@dataclass
class Sheet:
    id: str
    image: Image.Image
    role: Role = 'primary'
    note: str = ''
    units_per_px: Optional[float] = None

    def __post_init__(self):
        if self.role not in ROLE_POLICY:
            raise ValueError(f"Unknown role {self.role!r}; expected one of {ROLES}")


@dataclass
class ViewSlot:
    face: Face
    sheets: List[Sheet] = field(default_factory=list)
    note: str = ''

    def __post_init__(self):
        if self.face not in FACE_AXES:
            raise ValueError(f"Unknown face {self.face!r}; expected one of {FACES}")

    @property
    def primary(self) -> Optional[Sheet]:
        primaries = [s for s in self.sheets if s.role == 'primary']
        if len(primaries) > 1:
            raise ValueError(
                f"Face {self.face!r} has {len(primaries)} primary sheets; exactly one is required"
            )
        return primaries[0] if primaries else None

    def make_primary(self, sheet_id: str) -> None:
        """Reorder: change which sheet on this face is primary (§2.5)."""
        found = False
        for s in self.sheets:
            if s.id == sheet_id:
                s.role = 'primary'
                found = True
            elif s.role == 'primary':
                s.role = 'alternate'
        if not found:
            raise KeyError(f"Sheet {sheet_id!r} is not on face {self.face!r}")


@dataclass
class DrawingSet:
    slots: Dict[Face, ViewSlot] = field(default_factory=lambda: {f: ViewSlot(face=f) for f in FACES})
    project_note: str = ''
    units: Optional[Units] = None  # no default, by design -- see DEVELOPMENTPLAN.md §2.6

    def __post_init__(self):
        for f in FACES:
            self.slots.setdefault(f, ViewSlot(face=f))

    def assign(self, sheet: Sheet, face: Face) -> None:
        """
        Reassign a sheet to `face`, removing it from every other face first so a
        sheet lives on exactly one face at a time (§2.5). Atomic: either the sheet
        ends up solely on `face`, or the DrawingSet is left unchanged.
        """
        if face not in FACES:
            raise ValueError(f"Unknown face {face!r}; expected one of {FACES}")
        for slot in self.slots.values():
            slot.sheets = [s for s in slot.sheets if s.id != sheet.id]
        self.slots[face].sheets.append(sheet)

    def find_sheet(self, sheet_id: str) -> Optional[Tuple[Face, Sheet]]:
        for face, slot in self.slots.items():
            for s in slot.sheets:
                if s.id == sheet_id:
                    return face, s
        return None

    def occupied_faces(self) -> List[Face]:
        return [f for f in FACES if self.slots[f].sheets]

    def primary_sheets(self) -> Dict[Face, Sheet]:
        out = {}
        for f in FACES:
            p = self.slots[f].primary
            if p is not None:
                out[f] = p
        return out

    def constrained_axes(self) -> Dict[str, List[Face]]:
        """Which world axes are measured, and by which faces, given assigned primaries."""
        axes: Dict[str, List[Face]] = {'X': [], 'Y': [], 'Z': []}
        for f in self.primary_sheets():
            for ax in FACE_AXES[f]:
                axes[ax].append(f)
        return axes


@dataclass
class MetricFrame:
    extents_world: np.ndarray          # (3,) reconciled [X, Y, Z] size, in DrawingSet.units
    unit_cube_scale: float             # multiply a [-0.5,0.5]^3-normalised mesh by this
    residuals: Dict[str, float] = field(default_factory=dict)      # axis -> disagreement fraction
    per_view_transform: Dict[Face, Tuple[float, Tuple[float, float]]] = field(default_factory=dict)
    units: Optional[Units] = None
