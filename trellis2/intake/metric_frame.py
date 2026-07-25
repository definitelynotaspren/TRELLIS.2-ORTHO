"""
The metric frame solver (DEVELOPMENTPLAN.md §2.4). Replaces
`Trellis2ImageTo3DPipeline.preprocess_image()` for multi-view, to-scale input:
that function crops each image independently around its own alpha bbox, which
silently rescales views relative to one another (§1.4). This module shares one
normalisation frame across all views instead.

Pure numpy/Pillow -- no torch, no GPU required.
"""
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from .slots import DrawingSet, Face, FACE_AXES, MetricFrame, Sheet

_AXIS_ORDER = ('X', 'Y', 'Z')


def ink_mask(image: Image.Image, ink_threshold: float = 0.92, alpha_threshold: int = 20) -> np.ndarray:
    """
    Boolean mask of "ink" pixels: drawing content vs. background.

    - If the image carries real transparency, alpha > `alpha_threshold` is ink.
    - Otherwise, pixels darker than `ink_threshold` (fraction of white) are ink --
      the white-ground / black-stroke line-art convention used throughout the plan
      (§2.7 Engine 1, §4 Phase 4 render pipeline).
    """
    rgba = np.array(image.convert('RGBA'))
    alpha = rgba[:, :, 3]
    if not np.all(alpha == 255):
        return alpha > alpha_threshold
    gray = rgba[:, :, :3].astype(np.float64).mean(axis=2) / 255.0
    return gray < ink_threshold


def ink_bbox_px(image: Image.Image, **mask_kwargs) -> Tuple[int, int, int, int]:
    """Pixel bbox `(x0, y0, x1, y1)` of ink content; `x1`/`y1` are exclusive."""
    mask = ink_mask(image, **mask_kwargs)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("No ink found in sheet -- image appears blank")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def sheet_ink_extents_world(sheet: Sheet) -> np.ndarray:
    """Physical (width, height) of a sheet's ink bbox, in the sheet's declared units_per_px."""
    if sheet.units_per_px is None:
        raise ValueError(f"Sheet {sheet.id!r} has no units_per_px set; cannot compute physical extents")
    x0, y0, x1, y1 = ink_bbox_px(sheet.image)
    return np.array([x1 - x0, y1 - y0], dtype=np.float64) * sheet.units_per_px


def reconcile_extents(drawing_set: DrawingSet) -> MetricFrame:
    """
    Least-squares reconciliation of doubly-measured axes (§2.3, §2.4 step 1-3).

    Every world axis is spanned by up to four primary sheets (e.g. Z by front,
    back, left, and right). Each sheet contributes one direct scalar
    observation of that axis; solving that overdetermined system by ordinary
    least squares is exactly the mean of the observations. It's kept as an
    explicit `lstsq` rather than a hand-written average because that's the
    natural place to add weighted/anchor observations later (§7.3).

    Raises if `drawing_set.units` is unset (no default, by design -- §2.6) or
    if a primary sheet is missing `units_per_px`.
    """
    if drawing_set.units is None:
        raise ValueError("DrawingSet.units must be declared before reconciling extents")

    measurements: Dict[str, List[float]] = {ax: [] for ax in _AXIS_ORDER}
    for face, sheet in drawing_set.primary_sheets().items():
        u_axis, v_axis = FACE_AXES[face]
        w_units, h_units = sheet_ink_extents_world(sheet)
        measurements[u_axis].append(w_units)
        measurements[v_axis].append(h_units)

    extents = np.zeros(3, dtype=np.float64)
    residuals: Dict[str, float] = {}
    for i, ax in enumerate(_AXIS_ORDER):
        vals = measurements[ax]
        if not vals:
            continue
        vals_arr = np.asarray(vals, dtype=np.float64)
        design = np.ones((len(vals_arr), 1))
        est, *_ = np.linalg.lstsq(design, vals_arr, rcond=None)
        extents[i] = est[0]
        if len(vals_arr) > 1 and extents[i] > 0:
            residuals[ax] = float((vals_arr.max() - vals_arr.min()) / extents[i])

    L = float(extents.max()) if extents.max() > 0 else 0.0
    return MetricFrame(extents_world=extents, unit_cube_scale=L, residuals=residuals, units=drawing_set.units)


def normalize_views(
    drawing_set: DrawingSet,
    frame: MetricFrame,
    canvas_size: int = 1024,
    background: Tuple[int, int, int] = (255, 255, 255),
) -> Dict[Face, Image.Image]:
    """
    Scale every primary sheet by the *same* factor -- `canvas_size / max(extents_world)`
    -- and letterbox (never crop) onto a `canvas_size` square (§2.4 step 4).

    This is the direct replacement for `preprocess_image()`'s independent
    per-view square-crop-and-resize: padding instead of cropping keeps a tall
    thin object tall and thin in every view's tokens, and preserves relative
    scale across views, which independent cropping destroys (§1.4).

    Mutates `frame.per_view_transform` in place with, per occupied face, the
    `(scale, (offset_x, offset_y))` mapping original sheet pixel coordinates to
    canvas pixel coordinates: `canvas_xy = scale * (orig_xy - bbox_origin) + offset`.
    """
    L = max(float(frame.extents_world.max()), 1e-12)
    px_per_unit = canvas_size / L

    out: Dict[Face, Image.Image] = {}
    transforms: Dict[Face, Tuple[float, Tuple[float, float]]] = {}
    for face, sheet in drawing_set.primary_sheets().items():
        if sheet.units_per_px is None:
            raise ValueError(f"Sheet on face {face!r} has no units_per_px set")
        x0, y0, x1, y1 = ink_bbox_px(sheet.image)
        ink = sheet.image.crop((x0, y0, x1, y1))

        scale = sheet.units_per_px * px_per_unit
        w_canvas = max(1, round((x1 - x0) * scale))
        h_canvas = max(1, round((y1 - y0) * scale))
        if w_canvas > canvas_size or h_canvas > canvas_size:
            raise ValueError(
                f"Sheet on face {face!r} does not fit a {canvas_size}px canvas at the shared "
                f"scale ({w_canvas}x{h_canvas}); increase canvas_size"
            )

        resized = ink.resize((w_canvas, h_canvas), Image.LANCZOS)
        mode = 'RGBA' if ink.mode == 'RGBA' else 'RGB'
        bg = background + (0,) if mode == 'RGBA' else background
        canvas = Image.new(mode, (canvas_size, canvas_size), bg)
        off_x = (canvas_size - w_canvas) // 2
        off_y = (canvas_size - h_canvas) // 2
        canvas.paste(resized, (off_x, off_y), mask=resized if mode == 'RGBA' else None)

        out[face] = canvas
        transforms[face] = (scale, (off_x - x0 * scale, off_y - y0 * scale))

    frame.per_view_transform = transforms
    return out
