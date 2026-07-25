"""
Engine 2 -- semantic pass (DEVELOPMENTPLAN.md §2.7). "Useful, not authoritative":
one VLM call per sheet plus one global reconciliation call, against any
OpenAI-compatible chat-completions endpoint (LM Studio, vLLM, or a cloud API)
so it runs against a local model with no cloud dependency if desired.

Narrow, single-purpose prompts beat one giant prompt, especially on a local
model -- each call below asks for exactly one thing.

This module is plumbing only: it has no way to be exercised in this
environment without a live endpoint, so treat it as reviewed-but-untested.
Every extracted value keeps its `bbox_norm` so the approval window can render
the source crop next to the value (§2.7) -- never trust a read without that
crop, VLM digit reading fails confidently (§6.4).

Stdlib-only HTTP client (no extra dependency) since this is the one package
in the fork that talks to the network.
"""
import base64
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from PIL import Image

from .slots import DrawingSet, Face, ROLE_POLICY, Sheet

SHEET_SYSTEM_PROMPT = """You are reading one sheet from a hand-drawn orthographic engineering drawing.
Respond with JSON only, matching this exact shape, no prose, no markdown fences:
{
  "object_identification": {"reads_as": str, "implied_function": str, "confidence": "high"|"medium"|"low"},
  "annotations_read": [{"value": number, "raw_text": str, "axis_guess": "X"|"Y"|"Z"|null,
                         "bbox_norm": [x0,y0,x1,y1], "confidence": "high"|"medium"|"low"}],
  "ambiguous_regions": [{"bbox_norm": [x0,y0,x1,y1], "issue": str, "confidence": "high"|"medium"|"low"}],
  "unit_statements": [{"claim": str, "confidence": "high"|"medium"|"low"}]
}
bbox_norm is [x0,y0,x1,y1] normalised to [0,1] of this image. If a field has nothing to report, use an empty list."""

RECONCILE_SYSTEM_PROMPT = """You are given per-sheet JSON extractions from every view of one hand-drawn object,
plus the project note the author wrote. Merge them into one JSON object, no prose, no markdown fences:
{
  "object_identification": {"reads_as": str, "distinct_parts": [{"name": str, "evidence_slot": str,
                             "bbox_norm": [x0,y0,x1,y1]}], "implied_function": str, "confidence": "high"|"medium"|"low"},
  "annotations_read": [{"value": number, "raw_text": str, "slot": str, "axis_guess": "X"|"Y"|"Z"|null,
                         "bbox_norm": [x0,y0,x1,y1], "confidence": "high"|"medium"|"low"}],
  "ambiguous_regions": [{"slot": str, "bbox_norm": [x0,y0,x1,y1], "issue": str, "confidence": "high"|"medium"|"low"}],
  "unit_statements": [{"source": str, "claim": str, "confidence": "high"|"medium"|"low"}]
}
Resolve conflicts by preferring higher-confidence sources; do not invent parts or annotations not present in the input."""


@dataclass
class VLMConfig:
    base_url: str = "http://localhost:1234/v1"   # LM Studio default; point at any OpenAI-compatible endpoint
    model: str = "local-model"
    api_key: str = "not-needed"
    timeout: float = 120.0
    max_tokens: int = 2048
    temperature: float = 0.0


def _image_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert('RGB').save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{encoded}"


def _chat(config: VLMConfig, system_prompt: str, user_text: str, image: Optional[Image.Image] = None) -> dict:
    """POST one chat-completions request; parse the reply as JSON."""
    content = [{"type": "text", "text": user_text}]
    if image is not None:
        content.append({"type": "image_url", "image_url": {"url": _image_data_url(image)}})

    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    }
    req = urllib.request.Request(
        url=f"{config.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode('utf-8'),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            body = json.loads(resp.read().decode('utf-8'))
    except urllib.error.URLError as e:
        raise RuntimeError(f"VLM endpoint {config.base_url!r} unreachable: {e}") from e

    text = body['choices'][0]['message']['content']
    return _parse_json_reply(text)


def _parse_json_reply(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Local models often wrap JSON in ```json fences or add stray prose; fall
    # back to the outermost brace pair.
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"VLM reply was not JSON: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def extract_sheet(config: VLMConfig, sheet: Sheet, face: Face) -> dict:
    """Run the narrow per-sheet extraction prompt against one sheet's image."""
    user_text = f"This sheet is assigned to the '{face}' face, role '{sheet.role}'."
    if sheet.note:
        user_text += f" Author's note on this sheet: {sheet.note!r}"
    return _chat(config, SHEET_SYSTEM_PROMPT, user_text, image=sheet.image)


def reconcile_semantic(config: VLMConfig, per_sheet_results: Dict[str, dict], project_note: str) -> dict:
    """Global reconciliation call over every per-sheet extraction (§2.7 Engine 2)."""
    user_text = json.dumps({
        "project_note": project_note,
        "per_sheet": per_sheet_results,
    })
    return _chat(config, RECONCILE_SYSTEM_PROMPT, user_text)


def run_engine2(config: VLMConfig, drawing_set: DrawingSet) -> dict:
    """
    Run Engine 2 end to end: one extraction call per sheet whose role feeds the
    dimension source (§2.5 ROLE_POLICY -- primary, dimensioned, detail,
    section; 'alternate' sheets are skipped), then one reconciliation call.
    """
    per_sheet: Dict[str, dict] = {}
    for face, slot in drawing_set.slots.items():
        for sheet in slot.sheets:
            _cond, _sil, dimension_source = ROLE_POLICY[sheet.role]
            if not dimension_source:
                continue
            key = f"{face}:{sheet.id}"
            per_sheet[key] = extract_sheet(config, sheet, face)

    if not per_sheet:
        return {
            "object_identification": {"reads_as": "", "distinct_parts": [], "implied_function": "", "confidence": "low"},
            "annotations_read": [], "ambiguous_regions": [], "unit_statements": [],
        }
    return reconcile_semantic(config, per_sheet, drawing_set.project_note)
