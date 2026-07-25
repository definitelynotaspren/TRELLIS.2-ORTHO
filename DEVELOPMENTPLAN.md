# TRELLIS.2-Ortho — Development Plan

**Fork of `microsoft/TRELLIS.2`** · multi-view, to-scale, hand-drawn orthographic input
Supersedes `trellis2-orthoview-design.md` and `trellis2-intake-spec.md`.
Baseline: upstream `main`, 11 commits, 4B params, DINOv3 ViT-L/16 conditioning.

---

## 0. Scope and thesis

**In:** a set of hand-drawn orthographic views, assigned by hand to cube faces, with units declared in plain language.
**Out:** a 3D model at true physical size, plus an honest, itemised account of which geometry came from the drawings and which the machine invented.

The request decomposes into three problems with wildly different costs, and keeping them separate is what makes the project tractable:

| | Problem | Nature | Cost |
|---|---|---|---|
| **A** | Accept *N* conditioning images instead of 1 | Plumbing — half-built upstream already | Days |
| **B** | Make the model *know which view is which* | Architecture + training | Weeks + 8×80GB |
| **C** | Preserve metric scale; handle line art; know what's unknown | Mostly **not** an ML problem | Weeks, mostly CPU |

**Design thesis, and the thing that determines the whole build order:**

> Generative for *form*. Deterministic for *scale*. Optimisation for *accuracy*. Human for *authority*.

A flow-matching model will never hit ±0.5 mm, and asking it to is the mistake that sinks projects like this. It is excellent at plausible topology. Everything metric is arithmetic and least squares, everything precise is an optimiser fitting to your own linework, and everything ambiguous gets escalated to you before generation runs — not explained away afterward.

Consequence: **Phases 0–3 contain no training whatsoever and probably deliver most of the value.** Phases 4–5 are the expensive part and are explicitly gated on measurement.

---

## 1. Upstream findings

Verified by reading the tree, not the README. Line references are against upstream `main` at fork time.

### 1.1 Multi-image is already half-built, and dormant

Microsoft left the scaffolding in but never wired it up:

- `trellis2/datasets/components.py:135` — `MultiImageConditionedMixin`. Samples 1–4 random views, crops each, returns `pack['cond'] = [torch.stack(cond_images)]`. **Not composed into any dataset class, not registered in `datasets/__init__.py`.** Dead code.
- `trellis2/trainers/flow_matching/mixins/image_conditioned.py:178` — trainer-side `MultiImageConditionedMixin` with `encode_images()`: batches all views through DINOv3, splits by per-sample view count, flattens each to `(V·N_patch, 1024)`.
- `trellis2/trainers/__init__.py:18` — `MultiImageConditionedSparseFlowMatchingCFGTrainer` **is** registered, and exists at `sparse_flow_matching.py:288`.
- `trellis2/models/structured_latent_flow.py:179-180` — `SLatFlowModel.forward` already accepts `cond: List[torch.Tensor]` → `VarLenTensor`.
- `trellis2/modules/sparse/attention/modules.py:99` — sparse cross-attention takes `context: Union[VarLenTensor, torch.Tensor]`.

**Stages 2 (shape SLat) and 3 (texture SLat) accept a variable number of views today, with zero model changes.**

### 1.2 The stage-1 gap

`SparseStructureFlowModel.forward` (`models/sparse_structure_flow.py:224`) is typed `cond: torch.Tensor` with no `VarLenTensor` branch. But the dense attention it feeds (`modules/attention/modules.py:88`) reads `Lkv = context.shape[1]` — KV length is unconstrained.

So a **fixed**-slot protocol (always V slots, blank-padded) works at stage 1 with no model code change at all: concatenate view tokens along the sequence axis into `(B, V·N, 1024)`. Variable V at stage 1 would require adding varlen support to the dense DiT. This is one of two reasons the fixed six-face design below is correct anyway.

### 1.3 There is no camera conditioning. Anywhere.

Conditioning is *only* DINOv3 patch tokens (`cond_channels: 1024`, `facebook/dinov3-vitl16-pretrain-lvd1689m`). No pose tokens, no Plücker rays, no view index. The model infers orientation purely from appearance, having learned the canonical Objaverse frame.

Training views (`data_toolkit/render_cond.py:29-47`): 16 Hammersley sphere views, **perspective**, FOV randomised 10°–70°, radius set so the unit-sphere-bounded object fills frame. `blender_script/render_cond.py:405` sets `cam.data.lens = 16/tan(fov/2)`.

Two consequences:
- Feeding six unlabelled views into stock weights gives the model no way to bind view→direction. Expect it to **degrade**, not improve. This is the whole justification for Phase 5.
- FOV 10° at radius ≈ 9.9 is very nearly orthographic, so an ortho fine-tune is a mild distribution shift, not a cliff.

### 1.4 Scale is destroyed twice, deliberately

1. **Asset normalisation**, `data_toolkit/blender_script/dump_mesh.py:164`:
   ```python
   scale = 1 / max(bbox_max - bbox_min)   # then centre
   ```
   Isotropic fit to the unit cube. **Proportions survive; absolute size does not.** This turns out to be good news — §2.3.

2. **Inference preprocessing**, `pipelines/trellis2_image_to_3d.py:127` `preprocess_image()`: computes the alpha bbox *of that one image*, crops square around it, resizes. Run it independently on a front view and a top view of the same object and you have silently rescaled them relative to each other.

   > **This is the single most damaging function in the upstream codebase for our use case.** Independent per-view normalisation is exactly what a to-scale workflow cannot tolerate. It gets replaced, not patched.

3. Export is hard-coded to `aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]]` in the `o_voxel.postprocess.to_glb` call.

### 1.5 Summary

| Upstream asset | Status | Consequence for the fork |
|---|---|---|
| Varlen cond, stages 2–3 | Works | Free |
| Dense cond, stage 1 | Works if V fixed | Fixed six-slot protocol |
| Multi-image dataset mixin | Dead code, untested | Compose + audit, don't trust |
| Multi-image trainer (sparse) | Registered | Reuse |
| Multi-image trainer (dense) | **Missing** | Write it |
| Camera conditioning | Absent | New params → Phase 5 |
| Isotropic normalisation | Present | Only one scalar lost — recoverable |
| `preprocess_image()` | Hostile | Replace |

---

## 2. Architecture

### 2.1 End to end

```
  sheets ─┐
          ├─▶ [1] INTAKE ─────────▶ IntakeReport ──▶ [approval gate] ──┐
  notes ──┘     slot assignment                        ▲               │
  units ──┘     3 validation engines                   │               │
                                                    human              │
                                                                       ▼
                       ┌───────────────────────────────────── [2] METRIC FRAME
                       │                                       shared normalisation
                       │                                       axis reconciliation
                       ▼                                       unit_cube_scale
              [3] CONDITIONING ──▶ [4] GENERATION ──▶ [5] REFINEMENT ──▶ [6] EXPORT
               primary sheets       SS flow             nvdiffrast          × scale
               + slot embeddings    shape SLat          silhouette fit      GLB
               (6 × 1024 tokens)    tex SLat            dim constraints     + report
```

Stages 1, 2, 5 and 6 are deterministic. Stage 4 is the only place a neural network decides anything, and stage 5 exists to correct it against your own drawings.

### 2.2 Data model

One model, unified across intake and pipeline. This supersedes the earlier standalone `OrthoView`.

```python
# trellis2/intake/slots.py
Face = Literal['front','back','left','right','top','bottom']
Role = Literal['primary','dimensioned','detail','section','alternate']

FACE_AXES = {'front':('X','Z'), 'back':('X','Z'),
             'left':('Y','Z'),  'right':('Y','Z'),
             'top':('X','Y'),   'bottom':('X','Y')}
SLOT_INDEX = {'front':0,'back':1,'left':2,'right':3,'top':4,'bottom':5,'iso':6,'free':7}
FACE_R = { ... }   # 3×3 world→camera per face, Objaverse/Blender frame

@dataclass
class Sheet:
    id: str
    image: Image.Image
    role: Role = 'primary'
    note: str = ''
    units_per_px: float | None = None

@dataclass
class ViewSlot:
    face: Face
    sheets: list[Sheet]        # ordered; exactly one has role='primary'
    note: str = ''

@dataclass
class DrawingSet:
    slots: dict[Face, ViewSlot]
    project_note: str
    units: Literal['mm','cm','in','ft-in']    # no default. see §2.6

@dataclass
class MetricFrame:
    extents_world: np.ndarray   # (3,) reconciled object size in declared units
    unit_cube_scale: float      # multiply unit-cube mesh by this
    residuals: dict             # per-axis disagreement %
    per_view_transform: dict    # face → (scale, offset) into the shared canvas
```

### 2.3 The scale insight

Because upstream normalisation is *isotropic*, the pipeline loses exactly one scalar. Recovering true metric scale therefore needs only:

1. **Correct proportions** — guaranteed if all views share one normalisation frame, and
2. **One metric anchor** — a scale bar, a dimension annotation, or a typed "this edge is 120 mm."

**No retraining is required for metric output.** We just have to stop throwing the scalar away.

Better: with three orthographic views the extents are measured *redundantly*.

| View | Constrains |
|---|---|
| Front / Back | X, Z |
| Left / Right | Y, Z |
| Top / Bottom | X, Y |

Every axis is measured twice. Solve the overdetermined system by least squares and **the residual is a free consistency check on the human's drafting** — "your top view is 4.2% wider in Y than your right view; which do you trust?" No ML, high value, and it is the kind of feedback that makes the tool trustworthy for someone about to cut material.

### 2.4 The metric frame solver

Replaces `preprocess_image()`.

1. Ink/alpha bbox per primary sheet → pixel extents.
2. `units_per_px` → physical extents.
3. Least-squares reconcile doubly-measured axes → `extents_world`; keep residuals.
4. `L = max(extents_world)`. Scale **every** view by the same factor so `L` maps to the canvas, then **letterbox, never crop.**
5. `unit_cube_scale = L`.

Step 4 is the crux. Padding instead of cropping means a tall thin object stays tall and thin in the tokens, which is precisely the signal we want the model to see, and it preserves cross-view relative scale that upstream destroys.

### 2.5 Role policy — what feeds what

Real drawing sets contain the front view three times: once clean, once with dimensions crowding the linework, once as a detail of the corner they kept getting wrong. So a face holds *many* sheets, and the role field routes each one.

```python
ROLE_POLICY = {
    # role          cond_tokens  silhouette_loss  dimension_source
    'primary':      (True,       True,            True),
    'dimensioned':  (False,      False,           True),
    'detail':       (False,      False,           True),
    'section':      (False,      False,           True),
    'alternate':    (False,      False,           False),
}
```

**Only the `primary` sheet becomes conditioning tokens.** Six faces at 1024px is already ~25k KV tokens (§6.6); if every sheet fed the DiT the design collapses under its own compute. Everything else feeds the validator, the constraint set, and the optimiser — as *numbers and text*, not tokens.

Two things must never enter the silhouette loss, and the role field is what prevents it:
- **Dimensioned sheets** — leader lines and numerals are ink outside the object outline. Fitting to that mask inflates the part.
- **Sections** — a section is a cut plane, not a projection. Its outline is a lie about the silhouette.

**Sheet movement.** Two distinct operations, both needed:

| Operation | Meaning | Mechanism |
|---|---|---|
| Reassign | Move a sheet to a different face | `assign()` strips the id from every slot before pushing — a sheet lives on exactly one face |
| Reorder | Change which sheet on a face is primary | carousel + `Make primary` |

Reassignment must be atomic. Silent duplicates across faces would double-count a measurement and corrupt the reconciliation.

### 2.6 Free text in, structured out, confirmed in the middle

```
free text ──▶ VLM extraction ──▶ structured draft ──▶ HUMAN CONFIRMS ──▶ metric contract
```

Two note fields, different jobs:
- **Per-slot note** — what's on *this* sheet. "The wobble on the lower edge is a chamfer, not a curve."
- **Project note** — what the object *is*, where the numbers come from, what isn't drawn.

And one hard structured field: **the units dropdown has no default and blocks the gate.** Everything else can be inferred and corrected downstream. A unit mix-up is a 25.4× error that produces a perfectly plausible-looking model, and it is the most likely way this tool ruins someone's material.

### 2.7 Validation: three engines, differently trustworthy

Keep these strictly separate in code and in the UI, because they warrant very different levels of belief.

**Engine 1 — Geometry checks (`intake/geometry_checks.py`) — trust this.** Pure computation on ink masks.
- Extents per sheet; axis reconciliation per §2.3.
- **Projection-convention detection.** In a correct third-angle set, front and side align row-wise on Z, front and top align column-wise on X. If Z aligns but the side view is mirrored, they've drawn first-angle (European) and mislabelled left/right. Most common orthographic mistake, silently produces a mirrored part, and it is catchable.
- Mask closure, hole detection, bilateral symmetry hints.

**Engine 2 — Semantic pass (`intake/semantic.py`) — useful, not authoritative.** One VLM call per sheet plus one global reconciliation call. Narrow single-purpose prompts beat one giant prompt, especially on a local model; runs against an OpenAI-compatible endpoint so LM Studio serves it without cloud dependency.

```json
{
  "object_identification": {
    "reads_as": "L-profile shelf bracket",
    "distinct_parts": [
      {"name": "mounting leg", "evidence_slot": "front", "bbox_norm": [0.08,0.12,0.31,0.86]},
      {"name": "corner gusset", "evidence_slot": "detail:S-05", "bbox_norm": [0.18,0.20,0.55,0.82]}
    ],
    "implied_function": "wall-mounted; carries downward load on the arm",
    "confidence": "high"
  },
  "annotations_read": [
    {"value": 120.0, "raw_text": "120", "slot": "front", "axis_guess": "X",
     "bbox_norm": [0.40,0.90,0.52,0.97], "confidence": "high"}
  ],
  "ambiguous_regions": [
    {"slot": "front", "bbox_norm": [0.55,0.60,0.75,0.80],
     "issue": "curve or chamfer — line weight does not distinguish", "confidence": "low"}
  ],
  "unit_statements": [
    {"source": "project_note", "claim": "numbers on FRONT are millimetres", "confidence": "high"}
  ]
}
```

Every extracted number carries `bbox_norm`, so the approval window renders the crop it came from beside the value. **Verification becomes one glance instead of an act of faith.** This matters more than it sounds: VLM digit reading on hand-drawn numerals — 1/7, 4/9, misplaced decimals — is the failure mode that will bite, and it fails confidently.

**Engine 3 — Visual hull (`intake/visual_hull.py`) — the honest knowledge boundary.** Carve a voxel grid against the primary silhouettes under the assigned ortho cameras. Cheap, no learning.

> The hull is the maximal object consistent with the drawings. Anything the generative model puts *inside* it that the hull doesn't require is invention. Anything protruding *outside* it is the model contradicting the drawings, and is a hard error.

That gives a computable, non-hand-wavy definition of "geometry that needs to be imagined":

| Confidence | Condition |
|---|---|
| **High** | Constrained by ≥2 views |
| **Medium** | Constrained by 1 view |
| **Low** | Interior, or occluded in every supplied view |

Report `hull_volume` vs `generated_volume` as a single invention ratio.

### 2.8 The approval window and gate

Six sections, in this order. The order is the argument: identity first (does the machine even know what this is), then the number that determines physical size, then the honest list of what's being made up.

1. **What I think this is** — object, parts, implied function. User notes are quoted back as overriding authority, and their *absence* is called out.
2. **Metric contract** — units, anchor sheet, per-axis resolved value, reading count, disagreement %, and the single scalar applied at export.
3. **Slot map** — which sheet on which face, which is primary, what's excluded from the silhouette fit and why.
4. **Geometry I will have to invent** — the Engine 3 ledger, each line High/Medium/Low with a one-clause reason.
5. **Redline notes** — blockers and warnings, visually separated.
6. **Gate** — Approve / Go back.

**Blockers** (cannot proceed): units not declared · fewer than two faces on different axes · any of X/Y/Z unconstrained · a face with sheets but no primary.

**Warnings** (proceed with acknowledgement): axis disagreement > 3% · no anchor sheet · projection convention ambiguous · single-view axis · low-confidence annotation read.

Approving **freezes the metric contract into the run record.** When the mesh comes out 4% narrow, the contract shows whether that came from the model or from the top view disagreeing with the side view all along.

### 2.9 Pose embedding (new params → Phase 5)

```python
# trellis2/modules/image_feature_extractor.py  (extend)
class MultiViewDinoV3FeatureExtractor(DinoV3FeatureExtractor):
    def __init__(self, model_name, image_size=512, num_slots=8, cond_channels=1024):
        super().__init__(model_name, image_size)
        self.slot_emb = nn.Embedding(num_slots, cond_channels)   # NEW, trainable
        nn.init.zeros_(self.slot_emb.weight)   # zero-init = exact stock behaviour at step 0

    @torch.no_grad()
    def encode_views(self, slots) -> torch.Tensor:
        feats = self(...)                                   # (V, N, 1024)
        idx = torch.tensor([SLOT_INDEX[s.face] for s in slots])
        return feats + self.slot_emb(idx)[:, None, :]
```

Zero-init matters: at initialisation the multi-view model is bit-identical to stock on a single front view, so fine-tuning starts from a working point rather than a scrambled one. Free cameras later = swap the embedding for a Plücker-ray MLP (6 channels/patch → `Linear(6, 1024)`, added the same way).

### 2.10 Silhouette refinement — where accuracy actually comes from

`nvdiffrast` and `nvdiffrec` are already upstream dependencies.

```python
# trellis2/refine/silhouette_fit.py
# 1. Render the generated mesh under the assigned ortho cameras.
# 2. Loss = per-view silhouette IoU vs. each primary sheet's ink mask,
#    + hard penalties on confirmed dimension constraints.
# 3. Optimise, in increasing order of freedom:
#      a. global anisotropic scale        (3 params)   <- fixes most of it
#      b. per-axis piecewise-linear scale (~30 params)
#      c. low-frequency FFD cage          (~10^2-10^3, Laplacian-regularised)
# 4. Report final per-axis residual in declared units.
```

Stage (a) alone typically closes most of the proportion gap, costs seconds, needs no training data. **This is how multi-view drawings genuinely improve the output before any fine-tuning exists** — through the optimiser, not the network.

---

## 3. The fork's diff surface

### 3.1 New packages (ours, no upstream conflict)

```
trellis2/intake/
    slots.py            Sheet, ViewSlot, DrawingSet, face geometry
    geometry_checks.py  Engine 1
    semantic.py         Engine 2 (VLM adapter, OpenAI-compatible)
    visual_hull.py       Engine 3
    metric_frame.py     reconciliation + shared normalisation (§2.4)
    report.py           IntakeReport dataclass + JSON schema
    gate.py             blocking rules
trellis2/refine/
    silhouette_fit.py   §2.10
    constraints.py      dimension constraints → penalties
ui/
    intake_bench.html   cube assignment prototype (built)
eval/
    build_benchmark.py  synthetic ortho line-art test set
    metrics.py          dimension error, Chamfer, IoU, topology
```

### 3.2 Modified upstream files

| File | Line | Change |
|---|---|---|
| `pipelines/trellis2_image_to_3d.py` | 127 | Keep `preprocess_image` for back-compat. Add `preprocess_views(DrawingSet) → (list[Image], MetricFrame)` per §2.4 |
| " | 164 | `get_cond()` accepts slots. Stage 1: `(1, V·N, 1024)` flattened. Stages 2–3: `[feats.reshape(-1,1024)]` list form. **`neg_cond` for varlen must be a list of 1-token zeros** (matching `mixins/image_conditioned.py:213`), not `zeros_like` |
| " | 188 | No model change. **Latent bug:** `noise` is `(num_samples,…)` but `cond` is batch-1 — add `cond.expand(num_samples,-1,-1)` before ever using `num_samples>1` |
| " | 237/277/391 | Unchanged. CFG (`samplers/classifier_free_guidance_mixin.py`) calls the model twice separately rather than batch-concatenating, so varlen cond is safe |
| " | 456 | `decode_latent()` threads `MetricFrame`; `mesh.vertices *= frame.unit_cube_scale` |
| " | 489 | `run()` takes an **approved `IntakeReport`**, not raw images. Derives `MetricFrame` from the report's reconciled extents rather than recomputing — the number the human approved is the number that ships |
| `modules/image_feature_extractor.py` | — | Add `MultiViewDinoV3FeatureExtractor` (§2.9) |
| `datasets/components.py` | 190 | **Audit** `pack['cond'] = [torch.stack(cond_images)]` against the collate before trusting it — list-of-one wrapping a `(V,3,H,W)` tensor, untested |
| `datasets/__init__.py` | 9–16, 42–45 | Register `MultiImageConditionedSLatShape`, `…SparseStructureLatent`, `…SLatPbr` |
| `trainers/flow_matching/flow_matching.py` | ~316 | **Write** `MultiImageConditionedFlowMatchingCFGTrainer` (dense, stage 1) — the missing counterpart |
| `trainers/.../image_conditioned.py` | — | Add **per-view CFG dropout** so the model works with 2 views as well as 6 and doesn't collapse into requiring all six |
| `data_toolkit/blender_script/render_cond.py` | 405 | Ortho branch: `cam.data.type='ORTHO'; cam.data.ortho_scale=…` (≈1.15 for margin). Add Freestyle line-art pass |
| `data_toolkit/render_cond.py` | 29–47 | Emit canonical six ortho views + slot + `ortho_scale` into `transforms.json`; retain the 16 perspective views for mixed training |
| `app.py` | ~526 | Replace single `gr.Image` with the cube bench — see §3.3 |

### 3.3 The Gradio seam

Gradio has no 3D face picker, so the cube ships as `gr.HTML` bridged to `gr.State` through a hidden `gr.Textbox` carrying slot-map JSON. The prototype's state object is already shaped for that handoff. Plus: sheet tray (`gr.Gallery`), per-slot and project notes, units dropdown, `Run intake check`, approval accordion. **`Generate` stays disabled until an approved `IntakeReport` exists.**

This is the seam most likely to be annoying in practice — budget for it, and consider a proper Gradio custom component if the JSON bridge fights back.

### 3.4 Upstream tracking

Upstream is young and active (11 commits at fork time). Keep merges cheap:

- Nearly all new logic lives in `trellis2/intake/`, `trellis2/refine/`, `eval/` — files upstream will never touch.
- Modifications to upstream files are **additive wherever possible** (new methods beside old ones, not rewrites). `preprocess_image()` survives untouched precisely so upstream diffs land clean.
- `app.py` is the one file we materially rewrite; accept that it diverges and re-port upstream demo changes by hand.
- Pin the upstream commit in `FORK.md` and rebase deliberately, not continuously. Checkpoint compatibility matters more than being current.

---

## 4. Roadmap

| Phase | What | Hardware | Rough time |
|---|---|---|---|
| **0** | Geometry core + eval harness | CPU, 1 GPU for baseline | 1–2 wks |
| **1** | Intake bench + gate | CPU | 2–3 wks |
| **2** | Semantic pass + invention ledger | CPU + local VLM | 2–3 wks |
| **3** | Silhouette refinement | 1 GPU | 2–4 wks |
| — | **← DECISION GATE. Measure. Consider stopping.** | | |
| **4** | Ortho + line-art data generation | CPU farm, TB storage | 4–8 wks |
| **5** | Multi-view fine-tune | 8×80GB | 4–8 wks |
| **6** | Annotation constraints + fabrication export | CPU | open |

### Phase 0 — Instrument first
Build the eval harness *before* changing the model, or you'll have no idea whether anything helps.
- Synthetic benchmark: 200 held-out assets → ortho line-art renders → known ground-truth geometry.
- Metrics: per-axis dimension error % · Chamfer after metric alignment · per-view silhouette IoU · topology sanity (genus, watertightness) · invention ratio.
- Baseline: stock TRELLIS.2 on the front view alone.
- Ship the metric frame (§2.4) and export scaling headless — driven by a scripted slot map, no UI yet.
- Control experiment: naive multi-view token concat into stock weights. **Expect it to underperform single-view** (§1.3). Measure anyway.

**Deliverable:** correctly-scaled output from a single drawing.

### Phase 1 — The bench
Slot model, cube assignment UI, Engine 1, gate rules, approval window skeleton. The prototype in `ui/intake_bench.html` is the working reference; this phase makes it real against Gradio and the geometry checker.

**Deliverable:** a human can assign sheets, declare units, and get a numeric consistency report on their own drafting. Independently useful even with generation switched off.

### Phase 2 — Judgement
Engine 2 (VLM identification, parts, annotation reading with source crops) and Engine 3 (visual hull, invention ledger with confidence). Full approval window.

**Deliverable:** the system can say "I don't know," and say precisely where.

### Phase 3 — Accuracy
Silhouette refinement stages (a) and (b). Multi-view drawings now genuinely improve results.

**Deliverable:** dimensional error plausibly from ~15% down to low single digits, with zero training.

> **Decision gate.** Measure against the Phase 0 harness. If Phase 3 hits the accuracy you need, **stop.** Phases 4–5 are product work, not tool work, and cost two orders of magnitude more.

### Phase 4 — Data
Ortho + Freestyle line-art render pipeline (silhouette + crease + border edges, white ground, black strokes; randomised line weight, stroke jitter/taper, dropped interior creases, paper composite). Filtered corpus — man-made/mechanical/furniture, ~100–300k assets — beats all of Objaverse-XL for this domain. Budget thousands of CPU-hours of Blender and a few TB.

### Phase 5 — Fine-tune
Freeze the backbone. Train the slot embedding + LoRA (rank 64–128) on cross-attention `to_kv`/`to_q` across all three DiTs: ~10⁸ trainable against 4×10⁹ frozen. That's the difference between "8×A100 for a few days" and "not happening."

Curriculum: (a) shaded ortho multi-view — teach view binding; (b) synthetic line art — close the domain gap; (c) a few hundred real human drawings — close the last gap, which is that real people draw wobbly lines, inconsistent views, and construction marks they forgot to erase.

Train stage 1 and stage 2 (shape) only. **Leave texture single-view** — a pencil drawing carries no material information and pretending otherwise wastes compute.

### Phase 6 — Annotation and fabrication
Dimension-line + text detection promoted from advisory (Engine 2) to hard constraints in the Phase 3 optimiser. Mesh→primitive fitting for anything heading to a machine.

---

## 5. Evaluation

| Metric | Instrument | Phase-3 target | Phase-5 target |
|---|---|---|---|
| Per-axis dimension error | vs. ground-truth extents | < 5% | < 2% |
| Chamfer (metric-aligned) | normalised to object diagonal | — | baseline −40% |
| Per-view silhouette IoU | primary sheets only | > 0.90 | > 0.94 |
| Invention ratio | `1 − hull∩gen / gen` | reported | reported |
| Hull violation volume | geometry outside the hull | ≈ 0 | ≈ 0 |
| Watertight / genus sane | topology check | > 95% | > 95% |

Hull violation is the strongest single signal in the set: it's the only metric that catches the model contradicting the drawings rather than merely guessing beyond them, and it needs no ground truth, so it also works on real user submissions in production.

---

## 6. Risks and honest caveats

**6.1 Consider the cheap route first.** A sketch→render adapter (small img2img: ortho line art → plausible shaded render) feeding *stock* TRELLIS.2, plus Phase 3 refinement, gets most of the way with none of Phase 4–5's cost. Phases 4–5 are worth it only if this becomes a product rather than a tool for ourselves.

**6.2 A generative model may be the wrong tool entirely for the easy cases.** Reconstructing a solid from three consistent orthographic projections is a well-studied deterministic problem (extrusion + boolean intersection of swept view prisms), and for prismatic mechanical parts it will beat any diffusion model on accuracy by a wide margin. The generative path earns its place on organic, complex, or under-specified forms — where the drawing genuinely doesn't determine the object and something has to hallucinate plausibly. **A hybrid that routes on "is this drawing prismatic?" is probably the strongest system**, and Engine 1 already computes most of what that router needs.

**6.3 A generated mesh is not a CAD B-rep.** If the destination is fabrication, Phase 6's primitive fitting is mandatory, not optional. Otherwise the output is a reference model, and it should be labelled as one in the UI.

**6.4 VLM digit reading fails confidently.** Mitigated by source crops in the approval window (§2.7), never by trusting the extraction.

**6.5 Units are a 25.4× landmine.** Mitigated by a no-default dropdown that blocks the gate (§2.6).

**6.6 Compute.** Inference needs ≥24 GB VRAM, Linux only. DINOv3-L/16 at 1024px is 4096 tokens **per view**; six views ≈ 25k cross-attention KV tokens against up to 49k sparse query tokens. Upstream's ~17s at 1024³ will grow substantially. Mitigation: run stage 1 and the LR cascade at 512px conditioning (1024 tokens/view), reserve full-res tokens for the HR shape stage. Fine-tuning 4B at these token counts realistically wants 8×80 GB.

**6.7 Deployment path: GGUF via trellis.cpp — develop in PyTorch, ship quantized.** [`pwilkin/trellis.cpp`](https://github.com/pwilkin/trellis.cpp) is a from-scratch C++/GGML reimplementation of TRELLIS.2 inference, numerically validated against torch (DiT rel. err ~2.8e-3), running the 1024 cascade on 16 GB and reportedly ~6 GB with quantized weights ([Aero-Ex/Trellis2-GGUF](https://huggingface.co/Aero-Ex/Trellis2-GGUF), [ilintar/trellis2-gguf](https://huggingface.co/ilintar/trellis2-gguf)), with CUDA/ROCm/Vulkan backends (AMD and integrated GPUs work; Windows too). **This is the intended deployment target, not the development brain:**

- *It cannot be the development brain.* It is single-image only (every §3.2 conditioning change would need a hand-port to GGML), GGUF/GGML is inference-only so Phase 5 LoRA fine-tuning is impossible on it, and its auto-matte/crop input path is exactly the per-view normalisation §1.4 forbids.
- *It slots cleanly under our thesis.* Scale is a deterministic multiplication on the exported mesh and Phase 3 refinement corrects form against the drawings — neither cares which brain generated the mesh. So trellis.cpp works today as a low-VRAM baseline generation backend (generate GLB there → apply `unit_cube_scale` → validate/refine here), and quantization noise lands precisely where the design has a correction mechanism. Whether Q4/Q8 hurts dimensional accuracy is a Phase 0 harness measurement, not a guess.
- *Long-term shape:* fine-tune multi-view in PyTorch (rented GPU time), convert the checkpoint to GGUF (trellis.cpp documents custom-checkpoint conversion), and port the slot-embedding conditioning to trellis.cpp as a bounded, contribute-upstream-sized change. Note the six-view model's KV tokens grow ~6× (§6.6), so its quantized footprint will exceed the single-view "6 GB" figure.

---

## 7. Open decisions

1. **Third-angle or first-angle as the declared default?** Engine 1 can detect and offer to swap, but one has to be the assumed convention. (US/makerspace context suggests third-angle.)
2. **Does `iso`/`free` slot survive?** The eight-slot embedding reserves space for a free camera. Worth keeping for a reference photo alongside the drawings, or is that scope creep?
3. **Anchor policy** when the anchor sheet's number conflicts with reconciliation — anchor wins outright, or anchor weighted heavily in the least squares? Outright is more predictable; weighted is more accurate.
4. **Where does the invention ledger live post-generation?** Embedded in GLB extras, a sidecar JSON, or both — matters if these models circulate beyond the person who made them.
5. **Do we ship the deterministic prismatic reconstructor (§6.2) as a parallel path**, or stay generative-only and accept the accuracy ceiling on simple parts?
