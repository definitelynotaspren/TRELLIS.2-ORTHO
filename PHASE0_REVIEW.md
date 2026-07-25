# Phase 0/1 code review — state of the fork vs. DEVELOPMENTPLAN.md

Reviewed 2026-07-25 against the plan and the pinned upstream commit (`75fbf01`, see FORK.md).
Everything below was verified by running code, not by reading it.

---

## 1. What is verified working

### The plan's upstream findings (§1) all check out against the tree

Every line-level claim in the plan was re-verified against the actual source:

| Claim | Verified |
|---|---|
| `datasets/components.py:135` `MultiImageConditionedMixin` exists, is dead code, not registered in `datasets/__init__.py` | ✅ (grep for `MultiImage` in `datasets/__init__.py` returns nothing) |
| Trainer-side `MultiImageConditionedMixin.encode_images()` flattens per-sample views to `(V·N, 1024)` | ✅ `trainers/flow_matching/mixins/image_conditioned.py:178` |
| `MultiImageConditionedSparseFlowMatchingCFGTrainer` registered | ✅ `trainers/__init__.py:18` |
| `SLatFlowModel.forward` accepts `cond: list` → `VarLenTensor` (stages 2–3 multi-view is free) | ✅ `models/structured_latent_flow.py:179` |
| Stage-1 `SparseStructureFlowModel.forward` typed `cond: torch.Tensor` only (fixed-slot protocol needed) | ✅ `models/sparse_structure_flow.py:224` |
| `neg_cond` for varlen is a **list of 1-token zeros**, not `zeros_like` | ✅ `mixins/image_conditioned.py:216-218` — the plan's warning is real |
| Isotropic unit-cube normalisation destroys exactly one scalar | ✅ `data_toolkit/blender_script/dump_mesh.py:164` |
| Perspective-only training views, `lens = 16/tan(fov/2)` | ✅ `blender_script/render_cond.py:405` |
| `preprocess_image()` normalises each view independently (hostile to to-scale input) | ✅ `pipelines/trellis2_image_to_3d.py:127-162` |

### The new intake package (Phase 0/1 core)

All CPU-only, no torch import anywhere in `trellis2/intake/` or `eval/`. 32 unit tests pass in ~2 s.

- **Data model** (`slots.py`) — exclusive sheet↔face assignment (atomic `assign()`), one-primary-per-face invariant enforced, `FACE_R` matrices confirmed to be proper rotations.
- **Metric frame solver** (`metric_frame.py`) — least-squares axis reconciliation recovers a known box's extents to <0.5%; letterboxing confirmed to preserve aspect (a 100×40 front view fills the canvas in X and stays under half in Z).
- **Engine 1** (`geometry_checks.py`) — axis reconciliation, shared-axis profile alignment, and first-/third-angle mislabel detection all behave correctly on both consistent and deliberately mislabelled inputs.
- **Engine 3** (`visual_hull.py`) — **verified with an asymmetric round-trip**, the strongest check in the suite: a ground-truth voxel L-prism is rendered to six silhouettes using `FACE_R` directly, pushed through the full `DrawingSet → reconcile → carve` path, and the carved hull reproduces the ground truth **voxel-exactly (IoU 1.000)** in all three L-orientations. This rules out any mirror/flip inconsistency between `normalize_views`, the per-face cameras, and the carver — the class of bug that symmetric test objects can never catch, and that would otherwise surface months later as "the generated part is mirrored."
- **Gate + report** (`gate.py`, `report.py`) — all §2.8 blockers and warnings fire on the right inputs; `approve()` refuses while blockers exist; the approved metric contract serialises to JSON.
- **Eval metrics** (`eval/metrics.py`) — dimension error, Chamfer, silhouette IoU, topology, aggregate rollup all unit-tested.

---

## 2. Errors and gaps to correct before the next phase

Ordered by how hard they block progress. "Next phase" here means finishing Phase 0's deliverable
(*correctly-scaled output from a single drawing*) and Phase 2 (Engine 2 + full approval window).

### Blocking Phase 0 completion

1. **Pipeline integration does not exist yet.** None of the §3.2 modifications to
   `pipelines/trellis2_image_to_3d.py` have been made: no `preprocess_views(DrawingSet)`, no
   slot-aware `get_cond()`, no `MetricFrame` threading through `decode_latent()`
   (`mesh.vertices *= frame.unit_cube_scale`), no `run(IntakeReport)`. The intake package computes
   the right scale but **nothing applies it to a mesh yet**. This is the single item standing
   between the current code and Phase 0's deliverable. It needs a GPU machine with the model
   weights to implement and test — it cannot be validated in a CPU-only environment.

2. **The `num_samples > 1` latent bug (§3.2, line 188) is still present upstream.** `noise` is
   allocated `(num_samples, …)` while `cond` stays batch-1. Harmless while `num_samples=1`
   (the default), but it must be fixed (`cond.expand(num_samples, -1, -1)`) before anyone runs
   multi-sample generation with the intake path.

### Blocking Phase 2

3. **Engine 2 (`semantic.py`) has never run against a live endpoint.** The adapter is complete
   (per-sheet extraction prompt, global reconciliation prompt, JSON-fence fallback parsing) but is
   *reviewed-not-tested*. Before Phase 2 is called done it needs: a smoke test against LM Studio or
   any OpenAI-compatible server, schema validation of real model replies (local models will violate
   the JSON shape), and a retry/repair path for malformed output.

4. **`ui/intake_bench.html` is referenced by the plan as "(built)" but is not in the repository.**
   The plan treats it as the working reference for Phase 1's Gradio bench. Either the prototype
   needs to be recovered from wherever it lives, or Phase 1 starts by rebuilding it. Right now the
   approval window exists only as data (`IntakeReport.to_dict()`), which is fine for headless
   Phase 0 use but is not the §2.8 six-section window.

### Blocking Phase 3+

5. **`trellis2/refine/` does not exist.** `silhouette_fit.py` and `constraints.py` (§2.10) are
   unwritten — they need `nvdiffrast` and a GPU, which this environment doesn't have. Phase 3 is
   where accuracy actually comes from, so nothing generation-side should be *evaluated* for
   dimensional accuracy until at least stage (a) (global anisotropic scale, 3 params) lands.

6. **`eval/build_benchmark.py`'s `render_ortho_views()` calls a Blender interface that doesn't
   exist yet.** It passes `--faces` / `--ortho_scale` flags that `blender_script/render_cond.py`
   doesn't accept — the ortho+Freestyle branch is Phase 4 work (§3.2). The asset-selection and
   manifest I/O halves of the file are tested; the render half is a documented interface only.
   Anyone running it today gets a Blender argument error, by design, not silently wrong output.

### Known limitations that are working as intended (don't "fix" these)

- **`units_per_px` is user-supplied.** Automatic scale-bar/dimension reading is Engine 2 → Phase 6
  work. Until then the gate's `no_anchor_sheet` warning is the honest state of the world.
- **`possible_open_contour` in Engine 1 is a heuristic** (low fill ratio + no holes) and is
  documented as such; it cannot distinguish an outline drawing from a leaky contour.
- **Hull confidence uses axis families, not raw view counts** — front+back alone can never produce
  High confidence. This is deliberate (opposite views are a consistency check, not independent
  evidence) and now pinned by a test.

### Minor fixes already applied in this review

- `eval/metrics.aggregate_report` reported `worst = max` unconditionally, which is backwards for
  IoU-like scores where lower is worse. Now reports both `min` and `max` and lets the caller pick.
- The asymmetric round-trip check was promoted from a scratch script into the permanent suite
  (`test_carve_visual_hull_asymmetric_round_trip`, parametrised over all three L-orientations).

---

## 3. Recommended order of work

1. On a GPU machine: implement §3.2's `preprocess_views` / `decode_latent` scaling / `run(IntakeReport)` — closes Phase 0.
2. Baseline run: stock weights, front view only, through the metric frame — the plan's Phase 0 control.
3. Smoke-test `semantic.py` against LM Studio; add reply-schema validation — opens Phase 2.
4. Rebuild or recover `ui/intake_bench.html`; wire the Gradio bench (§3.3) — Phase 1 UI.
5. `trellis2/refine/silhouette_fit.py` stage (a) only — the 3-parameter anisotropic scale fit is
   most of Phase 3's value at a fraction of its effort.
