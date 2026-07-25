# TRELLIS.2-Ortho — fork notes

Fork of [`microsoft/TRELLIS.2`](https://github.com/microsoft/TRELLIS.2), extending it to accept a set of
hand-drawn orthographic views (assigned to cube faces, with units declared explicitly) and produce a
3D model at true physical size, together with an itemised account of which geometry came from the
drawings and which the machine invented.

Full rationale, architecture, and roadmap: see `DEVELOPMENTPLAN.md` (design thesis, upstream findings,
data model, phase-by-phase plan, evaluation, and open decisions). This file only tracks fork mechanics.

## Pinned upstream commit

```
75fbf0183001ed9876c8dbb35de6b68552ee08bd  (2026-06-05)
Merge pull request #166 from microsoft/copilot/fix-failing-github-actions-job
```

11 commits, 4B params, DINOv3 ViT-L/16 conditioning. Rebase deliberately against upstream `main`, not
continuously — checkpoint compatibility matters more than being current. Update this pin (and re-run the
eval harness in `eval/`) whenever a rebase lands.

## Design thesis

> Generative for *form*. Deterministic for *scale*. Optimisation for *accuracy*. Human for *authority*.

Concretely: `trellis2/intake/` and `trellis2/refine/` are new, upstream-free packages doing the
deterministic/optimisation work (metric reconciliation, validation, silhouette fitting). Modifications to
upstream files are additive wherever possible (new methods beside old ones) so upstream diffs keep
landing clean. `app.py` is the one file expected to diverge materially.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Geometry core + eval harness | In progress — `trellis2/intake/metric_frame.py`, `eval/metrics.py` |
| 1 | Intake bench + gate | In progress — `trellis2/intake/{slots,geometry_checks,gate,report}.py` |
| 2 | Semantic pass + invention ledger | Partial — `visual_hull.py` (Engine 3) done; `semantic.py` (Engine 2) is a functional VLM-adapter stub, untested against a live endpoint |
| 3 | Silhouette refinement | Not started (needs `nvdiffrast` + GPU) |
| 4 | Ortho + line-art data generation | Not started |
| 5 | Multi-view fine-tune | Not started |
| 6 | Annotation constraints + fabrication export | Not started |

See `DEVELOPMENTPLAN.md` §4 for the full roadmap and §7 for open decisions that need a call before
Phase 2+ (third- vs first-angle default, anchor-conflict policy, invention-ledger storage location, etc).

## New packages (no upstream conflict)

```
trellis2/intake/       Sheet/ViewSlot/DrawingSet model, metric-frame solver, three validation engines,
                        IntakeReport, gate rules
eval/                  metrics.py (dimension error, IoU, invention ratio); build_benchmark.py (skeleton)
```

Nothing under `trellis2/intake/` or `eval/` touches upstream files or requires a GPU; it's importable and
testable with only `numpy`, `scipy`, and `Pillow`.
