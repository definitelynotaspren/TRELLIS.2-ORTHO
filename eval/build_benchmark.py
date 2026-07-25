"""
Synthetic ortho line-art benchmark (DEVELOPMENTPLAN.md §3.1, Phase 0):
~200 held-out assets, rendered to six-face ortho line art with known
ground-truth extents, for measuring dimension error / Chamfer / silhouette
IoU / invention ratio against `eval/metrics.py` before touching the model.

Asset selection and manifest I/O here are pure pandas/numpy and are exercised
by the test suite. `render_ortho_views` shells out to Blender via the existing
`data_toolkit/blender_script` machinery (the same pattern
`data_toolkit/render_cond.py` uses for training-view rendering) and is not
runnable in this environment -- it needs Blender and the source asset corpus.
Treat it as a documented interface, not a tested one.
"""
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trellis2.intake.slots import FACES, Face

BLENDER_SCRIPT = Path(__file__).resolve().parent.parent / 'data_toolkit' / 'blender_script' / 'render_cond.py'


@dataclass
class BenchmarkItem:
    sha256: str
    source_path: str
    ground_truth_extents: List[float]     # [X, Y, Z], real-world units of the source asset
    render_paths: Dict[str, str] = field(default_factory=dict)   # face -> rendered line-art image path


def select_holdout_assets(metadata_csv: str, n: int = 200, seed: int = 0, exclude_sha256: Optional[List[str]] = None) -> List[str]:
    """
    Sample `n` assets from a `metadata.csv` (upstream `data_toolkit` format) to
    hold out for the benchmark, excluding anything already used for training.
    """
    metadata = pd.read_csv(metadata_csv)
    if exclude_sha256:
        metadata = metadata[~metadata['sha256'].isin(set(exclude_sha256))]
    n = min(n, len(metadata))
    sampled = metadata.sample(n=n, random_state=seed)
    return sampled['sha256'].tolist()


def ground_truth_extents(mesh_path: str) -> np.ndarray:
    """
    Real-world [X, Y, Z] extents of the *unnormalised* source mesh. Must be
    read from the original asset, not the unit-cube-normalised training mesh
    `data_toolkit/blender_script/dump_mesh.py` produces, since normalisation
    deliberately discards absolute scale (§1.4).
    """
    import trimesh
    mesh = trimesh.load(mesh_path, force='mesh')
    bounds = mesh.bounds  # (2, 3): [min, max]
    return bounds[1] - bounds[0]


def render_ortho_views(source_path: str, out_dir: str, faces: List[Face] = FACES, ortho_scale: float = 1.15) -> Dict[Face, str]:
    """
    Render the six canonical ortho line-art views of `source_path` via Blender,
    matching the Phase 4 ortho + Freestyle render branch (DEVELOPMENTPLAN.md
    §3.2, `data_toolkit/blender_script/render_cond.py:405`). Requires that
    branch to exist (Phase 4) and a working Blender install; not exercised here.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        'blender', '-b', '-P', str(BLENDER_SCRIPT), '--',
        '--object', source_path,
        '--faces', json.dumps(list(faces)),
        '--ortho_scale', str(ortho_scale),
        '--output_folder', str(out),
    ]
    subprocess.run(cmd, check=True)
    return {face: str(out / f'{face}.png') for face in faces}


def build_manifest(metadata_csv: str, asset_root: str, out_dir: str, n: int = 200, seed: int = 0) -> str:
    """
    Select held-out assets, render their ortho views, record ground-truth
    extents, and write `benchmark_manifest.json` to `out_dir`. Returns the
    manifest path.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sha256s = select_holdout_assets(metadata_csv, n=n, seed=seed)

    items = []
    for sha256 in sha256s:
        source_path = str(Path(asset_root) / f'{sha256}.glb')
        item = BenchmarkItem(
            sha256=sha256,
            source_path=source_path,
            ground_truth_extents=ground_truth_extents(source_path).tolist(),
            render_paths=render_ortho_views(source_path, str(out / sha256)),
        )
        items.append(asdict(item))

    manifest_path = out / 'benchmark_manifest.json'
    manifest_path.write_text(json.dumps({'items': items}, indent=2))
    return str(manifest_path)


def load_manifest(manifest_path: str) -> List[BenchmarkItem]:
    data = json.loads(Path(manifest_path).read_text())
    return [BenchmarkItem(**item) for item in data['items']]
