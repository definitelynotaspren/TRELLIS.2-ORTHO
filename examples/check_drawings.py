"""
Check a set of scanned orthographic drawings — no GPU, no AI model needed.

Give it one image per view of your object (front, top, left, ...), tell it the
units and the real-world size of one pixel, and it prints a plain-English
report: the object's measured size, whether your views agree with each other,
and anything that would block generation.

Example:
    python examples/check_drawings.py \
        --front scans/front.png --top scans/top.png --right scans/right.png \
        --units mm --units-per-px 0.2

`--units-per-px 0.2` means each pixel in your scans is 0.2 mm — e.g. a drawing
scanned at 127 DPI of a part drawn at 1:1. If you scanned at 300 DPI and drew
at full scale, one pixel is 25.4/300 = 0.0847 mm.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from trellis2.intake import (
    DrawingSet, FACES, Sheet, build_intake_report,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for face in FACES:
        parser.add_argument(f'--{face}', type=str, help=f'Image file for the {face} view')
    parser.add_argument('--units', required=True, choices=['mm', 'cm', 'in', 'ft-in'],
                        help='The real-world units your drawings are in (required, no default, on purpose)')
    parser.add_argument('--units-per-px', required=True, type=float,
                        help='Real-world size of one pixel in your scans (same for all views)')
    parser.add_argument('--note', default='', help='Optional note about what the object is')
    args = parser.parse_args()

    ds = DrawingSet(units=args.units, project_note=args.note)
    n_views = 0
    for face in FACES:
        path = getattr(args, face)
        if path:
            image = Image.open(path)
            ds.assign(Sheet(id=Path(path).name, image=image, units_per_px=args.units_per_px), face)
            n_views += 1
    if n_views == 0:
        parser.error('Provide at least one view, e.g. --front scans/front.png')

    print(f'\nChecking {n_views} view(s), units: {args.units}, {args.units_per_px} {args.units}/pixel')
    print('=' * 64)

    report = build_intake_report(ds, has_anchor=False)

    # --- measured size --------------------------------------------------
    if report.metric_frame is not None:
        x, y, z = report.metric_frame.extents_world
        print(f'\nMeasured size (width x depth x height):')
        print(f'  {x:.1f} x {y:.1f} x {z:.1f} {args.units}')
        recon = report.engine1.get('axis_reconciliation') or {}
        for ax, info in recon.items():
            if info['residual'] is not None:
                pct = info['residual'] * 100
                verdict = 'OK' if pct <= 3 else 'CHECK YOUR DRAWINGS'
                print(f'  {ax} axis measured by {info["num_measurements"]} views, '
                      f'they disagree by {pct:.1f}% -- {verdict}')

    # --- how much is guesswork ------------------------------------------
    if report.engine3_summary is not None:
        conf = report.engine3_summary['confidence_volume_frac']
        print(f'\nHow well your views pin down the shape:')
        print(f'  well-determined (seen by 2+ view directions): {conf["high"]*100:.0f}%')
        print(f'  partly determined (1 view direction):          {conf["medium"]*100:.0f}%')
        print(f'  would be guessed (hidden in every view):       {conf["low"]*100:.0f}%')

    # --- problems ---------------------------------------------------------
    print()
    if report.blockers:
        print('BLOCKED -- these must be fixed before generation could run:')
        for issue in report.blockers:
            print(f'  [X] {issue.message}')
    if report.warnings:
        print('Warnings -- generation could proceed, but double-check:')
        for issue in report.warnings:
            print(f'  [!] {issue.message}')
    if not report.blockers and not report.warnings:
        print('No problems found.')

    print()
    print('Ready to generate.' if report.can_approve() else 'Not ready yet -- fix the blocked items above.')


if __name__ == '__main__':
    main()
