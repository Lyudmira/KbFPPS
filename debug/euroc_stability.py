"""Stability sweep: is the ~0.5% EuRoC result robust to stride and RANSAC
threshold, or a lucky parameter combination? Reuses reproduce_euroc.run_euroc.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import reproduce_euroc as re

mav = Path(r"E:\KFPPS_data\euroc\MH_01_easy")

print(f"{'stride':>6} {'ransac_px':>9} {'pairs':>6} {'feasible':>8} {'focal_med':>9} {'cx_med':>7} {'cy_med':>7} {'relK%':>7}")
for stride in [5, 7, 10]:
    for thr in [0.5, 1.0, 2.0]:
        s = re.run_euroc(
            mav, stride=stride, max_pairs=68, min_angle_deg=5.0,
            max_features=4000, ratio=0.8, center_radius_px=50.0,
            ransac_threshold_px=thr,
        )
        if "K_median" in s:
            print(f"{stride:>6} {thr:>9.1f} {s['num_pairs']:>6} {s['num_feasible']:>8} "
                  f"{s['focal_median']:>9.2f} {s['cx_median']:>7.2f} {s['cy_median']:>7.2f} "
                  f"{s['relative_K_error_percent']:>7.3f}")
        else:
            print(f"{stride:>6} {thr:>9.1f} {s['num_pairs']:>6} {s['num_feasible']:>8}  (no feasible)")
