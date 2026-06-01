"""Profile-optimizer slowness breakdown: vary num_focal_samples and max_nodes,
measure time AND accuracy, to see if multi_angle can be made overnight-feasible
without losing the result.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "vendor"))

import eccentricity_sweep as es

# Fixed off-axis, moderate noise scenario.
diag = 120.0 / math.sqrt(2.0)
cx, cy = es.W / 2.0 + diag, es.H / 2.0 + diag


def make_pairs(seed, npairs=4, noise=0.5):
    rng = np.random.default_rng(seed)
    return [es.sample_pair_correspondences(rng, cx=cx, cy=cy, angle_deg=20.0, noise=noise, baseline=0.1, n=200)
            for _ in range(npairs)]


print(f"scenario: ecc=120 (GT cx={cx:.0f},cy={cy:.0f}), 4 pairs, noise=0.5, multi_angle\n")
print(f"{'focal_samples':>13} {'max_nodes':>9} {'min_box':>7} {'time_s':>7} {'pp_err':>8} {'focal_err':>9}")
for nfs, mn, mbox in [(15, 8000, 1.5), (9, 4000, 2.0), (7, 2000, 3.0), (5, 1500, 4.0), (9, 2000, 3.0)]:
    # average over 3 seeds for a stable time/accuracy estimate
    times, pps, fes = [], [], []
    for seed in range(3):
        pairs = make_pairs(seed)
        t0 = time.perf_counter()
        est = es.estimate_profile(pairs, use_angle=True, num_focal_samples=nfs, max_nodes=mn, min_box_size_px=mbox)
        times.append(time.perf_counter() - t0)
        if est:
            pps.append(math.hypot(est[1] - cx, est[2] - cy))
            fes.append(abs(est[0] - 1000.0))
    tm = np.median(times)
    pp = np.median(pps) if pps else float("nan")
    fe = np.median(fes) if fes else float("nan")
    print(f"{nfs:>13} {mn:>9} {mbox:>7.1f} {tm:>7.1f} {pp:>8.2f} {fe:>9.2f}")
