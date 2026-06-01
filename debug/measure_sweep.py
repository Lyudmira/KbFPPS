"""Two quick measurements before committing to an overnight sweep:
  (A) correctness: Martyushev-only, noise-free, ecc=0 vs 240 -> expect ~1e-9.
  (B) timing: one profile solve vs one Martyushev solve, to budget the grid.
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

print("=== (A) correctness: Martyushev-only, noise-free ===")
for ecc in [0.0, 240.0]:
    diag = ecc / math.sqrt(2.0)
    cx, cy = es.W / 2.0 + diag, es.H / 2.0 + diag
    rng = np.random.default_rng(0)
    pairs = [es.sample_pair_correspondences(rng, cx=cx, cy=cy, angle_deg=20.0, noise=0.0, baseline=0.1, n=200)]
    est = es.estimate_martyushev(pairs, cx, cy, average=False)
    if est:
        pp = math.hypot(est[1] - cx, est[2] - cy)
        print(f"  ecc={ecc:5.0f}px GT=({cx:.1f},{cy:.1f}) -> f={est[0]:.4f} pp_err={pp:.2e} focal_err={abs(est[0]-1000):.2e}")
    else:
        print(f"  ecc={ecc:5.0f}px -> NO feasible solution")

print("\n=== (B) timing (single calls, ecc=120, noise=0.5) ===")
diag = 120.0 / math.sqrt(2.0)
cx, cy = es.W / 2.0 + diag, es.H / 2.0 + diag
rng = np.random.default_rng(1)
pairs4 = [es.sample_pair_correspondences(rng, cx=cx, cy=cy, angle_deg=20.0, noise=0.5, baseline=0.1, n=200) for _ in range(4)]

t0 = time.perf_counter()
es.estimate_martyushev(pairs4[:1], cx, cy, average=False)
print(f"  Martyushev single-pair : {(time.perf_counter()-t0)*1000:7.1f} ms")

t0 = time.perf_counter()
es.estimate_martyushev(pairs4, cx, cy, average=True)
print(f"  Martyushev 4-pair avg  : {(time.perf_counter()-t0)*1000:7.1f} ms")

t0 = time.perf_counter()
es.estimate_profile(pairs4[:1], use_angle=False, num_focal_samples=15, max_nodes=8000, min_box_size_px=1.5)
print(f"  profile f_only (1 pair): {(time.perf_counter()-t0)*1000:7.1f} ms")

t0 = time.perf_counter()
es.estimate_profile(pairs4, use_angle=True, num_focal_samples=15, max_nodes=8000, min_box_size_px=1.5)
print(f"  profile multi_angle(4) : {(time.perf_counter()-t0)*1000:7.1f} ms")
