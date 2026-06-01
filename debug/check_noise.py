"""Quick noise behavior check for the Martyushev solver + averaging."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# This script lives in papers/KFPPS/debug; the solver kernel is one level up.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))            # sibling debug modules (validate_martyushev)
sys.path.insert(0, str(ROOT.parent))     # martyushev_solver in papers/KFPPS

from martyushev_solver import solve_calibration_from_correspondences, select_feasible_solution
from validate_martyushev import sample_instance, K_GT, IMAGE_SIZE


def run(noise, n_trials, n_pairs):
    rng = np.random.default_rng(7)
    errs_single = []
    errs_avg = []
    for _ in range(n_trials):
        per_pair = []
        for _ in range(n_pairs):
            p0, p1, tau = sample_instance(rng, noise_px=noise)
            sols = solve_calibration_from_correspondences(p0, p1, tau)
            best = select_feasible_solution(sols, image_size=IMAGE_SIZE, center_radius_px=400)
            if best is not None:
                per_pair.append((best.cx, best.cy, best.focal))
        if not per_pair:
            continue
        per_pair = np.array(per_pair)
        errs_single.append(math.hypot(per_pair[0, 0] - 640, per_pair[0, 1] - 360))
        avg = per_pair.mean(axis=0)
        errs_avg.append(math.hypot(avg[0] - 640, avg[1] - 360))
    return errs_single, errs_avg


for noise in [0.0, 0.5, 1.0]:
    s, a = run(noise, 10, 4)
    print(f"noise={noise}: single pp err median={np.median(s):.2f}px  4-pair-avg pp err median={np.median(a):.2f}px  (n={len(s)})")
