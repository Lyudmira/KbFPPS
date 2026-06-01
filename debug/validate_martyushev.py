"""Validate the Martyushev solver against the paper's own numerical oracle.

Paper claims (ECCV 2018, Section 4):
  - noise-free median relative K error ~2.5e-9,
  - generically six solutions (real + complex),
  - the feasible solution (real, p>0) is usually unique.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))  # martyushev_solver in papers/KFPPS (one level up)

from martyushev_solver import (
    solve_calibration_from_correspondences,
    solve_calibration_from_fundamental,
    estimate_fundamental_candidates,
    select_feasible_solution,
)

K_GT = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
IMAGE_SIZE = (1280, 720)


def axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return (
        math.cos(angle_rad) * np.eye(3)
        + (1 - math.cos(angle_rad)) * np.outer(axis, axis)
        + math.sin(angle_rad) * skew
    )


def sample_instance(rng, *, noise_px=0.0, num_points=20):
    angle = rng.uniform(math.radians(10), math.radians(30))
    R = axis_angle(rng.normal(size=3), angle)
    t = rng.normal(size=3)
    t = 0.1 * t / np.linalg.norm(t)
    tau = float(np.trace(R))
    pts0, pts1 = [], []
    while len(pts0) < num_points:
        depth = rng.uniform(0.75, 1.25)
        u0, v0 = rng.uniform(0, IMAGE_SIZE[0]), rng.uniform(0, IMAGE_SIZE[1])
        X = np.array([(u0 - 640) / 1000 * depth, (v0 - 360) / 1000 * depth, depth])
        Xc = R @ X + t
        if Xc[2] <= 1e-6:
            continue
        u1 = 1000 * Xc[0] / Xc[2] + 640
        v1 = 1000 * Xc[1] / Xc[2] + 360
        if not (0 <= u1 < IMAGE_SIZE[0] and 0 <= v1 < IMAGE_SIZE[1]):
            continue
        pts0.append([u0, v0])
        pts1.append([u1, v1])
    pts0 = np.array(pts0)
    pts1 = np.array(pts1)
    if noise_px > 0:
        pts0 = pts0 + noise_px * rng.standard_normal(pts0.shape)
        pts1 = pts1 + noise_px * rng.standard_normal(pts1.shape)
    return pts0, pts1, tau


def rel_k_error(K):
    return np.linalg.norm(K - K_GT) / np.linalg.norm(K_GT)


def main():
    rng = np.random.default_rng(0)
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    best_errors = []
    n_feasible = []
    n_total = []
    solve_times = []
    for _ in range(n_trials):
        pts0, pts1, tau = sample_instance(rng, noise_px=0.0)
        t0 = time.perf_counter()
        sols = solve_calibration_from_correspondences(pts0, pts1, tau)
        solve_times.append(time.perf_counter() - t0)
        feasible = [s for s in sols if s.normalized_focal > 0 and np.isfinite(s.residual)]
        n_total.append(len(sols))
        n_feasible.append(len(feasible))
        if feasible:
            best = min(feasible, key=lambda s: rel_k_error(s.K))
            best_errors.append(rel_k_error(best.K))
        else:
            best_errors.append(float("nan"))

    valid = [e for e in best_errors if np.isfinite(e)]
    print(f"trials                : {n_trials}")
    print(f"recovered (any sol)   : {len(valid)}/{n_trials}")
    if valid:
        print(f"median rel K error    : {np.median(valid):.3e}   (paper ~2.5e-9)")
        print(f"max rel K error       : {np.max(valid):.3e}")
    print(f"mean feasible sols     : {np.mean(n_feasible):.2f}  (paper: usually 1)")
    print(f"mean total real sols   : {np.mean(n_total):.2f}")
    print(f"mean solve time        : {np.mean(solve_times)*1000:.1f} ms")


if __name__ == "__main__":
    main()
