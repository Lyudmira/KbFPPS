"""Profile where a single Martyushev solve actually spends its time."""

from __future__ import annotations

import cProfile
import math
import pstats
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import martyushev_solver as ms


def axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    s = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)
    return math.cos(angle) * np.eye(3) + (1 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * s


def make_instance(rng):
    R = axis_angle(rng.normal(size=3), math.radians(22))
    t = rng.normal(size=3); t = 0.1 * t / np.linalg.norm(t)
    tau = float(np.trace(R))
    pts0, pts1 = [], []
    while len(pts0) < 20:
        d = rng.uniform(0.75, 1.25)
        u0, v0 = rng.uniform(0, 1280), rng.uniform(0, 720)
        X = np.array([(u0 - 640) / 1000 * d, (v0 - 360) / 1000 * d, d])
        Xc = R @ X + t
        if Xc[2] <= 1e-6:
            continue
        u1, v1 = 1000 * Xc[0] / Xc[2] + 640, 1000 * Xc[1] / Xc[2] + 360
        if not (0 <= u1 < 1280 and 0 <= v1 < 720):
            continue
        pts0.append([u0, v0]); pts1.append([u1, v1])
    return np.array(pts0), np.array(pts1), tau


rng = np.random.default_rng(0)
instances = [make_instance(rng) for _ in range(10)]

# Stage-level wall timing.
def time_stage(fn, *a, n=10):
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*a)
    return (time.perf_counter() - t0) / n * 1000.0

p0, p1, tau = instances[0]
cands, S = ms.estimate_fundamental_candidates(p0, p1)
F = cands[0]

print("=== stage wall-clock (ms/call, avg of 10) ===")
print(f"estimate_fundamental_candidates : {time_stage(ms.estimate_fundamental_candidates, p0, p1):8.2f}")
print(f"_build_polynomials              : {time_stage(ms._build_polynomials, F, tau):8.2f}")
polys = ms._build_polynomials(F, tau)
print(f"_poly_coeff_dicts               : {time_stage(ms._poly_coeff_dicts, polys):8.2f}")
cd = ms._poly_coeff_dicts(polys)
print(f"_numeric_quotient_solve         : {time_stage(ms._numeric_quotient_solve, cd):8.2f}")
print(f"FULL solve_from_correspondences : {time_stage(lambda: ms.solve_calibration_from_correspondences(p0, p1, tau)):8.2f}")

print("\n=== cProfile top cumulative (full solve x10) ===")
pr = cProfile.Profile()
pr.enable()
for (a, b, t) in instances:
    ms.solve_calibration_from_correspondences(a, b, t)
pr.disable()
st = pstats.Stats(pr).sort_stats("cumulative")
st.print_stats(15)
