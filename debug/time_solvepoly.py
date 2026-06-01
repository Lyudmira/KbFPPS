"""Time sympy solve_poly_system over floating-point coefficients."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))  # martyushev_solver in papers/KFPPS (one level up)

import martyushev_solver as ms

rng = np.random.default_rng(1)


def axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    s = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)
    return math.cos(angle) * np.eye(3) + (1 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * s


K_GT = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
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
pts0, pts1 = np.array(pts0), np.array(pts1)
cands, S = ms.estimate_fundamental_candidates(pts0, pts1)
F = cands[0]
polys = ms._build_polynomials(F, tau)
a, b, p = ms._A, ms._B, ms._P
Kn = S @ K_GT; Kn = Kn / Kn[2, 2]
print(f"GT a,b,p = {Kn[0,2]:.5f}, {Kn[1,2]:.5f}, {Kn[0,0]*Kn[1,1]:.5f}")

# Saturate p numerically: add w*p - 1 = 0 with fresh var w, then solve over floats.
w = sp.Symbol("w")
exprs = [pp.as_expr() for pp in polys] + [w * p - 1]
print("solve_poly_system over floats (saturated, 4 vars)...")
t0 = time.perf_counter()
try:
    sols = sp.solve_poly_system(exprs, a, b, p, w)
    dt = time.perf_counter() - t0
    print(f"  solved in {dt:.2f}s, {len(sols)} solutions")
    for s in sols:
        vals = [complex(v) for v in s]
        if all(abs(v.imag) < 1e-6 for v in vals) and vals[2].real > 0:
            print(f"   REAL p>0: a={vals[0].real:.4f} b={vals[1].real:.4f} p={vals[2].real:.4f}")
except Exception as exc:
    print(f"  FAILED after {time.perf_counter()-t0:.2f}s: {exc}")
