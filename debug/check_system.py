"""Decisive cheap check: do the 4 polynomials vanish at ground truth?

If they do, the polynomial construction is correct and I only need a fast
numeric solver. If not, the bug is in the system itself (F<->E scale, angle
equation sign), which must be fixed first.
"""

from __future__ import annotations

import math
import sys
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
t = rng.normal(size=3)
t = 0.1 * t / np.linalg.norm(t)
tau = float(np.trace(R))

# Noise-free correspondences.
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
    pts0.append([u0, v0])
    pts1.append([u1, v1])
pts0, pts1 = np.array(pts0), np.array(pts1)

cands, S = ms.estimate_fundamental_candidates(pts0, pts1)
F = cands[0]
print(f"F candidates: {len(cands)}")

# Ground truth in the normalized frame: K_norm = S @ K_gt, scaled so [2,2]=1.
K_norm = S @ K_GT
K_norm = K_norm / K_norm[2, 2]
a_gt = K_norm[0, 2]
b_gt = K_norm[1, 2]
p_gt = K_norm[0, 0] * K_norm[1, 1]
print(f"normalized GT: a={a_gt:.5f} b={b_gt:.5f} p={p_gt:.5f} (f_norm={math.sqrt(p_gt):.5f})")

# Sanity: does F_norm reconstruct an essential matrix at GT with trace tau?
E = K_norm.T @ F @ K_norm
u, s, vh = np.linalg.svd(E)
print(f"E singular values (want s1~s2, s3~0): {s}")

# Evaluate the 4 polynomials at GT.
polys = ms._build_polynomials(F, tau)
subs = {ms._A: a_gt, ms._B: b_gt, ms._P: p_gt}
print("polynomial residuals at GT (want ~0):")
for i, poly in enumerate(polys):
    val = float(poly.as_expr().subs(subs))
    print(f"  f{i+1} = {val:.3e}")

# Also check the angle residual helper used for solution selection.
print(f"angle/cubic residual at GT: {ms._angle_residual(F, a_gt, b_gt, p_gt, tau):.3e}")
