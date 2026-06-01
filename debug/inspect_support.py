"""Inspect monomial support / degrees of f1..f4 to design the Macaulay matrix.

build_polynomials is fast (only symbolic *solving* was slow), so we use it to
read the exact structure, then design a numeric hidden-variable (in p) solver.
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

rng = np.random.default_rng(5)


def axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    s = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)
    return math.cos(angle) * np.eye(3) + (1 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * s


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
polys = ms._build_polynomials(cands[0], tau)
a, b, p = ms._A, ms._B, ms._P

for i, poly in enumerate(polys):
    monoms = poly.monoms()
    da = max(m[0] for m in monoms); db = max(m[1] for m in monoms); dp = max(m[2] for m in monoms)
    dab = max(m[0] + m[1] for m in monoms)
    print(f"f{i+1}: deg_a={da} deg_b={db} deg_p={dp} deg_ab={dab} #terms={len(monoms)}")

# Monomials in (a,b) appearing (p hidden), union over f1..f4
ab_monoms = set()
for poly in polys:
    for (ia, ib, ip) in poly.monoms():
        ab_monoms.add((ia, ib))
print(f"\n(a,b) monomial support (p hidden), {len(ab_monoms)} monomials:")
print(sorted(ab_monoms))
