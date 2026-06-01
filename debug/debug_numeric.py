"""Debug the numeric quotient solve stage by stage on a verified GT instance."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

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
coeff_dicts = ms._poly_coeff_dicts(polys)

Kn = S @ K_GT; Kn = Kn / Kn[2, 2]
gt = (Kn[0, 2], Kn[1, 2], Kn[0, 0] * Kn[1, 1])
print(f"GT (a,b,p) = {gt}")

for deg in [4, 5, 6, 7]:
    R_mac, cols, cidx = ms._build_macaulay(coeff_dicts, deg)
    sv = np.linalg.svd(R_mac, compute_uv=False)
    n_cols = len(cols)
    # count near-zero singular values
    tol = 1e-7 * max(sv.max(), 1.0)
    nnull = n_cols - np.sum(sv > tol) if len(sv) >= 1 else n_cols
    # also full rank deficiency = cols - rank
    rank = int(np.sum(sv > tol))
    print(f"deg={deg}: Macaulay {R_mac.shape}, #cols={n_cols}, rank={rank}, nullity={n_cols-rank}, smallest sv tail={sv[-8:] if len(sv)>=8 else sv}")
