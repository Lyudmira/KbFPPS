"""Does the solver work in EuRoC's REGIME (f=458, 752x480, small angle 5-8 deg)?
Isolates whether the failure is solver-vs-small-angle or real-data-specific.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import martyushev_solver as ms

K_GT = np.array([[458.654, 0, 367.215], [0, 457.296, 248.375], [0, 0, 1.0]])
W, H = 752, 480


def axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis); x, y, z = axis
    s = np.array([[0,-z,y],[z,0,-x],[-y,x,0]], float)
    return math.cos(angle)*np.eye(3)+(1-math.cos(angle))*np.outer(axis,axis)+math.sin(angle)*s


def trial(rng, angle_deg, noise, baseline, n=200):
    R = axis_angle(rng.normal(size=3), math.radians(angle_deg))
    t = rng.normal(size=3); t = baseline * t/np.linalg.norm(t)
    tau = float(np.trace(R))
    f, cx, cy = 458.654, 367.215, 248.375
    pts0, pts1 = [], []
    tries = 0
    while len(pts0) < n and tries < n*20:
        tries += 1
        d = rng.uniform(3.0, 8.0)  # EuRoC machine-hall depths ~meters
        u0, v0 = rng.uniform(0,W), rng.uniform(0,H)
        X = np.array([(u0-cx)/f*d, (v0-cy)/f*d, d])
        Xc = R@X + t
        if Xc[2] <= 1e-6: continue
        u1, v1 = f*Xc[0]/Xc[2]+cx, f*Xc[1]/Xc[2]+cy
        if not (0<=u1<W and 0<=v1<H): continue
        pts0.append([u0,v0]); pts1.append([u1,v1])
    pts0, pts1 = np.array(pts0), np.array(pts1)
    if noise>0:
        pts0 = pts0 + noise*rng.standard_normal(pts0.shape)
        pts1 = pts1 + noise*rng.standard_normal(pts1.shape)
    sols = ms.solve_calibration_from_correspondences(pts0, pts1, tau)
    best = ms.select_feasible_solution(sols, image_size=(W,H), center_radius_px=80)
    if best is None:
        return None
    return math.hypot(best.cx-cx, best.cy-cy), abs(best.focal-f)


for angle in [6.0, 10.0, 20.0]:
    for noise in [0.0, 0.5]:
        rng = np.random.default_rng(0)
        errs = [trial(rng, angle, noise, baseline=0.3) for _ in range(10)]
        ok = [e for e in errs if e is not None]
        if ok:
            pp = np.median([e[0] for e in ok]); fe = np.median([e[1] for e in ok])
            print(f"angle={angle:4.0f} noise={noise}: {len(ok)}/10 feasible, median pp_err={pp:.2f}px focal_err={fe:.2f}px")
        else:
            print(f"angle={angle:4.0f} noise={noise}: 0/10 feasible")
