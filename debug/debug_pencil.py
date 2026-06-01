"""Debug the hidden-variable p pencil on the verified GT instance."""

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
a_gt, b_gt, p_gt = Kn[0, 2], Kn[1, 2], Kn[0, 0] * Kn[1, 1]
print(f"GT a,b,p = {a_gt:.5f}, {b_gt:.5f}, {p_gt:.5f}")

for degree in [5, 6, 7]:
    matrices, columns, col_index = ms._build_p_pencil(coeff_dicts, degree)
    M0, M1, M2 = matrices[0], matrices[1], matrices[2]
    print(f"\n--- degree {degree}: pencil rows={M0.shape[0]}, cols={M0.shape[1]} (ab monomials) ---")
    # Build GT (a,b) monomial vector
    gt_vec = np.array([a_gt**ia * b_gt**ib for (ia, ib) in columns])
    Mp_gt = M0 + p_gt * M1 + p_gt**2 * M2
    resid = Mp_gt @ gt_vec
    print(f"  ||M(p_gt) @ gt_vec|| = {np.linalg.norm(resid):.3e}  (want ~0)")
    sv = np.linalg.svd(Mp_gt, compute_uv=False)
    print(f"  smallest sv of M(p_gt): {sv[-3:]}")
    # Is the columns count enough? need rows>=cols for QEP compression
    print(f"  rows>=cols: {M0.shape[0] >= M0.shape[1]}")

# Now inspect what eigenvalues the QEP actually returns at degree 7.
print("\n=== QEP eigenvalue inspection (degree 7) ===")
matrices, columns, col_index = ms._build_p_pencil(coeff_dicts, 7)
M0, M1, M2 = matrices[0], matrices[1], matrices[2]
n_rows, n_cols = M0.shape
rng2 = np.random.default_rng(0)
combiner = rng2.standard_normal((n_cols, n_rows))
A0, A1, A2 = combiner @ M0, combiner @ M1, combiner @ M2
print(f"cond(A2)={np.linalg.cond(A2):.3e}, rank(A2)={np.linalg.matrix_rank(A2)}/{n_cols}")
identity = np.eye(n_cols); zero = np.zeros((n_cols, n_cols))
A = np.block([[-A1, -A0], [identity, zero]])
B = np.block([[A2, zero], [zero, identity]])
from scipy.linalg import eig as geig
vals = geig(A, B, right=False)
finite = vals[np.isfinite(vals)]
real_pos = sorted([v.real for v in finite if abs(v.imag) < 1e-4 and v.real > 0])
print(f"total eigs={len(vals)}, finite={len(finite)}, real&pos={len(real_pos)}")
print(f"real positive eigenvalues (want one ~{p_gt:.3f}): {[f'{r:.3f}' for r in real_pos[:15]]}")

# Trace the full _numeric_quotient_solve + acceptance on this instance.
print("\n=== full solve trace ===")
raw = ms._numeric_quotient_solve(coeff_dicts, degree=7)
print(f"raw candidates from QEP: {len(raw)}")
coeff_scale = max((abs(c) for d in coeff_dicts for c in d.values()), default=1.0)
print(f"coeff_scale = {coeff_scale:.3e}")
for (a_r, b_r, p_r) in raw:
    a_p, b_p, p_p = ms._newton_polish(coeff_dicts, a_r, b_r, p_r)
    x = np.array([a_p, b_p, p_p])
    resid = max(abs(ms._eval_poly(d, x)) for d in coeff_dicts)
    tag = ""
    if abs(a_p - a_gt) < 0.05 and abs(b_p - b_gt) < 0.05 and abs(p_p - p_gt) < 1.0:
        tag = "  <-- GT!"
    print(f"  raw p={p_r:8.3f} -> polished a={a_p:8.4f} b={b_p:8.4f} p={p_p:8.4f}  resid={resid:.2e}  accept={resid <= 1e-6*coeff_scale}{tag}")

sols = ms.solve_calibration_from_fundamental(F, tau, S=S)
print(f"\nfinal solutions accepted: {len(sols)}")
for s in sols:
    print(f"  f={s.focal:.2f} cx={s.cx:.2f} cy={s.cy:.2f} resid={s.residual:.2e}")


