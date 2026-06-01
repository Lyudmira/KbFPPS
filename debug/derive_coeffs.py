"""Derive the polynomial coefficients as closed-form functions of (F entries, tau).

Goal: replace the per-instance sympy build with a one-time symbolic derivation,
then emit numpy code. This script just checks feasibility and size.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

a, b, p, tau = sp.symbols("a b p tau", real=True)
F = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"F{i}{j}", real=True))

omega = sp.Matrix([[a*a + p, a*b, a], [a*b, b*b + p, b], [a, b, sp.Integer(1)]])
fof = F * omega * F.T * omega
trace_term = fof.trace()
G = sp.Rational(1, 2) * trace_term * F - fof * F
wf = omega * F
angle = (sp.Rational(1, 2) * (tau*tau - 1) * trace_term
         + (tau + 1) * (wf * wf).trace()
         - tau * (wf.trace())**2)

t0 = time.perf_counter()
exprs = [sp.expand(G[0, 0]), sp.expand(G[1, 1]), sp.expand(G[2, 2]), sp.expand(angle)]
print(f"symbolic expand (one-time): {time.perf_counter()-t0:.2f}s")

# Represent each polynomial as dict {(i,j,k) exponent in (a,b,p): coeff_expr in F,tau}.
fsyms = list(F) + [tau]
total_terms = 0
for idx, e in enumerate(exprs):
    poly = sp.Poly(e, a, b, p)
    n = len(poly.monoms())
    total_terms += n
    # measure coefficient expression complexity
    csize = sum(sp.count_ops(c) for c in poly.coeffs())
    print(f"f{idx+1}: {n} monomials in (a,b,p), total coeff ops = {csize}")
print(f"total (a,b,p) monomials across 4 eqs: {total_terms}")

# Can we lambdify the whole coefficient vector cheaply?
poly0 = sp.Poly(exprs[0], a, b, p)
t0 = time.perf_counter()
fn = sp.lambdify(fsyms, [c for c in poly0.coeffs()], modules="numpy")
print(f"lambdify f1 coeffs: {time.perf_counter()-t0:.2f}s")
import numpy as np
vals = fn(*np.random.rand(10))
print(f"f1 coeff vector len = {len(vals)} (numpy eval works)")
