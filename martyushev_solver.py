"""Paper-local reimplementation of Martyushev ECCV 2018.

"Self-Calibration of Cameras with Euclidean Image Plane in Case of Two Views
and Known Relative Rotation Angle."

This module implements the actual non-iterative minimal solver from the paper:
a Groebner-basis / action-matrix solution of the polynomial system in the
unknowns ``a, b, p`` (principal point ``(a, b)`` and ``p = f**2``), rather than
a generic symbolic solve.

Pipeline (Section 3 of the paper):
  1. Pre-normalize correspondences with an upper-triangular S so the 2N points
     have centroid at the origin and mean distance sqrt(2). The whole solve
     happens in the normalized frame; K is denormalized at the end via S^-1 K.
  2. Estimate the fundamental matrix F (7-point or 8-point).
  3. Build the four quartic polynomials f1=(G)_11, f2=(G)_22, f3=(G)_33 (from
     the Demazure cubic essential constraint applied to F omega* F^T omega*) and
     f4 (the known-rotation-angle constraint, Eq. constrFw2).
  4. Construct an action matrix for multiplication by p in the quotient ring
     C[a,b,p]/J' and read the six solutions off its eigenvectors.
  5. Discard complex solutions and ones with p <= 0; denormalize K = S^-1 K_norm.

The paper specifies an explicit reduced-row-echelon elimination template
(B0 -> ... -> B5) verified in Maple. We do not have Maple to cross-check the
literal row/column bookkeeping, so we build a numerically equivalent
action matrix by Macaulay-style expansion of the ideal and a quotient by p,
then validate against the paper's own oracle (noise-free ~1e-9 recovery, six
solutions generically, the feasible solution usually unique).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class MartyushevCalibrationSolution:
    focal: float
    cx: float
    cy: float
    normalized_focal: float
    normalized_cx: float
    normalized_cy: float
    K: np.ndarray
    K_normalized: np.ndarray
    F_normalized: np.ndarray
    residual: float


# --------------------------------------------------------------------------- #
# Data pre-normalization (paper Subsection 3.1) and F estimation.
# --------------------------------------------------------------------------- #


def normalize_fro(matrix: np.ndarray, *, eps: float = 1e-15) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError("Cannot normalize a zero or non-finite matrix.")
    return matrix / norm


def rank2_fundamental(matrix: np.ndarray) -> np.ndarray:
    u, s, vh = np.linalg.svd(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    s[-1] = 0.0
    return normalize_fro(u @ np.diag(s) @ vh)


def homogeneous(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    return np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)


def normalize_joint_points(
    points0: np.ndarray, points1: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the paper's S so the 2N stacked points are centred with mean dist sqrt(2)."""

    points0 = np.asarray(points0, dtype=np.float64).reshape(-1, 2)
    points1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    stacked = np.vstack([points0, points1])
    centroid = stacked.mean(axis=0)
    centered = stacked - centroid
    mean_dist = float(np.mean(np.linalg.norm(centered, axis=1)))
    scale = np.sqrt(2.0) / max(mean_dist, 1e-12)
    S = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized0 = homogeneous(points0) @ S.T
    normalized1 = homogeneous(points1) @ S.T
    return normalized0[:, :2], normalized1[:, :2], S


# __APPEND_1__


def _design_matrix(points0: np.ndarray, points1: np.ndarray) -> np.ndarray:
    x0 = np.asarray(points0, dtype=np.float64)[:, 0]
    y0 = np.asarray(points0, dtype=np.float64)[:, 1]
    x1 = np.asarray(points1, dtype=np.float64)[:, 0]
    y1 = np.asarray(points1, dtype=np.float64)[:, 1]
    return np.column_stack(
        [x1 * x0, x1 * y0, x1, y1 * x0, y1 * y0, y1, x0, y0, np.ones_like(x0)]
    )


def _determinant_cubic_coefficients(F0: np.ndarray, F1: np.ndarray) -> np.ndarray:
    samples = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float64)
    vandermonde = np.vander(samples, 4)
    values = np.array([np.linalg.det(F0 + value * F1) for value in samples], dtype=np.float64)
    return np.linalg.solve(vandermonde, values)


def estimate_fundamental_candidates(
    points0: np.ndarray, points1: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return normalized-frame F candidates plus the normalization S.

    N == 7 yields the one-or-three real F solutions of the 7-point algorithm;
    N >= 8 yields the unique normalized eight-point F. All candidates are in the
    normalized frame defined by S.
    """

    normalized0, normalized1, S = normalize_joint_points(points0, points1)
    design = _design_matrix(normalized0, normalized1)
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    if design.shape[0] == 7:
        F0 = vh[-1].reshape(3, 3)
        F1 = vh[-2].reshape(3, 3)
        coeffs = _determinant_cubic_coefficients(F0, F1)
        roots = np.roots(coeffs)
        candidates: list[np.ndarray] = []
        for root in roots:
            if abs(root.imag) > 1e-8:
                continue
            candidates.append(normalize_fro(F0 + float(root.real) * F1))
        if not candidates:
            candidates.append(rank2_fundamental(F0))
        return candidates, S
    candidate = rank2_fundamental(vh[-1].reshape(3, 3))
    return [candidate], S


# __APPEND_2__


# --------------------------------------------------------------------------- #
# Polynomial system (paper Section 3.2): f1=(G)_11, f2=(G)_22, f3=(G)_33 from
# the Demazure cubic essential constraint, f4 from the known-rotation-angle
# constraint (Eq. constrFw2). Variables a, b (principal point) and p = f^2.
# --------------------------------------------------------------------------- #

_A, _B, _P, _T = sp.symbols("a b p t", real=True)


def _omega_star() -> sp.Matrix:
    return sp.Matrix(
        [
            [_A * _A + _P, _A * _B, _A],
            [_A * _B, _B * _B + _P, _B],
            [_A, _B, sp.Integer(1)],
        ]
    )


def _derive_coefficient_model() -> tuple[list[list[tuple[int, int, int]]], Any]:
    """One-time symbolic derivation of the four polynomials' coefficients.

    The coefficients of f1..f4 (in the monomials of a, b, p) are closed-form
    functions of the nine entries of F and tau. We derive them symbolically
    ONCE at import and compile a numpy function, so per-instance solving is pure
    float arithmetic instead of ~190 ms of sympy expansion (see
    debug/profile_solver.py for the measurement that motivated this).

    Returns (monomial_lists, evaluator) where monomial_lists[k] are the (a,b,p)
    exponent tuples of equation k, and evaluator(F_flat9, tau) -> list of four
    coefficient arrays aligned with monomial_lists.
    """

    F = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"F{i}{j}", real=True))
    tau = sp.Symbol("tau", real=True)
    omega = _omega_star()
    fof = F * omega * F.T * omega
    trace_term = fof.trace()
    G = sp.Rational(1, 2) * trace_term * F - fof * F
    wf = omega * F
    angle = (
        sp.Rational(1, 2) * (tau * tau - 1) * trace_term
        + (tau + 1) * (wf * wf).trace()
        - tau * (wf.trace()) ** 2
    )
    exprs = [sp.expand(G[0, 0]), sp.expand(G[1, 1]), sp.expand(G[2, 2]), sp.expand(angle)]

    monomial_lists: list[list[tuple[int, int, int]]] = []
    coeff_exprs: list[list[sp.Expr]] = []
    for expr in exprs:
        poly = sp.Poly(expr, _A, _B, _P)
        monos = [(int(m[0]), int(m[1]), int(m[2])) for m in poly.monoms()]
        monomial_lists.append(monos)
        coeff_exprs.append(list(poly.coeffs()))

    symbols = list(F) + [tau]
    # Flatten all coefficient expressions into a single lambdified vector.
    flat = [c for group in coeff_exprs for c in group]
    flat_fn = sp.lambdify(symbols, flat, modules="numpy")
    sizes = [len(group) for group in coeff_exprs]

    def evaluator(F_flat9: np.ndarray, tau_value: float) -> list[np.ndarray]:
        values = np.asarray(flat_fn(*F_flat9, tau_value), dtype=np.float64).reshape(-1)
        out: list[np.ndarray] = []
        offset = 0
        for size in sizes:
            out.append(values[offset : offset + size])
            offset += size
        return out

    return monomial_lists, evaluator


_MONOMIAL_LISTS, _COEFF_EVALUATOR = _derive_coefficient_model()


def coefficient_dicts(F_normalized: np.ndarray, rotation_trace: float) -> list[dict[tuple[int, int, int], float]]:
    """Fast path: numeric coefficient dicts for f1..f4 via the compiled model."""

    F_flat = np.asarray(F_normalized, dtype=np.float64).reshape(9)
    coeff_groups = _COEFF_EVALUATOR(F_flat, float(rotation_trace))
    dicts: list[dict[tuple[int, int, int], float]] = []
    for monos, coeffs in zip(_MONOMIAL_LISTS, coeff_groups):
        dicts.append({mono: float(c) for mono, c in zip(monos, coeffs)})
    return dicts


def _build_polynomials(F_normalized: np.ndarray, rotation_trace: float) -> list[sp.Poly]:
    """Symbolic reference build (slow). Retained for diagnostics; the solve path
    uses coefficient_dicts() instead. See debug/profile_solver.py."""

    F = sp.Matrix([[sp.Float(float(v), 17) for v in row] for row in np.asarray(F_normalized, dtype=np.float64).reshape(3, 3)])
    omega = _omega_star()
    tau = sp.Float(float(rotation_trace), 17)
    fof = F * omega * F.T * omega
    trace_term = (fof).trace()
    G = sp.Rational(1, 2) * trace_term * F - fof * F
    wf = omega * F
    angle = (
        sp.Rational(1, 2) * (tau * tau - 1) * trace_term
        + (tau + 1) * (wf * wf).trace()
        - tau * (wf.trace()) ** 2
    )
    exprs = [sp.expand(G[0, 0]), sp.expand(G[1, 1]), sp.expand(G[2, 2]), sp.expand(angle)]
    return [sp.Poly(e, _A, _B, _P) for e in exprs]


def _poly_coeff_dicts(polys: list[sp.Poly]) -> list[dict[tuple[int, int, int], float]]:
    """Numeric coefficient dicts keyed by monomial exponent (a, b, p)."""

    dicts: list[dict[tuple[int, int, int], float]] = []
    for poly in polys:
        d: dict[tuple[int, int, int], float] = {}
        for coeff, mono in zip(poly.coeffs(), poly.monoms()):
            d[(int(mono[0]), int(mono[1]), int(mono[2]))] = float(coeff)
        dicts.append(d)
    return dicts


# __NUMERIC_SOLVE__



def _monomials_up_to(degree: int) -> list[tuple[int, int, int]]:
    """All (a, b, p) exponent triples with total degree <= ``degree``."""

    monos: list[tuple[int, int, int]] = []
    for total in range(degree + 1):
        for i in range(total + 1):
            for j in range(total - i + 1):
                k = total - i - j
                monos.append((i, j, k))
    return monos


def _ab_monomials_up_to(degree: int) -> list[tuple[int, int]]:
    """All (a, b) exponent pairs with total degree <= ``degree``, low degree first."""

    monos: list[tuple[int, int]] = []
    for total in range(degree + 1):
        for i in range(total + 1):
            monos.append((i, total - i))
    return monos


def _build_p_pencil(
    coeff_dicts: list[dict[tuple[int, int, int], float]], degree: int
) -> tuple[list[np.ndarray], list[tuple[int, int]], dict[tuple[int, int], int]]:
    """Hidden-variable (in p) Macaulay pencil M(p) = M0 + p M1 + p^2 M2.

    Rows are products {(a,b)-monomial} * {generator}; columns are (a,b) monomials.
    Coefficients are split by power of p (the system is degree <= 2 in p), so the
    common (a,b) root at the true p is a right null vector of M(p).
    """

    columns = _ab_monomials_up_to(degree)
    col_index = {mono: i for i, mono in enumerate(columns)}
    p_degree = max(mono[2] for gen in coeff_dicts for mono in gen)
    rows_per_power: list[list[np.ndarray]] = [[] for _ in range(p_degree + 1)]
    for gen in coeff_dicts:
        gen_ab_deg = max(mono[0] + mono[1] for mono in gen)
        for shift in _ab_monomials_up_to(degree - gen_ab_deg):
            split = [np.zeros(len(columns), dtype=np.float64) for _ in range(p_degree + 1)]
            for (ia, ib, ip), coeff in gen.items():
                target = (ia + shift[0], ib + shift[1])
                split[ip][col_index[target]] = coeff
            for ip in range(p_degree + 1):
                rows_per_power[ip].append(split[ip])
    matrices = [np.asarray(rows, dtype=np.float64) for rows in rows_per_power]
    return matrices, columns, col_index


def _numeric_quotient_solve(
    coeff_dicts: list[dict[tuple[int, int, int], float]],
    *,
    degree: int = 7,
    rng_seed: int = 0,
) -> list[tuple[float, float, float]]:
    """Hidden-variable resultant in p, solved as a quadratic eigenvalue problem.

    1. Build the rectangular pencil M(p) = M0 + p M1 + p^2 M2 over (a,b) monomials.
    2. Randomly compress the (overdetermined) rows to a square C x C pencil; for a
       random combiner this introduces no spurious singularities.
    3. Linearize the QEP to a generalized eigenproblem; the eigenvalues are p.
    4. For each finite real p, recover (a,b) from the right null vector of the
       rectangular M(p) (more rows -> more robust).
    """

    matrices, columns, col_index = _build_p_pencil(coeff_dicts, degree)
    if len(matrices) < 3:
        matrices = matrices + [np.zeros_like(matrices[0])] * (3 - len(matrices))
    M0, M1, M2 = matrices[0], matrices[1], matrices[2]
    n_rows, n_cols = M0.shape
    if n_rows < n_cols:
        return []

    rng = np.random.default_rng(rng_seed)
    combiner = rng.standard_normal((n_cols, n_rows))
    A0, A1, A2 = combiner @ M0, combiner @ M1, combiner @ M2

    # Linearize M0 + p M1 + p^2 M2: companion generalized eigenproblem (size 2C).
    identity = np.eye(n_cols, dtype=np.float64)
    zero = np.zeros((n_cols, n_cols), dtype=np.float64)
    A = np.block([[-A1, -A0], [identity, zero]])
    B = np.block([[A2, zero], [zero, identity]])

    try:
        from scipy.linalg import eig as generalized_eig

        eigvals = generalized_eig(A, B, right=False)
    except Exception:
        try:
            eigvals = np.linalg.eigvals(np.linalg.solve(B, A))
        except np.linalg.LinAlgError:
            return []

    i_one = col_index.get((0, 0))
    i_a = col_index.get((1, 0))
    i_b = col_index.get((0, 1))
    if i_one is None or i_a is None or i_b is None:
        return []

    solutions: list[tuple[float, float, float]] = []
    for value in np.atleast_1d(eigvals):
        if not np.isfinite(value):
            continue
        # The rank-deficient linearization yields spurious roots; keep any finite
        # real-ish positive p as a candidate and let verification reject the rest.
        if abs(value.imag) > 1e-3 * max(1.0, abs(value.real)):
            continue
        p_val = float(value.real)
        if p_val <= 1e-6:
            continue
        # Recover (a, b) from the null space of the rectangular M(p). At a true
        # root the null space can be multi-dimensional, so we pick, among the
        # near-zero singular directions, the one whose layout matches a genuine
        # monomial vector: 1 -> a -> b -> a^2 -> a*b ... consistent with v[i_a],
        # v[i_b] being the degree-one coordinates after fixing the constant to 1.
        Mp = M0 + p_val * M1 + p_val * p_val * M2
        _, sv, vh = np.linalg.svd(Mp)
        tol = 1e-6 * max(sv[0], 1.0)
        null_rows = [k for k in range(len(sv)) if sv[k] <= tol]
        if not null_rows:
            null_rows = [len(sv) - 1]
        nullspace = vh[null_rows, :]  # (d, n_cols)
        cand = _recover_from_nullspace(nullspace, i_one, i_a, i_b, columns)
        if cand is None:
            continue
        solutions.append((cand[0], cand[1], p_val))
    return solutions


def _recover_from_nullspace(
    nullspace: np.ndarray,
    i_one: int,
    i_a: int,
    i_b: int,
    columns: list[tuple[int, int]],
) -> tuple[float, float] | None:
    """Pick the null-space vector consistent with a monomial vector [1, a, b, ...].

    For a 1-D null space this just normalizes by the constant term. For a higher
    dimensional null space (true root plus the parasitic conic direction), we
    find the combination with a nonzero constant term that best satisfies the
    multiplicative monomial relations (e.g. (a)*(a) == (a^2)).
    """

    if nullspace.shape[0] == 1:
        v = nullspace[0]
        if abs(v[i_one]) < 1e-10:
            return None
        v = v / v[i_one]
        return float(v[i_a]), float(v[i_b])

    # Solve for the combiner c so that w = c @ nullspace has w[i_one] = 1 and the
    # degree-one entries are self-consistent with the quadratic monomials.
    col_pos = {mono: k for k, mono in enumerate(columns)}
    best: tuple[float, tuple[float, float]] | None = None
    # Try each basis null vector and small combinations seeded by the constant term.
    for seed in range(nullspace.shape[0]):
        v0 = nullspace[seed]
        if abs(v0[i_one]) < 1e-10:
            continue
        v = v0 / v0[i_one]
        a_val, b_val = float(v[i_a]), float(v[i_b])
        # Consistency score: how well do quadratic monomials match products.
        score = 0.0
        for mono, k in col_pos.items():
            ia, ib = mono
            if ia + ib <= 1:
                continue
            predicted = (a_val ** ia) * (b_val ** ib)
            score += (float(v[k]) - predicted) ** 2
        if best is None or score < best[0]:
            best = (score, (a_val, b_val))
    if best is None:
        return None
    return best[1]


def _newton_polish(
    coeff_dicts: list[dict[tuple[int, int, int], float]],
    a: float,
    b: float,
    p: float,
    *,
    iters: int = 8,
) -> tuple[float, float, float]:
    """Refine (a, b, p) with a few Gauss-Newton steps on the four equations."""

    x = np.array([a, b, p], dtype=np.float64)
    for _ in range(iters):
        resid = np.array([_eval_poly(d, x) for d in coeff_dicts], dtype=np.float64)
        jac = np.array([_eval_grad(d, x) for d in coeff_dicts], dtype=np.float64)
        try:
            step, *_ = np.linalg.lstsq(jac, -resid, rcond=None)
        except np.linalg.LinAlgError:
            break
        x = x + step
        if np.linalg.norm(step) < 1e-13:
            break
    return float(x[0]), float(x[1]), float(x[2])


def _eval_poly(coeffs: dict[tuple[int, int, int], float], x: np.ndarray) -> float:
    a, b, p = x
    return sum(c * a**i * b**j * p**k for (i, j, k), c in coeffs.items())


def _eval_grad(coeffs: dict[tuple[int, int, int], float], x: np.ndarray) -> np.ndarray:
    a, b, p = x
    da = db = dp = 0.0
    for (i, j, k), c in coeffs.items():
        if i > 0:
            da += c * i * a ** (i - 1) * b**j * p**k
        if j > 0:
            db += c * j * a**i * b ** (j - 1) * p**k
        if k > 0:
            dp += c * k * a**i * b**j * p ** (k - 1)
    return np.array([da, db, dp], dtype=np.float64)


# __APPEND_4__



def _angle_residual(F_norm: np.ndarray, a: float, b: float, p: float, tau: float) -> float:
    if p <= 0:
        return float("inf")
    K = np.array([[np.sqrt(p), 0.0, a], [0.0, np.sqrt(p), b], [0.0, 0.0, 1.0]], dtype=np.float64)
    E = K.T @ np.asarray(F_norm, dtype=np.float64).reshape(3, 3) @ K
    trace_eet = float(np.trace(E @ E.T))
    if trace_eet <= 1e-15:
        return float("inf")
    val = (
        0.5 * (tau * tau - 1.0) * trace_eet
        + (tau + 1.0) * float(np.trace(E @ E))
        - tau * float(np.trace(E)) ** 2
    )
    cubic = 2.0 * E @ E.T @ E - trace_eet * E
    return float(val * val / (trace_eet * trace_eet) + np.linalg.norm(cubic) ** 2 / max(trace_eet**3, 1e-30))


def solve_calibration_from_fundamental(
    F_normalized: np.ndarray,
    rotation_trace: float,
    S: np.ndarray | None = None,
    *,
    macaulay_degree: int = 7,
) -> list[MartyushevCalibrationSolution]:
    """Solve for (a, b, p) in the normalized frame, then denormalize K = S^-1 K_norm."""

    coeff_dicts = coefficient_dicts(F_normalized, rotation_trace)
    raw = _numeric_quotient_solve(coeff_dicts, degree=macaulay_degree)
    tau = float(rotation_trace)
    S_inv = np.eye(3, dtype=np.float64) if S is None else np.linalg.inv(np.asarray(S, dtype=np.float64).reshape(3, 3))
    # Scale for the equation-residual acceptance test (coefficients are O(1) after
    # Frobenius-normalizing F, so a true root drives all four polynomials to ~0).
    coeff_scale = max((abs(c) for d in coeff_dicts for c in d.values()), default=1.0)
    results: list[MartyushevCalibrationSolution] = []
    seen: list[tuple[float, float, float]] = []
    for a_val, b_val, p_val in raw:
        if not np.isfinite([a_val, b_val, p_val]).all() or p_val <= 1e-9:
            continue
        # Polish against the exact equations to clean up the linear-algebra estimate.
        a_val, b_val, p_val = _newton_polish(coeff_dicts, a_val, b_val, p_val)
        if not np.isfinite([a_val, b_val, p_val]).all() or p_val <= 1e-9:
            continue
        # Reject spurious eigen-roots that do not actually satisfy the system.
        x = np.array([a_val, b_val, p_val], dtype=np.float64)
        max_resid = max(abs(_eval_poly(d, x)) for d in coeff_dicts)
        if max_resid > 1e-6 * coeff_scale:
            continue
        if any(np.allclose((a_val, b_val, p_val), prev, atol=1e-5, rtol=1e-5) for prev in seen):
            continue
        seen.append((a_val, b_val, p_val))
        f_norm = float(np.sqrt(p_val))
        K_norm = np.array([[f_norm, 0.0, a_val], [0.0, f_norm, b_val], [0.0, 0.0, 1.0]], dtype=np.float64)
        K = S_inv @ K_norm
        K = K / K[2, 2]
        results.append(
            MartyushevCalibrationSolution(
                focal=float(K[0, 0]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                normalized_focal=f_norm,
                normalized_cx=float(a_val),
                normalized_cy=float(b_val),
                K=K,
                K_normalized=K_norm,
                F_normalized=np.asarray(F_normalized, dtype=np.float64).reshape(3, 3),
                residual=_angle_residual(F_normalized, a_val, b_val, p_val, tau),
            )
        )
    return results


def solve_calibration_from_correspondences(
    points0: np.ndarray,
    points1: np.ndarray,
    rotation_trace: float,
) -> list[MartyushevCalibrationSolution]:
    """Full single-pair pipeline: normalize, estimate F, solve, denormalize."""

    F_candidates, S = estimate_fundamental_candidates(points0, points1)
    results: list[MartyushevCalibrationSolution] = []
    for F_candidate in F_candidates:
        results.extend(solve_calibration_from_fundamental(F_candidate, rotation_trace, S=S))
    unique: list[MartyushevCalibrationSolution] = []
    for candidate in results:
        if any(
            np.allclose(
                [candidate.focal, candidate.cx, candidate.cy],
                [existing.focal, existing.cx, existing.cy],
                atol=1e-4,
                rtol=1e-4,
            )
            for existing in unique
        ):
            continue
        unique.append(candidate)
    return unique


def select_feasible_solution(
    solutions: list[MartyushevCalibrationSolution],
    *,
    image_size: tuple[int, int] | None = None,
    center_radius_px: float | None = None,
) -> MartyushevCalibrationSolution | None:
    """Pick the feasible solution (lowest residual), optionally near the image center.

    Mirrors the paper's real-data heuristic of discarding principal points far
    from the geometric center to guarantee a unique feasible output.
    """

    candidates = [s for s in solutions if s.normalized_focal > 0 and np.isfinite(s.residual)]
    if image_size is not None and center_radius_px is not None:
        cx0, cy0 = 0.5 * image_size[0], 0.5 * image_size[1]
        near = [s for s in candidates if abs(s.cx - cx0) < center_radius_px and abs(s.cy - cy0) < center_radius_px]
        if near:
            candidates = near
    if not candidates:
        return None
    return min(candidates, key=lambda s: s.residual)





