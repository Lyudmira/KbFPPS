from __future__ import annotations

import numpy as np
import sympy as sp

from .geometry import householder_orthobasis_perp, left_epipole_from_fundamental, normalize_fundamental, skew
from .polynomial import coefficient_tensor_from_expr
from .types import FundamentalMatrixObservation, PairDiagnostics


def _validate_focal_lengths(fx: float, fy: float) -> tuple[float, float]:
    fx = float(fx)
    fy = float(fy)
    if not np.isfinite(fx) or not np.isfinite(fy) or fx == 0.0 or fy == 0.0:
        raise ValueError("Known focal lengths must be finite and non-zero.")
    return fx, fy


def _diac_matrix_expr(
    *,
    fx: float,
    fy: float,
    cx_symbol: sp.Symbol,
    cy_symbol: sp.Symbol,
) -> sp.Matrix:
    return sp.Matrix(
        [
            [fx * fx + cx_symbol * cx_symbol, cx_symbol * cy_symbol, cx_symbol],
            [cx_symbol * cy_symbol, fy * fy + cy_symbol * cy_symbol, cy_symbol],
            [cx_symbol, cy_symbol, 1.0],
        ]
    )


def _calibration_matrix_expr(
    *,
    fx: float,
    fy: float,
    cx_symbol: sp.Symbol,
    cy_symbol: sp.Symbol,
) -> sp.Matrix:
    return sp.Matrix(
        [
            [fx, 0.0, cx_symbol],
            [0.0, fy, cy_symbol],
            [0.0, 0.0, 1.0],
        ]
    )


def _known_rotation_term_polynomial(
    observation: FundamentalMatrixObservation,
    *,
    fx: float,
    fy: float,
    cx_symbol: sp.Symbol,
    cy_symbol: sp.Symbol,
) -> tuple[sp.Expr, float, bool]:
    rotation_trace = observation.resolved_rotation_trace()
    if rotation_trace is None:
        return sp.Float(0.0), 0.0, False
    rotation_weight = float(observation.known_rotation_weight)
    if not np.isfinite(rotation_weight) or rotation_weight < 0.0:
        raise ValueError("known_rotation_weight must be finite and non-negative.")
    if rotation_weight == 0.0:
        return sp.Float(0.0), 0.0, False
    F = sp.Matrix(observation.normalized_matrix().tolist())
    K = _calibration_matrix_expr(
        fx=fx,
        fy=fy,
        cx_symbol=cx_symbol,
        cy_symbol=cy_symbol,
    )
    E = (K.T * F * K).applyfunc(sp.expand)
    tau = sp.Float(rotation_trace)
    trace_eet = sp.expand((E * E.T).trace())
    trace_e2 = sp.expand((E * E).trace())
    trace_e = sp.expand(E.trace())
    residual = sp.expand(
        sp.Rational(1, 2) * (tau * tau - 1.0) * trace_eet
        + (tau + 1.0) * trace_e2
        - tau * trace_e * trace_e
    )
    return residual, float(observation.weight) * rotation_weight, True


def _support_discriminant(F: np.ndarray, basis: np.ndarray, epipole: np.ndarray) -> float:
    x_symbol, y_symbol = sp.symbols("x_inf y_inf", real=True)
    direction = sp.Matrix([x_symbol, y_symbol, 0.0])
    basis_expr = sp.Matrix(basis)
    f_expr = sp.Matrix(F)
    skew_epipole = sp.Matrix(skew(epipole))
    delta = sp.expand(
        sp.Matrix.hstack(
            basis_expr.T * (f_expr * direction),
            basis_expr.T * (skew_epipole * direction),
        ).det()
    )
    dehomogenized = sp.expand(delta.subs(y_symbol, 1.0))
    poly = sp.Poly(dehomogenized, x_symbol)
    if poly.is_zero or poly.degree() < 2:
        return 0.0
    return float(sp.N(sp.discriminant(poly.as_expr(), x_symbol)))


def _proportionality_minors(left_term: sp.Matrix, right_term: sp.Matrix) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    minor_01 = sp.expand(left_term[0, 0] * right_term[0, 1] - left_term[0, 1] * right_term[0, 0])
    minor_02 = sp.expand(left_term[0, 0] * right_term[1, 1] - left_term[1, 1] * right_term[0, 0])
    minor_12 = sp.expand(left_term[0, 1] * right_term[1, 1] - left_term[1, 1] * right_term[0, 1])
    return minor_01, minor_02, minor_12


def kruppa_pair_residual_polynomials(
    observation: FundamentalMatrixObservation,
    *,
    fx: float,
    fy: float,
    cx_symbol: sp.Symbol,
    cy_symbol: sp.Symbol,
    rank_threshold: float = 1e-8,
    support_discriminant_threshold: float = 1e-8,
) -> tuple[tuple[sp.Expr, ...], PairDiagnostics]:
    fx, fy = _validate_focal_lengths(fx, fy)
    restricted_matrices, diagnostics = kruppa_restricted_matrices(
        observation,
        fx=fx,
        fy=fy,
        cx_symbol=cx_symbol,
        cy_symbol=cy_symbol,
        rank_threshold=rank_threshold,
        support_discriminant_threshold=support_discriminant_threshold,
    )
    if not diagnostics.valid:
        return (sp.Float(0.0),), diagnostics
    left_term, right_term = restricted_matrices
    return _proportionality_minors(left_term, right_term), diagnostics


def kruppa_restricted_matrices(
    observation: FundamentalMatrixObservation,
    *,
    fx: float,
    fy: float,
    cx_symbol: sp.Symbol,
    cy_symbol: sp.Symbol,
    rank_threshold: float = 1e-8,
    support_discriminant_threshold: float = 1e-8,
) -> tuple[tuple[sp.Matrix, sp.Matrix], PairDiagnostics]:
    fx, fy = _validate_focal_lengths(fx, fy)
    try:
        F = normalize_fundamental(observation.normalized_matrix())
    except ValueError:
        diagnostics = PairDiagnostics(
            label=observation.label,
            valid=False,
            reason="rank_deficient_fundamental",
            original_weight=float(observation.weight),
            effective_weight=0.0,
            kruppa_valid=False,
        )
        zero = sp.zeros(2, 2)
        return (zero, zero), diagnostics
    singular_values = np.linalg.svd(F, compute_uv=False)
    epipole = np.zeros(3, dtype=np.float64)
    if singular_values[1] <= rank_threshold:
        diagnostics = PairDiagnostics(
            label=observation.label,
            valid=False,
            reason="rank_deficient_fundamental",
            original_weight=float(observation.weight),
            effective_weight=0.0,
            kruppa_valid=False,
            left_epipole=epipole,
            singular_values=singular_values,
        )
        zero = sp.zeros(2, 2)
        return (zero, zero), diagnostics

    epipole = left_epipole_from_fundamental(F)
    basis = householder_orthobasis_perp(epipole)
    support_disc = _support_discriminant(F, basis, epipole)
    if abs(support_disc) < support_discriminant_threshold:
        diagnostics = PairDiagnostics(
            label=observation.label,
            valid=False,
            reason="near_support_degenerate",
            original_weight=float(observation.weight),
            effective_weight=0.0,
            kruppa_valid=False,
            left_epipole=epipole,
            singular_values=singular_values,
            support_discriminant=support_disc,
        )
        zero = sp.zeros(2, 2)
        return (zero, zero), diagnostics

    basis_expr = sp.Matrix(basis)
    omega = _diac_matrix_expr(fx=fx, fy=fy, cx_symbol=cx_symbol, cy_symbol=cy_symbol)
    f_expr = sp.Matrix(F.tolist())
    skew_epipole = sp.Matrix(skew(epipole).tolist())
    left_term = basis_expr.T * (skew_epipole * omega * skew_epipole.T) * basis_expr
    right_term = basis_expr.T * (f_expr * omega * f_expr.T) * basis_expr
    diagnostics = PairDiagnostics(
        label=observation.label,
        valid=True,
        reason="ok",
        original_weight=float(observation.weight),
        effective_weight=float(observation.weight),
        kruppa_valid=True,
        left_epipole=epipole,
        singular_values=singular_values,
        support_discriminant=support_disc,
    )
    return (left_term, right_term), diagnostics


def build_kruppa_objective_coefficients(
    observations: list[FundamentalMatrixObservation],
    *,
    fx: float,
    fy: float,
    rank_threshold: float = 1e-8,
    support_discriminant_threshold: float = 1e-8,
) -> tuple[np.ndarray, list[PairDiagnostics]]:
    fx, fy = _validate_focal_lengths(fx, fy)
    cx_symbol, cy_symbol = sp.symbols("cx cy", real=True)
    objective = sp.Float(0.0)
    diagnostics: list[PairDiagnostics] = []
    for observation in observations:
        residuals, pair_diagnostics = kruppa_pair_residual_polynomials(
            observation,
            fx=fx,
            fy=fy,
            cx_symbol=cx_symbol,
            cy_symbol=cy_symbol,
            rank_threshold=rank_threshold,
            support_discriminant_threshold=support_discriminant_threshold,
        )
        angle_residual, angle_weight, angle_active = _known_rotation_term_polynomial(
            observation,
            fx=fx,
            fy=fy,
            cx_symbol=cx_symbol,
            cy_symbol=cy_symbol,
        )
        pair_diagnostics.angle_constraint_active = angle_active and angle_weight > 0.0
        diagnostics.append(pair_diagnostics)
        if pair_diagnostics.kruppa_valid and pair_diagnostics.effective_weight > 0.0:
            objective += pair_diagnostics.effective_weight * sum(residual * residual for residual in residuals)
        if pair_diagnostics.angle_constraint_active:
            objective += angle_weight * angle_residual * angle_residual
    coeffs = coefficient_tensor_from_expr(sp.expand(objective), cx_symbol, cy_symbol)
    return coeffs, diagnostics
