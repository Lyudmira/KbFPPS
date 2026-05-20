from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True, slots=True)
class ExactKruppaSystem:
    a: sp.Symbol
    b: sp.Symbol
    C: sp.Matrix
    D: sp.Matrix
    residuals: tuple[sp.Expr, sp.Expr, sp.Expr]


@dataclass(frozen=True, slots=True)
class ProjectiveKruppaSystem:
    X: sp.Symbol
    Y: sp.Symbol
    Z: sp.Symbol
    lam: sp.Symbol
    mu: sp.Symbol
    C: sp.Matrix
    D: sp.Matrix
    equations: tuple[sp.Expr, sp.Expr, sp.Expr]


def normalized_diac_expr(a: sp.Symbol, b: sp.Symbol) -> sp.Matrix:
    return sp.Matrix(
        [
            [1 + a * a, a * b, a],
            [a * b, 1 + b * b, b],
            [a, b, 1],
        ]
    )


def homogenized_diac_expr(X: sp.Symbol, Y: sp.Symbol, Z: sp.Symbol) -> sp.Matrix:
    return sp.Matrix(
        [
            [Z * Z + X * X, X * Y, X * Z],
            [X * Y, Z * Z + Y * Y, Y * Z],
            [X * Z, Y * Z, Z * Z],
        ]
    )


def normalized_kruppa_system(
    rotation: sp.Matrix,
    perpendicular_basis: sp.Matrix,
    *,
    a: sp.Symbol | None = None,
    b: sp.Symbol | None = None,
) -> ExactKruppaSystem:
    a = a or sp.Symbol("a")
    b = b or sp.Symbol("b")
    R = sp.Matrix(rotation)
    P = sp.Matrix(perpendicular_basis)
    if R.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix.")
    if P.shape != (3, 2):
        raise ValueError("perpendicular_basis must be a 3x2 matrix.")

    W = normalized_diac_expr(a, b)
    C = sp.expand(P.T * W * P)
    D = sp.expand(P.T * R * W * R.T * P)
    residuals = proportionality_residuals(C, D)
    return ExactKruppaSystem(a=a, b=b, C=C, D=D, residuals=residuals)


def projective_kruppa_system(
    rotation: sp.Matrix,
    perpendicular_basis: sp.Matrix,
    *,
    X: sp.Symbol | None = None,
    Y: sp.Symbol | None = None,
    Z: sp.Symbol | None = None,
    lam: sp.Symbol | None = None,
    mu: sp.Symbol | None = None,
) -> ProjectiveKruppaSystem:
    X = X or sp.Symbol("X")
    Y = Y or sp.Symbol("Y")
    Z = Z or sp.Symbol("Z")
    lam = lam or sp.Symbol("lambda")
    mu = mu or sp.Symbol("mu")
    R = sp.Matrix(rotation)
    P = sp.Matrix(perpendicular_basis)
    if R.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix.")
    if P.shape != (3, 2):
        raise ValueError("perpendicular_basis must be a 3x2 matrix.")

    W = homogenized_diac_expr(X, Y, Z)
    C = sp.expand(P.T * W * P)
    D = sp.expand(P.T * R * W * R.T * P)
    equations = (
        sp.expand(mu * D[0, 0] - lam * C[0, 0]),
        sp.expand(mu * D[0, 1] - lam * C[0, 1]),
        sp.expand(mu * D[1, 1] - lam * C[1, 1]),
    )
    return ProjectiveKruppaSystem(X=X, Y=Y, Z=Z, lam=lam, mu=mu, C=C, D=D, equations=equations)


def proportionality_residuals(left: sp.Matrix, right: sp.Matrix) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    if left.shape != (2, 2) or right.shape != (2, 2):
        raise ValueError("proportionality_residuals expects two 2x2 matrices.")
    minor_01 = sp.expand(left[0, 0] * right[0, 1] - left[0, 1] * right[0, 0])
    minor_02 = sp.expand(left[0, 0] * right[1, 1] - left[1, 1] * right[0, 0])
    minor_12 = sp.expand(left[0, 1] * right[1, 1] - left[1, 1] * right[0, 1])
    return minor_01, minor_02, minor_12


def lex_groebner_for_affine_candidates(
    system: ExactKruppaSystem,
    *,
    saturate: bool = True,
    saturator: sp.Expr | None = None,
) -> sp.GroebnerBasis:
    if not saturate:
        return sp.groebner(system.residuals, system.a, system.b, order="lex")

    saturation_variable = sp.Symbol("_sat")
    saturation_polynomial = saturator or system.C[0, 0]
    saturated_basis = sp.groebner(
        [*system.residuals, 1 - saturation_variable * saturation_polynomial],
        saturation_variable,
        system.a,
        system.b,
        order="lex",
    )
    eliminated = [
        poly.as_expr()
        for poly in saturated_basis.polys
        if saturation_variable not in poly.as_expr().free_symbols
    ]
    if not eliminated:
        raise ValueError("Saturation eliminated all affine candidate equations.")
    return sp.groebner(eliminated, system.a, system.b, order="lex")


def univariate_candidate_polynomial(groebner_basis: sp.GroebnerBasis, variable: sp.Symbol) -> sp.Poly:
    candidates = [
        sp.Poly(poly.as_expr(), variable)
        for poly in groebner_basis.polys
        if poly.as_expr().free_symbols <= {variable}
    ]
    if not candidates:
        raise ValueError(f"Groebner basis has no univariate polynomial in {variable}.")
    return max(candidates, key=lambda poly: poly.degree())


def affine_candidate_count(system: ExactKruppaSystem) -> int:
    basis = lex_groebner_for_affine_candidates(system)
    if not basis.is_zero_dimensional:
        raise ValueError("Kruppa candidate ideal is not zero-dimensional.")
    return univariate_candidate_polynomial(basis, system.b).degree()


def generic_projective_chart_groebner(system: ProjectiveKruppaSystem) -> sp.GroebnerBasis:
    x, y, rho = sp.symbols("x_chart y_chart rho_chart")
    substitutions = {
        system.X: x,
        system.Y: y,
        system.Z: 1 - x - y,
        system.lam: rho,
        system.mu: 1 - rho,
    }
    equations = [sp.expand(equation.subs(substitutions)) for equation in system.equations]
    return sp.groebner(equations, x, y, rho, order="lex")


def projective_candidate_count(system: ProjectiveKruppaSystem) -> int:
    basis = generic_projective_chart_groebner(system)
    if not basis.is_zero_dimensional:
        raise ValueError("Projective Kruppa chart ideal is not zero-dimensional.")
    rho = basis.gens[-1]
    return univariate_candidate_polynomial(basis, rho).degree()


def support_polynomial_at_infinity(
    rotation: sp.Matrix,
    perpendicular_basis: sp.Matrix,
    *,
    X: sp.Symbol | None = None,
    Y: sp.Symbol | None = None,
) -> sp.Expr:
    X = X or sp.Symbol("X")
    Y = Y or sp.Symbol("Y")
    R = sp.Matrix(rotation)
    P = sp.Matrix(perpendicular_basis)
    if R.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix.")
    if P.shape != (3, 2):
        raise ValueError("perpendicular_basis must be a 3x2 matrix.")
    direction = sp.Matrix([X, Y, 0])
    return sp.expand(sp.Matrix.hstack(P.T * R * direction, P.T * direction).det())


def dehomogenized_support_discriminant(
    rotation: sp.Matrix,
    perpendicular_basis: sp.Matrix,
    *,
    variable: sp.Symbol | None = None,
) -> sp.Expr:
    variable = variable or sp.Symbol("x")
    Y = sp.Symbol("Y")
    support = support_polynomial_at_infinity(rotation, perpendicular_basis, X=variable, Y=Y)
    dehomogenized = sp.expand(support.subs(Y, 1))
    return sp.factor(sp.discriminant(dehomogenized, variable))


def primitive_polynomial(expr: sp.Expr, *gens: sp.Symbol) -> sp.Expr:
    poly = sp.Poly(sp.expand(expr), *gens, domain=sp.QQ)
    _, primitive = poly.primitive()
    return primitive.as_expr()


def numerical_affine_candidates(
    system: ExactKruppaSystem,
    *,
    precision: int = 30,
    imaginary_tolerance: float = 1e-9,
) -> list[tuple[complex, complex]]:
    basis = lex_groebner_for_affine_candidates(system)
    if not basis.is_zero_dimensional:
        raise ValueError("Kruppa candidate ideal is not zero-dimensional.")

    b_poly = univariate_candidate_polynomial(basis, system.b)
    try:
        a_poly = _linear_polynomial_in_a(basis, system.a)
    except ValueError:
        return _solve_candidates_directly(system, imaginary_tolerance=imaginary_tolerance)
    candidates: list[tuple[complex, complex]] = []
    for b_root in sp.nroots(b_poly.as_expr(), n=precision, maxsteps=200):
        a_root = _solve_linear_a(a_poly, system.a, system.b, b_root)
        a_complex = complex(a_root)
        b_complex = complex(b_root)
        if abs(a_complex.imag) < imaginary_tolerance:
            a_complex = complex(a_complex.real, 0.0)
        if abs(b_complex.imag) < imaginary_tolerance:
            b_complex = complex(b_complex.real, 0.0)
        candidates.append((a_complex, b_complex))
    return candidates


def shared_real_candidates(
    systems: list[ExactKruppaSystem],
    *,
    precision: int = 30,
    tolerance: float = 1e-7,
) -> list[tuple[float, float]]:
    if not systems:
        return []
    shared = real_affine_candidates(systems[0], precision=precision, imaginary_tolerance=tolerance)
    for system in systems[1:]:
        roots = real_affine_candidates(system, precision=precision, imaginary_tolerance=tolerance)
        shared = [
            candidate
            for candidate in shared
            if any(_candidate_distance(candidate, root) <= tolerance for root in roots)
        ]
    return shared


def real_affine_candidates(
    system: ExactKruppaSystem,
    *,
    precision: int = 30,
    imaginary_tolerance: float = 1e-9,
) -> list[tuple[float, float]]:
    roots = numerical_affine_candidates(
        system,
        precision=precision,
        imaginary_tolerance=imaginary_tolerance,
    )
    return [
        (float(a_root.real), float(b_root.real))
        for a_root, b_root in roots
        if abs(a_root.imag) < imaginary_tolerance and abs(b_root.imag) < imaginary_tolerance
    ]


def _solve_candidates_directly(
    system: ExactKruppaSystem,
    *,
    imaginary_tolerance: float,
) -> list[tuple[complex, complex]]:
    candidates: list[tuple[complex, complex]] = []
    for root in sp.solve(system.residuals, (system.a, system.b), dict=True):
        if system.a not in root or system.b not in root:
            continue
        a_complex = complex(sp.N(root[system.a]))
        b_complex = complex(sp.N(root[system.b]))
        if abs(a_complex.imag) < imaginary_tolerance:
            a_complex = complex(a_complex.real, 0.0)
        if abs(b_complex.imag) < imaginary_tolerance:
            b_complex = complex(b_complex.real, 0.0)
        candidates.append((a_complex, b_complex))
    return candidates


def _candidate_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(abs(first[0] - second[0]), abs(first[1] - second[1]))


def _linear_polynomial_in_a(groebner_basis: sp.GroebnerBasis, a: sp.Symbol) -> sp.Poly:
    candidates = [
        sp.Poly(poly.as_expr(), a)
        for poly in groebner_basis.polys
        if a in poly.as_expr().free_symbols and sp.Poly(poly.as_expr(), a).degree() == 1
    ]
    if not candidates:
        raise ValueError(f"Groebner basis has no linear polynomial in {a}.")
    return candidates[0]


def _solve_linear_a(poly: sp.Poly, a: sp.Symbol, b: sp.Symbol, b_value: sp.Expr) -> sp.Expr:
    coeff_a = poly.coeff_monomial(a)
    coeff_0 = poly.coeff_monomial(1)
    return sp.N(-coeff_0.subs(b, b_value) / coeff_a.subs(b, b_value))


def paper_rational_witness_system() -> ExactKruppaSystem:
    rotation = sp.Matrix(
        [
            [-29, 2, 26],
            [22, -19, 26],
            [14, 34, 13],
        ]
    ) / 39
    perpendicular_basis = sp.Matrix(
        [
            [1, 0],
            [0, 1],
            [-sp.Rational(2, 3), sp.Rational(1, 3)],
        ]
    )
    return normalized_kruppa_system(rotation, perpendicular_basis)


def paper_rational_witness_projective_system() -> ProjectiveKruppaSystem:
    rotation = sp.Matrix(
        [
            [-29, 2, 26],
            [22, -19, 26],
            [14, 34, 13],
        ]
    ) / 39
    perpendicular_basis = sp.Matrix(
        [
            [1, 0],
            [0, 1],
            [-sp.Rational(2, 3), sp.Rational(1, 3)],
        ]
    )
    return projective_kruppa_system(rotation, perpendicular_basis)
