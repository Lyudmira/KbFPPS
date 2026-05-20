from __future__ import annotations

from math import comb

import numpy as np
import sympy as sp

from .types import SearchBox


def coefficient_tensor_from_expr(expr: sp.Expr, x_symbol: sp.Symbol, y_symbol: sp.Symbol) -> np.ndarray:
    poly = sp.Poly(sp.expand(expr), x_symbol, y_symbol)
    if poly.is_zero:
        return np.zeros((1, 1), dtype=np.float64)
    degree_x = int(poly.degree(x_symbol))
    degree_y = int(poly.degree(y_symbol))
    coeffs = np.zeros((degree_x + 1, degree_y + 1), dtype=np.float64)
    for (power_x, power_y), coeff in poly.as_dict().items():
        coeffs[int(power_x), int(power_y)] = float(sp.N(coeff))
    return coeffs


def evaluate_power_polynomial(coeffs: np.ndarray, x_value: float, y_value: float) -> float:
    coeffs = np.asarray(coeffs, dtype=np.float64)
    value = 0.0
    for power_x in range(coeffs.shape[0] - 1, -1, -1):
        row_value = 0.0
        for power_y in range(coeffs.shape[1] - 1, -1, -1):
            row_value = row_value * y_value + coeffs[power_x, power_y]
        value = value * x_value + row_value
    return float(value)


def derivative_tensors(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.shape[0] == 1:
        dx = np.zeros((1, coeffs.shape[1]), dtype=np.float64)
    else:
        dx = np.zeros((coeffs.shape[0] - 1, coeffs.shape[1]), dtype=np.float64)
        for power_x in range(1, coeffs.shape[0]):
            dx[power_x - 1, :] = power_x * coeffs[power_x, :]
    if coeffs.shape[1] == 1:
        dy = np.zeros((coeffs.shape[0], 1), dtype=np.float64)
    else:
        dy = np.zeros((coeffs.shape[0], coeffs.shape[1] - 1), dtype=np.float64)
        for power_y in range(1, coeffs.shape[1]):
            dy[:, power_y - 1] = power_y * coeffs[:, power_y]
    return dx, dy


def add_polynomials(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    degree_x = max(first.shape[0], second.shape[0])
    degree_y = max(first.shape[1], second.shape[1])
    result = np.zeros((degree_x, degree_y), dtype=np.float64)
    result[: first.shape[0], : first.shape[1]] += first
    result[: second.shape[0], : second.shape[1]] += second
    return _trim_coefficients(result)


def scale_polynomial(coeffs: np.ndarray, scale: float) -> np.ndarray:
    return _trim_coefficients(np.asarray(coeffs, dtype=np.float64) * float(scale))


def subtract_scalar(coeffs: np.ndarray, scalar: float) -> np.ndarray:
    result = np.asarray(coeffs, dtype=np.float64).copy()
    result[0, 0] -= float(scalar)
    return _trim_coefficients(result)


def multiply_polynomials(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    result = np.zeros(
        (first.shape[0] + second.shape[0] - 1, first.shape[1] + second.shape[1] - 1),
        dtype=np.float64,
    )
    for first_x in range(first.shape[0]):
        for first_y in range(first.shape[1]):
            first_coeff = first[first_x, first_y]
            if first_coeff == 0.0:
                continue
            for second_x in range(second.shape[0]):
                for second_y in range(second.shape[1]):
                    second_coeff = second[second_x, second_y]
                    if second_coeff == 0.0:
                        continue
                    result[first_x + second_x, first_y + second_y] += first_coeff * second_coeff
    return _trim_coefficients(result)


def square_polynomial(coeffs: np.ndarray) -> np.ndarray:
    return multiply_polynomials(coeffs, coeffs)


def hessian_tensors(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grad_x, grad_y = derivative_tensors(coeffs)
    hessian_xx, hessian_xy = derivative_tensors(grad_x)
    _, hessian_yy = derivative_tensors(grad_y)
    return hessian_xx, hessian_xy, hessian_yy


def gradient_squared_tensor(coeffs: np.ndarray) -> np.ndarray:
    grad_x, grad_y = derivative_tensors(coeffs)
    return add_polynomials(square_polynomial(grad_x), square_polynomial(grad_y))


def transformed_power_coefficients_on_unit_square(coeffs: np.ndarray, box: SearchBox) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float64)
    transformed = np.zeros_like(coeffs)
    x_offset = float(box.x_min)
    y_offset = float(box.y_min)
    x_scale = float(box.width)
    y_scale = float(box.height)
    for power_x in range(coeffs.shape[0]):
        for power_y in range(coeffs.shape[1]):
            coeff = coeffs[power_x, power_y]
            if coeff == 0.0:
                continue
            for reduced_x in range(power_x + 1):
                x_term = comb(power_x, reduced_x) * (x_offset ** (power_x - reduced_x)) * (x_scale ** reduced_x)
                for reduced_y in range(power_y + 1):
                    y_term = comb(power_y, reduced_y) * (y_offset ** (power_y - reduced_y)) * (y_scale ** reduced_y)
                    transformed[reduced_x, reduced_y] += coeff * x_term * y_term
    return transformed


def power_to_bernstein_unit(coeffs: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float64)
    degree_x = coeffs.shape[0] - 1
    degree_y = coeffs.shape[1] - 1
    bernstein = np.zeros_like(coeffs)
    for index_x in range(degree_x + 1):
        for index_y in range(degree_y + 1):
            total = 0.0
            for power_x in range(index_x + 1):
                x_factor = comb(index_x, power_x) / comb(degree_x, power_x)
                for power_y in range(index_y + 1):
                    y_factor = comb(index_y, power_y) / comb(degree_y, power_y)
                    total += coeffs[power_x, power_y] * x_factor * y_factor
            bernstein[index_x, index_y] = total
    return bernstein


def bernstein_coefficients_on_box(coeffs: np.ndarray, box: SearchBox) -> np.ndarray:
    transformed = transformed_power_coefficients_on_unit_square(coeffs, box)
    return power_to_bernstein_unit(transformed)


def _trim_coefficients(coeffs: np.ndarray, *, atol: float = 0.0) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float64)
    if coeffs.size == 0:
        return np.zeros((1, 1), dtype=np.float64)
    nonzero = np.argwhere(np.abs(coeffs) > atol)
    if nonzero.size == 0:
        return np.zeros((1, 1), dtype=np.float64)
    max_x, max_y = np.max(nonzero, axis=0)
    return coeffs[: max_x + 1, : max_y + 1]
