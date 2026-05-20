from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .polynomial import (
    bernstein_coefficients_on_box,
    gradient_squared_tensor,
    hessian_tensors,
    multiply_polynomials,
    subtract_scalar,
)
from .types import SearchBox


@dataclass(slots=True, frozen=True)
class GradientExclusionBoxCertificate:
    box: SearchBox
    lower_bound: float
    verified: bool


@dataclass(slots=True, frozen=True)
class HessianPositivityBoxCertificate:
    box: SearchBox
    mu: float
    lower_hxx_minus_mu: float
    lower_hyy_minus_mu: float
    lower_principal_minor: float
    verified: bool


def bernstein_lower_bound(coeffs: np.ndarray, box: SearchBox) -> float:
    return float(np.min(bernstein_coefficients_on_box(coeffs, box)))


def gradient_exclusion_certificate(
    objective_coeffs: np.ndarray,
    box: SearchBox,
    *,
    strict_margin: float = 0.0,
) -> GradientExclusionBoxCertificate:
    gradient_squared = gradient_squared_tensor(objective_coeffs)
    lower_bound = bernstein_lower_bound(gradient_squared, box)
    return GradientExclusionBoxCertificate(
        box=box,
        lower_bound=lower_bound,
        verified=bool(lower_bound > strict_margin),
    )


def hessian_positivity_certificate(
    objective_coeffs: np.ndarray,
    box: SearchBox,
    *,
    mu: float,
    strict_margin: float = 0.0,
) -> HessianPositivityBoxCertificate:
    hxx, hxy, hyy = hessian_tensors(objective_coeffs)
    hxx_minus_mu = subtract_scalar(hxx, mu)
    hyy_minus_mu = subtract_scalar(hyy, mu)
    principal_minor = subtract_scalar(
        multiply_polynomials(hxx_minus_mu, hyy_minus_mu),
        0.0,
    )
    principal_minor = _subtract_polynomial(principal_minor, multiply_polynomials(hxy, hxy))

    lower_hxx = bernstein_lower_bound(hxx_minus_mu, box)
    lower_hyy = bernstein_lower_bound(hyy_minus_mu, box)
    lower_minor = bernstein_lower_bound(principal_minor, box)
    verified = lower_hxx > strict_margin and lower_hyy > strict_margin and lower_minor > strict_margin
    return HessianPositivityBoxCertificate(
        box=box,
        mu=float(mu),
        lower_hxx_minus_mu=lower_hxx,
        lower_hyy_minus_mu=lower_hyy,
        lower_principal_minor=lower_minor,
        verified=bool(verified),
    )


def _subtract_polynomial(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    degree_x = max(first.shape[0], second.shape[0])
    degree_y = max(first.shape[1], second.shape[1])
    result = np.zeros((degree_x, degree_y), dtype=np.float64)
    result[: first.shape[0], : first.shape[1]] += first
    result[: second.shape[0], : second.shape[1]] -= second
    return result
