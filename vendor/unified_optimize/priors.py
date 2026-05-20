from __future__ import annotations

import math

import numpy as np

from .data import FocalPrior, PrincipalPointPrior


def cauchy_loss(value: float | np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    return float(np.sum(np.log1p(value * value)))


def prior_penalty_scalar(delta: float, scale: float, *, weight: float = 1.0, robust: bool = True) -> float:
    z = float(delta) / float(scale)
    if robust:
        return float(weight) * math.log1p(z * z)
    return float(weight) * z * z


def focal_prior_penalty(eta: float, prior: FocalPrior | None) -> float:
    if prior is None:
        return 0.0
    return prior_penalty_scalar(
        eta - prior.eta0,
        prior.scale_log,
        weight=prior.weight,
        robust=prior.robust,
    )


def principal_prior_penalty(cx: float, cy: float, prior: PrincipalPointPrior | None) -> float:
    if prior is None:
        return 0.0
    delta = (np.array([cx, cy], dtype=np.float64) - prior.center_array) / prior.scale_array
    if prior.robust:
        return float(prior.weight) * cauchy_loss(delta)
    return float(prior.weight) * float(np.dot(delta, delta))


def signed_cauchy_residual(delta: float, scale: float, *, weight: float = 1.0) -> float:
    z = float(delta) / float(scale)
    magnitude = math.sqrt(max(0.0, 2.0 * float(weight) * math.log1p(z * z)))
    return math.copysign(magnitude, z)


def focal_prior_residuals(eta: float, prior: FocalPrior | None) -> list[float]:
    if prior is None:
        return []
    if prior.robust:
        return [signed_cauchy_residual(eta - prior.eta0, prior.scale_log, weight=prior.weight)]
    return [math.sqrt(prior.weight) * (eta - prior.eta0) / prior.scale_log]


def principal_prior_residuals(cx: float, cy: float, prior: PrincipalPointPrior | None) -> list[float]:
    if prior is None:
        return []
    delta = np.array([cx, cy], dtype=np.float64) - prior.center_array
    scale = prior.scale_array
    if prior.robust:
        return [
            signed_cauchy_residual(delta[0], scale[0], weight=prior.weight),
            signed_cauchy_residual(delta[1], scale[1], weight=prior.weight),
        ]
    return (math.sqrt(prior.weight) * delta / scale).tolist()
