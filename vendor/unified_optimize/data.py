from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class FocalPrior:
    """A weak log-focal prior, centered on MoGe or another focal estimate."""

    focal_px: float
    scale_log: float = math.log(1.35)
    bias: float = 1.0
    weight: float = 1.0
    robust: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.focal_px) or self.focal_px <= 0.0:
            raise ValueError("focal_px must be finite and positive.")
        if not np.isfinite(self.scale_log) or self.scale_log <= 0.0:
            raise ValueError("scale_log must be finite and positive.")
        if not np.isfinite(self.bias) or self.bias <= 0.0:
            raise ValueError("bias must be finite and positive.")

    @property
    def eta0(self) -> float:
        return float(math.log(self.focal_px / self.bias))


@dataclass(slots=True, frozen=True)
class PrincipalPointPrior:
    """A broad proper prior for principal point.

    The scale is deliberately explicit: use a large value when the principal
    point may sit far outside the image.
    """

    center: tuple[float, float]
    scale: tuple[float, float] | float
    weight: float = 1.0
    robust: bool = True

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (2,) or np.any(~np.isfinite(center)):
            raise ValueError("center must contain two finite values.")
        scale = np.asarray(self.scale, dtype=np.float64)
        if scale.ndim == 0:
            if not np.isfinite(float(scale)) or float(scale) <= 0.0:
                raise ValueError("scale must be positive.")
        elif scale.shape != (2,) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("scale must be a positive scalar or two positive values.")

    @property
    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=np.float64)

    @property
    def scale_array(self) -> np.ndarray:
        scale = np.asarray(self.scale, dtype=np.float64)
        if scale.ndim == 0:
            return np.array([float(scale), float(scale)], dtype=np.float64)
        return scale.astype(np.float64)


@dataclass(slots=True, frozen=True)
class OptimizerBounds:
    eta: tuple[float, float]
    cx: tuple[float, float]
    cy: tuple[float, float]

    def __post_init__(self) -> None:
        for name, bounds in (("eta", self.eta), ("cx", self.cx), ("cy", self.cy)):
            lo, hi = bounds
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"{name} bounds must be finite and strictly ordered.")

    @classmethod
    def from_image_size(
        cls,
        image_size: tuple[int, int],
        focal_prior: FocalPrior,
        *,
        focal_ratio: tuple[float, float] = (0.35, 3.0),
        principal_margin: float = 3.0,
    ) -> "OptimizerBounds":
        width, height = image_size
        if focal_ratio[0] <= 0.0 or focal_ratio[0] >= focal_ratio[1]:
            raise ValueError("focal_ratio must be positive and ordered.")
        cx_center = 0.5 * width
        cy_center = 0.5 * height
        return cls(
            eta=(
                focal_prior.eta0 + math.log(focal_ratio[0]),
                focal_prior.eta0 + math.log(focal_ratio[1]),
            ),
            cx=(cx_center - principal_margin * width, cx_center + principal_margin * width),
            cy=(cy_center - principal_margin * height, cy_center + principal_margin * height),
        )

    def scipy_bounds(self) -> tuple[list[float], list[float]]:
        lower = [self.eta[0], self.cx[0], self.cy[0]]
        upper = [self.eta[1], self.cx[1], self.cy[1]]
        return lower, upper


@dataclass(slots=True)
class FundamentalObservation:
    F: np.ndarray
    weight: float = 1.0
    label: str = ""
    known_rotation_trace: float | None = None
    known_rotation_angle_deg: float | None = None
    known_rotation_weight: float = 1.0

    def matrix(self) -> np.ndarray:
        matrix = np.asarray(self.F, dtype=np.float64).reshape(3, 3)
        if np.any(~np.isfinite(matrix)):
            raise ValueError("Fundamental matrix contains non-finite values.")
        return matrix

    def resolved_rotation_trace(self) -> float | None:
        trace_value = self.known_rotation_trace
        angle_value = self.known_rotation_angle_deg
        if trace_value is None and angle_value is None:
            return None
        resolved_trace: float | None = None
        if trace_value is not None:
            resolved_trace = float(trace_value)
            if not np.isfinite(resolved_trace):
                raise ValueError("known_rotation_trace must be finite.")
        if angle_value is not None:
            angle_deg = float(angle_value)
            if not np.isfinite(angle_deg):
                raise ValueError("known_rotation_angle_deg must be finite.")
            angle_trace = 1.0 + 2.0 * math.cos(math.radians(angle_deg))
            if resolved_trace is not None and abs(resolved_trace - angle_trace) > 1e-6:
                raise ValueError("known_rotation_trace and known_rotation_angle_deg are inconsistent.")
            resolved_trace = angle_trace
        assert resolved_trace is not None
        if resolved_trace < -1.0 - 1e-9 or resolved_trace > 3.0 + 1e-9:
            raise ValueError("Resolved rotation trace must lie in [-1, 3].")
        return float(np.clip(resolved_trace, -1.0, 3.0))


@dataclass(slots=True)
class PairMatches:
    points0: np.ndarray
    points1: np.ndarray
    weight: float = 1.0
    label: str = ""
    initial_F: np.ndarray | None = None

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        points0 = np.asarray(self.points0, dtype=np.float64)
        points1 = np.asarray(self.points1, dtype=np.float64)
        if points0.ndim != 2 or points1.ndim != 2 or points0.shape[1] != 2 or points1.shape[1] != 2:
            raise ValueError("PairMatches points must have shape (N, 2).")
        if points0.shape != points1.shape:
            raise ValueError("points0 and points1 must have the same shape.")
        if points0.shape[0] < 8:
            raise ValueError("At least eight matches are required.")
        mask = np.isfinite(points0).all(axis=1) & np.isfinite(points1).all(axis=1)
        if mask.sum() < 8:
            raise ValueError("At least eight finite matches are required.")
        return points0[mask], points1[mask]


@dataclass(slots=True)
class ProfileSample:
    eta: float
    focal: float
    cx: float
    cy: float
    profile_loss: float
    total_loss: float
    success: bool
    message: str = ""
    laplace_logdet: float | None = None


@dataclass(slots=True)
class IntrinsicsEstimate:
    focal: float
    cx: float
    cy: float
    loss: float
    success: bool
    message: str
    fx: float | None = None
    fy: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def eta(self) -> float:
        return float(math.log(self.focal))

    @property
    def K(self) -> np.ndarray:
        fx = self.focal if self.fx is None else self.fx
        fy = self.focal if self.fy is None else self.fy
        return np.array(
            [
                [fx, 0.0, self.cx],
                [0.0, fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
