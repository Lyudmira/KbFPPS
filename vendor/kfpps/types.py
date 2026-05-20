from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(slots=True, frozen=True)
class SearchBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise ValueError("SearchBox bounds must be strictly ordered.")

    @property
    def width(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return float(self.y_max - self.y_min)

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max)],
            dtype=np.float64,
        )

    def as_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((self.x_min, self.x_max), (self.y_min, self.y_max))

    def expand(self, factor: float) -> "SearchBox":
        if factor <= 1.0:
            raise ValueError("Expansion factor must be greater than one.")
        center = self.center
        half_width = 0.5 * self.width * factor
        half_height = 0.5 * self.height * factor
        return SearchBox(
            x_min=float(center[0] - half_width),
            x_max=float(center[0] + half_width),
            y_min=float(center[1] - half_height),
            y_max=float(center[1] + half_height),
        )

    def on_boundary(self, x: float, y: float, tol: float) -> bool:
        return any(
            abs(value - bound) <= tol
            for value, bound in (
                (x, self.x_min),
                (x, self.x_max),
                (y, self.y_min),
                (y, self.y_max),
            )
        )

    def split_longest(self) -> tuple["SearchBox", "SearchBox"]:
        if self.width >= self.height:
            split = 0.5 * (self.x_min + self.x_max)
            return (
                SearchBox(self.x_min, split, self.y_min, self.y_max),
                SearchBox(split, self.x_max, self.y_min, self.y_max),
            )
        split = 0.5 * (self.y_min + self.y_max)
        return (
            SearchBox(self.x_min, self.x_max, self.y_min, split),
            SearchBox(self.x_min, self.x_max, split, self.y_max),
        )

    def sample_points(self) -> np.ndarray:
        center = self.center
        return np.array(
            [
                [self.x_min, self.y_min],
                [self.x_min, self.y_max],
                [self.x_max, self.y_min],
                [self.x_max, self.y_max],
                center,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_image_size(
        cls,
        image_size: tuple[int, int],
        *,
        fraction: float = 0.15,
    ) -> "SearchBox":
        width, height = image_size
        return cls(
            x_min=0.5 * width - fraction * width,
            x_max=0.5 * width + fraction * width,
            y_min=0.5 * height - fraction * height,
            y_max=0.5 * height + fraction * height,
        )


@dataclass(slots=True)
class FundamentalMatrixObservation:
    F: np.ndarray
    weight: float = 1.0
    label: str = ""
    known_rotation_trace: float | None = None
    known_rotation_angle_deg: float | None = None
    known_rotation_weight: float = 1.0

    def normalized_matrix(self) -> np.ndarray:
        matrix = np.asarray(self.F, dtype=np.float64).reshape(3, 3)
        fro_norm = float(np.linalg.norm(matrix, ord="fro"))
        if not np.isfinite(fro_norm) or fro_norm <= 0.0:
            raise ValueError("Fundamental matrix must have finite non-zero Frobenius norm.")
        return matrix / fro_norm

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
class PairDiagnostics:
    label: str
    valid: bool
    reason: str
    original_weight: float
    effective_weight: float
    kruppa_valid: bool = False
    angle_constraint_active: bool = False
    left_epipole: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    singular_values: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    support_discriminant: float = math.nan

    @property
    def contributes_objective(self) -> bool:
        return bool((self.kruppa_valid and self.effective_weight > 0.0) or self.angle_constraint_active)


@dataclass(slots=True)
class BernsteinNodeSummary:
    box: SearchBox
    depth: int
    lower_bound: float
    sampled_upper_bound: float


@dataclass(slots=True)
class BernsteinCertificate:
    best_point: np.ndarray
    best_value: float
    global_lower_bound: float
    global_upper_bound: float
    certified: bool
    explored_nodes: int
    pruned_nodes: int
    leaf_nodes: int
    node_summaries: list[BernsteinNodeSummary]


@dataclass(slots=True)
class CertifiedPrincipalPointResult:
    cx: float
    cy: float
    objective_value: float
    certificate_gap: float
    certified: bool
    search_box: SearchBox
    used_expanded_box: bool
    boundary_hit: bool
    valid_pairs: int
    diagnostics: list[PairDiagnostics]
    certificate: BernsteinCertificate
