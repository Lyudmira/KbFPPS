from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .data import IntrinsicsEstimate


def _to_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "detach") and hasattr(array, "cpu"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


@dataclass(slots=True)
class MogePointLMConfig:
    initial_focal: float = 400.0
    initial_shift: float = 0.0
    initial_principal: tuple[float, float] | None = None
    method: str = "lm"
    diff_step: float = 0.5
    ftol: float = 1e-15
    xtol: float = 1e-15
    gtol: float = 1e-15
    max_nfev: int = 1000
    pixel_center_offset: float = 0.0
    residual_normalization: str = "num_points"


class MogePointLMOptimizer:
    """Recover intrinsics from a MoGe-style camera point map.

    This is the small nonlinear least-squares model from
    dc_reality/splatting/utils/moge/_utils.py, written as a reusable optimizer:

        u = fx * X / (Z + t) + cx
        v = fy * Y / (Z + t) + cy

    The shift t is a depth-axis correction for affine/shift ambiguity in the
    predicted point map.
    """

    def __init__(self, config: MogePointLMConfig | None = None) -> None:
        self.config = config or MogePointLMConfig()

    def _pixel_grid(self, height: int, width: int) -> np.ndarray:
        xs = np.arange(width, dtype=np.float64) + self.config.pixel_center_offset
        ys = np.arange(height, dtype=np.float64) + self.config.pixel_center_offset
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        return np.stack([grid_x, grid_y], axis=-1)

    def _residuals(
        self,
        params: np.ndarray,
        point_x: np.ndarray,
        point_y: np.ndarray,
        point_z: np.ndarray,
        pixel_u: np.ndarray,
        pixel_v: np.ndarray,
    ) -> np.ndarray:
        fx, fy, shift, cx, cy = map(float, params)
        denom = point_z + shift
        safe = np.abs(denom) > 1e-12
        denom = np.where(safe, denom, np.sign(denom + 1e-30) * 1e-12)
        res_u = fx * point_x / denom - pixel_u + cx
        res_v = fy * point_y / denom - pixel_v + cy
        residuals = np.concatenate([res_u, res_v])
        if self.config.residual_normalization == "num_points":
            residuals = residuals / max(1, point_x.shape[0])
        elif self.config.residual_normalization == "sqrt_num_points":
            residuals = residuals / math.sqrt(max(1, point_x.shape[0]))
        elif self.config.residual_normalization != "none":
            raise ValueError(f"Unknown residual normalization: {self.config.residual_normalization!r}")
        return residuals

    def solve(self, points: Any, mask: Any | None = None) -> IntrinsicsEstimate:
        point_array = _to_numpy(points).astype(np.float64)
        if point_array.ndim != 3 or point_array.shape[-1] != 3:
            raise ValueError("points must have shape (H, W, 3).")
        height, width = point_array.shape[:2]
        if mask is None:
            mask_array = np.ones((height, width), dtype=bool)
        else:
            mask_array = _to_numpy(mask).astype(bool)
            if mask_array.shape != (height, width):
                raise ValueError("mask must have shape (H, W).")
        finite = np.isfinite(point_array).all(axis=-1)
        mask_array = mask_array & finite
        if int(mask_array.sum()) < 5:
            raise ValueError("At least five valid pixels are required.")

        uv = self._pixel_grid(height, width)
        pts = point_array[mask_array]
        uvs = uv[mask_array]
        point_x = pts[:, 0]
        point_y = pts[:, 1]
        point_z = pts[:, 2]
        pixel_u = uvs[:, 0]
        pixel_v = uvs[:, 1]

        if self.config.initial_principal is None:
            cx0, cy0 = width / 2.0, height / 2.0
        else:
            cx0, cy0 = self.config.initial_principal
        initial = np.array(
            [
                self.config.initial_focal,
                self.config.initial_focal,
                self.config.initial_shift,
                cx0,
                cy0,
            ],
            dtype=np.float64,
        )
        result = least_squares(
            fun=self._residuals,
            x0=initial,
            args=(point_x, point_y, point_z, pixel_u, pixel_v),
            method=self.config.method,
            diff_step=self.config.diff_step,
            ftol=self.config.ftol,
            xtol=self.config.xtol,
            gtol=self.config.gtol,
            max_nfev=self.config.max_nfev,
        )
        fx, fy, shift, cx, cy = map(float, result.x)
        shifted_points = point_array.copy()
        shifted_points[..., 2] += shift
        focal = 0.5 * (fx + fy)
        return IntrinsicsEstimate(
            focal=focal,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            loss=float(0.5 * np.dot(result.fun, result.fun)),
            success=bool(result.success),
            message=str(result.message),
            diagnostics={
                "shift": shift,
                "result": result,
                "num_valid_pixels": int(mask_array.sum()),
                "shifted_points": shifted_points,
            },
        )


def recover_intrinsic_with_shift(points: Any, mask: Any | None = None) -> tuple[np.ndarray, dict[str, float]]:
    """Compatibility helper mirroring the dc_reality MoGe utility."""

    estimate = MogePointLMOptimizer().solve(points, mask)
    shifted_points = estimate.diagnostics["shifted_points"]
    return shifted_points, {
        "focal_x": float(estimate.fx),
        "focal_y": float(estimate.fy),
        "cx": float(estimate.cx),
        "cy": float(estimate.cy),
        "t": float(estimate.diagnostics["shift"]),
    }
