from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy import optimize

from .data import FocalPrior, IntrinsicsEstimate, OptimizerBounds, PairMatches, PrincipalPointPrior
from .geometry import (
    estimate_fundamental_normalized_eight_point,
    essential_from_fundamental,
    essential_motion_candidates,
    fundamental_from_rt,
    make_pinhole_k,
    rotation_from_rotvec,
    rotvec_from_rotation,
    sampson_rms,
    sampson_signed_residuals,
)
from .priors import focal_prior_residuals, principal_prior_residuals


def _cauchy_residual_array(values: np.ndarray, scale: float, weight: float) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64) / float(scale)
    return np.sign(z) * np.sqrt(np.maximum(0.0, 2.0 * float(weight) * np.log1p(z * z)))


@dataclass(slots=True)
class RawSampsonConfig:
    image_size: tuple[int, int]
    focal_prior: FocalPrior
    principal_prior: PrincipalPointPrior | None = None
    bounds: OptimizerBounds | None = None
    focal_ratio: tuple[float, float] = (0.35, 3.0)
    principal_margin: float = 10.0
    match_robust_scale_px: float | None = 2.0
    normalize_pair_residuals: bool = True
    max_nfev: int = 500
    xtol: float = 1e-10
    ftol: float = 1e-10
    gtol: float = 1e-10

    def resolved_bounds(self) -> OptimizerBounds:
        if self.bounds is not None:
            return self.bounds
        return OptimizerBounds.from_image_size(
            self.image_size,
            self.focal_prior,
            focal_ratio=self.focal_ratio,
            principal_margin=self.principal_margin,
        )


class RawSampsonJointOptimizer:
    """Jointly estimate log focal, principal point, and per-pair relative poses.

    This is the "do not compress to F too early" estimator:

        min_{eta,c,R_ij,t_ij} sum robust Sampson residuals
            + robust log-focal prior
            + broad/proper principal-point prior.

    Each pair owns six nuisance variables: a rotation vector and an unconstrained
    translation vector. The translation is normalized internally because only
    its direction is observable from epipolar geometry.
    """

    def __init__(self, config: RawSampsonConfig) -> None:
        self.config = config
        self.bounds = config.resolved_bounds()

    def _initial_principal(self) -> np.ndarray:
        width, height = self.config.image_size
        if self.config.principal_prior is not None:
            point = self.config.principal_prior.center_array.copy()
        else:
            point = np.array([0.5 * width, 0.5 * height], dtype=np.float64)
        point[0] = np.clip(point[0], self.bounds.cx[0], self.bounds.cx[1])
        point[1] = np.clip(point[1], self.bounds.cy[0], self.bounds.cy[1])
        return point

    def _initial_motion(self, pair: PairMatches, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points0, points1 = pair.arrays()
        if pair.initial_F is not None:
            F = np.asarray(pair.initial_F, dtype=np.float64).reshape(3, 3)
        else:
            F = estimate_fundamental_normalized_eight_point(points0, points1)
        E = essential_from_fundamental(F, K)
        best_rotation = np.eye(3, dtype=np.float64)
        best_translation = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        best_rms = math.inf
        for rotation, translation in essential_motion_candidates(E):
            try:
                candidate_F = fundamental_from_rt(K, rotation, translation)
                rms = sampson_rms(points0, points1, candidate_F)
            except Exception:
                continue
            if rms < best_rms:
                best_rms = rms
                best_rotation = rotation
                best_translation = translation
        return rotvec_from_rotation(best_rotation), best_translation, F

    def _pack_initial(self, pairs: list[PairMatches]) -> tuple[np.ndarray, list[np.ndarray]]:
        eta0 = float(np.clip(self.config.focal_prior.eta0, self.bounds.eta[0], self.bounds.eta[1]))
        cx0, cy0 = self._initial_principal()
        K0 = make_pinhole_k(math.exp(eta0), cx0, cy0)
        params = [eta0, float(cx0), float(cy0)]
        initial_fundamentals = []
        for pair in pairs:
            rotvec, translation, F = self._initial_motion(pair, K0)
            params.extend(rotvec.tolist())
            params.extend(np.asarray(translation, dtype=np.float64).reshape(3).tolist())
            initial_fundamentals.append(F)
        return np.asarray(params, dtype=np.float64), initial_fundamentals

    @staticmethod
    def _pair_offset(pair_index: int) -> int:
        return 3 + 6 * pair_index

    def _unpack_pair(self, params: np.ndarray, pair_index: int) -> tuple[np.ndarray, np.ndarray]:
        offset = self._pair_offset(pair_index)
        rotvec = params[offset : offset + 3]
        translation = params[offset + 3 : offset + 6]
        return rotation_from_rotvec(rotvec), translation

    def _residuals(self, params: np.ndarray, pairs: list[PairMatches]) -> np.ndarray:
        eta, cx, cy = map(float, params[:3])
        K = make_pinhole_k(math.exp(eta), cx, cy)
        residuals: list[np.ndarray] = []
        for pair_index, pair in enumerate(pairs):
            points0, points1 = pair.arrays()
            rotation, translation = self._unpack_pair(params, pair_index)
            F = fundamental_from_rt(K, rotation, translation)
            pair_residuals = sampson_signed_residuals(points0, points1, F)
            pair_weight = float(pair.weight)
            if self.config.normalize_pair_residuals:
                pair_weight /= max(1, points0.shape[0])
            if self.config.match_robust_scale_px is not None:
                pair_residuals = _cauchy_residual_array(
                    pair_residuals,
                    self.config.match_robust_scale_px,
                    pair_weight,
                )
            else:
                pair_residuals = math.sqrt(pair_weight) * pair_residuals
            residuals.append(pair_residuals)
        prior = []
        prior.extend(focal_prior_residuals(eta, self.config.focal_prior))
        prior.extend(principal_prior_residuals(cx, cy, self.config.principal_prior))
        if prior:
            residuals.append(np.asarray(prior, dtype=np.float64))
        return np.concatenate(residuals) if residuals else np.empty(0, dtype=np.float64)

    def _bounds(self, num_pairs: int) -> tuple[np.ndarray, np.ndarray]:
        lower = [self.bounds.eta[0], self.bounds.cx[0], self.bounds.cy[0]]
        upper = [self.bounds.eta[1], self.bounds.cx[1], self.bounds.cy[1]]
        lower.extend([-np.inf] * (6 * num_pairs))
        upper.extend([np.inf] * (6 * num_pairs))
        return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    def solve(self, pairs: Iterable[PairMatches]) -> IntrinsicsEstimate:
        pair_list = list(pairs)
        if not pair_list:
            raise ValueError("At least one matched pair is required.")
        x0, initial_fundamentals = self._pack_initial(pair_list)
        lower, upper = self._bounds(len(pair_list))
        result = optimize.least_squares(
            lambda params: self._residuals(np.asarray(params, dtype=np.float64), pair_list),
            x0=x0,
            bounds=(lower, upper),
            method="trf",
            max_nfev=self.config.max_nfev,
            xtol=self.config.xtol,
            ftol=self.config.ftol,
            gtol=self.config.gtol,
        )
        eta, cx, cy = map(float, result.x[:3])
        focal = float(math.exp(eta))
        K = make_pinhole_k(focal, cx, cy)
        per_pair = []
        for pair_index, pair in enumerate(pair_list):
            points0, points1 = pair.arrays()
            rotation, translation = self._unpack_pair(result.x, pair_index)
            F = fundamental_from_rt(K, rotation, translation)
            per_pair.append(
                {
                    "label": pair.label,
                    "num_matches": int(points0.shape[0]),
                    "sampson_rms_px": sampson_rms(points0, points1, F),
                    "rotation": rotation,
                    "translation_direction": translation / max(np.linalg.norm(translation), 1e-12),
                    "F": F,
                }
            )
        hessian_eigvals = None
        try:
            hessian = result.jac.T @ result.jac
            hessian_eigvals = np.linalg.eigvalsh(hessian[:3, :3])
        except Exception:
            pass
        return IntrinsicsEstimate(
            focal=focal,
            cx=cx,
            cy=cy,
            loss=float(0.5 * np.dot(result.fun, result.fun)),
            success=bool(result.success),
            message=str(result.message),
            diagnostics={
                "num_pairs": len(pair_list),
                "initial_fundamentals": initial_fundamentals,
                "per_pair": per_pair,
                "result": result,
                "hessian_intrinsics_eigvals": hessian_eigvals,
                "bounds": self.bounds,
            },
        )
