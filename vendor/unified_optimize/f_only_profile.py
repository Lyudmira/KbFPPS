from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable
import warnings

import numpy as np
from scipy import optimize

from .data import (
    FocalPrior,
    FundamentalObservation,
    IntrinsicsEstimate,
    OptimizerBounds,
    PrincipalPointPrior,
    ProfileSample,
)
from .geometry import essential_residual, known_rotation_trace_residual, make_pinhole_k, numerical_hessian_2d
from .priors import focal_prior_penalty, principal_prior_penalty


def _as_observations(observations: Iterable[FundamentalObservation | object]) -> list[FundamentalObservation]:
    converted: list[FundamentalObservation] = []
    for obs in observations:
        matrix = obs.F
        weight = getattr(obs, "weight", 1.0)
        label = getattr(obs, "label", "")
        converted.append(
            FundamentalObservation(
                F=np.asarray(matrix, dtype=np.float64),
                weight=float(weight),
                label=str(label),
                known_rotation_trace=getattr(obs, "known_rotation_trace", None),
                known_rotation_angle_deg=getattr(obs, "known_rotation_angle_deg", None),
                known_rotation_weight=float(getattr(obs, "known_rotation_weight", 1.0)),
            )
        )
    if not converted:
        raise ValueError("At least one fundamental observation is required.")
    return converted


def _clip_point(point: np.ndarray, bounds: OptimizerBounds) -> np.ndarray:
    return np.array(
        [
            np.clip(point[0], bounds.cx[0], bounds.cx[1]),
            np.clip(point[1], bounds.cy[0], bounds.cy[1]),
        ],
        dtype=np.float64,
    )


def _resolved_rotation_trace(observation: object) -> float | None:
    resolver = getattr(observation, "resolved_rotation_trace", None)
    if callable(resolver):
        return resolver()
    trace_value = getattr(observation, "known_rotation_trace", None)
    angle_value = getattr(observation, "known_rotation_angle_deg", None)
    if trace_value is None and angle_value is None:
        return None
    if trace_value is not None:
        return float(trace_value)
    return float(1.0 + 2.0 * math.cos(math.radians(float(angle_value))))


@dataclass(slots=True)
class FOnlyProfileConfig:
    image_size: tuple[int, int]
    focal_prior: FocalPrior
    principal_prior: PrincipalPointPrior | None = None
    bounds: OptimizerBounds | None = None
    focal_ratio: tuple[float, float] = (0.35, 3.0)
    principal_margin: float = 3.0
    num_focal_samples: int = 41
    residual_kind: str = "manifold"
    laplace_correction: bool = False
    laplace_ridge: float = 1e-12
    hessian_step_px: float = 1e-2
    inner_maxiter: int = 400
    refine: bool = True

    def resolved_bounds(self) -> OptimizerBounds:
        if self.bounds is not None:
            return self.bounds
        return OptimizerBounds.from_image_size(
            self.image_size,
            self.focal_prior,
            focal_ratio=self.focal_ratio,
            principal_margin=self.principal_margin,
        )


class FOnlyProfileOptimizer:
    """Profile optimizer over log focal and principal point from F-only geometry.

    This is the cleaned-up version of the original idea:

        for f around MoGe f0:
            solve/optimize cx, cy
            score K^T F K by essential singular values
            add focal and optional principal-point prior

    Set residual_kind="original" to reproduce the first residual we discussed:
        ((s1 - s2)^2 + s3^2) / (s1^2 + s2^2)

    Set residual_kind="manifold" for the Frobenius projection-style version:
        (0.5 * (s1 - s2)^2 + s3^2) / (s1^2 + s2^2 + s3^2)
    """

    def __init__(self, config: FOnlyProfileConfig) -> None:
        self.config = config
        self.bounds = config.resolved_bounds()

    def _data_loss(self, observations: list[FundamentalObservation], eta: float, cx: float, cy: float) -> float:
        focal = math.exp(float(eta))
        K = make_pinhole_k(focal, cx, cy)
        total = 0.0
        for obs in observations:
            total += float(obs.weight) * essential_residual(obs.matrix(), K, kind=self.config.residual_kind)
        return float(total)

    def _profile_data_loss(self, observations: list[FundamentalObservation], eta: float, point: np.ndarray) -> float:
        cx, cy = float(point[0]), float(point[1])
        return self._data_loss(observations, eta, cx, cy) + principal_prior_penalty(cx, cy, self.config.principal_prior)

    def _initial_points(self, previous: np.ndarray | None = None) -> list[np.ndarray]:
        width, height = self.config.image_size
        points: list[np.ndarray] = []
        if previous is not None:
            points.append(_clip_point(previous, self.bounds))
        if self.config.principal_prior is not None:
            points.append(_clip_point(self.config.principal_prior.center_array, self.bounds))
        points.append(_clip_point(np.array([0.5 * width, 0.5 * height], dtype=np.float64), self.bounds))
        points.append(np.array([0.5 * (self.bounds.cx[0] + self.bounds.cx[1]), 0.5 * (self.bounds.cy[0] + self.bounds.cy[1])]))
        points.extend(
            [
                np.array([self.bounds.cx[0], self.bounds.cy[0]], dtype=np.float64),
                np.array([self.bounds.cx[0], self.bounds.cy[1]], dtype=np.float64),
                np.array([self.bounds.cx[1], self.bounds.cy[0]], dtype=np.float64),
                np.array([self.bounds.cx[1], self.bounds.cy[1]], dtype=np.float64),
            ]
        )
        unique: list[np.ndarray] = []
        for point in points:
            if not any(np.linalg.norm(point - other) < 1e-9 for other in unique):
                unique.append(point)
        return unique

    def _inner_optimize(
        self,
        observations: list[FundamentalObservation],
        eta: float,
        *,
        previous: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, bool, str, float | None]:
        bounds = [self.bounds.cx, self.bounds.cy]
        best_result = None
        for start in self._initial_points(previous):
            result = optimize.minimize(
                lambda point: self._profile_data_loss(observations, eta, np.asarray(point, dtype=np.float64)),
                x0=start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.config.inner_maxiter},
            )
            if best_result is None or float(result.fun) < float(best_result.fun):
                best_result = result
        assert best_result is not None
        point = np.asarray(best_result.x, dtype=np.float64)
        profile_loss = float(best_result.fun)
        laplace_logdet = None
        if self.config.laplace_correction:
            hessian = numerical_hessian_2d(
                lambda p: self._profile_data_loss(observations, eta, np.asarray(p, dtype=np.float64)),
                point,
                step=self.config.hessian_step_px,
            )
            eigvals = np.linalg.eigvalsh(hessian)
            eigvals = np.maximum(eigvals, self.config.laplace_ridge)
            laplace_logdet = float(np.sum(np.log(eigvals)))
            profile_loss += 0.5 * laplace_logdet
        return point, profile_loss, bool(best_result.success), str(best_result.message), laplace_logdet

    def _eta_grid(self) -> np.ndarray:
        lo, hi = self.bounds.eta
        if self.config.num_focal_samples < 2:
            return np.array([np.clip(self.config.focal_prior.eta0, lo, hi)], dtype=np.float64)
        return np.linspace(lo, hi, self.config.num_focal_samples, dtype=np.float64)

    def _score_eta(
        self,
        observations: list[FundamentalObservation],
        eta: float,
        *,
        previous: np.ndarray | None = None,
    ) -> ProfileSample:
        point, profile_loss, success, message, laplace_logdet = self._inner_optimize(observations, eta, previous=previous)
        total = profile_loss + focal_prior_penalty(eta, self.config.focal_prior)
        return ProfileSample(
            eta=float(eta),
            focal=float(math.exp(eta)),
            cx=float(point[0]),
            cy=float(point[1]),
            profile_loss=float(profile_loss),
            total_loss=float(total),
            success=success,
            message=message,
            laplace_logdet=laplace_logdet,
        )

    def solve(self, observations: Iterable[FundamentalObservation | KFPPSObservation]) -> IntrinsicsEstimate:
        obs = _as_observations(observations)
        samples: list[ProfileSample] = []
        previous_point: np.ndarray | None = None
        for eta in self._eta_grid():
            sample = self._score_eta(obs, float(eta), previous=previous_point)
            samples.append(sample)
            previous_point = np.array([sample.cx, sample.cy], dtype=np.float64)
        best_sample = min(samples, key=lambda sample: sample.total_loss)

        if self.config.refine and not self.config.laplace_correction:
            lower, upper = self.bounds.scipy_bounds()
            start = np.array([best_sample.eta, best_sample.cx, best_sample.cy], dtype=np.float64)

            def objective(values: np.ndarray) -> float:
                eta, cx, cy = map(float, values)
                return (
                    self._data_loss(obs, eta, cx, cy)
                    + focal_prior_penalty(eta, self.config.focal_prior)
                    + principal_prior_penalty(cx, cy, self.config.principal_prior)
                )

            result = optimize.minimize(
                objective,
                x0=start,
                method="L-BFGS-B",
                bounds=list(zip(lower, upper)),
                options={"maxiter": self.config.inner_maxiter},
            )
            eta, cx, cy = map(float, result.x)
            best_loss = float(result.fun)
            success = bool(result.success)
            message = str(result.message)
        elif self.config.refine and self.config.laplace_correction:
            eta_lo, eta_hi = self.bounds.eta

            def objective_eta(eta_value: float) -> float:
                return self._score_eta(obs, float(eta_value), previous=np.array([best_sample.cx, best_sample.cy])).total_loss

            result = optimize.minimize_scalar(
                objective_eta,
                bounds=(eta_lo, eta_hi),
                method="bounded",
                options={"maxiter": self.config.inner_maxiter},
            )
            refined = self._score_eta(obs, float(result.x), previous=np.array([best_sample.cx, best_sample.cy]))
            eta, cx, cy = refined.eta, refined.cx, refined.cy
            best_loss = refined.total_loss
            success = bool(result.success and refined.success)
            message = f"{result.message}; inner={refined.message}"
        else:
            eta, cx, cy = best_sample.eta, best_sample.cx, best_sample.cy
            best_loss = best_sample.total_loss
            success = best_sample.success
            message = best_sample.message

        focal = float(math.exp(eta))
        return IntrinsicsEstimate(
            focal=focal,
            cx=float(cx),
            cy=float(cy),
            loss=float(best_loss),
            success=success,
            message=message,
            diagnostics={
                "samples": samples,
                "bounds": self.bounds,
                "residual_kind": self.config.residual_kind,
                "laplace_correction": self.config.laplace_correction,
                "num_observations": len(obs),
            },
        )


@dataclass(slots=True)
class KFPPSFocalProfileConfig:
    image_size: tuple[int, int]
    focal_prior: FocalPrior
    search_box: object
    principal_prior: PrincipalPointPrior | None = None
    bounds: OptimizerBounds | None = None
    focal_ratio: tuple[float, float] = (0.35, 3.0)
    num_focal_samples: int = 31
    score_kind: str = "kfpps"
    support_discriminant_threshold: float = 1e-30
    min_box_size_px: float = 0.5
    objective_tolerance: float = 1e-7
    max_nodes: int = 100000
    max_expansions: int = 0
    polish_with_f_only: bool = True
    residual_kind_for_polish: str = "manifold"

    def resolved_bounds(self) -> OptimizerBounds:
        if self.bounds is not None:
            return self.bounds
        return OptimizerBounds.from_image_size(
            self.image_size,
            self.focal_prior,
            focal_ratio=self.focal_ratio,
            principal_margin=10.0,
        )


class KFPPSFocalProfileOptimizer:
    """Run the existing fixed-f KFPPS solver across a log-focal sweep."""

    def __init__(self, config: KFPPSFocalProfileConfig) -> None:
        self.config = config
        self.bounds = config.resolved_bounds()

    def _eta_grid(self) -> np.ndarray:
        lo, hi = self.bounds.eta
        if self.config.num_focal_samples < 2:
            return np.array([np.clip(self.config.focal_prior.eta0, lo, hi)], dtype=np.float64)
        return np.linspace(lo, hi, self.config.num_focal_samples, dtype=np.float64)

    def _to_kfpps_observations(
        self,
        observations: Iterable[FundamentalObservation | object],
    ) -> list[object]:
        from kfpps.types import FundamentalMatrixObservation as KFPPSObservation

        converted = []
        for obs in observations:
            converted.append(
                KFPPSObservation(
                    F=np.asarray(obs.F, dtype=np.float64).reshape(3, 3),
                    weight=float(getattr(obs, "weight", 1.0)),
                    label=str(getattr(obs, "label", "")),
                    known_rotation_trace=getattr(obs, "known_rotation_trace", None),
                    known_rotation_angle_deg=getattr(obs, "known_rotation_angle_deg", None),
                    known_rotation_weight=float(getattr(obs, "known_rotation_weight", 1.0)),
                )
            )
        if not converted:
            raise ValueError("At least one fundamental observation is required.")
        return converted

    def _score_result(self, observations: list[object], eta: float, cx: float, cy: float, kfpps_loss: float) -> float:
        focal = math.exp(float(eta))
        K = make_pinhole_k(focal, cx, cy)
        angle_augmented_score = 0.0
        has_angle_constraints = False
        for obs in observations:
            angle_augmented_score += float(obs.weight) * essential_residual(obs.F, K, kind="manifold")
            rotation_trace = _resolved_rotation_trace(obs)
            if rotation_trace is None:
                continue
            has_angle_constraints = True
            angle_weight = float(getattr(obs, "known_rotation_weight", 1.0))
            angle_augmented_score += float(obs.weight) * angle_weight * known_rotation_trace_residual(
                obs.F,
                K,
                rotation_trace,
            )
        if self.config.score_kind == "kfpps":
            score = float(angle_augmented_score) if has_angle_constraints else float(kfpps_loss)
        elif self.config.score_kind in {"original", "manifold"}:
            score = 0.0
            for obs in observations:
                score += float(obs.weight) * essential_residual(obs.F, K, kind=self.config.score_kind)
                rotation_trace = _resolved_rotation_trace(obs)
                if rotation_trace is None:
                    continue
                angle_weight = float(getattr(obs, "known_rotation_weight", 1.0))
                score += float(obs.weight) * angle_weight * known_rotation_trace_residual(obs.F, K, rotation_trace)
        else:
            raise ValueError(f"Unknown score_kind: {self.config.score_kind!r}")
        return score + focal_prior_penalty(eta, self.config.focal_prior) + principal_prior_penalty(cx, cy, self.config.principal_prior)

    def solve(self, observations: Iterable[FundamentalObservation | object]) -> IntrinsicsEstimate:
        from kfpps.solver import CertifiedPrincipalPointSolver

        obs = self._to_kfpps_observations(observations)
        samples: list[ProfileSample] = []
        raw_results = []
        for eta in self._eta_grid():
            focal = float(math.exp(float(eta)))
            try:
                result = CertifiedPrincipalPointSolver(
                    fx=focal,
                    fy=focal,
                    image_size=self.config.image_size,
                    search_box=self.config.search_box,
                    support_discriminant_threshold=self.config.support_discriminant_threshold,
                    min_box_size_px=self.config.min_box_size_px,
                    objective_tolerance=self.config.objective_tolerance,
                    max_nodes=self.config.max_nodes,
                    max_expansions=self.config.max_expansions,
                ).solve(obs)
            except Exception as exc:
                samples.append(
                    ProfileSample(
                        eta=float(eta),
                        focal=focal,
                        cx=math.nan,
                        cy=math.nan,
                        profile_loss=math.inf,
                        total_loss=math.inf,
                        success=False,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
                raw_results.append(None)
                continue
            total = self._score_result(obs, float(eta), result.cx, result.cy, result.objective_value)
            samples.append(
                ProfileSample(
                    eta=float(eta),
                    focal=focal,
                    cx=float(result.cx),
                    cy=float(result.cy),
                    profile_loss=float(result.objective_value),
                    total_loss=float(total),
                    success=True,
                    message="ok",
                )
            )
            raw_results.append(result)

        finite_samples = [sample for sample in samples if np.isfinite(sample.total_loss)]
        if not finite_samples:
            return IntrinsicsEstimate(
                focal=math.nan,
                cx=math.nan,
                cy=math.nan,
                loss=math.inf,
                success=False,
                message="All KFPPS focal samples failed.",
                diagnostics={"samples": samples, "raw_results": raw_results},
            )
        best_sample = min(finite_samples, key=lambda sample: sample.total_loss)

        if self.config.polish_with_f_only:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                polished = FOnlyProfileOptimizer(
                    FOnlyProfileConfig(
                        image_size=self.config.image_size,
                        focal_prior=self.config.focal_prior,
                        principal_prior=self.config.principal_prior,
                        bounds=self.bounds,
                        num_focal_samples=max(5, min(15, self.config.num_focal_samples)),
                        residual_kind=self.config.residual_kind_for_polish,
                        refine=True,
                    )
                ).solve(obs)
            polished.diagnostics.update(
                {
                    "kfpps_samples": samples,
                    "kfpps_raw_results": raw_results,
                    "kfpps_best_before_polish": best_sample,
                }
            )
            return polished

        return IntrinsicsEstimate(
            focal=best_sample.focal,
            cx=best_sample.cx,
            cy=best_sample.cy,
            loss=best_sample.total_loss,
            success=best_sample.success,
            message=best_sample.message,
            diagnostics={
                "samples": samples,
                "raw_results": raw_results,
                "bounds": self.bounds,
                "score_kind": self.config.score_kind,
            },
        )
