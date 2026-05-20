from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class MartyushevCalibrationSolution:
    focal: float
    cx: float
    cy: float
    normalized_focal: float
    normalized_cx: float
    normalized_cy: float
    K: np.ndarray
    K_normalized: np.ndarray
    F_normalized: np.ndarray


def normalize_fro(matrix: np.ndarray, *, eps: float = 1e-15) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError("Cannot normalize a zero or non-finite matrix.")
    return matrix / norm


def rank2_fundamental(matrix: np.ndarray) -> np.ndarray:
    u, s, vh = np.linalg.svd(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    s[-1] = 0.0
    return normalize_fro(u @ np.diag(s) @ vh)


def homogeneous(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    return np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)


def normalize_joint_points(points0: np.ndarray, points1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points0 = np.asarray(points0, dtype=np.float64).reshape(-1, 2)
    points1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    stacked = np.vstack([points0, points1])
    centroid = stacked.mean(axis=0)
    centered = stacked - centroid
    mean_dist = float(np.mean(np.linalg.norm(centered, axis=1)))
    scale = np.sqrt(2.0) / max(mean_dist, 1e-12)
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized0 = homogeneous(points0) @ transform.T
    normalized1 = homogeneous(points1) @ transform.T
    return normalized0[:, :2], normalized1[:, :2], transform


def _design_matrix(points0: np.ndarray, points1: np.ndarray) -> np.ndarray:
    x0 = np.asarray(points0, dtype=np.float64)[:, 0]
    y0 = np.asarray(points0, dtype=np.float64)[:, 1]
    x1 = np.asarray(points1, dtype=np.float64)[:, 0]
    y1 = np.asarray(points1, dtype=np.float64)[:, 1]
    return np.column_stack(
        [
            x1 * x0,
            x1 * y0,
            x1,
            y1 * x0,
            y1 * y0,
            y1,
            x0,
            y0,
            np.ones_like(x0),
        ]
    )


def _determinant_cubic_coefficients(F0: np.ndarray, F1: np.ndarray) -> np.ndarray:
    samples = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float64)
    vandermonde = np.vander(samples, 4)
    values = np.array([np.linalg.det(F0 + value * F1) for value in samples], dtype=np.float64)
    return np.linalg.solve(vandermonde, values)


def estimate_fundamental_candidates(points0: np.ndarray, points1: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    normalized0, normalized1, transform = normalize_joint_points(points0, points1)
    design = _design_matrix(normalized0, normalized1)
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    if design.shape[0] == 7:
        F0 = vh[-1].reshape(3, 3)
        F1 = vh[-2].reshape(3, 3)
        coeffs = _determinant_cubic_coefficients(F0, F1)
        roots = np.roots(coeffs)
        candidates: list[np.ndarray] = []
        for root in roots:
            if abs(root.imag) > 1e-8:
                continue
            candidate = normalize_fro(F0 + float(root.real) * F1)
            candidates.append(candidate)
        return candidates, transform
    candidate = rank2_fundamental(vh[-1].reshape(3, 3))
    return [candidate], transform


def _build_quartic_system(F_normalized: np.ndarray, rotation_trace: float) -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, list[sp.Expr]]:
    a, b, p = sp.symbols("a b p")
    F = sp.Matrix(np.asarray(F_normalized, dtype=np.float64).reshape(3, 3).tolist())
    omega = sp.Matrix(
        [
            [a * a + p, a * b, a],
            [a * b, b * b + p, b],
            [a, b, 1],
        ]
    )
    trace_term = sp.expand((F * omega * F.T * omega).trace())
    G = (sp.Rational(1, 2) * trace_term) * F - F * omega * F.T * omega * F
    tau = sp.Float(float(rotation_trace))
    angle_constraint = sp.expand(
        sp.Rational(1, 2) * (tau * tau - 1.0) * trace_term
        + (tau + 1.0) * (omega * F * omega * F).trace()
        - tau * (omega * F).trace() ** 2
    )
    system = [sp.expand(G[0, 0]), sp.expand(G[1, 1]), sp.expand(G[2, 2]), angle_constraint]
    return a, b, p, system


def _filter_real_positive_solutions(raw_solutions: list[tuple[complex, complex, complex]]) -> list[tuple[float, float, float]]:
    filtered: list[tuple[float, float, float]] = []
    for a_value, b_value, p_value in raw_solutions:
        if any(abs(complex(value).imag) > 1e-7 for value in (a_value, b_value, p_value)):
            continue
        a_real = float(complex(a_value).real)
        b_real = float(complex(b_value).real)
        p_real = float(complex(p_value).real)
        if not np.isfinite(a_real) or not np.isfinite(b_real) or not np.isfinite(p_real):
            continue
        if p_real <= 1e-9:
            continue
        filtered.append((a_real, b_real, p_real))
    unique: list[tuple[float, float, float]] = []
    for candidate in filtered:
        if any(np.allclose(candidate, existing, atol=1e-6, rtol=1e-6) for existing in unique):
            continue
        unique.append(candidate)
    return unique


def solve_calibration_from_fundamental(
    F_normalized: np.ndarray,
    rotation_trace: float,
    transform: np.ndarray | None = None,
) -> list[MartyushevCalibrationSolution]:
    a, b, p, system = _build_quartic_system(F_normalized, rotation_trace)
    raw_solutions = sp.solve_poly_system(system, a, b, p)
    if raw_solutions is None:
        return []
    transform_inv = np.eye(3, dtype=np.float64) if transform is None else np.linalg.inv(np.asarray(transform, dtype=np.float64).reshape(3, 3))
    results: list[MartyushevCalibrationSolution] = []
    for a_value, b_value, p_value in _filter_real_positive_solutions(raw_solutions):
        normalized_focal = float(np.sqrt(p_value))
        K_normalized = np.array(
            [
                [normalized_focal, 0.0, a_value],
                [0.0, normalized_focal, b_value],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        K = transform_inv @ K_normalized
        results.append(
            MartyushevCalibrationSolution(
                focal=float(K[0, 0]),
                cx=float(K[0, 2]),
                cy=float(K[1, 2]),
                normalized_focal=normalized_focal,
                normalized_cx=float(a_value),
                normalized_cy=float(b_value),
                K=K,
                K_normalized=K_normalized,
                F_normalized=np.asarray(F_normalized, dtype=np.float64).reshape(3, 3),
            )
        )
    return results


def solve_calibration_from_correspondences(
    points0: np.ndarray,
    points1: np.ndarray,
    rotation_trace: float,
) -> list[MartyushevCalibrationSolution]:
    F_candidates, transform = estimate_fundamental_candidates(points0, points1)
    results: list[MartyushevCalibrationSolution] = []
    for F_candidate in F_candidates:
        results.extend(solve_calibration_from_fundamental(F_candidate, rotation_trace, transform=transform))
    unique: list[MartyushevCalibrationSolution] = []
    for candidate in results:
        if any(
            np.allclose(
                [candidate.focal, candidate.cx, candidate.cy],
                [existing.focal, existing.cx, existing.cy],
                atol=1e-6,
                rtol=1e-6,
            )
            for existing in unique
        ):
            continue
        unique.append(candidate)
    return unique