from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


def make_pinhole_k(focal: float, cx: float, cy: float, fy: float | None = None) -> np.ndarray:
    fx = float(focal)
    fy_value = fx if fy is None else float(fy)
    return np.array(
        [
            [fx, 0.0, float(cx)],
            [0.0, fy_value, float(cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def normalize_fro(matrix: np.ndarray, *, eps: float = 1e-15) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError("Cannot normalize a zero or non-finite matrix.")
    return matrix / norm


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def rank2_fundamental(matrix: np.ndarray) -> np.ndarray:
    u, s, vh = np.linalg.svd(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    s[-1] = 0.0
    return normalize_fro(u @ np.diag(s) @ vh)


def essential_from_fundamental(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    return np.asarray(K, dtype=np.float64).T @ np.asarray(F, dtype=np.float64).reshape(3, 3) @ np.asarray(K, dtype=np.float64)


def essential_singular_values(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    return np.linalg.svd(essential_from_fundamental(F, K), compute_uv=False)


def original_essential_constraint(F: np.ndarray, K: np.ndarray, *, eps: float = 1e-15) -> float:
    """The first simple residual we discussed.

    It has the right zero set for an essential matrix: s1=s2 and s3=0.
    """

    s1, s2, s3 = essential_singular_values(F, K)
    denom = s1 * s1 + s2 * s2
    return float(((s1 - s2) ** 2 + s3 * s3) / max(denom, eps))


def essential_manifold_distance(F: np.ndarray, K: np.ndarray, *, eps: float = 1e-15) -> float:
    """Scale-normalized Frobenius distance to the essential singular-value pattern."""

    s1, s2, s3 = essential_singular_values(F, K)
    denom = s1 * s1 + s2 * s2 + s3 * s3
    return float((0.5 * (s1 - s2) ** 2 + s3 * s3) / max(denom, eps))


def essential_residual(
    F: np.ndarray,
    K: np.ndarray,
    *,
    kind: str = "manifold",
) -> float:
    if kind == "manifold":
        return essential_manifold_distance(F, K)
    if kind == "original":
        return original_essential_constraint(F, K)
    raise ValueError(f"Unknown essential residual kind: {kind!r}")


def known_rotation_trace_residual(
    F: np.ndarray,
    K: np.ndarray,
    rotation_trace: float,
    *,
    eps: float = 1e-15,
) -> float:
    E = essential_from_fundamental(F, K)
    tau = float(rotation_trace)
    trace_eet = float(np.trace(E @ E.T))
    trace_e2 = np.trace(E @ E)
    trace_e = np.trace(E)
    numerator = 0.5 * (tau * tau - 1.0) * trace_eet + (tau + 1.0) * trace_e2 - tau * trace_e * trace_e
    return float(np.abs(numerator) ** 2 / max(trace_eet * trace_eet, eps))


def homogeneous(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    return np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)


def sampson_signed_residuals(points0: np.ndarray, points1: np.ndarray, F: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x0 = homogeneous(points0)
    x1 = homogeneous(points1)
    F = np.asarray(F, dtype=np.float64).reshape(3, 3)
    Fx0 = x0 @ F.T
    Ftx1 = x1 @ F
    numerator = np.sum(x1 * Fx0, axis=1)
    denominator = Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2
    return numerator / np.sqrt(np.maximum(denominator, eps))


def sampson_rms(points0: np.ndarray, points1: np.ndarray, F: np.ndarray) -> float:
    residuals = sampson_signed_residuals(points0, points1, F)
    return float(np.sqrt(np.mean(residuals * residuals)))


def normalize_points_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid
    mean_dist = float(np.mean(np.linalg.norm(centered, axis=1)))
    scale = math.sqrt(2.0) / max(mean_dist, 1e-12)
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized = homogeneous(points) @ transform.T
    return normalized[:, :2], transform


def estimate_fundamental_normalized_eight_point(points0: np.ndarray, points1: np.ndarray) -> np.ndarray:
    points0 = np.asarray(points0, dtype=np.float64)
    points1 = np.asarray(points1, dtype=np.float64)
    if points0.shape[0] < 8 or points1.shape[0] < 8:
        raise ValueError("At least eight correspondences are required.")
    p0, t0 = normalize_points_2d(points0)
    p1, t1 = normalize_points_2d(points1)
    x0, y0 = p0[:, 0], p0[:, 1]
    x1, y1 = p1[:, 0], p1[:, 1]
    design = np.column_stack(
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
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    F_norm = vh[-1].reshape(3, 3)
    F_rank2 = rank2_fundamental(F_norm)
    F = t1.T @ F_rank2 @ t0
    return normalize_fro(F)


def essential_motion_candidates(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    u, _, vh = np.linalg.svd(np.asarray(E, dtype=np.float64).reshape(3, 3))
    if np.linalg.det(u @ vh) < 0.0:
        vh = -vh
    w = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotations = [u @ w @ vh, u @ w.T @ vh]
    translation = u[:, 2]
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for rotation in rotations:
        if np.linalg.det(rotation) < 0.0:
            rotation = -rotation
        candidates.append((rotation, translation.copy()))
        candidates.append((rotation, -translation.copy()))
    return candidates


def fundamental_from_rt(K: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    k_inv = np.linalg.inv(np.asarray(K, dtype=np.float64))
    t = np.asarray(translation, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(t))
    if not np.isfinite(norm) or norm <= 1e-12:
        t = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        t = t / norm
    F = k_inv.T @ skew(t) @ np.asarray(rotation, dtype=np.float64).reshape(3, 3) @ k_inv
    return normalize_fro(F)


def rotation_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_matrix()


def rotvec_from_rotation(rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(rotation, dtype=np.float64).reshape(3, 3)).as_rotvec()


def numerical_hessian_2d(func, point: np.ndarray, *, step: float = 1e-2) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64).reshape(2)
    hessian = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            ei = np.zeros(2, dtype=np.float64)
            ej = np.zeros(2, dtype=np.float64)
            ei[i] = step
            ej[j] = step
            hessian[i, j] = (
                func(point + ei + ej)
                - func(point + ei - ej)
                - func(point - ei + ej)
                + func(point - ei - ej)
            ) / (4.0 * step * step)
    return 0.5 * (hessian + hessian.T)
