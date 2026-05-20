from __future__ import annotations

import numpy as np


def normalize_fundamental(F: np.ndarray) -> np.ndarray:
    matrix = np.asarray(F, dtype=np.float64).reshape(3, 3)
    fro_norm = float(np.linalg.norm(matrix, ord="fro"))
    if not np.isfinite(fro_norm) or fro_norm <= 0.0:
        raise ValueError("Fundamental matrix must have finite non-zero Frobenius norm.")
    return matrix / fro_norm


def normalize_vector(vec: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= eps:
        raise ValueError("Vector norm is too small to normalize.")
    return vector / norm


def skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vec, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def left_epipole_from_fundamental(F: np.ndarray) -> np.ndarray:
    _, _, vh = np.linalg.svd(np.asarray(F, dtype=np.float64).T)
    return normalize_vector(vh[-1])


def householder_orthobasis_perp(vec: np.ndarray) -> np.ndarray:
    unit = normalize_vector(vec)
    target_index = int(np.argmin(np.abs(unit)))
    target = np.zeros(3, dtype=np.float64)
    target[target_index] = 1.0
    sign = 1.0 if unit[target_index] >= 0.0 else -1.0
    reflector_vec = unit - sign * target
    reflector_norm = float(np.linalg.norm(reflector_vec))
    if reflector_norm <= 1e-12:
        basis = np.eye(3, dtype=np.float64)
        return np.delete(basis, target_index, axis=1)
    reflector_vec /= reflector_norm
    householder = np.eye(3, dtype=np.float64) - 2.0 * np.outer(reflector_vec, reflector_vec)
    basis = np.delete(householder, target_index, axis=1)
    return basis


def make_calibration_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fundamental_from_motion(
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    normalize: bool = True,
) -> np.ndarray:
    K = make_calibration_matrix(fx, fy, cx, cy)
    K_inv = np.linalg.inv(K)
    matrix = K_inv.T @ skew(translation) @ np.asarray(rotation, dtype=np.float64) @ K_inv
    return normalize_fundamental(matrix) if normalize else matrix
