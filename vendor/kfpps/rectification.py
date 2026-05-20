from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geometry import make_calibration_matrix


def principal_point_homography(
    *,
    fx: float,
    fy: float,
    source_cx: float,
    source_cy: float,
    target_cx: float,
    target_cy: float,
) -> np.ndarray:
    source_k = make_calibration_matrix(fx, fy, source_cx, source_cy)
    target_k = make_calibration_matrix(fx, fy, target_cx, target_cy)
    return source_k @ np.linalg.inv(target_k)


def warp_image_to_intrinsics(
    image: np.ndarray,
    *,
    fx: float,
    fy: float,
    source_cx: float,
    source_cy: float,
    target_cx: float,
    target_cy: float,
    output_shape: tuple[int, int] | None = None,
    order: int = 1,
    mode: str = "nearest",
) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError("Expected image with shape [H, W] or [H, W, C].")
    if output_shape is None:
        output_height, output_width = image.shape[:2]
    else:
        output_height, output_width = output_shape
    homography = principal_point_homography(
        fx=fx,
        fy=fy,
        source_cx=source_cx,
        source_cy=source_cy,
        target_cx=target_cx,
        target_cy=target_cy,
    )
    yy, xx = np.meshgrid(
        np.arange(output_height, dtype=np.float64),
        np.arange(output_width, dtype=np.float64),
        indexing="ij",
    )
    ones = np.ones_like(xx)
    dest_h = np.stack([xx, yy, ones], axis=0).reshape(3, -1)
    source_h = homography @ dest_h
    source_x = (source_h[0] / source_h[2]).reshape(output_height, output_width)
    source_y = (source_h[1] / source_h[2]).reshape(output_height, output_width)
    coordinates = np.stack([source_y, source_x], axis=0)

    if image.ndim == 2:
        return ndimage.map_coordinates(image, coordinates, order=order, mode=mode)

    channels = [
        ndimage.map_coordinates(image[..., channel], coordinates, order=order, mode=mode)
        for channel in range(image.shape[2])
    ]
    return np.stack(channels, axis=-1)
