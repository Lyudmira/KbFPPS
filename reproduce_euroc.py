#!/usr/bin/env python
"""Paper-local EuRoC real-data reproduction for the Martyushev solver.

Mirrors the real-data protocol of Martyushev ECCV 2018 (Section 5), but takes
the relative rotation angle tau from the EuRoC ground-truth trajectory instead
of integrating the gyroscope. This is a deliberate, cleaner choice: our claim
needs the information "F + tau", not a particular tau source, and the Leica/
Vicon ground truth is far cleaner than gyro integration. The IMU-integrated tau
variant can be added later as a faithful-to-original control.

Pipeline:
  1. Use a local EuRoC MH_01_easy (ASL format) directory (--euroc-root), or the
     MH_01_easy.zip we range-extracted (--euroc-zip).
  2. Parse cam0/sensor.yaml: pinhole-radtan intrinsics + distortion.
  3. Parse state_groundtruth_estimate0/data.csv: timestamped body pose (p, q).
  4. Associate each cam0 image timestamp with the nearest GT pose.
  5. Undistort images with the GT intrinsics/distortion (paper assumes pinhole).
  6. Form image pairs with a stride; tau = trace of the relative body rotation
     (camera relative rotation has the same trace; T_BC cancels under similarity).
     Discard pairs with relative angle < 5 deg (paper).
  7. SIFT ratio-test matches per pair; run the Martyushev solver; keep the
     near-center feasible solution (paper's principal-point gate).
  8. Aggregate K over the sequence; report relative error vs GT (paper ~0.6%).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from martyushev_solver import solve_calibration_from_correspondences, select_feasible_solution

# __EUROC_APPEND__


def ensure_extracted(euroc_zip: Path, dest_root: Path) -> Path:
    """Extract MH_01_easy.zip (ASL format) if needed; return the mav0 parent dir."""

    # If already extracted, find the mav0 directory.
    existing = list(dest_root.rglob("mav0"))
    if existing:
        return existing[0].parent
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(euroc_zip) as zf:
        zf.extractall(dest_root)
    found = list(dest_root.rglob("mav0"))
    if not found:
        raise FileNotFoundError(f"No mav0/ directory found after extracting {euroc_zip}")
    return found[0].parent


def parse_sensor_yaml(path: Path) -> dict[str, Any]:
    """Minimal parser for EuRoC cam0/sensor.yaml (avoids a yaml dependency).

    Extracts intrinsics [fu, fv, cu, cv] and distortion_coefficients [k1,k2,p1,p2].
    """

    text = path.read_text(encoding="utf-8")

    def bracket_list(key: str) -> list[float]:
        idx = text.find(key)
        if idx < 0:
            raise KeyError(f"{key} not found in {path}")
        lo = text.find("[", idx)
        hi = text.find("]", lo)
        body = text[lo + 1:hi]
        return [float(v) for v in body.replace("\n", " ").split(",")]

    intrinsics = bracket_list("intrinsics")
    distortion = bracket_list("distortion_coefficients")
    model = "radtan"
    if "distortion_model" in text:
        after = text.split("distortion_model", 1)[1]
        model = after.split(":", 1)[1].strip().split()[0]
    return {"intrinsics": intrinsics, "distortion": distortion, "distortion_model": model}


def quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


@dataclass
class PoseSample:
    timestamp: int
    rotation: np.ndarray  # body-to-world R_WS


def parse_ground_truth(path: Path) -> list[PoseSample]:
    """Parse state_groundtruth_estimate0/data.csv.

    Columns: timestamp[ns], p_RS_R x,y,z, q_RS_R w,x,y,z, (v, b_w, b_a ...).
    We only need timestamp and quaternion (rotation) for tau.
    """

    samples: list[PoseSample] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                ts = int(row[0])
                qw, qx, qy, qz = (float(row[4]), float(row[5]), float(row[6]), float(row[7]))
            except (ValueError, IndexError):
                continue
            samples.append(PoseSample(ts, quaternion_to_rotation(qw, qx, qy, qz)))
    samples.sort(key=lambda s: s.timestamp)
    return samples


def list_cam0_images(mav_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted (timestamp_ns, image_path) for cam0."""

    data_dir = mav_dir / "mav0" / "cam0" / "data"
    images: list[tuple[int, Path]] = []
    for p in data_dir.glob("*.png"):
        try:
            images.append((int(p.stem), p))
        except ValueError:
            continue
    images.sort(key=lambda x: x[0])
    return images


def associate_pose(poses: list[PoseSample], timestamps: list[int], ts: int, max_dt_ns: int = 20_000_000) -> np.ndarray | None:
    """Nearest-timestamp GT rotation for an image timestamp (default tol 20 ms)."""

    idx = bisect.bisect_left(timestamps, ts)
    best = None
    for cand in (idx - 1, idx):
        if 0 <= cand < len(poses):
            dt = abs(poses[cand].timestamp - ts)
            if best is None or dt < best[0]:
                best = (dt, poses[cand].rotation)
    if best is None or best[0] > max_dt_ns:
        return None
    return best[1]


# __EUROC_APPEND_2__


def make_k(intrinsics: list[float]) -> np.ndarray:
    fu, fv, cu, cv = intrinsics
    return np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]], dtype=np.float64)


def undistort_image(image: np.ndarray, K: np.ndarray, dist: list[float], cv2_mod: Any) -> np.ndarray:
    """Undistort with GT intrinsics/distortion, keeping the same K (paper step)."""

    d = np.array(dist + [0.0] * (5 - len(dist)), dtype=np.float64)[:5]
    return cv2_mod.undistort(image, K, d, None, K)


def sift_match(image0: np.ndarray, image1: np.ndarray, cv2_mod: Any, *, max_features: int, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """SIFT + ratio-test matches (same recipe as PCCC's detect_sift_matches)."""

    if hasattr(cv2_mod, "SIFT_create"):
        detector = cv2_mod.SIFT_create(nfeatures=max_features)
        norm = cv2_mod.NORM_L2
    else:
        detector = cv2_mod.ORB_create(nfeatures=max_features)
        norm = cv2_mod.NORM_HAMMING
    kp0, desc0 = detector.detectAndCompute(image0, None)
    kp1, desc1 = detector.detectAndCompute(image1, None)
    if desc0 is None or desc1 is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    matcher = cv2_mod.BFMatcher(norm)
    good = []
    for pair in matcher.knnMatch(desc0, desc1, k=2):
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    pts0 = np.array([kp0[m.queryIdx].pt for m in good], dtype=np.float64)
    pts1 = np.array([kp1[m.trainIdx].pt for m in good], dtype=np.float64)
    return pts0, pts1


def ransac_filter(pts0: np.ndarray, pts1: np.ndarray, cv2_mod: Any, *, threshold_px: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Reject outlier matches with a RANSAC fundamental estimate.

    Real SIFT matches on EuRoC carry ~30% outliers (verified in
    debug/check_inliers.py). The minimal solver has zero outlier tolerance, so a
    single bad match destroys the 8-point F and the downstream polynomials. This
    robustifies the correspondence set; the synthetic benchmark (clean matches)
    is unaffected since it never calls this.
    """

    if pts0.shape[0] < 8:
        return pts0, pts1
    F, mask = cv2_mod.findFundamentalMat(pts0, pts1, cv2_mod.FM_RANSAC, threshold_px, 0.999)
    if mask is None:
        return pts0, pts1
    inliers = mask.ravel().astype(bool)
    return pts0[inliers], pts1[inliers]


def relative_angle_deg(R_i: np.ndarray, R_j: np.ndarray) -> tuple[float, float]:
    """Relative rotation angle and tau=trace between two body orientations.

    The camera-to-body extrinsic T_BC is constant, so the camera relative
    rotation R_C = R_BC^T R_S R_BC is a similarity transform of the body
    relative rotation R_S; trace (hence tau and the angle) is invariant. So we
    can use body rotations directly without knowing T_BC.
    """

    R_rel = R_j @ R_i.T
    tau = float(np.trace(R_rel))
    cos_theta = max(-1.0, min(1.0, (tau - 1.0) / 2.0))
    return math.degrees(math.acos(cos_theta)), tau


# __EUROC_APPEND_3__


@dataclass
class PairResult:
    idx_i: int
    idx_j: int
    angle_deg: float
    num_matches: int
    focal: float
    cx: float
    cy: float
    success: bool


def run_euroc(
    mav_dir: Path,
    *,
    stride: int,
    max_pairs: int,
    min_angle_deg: float,
    max_features: int,
    ratio: float,
    center_radius_px: float,
    ransac_threshold_px: float = 1.0,
) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV required: pip install opencv-python") from exc

    sensor = parse_sensor_yaml(mav_dir / "mav0" / "cam0" / "sensor.yaml")
    K_gt = make_k(sensor["intrinsics"])
    dist = sensor["distortion"]
    print(f"GT K (from sensor.yaml):\n{K_gt}")
    print(f"distortion ({sensor['distortion_model']}): {dist}")

    poses = parse_ground_truth(mav_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv")
    pose_ts = [p.timestamp for p in poses]
    images = list_cam0_images(mav_dir)
    print(f"loaded {len(poses)} GT poses, {len(images)} cam0 images")
    if not poses or not images:
        raise RuntimeError("Missing GT poses or images.")

    # Cache undistorted images lazily.
    undistort_cache: dict[int, np.ndarray] = {}

    def get_undistorted(index: int) -> np.ndarray | None:
        if index in undistort_cache:
            return undistort_cache[index]
        ts, path = images[index]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        u = undistort_image(img, K_gt, dist, cv2)
        undistort_cache[index] = u
        return u

    results: list[PairResult] = []
    i = 0
    while i + stride < len(images) and len(results) < max_pairs:
        j = i + stride
        ts_i, _ = images[i]
        ts_j, _ = images[j]
        R_i = associate_pose(poses, pose_ts, ts_i)
        R_j = associate_pose(poses, pose_ts, ts_j)
        if R_i is None or R_j is None:
            i += stride
            continue
        angle, tau = relative_angle_deg(R_i, R_j)
        if angle < min_angle_deg:
            i += stride
            continue
        img_i = get_undistorted(i)
        img_j = get_undistorted(j)
        if img_i is None or img_j is None:
            i += stride
            continue
        pts0, pts1 = sift_match(img_i, img_j, cv2, max_features=max_features, ratio=ratio)
        pts0, pts1 = ransac_filter(pts0, pts1, cv2, threshold_px=ransac_threshold_px)
        if pts0.shape[0] < 8:
            i += stride
            continue
        solutions = solve_calibration_from_correspondences(pts0, pts1, tau)
        h, w = img_i.shape[:2]
        best = select_feasible_solution(solutions, image_size=(w, h), center_radius_px=center_radius_px)
        if best is None:
            results.append(PairResult(i, j, angle, pts0.shape[0], math.nan, math.nan, math.nan, False))
        else:
            results.append(PairResult(i, j, angle, pts0.shape[0], best.focal, best.cx, best.cy, True))
        last = results[-1]
        print(f"pair {i:5d}->{j:5d} angle={angle:5.1f} matches={pts0.shape[0]:4d} "
              f"{'OK f=%.2f cx=%.2f cy=%.2f' % (last.focal, last.cx, last.cy) if last.success else 'no feasible sol'}",
              flush=True)
        i += stride

    return summarize(results, K_gt)


# __EUROC_APPEND_4__


def summarize(results: list[PairResult], K_gt: np.ndarray) -> dict[str, Any]:
    ok = [r for r in results if r.success and math.isfinite(r.focal)]
    summary: dict[str, Any] = {
        "num_pairs": len(results),
        "num_feasible": len(ok),
        "K_gt": K_gt.tolist(),
    }
    if not ok:
        summary["note"] = "no feasible solutions"
        return summary

    focals = np.array([r.focal for r in ok])
    cxs = np.array([r.cx for r in ok])
    cys = np.array([r.cy for r in ok])
    K_med = np.array(
        [[float(np.median(focals)), 0.0, float(np.median(cxs))],
         [0.0, float(np.median(focals)), float(np.median(cys))],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rel_err = float(np.linalg.norm(K_med - K_gt) / np.linalg.norm(K_gt))
    summary.update(
        {
            "K_median": K_med.tolist(),
            "focal_median": float(np.median(focals)),
            "cx_median": float(np.median(cxs)),
            "cy_median": float(np.median(cys)),
            "relative_K_error": rel_err,
            "relative_K_error_percent": rel_err * 100.0,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="EuRoC MH_01_easy real-data reproduction for the Martyushev solver.")
    parser.add_argument("--euroc-root", type=Path, default=None, help="Directory containing mav0/ (skips extraction).")
    parser.add_argument("--euroc-zip", type=Path, default=Path(r"E:\KFPPS_data\euroc\MH_01_easy.zip"))
    parser.add_argument("--extract-dest", type=Path, default=Path(r"E:\KFPPS_data\euroc\MH_01_easy"))
    parser.add_argument("--stride", type=int, default=7, help="Frame gap between paired images.")
    parser.add_argument("--max-pairs", type=int, default=68, help="Paper used 68 pairs.")
    parser.add_argument("--min-angle-deg", type=float, default=5.0, help="Discard near-pure-translation pairs (paper).")
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--ratio-test", type=float, default=0.8)
    parser.add_argument("--center-radius-px", type=float, default=50.0, help="Principal-point feasibility gate (paper).")
    parser.add_argument("--ransac-threshold-px", type=float, default=1.0, help="RANSAC fundamental inlier threshold for match cleanup.")
    args = parser.parse_args()

    if args.euroc_root is not None:
        mav_dir = args.euroc_root
    else:
        mav_dir = ensure_extracted(args.euroc_zip, args.extract_dest)
    print(f"EuRoC mav dir: {mav_dir}")

    summary = run_euroc(
        mav_dir,
        stride=args.stride,
        max_pairs=args.max_pairs,
        min_angle_deg=args.min_angle_deg,
        max_features=args.max_features,
        ratio=args.ratio_test,
        center_radius_px=args.center_radius_px,
        ransac_threshold_px=args.ransac_threshold_px,
    )

    print("\n=== EuRoC MH_01_easy summary ===")
    print(f"pairs attempted : {summary['num_pairs']}")
    print(f"feasible solves : {summary['num_feasible']}")
    if "K_median" in summary:
        print(f"median focal    : {summary['focal_median']:.3f}  (GT fu={summary['K_gt'][0][0]:.3f}, fv={summary['K_gt'][1][1]:.3f})")
        print(f"median cx, cy   : {summary['cx_median']:.3f}, {summary['cy_median']:.3f}  (GT {summary['K_gt'][0][2]:.3f}, {summary['K_gt'][1][2]:.3f})")
        print(f"relative K error: {summary['relative_K_error_percent']:.3f} %   (paper ~0.6%)")


if __name__ == "__main__":
    main()




