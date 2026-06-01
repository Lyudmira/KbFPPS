"""Check inlier ratio of real SIFT matches vs GT epipolar geometry, and whether
a RANSAC-filtered F fixes the solve. Confirms the outlier hypothesis.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import cv2
import martyushev_solver as ms
import reproduce_euroc as re

mav = Path(r"E:\KFPPS_data\euroc\MH_01_easy")
sensor = re.parse_sensor_yaml(mav / "mav0" / "cam0" / "sensor.yaml")
K_gt = re.make_k(sensor["intrinsics"]); dist = sensor["distortion"]
poses = re.parse_ground_truth(mav / "mav0" / "state_groundtruth_estimate0" / "data.csv")
pose_ts = [p.timestamp for p in poses]
images = re.list_cam0_images(mav)

count = 0; i = 0
while count < 4 and i + 7 < len(images):
    j = i + 7
    R_i = re.associate_pose(poses, pose_ts, images[i][0])
    R_j = re.associate_pose(poses, pose_ts, images[j][0])
    if R_i is None or R_j is None: i += 7; continue
    angle, tau = re.relative_angle_deg(R_i, R_j)
    if angle < 5.0: i += 7; continue
    img_i = re.undistort_image(cv2.imread(str(images[i][1]), cv2.IMREAD_GRAYSCALE), K_gt, dist, cv2)
    img_j = re.undistort_image(cv2.imread(str(images[j][1]), cv2.IMREAD_GRAYSCALE), K_gt, dist, cv2)
    pts0, pts1 = re.sift_match(img_i, img_j, cv2, max_features=4000, ratio=0.8)
    if pts0.shape[0] < 8: i += 7; continue

    # RANSAC fundamental with OpenCV to estimate inlier ratio.
    F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, 1.0, 0.999)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"\n=== pair {i}->{j} angle={angle:.1f} matches={pts0.shape[0]} RANSAC inliers={inliers} ({100*inliers/pts0.shape[0]:.0f}%) ===")

    # Re-solve using ONLY RANSAC inliers.
    if mask is not None and inliers >= 8:
        m = mask.ravel().astype(bool)
        sols = ms.solve_calibration_from_correspondences(pts0[m], pts1[m], tau)
        best = ms.select_feasible_solution(sols, image_size=(752,480), center_radius_px=80)
        if best:
            print(f"  inlier-only solve: f={best.focal:.2f} cx={best.cx:.2f} cy={best.cy:.2f} (GT 458.65/367.22/248.38)")
        else:
            print(f"  inlier-only solve: {len(sols)} raw sols, none feasible")
    count += 1; i += 7
