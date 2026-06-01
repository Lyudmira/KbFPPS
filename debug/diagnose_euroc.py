"""Diagnose why the Martyushev solver returns garbage on real EuRoC pairs.
Dump ALL raw solutions for a few pairs, before feasibility selection.
"""

from __future__ import annotations

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
K_gt = re.make_k(sensor["intrinsics"])
dist = sensor["distortion"]
poses = re.parse_ground_truth(mav / "mav0" / "state_groundtruth_estimate0" / "data.csv")
pose_ts = [p.timestamp for p in poses]
images = re.list_cam0_images(mav)

print(f"GT K: fu={K_gt[0,0]:.2f} cx={K_gt[0,2]:.2f} cy={K_gt[1,2]:.2f}, image 752x480, center=(376,240)")

# Look at a few pairs with decent angle.
count = 0
i = 0
while count < 4 and i + 7 < len(images):
    j = i + 7
    R_i = re.associate_pose(poses, pose_ts, images[i][0])
    R_j = re.associate_pose(poses, pose_ts, images[j][0])
    if R_i is None or R_j is None:
        i += 7; continue
    angle, tau = re.relative_angle_deg(R_i, R_j)
    if angle < 5.0:
        i += 7; continue
    img_i = re.undistort_image(cv2.imread(str(images[i][1]), cv2.IMREAD_GRAYSCALE), K_gt, dist, cv2)
    img_j = re.undistort_image(cv2.imread(str(images[j][1]), cv2.IMREAD_GRAYSCALE), K_gt, dist, cv2)
    pts0, pts1 = re.sift_match(img_i, img_j, cv2, max_features=4000, ratio=0.8)
    if pts0.shape[0] < 8:
        i += 7; continue
    print(f"\n=== pair {i}->{j} angle={angle:.1f} tau={tau:.4f} matches={pts0.shape[0]} ===")
    sols = ms.solve_calibration_from_correspondences(pts0, pts1, tau)
    print(f"  {len(sols)} raw solutions:")
    for s in sols:
        print(f"    f={s.focal:10.2f} cx={s.cx:9.2f} cy={s.cy:9.2f} p_norm={s.normalized_focal:.3f} resid={s.residual:.3e}")
    count += 1
    i += 7
