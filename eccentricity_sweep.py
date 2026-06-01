#!/usr/bin/env python
"""Eccentricity sweep: does angle-conditioned self-calibration survive an
off-axis principal point?

This is the decisive experiment for the unified-paper hypothesis. The principal
point is pushed from the image center outward along the diagonal; we measure how
the recovered principal point degrades for three information regimes:

  - f_only          : F-only KFPPS profile (no angle) -- expected to be
                       unidentifiable / unstable.
  - single_angle    : single-pair F + theta (Martyushev information mode).
  - multi_angle     : multi-pair F + theta (KFPPS profile accumulation).

Noise-free, the minimal/least-squares solve is exact regardless of eccentricity
(the solver works in the normalized frame), so the informative signal is the
NOISY degradation curve: does adding angle information keep the principal point
recoverable as it moves off-axis?

Selection note: we deliberately do NOT use the near-center feasibility gate
(that would filter out the true off-center solution). We select the feasible
solution closest to GT -- a DIAGNOSTIC upper bound, to be labeled as such.

Outputs JSON + a markdown table under work/results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))

import martyushev_solver as ms
from unified_optimize import FocalPrior, FundamentalObservation, KFPPSFocalProfileConfig, KFPPSFocalProfileOptimizer
from unified_optimize.data import OptimizerBounds
from kfpps.types import SearchBox
from unified_optimize.geometry import estimate_fundamental_normalized_eight_point

# __SWEEP_APPEND__

W, H = 1280, 720
FOCAL = 1000.0


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    s = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return math.cos(angle) * np.eye(3) + (1 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * s


def sample_pair_correspondences(rng, *, cx, cy, angle_deg, noise, baseline, n):
    """One synthetic pair: returns (pts0, pts1, tau) with given principal point."""
    R = axis_angle(rng.normal(size=3), math.radians(angle_deg))
    t = rng.normal(size=3)
    t = baseline * t / np.linalg.norm(t)
    tau = float(np.trace(R))
    pts0, pts1 = [], []
    tries = 0
    while len(pts0) < n and tries < n * 30:
        tries += 1
        d = rng.uniform(0.75, 1.25)
        u0, v0 = rng.uniform(0, W), rng.uniform(0, H)
        X = np.array([(u0 - cx) / FOCAL * d, (v0 - cy) / FOCAL * d, d])
        Xc = R @ X + t
        if Xc[2] <= 1e-6:
            continue
        u1, v1 = FOCAL * Xc[0] / Xc[2] + cx, FOCAL * Xc[1] / Xc[2] + cy
        if not (0 <= u1 < W and 0 <= v1 < H):
            continue
        pts0.append([u0, v0])
        pts1.append([u1, v1])
    pts0 = np.array(pts0, dtype=np.float64)
    pts1 = np.array(pts1, dtype=np.float64)
    if noise > 0 and pts0.shape[0]:
        pts0 = pts0 + noise * rng.standard_normal(pts0.shape)
        pts1 = pts1 + noise * rng.standard_normal(pts1.shape)
    return pts0, pts1, tau


def closest_to_gt(solutions, cx, cy):
    """Diagnostic selection: feasible solution nearest GT principal point."""
    feas = [s for s in solutions if s.normalized_focal > 0 and np.isfinite(s.focal)]
    if not feas:
        return None
    return min(feas, key=lambda s: math.hypot(s.cx - cx, s.cy - cy))


def estimate_martyushev(pairs, cx, cy, *, average: bool):
    per = []
    for pts0, pts1, tau in pairs:
        if pts0.shape[0] < 8:
            continue
        sols = ms.solve_calibration_from_correspondences(pts0, pts1, tau)
        best = closest_to_gt(sols, cx, cy)
        if best is not None:
            per.append((best.focal, best.cx, best.cy))
        if not average and per:
            break
    if not per:
        return None
    arr = np.array(per)
    if average:
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1])), float(np.median(arr[:, 2]))
    return float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2])


# __SWEEP_APPEND_2__


def build_search_box(image_size):
    w, h = image_size
    return SearchBox(-0.5 * w, 1.5 * w, -0.5 * h, 1.5 * h)


def estimate_profile(pairs, *, use_angle: bool, num_focal_samples: int, max_nodes: int, min_box_size_px: float):
    """KFPPS focal-profile optimizer over the pairs' fundamental matrices."""
    observations = []
    for idx, (pts0, pts1, tau) in enumerate(pairs):
        if pts0.shape[0] < 8:
            continue
        F = estimate_fundamental_normalized_eight_point(pts0, pts1)
        kwargs = {"F": F, "weight": 1.0, "label": f"pair_{idx}"}
        if use_angle:
            angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, (tau - 1.0) / 2.0))))
            kwargs["known_rotation_angle_deg"] = angle_deg
            kwargs["known_rotation_weight"] = 1.0
        observations.append(FundamentalObservation(**kwargs))
    if not observations:
        return None
    bounds = OptimizerBounds(
        eta=(math.log(FOCAL / 2.5), math.log(FOCAL * 2.5)),
        cx=(-0.5 * W, 1.5 * W),
        cy=(-0.5 * H, 1.5 * H),
    )
    config = KFPPSFocalProfileConfig(
        image_size=(W, H),
        focal_prior=FocalPrior(focal_px=FOCAL, scale_log=math.log(2.5), robust=True, weight=0.0),
        search_box=build_search_box((W, H)),
        principal_prior=None,
        bounds=bounds,
        num_focal_samples=num_focal_samples,
        score_kind="kfpps",
        support_discriminant_threshold=1e-30,
        min_box_size_px=min_box_size_px,
        objective_tolerance=1e-6,
        max_nodes=max_nodes,
        max_expansions=0,
        polish_with_f_only=False,
    )
    est = KFPPSFocalProfileOptimizer(config).solve(observations)
    return float(est.focal), float(est.cx), float(est.cy)


@dataclass
class RegimeResult:
    name: str
    eccentricity_px: float
    noise: float
    trials: int
    feasible: int
    median_pp_err: float
    median_focal_err: float


def run_sweep(*, eccentricities, noises, trials, num_pairs, angle_deg, baseline, num_points,
              num_focal_samples, max_nodes, min_box_size_px, seed):
    cx0, cy0 = W / 2.0, H / 2.0  # image center (240,360)->(640,360); diagonal offset
    regimes = ["f_only", "single_angle", "multi_angle"]
    results: list[RegimeResult] = []
    for ecc in eccentricities:
        # Push principal point diagonally off the image center.
        diag = ecc / math.sqrt(2.0)
        cx, cy = cx0 + diag, cy0 + diag
        for noise in noises:
            buckets = {r: [] for r in regimes}
            for trial in range(trials):
                rng = np.random.default_rng(seed + trial * 100003 + int(ecc) * 7 + int(noise * 1000))
                pairs = [sample_pair_correspondences(rng, cx=cx, cy=cy, angle_deg=angle_deg,
                                                     noise=noise, baseline=baseline, n=num_points)
                         for _ in range(num_pairs)]
                # f_only: single-pair profile, no angle.
                est = estimate_profile(pairs[:1], use_angle=False,
                                       num_focal_samples=num_focal_samples, max_nodes=max_nodes,
                                       min_box_size_px=min_box_size_px)
                if est:
                    buckets["f_only"].append((math.hypot(est[1] - cx, est[2] - cy), abs(est[0] - FOCAL)))
                # single_angle: single-pair Martyushev.
                est = estimate_martyushev(pairs[:1], cx, cy, average=False)
                if est:
                    buckets["single_angle"].append((math.hypot(est[1] - cx, est[2] - cy), abs(est[0] - FOCAL)))
                # multi_angle: multi-pair KFPPS profile with angle.
                est = estimate_profile(pairs, use_angle=True,
                                       num_focal_samples=num_focal_samples, max_nodes=max_nodes,
                                       min_box_size_px=min_box_size_px)
                if est:
                    buckets["multi_angle"].append((math.hypot(est[1] - cx, est[2] - cy), abs(est[0] - FOCAL)))
            for r in regimes:
                ok = buckets[r]
                if ok:
                    pp = float(np.median([e[0] for e in ok]))
                    fe = float(np.median([e[1] for e in ok]))
                else:
                    pp = fe = float("nan")
                results.append(RegimeResult(r, float(ecc), float(noise), trials, len(ok), pp, fe))
                print(f"ecc={ecc:5.0f}px noise={noise:.2f} {r:13s} feasible={len(ok):3d}/{trials} "
                      f"pp_err={pp:8.2f} focal_err={fe:8.2f}", flush=True)
    return results


# __SWEEP_APPEND_3__


def write_outputs(results, work_dir: Path, meta: dict):
    out_dir = work_dir / "results" / "eccentricity_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [vars(r) for r in results]
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2) + "\n", encoding="utf-8")

    # Markdown pivot: pp error by regime x eccentricity, per noise level.
    lines = ["# Eccentricity sweep: principal-point error (px)", ""]
    lines.append("Diagnostic selection (feasible solution nearest GT). "
                 "Principal point pushed diagonally off image center.")
    lines.append("")
    noises = sorted({r.noise for r in results})
    eccs = sorted({r.eccentricity_px for r in results})
    regimes = ["f_only", "single_angle", "multi_angle"]
    for noise in noises:
        lines.append(f"## noise sigma = {noise:.2f} px")
        lines.append("")
        header = "| regime | " + " | ".join(f"ecc={int(e)}px" for e in eccs) + " |"
        sep = "| --- | " + " | ".join("---:" for _ in eccs) + " |"
        lines.append(header)
        lines.append(sep)
        for reg in regimes:
            cells = []
            for e in eccs:
                match = [r for r in results if r.eccentricity_px == e and r.noise == noise and r.name == reg]
                if match and math.isfinite(match[0].median_pp_err):
                    cells.append(f"{match[0].median_pp_err:.2f}")
                else:
                    cells.append("--")
            lines.append(f"| {reg} | " + " | ".join(cells) + " |")
        lines.append("")
    (work_dir / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (work_dir / "results" / "tables" / "eccentricity_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWROTE {out_dir / 'results.json'}")
    print(f"WROTE {work_dir / 'results' / 'tables' / 'eccentricity_sweep.md'}")


def main():
    parser = argparse.ArgumentParser(description="Principal-point eccentricity sweep across motion-information regimes.")
    parser.add_argument("--eccentricities", type=float, nargs="*", default=[0, 60, 120, 180, 240])
    parser.add_argument("--noises", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--num-pairs", type=int, default=4)
    parser.add_argument("--angle-deg", type=float, default=20.0)
    parser.add_argument("--baseline", type=float, default=0.1)
    parser.add_argument("--num-points", type=int, default=200)
    parser.add_argument("--num-focal-samples", type=int, default=15)
    parser.add_argument("--max-nodes", type=int, default=8000)
    parser.add_argument("--min-box-size-px", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    t0 = time.time()
    results = run_sweep(
        eccentricities=args.eccentricities, noises=args.noises, trials=args.trials,
        num_pairs=args.num_pairs, angle_deg=args.angle_deg, baseline=args.baseline,
        num_points=args.num_points, num_focal_samples=args.num_focal_samples,
        max_nodes=args.max_nodes, min_box_size_px=args.min_box_size_px, seed=args.seed,
    )
    meta = {
        "eccentricities_px": args.eccentricities, "noises_px": args.noises, "trials": args.trials,
        "num_pairs": args.num_pairs, "angle_deg": args.angle_deg, "baseline": args.baseline,
        "num_points": args.num_points, "focal": FOCAL, "image_size": [W, H],
        "selection": "feasible-nearest-GT (diagnostic upper bound)",
        "elapsed_sec": time.time() - t0,
    }
    write_outputs(results, ROOT / "work", meta)
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()



