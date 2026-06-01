#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
sys.path.insert(0, str(ROOT / "vendor"))

from kfpps.types import SearchBox
from unified_optimize import FocalPrior, FundamentalObservation, KFPPSFocalProfileConfig, KFPPSFocalProfileOptimizer
from unified_optimize.data import OptimizerBounds
from unified_optimize.geometry import estimate_fundamental_normalized_eight_point, fundamental_from_rt

from martyushev_solver import solve_calibration_from_correspondences, select_feasible_solution


MARTYUSHEV_PDF_URL = (
    "https://openaccess.thecvf.com/content_ECCV_2018/papers/"
    "Martyushev_Self-Calibration_of_Cameras_ECCV_2018_paper.pdf"
)
MARTYUSHEV_ARXIV_SOURCE_URL = "https://arxiv.org/e-print/1807.11279v1"


def info(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def try_download(url: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return {"url": url, "path": str(destination), "status": "cached"}
    try:
        info(f"download {url}")
        with urllib.request.urlopen(url) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:  # pragma: no cover - network availability is environment-specific.
        return {"url": url, "path": str(destination), "status": "failed", "error": str(exc)}
    return {"url": url, "path": str(destination), "status": "downloaded"}


def extract_tar_archive(archive_path: Path, destination: Path) -> dict[str, Any]:
    if not archive_path.exists():
        return {"archive": str(archive_path), "status": "missing"}
    if destination.exists() and any(destination.iterdir()):
        return {"archive": str(archive_path), "status": "cached", "destination": str(destination)}
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path, mode="r:*") as tar:
            tar.extractall(destination)
    except Exception as exc:  # pragma: no cover - archive format is controlled externally.
        return {
            "archive": str(archive_path),
            "status": "failed",
            "destination": str(destination),
            "error": str(exc),
        }
    return {"archive": str(archive_path), "status": "extracted", "destination": str(destination)}


def download_martyushev_artifacts(work_dir: Path) -> dict[str, Any]:
    external_root = work_dir / "external/martyushev"
    external_root.mkdir(parents=True, exist_ok=True)
    pdf_result = try_download(MARTYUSHEV_PDF_URL, external_root / "martyushev_eccv2018.pdf")
    source_archive = external_root / "martyushev_arxiv_source.tar.gz"
    source_result = try_download(MARTYUSHEV_ARXIV_SOURCE_URL, source_archive)
    extract_result = extract_tar_archive(source_archive, external_root / "arxiv_source")
    manifest = {
        "paper": "Self-Calibration of Cameras with Euclidean Image Plane in Case of Two Views and Known Relative Rotation Angle",
        "author": "Evgeniy Martyushev",
        "public_code_found": False,
        "notes": [
            "GitHub repository search returned no public repository for the paper title.",
            "This paper-local package therefore starts from the published PDF and arXiv source.",
        ],
        "downloads": {
            "pdf": pdf_result,
            "arxiv_source": source_result,
            "arxiv_source_extract": extract_result,
        },
    }
    write_json(external_root / "manifest.json", manifest)
    return manifest


def axis_angle_rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis = axis / np.linalg.norm(axis)
    angle_rad = math.radians(angle_deg)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    outer = np.outer(axis, axis)
    return cosine * np.eye(3, dtype=np.float64) + (1.0 - cosine) * outer + sine * skew


def sample_motion(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    axis = rng.normal(size=3)
    while np.linalg.norm(axis) < 1e-9:
        axis = rng.normal(size=3)
    angle_deg = float(rng.uniform(6.0, 35.0))
    rotation = axis_angle_rotation(axis, angle_deg)
    translation = rng.normal(size=3)
    while np.linalg.norm(translation) < 1e-9:
        translation = rng.normal(size=3)
    translation = translation / np.linalg.norm(translation)
    return rotation, translation, angle_deg


def build_search_box(image_size: tuple[int, int]) -> SearchBox:
    width, height = image_size
    return SearchBox(
        -0.5 * width,
        1.5 * width,
        -0.5 * height,
        1.5 * height,
    )


def run_profile_method(
    observations: list[FundamentalObservation],
    *,
    image_size: tuple[int, int],
    focal_prior_px: float,
    runtime: "ProfileRuntimeConfig",
) -> Any:
    bounds = OptimizerBounds(
        eta=(math.log(max(50.0, focal_prior_px / 2.5)), math.log(focal_prior_px * 2.5)),
        cx=(-0.5 * image_size[0], 1.5 * image_size[0]),
        cy=(-0.5 * image_size[1], 1.5 * image_size[1]),
    )
    config = KFPPSFocalProfileConfig(
        image_size=image_size,
        focal_prior=FocalPrior(focal_px=focal_prior_px, scale_log=math.log(2.5), robust=True, weight=0.0),
        search_box=build_search_box(image_size),
        principal_prior=None,
        bounds=bounds,
        num_focal_samples=runtime.num_focal_samples,
        score_kind="kfpps",
        support_discriminant_threshold=1e-30,
        min_box_size_px=runtime.min_box_size_px,
        objective_tolerance=1e-6,
        max_nodes=runtime.max_nodes,
        max_expansions=0,
        polish_with_f_only=False,
    )
    return KFPPSFocalProfileOptimizer(config).solve(observations)


@dataclass(frozen=True)
class BenchmarkCondition:
    name: str
    pair_count: int
    use_angle: bool
    method: str = "profile"  # "profile" | "martyushev_single" | "martyushev_avg"


@dataclass(frozen=True)
class ProfileRuntimeConfig:
    name: str
    num_focal_samples: int
    max_nodes: int
    min_box_size_px: float


def resolve_runtime_config(name: str) -> ProfileRuntimeConfig:
    if name == "paper":
        return ProfileRuntimeConfig(
            name="paper",
            num_focal_samples=21,
            max_nodes=20000,
            min_box_size_px=1.0,
        )
    if name == "smoke":
        return ProfileRuntimeConfig(
            name="smoke",
            num_focal_samples=9,
            max_nodes=6000,
            min_box_size_px=2.0,
        )
    raise ValueError(f"Unsupported runtime preset: {name}")


@dataclass(frozen=True)
class TrialRecord:
    method: str
    pair_count: int
    use_angle: bool
    noise_sigma: float
    trial_index: int
    focal_true: float
    focal_est: float
    focal_error_percent: float
    cx_true: float
    cy_true: float
    cx_est: float
    cy_est: float
    principal_error_px: float
    success: bool
    loss: float
    message: str


def project_points(
    points_3d: np.ndarray,
    *,
    focal: float,
    cx: float,
    cy: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 3)
    camera_points = (np.asarray(rotation, dtype=np.float64) @ points_3d.T).T + np.asarray(translation, dtype=np.float64).reshape(1, 3)
    depths = camera_points[:, 2].copy()
    image_points = np.column_stack(
        [
            focal * camera_points[:, 0] / depths + cx,
            focal * camera_points[:, 1] / depths + cy,
        ]
    )
    return image_points, depths


def sample_visible_correspondences(
    rng: np.random.Generator,
    *,
    num_points: int,
    image_size: tuple[int, int],
    focal: float,
    cx: float,
    cy: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    scene_distance: float,
    scene_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = image_size
    lower_depth = scene_distance - 0.5 * scene_depth
    upper_depth = scene_distance + 0.5 * scene_depth
    points0: list[list[float]] = []
    points1: list[list[float]] = []
    while len(points0) < num_points:
        depth = float(rng.uniform(lower_depth, upper_depth))
        u0 = float(rng.uniform(0.0, width))
        v0 = float(rng.uniform(0.0, height))
        point_3d = np.array(
            [
                (u0 - cx) / focal * depth,
                (v0 - cy) / focal * depth,
                depth,
            ],
            dtype=np.float64,
        )
        projected1, depths1 = project_points(
            point_3d[None, :],
            focal=focal,
            cx=cx,
            cy=cy,
            rotation=rotation,
            translation=translation,
        )
        if depths1[0] <= 1e-6:
            continue
        u1, v1 = map(float, projected1[0])
        if not (0.0 <= u1 < width and 0.0 <= v1 < height):
            continue
        points0.append([u0, v0])
        points1.append([u1, v1])
    return np.asarray(points0, dtype=np.float64), np.asarray(points1, dtype=np.float64)


def estimate_pairwise_fundamental(
    rng: np.random.Generator,
    *,
    num_points: int,
    image_size: tuple[int, int],
    focal: float,
    cx: float,
    cy: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    image_noise_px: float,
    scene_distance: float,
    scene_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (F, noisy_points0, noisy_points1) for one synthetic image pair."""

    points0, points1 = sample_visible_correspondences(
        rng,
        num_points=num_points,
        image_size=image_size,
        focal=focal,
        cx=cx,
        cy=cy,
        rotation=rotation,
        translation=translation,
        scene_distance=scene_distance,
        scene_depth=scene_depth,
    )
    if image_noise_px > 0.0:
        noise0 = image_noise_px * rng.standard_normal(points0.shape)
        noise1 = image_noise_px * rng.standard_normal(points1.shape)
        points0 = points0 + noise0
        points1 = points1 + noise1
    F = estimate_fundamental_normalized_eight_point(points0, points1)
    return F, points0, points1


@dataclass
class _PairData:
    F: np.ndarray
    points0: np.ndarray
    points1: np.ndarray
    angle_deg: float
    observed_angle_deg: float
    rotation_trace: float


def _generate_pairs(
    rng: np.random.Generator,
    *,
    condition: BenchmarkCondition,
    image_size: tuple[int, int],
    focal_true: float,
    cx_true: float,
    cy_true: float,
    image_noise_px: float,
    angle_noise_sigma: float,
    num_points: int,
    baseline_length: float,
    scene_distance: float,
    scene_depth: float,
) -> list[_PairData]:
    pairs: list[_PairData] = []
    for _ in range(condition.pair_count):
        rotation, translation_direction, angle_deg = sample_motion(rng)
        translation = baseline_length * translation_direction
        F, points0, points1 = estimate_pairwise_fundamental(
            rng,
            num_points=num_points,
            image_size=image_size,
            focal=focal_true,
            cx=cx_true,
            cy=cy_true,
            rotation=rotation,
            translation=translation,
            image_noise_px=image_noise_px,
            scene_distance=scene_distance,
            scene_depth=scene_depth,
        )
        observed_angle_deg = float(angle_deg * (1.0 + angle_noise_sigma * rng.normal()))
        rotation_trace = 1.0 + 2.0 * math.cos(math.radians(observed_angle_deg))
        pairs.append(
            _PairData(
                F=F,
                points0=points0,
                points1=points1,
                angle_deg=angle_deg,
                observed_angle_deg=observed_angle_deg,
                rotation_trace=rotation_trace,
            )
        )
    return pairs


def _estimate_martyushev(
    pairs: list[_PairData],
    *,
    image_size: tuple[int, int],
    average: bool,
) -> tuple[float, float, float, bool, str]:
    """Per-pair minimal solve; single-pair takes pair 0, multi-pair averages.

    This is the paper's information-usage model: solve each pair independently,
    then combine multiple pairs by averaging the feasible solutions. It is the
    point of comparison against the profile optimizer, which instead accumulates
    all pairs in one joint objective.
    """

    per_pair: list[tuple[float, float, float]] = []
    for pair in pairs:
        solutions = solve_calibration_from_correspondences(pair.points0, pair.points1, pair.rotation_trace)
        best = select_feasible_solution(solutions, image_size=image_size, center_radius_px=0.5 * image_size[0])
        if best is None and solutions:
            best = min(solutions, key=lambda s: s.residual)
        if best is not None and math.isfinite(best.focal):
            per_pair.append((best.focal, best.cx, best.cy))
        if not average and per_pair:
            break
    if not per_pair:
        return float("nan"), float("nan"), float("nan"), False, "no feasible solution"
    arr = np.asarray(per_pair, dtype=np.float64)
    if average:
        focal_est, cx_est, cy_est = (float(v) for v in np.median(arr, axis=0))
        message = f"median over {len(per_pair)} feasible pairs"
    else:
        focal_est, cx_est, cy_est = (float(v) for v in arr[0])
        message = "single-pair minimal solve"
    return focal_est, cx_est, cy_est, True, message


def run_single_trial(
    rng: np.random.Generator,
    *,
    trial_index: int,
    condition: BenchmarkCondition,
    runtime: ProfileRuntimeConfig,
    image_size: tuple[int, int],
    focal_true: float,
    cx_true: float,
    cy_true: float,
    focal_prior_px: float,
    image_noise_px: float,
    angle_noise_sigma: float,
    num_points: int,
    baseline_length: float,
    scene_distance: float,
    scene_depth: float,
) -> TrialRecord:
    pairs = _generate_pairs(
        rng,
        condition=condition,
        image_size=image_size,
        focal_true=focal_true,
        cx_true=cx_true,
        cy_true=cy_true,
        image_noise_px=image_noise_px,
        angle_noise_sigma=angle_noise_sigma,
        num_points=num_points,
        baseline_length=baseline_length,
        scene_distance=scene_distance,
        scene_depth=scene_depth,
    )

    if condition.method == "profile":
        observations: list[FundamentalObservation] = []
        for pair_index, pair in enumerate(pairs):
            kwargs: dict[str, Any] = {"F": pair.F, "weight": 1.0, "label": f"pair_{pair_index}"}
            if condition.use_angle:
                kwargs["known_rotation_angle_deg"] = pair.observed_angle_deg
                kwargs["known_rotation_weight"] = 1.0
            observations.append(FundamentalObservation(**kwargs))
        estimate = run_profile_method(
            observations,
            image_size=image_size,
            focal_prior_px=focal_prior_px,
            runtime=runtime,
        )
        focal_est = float(estimate.focal)
        cx_est = float(estimate.cx)
        cy_est = float(estimate.cy)
        success = bool(estimate.success)
        loss = float(estimate.loss)
        message = str(estimate.message)
    elif condition.method in ("martyushev_single", "martyushev_avg"):
        focal_est, cx_est, cy_est, success, message = _estimate_martyushev(
            pairs,
            image_size=image_size,
            average=(condition.method == "martyushev_avg"),
        )
        loss = float("nan")
    else:
        raise ValueError(f"Unknown method: {condition.method}")

    principal_error_px = float(math.hypot(cx_est - cx_true, cy_est - cy_true)) if math.isfinite(cx_est) else float("nan")
    focal_error_percent = (focal_est - focal_true) / focal_true * 100.0 if math.isfinite(focal_est) else float("nan")
    return TrialRecord(
        method=condition.name,
        pair_count=condition.pair_count,
        use_angle=condition.use_angle,
        noise_sigma=float(image_noise_px),
        trial_index=int(trial_index),
        focal_true=float(focal_true),
        focal_est=focal_est,
        focal_error_percent=focal_error_percent,
        cx_true=float(cx_true),
        cy_true=float(cy_true),
        cx_est=cx_est,
        cy_est=cy_est,
        principal_error_px=principal_error_px,
        success=bool(success),
        loss=loss,
        message=message,
    )


def aggregate_records(records: list[TrialRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[TrialRecord]] = {}
    for record in records:
        grouped.setdefault((record.method, record.noise_sigma), []).append(record)
    rows: list[dict[str, Any]] = []
    for (method, noise_sigma), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        focal_errors = [abs(record.focal_error_percent) for record in group]
        principal_errors = [record.principal_error_px for record in group]
        successes = [record.success for record in group]
        rows.append(
            {
                "method": method,
                "noise_sigma": noise_sigma,
                "num_trials": len(group),
                "success_rate": float(np.mean(successes)),
                "mean_abs_focal_error_percent": float(np.mean(focal_errors)),
                "median_abs_focal_error_percent": float(np.median(focal_errors)),
                "mean_principal_error_px": float(np.mean(principal_errors)),
                "median_principal_error_px": float(np.median(principal_errors)),
            }
        )
    return rows


def write_synthetic_table(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Synthetic known-angle comparison",
        "",
        "| method | noise sigma (px) | trials | success rate | mean focal err % | median focal err % | mean pp err px | median pp err px |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {method} | {noise_sigma:.4f} | {num_trials} | {success_rate:.2f} | {mean_abs_focal_error_percent:.3f} | "
            "{median_abs_focal_error_percent:.3f} | {mean_principal_error_px:.3f} | {median_principal_error_px:.3f} |".format(
                **row
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_synthetic_known_angle_benchmark(
    *,
    work_dir: Path,
    trials: int,
    noise_sigmas: list[float],
    seed: int,
    num_points: int,
    angle_noise_sigma: float,
    runtime: ProfileRuntimeConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    image_size = (1280, 720)
    focal_true = 1000.0
    cx_true = 640.0
    cy_true = 360.0
    scene_distance = 1.0
    scene_depth = 0.5
    baseline_length = 0.1
    conditions = [
        BenchmarkCondition(name="single_pair_f_only", pair_count=1, use_angle=False, method="profile"),
        BenchmarkCondition(name="single_pair_angle", pair_count=1, use_angle=True, method="profile"),
        BenchmarkCondition(name="multi_pair_f_only", pair_count=4, use_angle=False, method="profile"),
        BenchmarkCondition(name="multi_pair_angle", pair_count=4, use_angle=True, method="profile"),
        BenchmarkCondition(name="martyushev_single_pair", pair_count=1, use_angle=True, method="martyushev_single"),
        BenchmarkCondition(name="martyushev_multi_pair_avg", pair_count=4, use_angle=True, method="martyushev_avg"),
    ]
    records: list[TrialRecord] = []
    for noise_sigma in noise_sigmas:
        for trial_index in range(trials):
            focal_prior_px = 1000.0
            for condition in conditions:
                records.append(
                    run_single_trial(
                        rng,
                        trial_index=trial_index,
                        condition=condition,
                        runtime=runtime,
                        image_size=image_size,
                        focal_true=focal_true,
                        cx_true=cx_true,
                        cy_true=cy_true,
                        focal_prior_px=focal_prior_px,
                        image_noise_px=noise_sigma,
                        angle_noise_sigma=angle_noise_sigma,
                        num_points=num_points,
                        baseline_length=baseline_length,
                        scene_distance=scene_distance,
                        scene_depth=scene_depth,
                    )
                )
    summary_rows = aggregate_records(records)
    results_root = work_dir / "results/synthetic_known_angle"
    write_json(results_root / "results.json", [asdict(record) for record in records])
    write_json(results_root / "summary.json", summary_rows)
    write_synthetic_table(work_dir / "results/tables/synthetic_known_angle_summary.md", summary_rows)
    return {
        "trials": trials,
        "noise_sigmas": noise_sigmas,
        "seed": seed,
        "num_points": num_points,
        "angle_noise_sigma": angle_noise_sigma,
        "runtime": asdict(runtime),
        "summary": summary_rows,
        "results_path": str(results_root / "results.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-local KFPPS reproduction scaffold: Martyushev artifacts and synthetic known-angle benchmarks.")
    parser.add_argument("--skip-download-martyushev", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--runtime-preset", choices=["smoke", "paper"], default="smoke")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--noise-sigmas", type=float, nargs="*", default=[0.0, 0.5])
    parser.add_argument("--angle-noise-sigma", type=float, default=0.0)
    parser.add_argument("--num-points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-work", action="store_true")
    args = parser.parse_args()
    runtime = resolve_runtime_config(args.runtime_preset)

    if args.reset_work:
        reset_dir(WORK)
    else:
        WORK.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {}
    if not args.skip_download_martyushev:
        manifest["martyushev"] = download_martyushev_artifacts(WORK)

    if not args.skip_synthetic:
        manifest["synthetic_known_angle"] = run_synthetic_known_angle_benchmark(
            work_dir=WORK,
            trials=args.trials,
            noise_sigmas=list(args.noise_sigmas),
            seed=args.seed,
            num_points=args.num_points,
            angle_noise_sigma=args.angle_noise_sigma,
            runtime=runtime,
        )

    write_json(WORK / "manifest.json", manifest)
    info(f"WROTE {WORK}")


if __name__ == "__main__":
    main()
