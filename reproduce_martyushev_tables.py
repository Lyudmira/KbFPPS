#!/usr/bin/env python
"""Reproduce the synthetic experiment tables of Martyushev ECCV 2018, with a
side-by-side comparison of three methods on identical data:

  - martyushev        : single-pair non-iterative minimal solver (paper method).
  - martyushev_multi  : per-pair minimal solve + robust median fusion.
  - kfpps_profile     : our KFPPS focal-profile optimizer (joint accumulation).

Experiments (aligned to the paper's figures):
  E1  numerical accuracy   (Fig. numerical_error): noise-free K relative error.
  E2  number of solutions  (Fig. nsols)          : real / feasible solution counts.
  E3a image-noise sweep    (Fig. error_K left)   : K rel. error vs sigma in [0,1] px.
  E3b point-count sweep    (Fig. error_K right)  : K rel. error vs N.
  E3c angle-noise sweep                          : K rel. error vs angle noise sigma.
  E4  focal + miscalibration (Fig. error_f)      : |f-1000|/1000 vs noise, alpha=0.1.
  E6  speed                                      : mean solve time per method.

The paper's pose-vs-5pt/6pt comparison (Fig. error_RT) is intentionally out of
scope: it benchmarks pose estimation, not internal self-calibration.

Metric (paper Eq. for numerical accuracy):
  rel_K_error = ||K_est - K_gt||_F / ||K_gt||_F

Trial budgets are per-method: the millisecond-scale Martyushev solver runs many
trials; the slow KFPPS profile optimizer runs fewer (its accumulated estimate
has low variance, so it does not need the same sample count).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))

import martyushev_solver as ms
from kfpps.types import SearchBox
from unified_optimize import FocalPrior, FundamentalObservation, KFPPSFocalProfileConfig, KFPPSFocalProfileOptimizer
from unified_optimize.data import OptimizerBounds
from unified_optimize.geometry import estimate_fundamental_normalized_eight_point

# Paper default synthetic setup (Section 4).
IMAGE_W, IMAGE_H = 1280, 720
FOCAL_GT = 1000.0
CX_GT, CY_GT = 640.0, 360.0
SCENE_DISTANCE = 1.0
SCENE_DEPTH = 0.5
BASELINE = 0.1
K_GT = np.array([[FOCAL_GT, 0.0, CX_GT], [0.0, FOCAL_GT, CY_GT], [0.0, 0.0, 1.0]], dtype=np.float64)

# __MT_APPEND__


def axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    s = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return math.cos(angle_rad) * np.eye(3) + (1 - math.cos(angle_rad)) * np.outer(axis, axis) + math.sin(angle_rad) * s


def k_relative_error(focal: float, cx: float, cy: float) -> float:
    """Paper metric: ||K_est - K_gt||_F / ||K_gt||_F."""
    K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return float(np.linalg.norm(K - K_GT) / np.linalg.norm(K_GT))


@dataclass
class PairData:
    pts0: np.ndarray
    pts1: np.ndarray
    tau: float
    angle_deg: float


def sample_pair(rng, *, cx, cy, num_points, image_noise_px, angle_noise_sigma,
                angle_range=(6.0, 35.0)) -> PairData:
    """One synthetic image pair under the paper's setup, with given principal point."""
    angle_deg = float(rng.uniform(*angle_range))
    R = axis_angle(rng.normal(size=3), math.radians(angle_deg))
    t = rng.normal(size=3)
    t = BASELINE * t / np.linalg.norm(t)
    pts0, pts1 = [], []
    lo, hi = SCENE_DISTANCE - 0.5 * SCENE_DEPTH, SCENE_DISTANCE + 0.5 * SCENE_DEPTH
    tries = 0
    while len(pts0) < num_points and tries < num_points * 40:
        tries += 1
        d = rng.uniform(lo, hi)
        u0, v0 = rng.uniform(0, IMAGE_W), rng.uniform(0, IMAGE_H)
        X = np.array([(u0 - cx) / FOCAL_GT * d, (v0 - cy) / FOCAL_GT * d, d])
        Xc = R @ X + t
        if Xc[2] <= 1e-6:
            continue
        u1, v1 = FOCAL_GT * Xc[0] / Xc[2] + cx, FOCAL_GT * Xc[1] / Xc[2] + cy
        if not (0 <= u1 < IMAGE_W and 0 <= v1 < IMAGE_H):
            continue
        pts0.append([u0, v0])
        pts1.append([u1, v1])
    pts0 = np.array(pts0, dtype=np.float64)
    pts1 = np.array(pts1, dtype=np.float64)
    if image_noise_px > 0 and pts0.shape[0]:
        pts0 = pts0 + image_noise_px * rng.standard_normal(pts0.shape)
        pts1 = pts1 + image_noise_px * rng.standard_normal(pts1.shape)
    # Angle noise: observed tau from a perturbed angle (paper models theta*s).
    observed_angle = angle_deg * (1.0 + angle_noise_sigma * rng.normal()) if angle_noise_sigma > 0 else angle_deg
    tau = 1.0 + 2.0 * math.cos(math.radians(observed_angle))
    return PairData(pts0, pts1, tau, angle_deg)


# __MT_APPEND_2__


@dataclass
class ProfileRuntime:
    num_focal_samples: int
    max_nodes: int
    min_box_size_px: float


@dataclass
class SolveOutcome:
    focal: float
    cx: float
    cy: float
    feasible: bool
    n_real: int = 0
    n_feasible: int = 0


def solve_martyushev_pair(pair: PairData, *, miscalib_alpha: float = 0.0) -> SolveOutcome:
    """Single-pair minimal solve. Returns the feasible solution nearest GT center.

    miscalib_alpha implements the paper's Fig. error_f control: the principal
    point used for feasibility gating is offset by alpha (the solver itself is
    blind to GT; this only emulates a miscalibrated center assumption).
    """
    if pair.pts0.shape[0] < 8:
        return SolveOutcome(math.nan, math.nan, math.nan, False)
    sols = ms.solve_calibration_from_correspondences(pair.pts0, pair.pts1, pair.tau)
    feas = [s for s in sols if s.normalized_focal > 0 and np.isfinite(s.focal)]
    n_real = len(sols)
    n_feas = len(feas)
    if not feas:
        return SolveOutcome(math.nan, math.nan, math.nan, False, n_real, n_feas)
    gate_cx = CX_GT * (1.0 + miscalib_alpha)
    gate_cy = CY_GT * (1.0 + miscalib_alpha)
    best = min(feas, key=lambda s: math.hypot(s.cx - gate_cx, s.cy - gate_cy))
    return SolveOutcome(best.focal, best.cx, best.cy, True, n_real, n_feas)


def estimate_martyushev_single(pairs: list[PairData], **kw) -> SolveOutcome:
    return solve_martyushev_pair(pairs[0], **kw)


def estimate_martyushev_multi(pairs: list[PairData], **kw) -> SolveOutcome:
    """Per-pair minimal solve + robust median fusion across pairs."""
    per = []
    for p in pairs:
        out = solve_martyushev_pair(p, **kw)
        if out.feasible:
            per.append((out.focal, out.cx, out.cy))
    if not per:
        return SolveOutcome(math.nan, math.nan, math.nan, False)
    arr = np.array(per)
    return SolveOutcome(float(np.median(arr[:, 0])), float(np.median(arr[:, 1])),
                        float(np.median(arr[:, 2])), True)


def estimate_kfpps_profile(pairs: list[PairData], *, runtime: ProfileRuntime, use_angle: bool = True, **kw) -> SolveOutcome:
    """KFPPS focal-profile optimizer accumulating all pairs jointly."""
    observations = []
    for idx, p in enumerate(pairs):
        if p.pts0.shape[0] < 8:
            continue
        F = estimate_fundamental_normalized_eight_point(p.pts0, p.pts1)
        kwargs: dict[str, Any] = {"F": F, "weight": 1.0, "label": f"pair_{idx}"}
        if use_angle:
            kwargs["known_rotation_angle_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, (p.tau - 1.0) / 2.0))))
            kwargs["known_rotation_weight"] = 1.0
        observations.append(FundamentalObservation(**kwargs))
    if not observations:
        return SolveOutcome(math.nan, math.nan, math.nan, False)
    bounds = OptimizerBounds(
        eta=(math.log(FOCAL_GT / 2.5), math.log(FOCAL_GT * 2.5)),
        cx=(-0.5 * IMAGE_W, 1.5 * IMAGE_W),
        cy=(-0.5 * IMAGE_H, 1.5 * IMAGE_H),
    )
    config = KFPPSFocalProfileConfig(
        image_size=(IMAGE_W, IMAGE_H),
        focal_prior=FocalPrior(focal_px=FOCAL_GT, scale_log=math.log(2.5), robust=True, weight=0.0),
        search_box=SearchBox(-0.5 * IMAGE_W, 1.5 * IMAGE_W, -0.5 * IMAGE_H, 1.5 * IMAGE_H),
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
    est = KFPPSFocalProfileOptimizer(config).solve(observations)
    return SolveOutcome(float(est.focal), float(est.cx), float(est.cy), bool(est.success))


# Method registry: name -> (estimator, default num_pairs, is_slow).
METHODS: dict[str, dict[str, Any]] = {
    "martyushev": {"fn": estimate_martyushev_single, "num_pairs": 1, "slow": False},
    "martyushev_multi": {"fn": estimate_martyushev_multi, "num_pairs": 4, "slow": False},
    "kfpps_profile": {"fn": estimate_kfpps_profile, "num_pairs": 4, "slow": True},
}


# __MT_APPEND_3__


@dataclass
class TrialBudget:
    fast_trials: int   # for millisecond Martyushev methods
    slow_trials: int   # for the slow KFPPS profile method
    profile_runtime: ProfileRuntime


def trials_for(method: str, budget: TrialBudget) -> int:
    return budget.slow_trials if METHODS[method]["slow"] else budget.fast_trials


def run_method_trials(
    method: str,
    budget: TrialBudget,
    *,
    seed: int,
    num_points: int,
    image_noise_px: float,
    angle_noise_sigma: float,
    cx: float = CX_GT,
    cy: float = CY_GT,
    miscalib_alpha: float = 0.0,
    num_pairs: int | None = None,
) -> dict[str, Any]:
    """Run one method over its trial budget; collect K-rel-error, focal err, etc."""
    spec = METHODS[method]
    fn = spec["fn"]
    npairs = num_pairs if num_pairs is not None else spec["num_pairs"]
    n_trials = trials_for(method, budget)
    kerrs, ferrs, pperrs, times = [], [], [], []
    n_real_list, n_feas_list = [], []
    feasible = 0
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t * 99991)
        pairs = [sample_pair(rng, cx=cx, cy=cy, num_points=num_points,
                             image_noise_px=image_noise_px, angle_noise_sigma=angle_noise_sigma)
                 for _ in range(npairs)]
        t0 = time.perf_counter()
        if method == "kfpps_profile":
            out = fn(pairs, runtime=budget.profile_runtime)
        else:
            out = fn(pairs, miscalib_alpha=miscalib_alpha)
        times.append(time.perf_counter() - t0)
        n_real_list.append(out.n_real)
        n_feas_list.append(out.n_feasible)
        if out.feasible and math.isfinite(out.focal):
            feasible += 1
            kerrs.append(k_relative_error(out.focal, out.cx, out.cy))
            ferrs.append(abs(out.focal - FOCAL_GT) / FOCAL_GT)
            pperrs.append(math.hypot(out.cx - cx, out.cy - cy))

    def med(x):
        return float(np.median(x)) if x else float("nan")

    return {
        "method": method,
        "num_pairs": npairs,
        "trials": n_trials,
        "feasible_rate": feasible / n_trials if n_trials else 0.0,
        "median_K_rel_error": med(kerrs),
        "median_focal_rel_error": med(ferrs),
        "median_pp_error_px": med(pperrs),
        "mean_solve_ms": float(np.mean(times) * 1000) if times else float("nan"),
        "mean_n_real": float(np.mean(n_real_list)) if n_real_list else 0.0,
        "mean_n_feasible": float(np.mean(n_feas_list)) if n_feas_list else 0.0,
    }


def experiment_E1_numerical_accuracy(budget: TrialBudget, methods: list[str], seed: int) -> dict[str, Any]:
    """Noise-free K relative error (paper median 2.5e-9)."""
    rows = [run_method_trials(m, budget, seed=seed, num_points=20, image_noise_px=0.0, angle_noise_sigma=0.0)
            for m in methods]
    return {"experiment": "E1_numerical_accuracy", "noise_free": True, "rows": rows}


def experiment_E2_number_of_solutions(budget: TrialBudget, seed: int) -> dict[str, Any]:
    """Real / feasible solution counts (Martyushev only; KFPPS has no notion)."""
    row = run_method_trials("martyushev", budget, seed=seed, num_points=20,
                            image_noise_px=0.0, angle_noise_sigma=0.0)
    return {"experiment": "E2_number_of_solutions",
            "mean_n_real": row["mean_n_real"], "mean_n_feasible": row["mean_n_feasible"],
            "note": "paper: real solutions usually 2 or 4; feasible usually 1"}


# __MT_APPEND_4__


def experiment_E3a_image_noise(budget, methods, seed, sigmas) -> dict[str, Any]:
    rows = []
    for sigma in sigmas:
        for m in methods:
            r = run_method_trials(m, budget, seed=seed, num_points=20,
                                  image_noise_px=sigma, angle_noise_sigma=0.0)
            r["image_noise_px"] = sigma
            rows.append(r)
    return {"experiment": "E3a_image_noise", "sigmas": sigmas, "rows": rows}


def experiment_E3b_point_count(budget, methods, seed, point_counts, fixed_noise=1.0) -> dict[str, Any]:
    rows = []
    for n in point_counts:
        for m in methods:
            r = run_method_trials(m, budget, seed=seed, num_points=n,
                                  image_noise_px=fixed_noise, angle_noise_sigma=0.0)
            r["num_points"] = n
            rows.append(r)
    return {"experiment": "E3b_point_count", "point_counts": point_counts, "fixed_noise_px": fixed_noise, "rows": rows}


def experiment_E3c_angle_noise(budget, methods, seed, angle_sigmas, fixed_noise=0.5) -> dict[str, Any]:
    rows = []
    for asig in angle_sigmas:
        for m in methods:
            r = run_method_trials(m, budget, seed=seed, num_points=20,
                                  image_noise_px=fixed_noise, angle_noise_sigma=asig)
            r["angle_noise_sigma"] = asig
            rows.append(r)
    return {"experiment": "E3c_angle_noise", "angle_sigmas": angle_sigmas, "fixed_noise_px": fixed_noise, "rows": rows}


def experiment_E4_focal_miscalib(budget, methods, seed, sigmas, alpha=0.1) -> dict[str, Any]:
    """Focal relative error vs image noise, with a 10% principal-point miscalibration."""
    rows = []
    for sigma in sigmas:
        for m in methods:
            r = run_method_trials(m, budget, seed=seed, num_points=20,
                                  image_noise_px=sigma, angle_noise_sigma=0.0, miscalib_alpha=alpha)
            r["image_noise_px"] = sigma
            rows.append(r)
    return {"experiment": "E4_focal_miscalibration", "alpha": alpha, "sigmas": sigmas, "rows": rows}


def experiment_E6_speed(budget, methods, seed) -> dict[str, Any]:
    rows = [run_method_trials(m, budget, seed=seed, num_points=20, image_noise_px=0.5, angle_noise_sigma=0.0)
            for m in methods]
    return {"experiment": "E6_speed",
            "rows": [{"method": r["method"], "mean_solve_ms": r["mean_solve_ms"], "num_pairs": r["num_pairs"]} for r in rows]}


# __MT_APPEND_5__


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def fmt(x: float, sci: bool = False) -> str:
    if not isinstance(x, (int, float)) or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.3e}" if sci else f"{x:.4f}"


def write_tables(results: dict[str, Any], tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Martyushev 2018 table reproduction: method comparison", "",
             "Methods: `martyushev` (single-pair minimal solver), "
             "`martyushev_multi` (per-pair + median fusion), "
             "`kfpps_profile` (joint profile accumulation).", ""]

    e1 = results.get("E1")
    if e1:
        lines += ["## E1 numerical accuracy (noise-free)", "",
                  "| method | trials | feasible | median K rel err | median focal rel err | median pp err px |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for r in e1["rows"]:
            lines.append(f"| {r['method']} | {r['trials']} | {r['feasible_rate']:.2f} | "
                         f"{fmt(r['median_K_rel_error'], sci=True)} | {fmt(r['median_focal_rel_error'], sci=True)} | "
                         f"{fmt(r['median_pp_error_px'])} |")
        lines += ["", "Paper reference: median numerical error ~2.5e-9 (noise-free).", ""]

    e2 = results.get("E2")
    if e2:
        lines += ["## E2 number of solutions (Martyushev, noise-free)", "",
                  f"mean real solutions: {e2['mean_n_real']:.2f}, mean feasible: {e2['mean_n_feasible']:.2f}",
                  "", f"_{e2['note']}_", ""]

    def sweep_table(key, title, axis_key, axis_label, value_key="median_K_rel_error", sci=True, ref=""):
        block = results.get(key)
        if not block:
            return []
        axis_vals = sorted({r[axis_key] for r in block["rows"]})
        methods = []
        for r in block["rows"]:
            if r["method"] not in methods:
                methods.append(r["method"])
        out = [f"## {title}", "",
               "| method | " + " | ".join(f"{axis_label}={v}" for v in axis_vals) + " |",
               "| --- | " + " | ".join("---:" for _ in axis_vals) + " |"]
        for m in methods:
            cells = []
            for v in axis_vals:
                match = [r for r in block["rows"] if r["method"] == m and r[axis_key] == v]
                cells.append(fmt(match[0][value_key], sci=sci) if match else "--")
            out.append(f"| {m} | " + " | ".join(cells) + " |")
        if ref:
            out += ["", ref]
        out.append("")
        return out

    lines += sweep_table("E3a", "E3a image-noise sweep (median K rel error)", "image_noise_px", "sigma")
    lines += sweep_table("E3b", "E3b point-count sweep (median K rel error, noise=1px)", "num_points", "N")
    lines += sweep_table("E3c", "E3c angle-noise sweep (median K rel error)", "angle_noise_sigma", "asig")
    lines += sweep_table("E4", "E4 focal rel error with 10% miscalibration", "image_noise_px", "sigma",
                         value_key="median_focal_rel_error")

    e6 = results.get("E6")
    if e6:
        lines += ["## E6 speed", "", "| method | mean solve ms | pairs |", "| --- | ---: | ---: |"]
        for r in e6["rows"]:
            lines.append(f"| {r['method']} | {fmt(r['mean_solve_ms'])} | {r['num_pairs']} |")
        lines.append("")

    (tables_dir / "martyushev_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {tables_dir / 'martyushev_comparison.md'}")


# __MT_APPEND_6__


def resolve_budget(preset: str, fast_trials: int | None, slow_trials: int | None) -> TrialBudget:
    if preset == "paper":
        rt = ProfileRuntime(num_focal_samples=9, max_nodes=2000, min_box_size_px=3.0)
        return TrialBudget(fast_trials or 2000, slow_trials or 50, rt)
    if preset == "smoke":
        rt = ProfileRuntime(num_focal_samples=7, max_nodes=1500, min_box_size_px=4.0)
        return TrialBudget(fast_trials or 20, slow_trials or 3, rt)
    raise ValueError(preset)


def main() -> None:
    p = argparse.ArgumentParser(description="Reproduce Martyushev 2018 synthetic tables with method comparison.")
    p.add_argument("--preset", choices=["smoke", "paper"], default="smoke")
    p.add_argument("--fast-trials", type=int, default=None, help="Override trial count for fast (Martyushev) methods.")
    p.add_argument("--slow-trials", type=int, default=None, help="Override trial count for the slow KFPPS profile method.")
    p.add_argument("--methods", nargs="*", default=["martyushev", "martyushev_multi", "kfpps_profile"])
    p.add_argument("--experiments", nargs="*", default=["E1", "E2", "E3a", "E3b", "E3c", "E4", "E6"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--image-sigmas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--point-counts", type=int, nargs="*", default=[8, 10, 15, 20, 40])
    p.add_argument("--angle-sigmas", type=float, nargs="*", default=[0.0, 0.03, 0.06, 0.09])
    args = p.parse_args()

    budget = resolve_budget(args.preset, args.fast_trials, args.slow_trials)
    methods = args.methods
    t_start = time.time()
    print(f"preset={args.preset} fast_trials={budget.fast_trials} slow_trials={budget.slow_trials}")
    print(f"methods={methods} experiments={args.experiments}\n")

    results: dict[str, Any] = {}
    out_root = WORK / "results" / "martyushev_tables"

    def checkpoint(tag: str, payload: Any) -> None:
        results[tag] = payload
        write_json(out_root / f"{tag}.json", payload)
        write_tables(results, WORK / "results" / "tables")
        print(f"[{time.time()-t_start:7.0f}s] {tag} done", flush=True)

    if "E1" in args.experiments:
        checkpoint("E1", experiment_E1_numerical_accuracy(budget, methods, args.seed))
    if "E2" in args.experiments:
        checkpoint("E2", experiment_E2_number_of_solutions(budget, args.seed))
    if "E3a" in args.experiments:
        checkpoint("E3a", experiment_E3a_image_noise(budget, methods, args.seed, args.image_sigmas))
    if "E3b" in args.experiments:
        checkpoint("E3b", experiment_E3b_point_count(budget, methods, args.seed, args.point_counts))
    if "E3c" in args.experiments:
        checkpoint("E3c", experiment_E3c_angle_noise(budget, methods, args.seed, args.angle_sigmas))
    if "E4" in args.experiments:
        checkpoint("E4", experiment_E4_focal_miscalib(budget, methods, args.seed, args.image_sigmas))
    if "E6" in args.experiments:
        checkpoint("E6", experiment_E6_speed(budget, methods, args.seed))

    write_json(out_root / "all_results.json", {"preset": args.preset, "elapsed_sec": time.time() - t_start, "results": results})
    print(f"\nDONE in {time.time()-t_start:.0f}s -> {out_root}")


if __name__ == "__main__":
    main()






