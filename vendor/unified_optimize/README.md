# Unified Pinhole Intrinsics Optimizers

This folder collects several candidate optimizers for the case where MoGe focal
is useful but not trusted as a hard value, and the principal point may be far
outside the image.

The shared pinhole model is

```text
K(eta, c) = [[exp(eta), 0, cx],
             [0, exp(eta), cy],
             [0, 0, 1]]
```

where `eta = log(f)`.

## 1. F-only Profile Optimizer

File: `f_only_profile.py`

This is the first idea written as code:

```text
for eta in log-focal range around MoGe:
    optimize cx, cy
    score E = K.T @ F @ K
    add log-focal prior and optional principal-point prior
refine eta, cx, cy
```

Two essential residuals are available:

```text
original:
    ((s1 - s2)^2 + s3^2) / (s1^2 + s2^2)

manifold:
    (0.5 * (s1 - s2)^2 + s3^2) / (s1^2 + s2^2 + s3^2)
```

Use `residual_kind="original"` to reproduce the original draft. Use
`residual_kind="manifold"` for the Frobenius projection-style distance.

`laplace_correction=True` adds a profile-to-marginal approximation over
principal point:

```text
D(eta, c_eta*) + 0.5 log det H_c(eta)
```

This rewards broad valleys over accidental needle minima.

## 2. KFPPS Focal Profile Optimizer

File: `f_only_profile.py`

This wraps the existing certified fixed-focal KFPPS solver:

```text
for eta in log-focal range:
    run CertifiedPrincipalPointSolver(f=exp(eta))
    score KFPPS objective or essential residual
    add focal/principal priors
optionally polish with FOnlyProfileOptimizer
```

This is closest to the current exact-F pipeline, while no longer treating MoGe
focal as known truth.

## 3. Raw Sampson Joint Optimizer

File: `raw_sampson.py`

This is the currently preferred estimator when raw matches are available:

```text
min_{eta,c,R_ij,t_ij}
    sum robust_sampson(x_i, x'_i; K^-T [t_ij]x R_ij K^-1)
    + robust_log_focal_prior
    + broad_principal_prior
```

Each image pair owns nuisance variables `R_ij,t_ij`. The optimizer estimates
intrinsics without first compressing all matches into a single fundamental
matrix.

## 4. MoGe Point-map LM Optimizer

File: `moge_point_lm.py`

Standalone version of the useful dc_reality helper:

```text
u = fx * X / (Z + t) + cx
v = fy * Y / (Z + t) + cy
```

It fits `fx, fy, t, cx, cy` with `scipy.optimize.least_squares(method="lm")`.
This is not an epipolar optimizer; it is a low-dimensional intrinsics recovery
from a MoGe-style camera point map.

## 5. COLMAP Bridge

File: `colmap_bridge.py`

Tiny helpers to write an `IntrinsicsEstimate` back to a COLMAP `cameras.txt`
PINHOLE line.

## Minimal Examples

F-only profile:

```python
from unified_optimize import (
    FOnlyProfileConfig,
    FOnlyProfileOptimizer,
    FocalPrior,
    FundamentalObservation,
    PrincipalPointPrior,
)

observations = [FundamentalObservation(F=F01, label="0-1")]
config = FOnlyProfileConfig(
    image_size=(width, height),
    focal_prior=FocalPrior(focal_px=moge_f, scale_log=0.35, robust=True),
    principal_prior=PrincipalPointPrior(
        center=(width / 2, height / 2),
        scale=(5 * width, 5 * height),
        robust=True,
    ),
    residual_kind="manifold",
    laplace_correction=True,
)
estimate = FOnlyProfileOptimizer(config).solve(observations)
```

Raw matches:

```python
from unified_optimize import PairMatches, RawSampsonConfig, RawSampsonJointOptimizer

pairs = [PairMatches(points0=mkpts0, points1=mkpts1, label="0-1")]
estimate = RawSampsonJointOptimizer(
    RawSampsonConfig(
        image_size=(width, height),
        focal_prior=FocalPrior(focal_px=moge_f, scale_log=0.35, robust=True),
        principal_prior=PrincipalPointPrior(
            center=(width / 2, height / 2),
            scale=(5 * width, 5 * height),
            robust=True,
        ),
    )
).solve(pairs)
```

MoGe point map:

```python
from unified_optimize import MogePointLMOptimizer

estimate = MogePointLMOptimizer().solve(points_hw3, mask_hw)
```
