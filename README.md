# KFPPS paper reproduction

This directory is the paper-local reproduction package for the KFPPS vs.
Martyushev 2018 study.

It keeps all paper assets, downloaded references, experiment outputs, and
paper-ready tables under `work/`, while bundling the KFPPS algorithm kernel
under `vendor/`.

One command on Windows:

```powershell
.\run.ps1
```

Equivalent Python command:

```powershell
python .\reproduce_kfpps.py
```

What it does right now:

1. Downloads Martyushev ECCV 2018 public artifacts that are directly available:
   the CVF PDF and the arXiv source tarball.
2. Records a paper-local manifest noting that no official public repository was
   located from GitHub repository search.
3. Runs a synthetic known-angle benchmark around the main paper regimes:
   single-pair `F + theta`, multi-pair `F + theta`, and multi-pair `F`-only.
   The synthetic defaults are aligned to Martyushev's published setup:
   `1280 x 720`, `K_gt = [[1000,0,640],[0,1000,360],[0,0,1]]`, scene distance
   `1`, scene depth `0.5`, baseline `0.1`, and noisy image correspondences.
   The default command uses a `smoke` runtime preset so the paper-local run
   finishes in a practical amount of time; use `paper` for heavier sweeps.
4. Uses `KFPPSFocalProfileOptimizer` as the main method in all regimes, so the
   comparison is anchored on the same optimization machinery.
5. Writes JSON summaries and paper-ready markdown tables under
   `work/results/tables`.

Useful shorter commands:

```powershell
# Default paper-local smoke run.
python .\reproduce_kfpps.py --skip-download-martyushev

# Skip artifact downloads and only re-run the benchmark.
python .\reproduce_kfpps.py --skip-download-martyushev

# Heavier synthetic pass for paper numbers.
python .\reproduce_kfpps.py --runtime-preset paper --trials 16 --noise-sigmas 0 0.5 1.0
```

Outputs:

- `work/external/martyushev`: downloaded PDF/source and a manifest.
- `work/results/synthetic_known_angle/results.json`: per-trial benchmark rows.
- `work/results/tables/synthetic_known_angle_summary.md`: aggregate table.

Current scope:

- This package is now responsible for the paper-local experiment harness.
- The KFPPS angle-aware kernel lives in `vendor/`, which is the local copy of
  the actual solver implementation used by the paper.
- Direct reproduction of Martyushev's minimal Groebner/action-matrix solver is
  not wired in yet; since no official public code was found, that block will be
  added as a paper-local reimplementation against the downloaded paper source.
