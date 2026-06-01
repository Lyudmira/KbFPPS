# debug/

Throwaway diagnostic scripts kept as a record of how the Martyushev solver was
brought up and validated. They are **not** part of the paper pipeline
(`reproduce_kfpps.py`) and are not imported by it. Run them from the
`papers/KFPPS` directory, e.g. `python .\debug\validate_martyushev.py 20`.

What each one established, in the order they mattered:

- `check_system.py` — Decisive check that the four polynomials (Demazure cubic
  `f1,f2,f3` plus the known-angle constraint `f4`) vanish at ground truth in the
  normalized frame (~1e-14). Confirmed the **polynomial system is correct**, so
  any failure was in the solve, not the math.
- `inspect_support.py` — Reads the monomial support / degrees (deg 4 in a,b,
  deg 2 in p, 22 terms), matching the paper's `y0` monomial vector. Used to size
  the Macaulay / hidden-variable construction.
- `time_solvepoly.py` — Evidence that symbolic `solve_poly_system` (even over
  floats, with p saturated) is not a viable runtime path. Together with the
  deleted grobner/resultant probes, this is why the solver is numeric.
- `debug_numeric.py` — Showed the truncated-normal-form attempt fails: the ideal
  is not zero-dimensional because of the parasitic `p = 0` conic, so a fixed
  6-dim quotient does not exist without saturation.
- `debug_pencil.py` — Showed the hidden-variable-in-p quadratic eigenvalue
  problem **does** contain the true root as an eigenvalue, and that the
  rank-deficient linearization is why spurious roots appear (filtered downstream
  by Newton polish + equation-residual rejection).
- `check_noise.py` — Quick look at noise behaviour and the multi-pair averaging
  baseline. Note: naive averaging is *worse* than single-pair under noise, which
  motivates a robust-fusion baseline rather than a mean.
- `validate_martyushev.py` — The real regression test against the paper's oracle:
  noise-free ~1e-9 (we get ~1e-13) K recovery, six solutions generically, the
  feasible solution usually unique. Run this after touching the solver.
