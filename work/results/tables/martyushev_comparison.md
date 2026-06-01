# Martyushev 2018 table reproduction: method comparison

Methods: `martyushev` (single-pair minimal solver), `martyushev_multi` (per-pair + median fusion), `kfpps_profile` (joint profile accumulation).

## E1 numerical accuracy (noise-free)

| method | trials | feasible | median K rel err | median focal rel err | median pp err px |
| --- | ---: | ---: | ---: | ---: | ---: |
| kfpps_profile | 3 | 1.00 | 2.018e-16 | 2.274e-16 | 0.0000 |

Paper reference: median numerical error ~2.5e-9 (noise-free).

