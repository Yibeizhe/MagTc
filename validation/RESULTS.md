# Validation results (MagTc 1.0.0, 2026-11-03, Numba backend, 4-core cloud container)

## Regression tests

`pytest` → 27 passed (including the `slow` large-anisotropy test), about 60 s.
`NUMBA_DISABLE_JIT=1 pytest -m "not slow"` on the light test files → 21 passed, 3 skipped
(the skipped ones are the Numba bit-for-bit cases).

* Bit-for-bit identity with `kernels/tc_jcp_rect.py` for the same seeds: rectangular FM J1–J3
  (VN), FM J1–J3 with triaxial K, and J1–J5 with the stripe_a order parameter. Checked with Numba
  and with the pure-Python fallback.
* Square-lattice Ising model (`spin: ising`), J = 1 meV, Onsager T_c = 26.333 K:
  * chi_abs peak at 1.052, 1.044 and 1.022 T_c for L = 16, 24 and 32 (4 seeds, 2e3 + 2e4 sweeps),
    converging to T_c from above, as the finite-size shift requires;
  * C_v peak at 1.010, 1.012 and 1.006 T_c;
  * energy per spin at T = 20 K and 40 K within 1 % of Onsager's exact result (L = 24).
* Heisenberg spins with K_z = −30 J (easy axis): the C_v peak lies within [0.90, 1.05] T_c(Ising).

## VN, b = 2.7342 Å, L = 50 (`python validation/validate_vn.py --reduced`)

8 seeds (20260616 + 10000 k), ladder 445 → 395 K in 5 K steps, **2e4 + 4e4 sweeps** per T
(one fifth of the JCP sweeps), measured every 20 sweeps. Wall time 7.3 min on 4 cores.

| quantity | value |
|---|---|
| T_C (chi_abs peak) | **419.1 ± 2.1 K** (jackknife 1.7 K, single-seed SEM 2.1 K) |
| T_C (C_v peak) | 420.6 ± 12.8 K |
| JCP 2026 | 419.7 K |
| `kernels/tc_jcp_rect.py`, full protocol | 420.1 ± 1.7 K |

The full protocol (`python validation/validate_vn.py`, 1e5 + 2e5 sweeps) was not run here.

## CrI3 example (`examples/cri3_lado2017.yaml`), reduced size

Honeycomb nearest-neighbour XXZ model, Jxy = 4.95 meV and Jz = 5.1525 meV per pair (unit vectors),
K = 0. L = 24 (1152 spins), 8 seeds, 70 → 20 K in 2.5 K steps, 2e4 + 4e4 sweeps.

| quantity | value |
|---|---|
| T_C (chi_abs peak) | **28.0 ± 0.2 K** |
| T_C (C_v peak) | 27.5 ± 0.03 K |
| experiment (monolayer, Huang et al. 2017) | ~45 K |

**Caveat:** J = 2.2 meV, λ = 0.09 meV and the conversion convention (factor 1/2, S² = 9/4) could not
be checked against the paper text in this session (network policy), so this number is provisional.
The parameters were not tuned. The result is for a single size, without finite-size extrapolation.
