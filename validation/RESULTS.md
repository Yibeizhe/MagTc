# Validation results — MagTc 2.0.0 (2026-11-13, Numba backend, 4-core cloud container)

## Test suite

`pytest` → 41 passed, including the slow 3D Heisenberg Binder test (about 100 s).
`NUMBA_DISABLE_JIT=1 pytest -m "not slow"` on the light files → 27 passed, 3 skipped (Numba-only).

| check | result |
|---|---|
| JCP mode against `kernels/tc_jcp_rect.py`, same seeds | identical bit for bit (VN J1–J3; FM with triaxial K; J1–J5 stripe_a), with Numba and in pure Python |
| MagTc 1.x model file against the same model in 2.x format | identical run |
| exact dimer (XXZ Jz/Jxy = 1.6 with Kz; isotropic J with K = (D, D, 0)) | every move set (uniform, cone, over-relaxation, Wolff, combinations, parallel tempering) within 1.4σ at about 4e-4 relative precision (16–24 seeds) |
| 2D Ising, L = 16 | energy within 1 % of Onsager at 20 and 40 K; chi_abs peak 1.00–1.09 T_c; C_v peak within 4 % |
| 3D sc Ising (T_c = 4.5115 J) | Binder crossing L = 8/12 at **0.997 T_c**; chi_abs peak 0.995 (L = 8), 0.997 (L = 12) |
| 3D sc Heisenberg (T_c = 1.4430 J) | Binder crossing L = 8/12 at **0.997 T_c**; chi_abs peak 1.033 (L = 8), 1.023 (L = 12) |
| VASP mapping, synthetic honeycomb J1–J3 + K | least squares (2×2) and four-state (3×3) recover J and K to 1e-6 meV |
| VASP mapping, synthetic bcc 3D, 2×2×2 | J1–J3 recovered exactly |
| VASP mapping with 20 µeV noise | J within 5σ of the reported errors; a flipped moment in OUTCAR is detected; `--strict` stops |

## Speed (`magtc bench`): VN, L = 50, T = 425 K

| moves | ms/sweep | τ_int(\|M\|) sweeps | ms per independent \|M\| | speedup |
|---|---|---|---|---|
| Metropolis uniform (JCP) | 0.45 | 170 | 152 | 1 |
| cone | 0.64 | 188 | 240 | 0.6 |
| Metropolis + 2 over-relaxation | 0.74 | 6.3 | 9 | 17 |
| cone + 2 over-relaxation | 0.95 | 7.1 | 14 | 11 |
| Metropolis + 20 Wolff | 1.36 | 1.3 | 3 | 50 |
| cone + 2 over-relaxation + 20 Wolff | 1.85 | 1.3 | 5 | 30 |

## VN, b = 2.7342 Å, L = 50, 8 seeds, ladder 445 → 395 K in 5 K steps

| run | sweeps per T | wall time (4 cores) | T_C (chi_abs) |
|---|---|---|---|
| JCP paper | 1e5 + 2e5 | — | 419.7 K |
| `kernels/tc_jcp_rect.py`, full protocol | 1e5 + 2e5 | — | 420.1 ± 1.7 K |
| JCP moves, reduced (identical arithmetic in 1.x and 2.x) | 2e4 + 4e4 | 7.3 min | 419.1 ± 2.1 K |
| Metropolis + 2 over-relaxation + 10 Wolff (`--fast`) | 1e3 + 5e3 | 2.8 min | 416.2 ± 0.8 K |
| Metropolis + 2 over-relaxation + 10 Wolff | 2e3 + 2e4 | 9.8 min | **417.7 ± 0.9 K** (C_v 418.2 ± 6.4 K) |

* Point by point, the χ_abs(T) and ⟨|M|⟩(T) curves of the fast runs agree with the JCP-move run
  within its statistical errors.
* The fast run has at least 2400 independent samples per temperature and seed; the JCP-move
  reduced run has about 100.
* The fast and JCP values differ by 1.2–2σ combined. The χ_abs peak is flat over 415–420 K. The
  shift from 416.2 to 417.7 K when the statistics were quadrupled is larger than the quoted 0.8 K.
  So for a peak this flat, the jackknife over 8 seeds underestimates the error, and a 5 K ladder
  step adds a systematic uncertainty of about ±2 K from the parabola fit.
* For publication-level comparisons with the JCP numbers, use the JCP moves, or a finer ladder
  (2 K steps) with the fast moves.

## CrI3 example (`examples/cri3_lado2017.yaml`)

The MagTc 1.0 run gave 28.0 ± 0.2 K (L = 24, JCP moves). In 2.x the file has a real lattice, and
the bond list follows that geometry, so runs are not bit-identical to 1.x, only statistically
equivalent. **The parameters remain unverified against the paper**; see the header of the file.
