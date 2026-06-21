# MagTc

Monte Carlo code for estimating the Curie temperature (T_C) of two-dimensional
magnets from first-principles–derived exchange parameters and single-ion
anisotropy. This repository contains the production scripts used in our study of
monolayer VN, and is under continued development toward a more general-purpose
package for computing magnetic properties and transition temperatures of
two-dimensional and bulk magnets.

## Model

The spins are treated as classical three-component unit vectors (|S| = 1) on a
square lattice, evolving under the anisotropic Heisenberg Hamiltonian

    H = - sum_<ij> J_ij (S_i . S_j) - D sum_i (S_i^z)^2

with direction-resolved exchange couplings J_a, J_b, J_ab (J > 0 ferromagnetic)
and single-ion anisotropy D > 0 favoring the out-of-plane axis. The exchange
parameters and anisotropy are obtained from density-functional-theory total-energy
mapping (see the paper and its Supplementary Material).

## Method

- Single-spin-flip Metropolis Monte Carlo, accelerated with Numba.
- For each lattice parameter b, the temperature is scanned in a window of
  +/- 60 K around an initial estimate with a 5 K step, from high to low T.
- At each temperature: 1e5 thermalization sweeps, then 2e5 sampling sweeps,
  measured every 20 sweeps; one sweep = L x L attempted updates.
- The scan is repeated over 8 independent replicas (different seeds); reported
  quantities are replica averages with the standard error of the mean.
- Recorded observables: magnetization, susceptibility chi_abs and chi_z,
  specific heat C_v, Binder cumulants, and energy.

### Curie-temperature estimator

The code computes several T_C estimators (peaks of C_v, chi_abs, chi_z, and
-d|M|/dT). **In the associated paper, T_C is taken from the peak of the
absolute-magnetization susceptibility chi_abs** (the `Tc_chi_abs` column of the
output), with C_v and -d|M|/dT used only as consistency checks. Note that the
current script additionally prints a single "recommended" value based on the C_v
peak for convenience; for results consistent with the paper, use the
`Tc_chi_abs` column.

## Requirements

- Python 3.8+
- numpy
- numba
- matplotlib

Install:

    pip install numpy numba matplotlib

## Usage

Run all b points listed in the built-in table:

    python vn_tc_batch_L50_prod.py --all

Run a single b point (0-based index):

    python vn_tc_batch_L50_prod.py --index 0

On a PBS/SLURM array job, the array index is picked up automatically from
`PBS_ARRAYID`, `PBS_ARRAY_INDEX`, or `SLURM_ARRAY_TASK_ID`.

### Key options (defaults match the paper)

| Option            | Default | Meaning                                   |
|-------------------|---------|-------------------------------------------|
| `--L`             | 50      | linear lattice size (L x L spins)         |
| `--half-window`   | 60      | scan center +/- this many K               |
| `--step`          | 5       | temperature step (K)                      |
| `--n-therm`       | 100000  | thermalization sweeps per T               |
| `--n-meas`        | 200000  | sampling sweeps per T                     |
| `--measure-every` | 20      | measure once every this many sweeps       |
| `--replicas`      | 8       | independent replicas (averaged)           |
| `--mae-is-per-v`  | off     | treat input MAE as already per-V          |
| `--n-v-mae-cell`  | 4       | divisor for D = MAE_perp / n_v_mae_cell   |
| `--seed`          | 20260616| base random seed                          |

### Anisotropy convention

By default the code sets `D = MAE_perp / 4`, assuming the tabulated `MAE_perp`
is the spin–orbit total-energy difference of a 2 x 2 supercell. If your input
`MAE_perp` is already per V atom, pass `--mae-is-per-v`.

## Input data

The exchange parameters J_a, J_b, J_ab, the anisotropy MAE_perp, and a scan-center
temperature for each b are stored in the `TABLE_S3` array near the top of
`vn_tc_batch_L50_prod.py`. Edit this table to apply the code to other parameter
sets.

## Output

For each b, written to `tc_TableS3_L50_prod_results/`:

- `b_<value>_scan.csv` : temperature scan with means and standard errors of all
  observables.
- `b_<value>_scan.png` : diagnostic plots (magnetization, susceptibilities, C_v).

A combined summary across all b points is written to `Tc_summary_fixed.csv`.

## Citing

If you use this code, please cite the associated paper (citation to be added upon
publication) and this repository: https://github.com/Yibeizhe/MagTc

## License

Released under the MIT License (see `LICENSE`).
