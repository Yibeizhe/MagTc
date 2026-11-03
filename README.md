# MagTc

Classical Monte Carlo Curie/Néel temperatures of **two-dimensional magnets**, packaged
as one versioned entry point for all T_C runs.

MagTc v1 packages the kernel of our J. Chem. Phys. 2026 study of monolayer VN
(`vn_tc_batch_L50_prod.py`, generalised and validated in `kernels/tc_jcp_rect.py`),
re-implemented on general bond lists. For a rectangular lattice it reproduces that
kernel **bit for bit**, with the same seeds and the same random-number stream.

## Model and protocol

Classical unit vectors |S| = 1, energies in meV, J > 0 ferromagnetic, every pair once:

    H = - sum_<ij> [ Jxy (Sx_i Sx_j + Sy_i Sy_j) + Jz Sz_i Sz_j ] + sum_i sum_a K_a (S_i^a)^2

(Heisenberg: Jxy = Jz = J. An easy axis z is K = (D, D, 0) or K = (0, 0, -D).)

Monte Carlo protocol (defaults = JCP 2026):

| setting | default |
|---|---|
| lattice | L x L unit cells, L = 50, periodic |
| move | single-spin Metropolis, new direction uniform on the sphere |
| ladder | scanned from high to low T, configuration carried over |
| sweeps per T | 1e5 thermalisation + 2e5 measurement, measured every 20 |
| seeds | 8 independent seeds (seed0 = 20260616, stride 10000) |
| T_C | peak of chi_abs = N(<\|M\|^2> - <\|M\|>^2)/k_B T of the full-vector order parameter, three-point parabola |
| error | max(leave-one-seed-out jackknife, SEM of single-seed peaks) |
| extrapolation | none (single size) |

The C_v peak and chi_z are reported as consistency checks.
k_B = 0.0861733 meV/K.

## Install

    pip install -e .             # numpy + pyyaml
    pip install -e ".[fast]"     # + numba (strongly recommended: ~100x faster)
    pip install -e ".[plot,test]"

Without Numba the same algorithm runs as plain Python with NumPy's RNG. This is
only useful for tests and tiny lattices. Numba and NumPy draw different random
streams from the same seed, so a result is reproducible only on the same backend.
The output JSON records which backend was used, and `analyse` refuses to mix
backends.

## Quickstart

    magtc show   examples/vn_b2p7342.yaml           # normalised model, ladder, seeds
    magtc ground examples/vn_b2p7342.yaml           # T = 0 energies of collinear orders
    magtc run    examples/vn_b2p7342.yaml --seed-index 0 --outdir runs
    magtc analyse runs/VN_b2p7342_seed*.json --out vn.json --plot vn.png
    magtc batch  examples/vn_b2p7342.yaml --nproc 8 --outdir runs     # all seeds + analysis
    magtc batch  examples/vn_b2p7342.yaml --pbs vn.pbs --queue batch  # write PBS array script only

Every MC setting can be overridden on the command line (`--L 24`, `--Tmax`, `--Tmin`, `--step`,
`--ntherm`, `--nmeas`, `--every`, `--nseeds`, `--seed0`). `python -m magtc` works as well.

## Model file

YAML or JSON. Full examples are in `examples/`.

```yaml
name: VN_b2p7342
reference: "doi / citation of the parameters"
spin: heisenberg                 # or ising (S = +-z, flip moves)
lattice:
  type: rectangular              # square | rectangular | honeycomb | custom
  # sublattices: [[0,0],[1/3,1/3]]   fractional positions (honeycomb default)
  # a1: [1, 0]   a2: [0.5, 0.866]    optional; used by magtc.model.bond_lengths()
exchange:                        # shells; bonds are [from_sub, to_sub, da, db], each pair ONCE
  - {name: Ja,  J: 32.36, bonds: [[0,0,1,0]]}
  - {name: Jb,  J: -7.74, bonds: [[0,0,0,1]]}
  - {name: Jab, J: 32.51, bonds: [[0,0,1,1],[0,0,1,-1]]}
  # XXZ shell:  {name: J1, Jxy: 4.95, Jz: 5.15, bonds: [...]}
anisotropy:
  K: [0.142775, 0.142775, 0.0]   # (Kx, Ky, Kz) meV, or one triple per sublattice
order_parameters:                # eta = signs[sub] * (-1)^(qa*i + qb*j); or pattern: [pa][pb][nsub]
  FM:   {q: [0,0], signs: [1]}   # FM, stripe_a, stripe_b, Neel are predefined for 1 sublattice
order: FM                        # order parameter measured by MC
mc: {L: 50, Tmax: 480, Tmin: 360, step: 5}   # everything else defaults to the JCP protocol
```

A bond `[s1, s2, da, db]` couples sublattice `s1` in cell (i, j) to sublattice `s2`
in cell (i+da, j+db). Listing a pair twice (also as its reverse) is an error. Use
`magtc.model.bond_lengths(model)` to check that every bond in a shell has the same
length.

## Outputs

* `run` writes one JSON per seed: the full model, its hash, the MC settings, the backend and,
  per temperature, <|M|>, <|M|^2>, <|M|^4>, <|Mz|>, <Mz^2>, <Mz^4>, <E>, <E^2>.
* `analyse` checks that the files belong to the same model, size, settings, ladder and backend.
  It then prints T_C(chi_abs) +- error and T_C(C_v), and flags a peak at the edge of the ladder.
* `ground` lists the energies of every q in {0,1}^2 × sublattice-sign pattern, with the moments
  along x, y and z. It warns if the MC order parameter is not the lowest of these. Spirals and
  long-period states are not tested.

## Validation

* `tests/` (fast, run in CI with and without Numba): `pytest -m "not slow"`
  * bit-for-bit agreement with `kernels/tc_jcp_rect.py`, with Numba and in pure Python;
  * `ground` energies compared against the reference kernel and against analytic results
    (rectangular J1–J5, honeycomb XXZ);
  * the square-lattice Ising model (`spin: ising`): energy against Onsager's exact solution,
    and the chi_abs / C_v peaks near T_c = 2.269 J/k_B;
  * model validation, the analysis error rule, and the CLI (`run`, `analyse`, `batch`, PBS).
* `validation/validate_vn.py` (slow, not in CI): VN at b = 2.7342 Å, L = 50. Expected
  420 +- 2 K (JCP: 419.7 K; reference kernel: 420.1 +- 1.7 K). `--reduced` runs 2e4 + 4e4 sweeps.

## Examples

* `examples/vn_b2p7342.yaml`: monolayer VN, published JCP 2026 parameters.
* `examples/cri3_lado2017.yaml`: monolayer CrI3, honeycomb nearest-neighbour XXZ model of
  Lado & Fernández-Rossier, 2D Mater. 4, 035002 (2017). **Read the header.** The numerical values
  and the convention conversion still have to be checked against the paper.
* `examples/ising_square.yaml`: Onsager benchmark.

## Legacy scripts

`vn_tc_batch_L50_prod.py` is kept unchanged as the script of the JCP paper.
`kernels/tc_jcp_rect.py` is kept as the reference that the regression tests compare against.
Use `magtc` for new calculations.

## Citing

Please cite the J. Chem. Phys. (2026) paper on monolayer VN and https://github.com/Yibeizhe/MagTc.

## License

MIT (see `LICENSE`).
