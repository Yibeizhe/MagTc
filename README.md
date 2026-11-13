# MagTc 2

Curie and Néel temperatures of **2D and 3D magnets** from classical Monte Carlo, with a **VASP
energy-mapping** front end. The pipeline is:

    POSCAR ──magtc dft setup──▶ VASP runs ──magtc dft collect──▶ model.yaml (J1…Jn, K)
           ──magtc ground──▶ ground state ──magtc batch/run──▶ seeds ──magtc analyse──▶ T_C ± error

MagTc 2 is a redesign of MagTc 1, which was 2D only and needed hand-written bonds. The validated
kernel of our J. Chem. Phys. 2026 study of monolayer VN (`kernels/tc_jcp_rect.py`) is kept as the
benchmark. With the JCP protocol, MagTc 2 reproduces it **bit for bit**, with the same seeds and the
same random-number stream.

## Model

Spins are classical unit vectors, energies are in meV, J > 0 is ferromagnetic, and every pair is
counted once:

    H = - sum_<ij> [ Jxy (Sx_i Sx_j + Sy_i Sy_j) + Jz Sz_i Sz_j ] + sum_i sum_a K_a(i) (S_i^a)^2

For a source that uses |S| = S with a factor 1/2 for double counting, write J·S² per pair. Energy
mapping with unit spins (`magtc dft`) gives these per-pair values directly.

## Monte Carlo

| setting | default (JCP 2026) | options |
|---|---|---|
| protocol | `anneal`: ladder from high to low T, configuration carried over | `pt`: parallel tempering over the ladder |
| moves per sweep | Metropolis, uniform new direction | `cone` (adaptive Gaussian step), `overrelax: n`, `wolff: n` |
| sweeps per T | 1e5 thermalisation + 2e5 measurement, measure every 20 | any |
| seeds | 8 (seed0 20260616, stride 10000) | any |
| T_C | chi_abs peak (three-point parabola); error = max(jackknife, single-seed SEM) | C_v peak, chi_z and Binder cumulant also reported |

How the faster moves work:

* **Over-relaxation** reflects a spin about its exchange field. This conserves the exchange energy
  exactly, also for XXZ. The single-ion part is accepted with a Metropolis test.
* **Wolff clusters** use a random reflection plane and the isotropic Jxy. Bonds of either sign are
  allowed. The rest of the energy, (Jz − Jxy) SzSz′ plus the K terms, is accepted with a Metropolis
  test on the whole cluster, so XXZ and single-ion anisotropy stay exact.

All moves are checked against an exactly solvable XXZ dimer with anisotropy (`tests/test_moves.py`).
Every run records τ_int of |M| and E (binning analysis), the Metropolis acceptance, the cluster sizes
and the parallel-tempering swap rates.

**Speed, VN at L = 50 and T = 425 K** (near T_C), measured as CPU time per independent |M| sample
(`magtc bench`):

| moves | ms/sweep | τ(|M|) sweeps | cost per independent sample | speedup |
|---|---|---|---|---|
| Metropolis (JCP) | 0.45 | 170 | 152 ms | 1× |
| Metropolis + 2 over-relaxation | 0.74 | 6.3 | 9 ms | ≈17× |
| Metropolis + 20 Wolff | 1.36 | 1.3 | 3 ms | ≈50× |

## Install

    pip install -e ".[fast]"      # numpy, pyyaml, numba (strongly recommended)
    pip install -e ".[plot,test]"

Without Numba the same algorithms run in plain Python. That is useful for tests only. Numba and
NumPy produce different random streams, so a result is reproducible on one backend only. `analyse`
refuses to mix backends, move sets or protocols.

## Quickstart

### From VASP to T_C

1. Write the VASP inputs: magnetic configurations for J1–J3 in a 2×2 supercell, plus three SOC runs
   with FM moments along x, y and z for K.

       magtc dft setup POSCAR --magnetic Cr --nshell 3 --supercell 2 2 --template vasp_template/ --outdir map

   `vasp_template/` holds your INCAR (functional, U, ENCUT, smearing …), KPOINTS and POTCAR. MagTc only
   sets ISPIN, MAGMOM and LSORBIT/LNONCOLLINEAR, and divides an automatic k-mesh by the supercell size.

2. Run VASP in every `map/*/` directory.

3. Fit J and K and write the model:

       magtc dft collect map --out CrX3.yaml

   This reads the final energies (E0 by default; `--energy F` for the free energy), checks the local
   moments in OUTCAR against the intended configuration, and fits J by least squares. It reports
   standard errors, the leave-one-out residual and the condition number of the fit. It writes
   `CrX3.yaml` with the ground state set as the MC order, and a coarse T ladder placed from the
   mean-field T_C.

   * `--method fourstate` in `setup` gives the four-state method instead. It needs a supercell in
     which each pair is linked only once.
   * `--dim 3` handles bulk.

4. Check the ground state and run:

       magtc ground CrX3.yaml
       magtc batch  CrX3.yaml --nproc 8 --overrelax 2 --wolff 10      # coarse scan, then refine the ladder
       magtc batch  CrX3.yaml --pbs job.pbs --queue batch             # or write a PBS array (never submitted)
       magtc analyse magtc_out/*_seed*.json --plot tc.png

### Other commands

    magtc show  model.yaml                    # normalised model, ladder, seeds
    magtc run   model.yaml --seed-index 0     # one seed
    magtc bench model.yaml --T 425 --L 50     # choose moves: cost per independent sample

Every MC setting can be overridden on the command line: `--L 16 16 16 --Tmax --Tmin --step --ntherm
--nmeas --every --nseeds --protocol pt --metropolis cone --overrelax 2 --wolff 10`.

## Model file

```yaml
name: CrX3
spin: heisenberg                     # or ising
structure:
  dim: 2                             # 2 (a1, a2 periodic) or 3
  lattice: [[6.9, 0, 0], [-3.45, 5.976, 0], [0, 0, 20]]   # Angstrom
  sites: [{label: Cr, frac: [0.333, 0.667, 0.5]}, {label: Cr, frac: [0.667, 0.333, 0.5]}]
interactions:                        # meV per pair, each pair once
  - {name: J1, J: 2.5, shell: 1}     # every bond of the 1st distance shell
  - {name: J2, Jxy: 0.6, Jz: 0.65, shell: 2}               # XXZ
  - {name: J3, J: -0.15, bonds: [[0, 1, 1, 1, 0], [0, 1, -1, 1, 0], [0, 1, -1, -1, 0]]}   # explicit [s1, s2, da, db, dc]
anisotropy: {K: [0.4, 0.4, 0.0]}     # (Kx, Ky, Kz), or one triple per sublattice
orders: {zigzag: {q: [1, 0, 0], signs: [1, 1]}}            # eta = signs[u] (-1)^(q.R); FM is predefined
order: FM
mc: {L: 50, Tmax: 60, Tmin: 20, step: 2, moves: {metropolis: uniform, overrelax: 2, wolff: 10}}
```

`shell: n` groups bonds by distance. Bonds of equal length that are not related by symmetry end up
in the same shell; give explicit bonds when that distinction matters. MagTc 1.x files (`lattice`,
`exchange`, 4-index bonds) still load.

## Validation (`validation/RESULTS.md`)

* JCP mode against `kernels/tc_jcp_rect.py`: identical bit for bit (Numba and pure Python).
* VN, b = 2.7342 Å, L = 50:
  * JCP moves, reduced sweeps: 419.1 ± 2.1 K;
  * fast moves: see RESULTS.md;
  * JCP paper: 419.7 K.
* Exact XXZ + K dimer: every move set and parallel tempering agree within statistical error.
* 2D Ising: Onsager energy within 1 %; chi_abs and C_v peaks near T_c.
* 3D simple cubic: the Binder crossing of L = 8 and 12 lies at 0.997 T_c for Ising (4.5115 J) and for
  Heisenberg (1.443 J).
* VASP mapping, on synthetic energies: least squares and four-state recover J1–J3 and K to 1e-6 meV.
  With noise, the reported errors cover the true values, and flipped moments are detected.

## Limits

* Collinear energy mapping and Heisenberg/XXZ exchange only. No DMI, Kitaev or biquadratic terms.
* `ground` tries collinear states on a 2×2(×2) cell only. Spirals are not detected.
* No finite-size extrapolation. T_C is the single-size chi_abs peak (JCP protocol); Binder cumulants
  are stored for your own crossing analysis.
* Classical spins. Quantum corrections (RPA, S(S+1) rescaling) are not applied.
* VASP is the only DFT code supported.

## Legacy

`vn_tc_batch_L50_prod.py` (the script of the JCP paper) and `kernels/tc_jcp_rect.py` (the reference
kernel) are unchanged.

## License

MIT
