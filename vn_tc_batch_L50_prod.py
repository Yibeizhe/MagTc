#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VN Monte Carlo Tc recalculation from Table S3, L50 production version.

Main fixes relative to the previous script:
1. Use the b-dependent MAE_perp from Table S3 to construct D for each b.
2. Do NOT use noisy chi_z coarse peak to center the fine scan.
3. Center the production temperature window around the old Table-S3 Tc only as a scan guide.
4. Use Cv peak as the recommended single-size Tc estimator; chi_abs, chi_z, and -d|M|/dT are saved as cross-checks.

Model:
    H = -sum_ij Jij Si.Sj - D sum_i (Si_z)^2
    J > 0 is ferromagnetic. D > 0 favors out-of-plane magnetization.

D convention:
    By default D = MAE_perp / 4, assuming Table S3 MAE_perp is a 2x2-cell total energy difference.
    If Table S3 MAE_perp is already per V atom, run with --mae-is-per-v.
"""

import os
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

kB = 0.08617  # meV/K

# b, Ja, Jb, Jab, MAE_perp, old_Tc
TABLE_S3 = np.array([
    [2.7342, 32.36, -7.74, 32.51, 0.5711, 415.7],
    [2.7889, 45.09, 14.00, 24.66, 0.5493, 477.1],
    [2.8436, 42.53, 21.42, 23.31, 0.5241, 492.5],
    [2.8983, 40.52, 29.76, 21.05, 0.4924, 492.5],
    [2.9529, 39.11, 37.67, 18.87, 0.4906, 492.5],
    [2.9624, 38.95, 38.95, 18.46, 0.4987, 492.5],
    [3.0076, 38.21, 44.45, 16.40, 0.4710, 492.5],
    [3.0623, 37.92, 50.18, 14.03, 0.4045, 477.1],
    [3.1170, 38.02, 55.31, 12.10, 0.3393, 477.1],
    [3.1717, 38.74, 59.63, 10.50, 0.2610, 477.1],
], dtype=np.float64)

# ============================================================
# Numba core
# ============================================================

@njit(cache=True, fastmath=True)
def seed_numba(seed):
    np.random.seed(seed)

@njit(cache=True, fastmath=True)
def random_spin_nb():
    z = 2.0 * np.random.random() - 1.0
    phi = 2.0 * np.pi * np.random.random()
    r = np.sqrt(1.0 - z*z)
    return r*np.cos(phi), r*np.sin(phi), z

@njit(cache=True, fastmath=True)
def random_spins_nb(L):
    s = np.empty((L, L, 3), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            sx, sy, sz = random_spin_nb()
            s[i, j, 0] = sx
            s[i, j, 1] = sy
            s[i, j, 2] = sz
    return s

@njit(cache=True, fastmath=True)
def mc_sweep_nb(s, beta, L, Ja, Jb, Jab, D):
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        ip = (i + 1) % L
        im = (i - 1) % L
        jp = (j + 1) % L
        jm = (j - 1) % L

        hx = (Ja*(s[ip,j,0] + s[im,j,0])
            + Jb*(s[i,jp,0] + s[i,jm,0])
            + Jab*(s[ip,jp,0] + s[ip,jm,0] + s[im,jp,0] + s[im,jm,0]))
        hy = (Ja*(s[ip,j,1] + s[im,j,1])
            + Jb*(s[i,jp,1] + s[i,jm,1])
            + Jab*(s[ip,jp,1] + s[ip,jm,1] + s[im,jp,1] + s[im,jm,1]))
        hz = (Ja*(s[ip,j,2] + s[im,j,2])
            + Jb*(s[i,jp,2] + s[i,jm,2])
            + Jab*(s[ip,jp,2] + s[ip,jm,2] + s[im,jp,2] + s[im,jm,2]))

        sx0 = s[i,j,0]
        sy0 = s[i,j,1]
        sz0 = s[i,j,2]

        sx1, sy1, sz1 = random_spin_nb()

        dE = -((sx1-sx0)*hx + (sy1-sy0)*hy + (sz1-sz0)*hz)
        dE += -D*(sz1*sz1 - sz0*sz0)

        if dE <= 0.0 or np.random.random() < np.exp(-beta*dE):
            s[i,j,0] = sx1
            s[i,j,1] = sy1
            s[i,j,2] = sz1

@njit(cache=True, fastmath=True)
def measure_magnetization_nb(s, L):
    mx = 0.0
    my = 0.0
    mz = 0.0
    for i in range(L):
        for j in range(L):
            mx += s[i,j,0]
            my += s[i,j,1]
            mz += s[i,j,2]
    N = L * L
    mx /= N
    my /= N
    mz /= N
    mt = np.sqrt(mx*mx + my*my + mz*mz)
    return mt, abs(mz), mz

@njit(cache=True, fastmath=True)
def total_energy_per_spin_nb(s, L, Ja, Jb, Jab, D):
    E = 0.0
    for i in range(L):
        ip = (i + 1) % L
        for j in range(L):
            jp = (j + 1) % L
            jm = (j - 1) % L
            sx = s[i,j,0]
            sy = s[i,j,1]
            sz = s[i,j,2]
            E += -Ja  * (sx*s[ip,j,0]  + sy*s[ip,j,1]  + sz*s[ip,j,2])
            E += -Jb  * (sx*s[i,jp,0]  + sy*s[i,jp,1]  + sz*s[i,jp,2])
            E += -Jab * (sx*s[ip,jp,0] + sy*s[ip,jp,1] + sz*s[ip,jp,2])
            E += -Jab * (sx*s[ip,jm,0] + sy*s[ip,jm,1] + sz*s[ip,jm,2])
            E += -D * sz * sz
    return E / (L * L)

@njit(cache=True, fastmath=True)
def run_T_nb(s, T, L, Ja, Jb, Jab, D, n_therm, n_meas, every):
    beta = 1.0 / (kB * T)

    for _ in range(n_therm):
        mc_sweep_nb(s, beta, L, Ja, Jb, Jab, D)

    cnt = 0
    mz1 = 0.0
    mz2 = 0.0
    mz4 = 0.0
    mt1 = 0.0
    mt2 = 0.0
    mt4 = 0.0
    e1 = 0.0
    e2 = 0.0

    for k in range(n_meas):
        mc_sweep_nb(s, beta, L, Ja, Jb, Jab, D)
        if k % every == 0:
            Mt, Mz_abs, Mz = measure_magnetization_nb(s, L)
            E = total_energy_per_spin_nb(s, L, Ja, Jb, Jab, D)
            mz1 += Mz_abs
            mz2 += Mz * Mz
            mz4 += Mz * Mz * Mz * Mz
            mt1 += Mt
            mt2 += Mt * Mt
            mt4 += Mt * Mt * Mt * Mt
            e1 += E
            e2 += E * E
            cnt += 1

    mz1 /= cnt
    mz2 /= cnt
    mz4 /= cnt
    mt1 /= cnt
    mt2 /= cnt
    mt4 /= cnt
    e1 /= cnt
    e2 /= cnt

    N = L * L
    chi_z = N * (mz2 - mz1*mz1) / (kB*T)
    chi_abs = N * (mt2 - mt1*mt1) / (kB*T)
    Cv = N * (e2 - e1*e1) / (kB*T*T)
    U4z = 1.0 - mz4/(3.0*mz2*mz2) if mz2 > 1.0e-14 else 0.0
    U4abs = 1.0 - mt4/(3.0*mt2*mt2) if mt2 > 1.0e-14 else 0.0

    return mz1, mt1, chi_z, chi_abs, U4z, U4abs, Cv, e1

# ============================================================
# Analysis utilities
# ============================================================

def parabolic_peak(T, X):
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    i = int(np.argmax(X))
    if i == 0 or i == len(T) - 1:
        return float(T[i])
    xs = T[i-1:i+2]
    ys = X[i-1:i+2]
    a, b, c = np.polyfit(xs, ys, 2)
    if a >= 0:
        return float(T[i])
    return float(-b/(2*a))


def slope_peak(T, M):
    """Estimate Tc from the maximum magnitude of -dM/dT. Handles descending T."""
    T = np.asarray(T, dtype=float)
    M = np.asarray(M, dtype=float)
    order = np.argsort(T)
    Ts = T[order]
    Ms = M[order]
    dMdT = np.gradient(Ms, Ts)
    X = -dMdT
    Tc = parabolic_peak(Ts, X)
    return Tc


def convert_D(mae_perp, args):
    if args.mae_is_per_v:
        return mae_perp
    return mae_perp / args.n_v_mae_cell


def scan_one_replica(temps, L, Ja, Jb, Jab, D, args, seed):
    seed_numba(seed)
    s = random_spins_nb(L)
    rows = []
    for T in temps:
        mz, mt, chi_z, chi_abs, U4z, U4abs, Cv, E = run_T_nb(
            s, float(T), L, Ja, Jb, Jab, D, args.n_therm, args.n_meas, args.measure_every
        )
        rows.append([T, mz, mt, chi_z, chi_abs, U4z, U4abs, Cv, E])
        print(f"        T={T:7.2f}  |Mz|={mz:.4f}  |M|={mt:.4f}  chi_z={chi_z:.4f}  chi_abs={chi_abs:.4f}  Cv={Cv:.4f}", flush=True)
    return np.array(rows, dtype=float)


def scan_replicas(temps, L, Ja, Jb, Jab, D, args, seed0):
    reps = []
    for r in range(args.replicas):
        print(f"      replica {r+1}/{args.replicas}", flush=True)
        reps.append(scan_one_replica(temps, L, Ja, Jb, Jab, D, args, seed0 + 10000*r))
    arr = np.array(reps)
    mean = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1)/np.sqrt(args.replicas) if args.replicas > 1 else np.zeros_like(mean)
    return mean, sem


def run_one_case(row, args, case_index):
    bval, Ja, Jb, Jab, mae_perp, old_tc = row
    D = convert_D(mae_perp, args)

    print("\n============================================================", flush=True)
    print(f"b = {bval:.4f} A", flush=True)
    print(f"Ja={Ja:.2f}, Jb={Jb:.2f}, Jab={Jab:.2f} meV", flush=True)
    print(f"MAE_perp={mae_perp:.4f} meV, D used={D:.6f} meV per spin", flush=True)
    print(f"old Tc used only as scan center = {old_tc:.1f} K", flush=True)
    print("============================================================", flush=True)

    # Critical fix: use old Tc only as a broad scan center. Do not use noisy chi_z coarse peak.
    temps = np.arange(old_tc + args.half_window, old_tc - args.half_window - 1e-9, -args.step)
    print(f"  Scan window: {temps[0]:.1f} K -> {temps[-1]:.1f} K, step={args.step} K", flush=True)

    mean, sem = scan_replicas(temps, args.L, Ja, Jb, Jab, D, args, args.seed + 100000*case_index)

    Tc_chiz = parabolic_peak(mean[:,0], mean[:,3])
    Tc_chiabs = parabolic_peak(mean[:,0], mean[:,4])
    Tc_Cv = parabolic_peak(mean[:,0], mean[:,7])
    Tc_slope_abs = slope_peak(mean[:,0], mean[:,2])
    Tc_slope_z = slope_peak(mean[:,0], mean[:,1])

    # Recommended for the present single-size production scan:
    # Cv is less sensitive to the noisy z-axis component in weak-anisotropy finite systems.
    Tc_rec = Tc_Cv

    print("  Tc estimates:", flush=True)
    print(f"    Tc_Cv          = {Tc_Cv:.2f} K  recommended", flush=True)
    print(f"    Tc_chi_abs     = {Tc_chiabs:.2f} K", flush=True)
    print(f"    Tc_chi_z       = {Tc_chiz:.2f} K  auxiliary", flush=True)
    print(f"    Tc_slope_abs   = {Tc_slope_abs:.2f} K", flush=True)
    print(f"    Tc_slope_z     = {Tc_slope_z:.2f} K", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    prefix = f"b_{bval:.4f}".replace('.', 'p')

    csv_path = os.path.join(args.outdir, f"{prefix}_scan.csv")
    header = 'T_K,Mz_abs,M_abs,chi_z,chi_abs,U4z,U4abs,Cv,E,Mz_sem,Mabs_sem,chi_z_sem,chi_abs_sem,U4z_sem,U4abs_sem,Cv_sem,E_sem'
    data = np.column_stack([mean, sem[:,1:]])
    np.savetxt(csv_path, data, delimiter=',', header=header, comments='')

    png_path = os.path.join(args.outdir, f"{prefix}_scan.png")
    plot_case(mean, sem, bval, old_tc, Tc_rec, Tc_Cv, Tc_chiabs, Tc_chiz, png_path)

    return [bval, Ja, Jb, Jab, mae_perp, D, old_tc, Tc_rec, Tc_Cv, Tc_chiabs, Tc_chiz, Tc_slope_abs, Tc_slope_z]


def plot_case(mean, sem, bval, old_tc, Tc_rec, Tc_Cv, Tc_chiabs, Tc_chiz, png_path):
    T = mean[:,0]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))

    axes[0].errorbar(T, mean[:,1], yerr=sem[:,1], fmt='o-', ms=4, lw=1.3, capsize=2, label='|Mz|')
    axes[0].errorbar(T, mean[:,2], yerr=sem[:,2], fmt='s-', ms=4, lw=1.3, capsize=2, label='|M|')
    axes[0].axvline(Tc_rec, ls='--', color='0.25', lw=1.0)
    axes[0].set_xlabel('T (K)')
    axes[0].set_ylabel('magnetization')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(T, mean[:,4], yerr=sem[:,4], fmt='o-', ms=4, lw=1.3, capsize=2, label='chi_abs')
    axes[1].errorbar(T, mean[:,3], yerr=sem[:,3], fmt='s-', ms=3.5, lw=1.0, capsize=2, label='chi_z')
    axes[1].axvline(Tc_chiabs, ls='--', color='C0', lw=1.0)
    axes[1].axvline(Tc_chiz, ls=':', color='C1', lw=1.0)
    axes[1].set_xlabel('T (K)')
    axes[1].set_ylabel('susceptibility')
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].errorbar(T, mean[:,7], yerr=sem[:,7], fmt='o-', ms=4, lw=1.3, capsize=2, label='Cv')
    axes[2].axvline(Tc_Cv, ls='--', color='0.25', lw=1.0)
    axes[2].set_xlabel('T (K)')
    axes[2].set_ylabel('Cv')
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.suptitle(f'b={bval:.4f} A, old Tc={old_tc:.1f} K, recommended Tc={Tc_rec:.1f} K')
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description='Fixed VN Tc scan from Table S3 with b-dependent D and old-Tc-centered broad window.')
    p.add_argument('--all', action='store_true', help='run all b points')
    p.add_argument('--index', type=int, default=None, help='run one b point, 0-based index')
    p.add_argument('--outdir', default='tc_TableS3_L50_prod_results')
    p.add_argument('--L', type=int, default=50)
    p.add_argument('--half-window', type=float, default=60.0, help='scan old_Tc +/- half_window')
    p.add_argument('--step', type=float, default=5.0)
    p.add_argument('--n-therm', type=int, default=100000)
    p.add_argument('--n-meas', type=int, default=200000)
    p.add_argument('--measure-every', type=int, default=20)
    p.add_argument('--replicas', type=int, default=8)
    p.add_argument('--mae-is-per-v', action='store_true')
    p.add_argument('--n-v-mae-cell', type=float, default=4.0)
    p.add_argument('--seed', type=int, default=20260616)
    return p.parse_args()


def main():
    args = parse_args()
    if args.measure_every <= 0:
        raise SystemExit('measure_every must be positive')
    if args.n_meas <= 0 or args.n_therm < 0:
        raise SystemExit('n_meas must be positive and n_therm must be non-negative')

    print('Production defaults: L=%d, n_therm=%d, n_meas=%d, measure_every=%d, replicas=%d' % (args.L, args.n_therm, args.n_meas, args.measure_every, args.replicas), flush=True)

    # warm-up compile
    seed_numba(args.seed)
    s0 = random_spins_nb(4)
    _ = run_T_nb(s0, 450.0, 4, 1.0, 1.0, 1.0, 0.1, 2, 2, 1)

    env_idx = os.environ.get('PBS_ARRAYID') or os.environ.get('PBS_ARRAY_INDEX') or os.environ.get('SLURM_ARRAY_TASK_ID')
    if args.index is None and (not args.all) and env_idx is not None:
        args.index = int(env_idx)

    summaries = []
    if args.all:
        for idx, row in enumerate(TABLE_S3):
            summaries.append(run_one_case(row, args, idx))
    else:
        if args.index is None:
            raise SystemExit('Use --all, --index i, or a PBS/SLURM array index.')
        if args.index < 0 or args.index >= len(TABLE_S3):
            raise SystemExit('index out of range')
        summaries.append(run_one_case(TABLE_S3[args.index], args, args.index))

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, 'Tc_summary_fixed.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['b', 'Ja', 'Jb', 'Jab', 'MAE_perp_table', 'D_used', 'old_Tc_scan_center',
                    'Tc_recommended_Cv', 'Tc_Cv', 'Tc_chi_abs', 'Tc_chi_z', 'Tc_slope_abs', 'Tc_slope_z'])
        w.writerows(summaries)
    print(f"Summary saved to {path}", flush=True)
    print('Recommended column: Tc_recommended_Cv', flush=True)

if __name__ == '__main__':
    main()
