#!/usr/bin/env python3
"""T_C by the protocol of the VN JCP 2026 paper (SI S4), generalised to rectangular J1-J5 lattices with triaxial anisotropy
(ledger 4.48, 2026-10-30).

Protocol (as vn_tc_batch_L50_prod.py of github.com/Yibeizhe/MagTc): classical unit spins on an L x L lattice (L = 50),
single-spin Metropolis with a uniformly random new direction, temperature scanned from high to low carrying the
configuration over, n_therm = 1e5 sweeps and n_meas = 2e5 sweeps per temperature, measurement every 20 sweeps,
independent seeds averaged. T_C = peak of chi_abs = N (<|M|^2> - <|M|>^2) / (k_B T) of the full-vector order parameter
M = (1/N) sum_i eta_i S_i (eta = 1 for FM, (-1)^ix for stripe_a, (-1)^iy for stripe_b, (-1)^(ix+iy) for Neel),
three-point parabolic interpolation; C_v and |M| are consistency checks. No finite-size extrapolation.

Hamiltonian (each pair once, J > 0 ferromagnetic):
  H = - sum_<ij> J_ij S_i.S_j + sum_i sum_a K_a (S_i^a)^2
  shells: J1 (+-1,0), J2 (0,+-1), J3 (+-1,+-1), J4 (+-2,0), J5 (0,+-2) on the metal sublattice (index ix along a).
VN check: Ja -> J1, Jb -> J2, Jab -> J3, K = (D, D, 0), which equals -D (S^z)^2 up to a constant.

Usage:
  python3 tc_jcp_rect.py run --tag NAME --J j1 j2 j3 j4 j5 --K kx ky kz --order FM|stripe_a|stripe_b|Neel
                             --Tmax T --Tmin T --step 5 --seed S [--L 50 --ntherm 100000 --nmeas 200000 --every 20] --out f.json
  python3 tc_jcp_rect.py ground --J ... --K ...          (zero-temperature energies of the four collinear orders)
  python3 tc_jcp_rect.py analyse f1.json f2.json ...      (seed average, chi_abs peak, jackknife over seeds)
"""
import argparse, json, math, sys, time

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

kB = 0.0861733  # meV/K


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def rand_spin():
    z = 2.0 * np.random.random() - 1.0
    phi = 2.0 * np.pi * np.random.random()
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return r * math.cos(phi), r * math.sin(phi), z


@njit(cache=True)
def field(s, i, j, L, J):
    hx = hy = hz = 0.0
    nb = ((1, 0, 0), (-1, 0, 0), (0, 1, 1), (0, -1, 1), (1, 1, 2), (1, -1, 2), (-1, 1, 2), (-1, -1, 2),
          (2, 0, 3), (-2, 0, 3), (0, 2, 4), (0, -2, 4))
    for di, dj, n in nb:
        if J[n] == 0.0:
            continue
        a = (i + di) % L
        b = (j + dj) % L
        hx += J[n] * s[a, b, 0]
        hy += J[n] * s[a, b, 1]
        hz += J[n] * s[a, b, 2]
    return hx, hy, hz


@njit(cache=True)
def sweep(s, beta, L, J, K):
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        hx, hy, hz = field(s, i, j, L, J)
        x0, y0, z0 = s[i, j, 0], s[i, j, 1], s[i, j, 2]
        x1, y1, z1 = rand_spin()
        dE = -((x1 - x0) * hx + (y1 - y0) * hy + (z1 - z0) * hz)
        dE += K[0] * (x1 * x1 - x0 * x0) + K[1] * (y1 * y1 - y0 * y0) + K[2] * (z1 * z1 - z0 * z0)
        if dE <= 0.0 or np.random.random() < math.exp(-beta * dE):
            s[i, j, 0] = x1
            s[i, j, 1] = y1
            s[i, j, 2] = z1


@njit(cache=True)
def energy(s, L, J, K):
    E = 0.0
    fw = ((1, 0, 0), (0, 1, 1), (1, 1, 2), (1, -1, 2), (2, 0, 3), (0, 2, 4))
    for i in range(L):
        for j in range(L):
            for di, dj, n in fw:
                if J[n] == 0.0:
                    continue
                a = (i + di) % L
                b = (j + dj) % L
                E -= J[n] * (s[i, j, 0] * s[a, b, 0] + s[i, j, 1] * s[a, b, 1] + s[i, j, 2] * s[a, b, 2])
            E += K[0] * s[i, j, 0] ** 2 + K[1] * s[i, j, 1] ** 2 + K[2] * s[i, j, 2] ** 2
    return E / (L * L)


@njit(cache=True)
def order(s, L, eta):
    mx = my = mz = 0.0
    for i in range(L):
        for j in range(L):
            mx += eta[i, j] * s[i, j, 0]
            my += eta[i, j] * s[i, j, 1]
            mz += eta[i, j] * s[i, j, 2]
    N = L * L
    return mx / N, my / N, mz / N


@njit(cache=True)
def run_T(s, T, L, J, K, eta, ntherm, nmeas, every):
    beta = 1.0 / (kB * T)
    for _ in range(ntherm):
        sweep(s, beta, L, J, K)
    acc = np.zeros(9)   # <|M|>, <|M|^2>, <|M|^4>, <|Mz|>, <Mz^2>, <Mz^4>, <E>, <E^2>, count
    for k in range(nmeas):
        sweep(s, beta, L, J, K)
        if k % every == 0:
            mx, my, mz = order(s, L, eta)
            m2 = mx * mx + my * my + mz * mz
            e = energy(s, L, J, K)
            acc[0] += math.sqrt(m2); acc[1] += m2; acc[2] += m2 * m2
            acc[3] += abs(mz); acc[4] += mz * mz; acc[5] += mz ** 4
            acc[6] += e; acc[7] += e * e; acc[8] += 1.0
    return acc


def eta_of(order_name, L):
    ix, iy = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    return {"FM": np.ones((L, L)), "stripe_a": (-1.0) ** ix, "stripe_b": (-1.0) ** iy,
            "Neel": (-1.0) ** (ix + iy)}[order_name].astype(np.float64)


def cmd_run(a):
    J = np.array(a.J, dtype=np.float64); K = np.array(a.K, dtype=np.float64)
    assert len(J) == 5 and len(K) == 3 and a.L % 2 == 0
    eta = eta_of(a.order, a.L)
    seed_rng(a.seed)
    s = np.empty((a.L, a.L, 3))
    for i in range(a.L):
        for j in range(a.L):
            s[i, j] = rand_spin()
    temps = np.arange(a.Tmax, a.Tmin - 1e-9, -a.step)
    rows, t0 = [], time.time()
    for T in temps:
        acc = run_T(s, float(T), a.L, J, K, eta, a.ntherm, a.nmeas, a.every)
        n = acc[8]
        rows.append(dict(T=float(T), M=acc[0] / n, M2=acc[1] / n, M4=acc[2] / n, Mz=acc[3] / n, Mz2=acc[4] / n,
                         Mz4=acc[5] / n, E=acc[6] / n, E2=acc[7] / n, n=int(n)))
        print("%s seed %d T=%.1f |M|=%.4f  (%.0f s)" % (a.tag, a.seed, T, acc[0] / n, time.time() - t0), flush=True)
    json.dump(dict(tag=a.tag, J=a.J, K=a.K, order=a.order, L=a.L, seed=a.seed, ntherm=a.ntherm, nmeas=a.nmeas,
                   every=a.every, numba=HAVE_NUMBA, wall_s=time.time() - t0, rows=rows), open(a.out, "w"), indent=1)


def cmd_ground(a):
    J = np.array(a.J, dtype=np.float64); K = np.array(a.K, dtype=np.float64)
    L = 8
    out = {}
    for o in ("FM", "stripe_a", "stripe_b", "Neel"):
        s = np.zeros((L, L, 3)); s[:, :, 2] = eta_of(o, L)
        out[o] = float(energy(s, L, J, K))
    g = min(out, key=out.get)
    print(json.dumps(dict(energies_meV_per_site=out, ground=g)))
    return g


def peak(T, X):
    i = int(np.argmax(X))
    if i in (0, len(T) - 1):
        return float(T[i]), True
    c = np.polyfit(T[i - 1:i + 2], X[i - 1:i + 2], 2)
    return (float(-c[1] / (2 * c[0])) if c[0] < 0 else float(T[i])), False


def curves(runs):
    T = np.array([r["T"] for r in runs[0]["rows"]])
    N = runs[0]["L"] ** 2
    g = lambda k: np.mean([[row[k] for row in r["rows"]] for r in runs], axis=0)
    M, M2, M4, Mz, Mz2, E, E2 = g("M"), g("M2"), g("M4"), g("Mz"), g("Mz2"), g("E"), g("E2")
    chi_abs = N * (M2 - M ** 2) / (kB * T)
    chi_z = N * (Mz2 - Mz ** 2) / (kB * T)
    Cv = N * (E2 - E ** 2) / (kB * T ** 2)
    U = 1 - M4 / (3 * M2 ** 2)
    return T, M, chi_abs, chi_z, Cv, U


def cmd_analyse(a):
    runs = [json.load(open(f)) for f in a.files]
    assert len({(tuple(r["J"]), tuple(r["K"]), r["order"], r["L"]) for r in runs}) == 1
    T, M, chi, chiz, Cv, U = curves(runs)
    Tc, edge = peak(T, chi)
    Tcv, edgec = peak(T, Cv)
    Tz, _ = peak(T, chiz)
    n = len(runs)
    jk = [peak(*curves([r for k, r in enumerate(runs) if k != i])[0:3:2])[0] for i in range(n)] if n > 1 else []
    err = math.sqrt((n - 1) / n * sum((x - np.mean(jk)) ** 2 for x in jk)) if n > 1 else None
    per_seed = [peak(*curves([r])[0:3:2])[0] for r in runs]
    res = dict(tag=runs[0]["tag"], n_seeds=n, Tc_chi_abs=Tc, Tc_chi_abs_jackknife_err=err, peak_at_edge=edge,
               Tc_Cv=Tcv, Cv_peak_at_edge=edgec, Tc_chi_z=Tz, per_seed_chi_abs_peak=per_seed,
               sem_per_seed=float(np.std(per_seed, ddof=1) / math.sqrt(n)) if n > 1 else None,
               curve=dict(T=T.tolist(), M=M.tolist(), chi_abs=chi.tolist(), chi_z=chiz.tolist(), Cv=Cv.tolist(), U=U.tolist()))
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return res


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True); r.add_argument("--J", type=float, nargs=5, required=True)
    r.add_argument("--K", type=float, nargs=3, required=True); r.add_argument("--order", required=True)
    r.add_argument("--Tmax", type=float, required=True); r.add_argument("--Tmin", type=float, required=True)
    r.add_argument("--step", type=float, default=5.0); r.add_argument("--seed", type=int, required=True)
    r.add_argument("--L", type=int, default=50); r.add_argument("--ntherm", type=int, default=100000)
    r.add_argument("--nmeas", type=int, default=200000); r.add_argument("--every", type=int, default=20)
    r.add_argument("--out", required=True)
    g = sub.add_parser("ground")
    g.add_argument("--J", type=float, nargs=5, required=True); g.add_argument("--K", type=float, nargs=3, required=True)
    an = sub.add_parser("analyse")
    an.add_argument("files", nargs="+"); an.add_argument("--out", default=None)
    a = p.parse_args()
    {"run": cmd_run, "ground": cmd_ground, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
