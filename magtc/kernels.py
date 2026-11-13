"""Monte Carlo kernels (Numba; plain-Python fallback when Numba is absent).

Moves (all satisfy detailed balance individually; a sweep applies them in this order):
  * Metropolis, random-sequential, N attempts per pass:
      mode 0  uniform new direction (JCP 2026 protocol; bit-for-bit with kernels/tc_jcp_rect.py),
      mode 1  adaptive cone: S' = (S + sigma g)/|...|, g Gaussian (symmetric proposal),
      mode 2  Ising flip S -> -S;
  * over-relaxation: reflection of S about its local exchange field, which conserves the exchange
    energy exactly (also for XXZ); the single-ion change is accepted with a Metropolis test;
  * Wolff clusters: random reflection plane r, bonds activated with p = 1 - exp(-2 beta Jxy (r.Si)(r.Sj))
    for satisfied bonds (any sign of J); the residual energy (Jz - Jxy) SzSz' and K terms is
    accepted with a Metropolis test on the whole cluster.
Random numbers: one global stream per process (np.random), seeded once per run.
"""
import math
import os

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

if os.environ.get("NUMBA_DISABLE_JIT", "0") not in ("", "0"):
    HAVE_NUMBA = False

kB = 0.0861733  # meV/K (value of the validated reference kernel)

MET_UNIFORM, MET_CONE, MET_FLIP, MET_NONE = 0, 1, 2, 3


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
def init_spins(nsite, ising):
    s = np.empty((nsite, 3))
    for n in range(nsite):
        if ising:
            s[n, 0] = 0.0
            s[n, 1] = 0.0
            s[n, 2] = 1.0 if np.random.random() < 0.5 else -1.0
        else:
            x, y, z = rand_spin()
            s[n, 0] = x
            s[n, 1] = y
            s[n, 2] = z
    return s


@njit(cache=True)
def metropolis_pass(s, beta, L1, L2, L3, nsub, nnb, nbr, jxy, jz, K, mode, sigma):
    """One pass of N random-sequential single-spin attempts.  Returns the number accepted."""
    nacc = 0
    for _ in range(L1 * L2 * L3 * nsub):
        i = np.random.randint(0, L1)
        j = np.random.randint(0, L2)
        k = 0
        if L3 > 1:
            k = np.random.randint(0, L3)
        u = 0
        if nsub > 1:
            u = np.random.randint(0, nsub)
        n = ((i * L2 + j) * L3 + k) * nsub + u
        hx = hy = hz = 0.0
        for c in range(nnb[n]):
            t = nbr[n, c]
            hx += jxy[n, c] * s[t, 0]
            hy += jxy[n, c] * s[t, 1]
            hz += jz[n, c] * s[t, 2]
        x0, y0, z0 = s[n, 0], s[n, 1], s[n, 2]
        if mode == 2:
            x1, y1, z1 = -x0, -y0, -z0
        elif mode == 1:
            gx = x0 + sigma * np.random.standard_normal()
            gy = y0 + sigma * np.random.standard_normal()
            gz = z0 + sigma * np.random.standard_normal()
            g = math.sqrt(gx * gx + gy * gy + gz * gz)
            x1, y1, z1 = gx / g, gy / g, gz / g
        else:
            x1, y1, z1 = rand_spin()
        dE = -((x1 - x0) * hx + (y1 - y0) * hy + (z1 - z0) * hz)
        dE += K[u, 0] * (x1 * x1 - x0 * x0) + K[u, 1] * (y1 * y1 - y0 * y0) + K[u, 2] * (z1 * z1 - z0 * z0)
        if dE <= 0.0 or np.random.random() < math.exp(-beta * dE):
            s[n, 0] = x1
            s[n, 1] = y1
            s[n, 2] = z1
            nacc += 1
    return nacc


@njit(cache=True)
def overrelax_pass(s, beta, nsub, nnb, nbr, jxy, jz, K, hasK):
    """Sequential reflections S -> 2 (S.h) h/|h|^2 - S about the exchange field h."""
    for n in range(s.shape[0]):
        hx = hy = hz = 0.0
        for c in range(nnb[n]):
            t = nbr[n, c]
            hx += jxy[n, c] * s[t, 0]
            hy += jxy[n, c] * s[t, 1]
            hz += jz[n, c] * s[t, 2]
        hh = hx * hx + hy * hy + hz * hz
        if hh < 1e-300:
            continue
        x0, y0, z0 = s[n, 0], s[n, 1], s[n, 2]
        p = 2.0 * (x0 * hx + y0 * hy + z0 * hz) / hh
        x1, y1, z1 = p * hx - x0, p * hy - y0, p * hz - z0
        g = math.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        x1, y1, z1 = x1 / g, y1 / g, z1 / g
        if hasK:
            u = n % nsub
            dE = K[u, 0] * (x1 * x1 - x0 * x0) + K[u, 1] * (y1 * y1 - y0 * y0) + K[u, 2] * (z1 * z1 - z0 * z0)
            if not (dE <= 0.0 or np.random.random() < math.exp(-beta * dE)):
                continue
        s[n, 0] = x1
        s[n, 1] = y1
        s[n, 2] = z1


@njit(cache=True)
def wolff_cluster(s, beta, nsub, nnb, nbr, jxy, jz, K, ising, stack, member, inc):
    """Grow and (if the residual-energy test passes) flip one Wolff cluster.  Returns its size,
    negative if rejected."""
    nsite = s.shape[0]
    if ising:
        rx, ry, rz = 0.0, 0.0, 1.0
    else:
        rx, ry, rz = rand_spin()
    n0 = np.random.randint(0, nsite)
    inc[n0] = True
    member[0] = n0
    size = 1
    top = 1
    stack[0] = n0
    while top > 0:
        top -= 1
        n = stack[top]
        pn = rx * s[n, 0] + ry * s[n, 1] + rz * s[n, 2]
        for c in range(nnb[n]):
            t = nbr[n, c]
            if inc[t]:
                continue
            x = 2.0 * beta * jxy[n, c] * pn * (rx * s[t, 0] + ry * s[t, 1] + rz * s[t, 2])
            if ising:
                x = 2.0 * beta * jz[n, c] * pn * (rx * s[t, 0] + ry * s[t, 1] + rz * s[t, 2])
            if x > 0.0 and np.random.random() < 1.0 - math.exp(-x):
                inc[t] = True
                member[size] = t
                size += 1
                stack[top] = t
                top += 1
    # residual energy: anisotropic exchange (Jz - Jxy) SzSz' and single-ion terms
    dE = 0.0
    if not ising:
        for a in range(size):
            n = member[a]
            pn = rx * s[n, 0] + ry * s[n, 1] + rz * s[n, 2]
            zn1 = s[n, 2] - 2.0 * pn * rz
            u = n % nsub
            xn1 = s[n, 0] - 2.0 * pn * rx
            yn1 = s[n, 1] - 2.0 * pn * ry
            dE += K[u, 0] * (xn1 * xn1 - s[n, 0] ** 2) + K[u, 1] * (yn1 * yn1 - s[n, 1] ** 2) \
                + K[u, 2] * (zn1 * zn1 - s[n, 2] ** 2)
            for c in range(nnb[n]):
                d = jz[n, c] - jxy[n, c]
                if d == 0.0:
                    continue
                t = nbr[n, c]
                if inc[t]:
                    pt = rx * s[t, 0] + ry * s[t, 1] + rz * s[t, 2]
                    zt1 = s[t, 2] - 2.0 * pt * rz
                    dE -= 0.5 * d * (zn1 * zt1 - s[n, 2] * s[t, 2])
                else:
                    dE -= d * (zn1 - s[n, 2]) * s[t, 2]
    acc = dE <= 0.0 or np.random.random() < math.exp(-beta * dE)
    for a in range(size):
        n = member[a]
        if acc:
            pn = rx * s[n, 0] + ry * s[n, 1] + rz * s[n, 2]
            s[n, 0] -= 2.0 * pn * rx
            s[n, 1] -= 2.0 * pn * ry
            s[n, 2] -= 2.0 * pn * rz
        inc[n] = False
    return size if acc else -size


@njit(cache=True)
def energy(s, nsub, nfw, fw, fjxy, fjz, K):
    """Energy per spin (meV)."""
    E = 0.0
    nsite = s.shape[0]
    for n in range(nsite):
        for c in range(nfw[n]):
            t = fw[n, c]
            a = fjxy[n, c]
            z = fjz[n, c]
            if a == z:
                E -= a * (s[n, 0] * s[t, 0] + s[n, 1] * s[t, 1] + s[n, 2] * s[t, 2])
            else:
                E -= a * (s[n, 0] * s[t, 0] + s[n, 1] * s[t, 1]) + z * (s[n, 2] * s[t, 2])
        u = n % nsub
        E += K[u, 0] * s[n, 0] ** 2 + K[u, 1] * s[n, 1] ** 2 + K[u, 2] * s[n, 2] ** 2
    return E / nsite


@njit(cache=True)
def order(s, eta):
    mx = my = mz = 0.0
    nsite = s.shape[0]
    for n in range(nsite):
        mx += eta[n] * s[n, 0]
        my += eta[n] * s[n, 1]
        mz += eta[n] * s[n, 2]
    return mx / nsite, my / nsite, mz / nsite


@njit(cache=True)
def sweep(s, beta, L1, L2, L3, nsub, nnb, nbr, jxy, jz, K, hasK, ising, met, npass, nover, nwolff, sigma,
          stack, member, inc):
    """One composite sweep.  Returns (Metropolis acceptance fraction, mean Wolff cluster size)."""
    acc = 0
    if met != 3:
        for _ in range(npass):
            acc += metropolis_pass(s, beta, L1, L2, L3, nsub, nnb, nbr, jxy, jz, K, met, sigma)
    if not ising:
        for _ in range(nover):
            overrelax_pass(s, beta, nsub, nnb, nbr, jxy, jz, K, hasK)
    csize = 0
    for _ in range(nwolff):
        c = wolff_cluster(s, beta, nsub, nnb, nbr, jxy, jz, K, ising, stack, member, inc)
        csize += abs(c)
    nsite = s.shape[0]
    fr = acc / (nsite * npass) if (met != 3 and npass > 0) else 0.0
    return fr, (csize / nwolff if nwolff > 0 else 0.0)


@njit(cache=True)
def run_T(s, T, L1, L2, L3, nsub, nnb, nbr, jxy, jz, nfw, fw, fjxy, fjz, K, eta, ising, met, npass, nover,
          nwolff, sigma0, target, ntherm, nmeas, every):
    """Anneal protocol at one temperature, continuing from s.

    Returns acc[9] (sums of |M|, |M|^2, |M|^4, |Mz|, Mz^2, Mz^4, E, E^2, count), the time series
    ser[nsample, 2] of (|M|, E), and diag[3] = (final cone width, Metropolis acceptance, cluster size).
    """
    beta = 1.0 / (kB * T)
    nsite = s.shape[0]
    hasK = False
    for u in range(nsub):
        for a in range(3):
            if K[u, a] != 0.0:
                hasK = True
    stack = np.empty(nsite, np.int64)
    member = np.empty(nsite, np.int64)
    inc = np.zeros(nsite, np.bool_)
    sigma = sigma0
    for _ in range(ntherm):
        fr, _c = sweep(s, beta, L1, L2, L3, nsub, nnb, nbr, jxy, jz, K, hasK, ising, met, npass, nover, nwolff,
                       sigma, stack, member, inc)
        if met == 1:   # adapt the cone during thermalisation only
            sigma *= math.exp(fr - target)
            sigma = min(max(sigma, 1e-3), 10.0)
    acc = np.zeros(9)
    nser = (nmeas + every - 1) // every
    ser = np.zeros((nser, 2))
    facc = 0.0
    fcl = 0.0
    m = 0
    for k in range(nmeas):
        fr, cl = sweep(s, beta, L1, L2, L3, nsub, nnb, nbr, jxy, jz, K, hasK, ising, met, npass, nover, nwolff,
                       sigma, stack, member, inc)
        facc += fr
        fcl += cl
        if k % every == 0:
            mx, my, mz = order(s, eta)
            m2 = mx * mx + my * my + mz * mz
            e = energy(s, nsub, nfw, fw, fjxy, fjz, K)
            acc[0] += math.sqrt(m2); acc[1] += m2; acc[2] += m2 * m2
            acc[3] += abs(mz); acc[4] += mz * mz; acc[5] += mz ** 4
            acc[6] += e; acc[7] += e * e; acc[8] += 1.0
            ser[m, 0] = math.sqrt(m2)
            ser[m, 1] = e
            m += 1
    diag = np.array([sigma, facc / nmeas, fcl / nmeas])
    return acc, ser, diag


@njit(cache=True)
def run_pt(S, temps, L1, L2, L3, nsub, nnb, nbr, jxy, jz, nfw, fw, fjxy, fjz, K, eta, ising, met, npass, nover,
           nwolff, sigma0, target, ntherm, nmeas, every, swap_every):
    """Parallel tempering: replica r sits at temps[r]; configurations are exchanged between neighbouring
    temperatures every `swap_every` sweeps (alternating even/odd pairs).  Returns acc[R, 9],
    ser[R, nsample, 2], diag[R, 3] and the swap acceptance per pair, swp[R-1]."""
    R = temps.shape[0]
    nsite = S.shape[1]
    hasK = False
    for u in range(nsub):
        for a in range(3):
            if K[u, a] != 0.0:
                hasK = True
    stack = np.empty(nsite, np.int64)
    member = np.empty(nsite, np.int64)
    inc = np.zeros(nsite, np.bool_)
    betas = 1.0 / (kB * temps)
    sig = np.full(R, sigma0)
    Ecur = np.zeros(R)
    nser = (nmeas + every - 1) // every
    acc = np.zeros((R, 9))
    ser = np.zeros((R, nser, 2))
    facc = np.zeros(R)
    fcl = np.zeros(R)
    swp = np.zeros(R - 1 if R > 1 else 1)
    natt = np.zeros(R - 1 if R > 1 else 1)
    tmp = np.empty((nsite, 3))
    m = 0
    parity = 0
    for it in range(ntherm + nmeas):
        meas = it >= ntherm
        for r in range(R):
            fr, cl = sweep(S[r], betas[r], L1, L2, L3, nsub, nnb, nbr, jxy, jz, K, hasK, ising, met, npass, nover,
                           nwolff, sig[r], stack, member, inc)
            if not meas and met == 1:
                sig[r] = min(max(sig[r] * math.exp(fr - target), 1e-3), 10.0)
            if meas:
                facc[r] += fr
                fcl[r] += cl
        if R > 1 and (it + 1) % swap_every == 0:
            for r in range(R):
                Ecur[r] = energy(S[r], nsub, nfw, fw, fjxy, fjz, K) * nsite
            for r in range(parity, R - 1, 2):
                d = (betas[r] - betas[r + 1]) * (Ecur[r] - Ecur[r + 1])
                natt[r] += 1.0
                if d >= 0.0 or np.random.random() < math.exp(d):
                    tmp[:, :] = S[r]
                    S[r][:, :] = S[r + 1]
                    S[r + 1][:, :] = tmp
                    swp[r] += 1.0
            parity = 1 - parity
        if meas and (it - ntherm) % every == 0:
            for r in range(R):
                mx, my, mz = order(S[r], eta)
                m2 = mx * mx + my * my + mz * mz
                e = energy(S[r], nsub, nfw, fw, fjxy, fjz, K)
                acc[r, 0] += math.sqrt(m2); acc[r, 1] += m2; acc[r, 2] += m2 * m2
                acc[r, 3] += abs(mz); acc[r, 4] += mz * mz; acc[r, 5] += mz ** 4
                acc[r, 6] += e; acc[r, 7] += e * e; acc[r, 8] += 1.0
                ser[r, m, 0] = math.sqrt(m2)
                ser[r, m, 1] = e
            m += 1
    diag = np.zeros((R, 3))
    for r in range(R):
        diag[r, 0] = sig[r]
        diag[r, 1] = facc[r] / nmeas
        diag[r, 2] = fcl[r] / nmeas
    for r in range(R - 1):
        if natt[r] > 0:
            swp[r] /= natt[r]
    return acc, ser, diag, swp
