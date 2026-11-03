"""Monte Carlo kernel of MagTc.

This is kernels/tc_jcp_rect.py (the validated JCP-protocol kernel) rewritten on a
general site/neighbour table, so that any 2D Bravais lattice with any number of
sublattices and any explicit bond list can be simulated.  The arithmetic and the
random-number sequence are kept identical to the reference kernel:

* one sweep = Lx*Ly*nsub single-spin attempts; each attempt draws
  i = randint(0, Lx), j = randint(0, Ly) and, only if nsub > 1, u = randint(0, nsub);
* the new direction is uniform on the sphere (z = 2r-1, phi = 2 pi r');
* Metropolis acceptance  dE <= 0 or random() < exp(-beta dE);
* local fields are summed shell by shell in the order of the model file and, inside
  a shell, in descending (da, db) order, which is the order of the reference kernel;
* energies are summed site by site over the forward bonds in model-file order, then
  the single-ion term.

For the rectangular J1-J3 FM case this reproduces the reference kernel bit for bit
(tests/test_regression_reference.py).

Numba is optional.  Without it the very same functions run as plain Python with
NumPy's legacy global RNG (slow, intended for tests and tiny systems).  Note that
Numba's RNG and NumPy's RNG do not give the same stream for the same seed, so a run
is reproducible only with the same backend; the JSON output records the backend.

Hamiltonian (each pair once, J > 0 ferromagnetic, classical unit vectors):
    H = - sum_<ij> [ Jxy (Sx Sx' + Sy Sy') + Jz Sz Sz' ] + sum_i sum_a K_a(u_i) (S_i^a)^2
"""
import math
import os

import numpy as np

try:  # pragma: no cover - exercised through both branches in CI matrix
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

kB = 0.0861733  # meV/K  (value of the validated reference kernel)

MOVE_UNIFORM = 0   # JCP protocol: new direction uniform on the sphere
MOVE_FLIP = 1      # Ising mode: S -> -S (spins stay on +-z)


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
def init_spins(nsite, move):
    """Random initial state, drawn site by site (same order as the reference)."""
    s = np.empty((nsite, 3))
    for n in range(nsite):
        if move == 1:
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
def sweep(s, beta, Lx, Ly, nsub, nnb, nbr, jxy, jz, K, move):
    for _ in range(Lx * Ly * nsub):
        i = np.random.randint(0, Lx)
        j = np.random.randint(0, Ly)
        if nsub > 1:
            u = np.random.randint(0, nsub)
        else:
            u = 0
        n = (i * Ly + j) * nsub + u
        hx = hy = hz = 0.0
        for k in range(nnb[n]):
            t = nbr[n, k]
            hx += jxy[n, k] * s[t, 0]
            hy += jxy[n, k] * s[t, 1]
            hz += jz[n, k] * s[t, 2]
        x0, y0, z0 = s[n, 0], s[n, 1], s[n, 2]
        if move == 1:
            x1, y1, z1 = -x0, -y0, -z0
        else:
            x1, y1, z1 = rand_spin()
        dE = -((x1 - x0) * hx + (y1 - y0) * hy + (z1 - z0) * hz)
        dE += K[u, 0] * (x1 * x1 - x0 * x0) + K[u, 1] * (y1 * y1 - y0 * y0) + K[u, 2] * (z1 * z1 - z0 * z0)
        if dE <= 0.0 or np.random.random() < math.exp(-beta * dE):
            s[n, 0] = x1
            s[n, 1] = y1
            s[n, 2] = z1


@njit(cache=True)
def energy(s, nsub, nfw, fw, fjxy, fjz, K):
    """Energy per spin (meV)."""
    E = 0.0
    nsite = s.shape[0]
    for n in range(nsite):
        for k in range(nfw[n]):
            t = fw[n, k]
            a = fjxy[n, k]
            c = fjz[n, k]
            if a == c:
                E -= a * (s[n, 0] * s[t, 0] + s[n, 1] * s[t, 1] + s[n, 2] * s[t, 2])
            else:
                E -= a * (s[n, 0] * s[t, 0] + s[n, 1] * s[t, 1]) + c * (s[n, 2] * s[t, 2])
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
def run_T(s, T, Lx, Ly, nsub, nnb, nbr, jxy, jz, nfw, fw, fjxy, fjz, K, eta, move, ntherm, nmeas, every):
    """Thermalise and measure at one temperature, starting from (and updating) s.

    Returns acc[9] = sums of |M|, |M|^2, |M|^4, |Mz|, Mz^2, Mz^4, E, E^2 and the count.
    """
    beta = 1.0 / (kB * T)
    for _ in range(ntherm):
        sweep(s, beta, Lx, Ly, nsub, nnb, nbr, jxy, jz, K, move)
    acc = np.zeros(9)
    for k in range(nmeas):
        sweep(s, beta, Lx, Ly, nsub, nnb, nbr, jxy, jz, K, move)
        if k % every == 0:
            mx, my, mz = order(s, eta)
            m2 = mx * mx + my * my + mz * mz
            e = energy(s, nsub, nfw, fw, fjxy, fjz, K)
            acc[0] += math.sqrt(m2); acc[1] += m2; acc[2] += m2 * m2
            acc[3] += abs(mz); acc[4] += mz * mz; acc[5] += mz ** 4
            acc[6] += e; acc[7] += e * e; acc[8] += 1.0
    return acc
