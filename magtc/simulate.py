"""One seed over one temperature ladder (`magtc run`): anneal (JCP) or parallel-tempering protocol."""
import json
import time

import numpy as np

from . import __version__
from . import kernels as kn
from .lattice import build_tables, eta_array
from .model import lattice_size, model_hash, temperatures

MET = {"uniform": kn.MET_UNIFORM, "cone": kn.MET_CONE, "none": kn.MET_NONE}


def tau_int(x, min_bins=32):
    """Integrated autocorrelation time (in samples) from a binning analysis:
    tau = 0.5 * var(mean, binned) / var(mean, naive), at the largest block with >= min_bins blocks."""
    x = np.asarray(x, float)
    n = len(x)
    v0 = x.var(ddof=1) / n if n > 1 else 0.0
    if v0 <= 0:
        return 0.5
    best = v0
    b = 1
    while n // (2 * b) >= min_bins:
        b *= 2
        nb = n // b
        y = x[:nb * b].reshape(nb, b).mean(axis=1)
        best = max(best, y.var(ddof=1) / nb)
    return 0.5 * best / v0


def _row(T, a, ser, diag, every):
    n = a[8]
    return dict(T=float(T), M=a[0] / n, M2=a[1] / n, M4=a[2] / n, Mz=a[3] / n, Mz2=a[4] / n, Mz4=a[5] / n,
                E=a[6] / n, E2=a[7] / n, n=int(n),
                tau_M=float(tau_int(ser[:, 0]) * every), tau_E=float(tau_int(ser[:, 1]) * every),
                cone=float(diag[0]), acc_metropolis=float(diag[1]), cluster=float(diag[2]))


def run_seed(m, seed, out=None, verbose=True, log=print):
    L = lattice_size(m)
    tb = build_tables(m, L)
    eta = eta_array(m["orders"][m["order"]], L, tb["nsub"])
    mc, mv = m["mc"], m["mc"]["moves"]
    ising = m["spin"] == "ising"
    met = kn.MET_FLIP if (ising and mv["metropolis"] != "none") else MET[mv["metropolis"]]
    temps = temperatures(m)
    args = (L[0], L[1], L[2], tb["nsub"], tb["nnb"], tb["nbr"], tb["jxy"], tb["jz"], tb["nfw"], tb["fw"],
            tb["fjxy"], tb["fjz"], tb["K"], eta, ising, met, int(mv["metropolis_passes"]), int(mv["overrelax"]),
            int(mv["wolff"]), 0.5, float(mv["cone_target"]))
    kn.seed_rng(int(seed))
    t0 = time.time()
    rows = []
    extra = {}
    if mc["protocol"] == "anneal":
        s = kn.init_spins(tb["nsite"], ising)
        for T in temps:
            a, ser, diag = kn.run_T(s, float(T), *args, int(mc["ntherm"]), int(mc["nmeas"]), int(mc["every"]))
            rows.append(_row(T, a, ser, diag, mc["every"]))
            if verbose:
                log("%s seed %d T=%.2f |M|=%.4f tau_M=%.0f sweeps (%.0f s)"
                    % (m["name"], seed, T, rows[-1]["M"], rows[-1]["tau_M"], time.time() - t0))
    else:
        S = np.empty((len(temps), tb["nsite"], 3))
        for r in range(len(temps)):
            S[r] = kn.init_spins(tb["nsite"], ising)
        a, ser, diag, swp = kn.run_pt(S, np.asarray(temps, float), *args, int(mc["ntherm"]), int(mc["nmeas"]),
                                      int(mc["every"]), int(mc["pt_swap_every"]))
        rows = [_row(T, a[r], ser[r], diag[r], mc["every"]) for r, T in enumerate(temps)]
        extra["pt_swap_acceptance"] = swp[:max(0, len(temps) - 1)].tolist()
        if verbose:
            log("%s seed %d PT done, min swap acceptance %.2f (%.0f s)"
                % (m["name"], seed, min(extra["pt_swap_acceptance"] or [1.0]), time.time() - t0))
    res = dict(magtc_version=__version__, tag=m["name"], model_hash=model_hash(m), model=m, order=m["order"],
               spin=m["spin"], L=list(L), dim=m["structure"]["dim"], nsub=tb["nsub"], nsite=tb["nsite"],
               seed=int(seed), protocol=mc["protocol"], moves=mv, ntherm=int(mc["ntherm"]), nmeas=int(mc["nmeas"]),
               every=int(mc["every"]), numba=kn.HAVE_NUMBA, kB=kn.kB, wall_s=time.time() - t0, rows=rows, **extra)
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
    return res
