"""One seed, one temperature ladder (the `run` command)."""
import copy
import json
import time

from . import __version__
from . import kernel as kn
from .model import build_tables, eta_array, lattice_size, model_hash, temperatures


def apply_overrides(m, **ov):
    """Return a copy of model m with mc settings overridden (None values are ignored)."""
    m = copy.deepcopy(m)
    for k, v in ov.items():
        if v is not None:
            m["mc"][k] = v
    if ov.get("Tmax") is not None or ov.get("Tmin") is not None:
        m["mc"]["temperatures"] = None
    if ov.get("nseeds") is not None or ov.get("seed0") is not None:
        m["mc"]["seeds"] = None
    return m


def run_seed(m, seed, out=None, verbose=True, log=print):
    """Run the full temperature ladder for one seed.  Returns the result dict (also written to `out`)."""
    Lx, Ly = lattice_size(m)
    tb = build_tables(m, Lx, Ly)
    op = m["order_parameters"][m["order"]]
    eta = eta_array(op, Lx, Ly, tb["nsub"])
    move = kn.MOVE_FLIP if m["spin"] == "ising" else kn.MOVE_UNIFORM
    mc = m["mc"]
    temps = temperatures(m)

    kn.seed_rng(int(seed))
    s = kn.init_spins(tb["nsite"], move)
    rows, t0 = [], time.time()
    for T in temps:
        acc = kn.run_T(s, float(T), Lx, Ly, tb["nsub"], tb["nnb"], tb["nbr"], tb["jxy"], tb["jz"],
                       tb["nfw"], tb["fw"], tb["fjxy"], tb["fjz"], tb["K"], eta, move,
                       int(mc["ntherm"]), int(mc["nmeas"]), int(mc["every"]))
        n = acc[8]
        rows.append(dict(T=float(T), M=acc[0] / n, M2=acc[1] / n, M4=acc[2] / n, Mz=acc[3] / n,
                         Mz2=acc[4] / n, Mz4=acc[5] / n, E=acc[6] / n, E2=acc[7] / n, n=int(n)))
        if verbose:
            log("%s seed %d T=%.2f |M|=%.4f  (%.0f s)" % (m["name"], seed, T, acc[0] / n, time.time() - t0))
    res = dict(magtc_version=__version__, tag=m["name"], model_hash=model_hash(m), model=m,
               order=m["order"], spin=m["spin"], L=[Lx, Ly], nsub=tb["nsub"], nsite=tb["nsite"],
               seed=int(seed), ntherm=int(mc["ntherm"]), nmeas=int(mc["nmeas"]), every=int(mc["every"]),
               numba=kn.HAVE_NUMBA, kB=kn.kB, wall_s=time.time() - t0, rows=rows)
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
    return res
