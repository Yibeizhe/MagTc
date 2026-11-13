"""Every move set must sample the Boltzmann distribution: exact XXZ + anisotropy dimer."""
import numpy as np
import pytest

import magtc
from magtc import apply_overrides, run_seed
from magtc.kernels import kB

Jxy, Jz, Kz, T = 1.0, 1.6, -0.4, 10.0


def exact_energy_per_spin():
    x, w = np.polynomial.legendre.leggauss(80)
    phi = np.linspace(0, 2 * np.pi, 161)[:-1]
    z1, z2, p = np.meshgrid(x, x, phi, indexing="ij")
    W = np.einsum("i,j->ij", w, w)[:, :, None]
    s1, s2 = np.sqrt(1 - z1 ** 2), np.sqrt(1 - z2 ** 2)
    Ed = -(Jxy * s1 * s2 * np.cos(p) + Jz * z1 * z2) + Kz * (z1 ** 2 + z2 ** 2)   # dimer energy
    bw = W * np.exp(-(Ed - Ed.min()) / (kB * T))
    return float((Ed * bw).sum() / bw.sum()) / 2


BASE = dict(name="dimer", structure=dict(dim=2, sites=[{"frac": [0, 0, 0]}, {"frac": [0.5, 0, 0]}]),
            interactions=[dict(Jxy=Jxy, Jz=Jz, bonds=[[0, 1, 0, 0, 0]])], anisotropy={"K": [0, 0, Kz]},
            mc=dict(L=[4, 1], temperatures=[T], ntherm=2000, nmeas=100000, every=2, nseeds=1))
EXACT = exact_energy_per_spin()


@pytest.mark.parametrize("moves,protocol", [
    ({"metropolis": "uniform"}, "anneal"),
    ({"metropolis": "cone"}, "anneal"),
    ({"metropolis": "uniform", "overrelax": 2}, "anneal"),
    ({"metropolis": "none", "wolff": 4}, "anneal"),
    ({"metropolis": "cone", "overrelax": 1, "wolff": 2}, "anneal"),
    ({"metropolis": "uniform", "wolff": 1}, "pt"),
])
def test_dimer_exact(moves, protocol):
    m = apply_overrides(magtc.load_model(BASE), moves=moves, protocol=protocol)
    if protocol == "pt":
        m = apply_overrides(m, temperatures=[14.0, 12.0, T])
    row = run_seed(m, 7, verbose=False)["rows"][-1]
    # variance of the mean x 2 tau (tau in samples = tau_E / every, every = 2)
    err = np.sqrt((row["E2"] - row["E"] ** 2) / row["n"] * max(1.0, 2 * row["tau_E"] / 2))
    assert abs(row["E"] - EXACT) < 4.5 * err + 1e-4, (row["E"], EXACT, err)


def test_pt_swaps_and_overrelax_conserve_exchange():
    m = apply_overrides(magtc.load_model(BASE), temperatures=[14.0, 12.0, T], protocol="pt", nmeas=2000)
    r = run_seed(m, 1, verbose=False)
    assert len(r["pt_swap_acceptance"]) == 2 and all(0 < a <= 1 for a in r["pt_swap_acceptance"])
    from magtc import kernels as kn
    from magtc.lattice import build_tables
    mm = magtc.load_model(dict(BASE, anisotropy={"K": [0, 0, 0]}))
    tb = build_tables(mm, (4, 1, 1))
    kn.seed_rng(3)
    s = kn.init_spins(tb["nsite"], False)
    e0 = kn.energy(s, 2, tb["nfw"], tb["fw"], tb["fjxy"], tb["fjz"], tb["K"])
    kn.overrelax_pass(s, 1.0, 2, tb["nnb"], tb["nbr"], tb["jxy"], tb["jz"], tb["K"], False)
    assert kn.energy(s, 2, tb["nfw"], tb["fw"], tb["fjxy"], tb["fjz"], tb["K"]) == pytest.approx(e0, abs=1e-12)
