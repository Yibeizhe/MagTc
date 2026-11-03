"""2D square-lattice Ising limit against Onsager's exact solution."""
import math

import numpy as np
import pytest

import magtc
from magtc.analysis import analyse
from magtc.kernel import kB
from magtc.model import seeds
from magtc.runner import apply_overrides, run_seed

J = 1.0  # meV
TC = 2 * J / math.log(1 + math.sqrt(2)) / kB   # 26.333 K


def onsager_energy(T):
    """Exact energy per spin of the infinite square-lattice Ising model (units of J)."""
    K = J / (kB * T)
    k = 2 * math.sinh(2 * K) / math.cosh(2 * K) ** 2
    a, b = 1.0, math.sqrt(max(1e-300, 1 - k * k))
    for _ in range(60):
        a, b = (a + b) / 2, math.sqrt(a * b)
    Kk = math.pi / (2 * a)
    return -J / math.tanh(2 * K) * (1 + 2 / math.pi * (2 * math.tanh(2 * K) ** 2 - 1) * Kk)


def test_onsager_helper():
    assert onsager_energy(TC * 1.0001) == pytest.approx(-math.sqrt(2), abs=2e-3)
    assert onsager_energy(1e4) == pytest.approx(-2 * J / (kB * 1e4), rel=1e-3)   # high-T: -2 J^2/kT


def test_energy_away_from_tc():
    m = apply_overrides(magtc.load_model("examples/ising_square.yaml"), L=24,
                        temperatures=[40.0, 20.0], ntherm=2000, nmeas=6000, every=5, nseeds=1)
    rows = run_seed(m, 5, verbose=False)["rows"]
    for row in rows:
        assert row["E"] == pytest.approx(onsager_energy(row["T"]), rel=0.01)


def test_ising_tc_near_onsager():
    m = apply_overrides(magtc.load_model("examples/ising_square.yaml"), L=16, Tmax=30.0, Tmin=23.0, step=0.5,
                        ntherm=1000, nmeas=12000, every=5, nseeds=4)
    res = analyse([run_seed(m, s, verbose=False) for s in seeds(m)])
    # finite-size shift of the chi_abs peak at L = 16 is about +5 %; C_v peak lies within ~2 %
    assert 1.0 < res["Tc_chi_abs"] / TC < 1.09
    assert abs(res["Tc_Cv"] / TC - 1) < 0.04
    assert not res["peak_at_edge"]


@pytest.mark.slow
def test_heisenberg_large_anisotropy_limit():
    """Heisenberg spins with a strong easy axis approach the Ising transition from below."""
    m = magtc.load_model(dict(name="hz", lattice={"type": "square"},
                              exchange=[dict(J=J, bonds=[[0, 0, 1, 0], [0, 0, 0, 1]])],
                              anisotropy={"K": [0.0, 0.0, -30.0]}))
    m = apply_overrides(m, L=16, Tmax=29.0, Tmin=21.0, step=1.0, ntherm=4000, nmeas=40000, every=10, nseeds=4)
    res = analyse([run_seed(m, s, verbose=False) for s in seeds(m)])
    assert 0.9 < res["Tc_Cv"] / TC < 1.05
