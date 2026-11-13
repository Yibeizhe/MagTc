"""Known transition temperatures: 2D Ising (Onsager), 3D Ising and Heisenberg on the simple-cubic lattice."""
import math

import numpy as np
import pytest

import magtc
from magtc import analyse, apply_overrides, run_seed
from magtc.kernels import kB
from magtc.model import seeds

J = 1.0
TC2D = 2 * J / math.log(1 + math.sqrt(2)) / kB


def onsager_energy(T):
    K = J / (kB * T)
    k = 2 * math.sinh(2 * K) / math.cosh(2 * K) ** 2
    a, b = 1.0, math.sqrt(max(1e-300, 1 - k * k))
    for _ in range(60):
        a, b = (a + b) / 2, math.sqrt(a * b)
    return -J / math.tanh(2 * K) * (1 + 2 / math.pi * (2 * math.tanh(2 * K) ** 2 - 1) * math.pi / (2 * a))


def test_onsager_energy():
    m = apply_overrides(magtc.load_model("examples/ising_square.yaml"), L=24, temperatures=[40.0, 20.0],
                        ntherm=2000, nmeas=6000, every=5, nseeds=1)
    for row in run_seed(m, 5, verbose=False)["rows"]:
        assert row["E"] == pytest.approx(onsager_energy(row["T"]), rel=0.01)


def test_ising_2d_tc():
    m = apply_overrides(magtc.load_model("examples/ising_square.yaml"), L=16, Tmax=30.0, Tmin=23.0, step=0.5,
                        ntherm=1000, nmeas=8000, every=5, nseeds=4)
    res = analyse([run_seed(m, s, verbose=False) for s in seeds(m)])
    assert 1.0 < res["Tc_chi_abs"] / TC2D < 1.09 and abs(res["Tc_Cv"] / TC2D - 1) < 0.04


def test_ising_3d_tc():
    tc = 4.511528 / kB
    m = apply_overrides(magtc.load_model("examples/sc_ising.yaml"), L=6, Tmax=60.0, Tmin=45.0, step=1.0,
                        ntherm=500, nmeas=4000, every=2, nseeds=2)
    res = analyse([run_seed(m, s, verbose=False) for s in seeds(m)])
    assert abs(res["Tc_chi_abs"] / tc - 1) < 0.06


def _binder_crossing(r1, r2):
    T = np.array(r1["curve"]["T"])
    d = np.array(r1["curve"]["U"]) - np.array(r2["curve"]["U"])
    i = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
    k = i[0]
    return T[k] - d[k] * (T[k + 1] - T[k]) / (d[k + 1] - d[k])


@pytest.mark.slow
def test_heisenberg_3d_binder_crossing():
    tc = 1.4430 / kB
    res = {}
    for L in (6, 10):
        m = apply_overrides(magtc.load_model("examples/sc_heisenberg.yaml"), L=L, Tmax=18.5, Tmin=15.0, step=0.5,
                            ntherm=1000, nmeas=10000, every=2, nseeds=2)
        res[L] = analyse([run_seed(m, s, verbose=False) for s in seeds(m)])
    assert abs(_binder_crossing(res[6], res[10]) / tc - 1) < 0.02
