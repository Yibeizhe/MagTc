"""The JCP mode must reproduce kernels/tc_jcp_rect.py (validated, VN JCP 2026) bit for bit."""
import os
import subprocess
import sys

import numpy as np
import pytest

import magtc
from magtc import apply_overrides, kernels as kn, run_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ("M", "M2", "M4", "Mz", "Mz2", "Mz4", "E", "E2")
RECT = [[[0, 0, 1, 0, 0]], [[0, 0, 0, 1, 0]], [[0, 0, 1, 1, 0], [0, 0, 1, -1, 0]], [[0, 0, 2, 0, 0]], [[0, 0, 0, 2, 0]]]


def rect_model(J, K, order="FM"):
    return magtc.load_model(dict(name="rect", structure=dict(dim=2, sites=[{"frac": [0, 0, 0]}]),
                                 interactions=[dict(name="J%d" % (i + 1), J=j, bonds=b)
                                               for i, (j, b) in enumerate(zip(J, RECT)) if j != 0],
                                 anisotropy={"K": K}, order=order))


def reference_rows(J, K, order, L, seed, temps, ntherm, nmeas, every):
    import tc_jcp_rect as ref
    eta = ref.eta_of(order, L)
    ref.seed_rng(seed)
    s = np.empty((L, L, 3))
    for i in range(L):
        for j in range(L):
            s[i, j] = ref.rand_spin()
    out = []
    for T in temps:
        acc = ref.run_T(s, float(T), L, np.array(J, float), np.array(K, float), eta, ntherm, nmeas, every)
        out.append(acc[:8] / acc[8])
    return np.array(out)


CASES = [([32.36, -7.74, 32.51, 0.0, 0.0], [0.142775, 0.142775, 0.0], "FM", 8, 20261104),
         ([5.0, 2.0, 1.0, 0.0, 0.0], [0.3, -0.1, 0.05], "FM", 6, 7),
         ([-4.0, 3.0, 0.5, 0.7, -0.4], [0.0, 0.2, -0.3], "stripe_a", 10, 99)]


@pytest.mark.skipif(not kn.HAVE_NUMBA, reason="needs numba")
@pytest.mark.parametrize("J,K,order,L,seed", CASES)
def test_bit_for_bit_numba(J, K, order, L, seed):
    m = apply_overrides(rect_model(J, K, order), L=L, Tmax=500.0, Tmin=300.0, step=100.0, ntherm=30, nmeas=60, every=7)
    new = np.array([[row[k] for k in KEYS] for row in run_seed(m, seed, verbose=False)["rows"]])
    ref = reference_rows(J, K, order, L, seed, [500.0, 400.0, 300.0], 30, 60, 7)
    assert np.array_equal(new, ref)


def test_bit_for_bit_pure_python():
    code = r"""
import sys, numpy as np
sys.path.insert(0, %r); sys.path.insert(0, %r)
from tests.test_reference import rect_model, reference_rows, KEYS
from magtc import apply_overrides, run_seed, kernels
assert not kernels.HAVE_NUMBA
J = [5.0, 2.0, 1.0, 0.0, 0.0]; K = [0.3, -0.1, 0.05]
m = apply_overrides(rect_model(J, K), L=4, Tmax=400.0, Tmin=300.0, step=100.0, ntherm=5, nmeas=10, every=3)
new = np.array([[r[k] for k in KEYS] for r in run_seed(m, 3, verbose=False)["rows"]])
assert np.array_equal(new, reference_rows(J, K, "FM", 4, 3, [400.0, 300.0], 5, 10, 3))
print("OK")
""" % (ROOT, os.path.join(ROOT, "kernels"))
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, NUMBA_DISABLE_JIT="1"),
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr


def test_v1_model_file_gives_same_run():
    v1 = dict(name="rect", lattice={"type": "rectangular"},
              exchange=[dict(name="J1", J=5.0, bonds=[[0, 0, 1, 0]]), dict(name="J2", J=2.0, bonds=[[0, 0, 0, 1]])],
              anisotropy={"K": [0, 0, -0.2]})
    v2 = rect_model([5.0, 2.0, 0, 0, 0], [0, 0, -0.2])
    ov = dict(L=6, Tmax=300.0, Tmin=200.0, step=100.0, ntherm=10, nmeas=20, every=5)
    a = run_seed(apply_overrides(magtc.load_model(v1), **ov), 4, verbose=False)["rows"]
    b = run_seed(apply_overrides(v2, **ov), 4, verbose=False)["rows"]
    assert [r["E"] for r in a] == [r["E"] for r in b]
