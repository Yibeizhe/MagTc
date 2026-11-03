"""MagTc must reproduce kernels/tc_jcp_rect.py (the validated JCP kernel) bit for bit."""
import os
import subprocess
import sys

import numpy as np
import pytest

import magtc
from magtc import kernel as kn
from magtc.runner import apply_overrides, run_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ("M", "M2", "M4", "Mz", "Mz2", "Mz4", "E", "E2")


def rect_model(J, K, order="FM"):
    names = ["J1", "J2", "J3", "J4", "J5"]
    bonds = [[[0, 0, 1, 0]], [[0, 0, 0, 1]], [[0, 0, 1, 1], [0, 0, 1, -1]], [[0, 0, 2, 0]], [[0, 0, 0, 2]]]
    return magtc.load_model(dict(name="rect", lattice={"type": "rectangular"},
                                 exchange=[dict(name=n, J=j, bonds=b) for n, j, b in zip(names, J, bonds) if j != 0],
                                 anisotropy={"K": K}, order=order))


def reference_rows(J, K, order, L, seed, temps, ntherm, nmeas, every):
    import tc_jcp_rect as ref
    J = np.array(J, dtype=float)
    K = np.array(K, dtype=float)
    eta = ref.eta_of(order, L)
    ref.seed_rng(seed)
    s = np.empty((L, L, 3))
    for i in range(L):
        for j in range(L):
            s[i, j] = ref.rand_spin()
    out = []
    for T in temps:
        acc = ref.run_T(s, float(T), L, J, K, eta, ntherm, nmeas, every)
        out.append(acc[:8] / acc[8])
    return np.array(out)


CASES = [
    # VN b = 2.7342 (rectangular FM J1-J3, K = (D, D, 0))
    ([32.36, -7.74, 32.51, 0.0, 0.0], [0.142775, 0.142775, 0.0], "FM", 8, 20261104),
    # generic rectangular FM J1-J3 with triaxial K
    ([5.0, 2.0, 1.0, 0.0, 0.0], [0.3, -0.1, 0.05], "FM", 6, 7),
    # J1-J5, stripe order parameter
    ([-4.0, 3.0, 0.5, 0.7, -0.4], [0.0, 0.2, -0.3], "stripe_a", 10, 99),
]


@pytest.mark.skipif(not kn.HAVE_NUMBA, reason="bit-for-bit numba comparison needs numba")
@pytest.mark.parametrize("J,K,order,L,seed", CASES)
def test_bit_for_bit_numba(J, K, order, L, seed):
    m = apply_overrides(rect_model(J, K, order), L=L, Tmax=500.0, Tmin=300.0, step=100.0,
                        ntherm=30, nmeas=60, every=7)
    res = run_seed(m, seed, verbose=False)
    new = np.array([[row[k] for k in KEYS] for row in res["rows"]])
    ref = reference_rows(J, K, order, L, seed, [500.0, 400.0, 300.0], 30, 60, 7)
    assert np.array_equal(new, ref), np.max(np.abs(new - ref))


def test_bit_for_bit_pure_python():
    """Same comparison with Numba disabled (NumPy RNG, plain Python) in a subprocess."""
    code = r"""
import sys, numpy as np
sys.path.insert(0, %r); sys.path.insert(0, %r)
from tests.test_regression_reference import rect_model, reference_rows, KEYS
from magtc.runner import apply_overrides, run_seed
from magtc import kernel
assert not kernel.HAVE_NUMBA
J = [5.0, 2.0, 1.0, 0.0, 0.0]; K = [0.3, -0.1, 0.05]
m = apply_overrides(rect_model(J, K), L=4, Tmax=400.0, Tmin=300.0, step=100.0, ntherm=5, nmeas=10, every=3)
new = np.array([[r[k] for k in KEYS] for r in run_seed(m, 3, verbose=False)["rows"]])
ref = reference_rows(J, K, "FM", 4, 3, [400.0, 300.0], 5, 10, 3)
assert np.array_equal(new, ref)
print("OK")
""" % (ROOT, os.path.join(ROOT, "kernels"))
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr


def test_reproducible_same_seed():
    m = apply_overrides(rect_model([5.0, 2.0, 1.0, 0, 0], [0, 0, -0.2]), L=6, Tmax=300.0, Tmin=200.0,
                        step=100.0, ntherm=10, nmeas=20, every=5)
    a = run_seed(m, 11, verbose=False)["rows"]
    b = run_seed(m, 11, verbose=False)["rows"]
    c = run_seed(m, 12, verbose=False)["rows"]
    assert a == b and a != c
