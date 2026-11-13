"""Neighbour shells, collinear ground states, mean-field estimate."""
import numpy as np
import pytest

import magtc
from magtc.ground import ground, mean_field_tc
from magtc.structure import Structure
from tests.test_reference import rect_model


def energies(g):
    return {(r["order"], r["axis"]): r["E_meV_per_spin"] for r in g["energies"]}


def test_shells_honeycomb_and_cubic():
    a = 6.9
    hc = Structure([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 20]], [[1 / 3, 2 / 3, 0], [2 / 3, 1 / 3, 0]], 2)
    sh = hc.shells(3)
    assert [len(b) for _, b in sh] == [3, 6, 3]
    assert [r for r, _ in sh] == pytest.approx([a / np.sqrt(3), a, 2 * a / np.sqrt(3)])
    sc = Structure(np.eye(3) * 4.0, [[0, 0, 0]], 3)
    assert [len(b) for _, b in sc.shells(3)] == [3, 6, 4]      # 6, 12, 8 neighbours
    bcc = Structure(np.eye(3) * 3.0, [[0, 0, 0], [0.5, 0.5, 0.5]], 3)
    r, b = bcc.shells(1)[0]
    assert len(b) == 8 and r == pytest.approx(3.0 * np.sqrt(3) / 2)   # 8 neighbours per site, 2 sites


@pytest.mark.parametrize("J,K", [([32.36, -7.74, 32.51, 0, 0], [0.142775, 0.142775, 0.0]),
                                 ([-4.0, 3.0, 0.5, 0.7, -0.4], [0.1, 0.2, -0.3])])
def test_rect_matches_reference_and_analytic(J, K):
    import tc_jcp_rect as ref
    J1, J2, J3, J4, J5 = J
    E = energies(ground(rect_model(J, K)))
    analytic = {"FM": -(J1 + J2 + 2 * J3 + J4 + J5), "stripe_a": -(-J1 + J2 - 2 * J3 + J4 + J5),
                "stripe_b": -(J1 - J2 - 2 * J3 + J4 + J5), "Neel": -(-J1 - J2 + 2 * J3 + J4 + J5)}
    for o, e0 in analytic.items():
        s = np.zeros((8, 8, 3)); s[:, :, 2] = ref.eta_of(o, 8)
        assert E[(o, "z")] == pytest.approx(e0 + K[2], abs=1e-12)
        assert E[(o, "z")] == pytest.approx(float(ref.energy(s, 8, np.array(J, float), np.array(K, float))), abs=1e-12)
        assert E[(o, "x")] == pytest.approx(e0 + K[0], abs=1e-12)


def test_honeycomb_xxz_and_cubic_afm():
    m = magtc.load_model("examples/cri3_lado2017.yaml")
    E = energies(ground(m))
    for o, f in (("FM", -1.5), ("Neel", 1.5), ("zigzag", -0.5), ("stripy", 0.5)):
        assert E[(o, "z")] == pytest.approx(f * 5.1525, abs=1e-12)
        assert E[(o, "x")] == pytest.approx(f * 4.95, abs=1e-12)
    afm = magtc.load_model(dict(name="g", structure=dict(dim=3, lattice=np.eye(3).tolist(), sites=[{"frac": [0, 0, 0]}]),
                                interactions=[dict(J=-1.0, shell=1)]))
    g = ground(afm)
    assert g["ground_order"] == "q=(1,1,1) signs=(+)" and g["E_ground"] == pytest.approx(-3.0)
    assert not g["mc_order_is_ground"]


def test_mean_field():
    m = magtc.load_model("examples/sc_heisenberg.yaml")      # z J / (3 kB) = 6/(3 kB)
    assert mean_field_tc(m) == pytest.approx(2.0 / 0.0861733)
