"""Zero-temperature energies of collinear orders."""
import numpy as np
import pytest

import magtc
from magtc.ground import ground
from tests.test_regression_reference import rect_model


def energies(g):
    return {(r["order"], r["axis"]): r["E_meV_per_spin"] for r in g["energies"]}


@pytest.mark.parametrize("J,K", [([32.36, -7.74, 32.51, 0, 0], [0.142775, 0.142775, 0.0]),
                                 ([-4.0, 3.0, 0.5, 0.7, -0.4], [0.1, 0.2, -0.3])])
def test_rect_matches_reference_and_analytic(J, K):
    import tc_jcp_rect as ref
    J1, J2, J3, J4, J5 = J
    E = energies(ground(rect_model(J, K)))
    analytic = {"FM": -(J1 + J2 + 2 * J3 + J4 + J5), "stripe_a": -(-J1 + J2 - 2 * J3 + J4 + J5),
                "stripe_b": -(J1 - J2 - 2 * J3 + J4 + J5), "Neel": -(-J1 - J2 + 2 * J3 + J4 + J5)}
    L = 8
    for o, e0 in analytic.items():
        s = np.zeros((L, L, 3)); s[:, :, 2] = ref.eta_of(o, L)
        e_ref = float(ref.energy(s, L, np.array(J, float), np.array(K, float)))
        assert E[(o, "z")] == pytest.approx(e0 + K[2], abs=1e-12)
        assert E[(o, "z")] == pytest.approx(e_ref, abs=1e-12)
        assert E[(o, "x")] == pytest.approx(e0 + K[0], abs=1e-12)


def test_vn_ground_is_fm_out_of_plane():
    g = ground(magtc.load_model("examples/vn_b2p7342.yaml"))
    assert g["ground_order"] == "FM" and g["ground_axis"] == "z" and g["mc_order_is_ground"]
    assert g["E_ground"] == pytest.approx(-(32.36 - 7.74 + 2 * 32.51), abs=1e-10)


def test_honeycomb_xxz():
    m = magtc.load_model("examples/cri3_lado2017.yaml")
    Jxy, Jz = 4.95, 5.1525
    E = energies(ground(m))
    # 3 NN bonds per 2 spins: FM -3/2 J, Neel +3/2 J, zigzag -1/2 J, stripy +1/2 J
    for o, f in (("FM", -1.5), ("Neel", 1.5), ("zigzag", -0.5), ("stripy", 0.5)):
        assert E[(o, "z")] == pytest.approx(f * Jz, abs=1e-12)
        assert E[(o, "x")] == pytest.approx(f * Jxy, abs=1e-12)
    assert ground(m)["ground_order"] == "FM"


def test_afm_detected_and_warning():
    m = magtc.load_model(dict(name="afm", lattice={"type": "honeycomb"},
                              exchange=[dict(J=-1.0, bonds=[[0, 1, 0, 0], [0, 1, -1, 0], [0, 1, 0, -1]])],
                              order_parameters={"Neel": {"q": [0, 0], "signs": [1, -1]}}, order="FM"))
    g = ground(m)
    assert g["ground_order"] == "Neel" and not g["mc_order_is_ground"]
