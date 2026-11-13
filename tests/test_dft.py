"""VASP energy mapping on synthetic outputs with known J and K."""
import os

import numpy as np
import pytest
import yaml

import magtc
from magtc.dft import collect, mapping, setup, vasp

POSCAR = """CrX3-like test
1.0
 6.9 0.0 0.0
 -3.45 5.975575 0.0
 0.0 0.0 20.0
Cr I
2 1
Direct
0.333333333333 0.666666666667 0.5
0.666666666667 0.333333333333 0.5
0.0 0.0 0.55
"""
JTRUE, KTRUE, E0 = [2.5, 0.6, -0.15], [0.4, 0.4, 0.0], -21.234


def write_fake_outputs(out, plan, noise=0.0, flip=None, rng=None):
    shells = [sh["bonds"] for sh in plan["shells"]]
    M = plan["supercell"]
    fm = mapping.correlations(np.ones(tuple(M) + (len(plan["sites"]),)), shells, M)
    for job in plan["jobs"]:
        if job["config"] is not None:
            X = mapping.correlations(np.array(job["config"]), shells, M)
            E = E0 + np.dot(JTRUE[:len(X)], X) * 1e-3
        else:
            E = E0 + np.dot(JTRUE[:len(fm)], fm) * 1e-3 + plan["n_mag_supercell"] * KTRUE[job["axis"]] * 1e-3
        if noise:
            E += rng.normal(0, noise)
        d = os.path.join(out, job["dir"])
        with open(os.path.join(d, "OSZICAR"), "w") as f:
            f.write("   1 F= %.10E E0= %.10E  d E =0.0  mag= 1.0\n" % (E + 0.01, E))
        if flip and job["dir"] == flip:
            with open(os.path.join(d, "OUTCAR"), "w") as f:
                f.write(" magnetization (x)\n\n# of ion       s       p       d       tot\n"
                        "------------------------------------------\n"
                        "    1       -0.01   -0.02   -2.90   -2.93\n    2        0.0     0.0     0.10    0.10\n")


@pytest.fixture
def workdir(tmp_path):
    (tmp_path / "POSCAR").write_text(POSCAR)
    (tmp_path / "tmpl").mkdir()
    (tmp_path / "tmpl" / "INCAR").write_text("ENCUT = 400\nISPIN = 1 ; LDAU = .TRUE.  # U on Cr\nLSORBIT = .TRUE.\n")
    (tmp_path / "tmpl" / "KPOINTS").write_text("auto\n0\nGamma\n12 12 1\n")
    return tmp_path


@pytest.mark.parametrize("method,sc", [("fit", [2, 2]), ("fourstate", [3, 3])])
def test_recover_J_and_K(workdir, method, sc):
    out = str(workdir / method)
    plan = setup(str(workdir / "POSCAR"), ["Cr"], 3, sc, out, template=str(workdir / "tmpl"), method=method,
                 log=lambda s: None)
    write_fake_outputs(out, plan)
    model = collect(out, out=str(workdir / "model.yaml"), name="t", log=lambda s: None)
    assert [sh["J"] for sh in model["interactions"]] == pytest.approx(JTRUE, abs=1e-6)
    assert model["anisotropy"]["K"] == pytest.approx(KTRUE, abs=1e-6)
    m = magtc.load_model(str(workdir / "model.yaml"))
    assert m["order"] == "FM" and m["mc"]["Tmax"] > 0
    incar = open(os.path.join(out, plan["jobs"][0]["dir"], "INCAR")).read()
    assert "ISPIN = 2" in incar and "LSORBIT" not in incar and "LDAU = .TRUE." in incar and "MAGMOM" in incar
    mae = open(os.path.join(out, "mae_x", "INCAR")).read()
    assert "LSORBIT = .TRUE." in mae and "LNONCOLLINEAR = .TRUE." in mae
    kp = open(os.path.join(out, plan["jobs"][0]["dir"], "KPOINTS")).read().splitlines()[3]
    assert kp.split() == [str(int(np.ceil(12 / sc[0]))), str(int(np.ceil(12 / sc[1]))), "1"]


def test_noisy_fit_errors_and_moment_check(workdir):
    out = str(workdir / "noisy")
    plan = setup(str(workdir / "POSCAR"), ["Cr"], 2, [2, 2], out, method="fit", nconf=8, mae=False,
                 log=lambda s: None)
    write_fake_outputs(out, plan, noise=2e-5, flip="col_000", rng=np.random.default_rng(0))
    logs = []
    model = collect(out, log=logs.append)
    pv = model["provenance"]
    assert pv["dof"] == 5 and pv["rms_meV"] > 0 and all(e > 0 for e in pv["J_err_meV"])
    for J, Jt, e in zip(pv["J_meV"], JTRUE, pv["J_err_meV"]):
        assert abs(J - Jt) < 5 * e
    assert any("local moments differ" in l for l in logs)
    with pytest.raises(RuntimeError):
        collect(out, strict=True, log=lambda s: None)


def test_supercell_too_small_is_reported(workdir):
    with pytest.raises(ValueError):
        setup(str(workdir / "POSCAR"), ["Cr"], 3, [1, 1], str(workdir / "x"), method="fit", log=lambda s: None)
    with pytest.raises(ValueError):
        setup(str(workdir / "POSCAR"), ["Cr"], 3, [2, 2], str(workdir / "y"), method="fourstate", log=lambda s: None)


def test_poscar_roundtrip_and_supercell(tmp_path):
    (tmp_path / "POSCAR").write_text(POSCAR.replace("Direct", "Cartesian").replace(
        "0.333333333333 0.666666666667 0.5", "0.0 3.98371666 10.0").replace(
        "0.666666666667 0.333333333333 0.5", "3.45 1.99185833 10.0").replace("0.0 0.0 0.55", "0.0 0.0 11.0"))
    p = vasp.read_poscar(str(tmp_path / "POSCAR"))
    assert p["frac"][0] == pytest.approx([1 / 3, 2 / 3, 0.5], abs=1e-6)
    q, index = vasp.supercell(p, [2, 3, 1])
    assert q["counts"] == [12, 6] and len(index) == 18
    vasp.write_poscar(str(tmp_path / "P2"), q)
    r = vasp.read_poscar(str(tmp_path / "P2"))
    assert np.allclose(r["lattice"], q["lattice"]) and np.allclose(r["frac"], q["frac"])
    assert vasp.magmom_string([3, 3, -3, 0, 0]) == "2*3 -3 2*0"


def test_energy_readers(tmp_path):
    (tmp_path / "OUTCAR").write_text("  free  energy   TOTEN  =      -10.5 eV\n  energy  without entropy=  -10.4  energy(sigma->0) = -10.45\n")
    assert vasp.read_energy(str(tmp_path), "F") == -10.5
    assert vasp.read_energy(str(tmp_path), "E0") == -10.45     # energy(sigma->0)
