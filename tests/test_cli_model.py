"""Model validation, analysis rules and the command line end to end."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from magtc import analyse
from magtc.analysis import peak
from magtc.model import ModelError, load_model, seeds, temperatures
from tests.test_dft import POSCAR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TINY = """
name: tiny
structure: {dim: 2, lattice: [[3, 0, 0], [0, 3, 0], [0, 0, 15]], sites: [{label: Mn, frac: [0, 0, 0]}]}
interactions:
  - {name: J1, J: 2.0, shell: 1}
anisotropy: {K: [0, 0, -0.5]}
mc: {L: 6, Tmax: 80, Tmin: 40, step: 10, ntherm: 50, nmeas: 200, every: 5, nseeds: 3, seed0: 1}
"""


def base(**kw):
    d = dict(name="x", structure=dict(dim=2, sites=[{"frac": [0, 0, 0]}]),
             interactions=[dict(J=1.0, bonds=[[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])])
    d.update(kw)
    return d


def test_defaults_are_jcp_protocol():
    mc = load_model(base())["mc"]
    assert (mc["L"], mc["ntherm"], mc["nmeas"], mc["every"], mc["nseeds"], mc["protocol"]) == \
        (50, 100000, 200000, 20, 8, "anneal")
    assert mc["moves"]["metropolis"] == "uniform" and mc["moves"]["overrelax"] == 0 and mc["moves"]["wolff"] == 0
    assert seeds(load_model(base()))[:2] == [20260616, 20270616]


@pytest.mark.parametrize("bad", [
    dict(interactions=[dict(J=1.0, bonds=[[0, 0, 1, 0, 0], [0, 0, -1, 0, 0]])]),
    dict(interactions=[dict(J=1.0, bonds=[[0, 0, 0, 0, 0]])]),
    dict(interactions=[dict(J=1.0, bonds=[[0, 0, 0, 0, 1]])]),          # dc != 0 in 2D
    dict(interactions=[dict(J=1.0, Jz=2.0, bonds=[[0, 0, 1, 0, 0]])]),
    dict(interactions=[dict(J=1.0, bonds=[[0, 1, 1, 0, 0]])]),
    dict(order="zigzag"),
    dict(anisotropy={"K": [1, 2]}),
    dict(mc={"moves": {"metropolis": "none"}}),
    dict(mc={"protocol": "magic"}),
])
def test_invalid_models(bad):
    with pytest.raises(ModelError):
        load_model(base(**bad))


def test_ladder_idempotent_and_size_check():
    m = load_model(base(mc={"Tmax": 30, "Tmin": 20, "step": 5}))
    assert temperatures(m).tolist() == [30.0, 25.0, 20.0]
    assert load_model(m) == m
    from magtc.lattice import build_tables
    with pytest.raises(ModelError):
        build_tables(load_model(base(interactions=[dict(J=1.0, bonds=[[0, 0, 2, 0, 0]])])), (4, 8, 1))


def test_peak_and_error_rule():
    T = np.array([5.0, 4.0, 3.0, 2.0])
    assert peak(T, -(T - 3.3) ** 2) == (pytest.approx(3.3), False)

    def fake(seed, shift):
        TT = np.array([30.0, 25.0, 20.0, 15.0])
        chi = np.exp(-(TT - 22 - shift) ** 2 / 20)
        rows = [dict(T=float(t), M=0.5, M2=0.25 + c * 0.0861733 * t / 16, M4=0.1, Mz=0.0, Mz2=0.0, Mz4=0.0, E=0.0,
                     E2=c, n=100, tau_M=1.0) for t, c in zip(TT, chi)]
        return dict(tag="f", model_hash="h", L=[4, 4, 1], nsite=16, order="FM", seed=seed, ntherm=1, nmeas=1,
                    every=1, numba=True, protocol="anneal", moves={}, rows=rows)
    runs = [fake(s, d) for s, d in enumerate([-1.0, 0.0, 0.5, 1.5])]
    r = analyse(runs)
    assert r["Tc_chi_abs_err"] == max(r["Tc_chi_abs_jackknife_err"], r["Tc_chi_abs_sem_single_seed"])
    with pytest.raises(ValueError):
        analyse(runs + [fake(0, 0.0)])


def cli(*args, cwd):
    r = subprocess.run([sys.executable, "-m", "magtc", *args], capture_output=True, text=True, cwd=cwd,
                       env=dict(os.environ, PYTHONPATH=ROOT))
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_cli_end_to_end(tmp_path):
    (tmp_path / "tiny.yaml").write_text(TINY)
    assert "ground: FM along z" in cli("ground", "tiny.yaml", cwd=tmp_path)
    cli("run", "tiny.yaml", "--seed-index", "0", "--outdir", "runs", cwd=tmp_path)
    cli("run", "tiny.yaml", "--seed", "77", "--out", "runs/x.json", cwd=tmp_path)
    out = cli("analyse", "runs/tiny_seed1.json", "runs/x.json", "--out", "a.json", cwd=tmp_path)
    assert "T_C (chi_abs peak)" in out and json.load(open(tmp_path / "a.json"))["n_seeds"] == 2
    cli("batch", "tiny.yaml", "--nproc", "2", "--outdir", "b", "--overrelax", "1", "--wolff", "2", cwd=tmp_path)
    res = json.load(open(tmp_path / "b" / "tiny_analysis.json"))
    assert len(res["seeds"]) == 3
    cli("batch", "tiny.yaml", "--outdir", "pt", "--protocol", "pt", "--nproc", "1", cwd=tmp_path)
    cli("batch", "tiny.yaml", "--pbs", "job.pbs", "--queue", "batch", "--L", "8", "--wolff", "3", cwd=tmp_path)
    txt = (tmp_path / "job.pbs").read_text()
    assert "#PBS -J 0-2" in txt and "--L 8" in txt and "--wolff 3" in txt
    out = cli("bench", "tiny.yaml", "--T", "60", "--sweeps", "400", cwd=tmp_path)
    assert "speedup" in out
    (tmp_path / "POSCAR").write_text(POSCAR)
    cli("dft", "setup", "POSCAR", "--magnetic", "Cr", "--nshell", "2", "--supercell", "2", "2", "--outdir", "map",
        "--moment", "Cr=3.2", cwd=tmp_path)
    assert (tmp_path / "map" / "plan.json").exists() and (tmp_path / "map" / "mae_z" / "INCAR").exists()
    assert "3.2" in (tmp_path / "map" / "col_000" / "INCAR").read_text()


def test_examples_load():
    for f in os.listdir(os.path.join(ROOT, "examples")):
        if f.endswith((".yaml", ".json")):
            load_model(os.path.join(ROOT, "examples", f))
