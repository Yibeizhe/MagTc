"""Model validation, analysis rules and the CLI end to end."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import magtc
from magtc.analysis import analyse, peak
from magtc.model import ModelError, load_model, seeds, temperatures

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TINY = """
name: tiny
lattice: {type: square}
exchange:
  - {name: J1, J: 2.0, bonds: [[0, 0, 1, 0], [0, 0, 0, 1]]}
anisotropy: {K: [0, 0, -0.5]}
mc: {L: 6, Tmax: 80, Tmin: 40, step: 10, ntherm: 50, nmeas: 200, every: 5, nseeds: 3, seed0: 1}
"""


def base(**kw):
    d = dict(name="x", lattice={"type": "square"}, exchange=[dict(J=1.0, bonds=[[0, 0, 1, 0], [0, 0, 0, 1]])])
    d.update(kw)
    return d


def test_defaults_are_jcp_protocol():
    m = load_model(base())
    mc = m["mc"]
    assert (mc["L"], mc["ntherm"], mc["nmeas"], mc["every"], mc["nseeds"]) == (50, 100000, 200000, 20, 8)
    assert seeds(m)[:2] == [20260616, 20270616]


@pytest.mark.parametrize("bad", [
    dict(exchange=[dict(J=1.0, bonds=[[0, 0, 1, 0], [0, 0, -1, 0]])]),       # pair listed twice
    dict(exchange=[dict(J=1.0, bonds=[[0, 0, 0, 0]])]),                      # self bond
    dict(exchange=[dict(J=1.0, Jz=2.0, bonds=[[0, 0, 1, 0]])]),               # J and Jz
    dict(exchange=[dict(J=1.0, bonds=[[0, 1, 1, 0]])]),                      # sublattice out of range
    dict(order="zigzag"),
    dict(anisotropy={"K": [1, 2]}),
])
def test_invalid_models(bad):
    with pytest.raises(ModelError):
        load_model(base(**bad))


def test_ladder_and_idempotent():
    m = load_model(base(mc={"Tmax": 30, "Tmin": 20, "step": 5}))
    assert temperatures(m).tolist() == [30.0, 25.0, 20.0]
    assert load_model(m) == m
    with pytest.raises(ModelError):
        temperatures(load_model(base(mc={"temperatures": [1, 2]})))


def test_lattice_too_small():
    from magtc.model import build_tables
    m = load_model(base(exchange=[dict(J=1.0, bonds=[[0, 0, 2, 0]])]))
    with pytest.raises(ModelError):
        build_tables(m, 4, 8)


def test_peak_parabola_and_edge():
    T = np.array([5.0, 4.0, 3.0, 2.0])
    assert peak(T, -(T - 3.3) ** 2) == (pytest.approx(3.3), False)
    assert peak(T, T)[1] is True


def _fake_run(seed, shift):
    T = np.array([30.0, 25.0, 20.0, 15.0])
    chi = np.exp(-(T - 22 - shift) ** 2 / 20)
    rows = [dict(T=float(t), M=0.5, M2=0.25 + c * 0.0861733 * t / 16, M4=0.1, Mz=0.0, Mz2=0.0, Mz4=0.0,
                 E=0.0, E2=c, n=1) for t, c in zip(T, chi)]
    return dict(tag="f", model_hash="h", L=[4, 4], nsite=16, order="FM", seed=seed, ntherm=1, nmeas=1, every=1,
                numba=True, rows=rows)


def test_error_is_max_of_jackknife_and_sem():
    runs = [_fake_run(s, d) for s, d in enumerate([-1.0, 0.0, 0.5, 1.5])]
    r = analyse(runs)
    assert r["Tc_chi_abs_err"] == max(r["Tc_chi_abs_jackknife_err"], r["Tc_chi_abs_sem_single_seed"])
    assert r["n_seeds"] == 4 and r["Tc_Cv"] is not None
    with pytest.raises(ValueError):
        analyse(runs + [_fake_run(0, 0.0)])


def cli(*args, cwd):
    r = subprocess.run([sys.executable, "-m", "magtc", *args], capture_output=True, text=True, cwd=cwd,
                       env=dict(os.environ, PYTHONPATH=ROOT))
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_cli_end_to_end(tmp_path):
    (tmp_path / "tiny.yaml").write_text(TINY)
    out = cli("ground", "tiny.yaml", cwd=tmp_path)
    assert "ground: FM along z" in out
    cli("run", "tiny.yaml", "--seed-index", "0", "--outdir", "runs", cwd=tmp_path)
    cli("run", "tiny.yaml", "--seed", "77", "--out", "runs/x.json", cwd=tmp_path)
    out = cli("analyse", "runs/tiny_seed1.json", "runs/x.json", "--out", "a.json", "--json", cwd=tmp_path)
    assert "T_C (chi_abs peak)" in out and json.load(open(tmp_path / "a.json"))["n_seeds"] == 2
    out = cli("batch", "tiny.yaml", "--nproc", "2", "--outdir", "b", cwd=tmp_path)
    assert os.path.exists(tmp_path / "b" / "tiny_analysis.json")
    assert len(json.load(open(tmp_path / "b" / "tiny_analysis.json"))["seeds"]) == 3
    cli("batch", "tiny.yaml", "--pbs", "job.pbs", "--queue", "batch", "--L", "8", cwd=tmp_path)
    txt = (tmp_path / "job.pbs").read_text()
    assert "#PBS -J 0-2" in txt and "--seed-index" in txt and "--L 8" in txt and "#PBS -q batch" in txt


def test_examples_load():
    for f in os.listdir(os.path.join(ROOT, "examples")):
        if f.endswith((".yaml", ".json")):
            load_model(os.path.join(ROOT, "examples", f))
