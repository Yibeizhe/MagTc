"""All seeds of a model: local multiprocessing, or a PBS array script (written, never submitted)."""
import json
import os
import shlex
from multiprocessing import get_context

from .analysis import analyse, load_runs, summary_text
from .model import load_model, seeds as model_seeds
from .runner import apply_overrides, run_seed


def _worker(args):
    model_path, ov, seed, out = args
    m = apply_overrides(load_model(model_path), **ov)
    run_seed(m, seed, out=out, verbose=False)
    return out


def batch(model_path, outdir, nproc=None, overrides=None, log=print):
    overrides = overrides or {}
    m = apply_overrides(load_model(model_path), **overrides)
    os.makedirs(outdir, exist_ok=True)
    sd = model_seeds(m)
    jobs = [(model_path, overrides, s, os.path.join(outdir, "%s_seed%d.json" % (m["name"], s))) for s in sd]
    nproc = nproc or min(len(jobs), os.cpu_count() or 1)
    log("magtc batch: %s, %d seeds on %d processes -> %s" % (m["name"], len(jobs), nproc, outdir))
    if nproc == 1:
        files = [_worker(j) for j in jobs]
    else:
        with get_context("spawn").Pool(nproc) as pool:
            files = []
            for f in pool.imap(_worker, jobs):
                log("  done %s" % f)
                files.append(f)
    res = analyse(load_runs(files))
    with open(os.path.join(outdir, "%s_analysis.json" % m["name"]), "w") as f:
        json.dump(res, f, indent=1)
    log(summary_text(res))
    return res


PBS_TEMPLATE = """#!/bin/bash
#PBS -N {jobname}
#PBS -l nodes=1:ppn=1
#PBS -l walltime={walltime}
#PBS -J 0-{last}
{queue}#PBS -j oe
# Written by `magtc batch --pbs` (MagTc {version}); NOT submitted.
# Submit with:   qsub {script}
# One array element = one seed.  Torque users: replace '#PBS -J 0-{last}' by '#PBS -t 0-{last}'
# (the index is read from PBS_ARRAY_INDEX or PBS_ARRAYID).
# After all elements finish:
#   magtc analyse {outdir_q}/{name}_seed*.json --out {outdir_q}/{name}_analysis.json

cd "$PBS_O_WORKDIR"
{setup}IDX=${{PBS_ARRAY_INDEX:-$PBS_ARRAYID}}
mkdir -p {outdir_q}
{python} -m magtc run {model_q} --seed-index "$IDX" --outdir {outdir_q}{extra}
"""


def write_pbs(model_path, outdir, script, walltime="72:00:00", queue=None, setup=None, overrides=None,
              python=None):
    from . import __version__
    m = apply_overrides(load_model(model_path), **(overrides or {}))
    n = len(model_seeds(m))
    extra = ""
    for k, v in (overrides or {}).items():
        if v is None or k.startswith("_"):
            continue
        if k == "L" and isinstance(v, (list, tuple)):
            extra += " --L %d %d" % tuple(v)
        else:
            extra += " --%s %s" % (k, v)
    text = PBS_TEMPLATE.format(
        jobname=("mtc_" + m["name"])[:15], walltime=walltime, last=n - 1,
        queue=("#PBS -q %s\n" % queue) if queue else "", version=__version__, script=script,
        outdir_q=shlex.quote(outdir), name=m["name"], setup=(setup + "\n") if setup else "",
        python=python or "python3", model_q=shlex.quote(os.path.abspath(model_path)), extra=extra)
    with open(script, "w") as f:
        f.write(text)
    return script
