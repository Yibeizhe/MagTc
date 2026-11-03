"""Command-line interface:  magtc {run, analyse, ground, batch, show}."""
import argparse
import json
import os
import sys

from . import __version__
from . import kernel as kn


def _add_overrides(p):
    g = p.add_argument_group("Monte Carlo overrides (default: values in the model file / JCP protocol)")
    g.add_argument("--L", type=int, nargs="+", help="L or Lx Ly (unit cells)")
    g.add_argument("--Tmax", type=float)
    g.add_argument("--Tmin", type=float)
    g.add_argument("--step", type=float)
    g.add_argument("--ntherm", type=int)
    g.add_argument("--nmeas", type=int)
    g.add_argument("--every", type=int)
    g.add_argument("--nseeds", type=int)
    g.add_argument("--seed0", type=int)


def _overrides(a):
    L = None
    if a.L:
        if len(a.L) > 2:
            raise SystemExit("--L takes one or two integers")
        L = a.L[0] if len(a.L) == 1 else list(a.L)
    ov = dict(L=L, Tmax=a.Tmax, Tmin=a.Tmin, step=a.step, ntherm=a.ntherm, nmeas=a.nmeas, every=a.every,
              nseeds=a.nseeds, seed0=a.seed0)
    return {k: v for k, v in ov.items() if v is not None}


def _model(a):
    from .model import load_model
    from .runner import apply_overrides
    ov = _overrides(a)
    return apply_overrides(load_model(a.model), **ov), ov


def cmd_run(a):
    from .model import seeds
    from .runner import run_seed
    m, _ = _model(a)
    if (a.seed is None) == (a.seed_index is None):
        raise SystemExit("give exactly one of --seed or --seed-index")
    if a.seed_index is not None:
        sd = seeds(m)
        if not 0 <= a.seed_index < len(sd):
            raise SystemExit("seed index %d out of range (model has %d seeds)" % (a.seed_index, len(sd)))
        seed = sd[a.seed_index]
    else:
        seed = a.seed
    out = a.out or os.path.join(a.outdir, "%s_seed%d.json" % (m["name"], seed))
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    print("magtc %s run: %s seed %d L=%s backend=%s -> %s" % (__version__, m["name"], seed, m["mc"]["L"],
                                                              "numba" if kn.HAVE_NUMBA else "python", out),
          flush=True)
    run_seed(m, seed, out=out, log=lambda s: print(s, flush=True))


def cmd_analyse(a):
    from .analysis import analyse, load_runs, plot, summary_text
    res = analyse(load_runs(a.files))
    print(summary_text(res))
    if a.json:
        print(json.dumps({k: v for k, v in res.items() if k != "curve"}))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    if a.plot:
        plot(res, a.plot)


def cmd_ground(a):
    from .ground import ground, ground_text
    from .model import load_model
    g = ground(load_model(a.model))
    print(json.dumps(g) if a.json else ground_text(g))


def cmd_batch(a):
    from .batch import batch, write_pbs
    _, ov = _model(a)
    if a.pbs:
        write_pbs(a.model, a.outdir, a.pbs, walltime=a.walltime, queue=a.queue, setup=a.setup, overrides=ov)
        print("PBS script written to %s (not submitted)" % a.pbs)
        return
    batch(a.model, a.outdir, nproc=a.nproc, overrides=ov, log=lambda s: print(s, flush=True))


def cmd_show(a):
    from .model import load_model, model_hash, seeds, temperatures
    m = load_model(a.model)
    print(json.dumps(dict(name=m["name"], hash=model_hash(m), spin=m["spin"], lattice=m["lattice"],
                          exchange=m["exchange"], anisotropy=m["anisotropy"], order=m["order"],
                          mc=m["mc"], temperatures=temperatures(m).tolist() if m["mc"]["Tmax"] or
                          m["mc"]["temperatures"] else None, seeds=seeds(m)), indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(prog="magtc", description="Classical Monte Carlo T_C of 2D magnets (JCP 2026 protocol)")
    p.add_argument("--version", action="version", version="magtc " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one seed, one temperature ladder -> JSON")
    r.add_argument("model")
    r.add_argument("--seed", type=int)
    r.add_argument("--seed-index", type=int, help="k-th seed of the model's seed list")
    r.add_argument("--out", help="output JSON (default OUTDIR/NAME_seedS.json)")
    r.add_argument("--outdir", default=".")
    _add_overrides(r)

    an = sub.add_parser("analyse", aliases=["analyze"], help="seed average, chi_abs and C_v peaks")
    an.add_argument("files", nargs="+")
    an.add_argument("--out", help="write full analysis (with curves) to JSON")
    an.add_argument("--plot", help="write a PNG of |M|, chi_abs, C_v (needs matplotlib)")
    an.add_argument("--json", action="store_true", help="also print the summary as one JSON line")

    g = sub.add_parser("ground", help="zero-temperature energies of collinear orders")
    g.add_argument("model")
    g.add_argument("--json", action="store_true")

    b = sub.add_parser("batch", help="all seeds of a model (multiprocessing) or write a PBS script")
    b.add_argument("model")
    b.add_argument("--outdir", default="magtc_out")
    b.add_argument("--nproc", type=int)
    b.add_argument("--pbs", metavar="SCRIPT", help="write a PBS array script instead of running (never submitted)")
    b.add_argument("--walltime", default="72:00:00")
    b.add_argument("--queue")
    b.add_argument("--setup", help="shell line(s) inserted before the run, e.g. 'module load anaconda'")
    _add_overrides(b)

    s = sub.add_parser("show", help="print the normalised model, ladder and seeds")
    s.add_argument("model")

    a = p.parse_args(argv)
    fn = {"run": cmd_run, "analyse": cmd_analyse, "analyze": cmd_analyse, "ground": cmd_ground,
          "batch": cmd_batch, "show": cmd_show}[a.cmd]
    fn(a)


if __name__ == "__main__":
    main()
