"""magtc command line:  run | analyse | ground | batch | bench | show | dft {setup, collect}."""
import argparse
import json
import os
import time

from . import __version__
from . import kernels as kn


def _add_overrides(p):
    g = p.add_argument_group("Monte Carlo overrides (default: model file, else the JCP protocol)")
    g.add_argument("--L", type=int, nargs="+", help="L, or L1 L2 [L3] (unit cells)")
    for k in ("Tmax", "Tmin", "step"):
        g.add_argument("--" + k, type=float)
    for k in ("ntherm", "nmeas", "every", "nseeds", "seed0"):
        g.add_argument("--" + k, type=int)
    g.add_argument("--protocol", choices=["anneal", "pt"], help="anneal (JCP ladder) or parallel tempering")
    g.add_argument("--pt-swap-every", type=int)
    g.add_argument("--metropolis", choices=["uniform", "cone", "none"])
    g.add_argument("--metropolis-passes", type=int)
    g.add_argument("--overrelax", type=int, help="over-relaxation passes per sweep")
    g.add_argument("--wolff", type=int, help="Wolff clusters per sweep")


def _overrides(a):
    ov = {}
    if a.L:
        if len(a.L) > 3:
            raise SystemExit("--L takes 1 to 3 integers")
        ov["L"] = a.L[0] if len(a.L) == 1 else list(a.L)
    for k in ("Tmax", "Tmin", "step", "ntherm", "nmeas", "every", "nseeds", "seed0", "protocol", "pt_swap_every"):
        if getattr(a, k) is not None:
            ov[k] = getattr(a, k)
    mv = {k: getattr(a, k) for k in ("metropolis", "metropolis_passes", "overrelax", "wolff")
          if getattr(a, k) is not None}
    if mv:
        ov["moves"] = mv
    return ov


def _model(a):
    from .model import apply_overrides, load_model
    ov = _overrides(a)
    return apply_overrides(load_model(a.model), **ov), ov


def cmd_run(a):
    from .model import seeds
    from .simulate import run_seed
    m, _ = _model(a)
    if (a.seed is None) == (a.seed_index is None):
        raise SystemExit("give exactly one of --seed or --seed-index")
    if a.seed_index is not None:
        sd = seeds(m)
        if not 0 <= a.seed_index < len(sd):
            raise SystemExit("seed index %d out of range (%d seeds)" % (a.seed_index, len(sd)))
        seed = sd[a.seed_index]
    else:
        seed = a.seed
    out = a.out or os.path.join(a.outdir, "%s_seed%d.json" % (m["name"], seed))
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    print("magtc %s run: %s seed %d L=%s %s moves=%s backend=%s -> %s"
          % (__version__, m["name"], seed, m["mc"]["L"], m["mc"]["protocol"], m["mc"]["moves"],
             "numba" if kn.HAVE_NUMBA else "python", out), flush=True)
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
    from .ground import ground, ground_text, mean_field_tc
    from .model import load_model
    m = load_model(a.model)
    g = ground(m)
    g["mean_field_Tc_K"] = mean_field_tc(m)
    print(json.dumps(g) if a.json else ground_text(g) + "\nmean-field T_C (FM) = %.1f K" % g["mean_field_Tc_K"])


def cmd_batch(a):
    from .batch import batch, write_pbs
    _, ov = _model(a)
    if a.pbs:
        write_pbs(a.model, a.outdir, a.pbs, walltime=a.walltime, queue=a.queue, setup=a.setup, overrides=ov)
        print("PBS script written to %s (not submitted)" % a.pbs)
        return
    batch(a.model, a.outdir, nproc=a.nproc, overrides=ov, log=lambda s: print(s, flush=True))


BENCH_MOVES = [("metropolis (JCP)", dict(metropolis="uniform", overrelax=0, wolff=0)),
               ("metropolis + 2 overrelax", dict(metropolis="uniform", overrelax=2, wolff=0)),
               ("cone + 2 overrelax", dict(metropolis="cone", overrelax=2, wolff=0)),
               ("metropolis + Wolff", dict(metropolis="uniform", overrelax=0, wolff=-1)),
               ("metropolis + 2 overrelax + Wolff", dict(metropolis="uniform", overrelax=2, wolff=-1))]


def cmd_bench(a):
    """Cost per independent sample of |M| for several move sets at one temperature."""
    from .model import apply_overrides, load_model
    from .simulate import run_seed
    m0, _ = _model(a)
    rows = []
    for label, mv in BENCH_MOVES:
        mv = dict(mv)
        if mv["wolff"] < 0:
            mv["wolff"] = a.wolff_per_sweep
        m = apply_overrides(m0, temperatures=[a.T], ntherm=a.sweeps // 10, nmeas=a.sweeps, every=1, moves=mv,
                            protocol="anneal")
        run_seed(apply_overrides(m, ntherm=1, nmeas=2), 1, verbose=False)   # compile
        t = time.time()
        r = run_seed(m, 11, verbose=False)["rows"][0]
        dt = (time.time() - t) / (a.sweeps * 1.1)
        rows.append((label, dt * 1e3, r["tau_M"], r["tau_E"], 2 * r["tau_M"] * dt * 1e3, r["cluster"]))
    base = rows[0][4]
    print("%s  T = %.2f K  L = %s  (%d sweeps each)" % (m0["name"], a.T, m0["mc"]["L"], a.sweeps))
    print("%-34s %10s %9s %9s %14s %8s %8s" % ("moves", "ms/sweep", "tau_M", "tau_E", "ms/indep(M)", "speedup", "cluster"))
    for lab, ms, tm, te, ci, cl in rows:
        print("%-34s %10.3f %9.1f %9.1f %14.2f %8.1f %8.0f" % (lab, ms, tm, te, ci, base / ci, cl))
    print("tau in sweeps; speedup = (cost per independent |M| sample of the JCP moves) / (this move set)")


def cmd_show(a):
    from .model import load_model, model_hash, seeds, temperatures
    m = load_model(a.model)
    print(json.dumps(dict(name=m["name"], hash=model_hash(m), spin=m["spin"], structure=m["structure"],
                          interactions=[{k: v for k, v in sh.items()} for sh in m["interactions"]],
                          anisotropy=m["anisotropy"], order=m["order"], mc=m["mc"],
                          temperatures=temperatures(m).tolist() if (m["mc"]["Tmax"] is not None or m["mc"]["temperatures"]) else None,
                          seeds=seeds(m)), indent=1))


def cmd_dft_setup(a):
    from .dft import setup
    mom = {}
    for item in a.moment or []:
        k, v = item.split("=")
        mom[k] = float(v)
    setup(a.poscar, a.magnetic, a.nshell, a.supercell, a.outdir, template=a.template, method=a.method,
          nconf=a.nconf, moment=mom, mae=not a.no_mae, dim=a.dim)


def cmd_dft_collect(a):
    from .dft import collect
    collect(a.plan_dir, energy=a.energy, out=a.out, name=a.name, strict=a.strict)


def main(argv=None):
    p = argparse.ArgumentParser(prog="magtc", description="Classical Monte Carlo T_C of 2D and 3D magnets; VASP energy mapping")
    p.add_argument("--version", action="version", version="magtc " + __version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="one seed, one temperature ladder -> JSON")
    r.add_argument("model")
    r.add_argument("--seed", type=int)
    r.add_argument("--seed-index", type=int)
    r.add_argument("--out")
    r.add_argument("--outdir", default=".")
    _add_overrides(r)

    an = sub.add_parser("analyse", aliases=["analyze"], help="seed average; chi_abs and C_v peaks")
    an.add_argument("files", nargs="+")
    an.add_argument("--out")
    an.add_argument("--plot")
    an.add_argument("--json", action="store_true")

    g = sub.add_parser("ground", help="T = 0 energies of collinear orders, mean-field T_C")
    g.add_argument("model")
    g.add_argument("--json", action="store_true")

    b = sub.add_parser("batch", help="all seeds (multiprocessing) or write a PBS array script")
    b.add_argument("model")
    b.add_argument("--outdir", default="magtc_out")
    b.add_argument("--nproc", type=int)
    b.add_argument("--pbs", metavar="SCRIPT")
    b.add_argument("--walltime", default="72:00:00")
    b.add_argument("--queue")
    b.add_argument("--setup")
    _add_overrides(b)

    be = sub.add_parser("bench", help="compare move sets: cost per independent sample at one T")
    be.add_argument("model")
    be.add_argument("--T", type=float, required=True)
    be.add_argument("--sweeps", type=int, default=20000)
    be.add_argument("--wolff-per-sweep", type=int, default=10)
    _add_overrides(be)

    s = sub.add_parser("show", help="print the normalised model")
    s.add_argument("model")

    d = sub.add_parser("dft", help="VASP energy mapping")
    dsub = d.add_subparsers(dest="dcmd", required=True)
    ds = dsub.add_parser("setup", help="write VASP inputs for the mapping configurations")
    ds.add_argument("poscar")
    ds.add_argument("--magnetic", nargs="+", required=True, help="magnetic species, e.g. Mn")
    ds.add_argument("--nshell", type=int, default=3)
    ds.add_argument("--supercell", type=int, nargs="+", required=True, help="M1 M2 [M3]")
    ds.add_argument("--dim", type=int, choices=[2, 3], default=2)
    ds.add_argument("--method", choices=["fit", "fourstate"], default="fit")
    ds.add_argument("--nconf", type=int, help="configurations for the fit (default 2*nshell+2)")
    ds.add_argument("--moment", nargs="*", help="initial moments, e.g. Mn=4.0 (default 3)")
    ds.add_argument("--template", help="directory with INCAR, KPOINTS (and POTCAR)")
    ds.add_argument("--no-mae", action="store_true", help="skip the three SOC runs for K")
    ds.add_argument("--outdir", default="mapping")
    dc = dsub.add_parser("collect", help="read energies, fit J and K, write a model file")
    dc.add_argument("plan_dir")
    dc.add_argument("--energy", choices=["E0", "F"], default="E0")
    dc.add_argument("--out", default="model.yaml")
    dc.add_argument("--name")
    dc.add_argument("--strict", action="store_true", help="stop if any energy is missing or a moment flipped")

    a = p.parse_args(argv)
    if a.cmd == "dft":
        {"setup": cmd_dft_setup, "collect": cmd_dft_collect}[a.dcmd](a)
        return
    {"run": cmd_run, "analyse": cmd_analyse, "analyze": cmd_analyse, "ground": cmd_ground, "batch": cmd_batch,
     "bench": cmd_bench, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
