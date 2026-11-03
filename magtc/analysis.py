"""Seed average and T_C estimators (the `analyse` command).

T_C is the peak of chi_abs = N (<|M|^2> - <|M|>^2) / (k_B T) of the seed-averaged moments,
located by a three-point parabola through the maximum and its two neighbours (JCP 2026).
Error = max(leave-one-seed-out jackknife, SEM of the single-seed peak positions).
The C_v peak is reported with the same procedure as a consistency check.
"""
import json
import math

import numpy as np

from .kernel import kB


def peak(T, X):
    """Three-point parabolic peak. Returns (T_peak, at_edge)."""
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    i = int(np.argmax(X))
    if i in (0, len(T) - 1):
        return float(T[i]), True
    c = np.polyfit(T[i - 1:i + 2], X[i - 1:i + 2], 2)
    return (float(-c[1] / (2 * c[0])) if c[0] < 0 else float(T[i])), False


def curves(runs):
    """Seed-averaged observables. All runs must share the temperature ladder."""
    T = np.array([r["T"] for r in runs[0]["rows"]])
    N = runs[0].get("nsite") or runs[0]["L"] ** 2
    g = lambda k: np.mean([[row[k] for row in r["rows"]] for r in runs], axis=0)
    M, M2, M4, Mz, Mz2, E, E2 = g("M"), g("M2"), g("M4"), g("Mz"), g("Mz2"), g("E"), g("E2")
    chi_abs = N * (M2 - M ** 2) / (kB * T)
    chi_z = N * (Mz2 - Mz ** 2) / (kB * T)
    Cv = N * (E2 - E ** 2) / (kB * T ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        U = 1 - M4 / (3 * M2 ** 2)
    return dict(T=T, M=M, chi_abs=chi_abs, chi_z=chi_z, Cv=Cv, U=U, E=E)


def _estimate(runs, key):
    c = curves(runs)
    Tc, edge = peak(c["T"], c[key])
    n = len(runs)
    jk_err = sem = None
    per_seed = [peak(curves([r])["T"], curves([r])[key])[0] for r in runs]
    if n > 1:
        jk = [peak(*(lambda cc: (cc["T"], cc[key]))(curves([r for k, r in enumerate(runs) if k != i])))[0]
              for i in range(n)]
        jk_err = math.sqrt((n - 1) / n * sum((x - np.mean(jk)) ** 2 for x in jk))
        sem = float(np.std(per_seed, ddof=1) / math.sqrt(n))
    err = max(jk_err, sem) if n > 1 else None
    return dict(Tc=Tc, err=err, jackknife_err=jk_err, sem_single_seed=sem, at_edge=edge, per_seed=per_seed)


def check_compatible(runs):
    keys = {(r.get("model_hash"), tuple(np.atleast_1d(r["L"]).tolist()), r.get("order"),
             r.get("ntherm"), r.get("nmeas"), r.get("every"), r.get("numba"),
             tuple(row["T"] for row in r["rows"])) for r in runs}
    if len(keys) != 1:
        raise ValueError("runs differ in model, size, order parameter, MC settings, backend or ladder")
    sd = [r["seed"] for r in runs]
    if len(set(sd)) != len(sd):
        raise ValueError("duplicate seeds among the input files: %s" % sd)


def analyse(runs):
    check_compatible(runs)
    chi = _estimate(runs, "chi_abs")
    cv = _estimate(runs, "Cv")
    chz = _estimate(runs, "chi_z")
    c = curves(runs)
    res = dict(tag=runs[0]["tag"], model_hash=runs[0].get("model_hash"), order=runs[0].get("order"),
               L=runs[0]["L"], n_seeds=len(runs), seeds=[r["seed"] for r in runs],
               Tc_chi_abs=chi["Tc"], Tc_chi_abs_err=chi["err"],
               Tc_chi_abs_jackknife_err=chi["jackknife_err"], Tc_chi_abs_sem_single_seed=chi["sem_single_seed"],
               peak_at_edge=chi["at_edge"], per_seed_chi_abs_peak=chi["per_seed"],
               Tc_Cv=cv["Tc"], Tc_Cv_err=cv["err"], Cv_peak_at_edge=cv["at_edge"],
               Tc_chi_z=chz["Tc"],
               curve={k: v.tolist() for k, v in c.items()})
    return res


def summary_text(res):
    def pm(x, e):
        return "%.2f" % x if e is None else "%.2f +- %.2f" % (x, e)
    lines = ["%s  order=%s  L=%s  seeds=%d" % (res["tag"], res["order"], res["L"], res["n_seeds"]),
             "  T_C (chi_abs peak) = %s K%s" % (pm(res["Tc_chi_abs"], res["Tc_chi_abs_err"]),
                                                "   [PEAK AT EDGE OF LADDER - extend the scan]" if res["peak_at_edge"] else ""),
             "  T_C (C_v peak)     = %s K%s" % (pm(res["Tc_Cv"], res["Tc_Cv_err"]),
                                                "   [at edge]" if res["Cv_peak_at_edge"] else "")]
    if res["n_seeds"] > 1:
        lines.append("  chi_abs error: jackknife %.2f K, single-seed SEM %.2f K (larger is quoted)"
                     % (res["Tc_chi_abs_jackknife_err"], res["Tc_chi_abs_sem_single_seed"]))
    return "\n".join(lines)


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = res["curve"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].plot(c["T"], c["M"], "o-", ms=3); ax[0].set_ylabel("<|M|>")
    ax[1].plot(c["T"], c["chi_abs"], "o-", ms=3); ax[1].set_ylabel("chi_abs")
    ax[1].axvline(res["Tc_chi_abs"], ls="--", c="0.3")
    ax[2].plot(c["T"], c["Cv"], "o-", ms=3); ax[2].set_ylabel("C_v")
    ax[2].axvline(res["Tc_Cv"], ls="--", c="0.3")
    for a in ax:
        a.set_xlabel("T (K)")
    fig.suptitle("%s: T_C(chi_abs) = %.1f K" % (res["tag"], res["Tc_chi_abs"]))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_runs(files):
    runs = []
    for f in files:
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs
