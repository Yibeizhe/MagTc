"""Zero-temperature energies of collinear candidate orders (`magtc ground`).

Candidates: every order in the model plus all eta = signs[u] * (-1)^(q.R) with q in {0,1}^dim and
sublattice signs (first fixed to +1), each with the moments along x, y and z.  Non-collinear and
long-period states (spirals) are not tried.
"""
import itertools

import numpy as np

from . import kernels as kn
from .lattice import build_tables, eta_array, max_range


def _size(m):
    r = max_range(m)
    dim = m["structure"]["dim"]
    L = []
    for a in range(3):
        if a == 2 and dim == 2:
            L.append(1)
            continue
        n = 4
        while n <= 2 * r[a]:
            n += 2
        L.append(n)
    return tuple(L)


def _same(a, b):
    if "pattern" in a or "pattern" in b:
        return False
    if a["q"] != b["q"]:
        return False
    sa, sb = np.array(a["signs"]), np.array(b["signs"])
    return bool(np.all(sa == sb) or np.all(sa == -sb))


def candidates(m):
    nsub = len(m["structure"]["sites"])
    dim = m["structure"]["dim"]
    cands = dict(m["orders"])
    qs = itertools.product((0, 1), (0, 1), (0, 1) if dim == 3 else (0,))
    for q in qs:
        for tail in itertools.product((1, -1), repeat=nsub - 1):
            signs = [1.0] + [float(x) for x in tail]
            op = {"q": list(q), "signs": signs}
            if not any(_same(op, o) for o in cands.values()):
                key = "q=(%s) signs=(%s)" % (",".join(map(str, q)), ",".join("+" if x > 0 else "-" for x in signs))
                cands[key] = op
    return cands


def ground(m):
    L = _size(m)
    tb = build_tables(m, L)
    out = {}
    for name, op in candidates(m).items():
        try:
            eta = eta_array(op, L, tb["nsub"])
        except Exception:
            continue
        for c, ax in enumerate("xyz"):
            s = np.zeros((tb["nsite"], 3))
            s[:, c] = eta
            out[(name, ax)] = float(kn.energy(s, tb["nsub"], tb["nfw"], tb["fw"], tb["fjxy"], tb["fjz"], tb["K"]))
    best = min(out, key=out.get)
    rows = sorted(({"order": k[0], "axis": k[1], "E_meV_per_spin": v} for k, v in out.items()),
                  key=lambda r: r["E_meV_per_spin"])
    ops = m["orders"]
    is_ground = best[0] == m["order"] or (best[0] in ops and _same(ops[best[0]], ops[m["order"]]))
    return dict(ground_order=best[0], ground_axis=best[1], E_ground=out[best], energies=rows,
                mc_order=m["order"], mc_order_is_ground=is_ground, supercell=list(L))


def mean_field_tc(m):
    """Mean-field T_C of the FM state (K), sum_shells z_s J_s / (3 k_B); an upper bound used to place ladders."""
    nsub = len(m["structure"]["sites"])
    zJ = 0.0
    for sh in m["interactions"]:
        zJ += 2.0 * len(sh["bonds"]) / nsub * (2 * sh["Jxy"] + sh["Jz"]) / 3.0
    return zJ / (3.0 * kn.kB)


def ground_text(g):
    lines = ["collinear candidates (meV per spin, lowest first):"]
    e0 = g["E_ground"]
    for r in g["energies"]:
        lines.append("  %-30s %s  % .6f   (+%.6f)" % (r["order"], r["axis"], r["E_meV_per_spin"],
                                                    r["E_meV_per_spin"] - e0))
    lines.append("ground: %s along %s" % (g["ground_order"], g["ground_axis"]))
    if not g["mc_order_is_ground"]:
        lines.append("WARNING: the MC order parameter '%s' is not the lowest collinear state" % g["mc_order"])
    return "\n".join(lines)
