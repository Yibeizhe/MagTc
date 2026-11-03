"""Zero-temperature energies of collinear candidate orders (the `ground` command).

Candidates are every order parameter defined in the model plus all patterns
eta = signs[u] * (-1)^(qa*i + qb*j) with q in {0,1}^2 and signs in {+1,-1}^nsub
(first sign fixed to +1), each with the moments along x, y and z.  Only collinear
states commensurate with a 2x2 supercell are tried: a non-collinear or long-period
ground state (spiral, etc.) is NOT detected.
"""
import itertools

import numpy as np

from . import kernel as kn
from .model import build_tables, eta_array, max_range


def _size(m):
    ra, rb = max_range(m)
    L = 8
    while L <= 2 * max(ra, rb):
        L += 2
    return L


def candidates(m):
    nsub = len(m["lattice"]["sublattices"])
    cands = dict(m["order_parameters"])
    for qa, qb in itertools.product((0, 1), (0, 1)):
        for tail in itertools.product((1, -1), repeat=nsub - 1):
            signs = [1.0] + [float(x) for x in tail]
            key = "q=(%d,%d) signs=(%s)" % (qa, qb, ",".join("+" if x > 0 else "-" for x in signs))
            op = {"q": [qa, qb], "signs": signs}
            if not any(_same(op, o) for o in cands.values()):
                cands[key] = op
    return cands


def _same(a, b):
    if "pattern" in a or "pattern" in b:
        return False
    if a["q"] != b["q"]:
        return False
    sa, sb = np.array(a["signs"]), np.array(b["signs"])
    return bool(np.all(sa == sb) or np.all(sa == -sb))


def ground(m):
    L = _size(m)
    tb = build_tables(m, L, L)
    out = {}
    for name, op in candidates(m).items():
        try:
            eta = eta_array(op, L, L, tb["nsub"])
        except Exception:
            continue
        for ax, c in zip("xyz", range(3)):
            s = np.zeros((tb["nsite"], 3))
            s[:, c] = eta
            out[(name, ax)] = float(kn.energy(s, tb["nsub"], tb["nfw"], tb["fw"], tb["fjxy"], tb["fjz"], tb["K"]))
    best = min(out, key=out.get)
    rows = sorted(({"order": k[0], "axis": k[1], "E_meV_per_spin": v} for k, v in out.items()),
                  key=lambda r: r["E_meV_per_spin"])
    return dict(ground_order=best[0], ground_axis=best[1], E_ground=out[best], energies=rows,
                mc_order=m["order"], mc_order_is_ground=_same_or_name(best[0], m), supercell=L)


def _same_or_name(name, m):
    if name == m["order"]:
        return True
    ops = m["order_parameters"]
    return name in ops and _same(ops[name], ops[m["order"]])


def ground_text(g):
    lines = ["collinear candidates (meV per spin, lowest first):"]
    e0 = g["E_ground"]
    for r in g["energies"]:
        lines.append("  %-28s %s  % .6f   (+%.6f)" % (r["order"], r["axis"], r["E_meV_per_spin"],
                                                    r["E_meV_per_spin"] - e0))
    lines.append("ground: %s along %s" % (g["ground_order"], g["ground_axis"]))
    if not g["mc_order_is_ground"]:
        lines.append("WARNING: the MC order parameter '%s' is not the lowest collinear state" % g["mc_order"])
    return "\n".join(lines)
