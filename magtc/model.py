"""Model files (YAML/JSON): loading, validation, defaults, and the v1 format.

    name, description, reference
    spin:        heisenberg (default) | ising
    structure:   dim: 2|3
                 lattice: [[a1], [a2], [a3]]       Angstrom (needed for `shell:` and geometry checks)
                 sites:   [{label: Mn, frac: [x, y, z]}, ...]     magnetic sublattices
    interactions:
      - {name: J1, J: 4.9, bonds: [[s1, s2, da, db, dc], ...]}   explicit bonds, each pair once
      - {name: J2, Jxy: 1.0, Jz: 1.1, shell: 2}                  all bonds of the 2nd distance shell
    anisotropy:  {K: [Kx, Ky, Kz]}  or one triple per sublattice
    orders:      {NAME: {q: [qa, qb, qc], signs: [...]}} or {NAME: {pattern: [pa][pb][pc][nsub]}}
    order:       order parameter measured by the MC run (default FM)
    mc:          Monte Carlo settings (defaults = JCP 2026 protocol, see MC_DEFAULTS)
    provenance:  free-form record (written by `magtc dft collect`)

Energies are in meV, J > 0 is ferromagnetic, spins are classical unit vectors:
    H = - sum_<ij> [Jxy (Sx Sx' + Sy Sy') + Jz Sz Sz'] + sum_i sum_a K_a (S_i^a)^2

Files in the MagTc 1.x format (lattice/exchange/order_parameters) are converted on load.
"""
import copy
import hashlib
import json
import os

import numpy as np

from .structure import Structure

MC_DEFAULTS = dict(
    L=50,               # int, or [L1, L2] (2D) / [L1, L2, L3] (3D), in unit cells
    Tmax=None, Tmin=None, step=5.0,
    temperatures=None,  # explicit descending ladder, overrides Tmax/Tmin/step
    ntherm=100000,      # thermalisation sweeps per temperature
    nmeas=200000,       # measurement sweeps per temperature
    every=20,           # measure every `every` sweeps
    nseeds=8, seed0=20260616, seed_stride=10000, seeds=None,
    protocol="anneal",  # anneal: JCP high->low ladder; pt: parallel tempering over the ladder
    pt_swap_every=1,    # sweeps between replica-exchange attempts (pt only)
    moves=None,         # see MOVE_DEFAULTS
)

#: One "sweep" = the moves below, in this order.  The JCP protocol is metropolis 'uniform' only.
MOVE_DEFAULTS = dict(
    metropolis="uniform",   # uniform (JCP) | cone (adaptive Gaussian step) | none
    metropolis_passes=1,    # Metropolis passes (N attempts each) per sweep
    overrelax=0,            # over-relaxation passes per sweep (exchange-energy-conserving reflections)
    wolff=0,                # Wolff clusters per sweep
    cone_target=0.5,        # acceptance targeted by the adaptive cone (thermalisation only)
)


class ModelError(ValueError):
    pass


def _load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if os.path.splitext(path)[1].lower() == ".json":
        return json.loads(text)
    import yaml
    return yaml.safe_load(text)


def load_model(src):
    if isinstance(src, dict):
        raw = copy.deepcopy(src)
    else:
        raw = _load_text(src)
        raw.setdefault("name", os.path.splitext(os.path.basename(src))[0])
    return normalise(raw)


def _f(x, what):
    try:
        return float(x)
    except Exception as e:
        raise ModelError("%s must be a number, got %r" % (what, x)) from e


def convert_v1(raw):
    """MagTc 1.x model (lattice/exchange/order_parameters, 4-index bonds) -> 2.x layout."""
    m = {k: v for k, v in raw.items() if k not in ("lattice", "exchange", "order_parameters")}
    lat = raw.get("lattice", {})
    typ = lat.get("type", "rectangular")
    subs = lat.get("sublattices") or ([[0, 0], [1 / 3, 1 / 3]] if typ == "honeycomb" else [[0, 0]])
    a1 = list(lat.get("a1", [1.0, 0.0])) + [0.0]
    a2 = list(lat.get("a2", [0.0, 1.0])) + [0.0]
    m["structure"] = dict(dim=2, lattice=[a1[:3], a2[:3], [0.0, 0.0, 10.0]],
                          sites=[dict(label="M%d" % i, frac=[float(s[0]), float(s[1]), 0.0])
                                 for i, s in enumerate(subs)])
    inter = []
    for sh in raw.get("exchange", []):
        e = {k: v for k, v in sh.items() if k != "bonds"}
        e["bonds"] = [list(b) + [0] if len(b) == 4 else list(b) for b in sh["bonds"]]
        inter.append(e)
    m["interactions"] = inter
    if "order_parameters" in raw:
        ords = {}
        for n, op in raw["order_parameters"].items():
            op = dict(op)
            if "pattern" in op:
                pat = np.array(op["pattern"], float)
                op["pattern"] = pat[:, :, None, :].tolist()
            else:
                op["q"] = list(op.get("q", [0, 0]))[:2] + [0]
            ords[n] = op
        m["orders"] = ords
    if raw.get("mc"):
        m["mc"] = dict(raw["mc"])
    return m


def normalise(raw):
    if "exchange" in raw and "interactions" not in raw:
        raw = convert_v1(raw)
    m = copy.deepcopy(raw)
    if "name" not in m:
        raise ModelError("model needs a 'name'")
    for k in ("description", "reference"):
        m.setdefault(k, "")
    m.setdefault("spin", "heisenberg")
    if m["spin"] not in ("heisenberg", "ising"):
        raise ModelError("spin must be heisenberg or ising")

    st = m.get("structure")
    if not st or not st.get("sites"):
        raise ModelError("model needs structure.sites")
    st.setdefault("dim", 2)
    st.setdefault("lattice", [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    st["sites"] = [dict(label=s.get("label", "M%d" % i), frac=[float(x) for x in (list(s["frac"]) + [0, 0])[:3]])
                   for i, s in enumerate(st["sites"])]
    try:
        S = Structure(st["lattice"], [s["frac"] for s in st["sites"]], st["dim"], [s["label"] for s in st["sites"]])
    except ValueError as e:
        raise ModelError(str(e)) from e
    st["lattice"] = S.lattice.tolist()
    nsub = S.nsub

    inter = m.get("interactions")
    if not isinstance(inter, list) or not inter:
        raise ModelError("'interactions' must be a non-empty list")
    need = [int(sh["shell"]) for sh in inter if "shell" in sh and not sh.get("bonds")]
    shell_cache = {i + 1: v for i, v in enumerate(S.shells(max(need)))} if need else {}
    seen = {}
    for k, sh in enumerate(inter):
        sh.setdefault("name", "J%d" % (k + 1))
        if sh.get("J") is not None:
            J = _f(sh["J"], "J of " + sh["name"])
            if any(x in sh and float(sh[x]) != J for x in ("Jxy", "Jz")):
                raise ModelError("shell %s: give either J or (Jxy, Jz)" % sh["name"])
            sh["J"] = sh["Jxy"] = sh["Jz"] = J
        else:
            if "Jxy" not in sh:
                raise ModelError("shell %s needs J or Jxy" % sh["name"])
            sh["Jxy"] = _f(sh["Jxy"], "Jxy")
            sh["Jz"] = _f(sh.get("Jz", sh["Jxy"]), "Jz")
            sh["J"] = sh["Jxy"] if sh["Jxy"] == sh["Jz"] else None
        if "shell" in sh and "bonds" not in sh:
            sh["bonds"] = [list(b) for b in shell_cache[int(sh["shell"])][1]]
        if not sh.get("bonds"):
            raise ModelError("shell %s has no bonds (give bonds or shell)" % sh["name"])
        clean = []
        for b in sh["bonds"]:
            b = [int(x) for x in b]
            if len(b) == 4:
                b.append(0)
            if len(b) != 5:
                raise ModelError("bond %r of %s must be [s1, s2, da, db, dc]" % (b, sh["name"]))
            s1, s2, da, db, dc = b
            if not (0 <= s1 < nsub and 0 <= s2 < nsub):
                raise ModelError("bond %r of %s: sublattice out of range" % (b, sh["name"]))
            if s1 == s2 and (da, db, dc) == (0, 0, 0):
                raise ModelError("bond %r of %s couples a site to itself" % (b, sh["name"]))
            if S.dim == 2 and dc != 0:
                raise ModelError("bond %r of %s has dc != 0 in a 2D model" % (b, sh["name"]))
            key, rkey = (s1, s2, da, db, dc), (s2, s1, -da, -db, -dc)
            if key in seen or rkey in seen:
                raise ModelError("bond %r of %s duplicates a bond of %s (list each pair once)"
                                 % (b, sh["name"], seen.get(key, seen.get(rkey))))
            seen[key] = sh["name"]
            clean.append(b)
        sh["bonds"] = clean
        lens = [S.bond_length(b) for b in clean]
        sh["distance"] = float(np.mean(lens))
        sh["distance_spread"] = float(max(lens) - min(lens))

    an = m.setdefault("anisotropy", {})
    K = np.array(an.get("K", [0.0, 0.0, 0.0]), dtype=float)
    if K.shape == (3,):
        K = np.tile(K, (nsub, 1))
    if K.shape != (nsub, 3):
        raise ModelError("anisotropy.K must be [Kx, Ky, Kz] or one triple per sublattice")
    an["K"] = K.tolist()

    ords = dict(m.get("orders") or {})
    ords.setdefault("FM", {"q": [0, 0, 0], "signs": [1] * nsub})
    if nsub == 1 and S.dim == 2:
        ords.setdefault("stripe_a", {"q": [1, 0, 0], "signs": [1]})
        ords.setdefault("stripe_b", {"q": [0, 1, 0], "signs": [1]})
        ords.setdefault("Neel", {"q": [1, 1, 0], "signs": [1]})
    for name, op in ords.items():
        if "pattern" in op:
            pat = np.array(op["pattern"], float)
            if pat.ndim != 4 or pat.shape[3] != nsub:
                raise ModelError("order %s: pattern must have shape [pa][pb][pc][nsub]" % name)
            op["pattern"] = pat.tolist()
        else:
            q = [int(x) for x in (list(op.get("q", [0, 0, 0])) + [0, 0])[:3]]
            if any(x not in (0, 1) for x in q) or (S.dim == 2 and q[2]):
                raise ModelError("order %s: q entries must be 0 or 1 (qc = 0 in 2D)" % name)
            op["q"] = q
            op["signs"] = [float(x) for x in op.get("signs", [1] * nsub)]
            if len(op["signs"]) != nsub:
                raise ModelError("order %s: one sign per sublattice" % name)
    m["orders"] = ords
    m.setdefault("order", "FM")
    if m["order"] not in ords:
        raise ModelError("order %r not among orders %s" % (m["order"], list(ords)))

    mc = dict(MC_DEFAULTS)
    mc.update(m.get("mc") or {})
    mv = dict(MOVE_DEFAULTS)
    mv.update(mc.get("moves") or {})
    mc["moves"] = mv
    if mv["metropolis"] not in ("uniform", "cone", "none"):
        raise ModelError("mc.moves.metropolis must be uniform, cone or none")
    if mc["protocol"] not in ("anneal", "pt"):
        raise ModelError("mc.protocol must be anneal or pt")
    if mc["every"] <= 0 or mc["nmeas"] <= 0 or mc["ntherm"] < 0:
        raise ModelError("mc: every, nmeas > 0 and ntherm >= 0 required")
    if mv["metropolis"] == "none" and not mv["wolff"]:
        raise ModelError("mc.moves: without Metropolis moves, Wolff clusters are needed for ergodicity")
    m["mc"] = mc
    m.setdefault("provenance", {})
    return m


def structure_of(m):
    st = m["structure"]
    return Structure(st["lattice"], [s["frac"] for s in st["sites"]], st["dim"], [s["label"] for s in st["sites"]])


def model_hash(m):
    keys = ("spin", "structure", "interactions", "anisotropy", "orders", "order")
    return hashlib.sha256(json.dumps({k: m[k] for k in keys}, sort_keys=True).encode()).hexdigest()[:16]


def lattice_size(m, L=None):
    L = m["mc"]["L"] if L is None else L
    dim = m["structure"]["dim"]
    L = [int(x) for x in (L if isinstance(L, (list, tuple)) else [L] * dim)]
    if len(L) == 2 and dim == 2:
        L = L + [1]
    if len(L) != 3 or (dim == 2 and L[2] != 1):
        raise ModelError("mc.L must be an int or [L1, L2]%s" % ("" if dim == 2 else " ... [L1, L2, L3]"))
    return tuple(L)


def temperatures(m):
    mc = m["mc"]
    if mc.get("temperatures"):
        T = np.array(mc["temperatures"], dtype=float)
        if np.any(np.diff(T) >= 0):
            raise ModelError("mc.temperatures must be strictly descending")
        return T
    if mc["Tmax"] is None or mc["Tmin"] is None:
        raise ModelError("mc needs Tmax and Tmin (or a 'temperatures' list)")
    return np.arange(float(mc["Tmax"]), float(mc["Tmin"]) - 1e-9, -float(mc["step"]))


def seeds(m):
    mc = m["mc"]
    if mc.get("seeds"):
        return [int(s) for s in mc["seeds"]]
    return [int(mc["seed0"]) + int(mc["seed_stride"]) * k for k in range(int(mc["nseeds"]))]


def apply_overrides(m, **ov):
    """Copy of m with mc settings overridden; `moves` entries are merged."""
    m = copy.deepcopy(m)
    for k, v in ov.items():
        if v is None:
            continue
        if k == "moves":
            m["mc"]["moves"].update(v)
        else:
            m["mc"][k] = v
    if ov.get("Tmax") is not None or ov.get("Tmin") is not None:
        m["mc"]["temperatures"] = None
    if ov.get("nseeds") is not None or ov.get("seed0") is not None:
        m["mc"]["seeds"] = None
    return normalise(m)
