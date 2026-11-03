"""Model files: loading, validation, defaults and construction of the lattice tables.

A model file (YAML or JSON) has these top-level keys; see docs and examples/.

    name:          short identifier (used as tag in output files)
    description:   free text (optional)
    reference:     citation / DOI of the parameters (optional, recommended)
    spin:          heisenberg (default) | ising
    lattice:       {type, sublattices, a1, a2}
    exchange:      list of shells {name, J | (Jxy, Jz), bonds: [[from_sub, to_sub, da, db], ...]}
    anisotropy:    {K: [Kx, Ky, Kz]}  or  {K: [[Kx, Ky, Kz] per sublattice]}
    order_parameters: {NAME: {q: [qa, qb], signs: [...]} | {pattern: [[[...]]]}}
    order:         name of the order parameter measured by the Monte Carlo run
    mc:            Monte Carlo settings (defaults = JCP 2026 protocol)

Conventions (JCP 2026): energies in meV; J > 0 ferromagnetic; every pair listed ONCE;
H = - sum_<ij> [Jxy (Sx Sx' + Sy Sy') + Jz Sz Sz'] + sum_i sum_a K_a (S_i^a)^2.
A bond [s1, s2, da, db] couples sublattice s1 in cell (i, j) to sublattice s2 in cell
(i + da, j + db); da counts cells along a1 and db along a2.
"""
import copy
import hashlib
import json
import os

import numpy as np

#: Monte Carlo defaults = protocol of the VN JCP 2026 paper (vn_tc_batch_L50_prod.py).
MC_DEFAULTS = dict(
    L=50,            # int or [Lx, Ly] (unit cells)
    Tmax=None,       # K, ladder scanned from high to low
    Tmin=None,
    step=5.0,
    temperatures=None,  # explicit ladder (overrides Tmax/Tmin/step); must be descending
    ntherm=100000,   # thermalisation sweeps per temperature
    nmeas=200000,    # measurement sweeps per temperature
    every=20,        # measure every `every` sweeps
    nseeds=8,
    seed0=20260616,
    seed_stride=10000,  # seed_k = seed0 + seed_stride * k (as in vn_tc_batch_L50_prod.py)
    seeds=None,      # explicit list overrides nseeds/seed0
)

LATTICE_DEFAULT_SUBLATTICES = {
    "square": [[0.0, 0.0]],
    "rectangular": [[0.0, 0.0]],
    "honeycomb": [[0.0, 0.0], [1.0 / 3.0, 1.0 / 3.0]],
    "custom": None,
}


class ModelError(ValueError):
    pass


def _load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ModelError("PyYAML is needed to read %s (pip install pyyaml) or use JSON" % path) from e
    return yaml.safe_load(text)


def load_model(src):
    """Load and validate a model from a path or a dict.  Returns a normalised dict."""
    if isinstance(src, dict):
        raw = copy.deepcopy(src)
    else:
        raw = _load_text(src)
        raw.setdefault("name", os.path.splitext(os.path.basename(src))[0])
    return normalise(raw)


def _as_float(x, what):
    try:
        return float(x)
    except Exception as e:
        raise ModelError("%s must be a number, got %r" % (what, x)) from e


def normalise(raw):
    m = copy.deepcopy(raw)
    if "name" not in m:
        raise ModelError("model needs a 'name'")
    m.setdefault("description", "")
    m.setdefault("reference", "")
    m.setdefault("spin", "heisenberg")
    if m["spin"] not in ("heisenberg", "ising"):
        raise ModelError("spin must be 'heisenberg' or 'ising'")

    lat = m.setdefault("lattice", {})
    lat.setdefault("type", "rectangular")
    if lat["type"] not in LATTICE_DEFAULT_SUBLATTICES:
        raise ModelError("lattice.type must be one of %s" % list(LATTICE_DEFAULT_SUBLATTICES))
    if "sublattices" not in lat:
        if LATTICE_DEFAULT_SUBLATTICES[lat["type"]] is None:
            raise ModelError("lattice.type custom needs explicit 'sublattices'")
        lat["sublattices"] = LATTICE_DEFAULT_SUBLATTICES[lat["type"]]
    nsub = len(lat["sublattices"])
    if nsub < 1:
        raise ModelError("at least one sublattice is required")
    if lat["type"] in ("square", "rectangular") and nsub != 1:
        raise ModelError("square/rectangular lattices have one sublattice; use type custom")
    if lat["type"] == "honeycomb" and nsub != 2:
        raise ModelError("honeycomb lattice has two sublattices")

    # exchange shells
    shells = m.get("exchange", [])
    if not isinstance(shells, list) or not shells:
        raise ModelError("'exchange' must be a non-empty list of shells")
    seen = {}
    for k, sh in enumerate(shells):
        sh.setdefault("name", "J%d" % (k + 1))
        if sh.get("J") is not None:
            J = _as_float(sh["J"], "J of " + sh["name"])
            if any(k in sh and float(sh[k]) != J for k in ("Jxy", "Jz")):
                raise ModelError("shell %s: give either J or (Jxy, Jz)" % sh["name"])
            sh["J"] = sh["Jxy"] = sh["Jz"] = J
        else:
            if "Jxy" not in sh:
                raise ModelError("shell %s needs J or Jxy" % sh["name"])
            sh["Jxy"] = _as_float(sh["Jxy"], "Jxy of " + sh["name"])
            sh["Jz"] = _as_float(sh.get("Jz", sh["Jxy"]), "Jz of " + sh["name"])
            sh["J"] = sh["Jxy"] if sh["Jxy"] == sh["Jz"] else None
        bonds = sh.get("bonds")
        if not bonds:
            raise ModelError("shell %s has no bonds" % sh["name"])
        clean = []
        for b in bonds:
            if len(b) != 4:
                raise ModelError("bond %r of %s must be [from_sub, to_sub, da, db]" % (b, sh["name"]))
            s1, s2, da, db = (int(x) for x in b)
            if not (0 <= s1 < nsub and 0 <= s2 < nsub):
                raise ModelError("bond %r of %s: sublattice index out of range" % (b, sh["name"]))
            if s1 == s2 and da == 0 and db == 0:
                raise ModelError("bond %r of %s couples a site to itself" % (b, sh["name"]))
            key = (s1, s2, da, db)
            rkey = (s2, s1, -da, -db)
            if key in seen or rkey in seen:
                raise ModelError("bond %r of %s duplicates a bond of %s (each pair must be listed once)"
                                 % (b, sh["name"], seen.get(key, seen.get(rkey))))
            seen[key] = sh["name"]
            clean.append([s1, s2, da, db])
        sh["bonds"] = clean

    # anisotropy
    an = m.setdefault("anisotropy", {})
    K = an.get("K", [0.0, 0.0, 0.0])
    K = np.array(K, dtype=float)
    if K.shape == (3,):
        K = np.tile(K, (nsub, 1))
    if K.shape != (nsub, 3):
        raise ModelError("anisotropy.K must be [Kx, Ky, Kz] or one such triple per sublattice")
    an["K"] = K.tolist()

    # order parameters
    ops = m.get("order_parameters") or {}
    ops = dict(ops)
    ops.setdefault("FM", {"q": [0, 0], "signs": [1] * nsub})
    if nsub == 1:
        ops.setdefault("stripe_a", {"q": [1, 0], "signs": [1]})
        ops.setdefault("stripe_b", {"q": [0, 1], "signs": [1]})
        ops.setdefault("Neel", {"q": [1, 1], "signs": [1]})
    for name, op in ops.items():
        if "pattern" in op:
            pat = np.array(op["pattern"], dtype=float)
            if pat.ndim != 3 or pat.shape[2] != nsub:
                raise ModelError("order %s: pattern must have shape [pa][pb][nsub]" % name)
            op["pattern"] = pat.tolist()
        else:
            op.setdefault("q", [0, 0])
            op.setdefault("signs", [1] * nsub)
            if len(op["q"]) != 2 or any(int(x) not in (0, 1) for x in op["q"]):
                raise ModelError("order %s: q must be [qa, qb] with entries 0 or 1" % name)
            if len(op["signs"]) != nsub:
                raise ModelError("order %s: need one sign per sublattice" % name)
            op["q"] = [int(x) for x in op["q"]]
            op["signs"] = [float(x) for x in op["signs"]]
    m["order_parameters"] = ops
    m.setdefault("order", "FM")
    if m["order"] not in ops:
        raise ModelError("order %r is not among order_parameters %s" % (m["order"], list(ops)))

    mc = dict(MC_DEFAULTS)
    mc.update(m.get("mc") or {})
    m["mc"] = mc
    if mc["every"] <= 0 or mc["nmeas"] <= 0 or mc["ntherm"] < 0:
        raise ModelError("mc: every and nmeas must be positive, ntherm non-negative")
    return m


def model_hash(m):
    """Hash of the physics content (lattice, spin, exchange, anisotropy, order parameter)."""
    keys = ("spin", "lattice", "exchange", "anisotropy", "order_parameters", "order")
    blob = json.dumps({k: m[k] for k in keys}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def lattice_size(m, L=None):
    L = m["mc"]["L"] if L is None else L
    if isinstance(L, (list, tuple)):
        Lx, Ly = int(L[0]), int(L[1])
    else:
        Lx = Ly = int(L)
    return Lx, Ly


def temperatures(m):
    mc = m["mc"]
    if mc.get("temperatures"):
        T = np.array(mc["temperatures"], dtype=float)
        if np.any(np.diff(T) >= 0):
            raise ModelError("mc.temperatures must be strictly descending (scan from high to low)")
        return T
    if mc["Tmax"] is None or mc["Tmin"] is None:
        raise ModelError("mc needs Tmax and Tmin (or an explicit 'temperatures' list)")
    return np.arange(float(mc["Tmax"]), float(mc["Tmin"]) - 1e-9, -float(mc["step"]))


def seeds(m):
    mc = m["mc"]
    if mc.get("seeds"):
        return [int(s) for s in mc["seeds"]]
    return [int(mc["seed0"]) + int(mc["seed_stride"]) * k for k in range(int(mc["nseeds"]))]


def eta_array(op, Lx, Ly, nsub):
    """Sign pattern eta_n of an order parameter, flattened in site order n = (i*Ly + j)*nsub + u."""
    eta = np.empty(Lx * Ly * nsub)
    if "pattern" in op:
        pat = np.array(op["pattern"], dtype=float)
        pa, pb = pat.shape[0], pat.shape[1]
        if Lx % pa or Ly % pb:
            raise ModelError("lattice %dx%d is not commensurate with a %dx%d pattern" % (Lx, Ly, pa, pb))
    else:
        qa, qb = op["q"]
        sg = op["signs"]
        if (qa and Lx % 2) or (qb and Ly % 2):
            raise ModelError("lattice %dx%d is not commensurate with q=%s" % (Lx, Ly, op["q"]))
    for i in range(Lx):
        for j in range(Ly):
            for u in range(nsub):
                n = (i * Ly + j) * nsub + u
                if "pattern" in op:
                    eta[n] = pat[i % pa, j % pb, u]
                else:
                    eta[n] = sg[u] * (-1.0) ** (qa * i + qb * j)
    return eta


def max_range(m):
    ra = rb = 0
    for sh in m["exchange"]:
        for _, _, da, db in sh["bonds"]:
            ra = max(ra, abs(da))
            rb = max(rb, abs(db))
    return ra, rb


def build_tables(m, Lx, Ly):
    """Neighbour tables for the sweep (all neighbours) and for the energy (forward bonds).

    Within each shell the neighbours of a site are ordered by descending (da, db, to_sub);
    shells keep model-file order.  Forward bonds keep model-file order.
    """
    nsub = len(m["lattice"]["sublattices"])
    ra, rb = max_range(m)
    if Lx <= 2 * ra or Ly <= 2 * rb:
        raise ModelError("lattice %dx%d too small for bonds of range (%d, %d): need L > 2*range"
                         % (Lx, Ly, ra, rb))
    shells = [sh for sh in m["exchange"] if not (sh["Jxy"] == 0.0 and sh["Jz"] == 0.0)]
    # per-sublattice neighbour templates
    nb_tmpl = [[] for _ in range(nsub)]   # (to_sub, da, db, Jxy, Jz)
    fw_tmpl = [[] for _ in range(nsub)]
    for sh in shells:
        per = [[] for _ in range(nsub)]
        for s1, s2, da, db in sh["bonds"]:
            per[s1].append((s2, da, db))
            per[s2].append((s1, -da, -db))
            fw_tmpl[s1].append((s2, da, db, sh["Jxy"], sh["Jz"]))
        for u in range(nsub):
            for s2, da, db in sorted(per[u], key=lambda t: (t[1], t[2], t[0]), reverse=True):
                nb_tmpl[u].append((s2, da, db, sh["Jxy"], sh["Jz"]))
    nsite = Lx * Ly * nsub
    maxnb = max(1, max(len(t) for t in nb_tmpl))
    maxfw = max(1, max(len(t) for t in fw_tmpl))
    nnb = np.zeros(nsite, dtype=np.int64)
    nbr = np.zeros((nsite, maxnb), dtype=np.int64)
    jxy = np.zeros((nsite, maxnb))
    jz = np.zeros((nsite, maxnb))
    nfw = np.zeros(nsite, dtype=np.int64)
    fw = np.zeros((nsite, maxfw), dtype=np.int64)
    fjxy = np.zeros((nsite, maxfw))
    fjz = np.zeros((nsite, maxfw))
    for i in range(Lx):
        for j in range(Ly):
            for u in range(nsub):
                n = (i * Ly + j) * nsub + u
                for k, (s2, da, db, a, c) in enumerate(nb_tmpl[u]):
                    nbr[n, k] = (((i + da) % Lx) * Ly + (j + db) % Ly) * nsub + s2
                    jxy[n, k] = a
                    jz[n, k] = c
                nnb[n] = len(nb_tmpl[u])
                for k, (s2, da, db, a, c) in enumerate(fw_tmpl[u]):
                    fw[n, k] = (((i + da) % Lx) * Ly + (j + db) % Ly) * nsub + s2
                    fjxy[n, k] = a
                    fjz[n, k] = c
                nfw[n] = len(fw_tmpl[u])
    K = np.array(m["anisotropy"]["K"], dtype=float).reshape(nsub, 3)
    return dict(nsub=nsub, nsite=nsite, nnb=nnb, nbr=nbr, jxy=jxy, jz=jz,
                nfw=nfw, fw=fw, fjxy=fjxy, fjz=fjz, K=K)


def bond_lengths(m):
    """Bond lengths (in units of a1, a2) per shell, if lattice.a1/a2 are given; None otherwise.

    Useful to check a hand-written bond list: all bonds of a shell must have the same length.
    """
    lat = m["lattice"]
    if "a1" not in lat or "a2" not in lat:
        return None
    a1, a2 = np.array(lat["a1"], float), np.array(lat["a2"], float)
    sub = np.array(lat["sublattices"], float)
    out = {}
    for sh in m["exchange"]:
        d = []
        for s1, s2, da, db in sh["bonds"]:
            f = sub[s2] - sub[s1] + np.array([da, db])
            d.append(float(np.linalg.norm(f[0] * a1 + f[1] * a2)))
        out[sh["name"]] = d
    return out
