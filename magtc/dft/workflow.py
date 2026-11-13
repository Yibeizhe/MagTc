"""`magtc dft setup` and `magtc dft collect`: VASP energy-mapping workflow.

setup:   POSCAR + template (INCAR, KPOINTS, optional POTCAR)  ->  one directory per configuration
         (collinear configurations for J; FM along x, y, z with SOC for the single-ion K) + plan.json
collect: final energies (+ local-moment check)  ->  fitted J_s, K_a  ->  MagTc model file (YAML)
"""
import datetime
import json
import os
import shutil

import numpy as np

from ..structure import Structure
from . import mapping, vasp

KMEV = 1000.0  # eV -> meV


def _mag_structure(p, magnetic, dim):
    spec = vasp.atom_species(p)
    idx = [a for a, s in enumerate(spec) if s in magnetic]
    if not idx:
        raise ValueError("no atoms of species %s in the POSCAR (species: %s)" % (magnetic, p["species"]))
    return idx, Structure(p["lattice"], p["frac"][idx], dim, [spec[a] for a in idx])


def _magmom(p, index, mag_idx, M, config, moment, axis=None):
    """Per-atom MAGMOM: +-moment on magnetic atoms by `config` [M1,M2,M3,nsub]; 0 elsewhere.
    With axis (0/1/2) a non-collinear MAGMOM (3 per atom) along that axis."""
    nat = sum(p["counts"]) * M[0] * M[1] * M[2]
    val = np.zeros(nat)
    for (c1, c2, c3, a), n in index.items():
        if a in mag_idx:
            u = mag_idx.index(a)
            val[n] = moment[vasp.atom_species(p)[a]] * (config[c1, c2, c3, u] if config is not None else 1.0)
    if axis is None:
        return vasp.magmom_string(val)
    vec = np.zeros((nat, 3))
    vec[:, axis] = val
    return vasp.magmom_string(vec.ravel())


def setup(poscar, magnetic, nshell, supercell, outdir, template=None, method="fit", nconf=None, moment=None,
          mae=True, dim=2, rmax_cells=4, log=print):
    p = vasp.read_poscar(poscar)
    M = [int(x) for x in supercell] + [1] * (3 - len(supercell))
    if dim == 2 and M[2] != 1:
        raise ValueError("2D: the supercell must be M1 x M2 x 1")
    mag_idx, S = _mag_structure(p, magnetic, dim)
    shells = S.shells(nshell, rmax_cells=rmax_cells)
    moment = {s: float((moment or {}).get(s, 3.0)) for s in magnetic}
    sup, index = vasp.supercell(p, M)
    nsub = S.nsub

    tmpl = {}
    if template:
        for f in ("INCAR", "KPOINTS", "POTCAR"):
            fp = os.path.join(template, f)
            if os.path.exists(fp):
                tmpl[f] = fp
    if "INCAR" not in tmpl:
        log("note: no template INCAR given; writing minimal INCARs - add your ENCUT, LDAU, smearing, ... before running")
    os.makedirs(outdir, exist_ok=True)

    jobs = []
    if method == "fit":
        nconf = nconf or (2 * nshell + 2)
        configs, A = mapping.choose_configs(M, nsub, [b for _, b in shells], nconf)
        for i, c in enumerate(configs):
            jobs.append(dict(dir="col_%03d" % i, kind="collinear", config=c.tolist(), row=A[i].tolist()))
    elif method == "fourstate":
        for s in range(nshell):
            for tag, c in zip(("uu", "ud", "du", "dd"), mapping.four_state_configs(M, nsub, [b for _, b in shells], s)):
                jobs.append(dict(dir="fs_J%d_%s" % (s + 1, tag), kind="fourstate", shell=s, state=tag,
                                 config=c.tolist()))
    else:
        raise ValueError("method must be fit or fourstate")
    if mae:
        for ax, name in enumerate("xyz"):
            jobs.append(dict(dir="mae_%s" % name, kind="mae", axis=ax, config=None))

    base_incar = open(tmpl["INCAR"]).read() if "INCAR" in tmpl else "SYSTEM = magtc\nPREC = Accurate\nEDIFF = 1E-7\n"
    kp_note = None
    for job in jobs:
        d = os.path.join(outdir, job["dir"])
        os.makedirs(d, exist_ok=True)
        vasp.write_poscar(os.path.join(d, "POSCAR"), sup, comment="%s %s" % (p["comment"], job["dir"]))
        cfg = np.array(job["config"]) if job["config"] is not None else None
        if job["kind"] == "mae":
            upd = dict(ISPIN=2, LSORBIT=".TRUE.", LNONCOLLINEAR=".TRUE.",
                       MAGMOM=_magmom(p, index, mag_idx, M, None, moment, axis=job["axis"]))
        else:
            upd = dict(ISPIN=2, LSORBIT=None, LNONCOLLINEAR=None, MAGMOM=_magmom(p, index, mag_idx, M, cfg, moment))
        with open(os.path.join(d, "INCAR"), "w") as f:
            f.write(vasp.incar_set(base_incar, upd))
        if "KPOINTS" in tmpl:
            txt, ok = vasp.kpoints_scaled(open(tmpl["KPOINTS"]).read(), M)
            kp_note = kp_note if ok else "KPOINTS is not an automatic mesh: copied unchanged, check it for the supercell"
            with open(os.path.join(d, "KPOINTS"), "w") as f:
                f.write(txt)
        if "POTCAR" in tmpl:
            shutil.copy(tmpl["POTCAR"], os.path.join(d, "POTCAR"))
    if kp_note:
        log("note: " + kp_note)
    if "POTCAR" not in tmpl:
        log("note: no POTCAR in the template; put one (species order %s) in every directory" % " ".join(sup["species"]))

    plan = dict(magtc_dft=1, created=datetime.datetime.now().isoformat(timespec="seconds"), poscar=os.path.abspath(poscar),
                primitive=dict(lattice=p["lattice"].tolist(), species=p["species"], counts=p["counts"],
                               frac=p["frac"].tolist()),
                magnetic=magnetic, mag_atoms=mag_idx, moment=moment, dim=dim, supercell=M, method=method,
                shells=[dict(name="J%d" % (k + 1), distance=r, bonds=b) for k, (r, b) in enumerate(shells)],
                sites=[dict(label=S.labels[u], frac=S.frac[u].tolist()) for u in range(nsub)],
                n_mag_supercell=nsub * M[0] * M[1] * M[2], jobs=jobs)
    with open(os.path.join(outdir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)
    log("magtc dft setup: %d directories in %s (%s, %d shells, supercell %s, %d magnetic atoms)"
        % (len(jobs), outdir, method, nshell, "x".join(map(str, M)), plan["n_mag_supercell"]))
    for k, (r, b) in enumerate(shells):
        log("  J%d: d = %.4f A, %d bonds per cell" % (k + 1, r, len(b)))
    if method == "fit":
        log("  design matrix condition number %.1f" % np.linalg.cond(np.array([j["row"] for j in jobs if j["kind"] == "collinear"])))
    return plan


def _moment_check(d, plan, job, index_cache):
    mom = vasp.read_moments(d)
    if mom is None or job["config"] is None:
        return None
    p = plan["primitive"]
    M = plan["supercell"]
    if index_cache.get("index") is None:
        _, index_cache["index"] = vasp.supercell(dict(p, frac=np.array(p["frac"]), lattice=np.array(p["lattice"]),
                                                       comment=""), M)
    index = index_cache["index"]
    cfg = np.array(job["config"])
    spec = vasp.atom_species(dict(species=p["species"], counts=p["counts"]))
    bad = []
    for (c1, c2, c3, a), n in index.items():
        if a in plan["mag_atoms"] and n < len(mom):
            u = plan["mag_atoms"].index(a)
            want = cfg[c1, c2, c3, u]
            m0 = plan["moment"][spec[a]]
            if np.sign(mom[n]) != np.sign(want) or abs(mom[n]) < 0.25 * abs(m0):
                bad.append((n + 1, round(mom[n], 3), int(want)))
    return bad


def collect(plan_dir, energy="E0", out=None, name=None, log=print, strict=False):
    plan = json.load(open(os.path.join(plan_dir, "plan.json")))
    E, warn, cache = {}, [], {}
    for job in plan["jobs"]:
        d = os.path.join(plan_dir, job["dir"])
        try:
            E[job["dir"]] = vasp.read_energy(d, energy)
        except FileNotFoundError as e:
            warn.append(str(e))
            continue
        bad = _moment_check(d, plan, job, cache)
        if bad:
            warn.append("%s: local moments differ from the intended configuration (atom, moment, intended sign): %s"
                        % (job["dir"], bad[:6]))
    for w in warn:
        log("WARNING: " + w)
    if strict and warn:
        raise RuntimeError("problems found (see warnings); rerun without --strict to fit anyway")

    nsh = len(plan["shells"])
    res = dict(method=plan["method"], energy=energy, supercell=plan["supercell"], energies_eV=E)
    if plan["method"] == "fit":
        jobs = [j for j in plan["jobs"] if j["kind"] == "collinear" and j["dir"] in E]
        A = np.array([j["row"] for j in jobs])
        f = mapping.fit(A, [E[j["dir"]] for j in jobs])
        J = [x * KMEV for x in f["J"]]
        res.update(J_meV=J, J_err_meV=[x * KMEV for x in f["J_err"]] if f["J_err"] else None,
                   rms_meV=f["rms"] * KMEV, loo_rms_meV=(f["loo_rms"] * KMEV if f["loo_rms"] is not None else None),
                   cond=f["cond"], dof=f["dof"],
                   residuals_meV={j["dir"]: r * KMEV for j, r in zip(jobs, f["residuals"])})
        if f["dof"] == 0:
            log("note: as many configurations as unknowns - no error estimate; use more configurations")
    else:
        J, Jerr = [], []
        for s in range(nsh):
            try:
                e4 = [E["fs_J%d_%s" % (s + 1, t)] for t in ("uu", "ud", "du", "dd")]
            except KeyError:
                raise RuntimeError("four-state energies of J%d incomplete" % (s + 1))
            J.append(mapping.four_state_J(e4) * KMEV)
        res.update(J_meV=J, J_err_meV=None)
    K = [0.0, 0.0, 0.0]
    if all("mae_%s" % a in E for a in "xyz"):
        e = np.array([E["mae_%s" % a] for a in "xyz"])
        K = ((e - e.min()) / plan["n_mag_supercell"] * KMEV).tolist()
        res["K_meV"] = K
    elif any(j["kind"] == "mae" for j in plan["jobs"]):
        log("WARNING: MAE energies incomplete; K = 0")

    model = dict(
        name=name or "dft_model",
        description="Energy-mapped model from VASP (%s, %s energies)" % (plan["method"], energy),
        reference="magtc dft collect on %s" % os.path.abspath(plan_dir),
        spin="heisenberg",
        structure=dict(dim=plan["dim"], lattice=plan["primitive"]["lattice"], sites=plan["sites"]),
        interactions=[dict(name=sh["name"], J=float(round(J[k], 6)), bonds=sh["bonds"]) for k, sh in enumerate(plan["shells"])],
        anisotropy=dict(K=[float(round(x, 6)) for x in K]),
        order="FM",
        provenance=dict(res, created=datetime.datetime.now().isoformat(timespec="seconds"), warnings=warn,
                        shells=[dict(name=s["name"], distance=s["distance"]) for s in plan["shells"]]),
    )
    from ..ground import ground, mean_field_tc
    from ..model import normalise
    m = normalise(model)
    g = ground(m)
    if not g["mc_order_is_ground"]:
        model["orders"] = {g["ground_order"]: dict(m["orders"].get(g["ground_order"]) or
                                                   _order_from_label(g["ground_order"]))}
        model["order"] = g["ground_order"]
        log("note: collinear ground state is %s (moments along %s); MC order set to it" % (g["ground_order"], g["ground_axis"]))
    tmf = mean_field_tc(m)
    model["mc"] = dict(L=50 if plan["dim"] == 2 else 16, Tmax=float(round(tmf, 1)), Tmin=float(round(0.05 * tmf, 1)),
                       step=float(round(tmf / 40, 2)))
    model["provenance"]["mean_field_Tc_K"] = tmf
    log("J (meV, per pair, unit spins): " + ", ".join(
        "%s = %.4f%s" % (sh["name"], J[k], "" if not res.get("J_err_meV") else " +- %.4f" % res["J_err_meV"][k])
        for k, sh in enumerate(plan["shells"])))
    if res.get("rms_meV") is not None:
        log("fit rms %.4f meV/supercell, leave-one-out rms %s, condition %.1f"
            % (res["rms_meV"], "%.4f" % res["loo_rms_meV"] if res["loo_rms_meV"] is not None else "n/a", res["cond"]))
    log("K = (%.4f, %.4f, %.4f) meV per magnetic atom; mean-field T_C (FM) = %.1f K" % (tuple(K) + (tmf,)))
    if out:
        import yaml
        with open(out, "w") as f:
            f.write("# Written by magtc dft collect. The mc ladder is a COARSE first scan (0.05-1 x mean-field T_C);\n"
                    "# refine Tmax/Tmin/step around the chi_abs peak before production runs.\n")
            yaml.safe_dump(model, f, sort_keys=False, default_flow_style=None, width=120)
        log("model written to %s" % out)
    return model


def _order_from_label(lbl):
    # "q=(1,0,0) signs=(+,-)"
    q = [int(x) for x in lbl.split("q=(")[1].split(")")[0].split(",")]
    sg = [1.0 if x == "+" else -1.0 for x in lbl.split("signs=(")[1].split(")")[0].split(",")]
    return dict(q=q, signs=sg)
