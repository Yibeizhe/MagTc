"""Minimal VASP file handling: POSCAR, INCAR tags, KPOINTS scaling, energies and moments."""
import math
import os
import re

import numpy as np


# ---------------------------------------------------------------- POSCAR
def read_poscar(path):
    """VASP 5 POSCAR (species line required).  Returns dict(comment, lattice[3x3] A, species, counts, frac[N,3])."""
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    comment = lines[0].strip()
    scale = float(lines[1].split()[0])
    lat = np.array([[float(x) for x in lines[2 + i].split()[:3]] for i in range(3)])
    f = (abs(scale) / abs(np.linalg.det(lat))) ** (1.0 / 3.0) if scale < 0 else scale   # negative = volume
    lat = lat * f
    species = lines[5].split()
    if all(re.fullmatch(r"\d+", x) for x in species):
        raise ValueError("%s: POSCAR without the species line (VASP 4 format) is not supported" % path)
    counts = [int(x) for x in lines[6].split()]
    i = 7
    if lines[i].strip()[:1].lower() == "s":
        i += 1  # selective dynamics
    cart = lines[i].strip()[:1].lower() in ("c", "k")
    n = sum(counts)
    pos = np.array([[float(x) for x in lines[i + 1 + a].split()[:3]] for a in range(n)])
    frac = np.linalg.solve(lat.T, (pos * f).T).T if cart else pos
    return dict(comment=comment, lattice=lat, species=species, counts=counts, frac=frac % 1.0)


def write_poscar(path, p, comment=None):
    with open(path, "w") as f:
        f.write((comment or p["comment"]) + "\n1.0\n")
        for v in p["lattice"]:
            f.write("  %20.12f %20.12f %20.12f\n" % tuple(v))
        f.write("  " + " ".join(p["species"]) + "\n  " + " ".join(str(c) for c in p["counts"]) + "\nDirect\n")
        for r in p["frac"]:
            f.write("  %18.12f %18.12f %18.12f\n" % tuple(r))


def atom_species(p):
    out = []
    for s, c in zip(p["species"], p["counts"]):
        out += [s] * c
    return out


def supercell(p, M):
    """Diagonal supercell M = (M1, M2, M3).  Atoms are grouped by species, then cell (c1, c2, c3) in
    C order, then the order of the primitive cell.  Returns (poscar dict, index) with
    index[(c1, c2, c3, a)] = atom number in the supercell of primitive atom a in cell (c1, c2, c3)."""
    M = [int(x) for x in M]
    spec = atom_species(p)
    frac, index = [], {}
    for s in p["species"]:
        for c1 in range(M[0]):
            for c2 in range(M[1]):
                for c3 in range(M[2]):
                    for a, sa in enumerate(spec):
                        if sa != s:
                            continue
                        index[(c1, c2, c3, a)] = len(frac)
                        frac.append((p["frac"][a] + [c1, c2, c3]) / M)
    q = dict(comment=p["comment"] + " supercell %dx%dx%d" % tuple(M), lattice=p["lattice"] * np.array(M)[:, None],
             species=list(p["species"]), counts=[c * M[0] * M[1] * M[2] for c in p["counts"]], frac=np.array(frac))
    return q, index


# ---------------------------------------------------------------- INCAR / KPOINTS
def magmom_string(vals):
    """Compress a MAGMOM list with VASP's n*value notation."""
    out, i = [], 0
    vals = ["%g" % (0.0 if abs(v) < 1e-12 else v) for v in vals]
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        out.append(vals[i] if j == i else "%d*%s" % (j - i + 1, vals[i]))
        i = j + 1
    return " ".join(out)


def incar_set(text, updates):
    """Set INCAR tags (case-insensitive): existing assignments are removed and the new values appended
    in a marked block.  A value of None only removes the tag.  Comments are kept."""
    keys = {k.upper(): v for k, v in updates.items()}
    out = []
    for line in text.splitlines():
        cut = min([i for i in (line.find("!"), line.find("#")) if i >= 0] or [len(line)])
        code, cmt = line[:cut], line[cut:]
        parts = [x.strip() for x in code.split(";")
                 if x.strip() and not ("=" in x and x.split("=", 1)[0].strip().upper() in keys)]
        newcode = "; ".join(parts)
        if newcode or cmt.strip():
            out.append((newcode + " " + cmt).strip())
        elif not line.strip():
            out.append("")
    out.append("# --- set by magtc dft setup ---")
    for k, v in keys.items():
        if v is not None:
            out.append("%s = %s" % (k, v))
    return "\n".join(out) + "\n"


def kpoints_scaled(text, M):
    """Divide an automatic Gamma/Monkhorst-Pack mesh by the supercell multiples (rounded up).
    Returns (new_text, scaled?)."""
    lines = text.splitlines()
    if len(lines) >= 4 and lines[1].strip() == "0" and lines[2].strip()[:1].lower() in ("g", "m"):
        try:
            n = [int(x) for x in lines[3].split()[:3]]
        except ValueError:
            return text, False
        new = [max(1, int(math.ceil(n[i] / M[i]))) for i in range(3)]
        lines[3] = "%d %d %d" % tuple(new)
        return "\n".join(lines) + "\n", True
    return text, False


# ---------------------------------------------------------------- outputs
def read_energy(d, kind="E0"):
    """Final energy (eV) of a VASP run in directory d.  kind: E0 (sigma -> 0) or F (free energy).
    OSZICAR is read first, OUTCAR as fallback."""
    osz = os.path.join(d, "OSZICAR")
    if os.path.exists(osz):
        last = None
        with open(osz) as f:
            for line in f:
                if "F=" in line and "E0=" in line:
                    last = line
        if last:
            mF = re.search(r"F=\s*([-+0-9.Ee]+)", last)
            mE = re.search(r"E0=\s*([-+0-9.Ee]+)", last)
            return float((mE if kind == "E0" else mF).group(1))
    out = os.path.join(d, "OUTCAR")
    if os.path.exists(out):
        val = None
        with open(out) as f:
            for line in f:
                if kind == "E0" and "energy  without entropy" in line:
                    val = float(line.split("=")[-1].split()[0])
                elif kind == "F" and "free  energy   TOTEN" in line:
                    val = float(line.split("=")[1].split()[0])
        if val is not None:
            return val
    raise FileNotFoundError("no final energy found in %s (OSZICAR/OUTCAR)" % d)


def read_moments(d):
    """Local moments (last 'magnetization (x)' block of OUTCAR, column 'tot'), or None."""
    out = os.path.join(d, "OUTCAR")
    if not os.path.exists(out):
        return None
    with open(out) as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("magnetization (x)"):
            start = i
    if start is None:
        return None
    mom = []
    for line in lines[start + 4:]:
        p = line.split()
        if not p or not p[0].isdigit():
            break
        mom.append(float(p[-1]))
    return mom
