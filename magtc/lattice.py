"""Supercell tables: site indexing, neighbour lists, forward bonds, order-parameter signs.

Site index n = ((i*L2 + j)*L3 + k)*nsub + u for cell (i, j, k) and sublattice u.
For L3 = 1 this is the indexing of MagTc 1.x and of kernels/tc_jcp_rect.py.
"""
import numpy as np

from .model import ModelError


def max_range(m):
    r = [0, 0, 0]
    for sh in m["interactions"]:
        for b in sh["bonds"]:
            for a in range(3):
                r[a] = max(r[a], abs(b[2 + a]))
    return r


def check_size(m, L):
    r = max_range(m)
    for a in range(3):
        if r[a] and L[a] <= 2 * r[a]:
            raise ModelError("supercell %s too small for bonds of range %s along axis %d: need L > 2*range"
                             % (list(L), r, a + 1))


def build_tables(m, L):
    """Tables for the kernels.  Within a shell, neighbours are ordered by descending (da, db, dc, s2);
    shells keep model order; forward bonds keep model order (this reproduces the reference kernel)."""
    L1, L2, L3 = L
    nsub = len(m["structure"]["sites"])
    check_size(m, L)
    shells = [sh for sh in m["interactions"] if not (sh["Jxy"] == 0.0 and sh["Jz"] == 0.0)]
    nb_t = [[] for _ in range(nsub)]
    fw_t = [[] for _ in range(nsub)]
    for sh in shells:
        per = [[] for _ in range(nsub)]
        for s1, s2, da, db, dc in sh["bonds"]:
            per[s1].append((s2, da, db, dc))
            per[s2].append((s1, -da, -db, -dc))
            fw_t[s1].append((s2, da, db, dc, sh["Jxy"], sh["Jz"]))
        for u in range(nsub):
            for s2, da, db, dc in sorted(per[u], key=lambda t: (t[1], t[2], t[3], t[0]), reverse=True):
                nb_t[u].append((s2, da, db, dc, sh["Jxy"], sh["Jz"]))
    nsite = L1 * L2 * L3 * nsub
    maxnb = max(1, max(len(t) for t in nb_t))
    maxfw = max(1, max(len(t) for t in fw_t))
    T = dict(nnb=np.zeros(nsite, np.int64), nbr=np.zeros((nsite, maxnb), np.int64),
             jxy=np.zeros((nsite, maxnb)), jz=np.zeros((nsite, maxnb)),
             nfw=np.zeros(nsite, np.int64), fw=np.zeros((nsite, maxfw), np.int64),
             fjxy=np.zeros((nsite, maxfw)), fjz=np.zeros((nsite, maxfw)))

    def idx(i, j, k, u):
        return (((i % L1) * L2 + (j % L2)) * L3 + (k % L3)) * nsub + u

    for i in range(L1):
        for j in range(L2):
            for k in range(L3):
                for u in range(nsub):
                    n = idx(i, j, k, u)
                    for c, (s2, da, db, dc, a, z) in enumerate(nb_t[u]):
                        T["nbr"][n, c] = idx(i + da, j + db, k + dc, s2)
                        T["jxy"][n, c] = a
                        T["jz"][n, c] = z
                    T["nnb"][n] = len(nb_t[u])
                    for c, (s2, da, db, dc, a, z) in enumerate(fw_t[u]):
                        T["fw"][n, c] = idx(i + da, j + db, k + dc, s2)
                        T["fjxy"][n, c] = a
                        T["fjz"][n, c] = z
                    T["nfw"][n] = len(fw_t[u])
    T.update(L=tuple(L), nsub=nsub, nsite=nsite,
             K=np.array(m["anisotropy"]["K"], float).reshape(nsub, 3))
    return T


def eta_array(op, L, nsub):
    """Sign pattern of an order parameter in site order."""
    L1, L2, L3 = L
    if "pattern" in op:
        pat = np.array(op["pattern"], float)
        pa, pb, pc = pat.shape[:3]
        if L1 % pa or L2 % pb or L3 % pc:
            raise ModelError("supercell %s not commensurate with a %dx%dx%d pattern" % (list(L), pa, pb, pc))
    else:
        q, sg = op["q"], op["signs"]
        if any(q[a] and L[a] % 2 for a in range(3)):
            raise ModelError("supercell %s not commensurate with q = %s" % (list(L), q))
    eta = np.empty(L1 * L2 * L3 * nsub)
    n = 0
    for i in range(L1):
        for j in range(L2):
            for k in range(L3):
                for u in range(nsub):
                    eta[n] = pat[i % pa, j % pb, k % pc, u] if "pattern" in op else \
                        sg[u] * (-1.0) ** (q[0] * i + q[1] * j + q[2] * k)
                    n += 1
    return eta
