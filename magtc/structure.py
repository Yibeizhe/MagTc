"""Crystal structure of the magnetic sublattice and automatic neighbour shells.

A Structure holds
  * dim      -- number of periodic directions (2: a1, a2; 3: a1, a2, a3),
  * lattice  -- 3x3 array, rows a1, a2, a3 in Angstrom (a3 is only used for geometry when dim = 2),
  * sites    -- magnetic sites (sublattices) with fractional coordinates and a label.

Bonds are written [s1, s2, da, db, dc]: sublattice s1 in cell (i, j, k) couples to sublattice s2
in cell (i+da, j+db, k+dc).  Each pair is listed once.
"""
import numpy as np

TOL_SHELL = 1e-3   # Angstrom: bonds whose lengths differ by less than this belong to one shell


class Structure:
    def __init__(self, lattice, sites, dim=2, labels=None):
        self.lattice = np.array(lattice, dtype=float).reshape(3, 3)
        self.frac = np.array(sites, dtype=float).reshape(-1, 3)
        self.dim = int(dim)
        if self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        self.labels = list(labels) if labels is not None else ["M%d" % i for i in range(len(self.frac))]

    @property
    def nsub(self):
        return len(self.frac)

    def cart(self, u, d=(0, 0, 0)):
        return (self.frac[u] + np.asarray(d, float)) @ self.lattice

    def bond_vector(self, s1, s2, da, db, dc=0):
        return self.cart(s2, (da, db, dc)) - self.cart(s1)

    def bond_length(self, b):
        s1, s2, da, db, dc = b
        return float(np.linalg.norm(self.bond_vector(s1, s2, da, db, dc)))

    def shells(self, nshell, rmax_cells=4, tol=TOL_SHELL):
        """The first `nshell` distance shells.  Returns [(distance, [bond, ...]), ...], each pair once.

        Bonds of equal length (within tol) are grouped together even if they are not related by
        symmetry; split such a shell by hand (explicit bonds) when the distinction matters.
        """
        rng = range(-rmax_cells, rmax_cells + 1)
        rc = rng if self.dim == 3 else [0]
        cand = []
        for s1 in range(self.nsub):
            for s2 in range(s1, self.nsub):
                for da in rng:
                    for db in rng:
                        for dc in rc:
                            if s1 == s2 and (da, db, dc) <= (0, 0, 0):
                                continue   # keep one of each (d, -d) pair and drop the self bond
                            b = (s1, s2, da, db, dc)
                            cand.append((self.bond_length(b), b))
        cand.sort(key=lambda t: t[0])
        out = []
        for r, b in cand:
            if out and abs(r - out[-1][0]) < tol:
                out[-1][1].append(list(b))
            else:
                if len(out) == nshell:
                    break
                out.append((r, [list(b)]))
        if len(out) < nshell:
            raise ValueError("only %d shells found within %d cells; increase rmax_cells" % (len(out), rmax_cells))
        far = max(max(abs(x) for x in b[2:]) for _, bl in out for b in bl)
        if far >= rmax_cells:
            raise ValueError("shell search reached the edge of the search box; increase rmax_cells")
        return [(float(r), sorted(bl, key=lambda b: (b[0], b[1], -b[2], -b[3], -b[4]))) for r, bl in out]
