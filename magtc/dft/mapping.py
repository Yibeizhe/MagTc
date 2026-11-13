"""Energy mapping of collinear DFT energies onto  E = E0 - sum_s J_s sum_<ij in s> sigma_i sigma_j.

Spins are unit vectors, so the fitted J_s are the per-pair couplings of the MagTc Hamiltonian
directly (|S|^2 absorbed).  Supercells are diagonal, (M1, M2, M3), M3 = 1 in 2D.
"""
import itertools

import numpy as np


def correlations(sigma, shells, M):
    """X_s = -sum_cells sum_{bonds b in shell s} sigma[c, s1] sigma[c + d, s2] for one configuration.
    sigma has shape (M1, M2, M3, nsub)."""
    X = np.zeros(len(shells))
    for k, bonds in enumerate(shells):
        for s1, s2, da, db, dc in bonds:
            shifted = np.roll(sigma[..., s2], shift=(-da, -db, -dc), axis=(0, 1, 2))
            X[k] -= float(np.sum(sigma[..., s1] * shifted))
    return X


def enumerate_configs(M, nsub, max_exhaustive=14, nrandom=4096, seed=1):
    """Collinear sign configurations of the supercell (first spin fixed to +1): exhaustive when
    small, otherwise all q-patterns plus random ones."""
    shape = tuple(M) + (nsub,)
    N = int(np.prod(shape))
    out = []
    if N - 1 <= max_exhaustive:
        for bits in itertools.product((1, -1), repeat=N - 1):
            out.append(np.array((1,) + bits, float).reshape(shape))
        return out
    idx = np.indices(tuple(M))
    for q in itertools.product((0, 1), repeat=3):
        if any(q[a] and M[a] % 2 for a in range(3)):
            continue
        ph = (-1.0) ** (q[0] * idx[0] + q[1] * idx[1] + q[2] * idx[2])
        for tail in itertools.product((1, -1), repeat=nsub - 1):
            sg = np.array((1,) + tail, float)
            out.append(ph[..., None] * sg)
    rng = np.random.default_rng(seed)
    for _ in range(nrandom):
        c = rng.choice((-1.0, 1.0), size=shape)
        c *= c.flat[0]
        out.append(c)
    return out


def choose_configs(M, nsub, shells, nconf, **kw):
    """Greedy D-optimal choice of `nconf` configurations (FM always first) for the fit
    E = E0 + sum_s J_s X_s.  Returns (configs, design matrix [nconf, 1 + nshell])."""
    cands = enumerate_configs(M, nsub, **kw)
    rows, keep = {}, []
    for c in cands:
        X = correlations(c, shells, M)
        key = tuple(np.round(X, 6))
        if key not in rows:
            rows[key] = len(keep)
            keep.append((c, np.concatenate([[1.0], X])))
    A = np.array([r for _, r in keep])
    p = A.shape[1]
    if np.linalg.matrix_rank(A) < p:
        raise ValueError("the %dx%dx%d supercell cannot separate these %d shells (design rank %d < %d); "
                         "use a larger supercell" % (M[0], M[1], M[2], len(shells), np.linalg.matrix_rank(A), p))
    chosen = [0]   # FM: all +1 is the first candidate in both enumerations
    while len(chosen) < min(nconf, len(keep)):
        best, bval = None, -np.inf
        for i in range(len(keep)):
            if i in chosen:
                continue
            S = A[chosen + [i]]
            val = np.linalg.slogdet(S.T @ S + 1e-9 * np.eye(p))[1]
            if val > bval + 1e-12:
                best, bval = i, val
        chosen.append(best)
    return [keep[i][0] for i in chosen], A[chosen]


def fit(A, E):
    """Least squares E = A @ [E0, J_1..J_n].  Returns dict with J, standard errors, residuals, LOO-CV."""
    A = np.asarray(A, float)
    E = np.asarray(E, float)
    n, p = A.shape
    if np.linalg.matrix_rank(A) < p:
        raise ValueError("design matrix is rank deficient: these configurations do not determine all J")
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    res = E - A @ coef
    out = dict(E0=float(coef[0]), J=coef[1:].tolist(), residuals=res.tolist(),
               rms=float(np.sqrt(np.mean(res ** 2))), cond=float(np.linalg.cond(A)), dof=n - p)
    if n > p:
        s2 = float(res @ res / (n - p))
        cov = s2 * np.linalg.inv(A.T @ A)
        out["J_err"] = np.sqrt(np.diag(cov))[1:].tolist()
        loo = []
        for i in range(n):
            m = np.arange(n) != i
            if np.linalg.matrix_rank(A[m]) == p:
                c, *_ = np.linalg.lstsq(A[m], E[m], rcond=None)
                loo.append(float(E[i] - A[i] @ c))
        out["loo_rms"] = float(np.sqrt(np.mean(np.square(loo)))) if loo else None
    else:
        out["J_err"] = None
        out["loo_rms"] = None
    return out


def four_state_configs(M, nsub, shells, s):
    """Four configurations isolating shell s with a pair (A, B) in a FM background.
    J_s = -(E_uu + E_dd - E_ud - E_du) / 4.  Raises if A and B are also coupled by another bond."""
    s1, s2, da, db, dc = shells[s][0]
    A = (0, 0, 0, s1)
    B = (da % M[0], db % M[1], dc % M[2], s2)
    if A == B:
        raise ValueError("four-state, shell %d: the pair maps onto one site in the %s supercell" % (s + 1, list(M)))
    links = []
    for k, bonds in enumerate(shells):
        for b1, b2, ea, eb, ec in bonds:
            for c in itertools.product(range(M[0]), range(M[1]), range(M[2])):
                p = c + (b1,)
                q = ((c[0] + ea) % M[0], (c[1] + eb) % M[1], (c[2] + ec) % M[2], b2)
                if {p, q} == {A, B}:
                    links.append(k)
    if links != [s]:
        raise ValueError("four-state, shell %d: the pair is linked %d times (shells %s) in the %s supercell; "
                         "use a larger supercell" % (s + 1, len(links), [x + 1 for x in links], list(M)))
    out = []
    for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        c = np.ones(tuple(M) + (nsub,))
        c[A] = sa
        c[B] = sb
        out.append(c)
    return out


def four_state_J(E):
    euu, eud, edu, edd = E
    return -(euu + edd - eud - edu) / 4.0
