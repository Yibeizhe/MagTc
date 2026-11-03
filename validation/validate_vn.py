#!/usr/bin/env python3
"""Slow validation (NOT run in CI): monolayer VN at b = 2.7342 A, L = 50.

Expected (JCP 2026 protocol, 8 seeds, 1e5 + 2e5 sweeps): T_C(chi_abs) = 420 +- 2 K;
published JCP value 419.7 K; kernels/tc_jcp_rect.py gave 420.1 +- 1.7 K.

    python validation/validate_vn.py               # full JCP protocol (hours on 8 cores)
    python validation/validate_vn.py --reduced     # L = 50, 8 seeds, 2e4 + 4e4 sweeps, 445 -> 395 K
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from magtc.batch import batch  # noqa: E402

JCP_TC = 419.7


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reduced", action="store_true")
    p.add_argument("--nproc", type=int, default=None)
    p.add_argument("--outdir", default="validation_vn_out")
    a = p.parse_args()
    model = os.path.join(os.path.dirname(HERE), "examples", "vn_b2p7342.yaml")
    ov = dict(Tmax=445.0, Tmin=395.0, step=5.0)
    if a.reduced:
        ov.update(ntherm=20000, nmeas=40000)
    res = batch(model, a.outdir, nproc=a.nproc, overrides=ov)
    tc, err = res["Tc_chi_abs"], res["Tc_chi_abs_err"]
    print("VN b=2.7342: T_C = %.1f +- %.1f K   (JCP 419.7 K, reference kernel 420.1 +- 1.7 K)" % (tc, err))
    ok = abs(tc - JCP_TC) <= max(2.0, 2 * err) + (3.0 if a.reduced else 0.0)
    print("PASS" if ok else "CHECK: deviation %.1f K" % (tc - JCP_TC))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
