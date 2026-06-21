#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot L=50 Monte Carlo M(T) and Cv(T) for equilibrium VN, b=2.7342 Å.
This script uses the new L50 scan CSV, not the old MCSolver result.txt.

Default input:
  tc_TableS3_L50_prod_results/b_2p7342_scan.csv

Output:
  VN_b273_L50_MC.png
  VN_b273_L50_MC.pdf
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parabolic_peak(T, Y, nfit=3):
    """Quadratic fit around the discrete maximum. Returns refined peak position."""
    T = np.asarray(T, dtype=float)
    Y = np.asarray(Y, dtype=float)
    imax = int(np.argmax(Y))
    half = nfit // 2
    i0 = max(0, imax - half)
    i1 = min(len(T), imax + half + 1)
    if i1 - i0 < 3:
        return float(T[imax])
    x = T[i0:i1]
    y = Y[i0:i1]
    a, b, c = np.polyfit(x, y, 2)
    if a >= 0:
        return float(T[imax])
    tc = -b / (2 * a)
    if tc < min(x) or tc > max(x):
        return float(T[imax])
    return float(tc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="tc_TableS3_L50_prod_results/b_2p7342_scan.csv",
                    help="L50 scan csv file for b=2.7342 Å")
    ap.add_argument("--out", default="VN_b273_L50_MC", help="output filename prefix")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--show-chi", action="store_true",
                    help="also plot normalized chi_abs as a thin dashed guide on the Cv axis")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).sort_values("T_K").reset_index(drop=True)

    T = df["T_K"].to_numpy()
    M = df["M_abs"].to_numpy()
    Cv = df["Cv"].to_numpy()
    M_sem = df["Mabs_sem"].to_numpy() if "Mabs_sem" in df.columns else None
    Cv_sem = df["Cv_sem"].to_numpy() if "Cv_sem" in df.columns else None

    Tc = parabolic_peak(T, Cv, nfit=3)

    color_M = "#e85d2a"
    color_Cv = "#5b4bd6"
    color_chi = "#2c7fb8"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.linewidth": 2.2,
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    })

    fig, ax1 = plt.subplots(figsize=(5.2, 4.2))
    ax2 = ax1.twinx()

    # left axis: normalized magnetization |M|
    ax1.errorbar(
        T, M, yerr=M_sem,
        color=color_M, marker="o", markersize=5.2,
        linestyle="-", linewidth=2.6,
        capsize=2.0, elinewidth=1.0,
        markerfacecolor=color_M, markeredgecolor="white", markeredgewidth=0.7,
        label=r"$|M|$"
    )
    ax1.set_xlabel("Temperature (K)", fontsize=15)
    ax1.set_ylabel(r"Magnetization $|M|$", color=color_M, fontsize=15)
    ax1.tick_params(axis="y", colors=color_M, labelcolor=color_M)
    ax1.set_ylim(0, max(M) * 1.12)

    # right axis: actual Cv from energy fluctuation
    ax2.errorbar(
        T, Cv, yerr=Cv_sem,
        color=color_Cv, marker="o", markersize=5.2,
        linestyle="-", linewidth=2.6,
        capsize=2.0, elinewidth=1.0,
        markerfacecolor=color_Cv, markeredgecolor="white", markeredgewidth=0.7,
        label=r"$C_v$"
    )

    if args.show_chi and "chi_abs" in df.columns:
        chi = df["chi_abs"].to_numpy()
        # rescale chi_abs onto the Cv range only for visual comparison
        chi_scaled = (chi - chi.min()) / (chi.max() - chi.min())
        chi_scaled = chi_scaled * (Cv.max() - Cv.min()) + Cv.min()
        ax2.plot(T, chi_scaled, color=color_chi, linestyle="--", linewidth=1.5,
                 label=r"$\chi_{|M|}$ (scaled)")

    ax2.set_ylabel(r"Heat capacity $C_v$", color=color_Cv, fontsize=15)
    ax2.tick_params(axis="y", colors=color_Cv, labelcolor=color_Cv)
    ax2.set_ylim(min(Cv) * 0.96, max(Cv) * 1.08)

    # Tc marker from Cv peak
    ax1.axvline(Tc, color="black", linestyle="--", linewidth=1.8)
    ax2.text(
        Tc + 7, max(Cv) * 0.985,
        rf"$T_C \approx {Tc:.0f}$ K",
        color=color_Cv, fontsize=14,
        ha="left", va="top"
    )

    # spine colors
    ax1.spines["left"].set_color(color_M)
    ax1.spines["left"].set_linewidth(2.4)
    ax2.spines["right"].set_color(color_Cv)
    ax2.spines["right"].set_linewidth(2.4)
    ax1.spines["bottom"].set_linewidth(2.2)
    ax1.spines["top"].set_linewidth(2.2)
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    ax1.set_xlim(T.min() - 5, T.max() + 5)
    ax1.grid(False)

    # compact legend, only if chi is shown
    if args.show_chi:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right", fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(f"{args.out}.pdf", bbox_inches="tight")
    print(f"Saved: {args.out}.png and {args.out}.pdf")
    print(f"Tc from Cv peak: {Tc:.2f} K")


if __name__ == "__main__":
    main()
