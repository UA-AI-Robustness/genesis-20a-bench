#!/usr/bin/env python3
"""Rebuild the preliminary figure from existing results JSON(s).

Figure design (in priority order):
  1. Three curves: CMA-ES agent (hero), PGD baseline, random noise control.
  2. CROSSING point eps* where the CMA-ES curve crosses tau_bad.
  3. HEADROOM: vulnerable quadrant above tau_bad.
  4. GAP arrow between CMA-ES and PGD at eps_max (AI advantage).

Torch-free, LaTeX text, compact wrapfigure aspect, warm panel bg.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pylab import rcParams


font_size = 11
FIG_W, FIG_H = 4.05, 2.75
rcParams.update({
    "axes.labelsize": font_size - 1,
    "axes.linewidth": 0.9,
    "font.size": font_size,
    "legend.fontsize": font_size - 2,
    "xtick.labelsize": font_size - 2,
    "xtick.major.size": 2.8,
    "ytick.labelsize": font_size - 2,
    "ytick.major.size": 2.8,
    "text.usetex": True,
    "figure.figsize": [FIG_W, FIG_H],
})

COL_CMAES      = "#1565C0"
COL_CMAES_FILL = "#42A5F5"
COL_PGD        = "#C62828"
COL_PGD_FILL   = "#EF5350"
COL_NOISE      = "#607D8B"
COL_TAU        = "#B71C1C"
COL_VULN       = "#FFCDD2"
COL_CLEAN      = "#666666"
PANEL_BG       = "#FFF8F0"


def _extract(rj: dict) -> Tuple[list, list, list, list]:
    eps = [r["epsilon"] for r in rj["results"]]
    mean = [r["mean"] for r in rj["results"]]
    lo = [r["lo"] for r in rj["results"]]
    hi = [r["hi"] for r in rj["results"]]
    return eps, mean, lo, hi


def _find_cross(eps_list, means, tau) -> Optional[float]:
    for i in range(1, len(eps_list)):
        if means[i - 1] <= tau < means[i]:
            frac = (tau - means[i - 1]) / (means[i] - means[i - 1])
            return eps_list[i - 1] + frac * (eps_list[i] - eps_list[i - 1])
    return None


def make_figure(
    pgd: dict,
    rand: Optional[dict],
    cmaes: Optional[dict],
    out_path: Path,
) -> None:
    eps_p, mean_p, lo_p, hi_p = _extract(pgd)
    clean = mean_p[0]
    tau = 1.5 * clean
    x_max = max(eps_p)

    pgd_end = mean_p[-1]

    if rand is not None:
        eps_r, mean_r, lo_r, hi_r = _extract(rand)
        x_max = max(x_max, max(eps_r))

    if cmaes is not None:
        eps_c, mean_c, lo_c, hi_c = _extract(cmaes)
        x_max = max(x_max, max(eps_c))
        cmaes_end = mean_c[-1]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_facecolor(PANEL_BG)

    hero_eps = eps_c if cmaes is not None else eps_p
    hero_mean = mean_c if cmaes is not None else mean_p
    eps_cross = _find_cross(hero_eps, hero_mean, tau)

    y_top_est = max(max(hi_p), tau * 1.18)
    if rand is not None:
        y_top_est = max(y_top_est, max(hi_r))
    if cmaes is not None:
        y_top_est = max(y_top_est, max(hi_c))
    y_top = y_top_est * 1.05

    if eps_cross is not None:
        ax.axvspan(eps_cross, x_max + 0.02,
                   ymin=(tau - 0) / y_top, ymax=1.0,
                   color=COL_VULN, alpha=0.55, zorder=1, linewidth=0)
        ax.text(
            (eps_cross + x_max) / 2.0,
            tau + (y_top - tau) * 0.55,
            r"\textbf{vulnerable}",
            ha="center", va="center",
            fontsize=font_size - 1, color=COL_TAU,
            zorder=5,
        )

    if rand is not None:
        ax.fill_between(eps_r, lo_r, hi_r, alpha=0.10, color=COL_NOISE,
                        linewidth=0, zorder=2)
        ax.plot(eps_r, mean_r, "s--", lw=1.0, ms=2.6,
                color=COL_NOISE, markerfacecolor=COL_NOISE,
                markeredgecolor=COL_NOISE, alpha=0.85,
                label=r"Random $L_\infty$ noise", zorder=3)

    ax.fill_between(eps_p, lo_p, hi_p, alpha=0.18, color=COL_PGD_FILL,
                    linewidth=0, zorder=3.5)
    ax.plot(eps_p, mean_p, "^--", lw=1.2, ms=3.0,
            color=COL_PGD, markerfacecolor=COL_PGD_FILL,
            markeredgecolor=COL_PGD, markeredgewidth=0.7,
            label=r"White-box PGD (20 steps)", zorder=4.5)

    if cmaes is not None:
        ax.fill_between(eps_c, lo_c, hi_c, alpha=0.22, color=COL_CMAES_FILL,
                        linewidth=0, zorder=4)
        ax.plot(eps_c, mean_c, "o-", lw=1.8, ms=3.6,
                color=COL_CMAES, markerfacecolor=COL_CMAES_FILL,
                markeredgecolor=COL_CMAES, markeredgewidth=0.8,
                label=r"\textbf{CMA-ES + DCT agent (ours)}", zorder=5)

    ax.axhline(tau, color=COL_TAU, ls="--", lw=1.0, zorder=2.5,
               label=rf"$\tau_{{\mathrm{{bad}}}}\!=\!{tau:.3f}$ "
                     rf"$(1.5\!\times\!$clean$)$")
    ax.axhline(clean, color=COL_CLEAN, ls=":", lw=0.7, alpha=0.7, zorder=2.5,
               label=rf"clean $= {clean:.3f}$")

    if eps_cross is not None:
        ax.axvline(eps_cross, color=COL_TAU, ls=":", lw=0.8, alpha=0.8,
                   zorder=4)
        ax.text(
            eps_cross + 0.003, clean * 0.25,
            rf"$\varepsilon^{{*}}\!=\!{eps_cross:.2f}$",
            ha="left", va="center",
            fontsize=font_size - 1, color=COL_TAU,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=PANEL_BG,
                      edgecolor="none", alpha=0.85),
            zorder=7,
        )

    if cmaes is not None:
        delta_cmaes = cmaes_end - clean
        delta_pgd = pgd_end - clean
        if delta_pgd > 1e-6 and delta_cmaes > delta_pgd:
            pct_more = (delta_cmaes / delta_pgd - 1) * 100
            arrow_x = x_max * 0.94
            arr = FancyArrowPatch(
                (arrow_x, pgd_end),
                (arrow_x, cmaes_end),
                arrowstyle="<->",
                mutation_scale=7,
                color="#333333", lw=1.0, zorder=6,
            )
            ax.add_patch(arr)
            ax.annotate(
                rf"$+{pct_more:.0f}\%$",
                xy=(arrow_x, (pgd_end + cmaes_end) / 2),
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right", va="center",
                fontsize=font_size - 1, color=COL_CMAES,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor=PANEL_BG,
                          edgecolor="none", alpha=0.9),
                zorder=7,
            )

    ax.set_xlabel(r"Perturbation budget $\varepsilon$ "
                  r"($L_\infty$ on BoxCox load)")
    ax.set_ylabel(r"24\,h NRMSE")
    ax.set_xlim(-0.005, x_max + 0.015)
    ax.set_ylim(0, y_top)
    ax.grid(axis="both", alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    preferred_order = [
        r"\textbf{CMA-ES + DCT agent (ours)}",
        r"White-box PGD (20 steps)",
        r"Random $L_\infty$ noise",
    ]
    order = []
    for target in preferred_order:
        for i, L in enumerate(labels):
            if L == target and i not in order:
                order.append(i); break
    for i in range(len(labels)):
        if i not in order:
            order.append(i)
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]

    leg = ax.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=2, fontsize=font_size - 2,
        framealpha=0.0, handlelength=1.8,
        borderpad=0.2, labelspacing=0.38, columnspacing=1.2,
    )
    leg.set_zorder(20)

    fig.text(
        0.99, 0.005,
        rf"ckpt {pgd.get('ckpt_sha256_8','?')} $\cdot$ "
        rf"seed {pgd.get('sampling_seed','?')}",
        ha="right", va="bottom",
        fontsize=font_size - 3.5, color="#888",
    )

    plt.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf",
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".png"),
                dpi=220, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path,
                    help="PGD results JSON (primary curve).")
    ap.add_argument("--random-json", type=Path, default=None,
                    help="Optional random-noise control JSON.")
    ap.add_argument("--cmaes-json", type=Path, default=None,
                    help="Optional CMA-ES agent results JSON.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    with open(args.json) as f:
        pgd_data = json.load(f)
    rand_data = None
    if args.random_json is not None:
        with open(args.random_json) as f:
            rand_data = json.load(f)
    cmaes_data = None
    if args.cmaes_json is not None:
        with open(args.cmaes_json) as f:
            cmaes_data = json.load(f)
    make_figure(pgd_data, rand_data, cmaes_data, args.out)


if __name__ == "__main__":
    main()
