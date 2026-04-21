#!/usr/bin/env python3
"""Rebuild the preliminary figure from existing results JSON(s).

Figure design (in priority order):
  1. GAP between targeted attack and matched-budget random noise -- the
     single "targeted attacks are not noise" claim.
  2. CROSSING point eps* where the model drops below tau_bad.
  3. HEADROOM -- the vulnerable quadrant (above tau_bad, right of eps*)
     is shaded as "this is what O1/O2 must beat".

The subtitle states the takeaway in words; the plot just reinforces it.
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


# ---------------- style ------------------------------------------------------
# Calibrated for 11pt documentclass, 1in margins (textwidth = 6.5 in).
# Wrapfigure at 0.48\textwidth -> graphic at 0.46\textwidth -> rendered ~2.99 in.
# Slightly wider canvas than the old (3.3, 2.5) so the 2-col legend below the
# axes has more room without compressing label spacing.
font_size = 11
FIG_W, FIG_H = 4.05, 2.55
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

# palette: PGD = hero red, noise = muted gray-blue, vulnerable = warm wash
COL_PGD       = "#C62828"   # hero red
COL_PGD_FILL  = "#EF5350"
COL_NOISE     = "#607D8B"   # muted gray-blue (de-emphasized control)
COL_TAU       = "#B71C1C"   # darker red for the threshold
COL_VULN      = "#FFCDD2"   # very pale red for vulnerable quadrant
COL_CLEAN     = "#666666"
PANEL_BG      = "#FFF8F0"


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


def make_figure(pgd: dict, rand: Optional[dict], out_path: Path) -> None:
    eps_p, mean_p, lo_p, hi_p = _extract(pgd)
    clean = mean_p[0]
    tau = 1.5 * clean
    x_max = max(eps_p)

    # stats for the takeaway subtitle
    pgd_end = mean_p[-1]
    delta_pgd = pgd_end - clean
    if rand is not None:
        eps_r, mean_r, lo_r, hi_r = _extract(rand)
        x_max = max(x_max, max(eps_r))
        rand_end = mean_r[-1]
        delta_rand = max(rand_end - clean, 1e-6)
        gap_ratio = delta_pgd / delta_rand
    else:
        gap_ratio = None

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_facecolor(PANEL_BG)

    # --- HEADROOM: vulnerable quadrant (above tau, right of eps*) -----------
    eps_cross = _find_cross(eps_p, mean_p, tau)
    y_top_est = max(max(hi_p), tau * 1.18)
    if rand is not None:
        y_top_est = max(y_top_est, max(hi_r))
    y_top = y_top_est * 1.05
    if eps_cross is not None:
        ax.axvspan(eps_cross, x_max + 0.02,
                   ymin=(tau - 0) / y_top, ymax=1.0,
                   color=COL_VULN, alpha=0.55, zorder=1, linewidth=0)
        # single-line label, centred inside the shaded quadrant
        ax.text(
            (eps_cross + x_max) / 2.0,
            tau + (y_top - tau) * 0.55,
            r"\textbf{vulnerable}",
            ha="center", va="center",
            fontsize=font_size - 1, color=COL_TAU,
            zorder=5,
        )

    # --- random-noise control (de-emphasized) -------------------------------
    if rand is not None:
        ax.fill_between(eps_r, lo_r, hi_r, alpha=0.10, color=COL_NOISE,
                        linewidth=0, zorder=2)
        ax.plot(eps_r, mean_r, "s--", lw=1.0, ms=2.6,
                color=COL_NOISE, markerfacecolor=COL_NOISE,
                markeredgecolor=COL_NOISE, alpha=0.85,
                label=r"Random $L_\infty$ noise (control)", zorder=3)

    # --- PGD hero curve ------------------------------------------------------
    ax.fill_between(eps_p, lo_p, hi_p, alpha=0.22, color=COL_PGD_FILL,
                    linewidth=0, zorder=3.5)
    ax.plot(eps_p, mean_p, "o-", lw=1.8, ms=3.6,
            color=COL_PGD, markerfacecolor=COL_PGD_FILL,
            markeredgecolor=COL_PGD, markeredgewidth=0.8,
            label=r"\textbf{Targeted white-box PGD}", zorder=5)

    # --- reference lines (labelled through the legend, not inline) ---------
    ax.axhline(tau, color=COL_TAU, ls="--", lw=1.0, zorder=2.5,
               label=rf"$\tau_{{\mathrm{{bad}}}}\!=\!{tau:.3f}$ "
                     rf"$(1.5\!\times\!$clean$)$")
    ax.axhline(clean, color=COL_CLEAN, ls=":", lw=0.7, alpha=0.7, zorder=2.5,
               label=rf"clean $= {clean:.3f}$")

    # --- CROSSING marker on x-axis (vertical line + in-plot label) ---------
    if eps_cross is not None:
        ax.axvline(eps_cross, color=COL_TAU, ls=":", lw=0.8, alpha=0.8,
                   zorder=4)
        # small in-plot label just right of the crossing line, low in the plot
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

    # --- GAP arrow between curves at eps = x_max ----------------------------
    if gap_ratio is not None:
        arrow_x = x_max * 0.96
        arr = FancyArrowPatch(
            (arrow_x, rand_end),
            (arrow_x, pgd_end),
            arrowstyle="<->",
            mutation_scale=7,
            color="#333333", lw=1.0, zorder=6,
        )
        ax.add_patch(arr)
        ax.annotate(
            rf"$\sim\!{gap_ratio:.0f}\times$ more",
            xy=(arrow_x, (rand_end + pgd_end) / 2),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right", va="center",
            fontsize=font_size - 1, color="#333333",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor=PANEL_BG,
                      edgecolor="none", alpha=0.9),
            zorder=7,
        )

    # --- axes ---------------------------------------------------------------
    ax.set_xlabel(r"Perturbation budget $\varepsilon$ "
                  r"($L_\infty$ on BoxCox load)")
    ax.set_ylabel(r"24\,h NRMSE")
    ax.set_xlim(-0.005, x_max + 0.015)
    ax.set_ylim(0, y_top)
    ax.grid(axis="both", alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # no on-figure title/subtitle -- LaTeX caption carries the message.

    # --- legend (2 columns, below axes so it never fights annotations) -----
    # reorder: hero PGD first, noise second, then reference lines
    handles, labels = ax.get_legend_handles_labels()
    order = []
    for target in (r"\textbf{Targeted white-box PGD}",
                   r"Random $L_\infty$ noise (control)"):
        for i, L in enumerate(labels):
            if L == target and i not in order:
                order.append(i); break
    for i in range(len(labels)):
        if i not in order:
            order.append(i)
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    # Centered below axes (no mode="expand" — expand squeezed columns and overlapped text).
    leg = ax.legend(
        handles, labels,
        loc="upper center", bbox_to_anchor=(0.5, -0.24),
        ncol=2, fontsize=font_size - 2,
        framealpha=0.0, handlelength=1.8,
        borderpad=0.2, labelspacing=0.38, columnspacing=1.55,
    )
    leg.set_zorder(20)

    # --- provenance stamp (below the axes, out of the plotting region) -----
    fig.text(
        0.99, 0.005,
        rf"ckpt {pgd.get('ckpt_sha256_8','?')} $\cdot$ "
        rf"git {pgd.get('git_rev','?')} $\cdot$ seed {pgd.get('sampling_seed','?')}",
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
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    with open(args.json) as f:
        pgd_data = json.load(f)
    rand_data = None
    if args.random_json is not None:
        with open(args.random_json) as f:
            rand_data = json.load(f)
    make_figure(pgd_data, rand_data, args.out)


if __name__ == "__main__":
    main()
