#!/usr/bin/env python3
"""
FIGURE 1 (paper Figure 3) - Boundary of the classical reduction and its
consequence for prediction.
=========================================================

Locks two things together in one plot:
  (a) how well the quantum kernel is described by a quadratic form, vs bandwidth
  (b) predictive disagreement between twin and quantum, at the SAME bandwidths
  (c) [optional] which bandwidths cross-validation actually selects

The visual claim: where (a) collapses, (b) rises, and (c) shows that region is
rarely selected anyway.

DATA: s_threshold.csv - contains r2_quad, mse_q, mse_t for every
      (dataset, mode, replicate, target, bandwidth). No binning; both curves
      live on the same 36-point grid.

KULLANIM
  python figure1.py                                   # defaults
  python figure1.py --panel-a residual                # 1-R² log-log
  python figure1.py --with-selection                  # C panelini ekle
  python figure1.py --figsize 7 5 --dpi 600 --out fig1
  python figure1.py --formats pdf png svg
"""
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =====================================================================
# CONFIG — buradan kontrol et
# =====================================================================
CFG = dict(
    csv="s_threshold.csv",
    out="figure1",
    formats=["pdf", "png"],
    figsize=(7.0, 6.2),          # inches; PeerJ single column ~6.8 in
    dpi=400,
    s_star=0.072,                # boundary of the classical regime
    r2_ref=0.99,                 # reference line in panel (a)
    colors={"Tecator": "#1f4e79", "Gasoline": "#c0504d"},
    markers={"Tecator": "o", "Gasoline": "s"},
    band=True,                   # interquartile band in panel (b)
    band_alpha=0.15,
    font_size=9,
    label_size=10,
    line_width=1.6,
    marker_size=3.5,
    grid_alpha=0.25,
    panel_a="r2",                # "r2" | "residual"
    with_selection=False,
    by_target=False,             # split panel (c) by target
    target_styles={             # target -> (line style, marker)
        "Octane": ("-", "s"), "Fat": ("-", "o"),
        "Water": ("--", "^"), "Protein": (":", "D"),
    },
    annotate=True,               # s* etiketini yaz
)


def style(cfg):
    mpl.rcParams.update({
        "font.size": cfg["font_size"],
        "axes.labelsize": cfg["label_size"],
        "axes.titlesize": cfg["label_size"],
        "xtick.labelsize": cfg["font_size"] - 1,
        "ytick.labelsize": cfg["font_size"] - 1,
        "legend.fontsize": cfg["font_size"] - 1,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,      # embedded TrueType - journals require it
        "ps.fonttype": 42,
    })


# =====================================================================
def prepare(cfg):
    d = pd.read_csv(cfg["csv"])
    d["lr"] = np.abs(np.log(d.mse_q / d.mse_t))

    # r2_quad does NOT depend on the target -> drop duplicates so the median
    # is not distorted by targets being counted several times
    r2 = (d.drop_duplicates(["dataset", "mode", "rep", "bw"])
            .groupby(["dataset", "bw"]).r2_quad.median().reset_index())

    lr = (d.groupby(["dataset", "bw"]).lr
            .agg(med="median", lo=lambda x: x.quantile(0.25),
                 hi=lambda x: x.quantile(0.75)).reset_index())

    # distribution of the bandwidth selected by cross-validation
    sel = (d.loc[d.groupby(["dataset", "mode", "rep", "target"]).cv_q.idxmin()]
             .groupby(["dataset", "bw"]).size().rename("n").reset_index())
    tot = sel.groupby("dataset").n.transform("sum")
    sel["frac"] = sel.n / tot

    # per target: each (dataset, target) normalised to sum to one
    selt = (d.loc[d.groupby(["dataset", "mode", "rep", "target"]).cv_q.idxmin()]
              .groupby(["dataset", "target", "bw"]).size().rename("n").reset_index())
    selt["frac"] = selt.n / selt.groupby(["dataset", "target"]).n.transform("sum")
    return r2, lr, sel, selt


def draw(cfg):
    style(cfg)
    r2, lr, sel, selt = prepare(cfg)
    n_panels = 3 if cfg["with_selection"] else 2
    h = cfg["figsize"][1] * (1.0 if n_panels == 2
                             else (1.55 if cfg["by_target"] else 1.35))
    fig, axes = plt.subplots(n_panels, 1, figsize=(cfg["figsize"][0], h),
                             sharex=True)
    axA, axB = axes[0], axes[1]

    # ---------------- Panel A ----------------
    for ds, g in r2.groupby("dataset"):
        g = g.sort_values("bw")
        y = g.r2_quad if cfg["panel_a"] == "r2" else (1 - g.r2_quad).clip(1e-12)
        axA.plot(g.bw, y, marker=cfg["markers"].get(ds, "o"),
                 ms=cfg["marker_size"], lw=cfg["line_width"],
                 color=cfg["colors"].get(ds), label=ds)
    if cfg["panel_a"] == "r2":
        axA.axhline(cfg["r2_ref"], ls=":", lw=1.0, color="0.45")
        axA.set_ylabel(r"$R^2$ of quadratic fit")
        axA.set_ylim(-0.35, 1.05)
        axA.text(1.05e-4, cfg["r2_ref"] + 0.03, r"$R^2=0.99$",
                 fontsize=cfg["font_size"] - 2, color="0.35")
    else:
        axA.set_yscale("log")
        axA.set_ylabel(r"$1-R^2$")
    axA.legend(frameon=False, loc="lower left")
    axA.set_title("(a) Exactness of the classical quadratic reduction",
                  loc="left", fontweight="bold")

    # ---------------- Panel B ----------------
    for ds, g in lr.groupby("dataset"):
        g = g.sort_values("bw")
        axB.plot(g.bw, g.med, marker=cfg["markers"].get(ds, "o"),
                 ms=cfg["marker_size"], lw=cfg["line_width"],
                 color=cfg["colors"].get(ds), label=ds)
        if cfg["band"]:
            axB.fill_between(g.bw, g.lo, g.hi, alpha=cfg["band_alpha"],
                             color=cfg["colors"].get(ds), lw=0)
    axB.set_yscale("log")
    axB.set_ylabel(r"$|\log(\mathrm{MSE}_Q/\mathrm{MSE}_{\mathrm{twin}})|$")
    axB.set_title("(b) Predictive disagreement between quantum kernel and twin",
                  loc="left", fontweight="bold")

    # ---------------- Panel C ----------------
    if cfg["with_selection"]:
        axC = axes[2]
        if cfg["by_target"]:
            # The pooled histogram shows two peaks, but the peak is NOT a
            # single phenomenon: a genuine CV failure in some targets, a flat
            # CV landscape in another. Splitting by target makes it legible.
            for (ds, tg), g in selt.groupby(["dataset", "target"]):
                g = g.sort_values("bw")
                ls, mk = cfg["target_styles"].get(tg, ("-", "o"))
                axC.step(g.bw, g.frac, where="mid", ls=ls,
                         lw=cfg["line_width"], color=cfg["colors"].get(ds),
                         label=f"{ds} · {tg}")
            axC.legend(frameon=False, ncol=2, loc="upper left",
                       fontsize=cfg["font_size"] - 2)
            axC.set_ylabel("fraction of splits\nselected by CV (per target)")
            axC.set_title("(c) Bandwidth chosen by cross-validation, by target",
                          loc="left", fontweight="bold")
        else:
            for ds, g in sel.groupby("dataset"):
                g = g.sort_values("bw")
                axC.step(g.bw, g.frac, where="mid", lw=cfg["line_width"],
                         color=cfg["colors"].get(ds), label=ds)
            axC.set_ylabel("fraction of splits\nselected by CV")
            axC.set_title("(c) Bandwidth chosen by cross-validation",
                          loc="left", fontweight="bold")

    # ---------------- ortak ----------------
    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(cfg["s_star"], ls="--", lw=1.1, color="0.3")
        ax.grid(True, which="major", alpha=cfg["grid_alpha"], lw=0.5)
    if cfg["annotate"]:
        axA.annotate(rf"$s^\ast={cfg['s_star']:g}$",
                     xy=(cfg["s_star"], 0.05),
                     xytext=(cfg["s_star"] * 1.35, 0.05),
                     fontsize=cfg["font_size"] - 1, color="0.25")
    axes[-1].set_xlabel(r"bandwidth $s$")

    for f in cfg["formats"]:
        fig.savefig(f"{cfg['out']}.{f}", dpi=cfg["dpi"])
        print(f"written: {cfg['out']}.{f}")
    return fig


# =====================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=CFG["csv"])
    p.add_argument("--out", default=CFG["out"])
    p.add_argument("--formats", nargs="+", default=CFG["formats"])
    p.add_argument("--figsize", nargs=2, type=float, default=CFG["figsize"])
    p.add_argument("--dpi", type=int, default=CFG["dpi"])
    p.add_argument("--s-star", type=float, default=CFG["s_star"])
    p.add_argument("--panel-a", choices=["r2", "residual"], default=CFG["panel_a"])
    p.add_argument("--with-selection", action="store_true")
    p.add_argument("--by-target", action="store_true",
                   help="split panel (c) by target")
    p.add_argument("--no-band", action="store_true")
    a = p.parse_args()

    cfg = dict(CFG)
    cfg.update(csv=a.csv, out=a.out, formats=a.formats,
               figsize=tuple(a.figsize), dpi=a.dpi, s_star=a.s_star,
               panel_a=a.panel_a, with_selection=a.with_selection,
               by_target=a.by_target,
               band=not a.no_band)
    if cfg["by_target"]:
        cfg["with_selection"] = True
    draw(cfg)


if __name__ == "__main__":
    main()
