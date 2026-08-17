#!/usr/bin/env python3
"""FIGURE 3 (paper Figure 1) - Structure and spectrum of the induced metric.

(a) heat maps of M = I + pi^2 (D+A): path, cycle, complete (shared colour scale)
(b) sorted eigenvalues lambda_i = 1 + pi^2 mu_i

The visual counterpart of Corollaries 2 and 3:
  - path and even cycles are BIPARTITE -> mu_1 = 0 -> lambda_1 = 1 (Euclidean direction)
  - complete: Q = (k-2)I + J -> only TWO distinct eigenvalues

Contains no data; it makes the content of the theorem visible.

Usage:  python figure3.py --formats pdf png --dpi 600 --k 6
"""
import argparse, itertools
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np

CFG = dict(out="figure3", formats=["pdf", "png"], figsize=(7.2, 4.6), dpi=400,
           k=6, cmap="viridis", colors=["#1f4e79", "#c0504d", "#4f7942"],
           font_size=9, label_size=10, annotate=True)
TOPOS = [("linear", r"path $P_k$"), ("circular", r"cycle $C_k$"),
         ("full", r"complete $K_k$")]


def edges(k, t):
    if t == "linear":   return [(i, i + 1) for i in range(k - 1)]
    if t == "circular": return [(i, (i + 1) % k) for i in range(k)]
    return list(itertools.combinations(range(k), 2))


def M_analytic(k, E):
    A = np.zeros((k, k))
    for i, j in E: A[i, j] = A[j, i] = 1.0
    return np.eye(k) + np.pi ** 2 * (np.diag(A.sum(1)) + A)


def draw(cfg):
    mpl.rcParams.update({"font.size": cfg["font_size"],
                         "axes.labelsize": cfg["label_size"],
                         "figure.constrained_layout.use": True,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    k = cfg["k"]
    Ms = [(lab, M_analytic(k, edges(k, t))) for t, lab in TOPOS]
    vmax = max(M.max() for _, M in Ms)
    fig = plt.figure(figsize=cfg["figsize"])
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0])

    # --- (a) heat maps ---
    for c, (lab, M) in enumerate(Ms):
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(M, cmap=cfg["cmap"], vmin=0, vmax=vmax)
        ax.set_title(lab.replace("_k", f"_{k}"), loc="left",
                     fontsize=cfg["font_size"])
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels(range(1, k + 1)); ax.set_yticklabels(range(1, k + 1))
        ax.tick_params(length=0, labelsize=cfg["font_size"] - 2)
        if cfg["annotate"] and k <= 8:
            for i in range(k):
                for j in range(k):
                    if M[i, j] > 0:
                        ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                                fontsize=cfg["font_size"] - 3.5,
                                color="w" if M[i, j] < 0.6 * vmax else "k")
        if c == 0: ax.set_ylabel("component")
    fig.colorbar(im, ax=fig.axes[:3], shrink=0.85, pad=0.015,
                 label=r"$M_{ij}=\left(I+\pi^{2}Q\right)_{ij}$")

    # --- (b) eigenvalues ---
    ax = fig.add_subplot(gs[1, :])
    w = 0.26
    for n, ((lab, M), col) in enumerate(zip(Ms, cfg["colors"])):
        ev = np.sort(np.linalg.eigvalsh(M))
        ax.bar(np.arange(k) + (n - 1) * w, ev, width=w, color=col,
               label=lab.replace("_k", f"_{k}"))
    ax.axhline(1.0, ls=":", lw=1.0, color="0.4")
    ax.text(-0.42, 1.0, r"$\lambda=1$", fontsize=cfg["font_size"] - 2,
            va="bottom", color="0.35")
    ax.set_xticks(range(k)); ax.set_xticklabels([rf"$\lambda_{{{i+1}}}$" for i in range(k)])
    ax.set_ylabel(r"$\lambda_i = 1+\pi^{2}\mu_i$")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("(b) spectrum of the induced metric", loc="left",
                 fontweight="bold", fontsize=cfg["font_size"])
    fig.axes[0].text(-0.18, 1.28, "(a) structure of the induced metric",
                     transform=fig.axes[0].transAxes, fontweight="bold",
                     fontsize=cfg["font_size"])
    for f in cfg["formats"]:
        fig.savefig(f"{cfg['out']}.{f}", dpi=cfg["dpi"]); print("written:", f"{cfg['out']}.{f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=CFG["out"]); p.add_argument("--formats", nargs="+", default=CFG["formats"])
    p.add_argument("--dpi", type=int, default=CFG["dpi"]); p.add_argument("--k", type=int, default=CFG["k"])
    p.add_argument("--figsize", nargs=2, type=float, default=CFG["figsize"])
    p.add_argument("--cmap", default=CFG["cmap"])
    a = p.parse_args(); c = dict(CFG)
    c.update(out=a.out, formats=a.formats, dpi=a.dpi, k=a.k,
             figsize=tuple(a.figsize), cmap=a.cmap)
    draw(c)
