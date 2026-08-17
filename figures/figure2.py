#!/usr/bin/env python3
"""FIGURE 2 - Direct visual verification of Proposition 1.

x: s^2 u^T M u   (closed-form prediction, M = I + pi^2 Q, NOTHING fitted)
y: 1 - K(x,x')   (exact state-vector simulation)
Each point is one training pair. Three topologies = three panels.

DESIGN NOTE: at a single small bandwidth the plot is a perfect 45-degree line
and therefore visually uninformative. TWO bandwidths are drawn: one inside the
classical regime (agreement) and one outside it (breakdown), so the figure
shows both the theorem and its boundary.

Usage:
  python figure2.py --formats pdf png --dpi 600
  python figure2.py --bandwidths 0.002 0.2 --n-points 70
"""
import argparse, itertools
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

CFG = dict(out="figure2", formats=["pdf", "png"], figsize=(7.2, 2.9), dpi=400,
           n_qubits=6, n_points=70, bandwidths=[0.002, 0.2], seed=0,
           colors=["#1f4e79", "#c0504d"], point_size=4, alpha=0.35,
           font_size=9, label_size=10)
TOPOS = [("linear", r"path $P_6$"), ("circular", r"cycle $C_6$"),
         ("full", r"complete $K_6$")]


def edges(k, t):
    if t == "linear":   return [(i, i + 1) for i in range(k - 1)]
    if t == "circular": return [(i, (i + 1) % k) for i in range(k)]
    return list(itertools.combinations(range(k), 2))


def M_analytic(k, E):
    A = np.zeros((k, k))
    for i, j in E: A[i, j] = A[j, i] = 1.0
    return np.eye(k) + np.pi ** 2 * (np.diag(A.sum(1)) + A)


def states(X, k, E):
    out = []
    for x in X:
        qc = QuantumCircuit(k); qc.h(range(k))
        for i in range(k): qc.rz(2.0 * x[i], i)
        for i, j in E:
            qc.cx(i, j); qc.rz(2.0 * (np.pi - x[i]) * (np.pi - x[j]), j); qc.cx(i, j)
        out.append(Statevector.from_instruction(qc).data)
    return np.array(out)


def draw(cfg):
    mpl.rcParams.update({"font.size": cfg["font_size"],
                         "axes.labelsize": cfg["label_size"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.constrained_layout.use": True,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    rng = np.random.default_rng(cfg["seed"])
    k, N = cfg["n_qubits"], cfg["n_points"]
    Z = rng.uniform(-1, 1, (N, k))
    iu = np.triu_indices(N, 1)
    fig, axes = plt.subplots(1, 3, figsize=cfg["figsize"], sharex=True, sharey=True)
    for ax, (topo, lab) in zip(axes, TOPOS):
        E = edges(k, topo); M = M_analytic(k, E)
        for c, s in zip(cfg["colors"], cfg["bandwidths"]):
            U = Z * s
            du = U[iu[0]] - U[iu[1]]
            pred = np.einsum("ij,jk,ik->i", du, M, du)      # s^2 u^T M u
            S = states(U, k, E); K = np.abs(S @ S.conj().T) ** 2
            sim = (1.0 - K)[iu]
            ax.scatter(pred, sim, s=cfg["point_size"], alpha=cfg["alpha"],
                       color=c, lw=0, label=rf"$s={s:g}$")
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color="0.35", zorder=0)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(lab, loc="left", fontsize=cfg["font_size"])
        ax.set_xlabel(r"$s^{2}\,\mathbf{u}^{\top}M\,\mathbf{u}$  (Proposition 1)")
    axes[0].set_ylabel(r"$1-K$  (simulated)")
    h, l = axes[0].get_legend_handles_labels()
    leg = axes[0].legend(h, l, frameon=False, loc="upper left", markerscale=3)
    for lh in leg.legend_handles: lh.set_alpha(1)
    for f in cfg["formats"]:
        fig.savefig(f"{cfg['out']}.{f}", dpi=cfg["dpi"]); print("written:", f"{cfg['out']}.{f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=CFG["out"]); p.add_argument("--formats", nargs="+", default=CFG["formats"])
    p.add_argument("--dpi", type=int, default=CFG["dpi"]); p.add_argument("--n-points", type=int, default=CFG["n_points"])
    p.add_argument("--bandwidths", nargs=2, type=float, default=CFG["bandwidths"])
    p.add_argument("--figsize", nargs=2, type=float, default=CFG["figsize"])
    a = p.parse_args(); c = dict(CFG)
    c.update(out=a.out, formats=a.formats, dpi=a.dpi, n_points=a.n_points,
             bandwidths=list(a.bandwidths), figsize=tuple(a.figsize))
    draw(c)
