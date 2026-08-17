#!/usr/bin/env python3
"""
CLASSICAL-REGIME THRESHOLD - FULL-SCALE RUN
======================================

CLAIM: the bandwidth regime in which the quantum kernel does NOT reduce to a
classical quadratic form (large s) is also the regime from which
cross-validation gains nothing predictively. In other words, no bandwidth is
both non-classical and useful.

TEST: select the quantum kernel's bandwidth by CV on (a) the full grid and
(b) a grid truncated at s <= s_cut, then compare PAIRED. If there is no loss,
the upper regime is inert.

DESIGN NOTE: this script records the FULL path at every bandwidth (cv, test
MSE, classicality R2), so s_cut can be swept without re-running. This shows
that the threshold was not chosen from the outcome and that the conclusion is
insensitive to it, which forestalls the obvious referee objection.

Usage:
  python s_threshold_full.py --n-reps 100 --n-jobs 10 --outdir ./sthr
  python s_threshold_full.py --n-reps 100 --n-jobs 10 --outdir ./sthr --resume
  python s_threshold_full.py --outdir ./sthr --report-only
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_HERE, "edf_study_v2.py")).read()
exec(_src[:_src.index("def main()")])                       # noqa: S102
sys.path.insert(0, _HERE)
from t3_twin_robustness import (MODES, M_analytic, edges,    # noqa: E402
                                k_twin, preprocess, quad_r2)

CUTS = [0.02, 0.036, 0.052, 0.072, 0.10, 0.14]   # threshold sweep for the report


# =====================================================================
def run_task(X0, ys, dsname, mode, rep, k, topo, with_r2=True, test_size=0.3):
    Xp = preprocess(X0, mode)
    idx = np.arange(len(Xp))
    tr, te = train_test_split(idx, test_size=test_size, random_state=1000 + rep)

    sx = StandardScaler().fit(Xp[tr])
    Xtr, Xte = sx.transform(Xp[tr]), sx.transform(Xp[te])
    pca = PCA(n_components=k).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    sc = np.abs(Ztr).max(0) + 1e-12
    Zn, Zen = np.clip(Ztr / sc, -1, 1), np.clip(Zte / sc, -1, 1)

    M = M_analytic(k, edges(k, topo))
    fm = QuantumFeatureMap(k, 1)

    Ks, r2 = {}, {}
    for bw in BW_GRID:
        S1, S2 = fm.states(Zn * bw), fm.states(Zen * bw)
        Kq, Kqt = fm.kernel(S1, S1), fm.kernel(S2, S1)
        Ks[bw] = (Kq, Kqt, k_twin(Zn, Zn, M, bw), k_twin(Zen, Zn, M, bw))
        r2[bw] = quad_r2(Kq, Zn) if with_r2 else np.nan

    rows = []
    for tgt, yv in ys.items():
        ytr, yte = yv[tr], yv[te]
        for bw in BW_GRID:
            Kq, Kqt, Kt, Ktt = Ks[bw]
            rec = {"dataset": dsname, "mode": mode, "rep": rep, "target": tgt,
                   "n_qubits": k, "topology": topo, "bw": bw, "r2_quad": r2[bw]}
            for tag, (K, Kte) in (("q", (Kq, Kqt)), ("t", (Kt, Ktt))):
                if is_degenerate(K):
                    rec[f"cv_{tag}"] = np.inf
                    rec[f"mse_{tag}"] = np.nan
                    continue
                cv = cv_scores(K, ytr, LAM_GRID)
                i = int(np.argmin(cv))
                _, mse, _ = eig_path(K, Kte, ytr, yte, LAM_GRID)
                rec[f"cv_{tag}"] = float(cv[i])
                rec[f"mse_{tag}"] = float(mse[i])
            rows.append(rec)
    return rows


def _guard(*a, **kw):
    try:
        return run_task(*a, **kw)
    except Exception as ex:                                  # noqa: BLE001
        print(f"  [warning] task failed: {ex}", flush=True)
        return []


# =====================================================================
def _boot(diff, n_boot=20000, seed=0):
    if len(diff) < 3:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    bt = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(1)
    return float(diff.mean()), (float(np.percentile(bt, 2.5)),
                                float(np.percentile(bt, 97.5)))


def _select(g, cut, tag):
    """Bandwidth selection by CV; cut=None means the full grid."""
    h = g if cut is None else g[g.bw <= cut]
    h = h[np.isfinite(h[f"cv_{tag}"])]
    if not len(h):
        return np.nan, np.nan
    r = h.loc[h[f"cv_{tag}"].idxmin()]
    return float(r[f"mse_{tag}"]), float(r["bw"])


def report(path, cuts=CUTS):
    d = pd.read_csv(path)
    key = ["dataset", "mode", "rep", "target"]

    # --- A) classicality boundary: R2 vs bandwidth, by dataset ---
    print("=" * 100)
    print("(A) CLASSICALITY BOUNDARY - median R2 of the quadratic fit, by bandwidth")
    print("=" * 100)
    piv = d.groupby(["dataset", "bw"]).r2_quad.median().unstack(0)
    show = piv.loc[piv.index[::4]]
    print(show.round(6).to_string())

    # --- B) selection vs truncation comparison ---
    sel = []
    for k_, g in d.groupby(key, sort=False):
        rec = dict(zip(key, k_))
        rec["q_full"], rec["bw_full"] = _select(g, None, "q")
        rec["t_full"], _ = _select(g, None, "t")
        for c in cuts:
            rec[f"q_{c}"], _ = _select(g, c, "q")
        sel.append(rec)
    s = pd.DataFrame(sel).dropna(subset=["q_full"])

    print("\n" + "=" * 100)
    print("(B) COST OF FORBIDDING THE UPPER REGIME - quantum, paired diff (full - truncated)")
    print("     diff > 0 => truncation is BETTER   |   CI covers zero => upper regime inert")
    print("=" * 100)
    hdr = f"{'dataset':<9}{'target':<9}{'mode':<7}{'full':>9}{'upper sel':>10}"
    for c in cuts:
        hdr += f"{('s≤' + str(c)):>10}"
    print(hdr)
    for (ds, tgt, m), g in s.groupby(["dataset", "target", "mode"], sort=False):
        line = f"{ds:<9}{tgt:<9}{m:<7}{g.q_full.mean():9.4f}{(g.bw_full > 0.072).mean():10.2f}"
        for c in cuts:
            line += f"{g[f'q_{c}'].mean():10.4f}"
        print(line)

    print("\n" + "-" * 100)
    print("Threshold sweep: paired difference pooled over all cells (full - truncated)")
    print(f"{'cut':>8}{'mean diff':>12}{'95% CI':>24}{'truncation better':>18}"
          f"{'upper selected':>18}")
    for c in cuts:
        diff = (s.q_full - s[f"q_{c}"]).values
        dm, ci = _boot(diff)
        tag = "" if (ci[0] <= 0 <= ci[1]) else "  *"
        print(f"{c:>8.3f}{dm:>12.4f}  [{ci[0]:+8.4f},{ci[1]:+8.4f}]"
              f"{(diff > 0).mean():>18.3f}{(s.bw_full > c).mean():>18.3f}{tag}")

    # --- C) twin equivalence inside vs outside the classical regime ---
    print("\n" + "=" * 100)
    print("(C) TWIN EQUIVALENCE - inside vs outside the classical regime")
    print("=" * 100)
    s2 = s.dropna(subset=["t_full"]).copy()
    if len(s2):
        s2["lr"] = np.abs(np.log(s2.q_full / s2.t_full))
        s2["regime"] = np.where(s2.bw_full > 0.072,
                               "s > 0.072 (non-classical)", "s <= 0.072 (classical)")
        out_rows = []
        for rj, gg in s2.groupby("regime"):
            out_rows.append({
                "regime": rj, "n": len(gg),
                "median_s": gg.bw_full.median(),
                "median_R2": d[d.bw.isin(gg.bw_full.unique())].r2_quad.median(),
                "median_|log ratio|": gg.lr.median(),
                "Q_win": float((np.log(gg.q_full / gg.t_full) < 0).mean()),
            })
        print(pd.DataFrame(out_rows).set_index("regime").round(4).to_string())
        if s2.regime.nunique() < 2:
            print("  (note: only one regime was observed in this run)")
    else:
        print("  (no twin results)")

    s.to_csv(path.replace(".csv", "_selected.csv"), index=False)
    print(f"\nSelection table written: {path.replace('.csv', '_selected.csv')}")


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--topology", default="linear",
                    choices=["linear", "circular", "full"])
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--datasets", nargs="+", default=["Gasoline", "Tecator"])
    ap.add_argument("--outdir", default="./sthr")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-r2", action="store_true",
                    help="skip the classicality R2 (~30%% faster)")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "s_threshold.csv")

    if a.report_only:
        report(out)
        return

    loaders = {"Tecator": load_tecator, "Gasoline": load_gasoline}
    data = {}
    for ds in a.datasets:
        X, ys = loaders[ds]()
        data[ds] = (X, ys)
        print(f"  {ds}: n={X.shape[0]}, p={X.shape[1]}, targets={list(ys)}")

    done = set()
    if a.resume and os.path.exists(out):
        prev = pd.read_csv(out)
        done = set(zip(prev.dataset, prev["mode"], prev.rep))
        print(f"  resuming: {len(done)} tasks already complete")

    tasks = [(ds, m, r) for ds in a.datasets for m in a.modes
             for r in range(a.n_reps) if (ds, m, r) not in done]
    print(f"Tasks: {len(tasks)}  |  qubits: {a.qubits}  |  "
          f"topology: {a.topology}  |  n_jobs: {a.n_jobs}")

    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(_guard)(data[ds][0], data[ds][1], ds, m, r,
                        a.qubits, a.topology, not a.no_r2)
        for ds, m, r in tasks)
    rows = [x for chunk in res for x in chunk]
    print(f"Done: {(time.time() - t0) / 60:.1f} min")

    df = pd.DataFrame(rows)
    if a.resume and os.path.exists(out) and len(df):
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    elif not len(df) and os.path.exists(out):
        df = pd.read_csv(out)
    df.to_csv(out, index=False)
    print(f"Written: {out}  ({len(df)} rows)")
    report(out)


if __name__ == "__main__":
    main()
