#!/usr/bin/env python3
"""
PROTEIN ANOMALY CONTROL - FULL SCALE
=========================================

HYPOTHESIS: for the Tecator protein target, the flatness of the cross-validation
objective in the bandwidth, and the frequent selection of the upper
(non-classical) regime, are not properties of the kernel but consequences of a
six-component representation that loses the target signal. As components are
added the CV curve should sharpen and upper-regime selection should fall to zero.

SURROGATE: the ANALYTICAL TWIN is used as the main family. Its equivalence to
the quantum kernel in the classical regime is established, and it costs O(k^2)
per pair; at k=20 circuit simulation (a 2^20-dimensional state vector) is not
practical.

QUANTUM ANCHOR (--with-quantum-anchor): for k <= anchor_max the quantum kernel
is run as well. If the flatness and upper-selection figures of the two families
agree, use of the surrogate is empirically justified.

Usage:
  python pc_control_full.py --n-reps 100 --n-jobs 10 --with-quantum-anchor
  python pc_control_full.py --n-reps 100 --n-jobs 10 --resume
  python pc_control_full.py --report-only --outdir ./pcc
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_HERE, "edf_study_v2.py")).read()
exec(_src[:_src.index("def main()")])                        # noqa: S102
sys.path.insert(0, _HERE)
from t3_twin_robustness import (MODES, M_analytic, edges,     # noqa: E402
                                k_twin, preprocess)

# k=4 gives the downward direction (the anomaly worsens) -> a dose-response trend
# 8->12 is the critical transition: without k=10 it is missed
# by k=20 ridge saturates (it matches the full spectrum)
KS_DEFAULT = [4, 6, 8, 10, 12, 16, 20]
S_STAR = 0.072
ANCHOR_MAX = 8          # run the quantum kernel up to this number of qubits


# =====================================================================
def _curve_stats(cv, mse, bw):
    ok = np.isfinite(cv)
    if ok.sum() < 5:
        return {}
    lo_m = ok & (bw <= S_STAR)
    hi_m = ok & (bw > S_STAR)
    lo = cv[lo_m].min() if lo_m.any() else np.nan
    hi = cv[hi_m].min() if hi_m.any() else np.nan
    j = int(np.argmin(np.where(ok, cv, np.inf)))
    return dict(flatness=float((cv[ok].max() - cv[ok].min()) / cv[ok].min()),
                arm_ratio=float(hi / lo) if np.isfinite(lo) and lo > 0 else np.nan,
                sel_bw=float(bw[j]), upper=int(bw[j] > S_STAR),
                cv_min=float(cv[ok].min()), test_mse=float(mse[j]))


def run_task(X0, ys, mode, rep, ks, anchor_max, test_size=0.3):
    Xp = preprocess(X0, mode)
    idx = np.arange(len(Xp))
    tr, te = train_test_split(idx, test_size=test_size, random_state=1000 + rep)
    sx = StandardScaler().fit(Xp[tr])
    A, B = sx.transform(Xp[tr]), sx.transform(Xp[te])
    pca = PCA(n_components=max(ks)).fit(A)
    Zf, Zef = pca.transform(A), pca.transform(B)
    bw = np.asarray(BW_GRID)

    rows, ridge = [], []

    # --- (1) ridge alone: how many components carry the signal? ---
    for tgt, yv in ys.items():
        ytr, yte = yv[tr], yv[te]
        for k in list(ks) + ["full"]:
            Ztr = A if k == "full" else Zf[:, :k]
            Zte = B if k == "full" else Zef[:, :k]
            m = RidgeCV(alphas=np.logspace(-6, 6, 60)).fit(Ztr, ytr)
            ridge.append(dict(mode=mode, rep=rep, target=tgt, k=k,
                              mse=float(np.mean((m.predict(Zte) - yte) ** 2))))

    # --- (2) kernel families ---
    for k in ks:
        Ztr, Zte = Zf[:, :k], Zef[:, :k]
        sc = np.abs(Ztr).max(0) + 1e-12
        Zn, Zen = np.clip(Ztr / sc, -1, 1), np.clip(Zte / sc, -1, 1)
        M = M_analytic(k, edges(k, "linear"))

        fams = {"twin": {bwv: (k_twin(Zn, Zn, M, bwv), k_twin(Zen, Zn, M, bwv))
                         for bwv in bw}}
        if anchor_max and k <= anchor_max:
            fm = QuantumFeatureMap(k, 1)
            fams["quantum"] = {}
            for bwv in bw:
                S1, S2 = fm.states(Zn * bwv), fm.states(Zen * bwv)
                fams["quantum"][bwv] = (fm.kernel(S1, S1), fm.kernel(S2, S1))

        for fam, Ks in fams.items():
            for tgt, yv in ys.items():
                ytr, yte = yv[tr], yv[te]
                cv = np.full(len(bw), np.inf)
                ms = np.full(len(bw), np.nan)
                for i, bwv in enumerate(bw):
                    K, Kt = Ks[bwv]
                    if is_degenerate(K):
                        continue
                    c = cv_scores(K, ytr, LAM_GRID)
                    j = int(np.argmin(c))
                    _, mm, _ = eig_path(K, Kt, ytr, yte, LAM_GRID)
                    cv[i], ms[i] = c[j], mm[j]
                st = _curve_stats(cv, ms, bw)
                if st:
                    rows.append(dict(mode=mode, rep=rep, k=k, target=tgt,
                                     family=fam, **st))
    return rows, ridge


def _guard(*a, **kw):
    try:
        return run_task(*a, **kw)
    except Exception as ex:                                   # noqa: BLE001
        print(f"  [warning] task failed: {ex}", flush=True)
        return [], []


# =====================================================================
def report(outdir):
    r = pd.read_csv(os.path.join(outdir, "pc_control.csv"))
    rg = pd.read_csv(os.path.join(outdir, "pc_control_ridge.csv"))
    tw = r[r.family == "twin"]
    n = tw.rep.nunique()
    print("=" * 92)
    print(f"PROTEIN ANOMALY - RETAINED-DIMENSION CONTROL ({n} replicates, "
          f"pipelines: {sorted(tw['mode'].unique())})")
    print("=" * 92)

    print("\n(1) Ridge alone, test MSE - how many components carry the signal?")
    print(rg.pivot_table(index="target", columns="k", values="mse",
                         aggfunc="mean").round(4).to_string())

    print("\n(2) CV curve flatness (max-min)/min - larger = sharper [twin]")
    print(tw.pivot_table(index="target", columns="k", values="flatness",
                         aggfunc="median").round(2).to_string())

    print("\n(3) Upper/lower arm CV ratio - near 1 = arms indistinguishable [twin]")
    print(tw.pivot_table(index="target", columns="k", values="arm_ratio",
                         aggfunc="median").round(3).to_string())

    print("\n(4) Upper-regime selection rate (%) [twin]")
    print((100 * tw.pivot_table(index="target", columns="k", values="upper",
                                aggfunc="mean")).round(1).to_string())

    print("\n(5) Twin test MSE (at the cross-validated optimum)")
    print(tw.pivot_table(index="target", columns="k", values="test_mse",
                         aggfunc="mean").round(4).to_string())

    if (r.family == "quantum").any():
        q = r[r.family == "quantum"]
        ks = sorted(q.k.unique())
        print("\n" + "=" * 92)
        print(f"QUANTUM ANCHOR - validity of the twin surrogate (k <= {max(ks)})")
        print("=" * 92)
        print(f"{'target':<10}{'k':>4}{'flat Q':>11}{'flat T':>11}"
              f"{'arm Q':>9}{'arm T':>9}{'up% Q':>9}{'up% T':>9}")
        for tgt in sorted(q.target.unique()):
            for k in ks:
                a = q[(q.target == tgt) & (q.k == k)]
                b = tw[(tw.target == tgt) & (tw.k == k)]
                if not len(a) or not len(b):
                    continue
                print(f"{tgt:<10}{k:>4}{a.flatness.median():>11.2f}"
                      f"{b.flatness.median():>11.2f}{a.arm_ratio.median():>9.3f}"
                      f"{b.arm_ratio.median():>9.3f}{100*a.upper.mean():>9.1f}"
                      f"{100*b.upper.mean():>9.1f}")
        print("\nIf the figures agree, using the twin as a surrogate for k > anchor")
        print("is empirically justified.")


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--ks", nargs="+", type=int, default=KS_DEFAULT)
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--with-quantum-anchor", action="store_true")
    ap.add_argument("--anchor-max", type=int, default=ANCHOR_MAX)
    ap.add_argument("--outdir", default="./pcc")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    f_main = os.path.join(a.outdir, "pc_control.csv")
    f_ridge = os.path.join(a.outdir, "pc_control_ridge.csv")
    if a.report_only:
        report(a.outdir)
        return

    X0, ys = load_tecator()
    print(f"  Tecator: n={X0.shape[0]}, p={X0.shape[1]}, targets={list(ys)}")
    anchor = a.anchor_max if a.with_quantum_anchor else 0

    done = set()
    if a.resume and os.path.exists(f_main):
        prev = pd.read_csv(f_main)
        done = set(zip(prev["mode"], prev.rep))
        print(f"  resuming: {len(done)} tasks already complete")

    tasks = [(m, r) for m in a.modes for r in range(a.n_reps)
             if (m, r) not in done]
    print(f"Tasks: {len(tasks)}  |  k: {a.ks}  |  "
          f"quantum anchor: {'k<=' + str(anchor) if anchor else 'off'}  |  "
          f"n_jobs: {a.n_jobs}")

    t0 = time.time()
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(
        delayed(_guard)(X0, ys, m, r, a.ks, anchor) for m, r in tasks)
    rows = [x for c, _ in res for x in c]
    ridge = [x for _, c in res for x in c]
    print(f"Done: {(time.time() - t0) / 60:.1f} min")

    for path, new in ((f_main, rows), (f_ridge, ridge)):
        df = pd.DataFrame(new)
        if a.resume and os.path.exists(path) and len(df):
            df = pd.concat([pd.read_csv(path), df], ignore_index=True)
        elif not len(df) and os.path.exists(path):
            df = pd.read_csv(path)
        df.to_csv(path, index=False)
        print(f"Written: {path} ({len(df)} rows)")
    report(a.outdir)


if __name__ == "__main__":
    main()
