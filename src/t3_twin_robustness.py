#!/usr/bin/env python3
"""
T3 - ROBUSTNESS OF THE EXPLICIT CLASSICAL TWIN TO PREPROCESSING
=====================================================

CLAIM: the twin kernel is predictively equivalent to the quantum kernel, and
this equivalence is INDEPENDENT OF PREPROCESSING, because the derivation rests
on the circuit rather than on the data.

Twin:   K = exp(-s^2 * du^T M du),   M = I + pi^2 (D + A)
        D+A = signless Laplacian of the entanglement graph (Proposition 1)
        NO fitted parameters. NO quantum simulation.

Preprocessing pipelines (all on the same splits, paired):
  raw   : raw absorbances + per-wavelength standardisation
  snv   : standard normal variate (scatter correction) + std
  sg1   : Savitzky-Golay first derivative + std
  sg2   : Savitzky-Golay second derivative + std
  sg2s  : SNV -> SG second derivative + std

Usage:
  python t3_twin_robustness.py --n-reps 100 --n-jobs 10 --outdir ./t3
  python t3_twin_robustness.py --n-reps 100 --n-jobs 10 --outdir ./t3 --resume
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- helpers from the main study (loaders, QuantumFeatureMap, cv_scores,
#     eig_path, is_degenerate, LAM_GRID, BW_GRID) ---------------------------
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edf_study_v2.py")
_src = open(_SRC).read()
exec(_src[:_src.index("def main()")])          # noqa: S102

MODES = ["raw", "snv", "sg1", "sg2", "sg2s"]


# =====================================================================
# Preprocessing
# =====================================================================
def snv(X):
    """Per-sample centring and scaling (multiplicative scatter correction)."""
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def preprocess(X, mode, window=11, poly=2):
    if mode == "raw":
        return X
    if mode == "snv":
        return snv(X)
    w = min(window, X.shape[1] - (1 - X.shape[1] % 2))
    if mode == "sg1":
        return savgol_filter(X, w, poly, deriv=1, axis=1)
    if mode == "sg2":
        return savgol_filter(X, w, poly, deriv=2, axis=1)
    if mode == "sg2s":
        return savgol_filter(snv(X), w, poly, deriv=2, axis=1)
    raise ValueError(mode)


# =====================================================================
# Entanglement graph and the ANALYTICAL metric  (Proposition 1)
# =====================================================================
def edges(k, topo):
    if topo == "linear":
        return [(i, i + 1) for i in range(k - 1)]
    if topo == "circular":
        return [(i, (i + 1) % k) for i in range(k)]
    if topo == "full":
        return list(itertools.combinations(range(k), 2))
    raise ValueError(topo)


def M_analytic(k, E):
    """M = I + pi^2 (D + A);  D+A = signless Laplacian."""
    A = np.zeros((k, k))
    for i, j in E:
        A[i, j] = A[j, i] = 1.0
    return np.eye(k) + np.pi ** 2 * (np.diag(A.sum(1)) + A)


def k_twin(P, Qm, M, s):
    """exp(-s^2 * du^T M du); P and Q are in normalised coordinates."""
    D = (np.einsum("ij,jk,ik->i", P, M, P)[:, None]
         + np.einsum("ij,jk,ik->i", Qm, M, Qm)[None, :]
         - 2.0 * P @ M @ Qm.T)
    return np.exp(-(s ** 2) * np.maximum(D, 0.0))


# =====================================================================
# Classicality diagnostic (secondary column): does 1-K fit a full quadratic form?
# =====================================================================
def quad_r2(K, U):
    n, k = len(K), U.shape[1]
    iu = np.triu_indices(n, 1)
    t = (1.0 - K)[iu]
    if t.std() < 1e-300:
        return np.nan
    du = U[iu[0]] - U[iu[1]]
    ii, jj = np.triu_indices(k, 1)
    A = np.hstack([du ** 2, du[:, ii] * du[:, jj]])
    w, *_ = np.linalg.lstsq(A, t, rcond=None)
    return float(1 - np.sum((t - A @ w) ** 2) / np.sum((t - t.mean()) ** 2))


# =====================================================================
# One task = (dataset, mode, replicate)
# =====================================================================
def run_task(X0, ys, dsname, mode, rep, k, topo, test_size=0.3):
    Xp = preprocess(X0, mode)
    idx = np.arange(len(Xp))
    tr, te = train_test_split(idx, test_size=test_size, random_state=1000 + rep)

    sx = StandardScaler().fit(Xp[tr])
    Xtr, Xte = sx.transform(Xp[tr]), sx.transform(Xp[te])
    pca = PCA(n_components=k).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    sc = np.abs(Ztr).max(0) + 1e-12
    Zn, Zen = np.clip(Ztr / sc, -1, 1), np.clip(Zte / sc, -1, 1)

    E = edges(k, topo)
    M = M_analytic(k, E)
    fm = QuantumFeatureMap(k, 1)

    # two kernels at each bandwidth
    Ks = {}
    for bw in BW_GRID:
        S1, S2 = fm.states(Zn * bw), fm.states(Zen * bw)
        Ks[("Kuantum", bw)] = (fm.kernel(S1, S1), fm.kernel(S2, S1))
        Ks[("Twin", bw)] = (k_twin(Zn, Zn, M, bw), k_twin(Zen, Zn, M, bw))

    rows = []
    for tgt, yv in ys.items():
        ytr, yte = yv[tr], yv[te]
        rec = {"dataset": dsname, "mode": mode, "rep": rep, "target": tgt,
               "n_qubits": k, "topology": topo}
        for fam in ("Quantum", "Twin"):
            best = (np.inf, np.nan, np.nan, np.nan)
            for bw in BW_GRID:
                K, Kt = Ks[(fam, bw)]
                if is_degenerate(K):
                    continue
                cv = cv_scores(K, ytr, LAM_GRID)
                i = int(np.argmin(cv))
                if cv[i] < best[0]:
                    edf, mse, _ = eig_path(K, Kt, ytr, yte, LAM_GRID)
                    best = (cv[i], mse[i], edf[i], bw)
            tag = "q" if fam == "Quantum" else "t"
            rec[f"{tag}_mse"] = best[1]
            rec[f"{tag}_edf"] = best[2]
            rec[f"{tag}_bw"] = best[3]
        # classicality diagnostic, at the bandwidth the quantum kernel selected
        if np.isfinite(rec.get("q_bw", np.nan)):
            rec["r2_quad"] = quad_r2(Ks[("Kuantum", rec["q_bw"])][0], Zn)
        rows.append(rec)
    return rows


def _guard(*a, **kw):
    try:
        return run_task(*a, **kw)
    except Exception as ex:                                     # noqa: BLE001
        print(f"  [warning] task failed: {ex}", flush=True)
        return []


# =====================================================================
# Raporlama
# =====================================================================
def split_effect(q, t, n_boot=20000, seed=0):
    d = q - t
    if len(d) < 3:
        return np.nan, (np.nan, np.nan), np.nan, np.nan
    rng = np.random.default_rng(seed)
    bt = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return (float(d.mean()),
            (float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))),
            float((d < 0).mean()),
            float(np.abs(np.log(q / t)).mean()))


def report(path):
    d = pd.read_csv(path)
    d = d.dropna(subset=["q_mse", "t_mse"])
    print("=" * 104)
    print("T3 - EXPLICIT CLASSICAL TWIN vs QUANTUM KERNEL, ACROSS PREPROCESSING")
    print("Claim holds iff the CIs contain zero (equivalence) in ALL pipelines")
    print("=" * 104)
    for ds in d.dataset.unique():
        for tgt in d[d.dataset == ds].target.unique():
            g0 = d[(d.dataset == ds) & (d.target == tgt)]
            print(f"\n### {ds} - {tgt}   ({g0.rep.nunique()} replicates)")
            print(f"{'mode':<7}{'Quantum':>10}{'Twin':>10}{'diff':>10}"
                  f"{'95% CI':>22}{'Q win':>11}{'<|log ratio|>':>14}"
                  f"{'R2_quad':>9}{'bw agree':>10}")
            for m in MODES:
                g = g0[g0["mode"] == m]
                if len(g) < 3:
                    continue
                a = g.sort_values("rep")
                q, t = a.q_mse.values, a.t_mse.values
                dm, ci, wr, lr = split_effect(q, t)
                agree = float((a.q_bw.values == a.t_bw.values).mean())
                flag = "" if (ci[0] <= 0 <= ci[1]) else "  <- CI EXCLUDES zero"
                print(f"{m:<7}{q.mean():10.4f}{t.mean():10.4f}{dm:+10.4f}"
                      f"  [{ci[0]:+8.4f},{ci[1]:+8.4f}]{wr:11.2f}{lr:14.4f}"
                      f"{a.r2_quad.median():9.5f}{agree:10.2f}{flag}")
    n_excl = 0
    tot = 0
    for (ds, tgt, m), g in d.groupby(["dataset", "target", "mode"]):
        if len(g) < 3:
            continue
        a = g.sort_values("rep")
        _, ci, _, _ = split_effect(a.q_mse.values, a.t_mse.values)
        tot += 1
        n_excl += int(not (ci[0] <= 0 <= ci[1]))
    print("\n" + "=" * 104)
    print(f"SUMMARY: the CI excludes zero in {n_excl} of {tot} cells "
          f"(equivalence in {tot - n_excl}).")
    print("If equivalence holds in every pipeline, the claim is preprocessing-independent.")


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
    ap.add_argument("--outdir", default="./t3")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, "t3_twin.csv")

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
                        a.qubits, a.topology)
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
