#!/usr/bin/env python3
"""
EFFECTIVE-DEGREES-OF-FREEDOM STUDY - FULL RUN (SINGLE FILE)
===============================================
Scalable version of the original study. Additions:

  1. CHECKPOINTING : results are appended to CSV as they complete, so work
     survives a sleeping laptop, a crashed worker or an interrupted run.
  2. RESUME        : completed (replicate, qubits, pipeline) tasks are skipped
     on restart; --resume continues where the run stopped.
  3. QUBIT SWEEP   : --qubits 4 6 8, to check whether the result is specific
     to six qubits.
  4. SUMMARISED OUTPUT: instead of the raw profile (millions of rows), results
     are aggregated into edf bins inside the worker, which reduces the output
     from about 3.5 million rows to roughly 50 thousand.

KULLANIM (macOS)
    python edf_study.py --n-reps 200 --n-jobs 10 --qubits 4 6 8 > run200.log

    caffeinate -i nohup python edf_study.py \
        --n-reps 200 --n-jobs 10 --qubits 4 6 8 > run200.log 2>&1 &

    # kesilirse:
    caffeinate -i nohup python edf_study.py \
        --n-reps 200 --n-jobs 10 --qubits 4 6 8 --resume >> run200.log 2>&1 &
"""

from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import urllib.request
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --- definitions inlined so that this file is self-contained ---

TECATOR_URL = ("https://raw.githubusercontent.com/moviedo5/fda.usc/"
               "master/data/tecator.rda")
DATA_DIR = Path("./data")
GASOLINE_URL = ("https://raw.githubusercontent.com/bhmevik/pls/"
                "master/data/gasoline.RData")
TARGETS = ["Fat", "Water", "Protein"]          # kept for backward compatibility


def load_tecator():
    """Download Tecator once and return (X, y_dict)."""
    import rdata

    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "tecator.rda"
    if not path.exists():
        print(f"  downloading Tecator: {TECATOR_URL}")
        path.write_bytes(urllib.request.urlopen(TECATOR_URL, timeout=60).read())

    obj = rdata.read_rda(str(path))["tecator"]
    X = np.asarray(obj["absorp.fdata"]["data"], dtype=float)
    y = {t: np.asarray(obj["y"][t], dtype=float) for t in TARGETS}
    return X, y


def load_gasoline():
    """
    Gasoline NIR (R package `pls`): n=60, p=401, target octane number.
    Complementary regime to Tecator (n=215 > p=100): here n=60 << p=401.
    The R object cannot be converted directly by rdata because it contains a
    matrix, so it is parsed at the raw level.
    """
    import rdata

    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "gasoline.RData"
    if not path.exists():
        print(f"  downloading Gasoline: {GASOLINE_URL}")
        path.write_bytes(urllib.request.urlopen(GASOLINE_URL, timeout=60).read())

    o = rdata.parser.parse_file(str(path)).object
    y = np.asarray(o.value[0].value[0].value, dtype=float)
    X = np.asarray(o.value[0].value[1].value, dtype=float).reshape(401, 60).T
    return X, {"Octane": y}


DATASETS = {"Tecator": load_tecator, "Gasoline": load_gasoline}


class QuantumFeatureMap:
    """ZZ-type feature map; K(x,x') = |<phi(x)|phi(x')>|^2."""

    def __init__(self, n_qubits=6, reps=1):
        self.n_qubits = n_qubits
        self.reps = reps

    def _circuit(self, x):
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(self.n_qubits)
        for _ in range(self.reps):
            qc.h(range(self.n_qubits))
            for i in range(self.n_qubits):
                qc.rz(2.0 * x[i], i)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(2.0 * (np.pi - x[i]) * (np.pi - x[i + 1]), i + 1)
                qc.cx(i, i + 1)
        return qc

    def states(self, Z):
        from qiskit.quantum_info import Statevector

        return np.array([Statevector.from_instruction(self._circuit(z)).data
                         for z in Z])

    @staticmethod
    def kernel(S_a, S_b):
        return np.abs(S_a @ S_b.conj().T) ** 2


def _sqdist(A, B):
    return np.maximum(
        np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2.0 * A @ B.T,
        0.0)


def k_rbf(A, B, ls):
    return np.exp(-_sqdist(A, B) / (2.0 * ls ** 2))


def k_matern(A, B, ls, nu):
    r = np.sqrt(_sqdist(A, B))
    if nu == 0.5:
        return np.exp(-r / ls)
    if nu == 1.5:
        a = np.sqrt(3.0) * r / ls
        return (1.0 + a) * np.exp(-a)
    a = np.sqrt(5.0) * r / ls
    return (1.0 + a + a ** 2 / 3.0) * np.exp(-a)


LAM_GRID = np.logspace(-10, 2, 40)
EDF_BINS = np.array([2, 4, 8, 16, 32, 64, 128, 256])

# --- FAIRNESS: EXACTLY N_SET settings in every competing family. Unequal grid
#     sizes made the oracle and CV selection overfitting incomparable across families.
N_SET = 36
BW_GRID = np.logspace(-4.0, 0.0, N_SET)   # quantum: 1e-4 .. 1.0 (brackets the interior minimum)
BW_PQK = BW_GRID[::6]                     # PQK: 6 bw x 6 gamma = 36
ARD_A = np.linspace(0.0, 1.0, 6)

# Reference baselines: context, NOT competitors. Their candidate sets are
# deliberately small and they are excluded from any equal-count comparison.
REF_FAMS = ["Linear (PC)", "Linear (full)", "Polynomial (PC)",
            "PLS (full)"]
PLS_MAX = 60   # chemometric standard: n_components selected by CV (must be interior)
FAMS = ["Quantum", "Twin", "PQK", "ARD-RBF", "RBF (isotropic)",
        "Matérn 5/2"] + REF_FAMS


PREP_MODES = ["raw", "snv", "sg1", "sg2", "sg2s"]


def _snv(X):
    """Per-sample centring and scaling (multiplicative scatter correction)."""
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def preprocess(X, mode, window=11, poly=2):
    """Spectral preprocessing. IMPORTANT: this is a hyperparameter, not a fixed
    choice. The optimal pipeline varies with the target AND with the kernel
    family, so fixing it would bias the comparison between families."""
    from scipy.signal import savgol_filter
    if mode == "raw":
        return X
    if mode == "snv":
        return _snv(X)
    w = min(window, X.shape[1] - (1 - X.shape[1] % 2))
    if mode == "sg1":
        return savgol_filter(X, w, poly, deriv=1, axis=1)
    if mode == "sg2":
        return savgol_filter(X, w, poly, deriv=2, axis=1)
    if mode == "sg2s":
        return savgol_filter(_snv(X), w, poly, deriv=2, axis=1)
    raise ValueError(mode)


def pls_baseline(Xtr, Xte, ytr, yte, n_folds=5, seed=0):
    """The de facto standard in chemometrics: PLS regression on the FULL SPECTRUM,
    the number of latent variables is selected by 5-fold CV.

    SPEED-UP: PLS scores are nested and orthogonal, so a SINGLE fit with kmax
    components and using only the first k scores gives the same projection as
    a k-component PLS: one fit per fold instead of kmax."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    kmax = int(min(PLS_MAX, Xtr.shape[1], len(ytr) - 2))
    if kmax < 1:
        return np.nan, np.nan, np.nan

    def _path(Xa, ya, Xb, yb, kk):
        """MSE on Xb for the first k = 1..kk components."""
        m = PLSRegression(n_components=kk, scale=False).fit(Xa, ya)
        Ta, Tb = m.x_scores_, m.transform(Xb)
        ym = ya.mean()
        out = np.full(kk, np.inf)
        for k in range(1, kk + 1):
            A = Ta[:, :k]
            beta, *_ = np.linalg.lstsq(A, ya - ym, rcond=None)
            out[k - 1] = np.mean((Tb[:, :k] @ beta + ym - yb) ** 2)
        return out

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    acc = []
    for a, b in kf.split(Xtr):
        kk = int(min(kmax, len(a) - 2))
        try:
            pth = _path(Xtr[a], ytr[a], Xtr[b], ytr[b], kk)
        except Exception:
            continue
        acc.append(np.pad(pth, (0, kmax - kk), constant_values=np.inf))
    if not acc:
        return np.nan, np.nan, np.nan
    cv = np.mean(acc, axis=0)
    kbest = int(np.argmin(cv)) + 1
    m = PLSRegression(n_components=kbest, scale=False).fit(Xtr, ytr)
    mse = float(np.mean((m.predict(Xte).ravel() - yte) ** 2))
    return mse, kbest, float(cv[kbest - 1])


def build_grids(Ztr):
    """Anchor the length-scale grids to the data. With a fixed logspace the
    classical families select the grid boundary in every split."""
    d = _sqdist(Ztr, Ztr)
    med = np.sqrt(np.median(d[d > 0]))
    iso_ls = med * np.logspace(-1.5, 1.5, N_SET)
    ard_c = med * np.logspace(-1.5, 1.5, N_SET // len(ARD_A))
    return iso_ls, ard_c


def pqk_gammas(d_tr):
    """gamma is divided by the Bloch scale of EACH bandwidth separately.
    Calibrating from a single bw would misscale gamma elsewhere, and a fixed
    gamma would confound the 4/6/8-qubit comparison."""
    m = np.median(d_tr[d_tr > 0]) + 1e-300
    return np.logspace(-1.5, 1.5, 6) / m


def entangle_edges(k, topo="linear"):
    """Edge set of the entanglement graph."""
    import itertools
    if topo == "linear":
        return [(i, i + 1) for i in range(k - 1)]
    if topo == "circular":
        return [(i, (i + 1) % k) for i in range(k)]
    if topo == "full":
        return list(itertools.combinations(range(k), 2))
    raise ValueError(topo)


def M_analytic(k, E):
    """Proposition 1: M = I + pi^2 (D + A);  D+A = signless Laplacian.
    NO fitted parameters; read directly off the entanglement graph."""
    A = np.zeros((k, k))
    for i, j in E:
        A[i, j] = A[j, i] = 1.0
    return np.eye(k) + np.pi ** 2 * (np.diag(A.sum(1)) + A)


def k_twin(P, Q, M, s):
    """EXPLICIT CLASSICAL TWIN:  K = exp(-s^2 * du^T M du).
    Requires no quantum simulation; costs O(k^2) per pair."""
    D = (np.einsum("ij,jk,ik->i", P, M, P)[:, None]
         + np.einsum("ij,jk,ik->i", Q, M, Q)[None, :]
         - 2.0 * P @ M @ Q.T)
    return np.exp(-(s ** 2) * np.maximum(D, 0.0))


def _safe_nanmean(v):
    """np.nanmean warns on an all-NaN array. The result would be NaN anyway,
    so we check first and avoid the warning."""
    if v is None:
        return np.nan
    a = np.asarray(v, dtype=float)
    return float(np.nanmean(a)) if a.size and np.isfinite(a).any() else np.nan


def quantum_classicality(K_q, Ztr, sigma_d=None, ard_setting=None):
    """
    CLASSICAL DECOMPOSITION OF THE QUANTUM KERNEL - nested model ladder.

    Since K_ii = 1, the kernel distance is d2_K(i,j) = 2(1 - K_ij).
    We fit it to quadratic forms in z-space:

      M1 (isotropic) : d2_K ~ a * sum_d (z_id - z_jd)^2       [1 parameter]
      M2 (ARD/diagonal): d2_K ~ sum_d w_d (z_id - z_jd)^2       [k parameters]
      M3 (+interaction): M2 + sum_{d<e} v_de dz_d dz_e          [k+k(k-1)/2]

    YORUM:
      R2(M2) ~ 1               -> the kernel is a CLASSICAL ARD kernel
      R2(M2) low, R2(M3) ~ 1   -> the structure lies in the entangling interactions
      R2(M3) low too           -> genuinely non-classical / saturated regime

    The length-scale profile recovered from M2, ls_d = 1/sqrt(2 w_d), is also
    compared with the profile ls_d = c * sigma_d^a that ARD-RBF selects by CV.
    Spearman is used because saturation is a monotone distortion: it lowers
    Pearson but preserves rank.
    """
    from scipy.optimize import nnls
    from scipy.stats import spearmanr

    n, k = len(K_q), Ztr.shape[1]
    iu = np.triu_indices(n, 1)
    t = (2.0 * (1.0 - K_q))[iu]
    if t.std() < 1e-300 or not np.all(np.isfinite(t)):
        return {}

    dz = Ztr[iu[0]] - Ztr[iu[1]]              # (m, k) pairwise differences
    Dg = dz ** 2                              # diagonal terms
    out = {}

    def _fit(A):
        w, _ = nnls(A, t)
        pred = A @ w
        ss = np.sum((t - t.mean()) ** 2)
        return w, pred, float(1.0 - np.sum((t - pred) ** 2) / max(ss, 1e-300))

    _, pr1, r2_iso = _fit(Dg.sum(1, keepdims=True))
    w2, pr2, r2_ard = _fit(Dg)
    ii, jj = np.triu_indices(k, 1)
    A3 = np.hstack([Dg, dz[:, ii] * dz[:, jj]])
    _, pr3, r2_int = _fit(A3)

    out["r2_iso"] = r2_iso
    out["r2_ard"] = r2_ard
    out["r2_int"] = r2_int
    out["gain_anis"] = r2_ard - r2_iso        # share explained by anisotropy
    out["gain_ent"] = r2_int - r2_ard         # share from interactions/entanglement
    out["rho_ard"] = float(spearmanr(pr2, t).statistic)

    # recovered length-scale profile
    ls_q = np.where(w2 > 0, 1.0 / np.sqrt(2.0 * np.maximum(w2, 1e-300)), np.nan)
    ok = np.isfinite(ls_q)
    out["ls_q_span"] = (float(np.nanmax(ls_q[ok]) / np.nanmin(ls_q[ok]))
                        if ok.sum() >= 2 else np.nan)

    # alignment with the profile ARD-RBF selected
    out["ard_align"] = np.nan
    if sigma_d is not None and ard_setting and ok.sum() >= 3:
        try:
            c = float(ard_setting.split("c=")[1].split(",")[0])
            a = float(ard_setting.split("a=")[1])
            ls_ard = c * (sigma_d ** a)
            u, v = np.log(ls_q[ok]), np.log(ls_ard[ok])
            # When ARD selects a=0, ls_ard is constant (ls = c*sigma^0 = c)
            # and Spearman is undefined; return NaN instead of warning.
            if u.std() > 1e-12 and v.std() > 1e-12:
                out["ard_align"] = float(spearmanr(u, v).statistic)
        except Exception:
            pass
    return out

SEL_COLS = ["dataset", "rep", "n_qubits", "target", "family", "prep",
            "cv_mse", "test_mse",
            "edf", "setting", "lambda", "mse_orc", "edf_orc", "sharp",
            "spec_decay", "decay_idx", "decay_exp", "n_settings", "med_dist",
            "r2_iso", "r2_ard", "r2_int", "gain_anis", "gain_ent",
            "rho_ard", "ls_q_span", "ard_align"]
PROF_COLS = ["dataset", "rep", "n_qubits", "target", "family", "edf_bin",
             "test_mse_mean", "n"]
SET_COLS = ["dataset", "rep", "n_qubits", "target", "family", "setting", "decay_exp",
            "decay_ratio_50", "sharp", "mse_min", "edf_at_min"]


def k_ard_spectral(A, B, sigma_d, c, a):
    ls = c * (sigma_d ** a)
    return np.exp(-_sqdist(A / ls, B / ls) / 2.0)


def _sqd_np(A, B):
    return np.maximum(np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :]
                      - 2.0 * A @ B.T, 0.0)


def bloch_features(states, n_qubits):
    """
    Feature extraction for the projected quantum kernel (Huang et al. 2021).
    Bloch components of the reduced density matrix of each qubit k:
        <X> = 2·Re(ρ01),  <Y> = −2·Im(ρ01),  <Z> = ρ00 − ρ11
    Returns a 3k-dimensional CLASSICAL feature vector per sample.
    """
    m = len(states)
    feats = np.empty((m, 3 * n_qubits))
    psi = states.reshape((m,) + (2,) * n_qubits)
    for k in range(n_qubits):
        ax = n_qubits - k          # qiskit: qubit 0 is the least significant bit
        A = np.moveaxis(psi, ax, 1).reshape(m, 2, -1)
        r01 = np.einsum("ij,ij->i", A[:, 0, :], A[:, 1, :].conj())
        r00 = np.einsum("ij,ij->i", A[:, 0, :], A[:, 0, :].conj()).real
        r11 = np.einsum("ij,ij->i", A[:, 1, :], A[:, 1, :].conj()).real
        feats[:, 3*k]     = 2.0 * r01.real
        feats[:, 3*k + 1] = -2.0 * r01.imag
        feats[:, 3*k + 2] = r00 - r11
    return feats


def _eigh(A):
    """
    Robust symmetric eigendecomposition.

    At many qubits and small bandwidth the kernel degenerates to K -> J
    (all ones): every state coincides, the matrix drops to rank one and the
    default LAPACK 'evd' driver fails to converge. Graceful fallback:
    evd -> ev (slower but robust) -> diagonal jitter -> SVD (always converges).
    """
    A = (A + A.T) / 2.0
    try:
        return np.linalg.eigh(A)
    except np.linalg.LinAlgError:
        pass
    try:
        from scipy.linalg import eigh as _sp_eigh
        return _sp_eigh(A, driver="ev", check_finite=False)
    except Exception:
        pass
    n = len(A)
    scale = max(np.trace(A) / n, 1e-300)
    for j in (1e-12, 1e-10, 1e-8):
        try:
            return np.linalg.eigh(A + j * scale * np.eye(n))
        except np.linalg.LinAlgError:
            continue
    U, sv, _ = np.linalg.svd(A)          # for symmetric PSD, eigenvalues = singular values
    return sv[::-1], U[:, ::-1]


def is_degenerate(K, tol=1e-12):
    """If K ~ J or K ~ I the kernel is dead and not worth computing."""
    n = len(K)
    if n < 3:
        return True
    off = K[~np.eye(n, dtype=bool)]
    return bool(off.std() < tol or not np.all(np.isfinite(K)))


def eig_path(K_tr, K_te, y_tr, y_te, lams):
    w, Q = _eigh(K_tr)
    w = np.maximum(w, 0.0)
    ybar = y_tr.mean()
    z = Q.T @ (y_tr - ybar)
    KQ = K_te @ Q
    edf = np.array([np.sum(w / (w + l)) for l in lams])
    mse = np.array([np.mean((KQ @ (z / (w + l)) + ybar - y_te) ** 2)
                    for l in lams])
    return edf, mse, w


def cv_scores(K, y, lams, n_folds=5, seed=0):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    tot = np.zeros(len(lams))
    for tr, va in kf.split(K):
        Ka, Kb = K[np.ix_(tr, tr)], K[np.ix_(va, tr)]
        w, Q = _eigh(Ka)
        w = np.maximum(w, 0.0)
        ybar = y[tr].mean()
        z = Q.T @ (y[tr] - ybar)
        KQ = Kb @ Q
        for i, l in enumerate(lams):
            tot[i] += np.mean((KQ @ (z / (w + l)) + ybar - y[va]) ** 2)
    return tot / n_folds


def decay_exponent(w, k=50):
    """
    Spectral decay exponent b:  log w_i ~ a - b*log i   (i = 1..k)
    Directly related to the capacity conditions of kernel learning theory and
    far more stable than single-point ratios such as w1/w50.
    """
    w = np.sort(np.maximum(w, 0.0))[::-1]
    k = min(k, len(w))
    w = w[:k]
    ok = w > w[0] * 1e-14          # drop the tail that falls to machine zero
    if ok.sum() < 5:
        return np.nan
    li = np.log(np.arange(1, k + 1)[ok])
    lw = np.log(w[ok])
    b, a = np.polyfit(li, lw, 1)
    return float(-b)


def sharpness(edf, mse):
    """How much log-MSE rises when edf is shifted by a factor of two around the optimum."""
    o = np.argsort(edf)
    e, m = np.log(edf[o]), np.log(np.maximum(mse[o], 1e-300))
    i = int(np.argmin(m))
    out = []
    for factor in (0.5, 2.0):
        t = e[i] + np.log(factor)
        j = int(np.argmin(np.abs(e - t)))
        if abs(e[j] - t) < np.log(1.6):
            out.append(m[j] - m[i])
    return float(np.mean(out)) if out else np.nan


def run_task(X, ys, rep, n_qubits, dataset="Tecator", mode="raw",
             test_size=0.3):
    idx = np.arange(len(X))
    tr_i, te_i = train_test_split(idx, test_size=test_size,
                                  random_state=1000 + rep)
    # Preprocessing is INDEPENDENT of the split (per-sample or along the
    # wavelength axis), so there is no leakage; standardisation is fitted on train only.
    X = preprocess(X, mode)
    sx = StandardScaler().fit(X[tr_i])
    Xtr_s, Xte_s = sx.transform(X[tr_i]), sx.transform(X[te_i])
    pca = PCA(n_components=n_qubits).fit(Xtr_s)
    Ztr, Zte = pca.transform(Xtr_s), pca.transform(Xte_s)
    s = np.abs(Ztr).max(0) + 1e-12
    Ztr_n, Zte_n = np.clip(Ztr / s, -1, 1), np.clip(Zte / s, -1, 1)
    sigma_d = Ztr.std(0) + 1e-12

    fm = QuantumFeatureMap(n_qubits, 1)
    M_tw = M_analytic(n_qubits, entangle_edges(n_qubits, "linear"))
    kernels = []
    q_kmap = {}          # setting -> K_tr (for the classicality diagnostic)

    iso_ls, ard_c = build_grids(Ztr)
    _dz = _sqdist(Ztr, Ztr)
    med_dist = float(np.sqrt(np.median(_dz[_dz > 0])))

    for bw in BW_GRID:
        S_tr, S_te = fm.states(Ztr_n * bw), fm.states(Zte_n * bw)
        Kq = fm.kernel(S_tr, S_tr)
        st = f"bw={bw:.4g}"
        q_kmap[st] = Kq
        kernels.append(("Quantum", st, Kq, fm.kernel(S_te, S_tr)))
        # Twin: the SAME grid as the quantum kernel -> equal candidate counts
        kernels.append(("Twin", st,
                        k_twin(Ztr_n, Ztr_n, M_tw, bw),
                        k_twin(Zte_n, Ztr_n, M_tw, bw)))

        if bw in BW_PQK:
            B_tr, B_te = bloch_features(S_tr, n_qubits), bloch_features(S_te, n_qubits)
            d_tr, d_te = _sqd_np(B_tr, B_tr), _sqd_np(B_te, B_tr)
            for g in pqk_gammas(d_tr):
                kernels.append(("PQK", f"bw={bw:.4g},g={g:.3g}",
                                np.exp(-g * d_tr), np.exp(-g * d_te)))

    for ls in iso_ls:
        kernels.append(("RBF (isotropic)", f"ls={ls:.3g}",
                        k_rbf(Ztr, Ztr, ls), k_rbf(Zte, Ztr, ls)))
        kernels.append(("Matérn 5/2", f"ls={ls:.3g}",
                        k_matern(Ztr, Ztr, ls, 2.5),
                        k_matern(Zte, Ztr, ls, 2.5)))
    for c in ard_c:
        for a in ARD_A:
            kernels.append(("ARD-RBF", f"c={c:.3g},a={a:g}",
                            k_ard_spectral(Ztr, Ztr, sigma_d, c, a),
                            k_ard_spectral(Zte, Ztr, sigma_d, c, a)))

    # --- REFERENCE BASELINES ---
    kernels.append(("Linear (PC)", "linear", Ztr @ Ztr.T, Zte @ Ztr.T))
    kernels.append(("Linear (full)", "linear_full",
                    Xtr_s @ Xtr_s.T, Xte_s @ Xtr_s.T))
    gpc = 1.0 / Ztr.shape[1]
    for deg in (2, 3):
        kernels.append(("Polynomial (PC)", f"deg={deg}",
                        (1 + gpc * Ztr @ Ztr.T) ** deg,
                        (1 + gpc * Zte @ Ztr.T) ** deg))

    n_settings = {}
    for f, _, _, _ in kernels:
        n_settings[f] = n_settings.get(f, 0) + 1

    sel_rows, prof_rows, set_rows = [], [], []
    for tgt in ys:
        y = ys[tgt]
        y_tr, y_te = y[tr_i], y[te_i]
        agg = {}          # (family, bin) -> [toplam mse, adet]
        best = {}         # family -> CV-selected
        orc = {}          # family -> oracle
        sharps = {}

        for fam, setting, K, K_te in kernels:
            if is_degenerate(K):
                continue          # K -> J or K -> I: dead kernel, skip
            edf, mse, w = eig_path(K, K_te, y_tr, y_te, LAM_GRID)

            b = np.digitize(edf, EDF_BINS) - 1
            for bi, mi in zip(b, mse):
                if 0 <= bi < len(EDF_BINS) - 1:
                    k = (fam, bi)
                    a_ = agg.get(k, [0.0, 0])
                    agg[k] = [a_[0] + mi, a_[1] + 1]

            sh = sharpness(edf, mse)
            b_exp = decay_exponent(w)
            sharps.setdefault(fam, []).append(sh)

            # SETTING-LEVEL RECORD - makes within-family decay variation visible
            jm = int(np.argmin(mse))
            set_rows.append({
                "dataset": dataset, "rep": rep, "n_qubits": n_qubits, "target": tgt,
                "prep": mode,
                "family": fam, "setting": setting,
                "decay_exp": b_exp,
                "decay_ratio_50": float(np.sort(np.maximum(w, 1e-300))[::-1][0]
                                        / np.sort(np.maximum(w, 1e-300))[::-1][min(49, len(w)-1)]),
                "sharp": sh,
                "mse_min": float(mse[jm]), "edf_at_min": float(edf[jm]),
            })

            j = int(np.argmin(mse))
            if fam not in orc or mse[j] < orc[fam][0]:
                orc[fam] = (float(mse[j]), float(edf[j]))

            cv = cv_scores(K, y_tr, LAM_GRID)
            i = int(np.argmin(cv))
            if fam not in best or cv[i] < best[fam][0]:
                ws = np.sort(np.maximum(w, 1e-300))[::-1]
                # The decay index adapts to the spectrum length: on Gasoline the
                # training set has 42 samples, so the 50th eigenvalue does NOT
                # exist and a fixed index divided by machine zero.
                k_dec = int(min(50, max(5, len(ws) // 3)))
                decay = float(ws[0] / max(ws[k_dec - 1], 1e-300))
                # A scale-free measure that is COMPARABLE across datasets: the
                # exponent b from the fit log w_i ~ a - b*log i.
                kk = int(min(50, len(ws)))
                wk = ws[:kk]
                ok = wk > wk[0] * 1e-14
                decay_b = (float(-np.polyfit(np.log(np.arange(1, kk + 1)[ok]),
                                             np.log(wk[ok]), 1)[0])
                           if ok.sum() >= 5 else np.nan)
                best[fam] = (float(cv[i]), float(mse[i]), float(edf[i]),
                             setting, float(LAM_GRID[i]), decay, decay_b,
                             k_dec)

        # --- PLS (tam spektrum), cekirdek disi kemometri zemini ---
        try:
            pls_mse, pls_k, pls_cv = pls_baseline(Xtr_s, Xte_s, y_tr, y_te)
            best["PLS (full)"] = (pls_cv, pls_mse, float(pls_k),
                                          f"ncomp={pls_k}", np.nan,
                                          np.nan, np.nan, np.nan)
            n_settings["PLS (full)"] = PLS_MAX
        except Exception as ex:
            print(f"  [warning] PLS failed: {ex}", flush=True)

        # The anisotropy profile ARD-RBF selected by CV, used as a reference
        ard_set = best["ARD-RBF"][3] if "ARD-RBF" in best else None
        for fam in best:
            cvv, mse_v, edf_v, setting, lam, decay, decay_b, k_dec = best[fam]
            diag = (quantum_classicality(q_kmap[setting], Ztr, sigma_d, ard_set)
                    if fam == "Quantum" else {})
            _o = orc.get(fam, (np.nan, np.nan))
            sel_rows.append({
                "dataset": dataset, "rep": rep, "n_qubits": n_qubits,
                "target": tgt, "prep": mode, "family": fam,
                "cv_mse": cvv, "test_mse": mse_v,
                "edf": edf_v, "setting": setting, "lambda": lam,
                "mse_orc": _o[0], "edf_orc": _o[1],
                "sharp": _safe_nanmean(sharps.get(fam)),
                "spec_decay": decay,
                "decay_idx": k_dec,
                "decay_exp": decay_b,
                "n_settings": n_settings.get(fam, np.nan),
                "med_dist": med_dist,
                **{c: diag.get(c, np.nan) for c in
                   ("r2_iso", "r2_ard", "r2_int", "gain_anis", "gain_ent",
                    "rho_ard", "ls_q_span", "ard_align")},
            })
        for (fam, bi), (tot, n) in agg.items():
            prof_rows.append({
                "dataset": dataset, "rep": rep, "n_qubits": n_qubits,
                "target": tgt, "prep": mode, "family": fam,
                "edf_bin": f"({EDF_BINS[bi]}, {EDF_BINS[bi + 1]}]",
                "test_mse_mean": tot / n, "n": n,
            })
    return sel_rows, prof_rows, set_rows


# ============================================================================

def _guarded(X, ys, rep, n_qubits, dataset, mode="raw"):
    """Wrapper around run_task so that one failing task does not kill the run."""
    try:
        return run_task(X, ys, rep, n_qubits, dataset, mode)
    except Exception as e:
        print(f"  [skipped] {dataset} rep={rep} q={n_qubits} prep={mode} "
              f"({type(e).__name__}: {str(e)[:60]})", flush=True)
        return [], [], []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reps", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--qubits", type=int, nargs="+", default=[6])
    ap.add_argument("--datasets", nargs="+", default=["Tecator", "Gasoline"],
                    choices=["Tecator", "Gasoline"])
    ap.add_argument("--outdir", type=str, default="./outputs")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--prep", nargs="+", default=PREP_MODES,
                    choices=PREP_MODES,
                    help="preprocessing pipelines; CV selects over their UNION")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sel_path = outdir / "edf_full_selected.csv"
    prof_path = outdir / "edf_full_profile.csv"
    set_path = outdir / "edf_full_settings.csv"

    tasks = [(ds, r, q, m) for ds in args.datasets
             for q in args.qubits for r in range(args.n_reps)
             for m in args.prep]

    done = set()
    if args.resume and sel_path.exists():
        d = pd.read_csv(sel_path, usecols=["dataset", "rep", "n_qubits", "prep"])
        done = {(a, int(b), int(c), m)
                for a, b, c, m in d.drop_duplicates().values}
        tasks = [t for t in tasks if t not in done]
        print(f"Resuming: {len(done)} tasks already complete, "
              f"{len(tasks)} remaining.")
    else:
        for p in (sel_path, prof_path, set_path):
            if p.exists():
                p.unlink()

    # datasets are loaded once and passed to the workers
    data = {}
    for ds in args.datasets:
        X, ys = DATASETS[ds]()
        data[ds] = (X, ys)
        print(f"  {ds}: n={X.shape[0]}, p={X.shape[1]}, "
              f"targets={list(ys)}", flush=True)
    print(f"Tasks: {len(tasks)}  |  qubits: {args.qubits}  |  "
          f"preprocessing: {args.prep}  |  n_jobs: {args.n_jobs}", flush=True)

    t0 = time.time()
    n_done = 0
    sel_buf, prof_buf, set_buf = [], [], []
    sel_hdr = not sel_path.exists()
    prof_hdr = not prof_path.exists()
    set_hdr = not set_path.exists()

    gen = Parallel(n_jobs=args.n_jobs, backend="loky",
                   return_as="generator")(
        delayed(_guarded)(data[ds][0], data[ds][1], r, q, ds, m)
        for ds, r, q, m in tasks)

    for sel_rows, prof_rows, set_rows in gen:
        sel_buf.extend(sel_rows)
        prof_buf.extend(prof_rows)
        set_buf.extend(set_rows)
        n_done += 1

        if len(sel_buf) >= 120 or n_done == len(tasks):
            pd.DataFrame(sel_buf).reindex(columns=SEL_COLS).to_csv(
                sel_path, mode="a", header=sel_hdr, index=False)
            pd.DataFrame(prof_buf).reindex(columns=PROF_COLS).to_csv(
                prof_path, mode="a", header=prof_hdr, index=False)
            pd.DataFrame(set_buf).reindex(columns=SET_COLS).to_csv(
                set_path, mode="a", header=set_hdr, index=False)
            sel_hdr = prof_hdr = set_hdr = False
            sel_buf, prof_buf, set_buf = [], [], []

        if n_done % 10 == 0 or n_done == len(tasks):
            el = time.time() - t0
            eta = (len(tasks) - n_done) / (n_done / el)
            print(f"  [{n_done:4d}/{len(tasks)}] {el/60:6.1f} dk | "
                  f"ETA {eta/60:6.1f} dk", flush=True)

    print(f"\nBitti: {(time.time() - t0)/60:.1f} dk\n")
    report(sel_path, prof_path, args.qubits)


def paired_on_rep(s, fam_a, fam_b):
    """Pairing BY POSITION causes a SILENT ERROR: if a family drops out of one
    replicate, zip aligns the wrong replicates. Merging on rep is mandatory."""
    a = s[s.family == fam_a][["rep", "test_mse"]]
    b = s[s.family == fam_b][["rep", "test_mse"]]
    m = a.merge(b, on="rep", suffixes=("_a", "_b"))
    return m["test_mse_a"].values, m["test_mse_b"].values


def split_effect(qv, v, n_boot=5000, seed=0):
    """Replicates are splits of the SAME dataset, not independent draws.
    A p-value has no sampling interpretation; we report the split-level
    difference distribution instead."""
    rng = np.random.default_rng(seed)
    d = qv - v
    if len(d) < 3:
        return np.nan, (np.nan, np.nan), np.nan
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bt = d[idx].mean(1)
    return (float(d.mean()),
            (float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))),
            float((d < 0).mean()))


def collapse_over_prep(sel):
    """TREAT PREPROCESSING AS A HYPERPARAMETER.

    CV scores are comparable across pipelines: the response, the fold
    assignment and the split are identical and only X changes. Selecting over
    the pooled (pipeline x setting) set is therefore IDENTICAL to selecting
    within each pipeline and then taking the minimum. This does the latter.

    The candidate count per family becomes n_settings x |pipelines| and stays
    equal across all competing families.
    """
    key = ["dataset", "rep", "n_qubits", "target", "family"]
    if "prep" not in sel.columns:
        return sel
    out = sel.loc[sel.groupby(key, sort=False)["cv_mse"].idxmin()].copy()
    n_modes = sel.groupby(key, sort=False)["prep"].nunique().max()
    if "n_settings" in out.columns:
        out["n_settings"] = out["n_settings"] * n_modes
    return out.reset_index(drop=True)


def report(sel_path, prof_path, qubits):
    sel = collapse_over_prep(pd.read_csv(sel_path))
    prof = pd.read_csv(prof_path)
    # NOTE: regret is NOT comparable across families (the oracle minimises over
    # the test set and is sensitive to candidate count). Read it within-family only.
    sel["sel_regret"] = np.log(sel["test_mse"] / sel["mse_orc"])

    for ds in sel.dataset.unique():
      for q in sorted(sel[sel.dataset == ds].n_qubits.unique()):
        for tgt in sel[sel.dataset == ds].target.unique():
            s = sel[(sel.dataset == ds) & (sel.n_qubits == q)
                    & (sel.target == tgt)]
            p = prof[(prof.dataset == ds) & (prof.n_qubits == q)
                     & (prof.target == tgt)]
            if len(s) == 0:
                continue
            print("=" * 100)
            print(f"{ds}   qubits={q}   TARGET: {tgt}   ({s.rep.nunique()} replicates)")
            print("=" * 100)

            comp = [f for f in FAMS if f not in REF_FAMS]
            piv = (p.groupby(["edf_bin", "family"])["test_mse_mean"]
                   .mean().unstack())
            order = [f"({EDF_BINS[i]}, {EDF_BINS[i+1]}]"
                     for i in range(len(EDF_BINS) - 1)]
            print("  (A) mean test MSE at matched edf")
            print(f"  {'edf':>12} " + "".join(f"{f:>17}" for f in comp))
            for b in order:
                if b not in piv.index:
                    continue
                r = piv.loc[b]
                print(f"  {b:>12} " + "".join(
                    f"{r.get(f, np.nan):17.4f}" for f in comp))

            print("\n  (B) at the cross-validated optimum  [bullet = reference, not competitor]")
            print(f"  {'Family':<24}{'sets':>5} {'test MSE':>10} {'SE':>8} "
                  f"{'edf':>7} {'diff':>9} {'95% CI':>19} {'Q win':>10}"
                  f" {'preprocessing':>16}")
            av = [f for f in FAMS if len(s[s.family == f])]
            for f in sorted(av, key=lambda k: s[s.family == k]["test_mse"].mean()):
                g = s[s.family == f]
                v = g["test_mse"].values
                se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
                ns = int(g["n_settings"].iloc[0]) if g["n_settings"].notna().any() else 0
                tag = "  ·" if f in REF_FAMS else "   "
                if f == "Quantum":
                    pm = (g["prep"].mode().iat[0] + f" ({(g['prep']==g['prep'].mode().iat[0]).mean():.0%})"
                          if "prep" in g else "—")
                    print(f"{tag}{f:<24}{ns:>5} {v.mean():10.4f} {se:8.4f} "
                          f"{g['edf'].median():7.1f} {'—':>9} {'—':>19} {'—':>10}"
                          f" {pm:>16}")
                else:
                    qv, vv = paired_on_rep(s, "Quantum", f)
                    dm, ci, wr = split_effect(qv, vv)
                    pm = (g["prep"].mode().iat[0] + f" ({(g['prep']==g['prep'].mode().iat[0]).mean():.0%})"
                          if "prep" in g else "—")
                    print(f"{tag}{f:<24}{ns:>5} {v.mean():10.4f} {se:8.4f} "
                          f"{g['edf'].median():7.1f} {dm:+9.4f} "
                          f"[{ci[0]:+8.4f},{ci[1]:+8.4f}] {wr:10.2f}"
                          f" {pm:>16}")

            qd = s[s.family == "Quantum"]
            if qd["r2_ard"].notna().any():
                def _m(c):
                    return qd[c].median()
                print("\n  (C) CLASSICALITY LADDER - quantum kernel at the CV-selected setting")
                print(f"      M1 isotropic    R2 = {_m('r2_iso'):.4f}")
                print(f"      M2 ARD/diagonal R2 = {_m('r2_ard'):.4f}"
                      f"   (anisotropy gain +{_m('gain_anis'):.4f})")
                print(f"      M3 +interaction R2 = {_m('r2_int'):.4f}"
                      f"   (interaction gain +{_m('gain_ent'):.4f})")
                print(f"      Spearman rho(M2 fit, kernel distance) = "
                      f"{_m('rho_ard'):.4f}")
                print(f"      recovered ls profile span = "
                      f"{_m('ls_q_span'):.1f}×")
                print(f"      ARD-RBF profiliyle hizalanma ρ = "
                      f"{_m('ard_align'):+.4f}")
            print()


if __name__ == "__main__":
    main()
