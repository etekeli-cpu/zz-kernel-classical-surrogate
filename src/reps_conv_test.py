"""Test of the quadratic-form reduction for reps=L and both phi conventions.

Conventions:
  'qiskit' : phi_ij = (pi - x_i)(pi - x_j)   <- Qiskit ZZFeatureMap
  'iqp'    : phi_ij = x_i x_j                <- Havlicek/IQP convention
Predictions:
  qiskit, L=1 -> M tridiagonal: M_dd = 1 + pi^2 deg(d), M_{d,d+1} = pi^2
  iqp,    L=1 -> no ZZ term at order O(s) -> M = I (isotropic)
  any L      -> R2 ~ 1 (the quadratic form persists), the structure of M may change
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def states(Z, nq, reps, conv):
    out = []
    for x in Z:
        qc = QuantumCircuit(nq)
        for _ in range(reps):
            qc.h(range(nq))
            for i in range(nq):
                qc.rz(2.0 * x[i], i)
            for i in range(nq - 1):
                ang = ((np.pi - x[i]) * (np.pi - x[i + 1]) if conv == "qiskit"
                       else x[i] * x[i + 1])
                qc.cx(i, i + 1); qc.rz(2.0 * ang, i + 1); qc.cx(i, i + 1)
        out.append(Statevector.from_instruction(qc).data)
    return np.array(out)


def fit_quadratic(K, Z):
    """Fit 1-K to the full quadratic form (diagonal + cross terms) by ORDINARY
    LEAST SQUARES.
    NOT NNLS: cross-term coefficients may be negative."""
    n, k = len(K), Z.shape[1]
    iu = np.triu_indices(n, 1)
    t = (1.0 - K)[iu]
    dz = Z[iu[0]] - Z[iu[1]]
    ii, jj = np.triu_indices(k, 1)
    A = np.hstack([dz ** 2, dz[:, ii] * dz[:, jj]])
    w, *_ = np.linalg.lstsq(A, t, rcond=None)
    pred = A @ w
    r2 = 1 - np.sum((t - pred) ** 2) / np.sum((t - t.mean()) ** 2)
    M = np.zeros((k, k))
    M[np.diag_indices(k)] = w[:k]
    M[ii, jj] = w[k:] / 2.0; M[jj, ii] = w[k:] / 2.0
    # isotropic submodel
    A1 = (dz ** 2).sum(1, keepdims=True)
    w1, *_ = np.linalg.lstsq(A1, t, rcond=None)
    r2_iso = 1 - np.sum((t - A1 @ w1) ** 2) / np.sum((t - t.mean()) ** 2)
    return r2, r2_iso, M


def M_analytic_L1(k):
    M = np.zeros((k, k))
    for d in range(k):
        M[d, d] = 1.0 + np.pi ** 2 * ((d > 0) + (d < k - 1))
    for d in range(k - 1):
        M[d, d + 1] = M[d + 1, d] = np.pi ** 2
    return M


rng = np.random.default_rng(0)
NQ, N, BW = 6, 90, 0.002
Z = rng.uniform(-1, 1, size=(N, NQ))

print("=" * 78)
print(f"n_qubits={NQ}, N={N}, bandwidth s={BW}")
print("=" * 78)
print(f"{'conv':<8}{'L':>3}{'R2_isotropic':>15}{'R2_quadratic':>15}{'1-R2':>12}"
      f"{'M/s^2 max off-diag':>24}")
res = {}
for conv in ("qiskit", "iqp"):
    for L in (1, 2, 3):
        S = states(Z * BW, NQ, L, conv)
        K = np.abs(S @ S.conj().T) ** 2
        r2, r2iso, M = fit_quadratic(K, Z * BW)
        off = np.abs(M - np.diag(np.diag(M))).max()
        res[(conv, L)] = M
        print(f"{conv:<8}{L:>3}{r2iso:>15.6f}{r2:>15.6f}{1-r2:>12.2e}"
              f"{off/max(np.diag(M).max(),1e-300):>24.4f}")

print("\n--- qiskit, L=1: fitted M / s^2  vs  ANALYTIC M ---")
Mf = res[("qiskit", 1)]
Ma = M_analytic_L1(NQ)
print("fitted:"); print(np.round(Mf, 2))
print("analytic:"); print(np.round(Ma, 2))
print("max relative error = %.3e" % (np.abs(Mf - Ma).max() / np.abs(Ma).max()))

print("\n--- iqp, L=1: fitted M / s^2 (prediction: identity) ---")
print(np.round(res[("iqp", 1)], 4))

for L in (2, 3):
    print(f"\n--- qiskit, L={L}: fitted M / s^2 (does it stay tridiagonal?) ---")
    M = res[("qiskit", L)]; print(np.round(M, 2))
    k = M.shape[0]
    band = [np.mean([abs(M[i, i+d]) for i in range(k-d)]) for d in range(k)]
    print("  band profile mean |M_{i,i+d}|:", " ".join("d=%d:%.2f" % (d, b) for d, b in enumerate(band)))
