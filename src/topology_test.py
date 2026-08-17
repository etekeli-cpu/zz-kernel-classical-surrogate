"""GENERAL FORM OF THE INDUCED METRIC:  M = I + pi^2 * Q,
Q = D + A = the SIGNLESS LAPLACIAN of the entanglement graph.
The linear chain is a special case. Path, cycle and complete topologies are tested here.
"""
import numpy as np, itertools
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def edges(k, topo):
    if topo == "linear":
        return [(i, i + 1) for i in range(k - 1)]
    if topo == "circular":
        return [(i, (i + 1) % k) for i in range(k)]
    if topo == "full":
        return list(itertools.combinations(range(k), 2))
    raise ValueError(topo)


def M_analytic(k, E):
    """M = I + pi^2 (D + A)"""
    A = np.zeros((k, k))
    for i, j in E:
        A[i, j] = A[j, i] = 1.0
    D = np.diag(A.sum(1))
    return np.eye(k) + np.pi ** 2 * (D + A)


def states(Z, k, E, reps):
    out = []
    for x in Z:
        qc = QuantumCircuit(k)
        for _ in range(reps):
            qc.h(range(k))
            for i in range(k):
                qc.rz(2.0 * x[i], i)
            for i, j in E:
                qc.cx(i, j)
                qc.rz(2.0 * (np.pi - x[i]) * (np.pi - x[j]), j)
                qc.cx(i, j)
        out.append(Statevector.from_instruction(qc).data)
    return np.array(out)


def fit_M(K, U):
    n, k = len(K), U.shape[1]
    iu = np.triu_indices(n, 1)
    t = (1.0 - K)[iu]
    du = U[iu[0]] - U[iu[1]]
    ii, jj = np.triu_indices(k, 1)
    A = np.hstack([du ** 2, du[:, ii] * du[:, jj]])
    w, *_ = np.linalg.lstsq(A, t, rcond=None)
    r2 = 1 - np.sum((t - A @ w) ** 2) / np.sum((t - t.mean()) ** 2)
    M = np.zeros((k, k))
    M[np.diag_indices(k)] = w[:k]
    M[ii, jj] = M[jj, ii] = w[k:] / 2.0
    return r2, M


rng = np.random.default_rng(0)
K_Q, N, BW = 6, 90, 0.002
Z = rng.uniform(-1, 1, size=(N, K_Q))
U = Z * BW

print("=" * 84)
print(f"PREDICTION:  M = I + pi^2 (D + A)     [n_qubits={K_Q}, s={BW}]")
print("=" * 84)
print(f"{'topology':<11}{'|E|':>4}{'reps':>5}{'R2_quadratic':>15}"
      f"{'max|M_fit - M_exact|':>26}{'rel.':>10}")
store = {}
for topo in ("linear", "circular", "full"):
    E = edges(K_Q, topo)
    Ma = M_analytic(K_Q, E)
    for reps in (1, 2):
        S = states(U, K_Q, E, reps)
        Kk = np.abs(S @ S.conj().T) ** 2
        r2, Mf = fit_M(Kk, U)
        store[(topo, reps)] = (Mf, Ma)
        if reps == 1:
            err = np.abs(Mf - Ma).max()
            print(f"{topo:<11}{len(E):>4}{reps:>5}{r2:>15.6f}"
                  f"{err:>26.3e}{err/np.abs(Ma).max():>10.2e}")
        else:
            print(f"{topo:<11}{len(E):>4}{reps:>5}{r2:>15.6f}"
                  f"{'(L=1 formula invalid)':>26}{'-':>10}")

for topo in ("circular", "full"):
    Mf, Ma = store[(topo, 1)]
    print(f"\n--- {topo}, reps=1 ---")
    print("fitted:"); print(np.round(Mf, 2))
    print("analytic I + pi^2(D+A):"); print(np.round(Ma, 2))
