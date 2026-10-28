"""Our dfthcsos pipeline: factorization -> angle words -> ANGLE_k circuits -> submission directory.

Scoring uses the official verifier modules (data/verifiers/dfthcsos) as the specification; the encoder
and the circuit compiler here are our own.

Usage:
    python dfthcsos/pipeline.py FACTORIZATION.npz OUT_DIR --brot 12 --batch 8 [--tree heap] [--verify]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import kickmix as km

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data' / 'verifiers'))
from dfthcsos import verify as V  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as DM  # noqa: E402
from dfthcsos.scoreinfra.circuits import SELECTION_REGISTER, word_register  # noqa: E402

N = 76


# Givens trees (rooted at 0, parent before child).

def heap_tree(n: int = N) -> np.ndarray:
    return np.array([((i - 1) // 2, i) for i in range(1, n)], dtype=np.int64)


def path_tree(n: int = N) -> np.ndarray:
    return np.array([(i - 1, i) for i in range(1, n)], dtype=np.int64)


TREES = {'heap': heap_tree, 'path': path_tree}


# Encoding.

def vectors_to_angles(vectors: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """theta/pi in [0, 2) per edge such that decoding e_0 on `edges` gives each (unit) row."""
    v = np.array(vectors, dtype=np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    theta = np.zeros((v.shape[0], len(edges)))
    for t in reversed(range(len(edges))):
        p, c = edges[t]
        theta[:, t] = np.arctan2(v[:, c], v[:, p])
        v[:, p] = np.hypot(v[:, p], v[:, c])
        v[:, c] = 0.0
    return np.mod(theta / np.pi, 2.0)


def angles_to_words(theta_over_pi: np.ndarray, brot: int) -> np.ndarray:
    return np.floor(np.mod(theta_over_pi, 2.0) * (1 << (brot - 1)) + 0.5).astype(np.int64) % (1 << brot)


def words_to_vectors(words: np.ndarray, edges: np.ndarray, brot: int) -> np.ndarray:
    return DM.reconstruct_u_vectors(words / (1 << (brot - 1)), edges)


def one_body_vectors(inputs: V.SubmissionInputs, two_body_vectors: np.ndarray) -> np.ndarray:
    """Eigenvectors (rows) of h1_prime built from the decoded two-body vectors, as the verifier does."""
    R, B, _ = inputs.w.shape
    X = DM.reconstruct_x(two_body_vectors.reshape(R, B, N), inputs.w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    oei, tei = inputs.ref['oei'], inputs.ref['tei']
    h1_thc = oei - 0.5 * np.einsum('prrq->pq', tei)
    h1p = DM.build_h1_prime(h1_thc, inputs.S, inputs.beta_1, inputs.ref['electrons'], inputs.W, X)
    _, vecs = np.linalg.eigh(h1p)
    return vecs.T


def encode(inputs: V.SubmissionInputs, edges: np.ndarray, brot: int,
           words2: np.ndarray | None = None) -> np.ndarray:
    """(N + R*B, N - 1) angle words: one-body rows first, then the two-body rows in (rho, b) order.
    If `words2` is given (e.g. from progressive quantization), it is used as-is for the
    two-body rows instead of re-encoding U (re-encoding decoded vectors can flip bits)."""
    R, B, _ = inputs.w.shape
    U = np.asarray(inputs.params['U'], dtype=np.float64).reshape(R * B, N)
    if words2 is None:
        words2 = angles_to_words(vectors_to_angles(U, edges), brot)
    u2 = words_to_vectors(words2, edges, brot)
    w1 = angles_to_words(vectors_to_angles(one_body_vectors(inputs, u2), edges), brot)
    return np.concatenate([w1, words2])


# Circuits: XOR oracle per batch, a plain segment-tree unary-iteration QROM.

def batch_deltas(words: np.ndarray, J: int) -> np.ndarray:
    A, n = words.shape
    K = math.ceil(n / J)
    padded = np.zeros((A, K * J), dtype=np.int64)
    padded[:, :n] = words
    batches = padded.reshape(A, K, J).transpose(1, 0, 2)
    deltas = batches.copy()
    deltas[1:] ^= batches[:-1]
    return deltas


def qrom_circuit(delta: np.ndarray, brot: int) -> km.Circuit:
    A, J = delta.shape
    nbits = (A - 1).bit_length()
    b = km.CircuitBuilder()
    sel = b.create_quantum_register(nbits, SELECTION_REGISTER)
    words = [b.create_quantum_register(brot, word_register(j)) for j in range(J)]
    targets = [words[j][bit] for j in range(J) for bit in range(brot)]
    flat = [[(int(delta[a, j]) >> bit) & 1 for j in range(J) for bit in range(brot)] for a in range(A)]
    with b.unary_iteration(sel, range(A)) as it:
        for a, match in it:
            qs = [t for t, v in zip(targets, flat[a]) if v]
            if qs:
                b.cx(km.array([match] * len(qs)), km.array(qs))
    return b.finish_circuit()


def build_family(words: np.ndarray, brot: int, J: int) -> list[km.Circuit]:
    return [qrom_circuit(d, brot) for d in batch_deltas(words, J)]


def write_submission(out: Path, factorization: dict, words: np.ndarray, edges: np.ndarray,
                     circuits: list[km.Circuit]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('angle_k_*'):
        old.unlink()
    for k, c in enumerate(circuits):
        c.to_file(str(out / f'angle_k_{k}.kmb'), format='kmb')
    np.savez(out / 'submission.npz', **{k: np.asarray(factorization[k]) for k in ('U', 'W', 'H2_I', 'H1_I', 'Hsym_lower')},
             tree_edges=edges, angle_words=words)


def load_factorization(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.array(z[k]) for k in ('U', 'W', 'H2_I', 'H1_I', 'Hsym_lower')}


def max_batch(A: int, brot: int, cap: int = 128) -> int:
    """Largest J such that sel + J*brot words + (nbits-1) ancilla fit in `cap` qubits."""
    nbits = (A - 1).bit_length()
    return max(1, (cap - 2 * nbits) // brot)


def predicted_cost(inputs: V.SubmissionInputs, metrics: dict, words: np.ndarray, brot: int, J: int) -> dict:
    """Score assuming plain-QROM circuits (A - 1 Toffolis per circuit, nbits - 1 ancillas)."""
    R, B, C = inputs.w.shape
    A = words.shape[0]
    K = math.ceil((N - 1) / J)
    h = V.heuristic_costs(N, R, B, C, brot)
    nbits = (A - 1).bit_length()
    T = 2 * K * (A - 2) + h['toffoli']
    Q = h['persistent_qubits'] + J * brot + max(h['temporary_qubits'], nbits - 1)
    return dict(T=T, Q=Q, score=metrics['lambda_eff'] * T * Q, angle_qubits=nbits + J * brot + nbits - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('factorization', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--brot', type=int, default=8)
    ap.add_argument('--batch', type=int, default=0, help='words per circuit; 0 = max under the 128-qubit cap')
    ap.add_argument('--tree', choices=sorted(TREES), default='heap')
    ap.add_argument('--verify', action='store_true')
    a = ap.parse_args()
    fac = load_factorization(a.factorization)
    a.batch = a.batch or max_batch(76 + fac['U'].shape[0] * fac['U'].shape[1], a.brot)
    inputs = V.load_inputs(fac)
    edges = TREES[a.tree]()
    with np.load(a.factorization, allow_pickle=False) as _z:
        w2 = np.asarray(_z['angle_words_2body']) if 'angle_words_2body' in _z.files else None
        if 'tree_edges' in _z.files:
            edges = np.asarray(_z['tree_edges'])  # the words were built for this tree
    words = encode(inputs, edges, a.brot, words2=w2)
    metrics = V.evaluate_words(inputs, words, edges, a.brot)
    print('metrics', {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()})
    print('accuracy ok:', V.passes_accuracy(metrics), '| predicted', predicted_cost(inputs, metrics, words, a.brot, a.batch))
    circuits = build_family(words, a.brot, a.batch)
    write_submission(a.out, fac, words, edges, circuits)
    print(f'wrote {len(circuits)} circuits to {a.out}')
    if a.verify:
        res = V.verify(str(a.out))
        print(json.dumps({k: res[k] for k in ('status', 'score', 'lambda_eff', 'toffoli_count', 'num_qubits')}))


if __name__ == '__main__':
    main()
