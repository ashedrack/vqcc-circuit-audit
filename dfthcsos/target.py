"""The two-body fitting target in weighted packed-pair form.

error_2 = || sum_(rho c) X (x) X - (S (x) I + I (x) S)/2 - tei ||_F over all N^4 entries.
With pairs P = (p >= q) weighted by sqrt(2) when p != q, the N^4 Frobenius norm equals the plain
Frobenius norm of P x P matrices, and X (x) X becomes x x^T with x = pack(X).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data' / 'verifiers'))
from dfthcsos.scoreinfra import reference as REF  # noqa: E402
from dfthcsos import verify as V  # noqa: E402

N = 76
ROWS, COLS = np.tril_indices(N)
PAIR_W = np.where(ROWS == COLS, 1.0, np.sqrt(2.0))


def pack(X: np.ndarray) -> np.ndarray:
    """(..., N, N) symmetric -> (..., P) weighted packed."""
    return X[..., ROWS, COLS] * PAIR_W


@lru_cache(maxsize=1)
def reference():
    ref = REF.load_reference(REF.check_fcidump())
    shifts = REF.load_fixed_shifts()
    fac = {'U': np.zeros((1, 1, N)), 'W': np.zeros((1, 1, 1)), 'H2_I': np.zeros((1, 1)), **shifts}
    _, _, S, beta_1 = V.native_inputs(fac)
    return ref, np.asarray(S, dtype=np.float64), float(beta_1)


@lru_cache(maxsize=1)
def target_matrix() -> np.ndarray:
    """T[P, Q] = tei[p,q,r,s] + (S_pq delta_rs + delta_pq S_rs)/2, weighted packed both sides."""
    ref, S, _ = reference()
    tei = np.asarray(ref['tei'], dtype=np.float64)
    T = tei[ROWS, COLS][:, ROWS, COLS]
    d = (ROWS == COLS).astype(np.float64)
    s = S[ROWS, COLS]
    T = T + 0.5 * (np.outer(s, d) + np.outer(d, s))
    return T * np.outer(PAIR_W, PAIR_W)


if __name__ == '__main__':
    T = target_matrix()
    ev = np.linalg.eigvalsh(T)[::-1]
    print('P =', T.shape[0], ' top eigenvalues', np.round(ev[:5], 3), ' most negative', np.round(ev[-3:], 5))
    pos = np.clip(ev, 0, None)
    neg2 = np.sum(np.minimum(ev, 0) ** 2)
    tail = np.sqrt(np.cumsum((pos[::-1] ** 2))[::-1] + neg2)  # residual if keeping the top L positive
    for eps in (0.05, 0.02, 0.01):
        L = int(np.argmax(np.append(tail, 0) <= eps))
        print(f'leaves needed for error_2 <= {eps * 1e3:.0f} mHa (unconstrained X): L = {L}')
    print('floor from negative eigenvalues:', np.sqrt(neg2))
