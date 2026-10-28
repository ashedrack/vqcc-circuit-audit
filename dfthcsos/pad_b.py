"""Pad each group's rows to a larger B (e.g. 120 -> 129 to cross the QROAM workspace
boundary). New rows are the strongest unused baseline rows, then unit-noise; W gets one
extra zero column... actually extra rows get zero weight initially via W padded with 0 rows.

Usage: python dfthcsos/pad_b.py IN.npz OUT.npz --B 129 [--refit 300]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from target import N, target_matrix  # noqa: E402
from fit_als import error2, refine_joint  # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / 'data/starter_kit/starter_kit/starter/task1_factorization/FeMoco76_65_60_60.npz'


def pad_rows(U, W, B_new, donor):
    R, B, N = U.shape
    C = W.shape[2]
    extra = B_new - B
    Un = U / np.linalg.norm(U, axis=-1, keepdims=True)
    # candidate donor rows: baseline rows not already close to any row in this group
    seen = set()
    rng = np.random.default_rng(0)
    U2 = np.empty((R, B_new, N))
    U2[:, :B] = Un
    W2 = np.zeros((R, B_new, C))
    W2[:, :B] = W
    Di = 0
    for r in range(R):
        added = 0
        while added < extra and Di < donor.shape[0]:
            cand = donor[Di]
            Di += 1
            if np.abs(Un[r] @ cand).max() < 0.999:  # skip near-duplicates
                U2[r, B + added] = cand
                added += 1
        while added < extra:
            v = rng.standard_normal(N)
            U2[r, B + added] = v / np.linalg.norm(v)
            added += 1
    return U2, W2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('inp', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--B', type=int, default=129)
    ap.add_argument('--refit', type=int, default=300)
    a = ap.parse_args()
    z = np.load(a.inp)
    U, W = np.asarray(z['U'], dtype=np.float64), np.asarray(z['W'], dtype=np.float64)
    b = np.load(BASELINE)
    donor = b['U'].reshape(-1, N)
    donor /= np.linalg.norm(donor, axis=-1, keepdims=True)
    T = jnp.asarray(target_matrix())
    T2 = float(jnp.sum(T * T))
    U2, W2 = pad_rows(U, W, a.B, donor)
    e0 = error2(U2, W2, T, T2)
    print(f'padded -> {U2.shape}: {e0 * 1e3:.2f} mHa (new rows zero-weighted)', flush=True)
    if a.refit:
        e0, U2, W2 = refine_joint(jnp.asarray(U2), jnp.asarray(W2), a.refit, every=50, f32=True)
        U2, W2 = np.asarray(U2), np.asarray(W2)
        print(f'refit -> {e0 * 1e3:.2f} mHa')
    np.savez(a.out, U=U2, W=W2)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
