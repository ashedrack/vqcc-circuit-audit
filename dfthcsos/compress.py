"""Group-compression chain for the DF-THC factorization (our own implementation).

  baseline (65,60,60) -> drop weakest -> (64,60,60) -> exact pair-merge -> (32,120,120)
  -> copy-truncate -> (32,120,60) -> drop+refit loop -> (R*,120,60)

Each drop removes the group with the smallest contribution norm ||M_r G_r M_r^T||_F
(= sqrt(tr((G K)^2)), K the in-group squared-inner-product Gram), then refits (U, W)
jointly with L-BFGS. Checkpoints are written after every accepted drop.

Usage:
  python dfthcsos/compress.py --target-r 12 --e-stop 0.013 --refit 400 --out runs/chain
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from target import target_matrix  # noqa: E402
from fit_als import gram_blocks, project_psd, error2, refine_joint, unit  # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / 'data/starter_kit/starter_kit/starter/task1_factorization/FeMoco76_65_60_60.npz'


def affinity_pairs(G):
    """Greedy max-inner-product pairing of group leaf Grams (R, C, C) -> list of pairs."""
    g = G.reshape(G.shape[0], -1)
    n = np.linalg.norm(g, axis=1)
    free = set(range(G.shape[0]))
    pairs = []
    while len(free) >= 2:
        best, pair = -1.0, None
        fs = sorted(free)
        for i in fs:
            for j in fs:
                if j <= i:
                    continue
                a = float(np.dot(g[i], g[j]) / (n[i] * n[j] + 1e-30))
                if a > best:
                    best, pair = a, (i, j)
        pairs.append(pair)
        free -= set(pair)
    return pairs, sorted(free)


def leaf_gram(U, W):
    """(R, C, C) leaf Gram X X^T with X[r,c] the packed leaf."""
    Un = unit(jnp.asarray(U))
    D = jnp.einsum('rbp,rqp->rbq', Un, Un)
    K = D * D
    Gm = jnp.einsum('rbc,rbd->rcd', jnp.asarray(W), jnp.asarray(W))
    return np.asarray(jnp.einsum('rbc,rbd->rcd', Gm, K))


def merge_exact(U, W, pairs, leftover):
    """Merge paired groups: rows concat, W block-diagonal -> exact same operator."""
    Rb, Bb, N = U.shape
    Cb = W.shape[2]
    U2, W2 = [], []
    for i, j in pairs:
        w = np.zeros((2 * Bb, 2 * Cb))
        w[:Bb, :Cb] = W[i]
        w[Bb:, Cb:] = W[j]
        U2.append(np.concatenate([U[i], U[j]]))
        W2.append(w)
    for r in leftover:  # unpaired group: pad rows with copies of its weakest rows, cols with zeros
        w = np.zeros((2 * Bb, 2 * Cb))
        w[:Bb, :Cb] = W[r]
        U2.append(np.concatenate([U[r], U[r][::-1]]))
        W2.append(w)
    return np.stack(U2), np.stack(W2)


def ctrunc(U, W, C):
    """Copy-truncate every group to C copies (top eigenvectors of G_r = W_r W_r^T)."""
    G = np.einsum('rbc,rdc->rbd', W, W)
    return project_psd(G, C)


def contrib(U, W):
    """Per-group contribution norm ||M_r G_r M_r^T||_F = sqrt(tr((G K)^2))."""
    Un = unit(jnp.asarray(U))
    D = jnp.einsum('rbp,rqp->rbq', Un, Un)
    K = np.asarray(D * D)
    G = np.einsum('rbc,rdc->rbd', W, W)
    GK = np.einsum('rbd,rdb->rbd', G, K)
    return np.sqrt(np.einsum('rbd,rdb->r', GK, GK))


def save(path, U, W):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, U=np.asarray(U), W=np.asarray(W))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', type=Path, default=BASELINE)
    ap.add_argument('--init', type=Path, help='resume from a checkpoint npz (U, W)')
    ap.add_argument('--target-r', type=int, default=12)
    ap.add_argument('--e-stop', type=float, default=0.013, help='stop dropping above this error_2 (Ha)')
    ap.add_argument('--refit', type=int, default=400, help='L-BFGS steps after each structural change')
    ap.add_argument('--out', type=Path, default=Path('runs/chain'))
    ap.add_argument('--no-ctrunc', action='store_true', help='keep merged C=2Cb')
    a = ap.parse_args()

    T = jnp.asarray(target_matrix())
    T2 = float(jnp.sum(T * T))
    err = lambda U, W: error2(U, W, T, T2)  # noqa: E731

    if a.init:
        z = np.load(a.init)
        U, W = np.asarray(z['U']), np.asarray(z['W'])
    else:
        z = np.load(a.baseline)
        U, W = np.asarray(z['U'], dtype=np.float64), np.asarray(z['W'], dtype=np.float64)
        imp = np.linalg.norm(W, axis=(1, 2))
        d = int(np.argmin(imp))  # make the count even
        U, W = np.delete(U, d, 0), np.delete(W, d, 0)
        print(f'dropped weakest baseline group {d}: {err(U, W) * 1e3:.2f} mHa', flush=True)
        pairs, left = affinity_pairs(leaf_gram(U, W))
        U, W = merge_exact(U, W, pairs, left)
        print(f'merged -> {U.shape[0]}x{U.shape[1]}x{W.shape[2]}: {err(U, W) * 1e3:.2f} mHa', flush=True)
        if not a.no_ctrunc:
            W = ctrunc(U, W, W.shape[2] // 2)
            print(f'ctrunc -> C={W.shape[2]}: {err(U, W) * 1e3:.2f} mHa (before refit)', flush=True)

    e, U_j, W_j = refine_joint(U, W, a.refit, every=50)
    U, W = U_j, W_j
    save(a.out / f'chk_R{U.shape[0]}_B{U.shape[1]}_C{W.shape[2]}_{e * 1e3:.1f}mHa.npz', U, W)
    print(f'refit -> {e * 1e3:.2f} mHa', flush=True)

    while U.shape[0] > a.target_r:
        cn = contrib(U, W)
        d = int(np.argmin(cn))
        U2 = np.delete(U, d, 0)
        W2 = np.delete(W, d, 0)
        e0 = err(U2, W2)
        print(f'drop group {d} (contrib {cn[d]:.3f}): raw {e0 * 1e3:.2f} mHa', flush=True)
        e, U2, W2 = refine_joint(U2, W2, a.refit, every=100)
        print(f'  refit -> {e * 1e3:.2f} mHa', flush=True)
        U, W = U2, W2
        save(a.out / f'chk_R{U.shape[0]}_B{U.shape[1]}_C{W.shape[2]}_{e * 1e3:.1f}mHa.npz', U, W)
        if e > a.e_stop:
            print(f'stopping at R={U.shape[0]}: error {e * 1e3:.2f} mHa exceeds stop {a.e_stop * 1e3:.1f} mHa')
            break
    print('final', U.shape, W.shape, f'{e * 1e3:.2f} mHa')


if __name__ == '__main__':
    main()
