"""Cluster-then-truncate seed: partition the baseline's R_b groups into R clusters by
leaf-space similarity, pool each cluster's rows, keep the strongest B per new group.

Group r's leaf space is X_r = M_r W_r (P x C). Affinity between groups uses the
Frobenius inner product of their leaf Gram matrices; clustering is greedy agglomerative
on the number of target clusters.

Usage: python dfthcsos/seed_cluster.py --R 10 --B 129 --out runs/seed_c_10_129.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from target import pack  # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / 'data/starter_kit/starter_kit/starter/task1_factorization/FeMoco76_65_60_60.npz'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--R', type=int, required=True)
    ap.add_argument('--B', type=int, required=True)
    ap.add_argument('--baseline', type=Path, default=BASELINE)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    z = np.load(a.baseline)
    U, W = z['U'], z['W']           # (Rb, Bb, N), (Rb, Bb, Cb)
    Rb, Bb, N = U.shape
    Un = U / np.linalg.norm(U, axis=-1, keepdims=True)
    M = np.stack([pack(np.einsum('p,q->pq', Un[r, b], Un[r, b])) for r in range(Rb) for b in range(Bb)])
    M = M.reshape(Rb, Bb, -1)        # (Rb, Bb, P)
    X = np.einsum('rbp,rbc->rcp', M, W)  # (Rb, Cb, P) leaf space per group
    G = np.einsum('rcp,rdp->rcd', X, X)  # (Rb, Cb, Cb) leaf Gram per group

    # greedy agglomeration: merge the highest-affinity pair whose combined size stays <= cap,
    # so each cluster keeps a large enough row pool; merge the two smallest only when stuck.
    clusters = [[r] for r in range(Rb)]
    grams = [G[r].ravel().astype(np.float64) for r in range(Rb)]
    norms = [np.linalg.norm(g) for g in grams]
    min_pool = -(-a.B // Bb)  # parents needed per cluster to supply B rows
    cap = max(min_pool, -(-Rb // a.R) + 1)
    while len(clusters) > a.R:
        best = (-1.0, 0, 1)
        for i in range(len(clusters)):
            gi = grams[i]
            for j in range(i + 1, len(clusters)):
                if len(clusters[i]) + len(clusters[j]) > cap:
                    continue
                aff = float(np.dot(gi, grams[j]) / (norms[i] * norms[j] + 1e-30))
                if aff > best[0]:
                    best = (aff, i, j)
        if best[0] < 0:  # nothing fits under the cap; merge the two smallest
            sz = sorted(range(len(clusters)), key=lambda t: len(clusters[t]))
            i, j = sz[0], sz[1]
        else:
            _, i, j = best
        clusters[i] += clusters[j]
        grams[i] = grams[i] + grams[j]
        norms[i] = np.linalg.norm(grams[i])
        del clusters[j], grams[j], norms[j]

    imp = np.linalg.norm(W, axis=-1)  # (Rb, Bb)
    used = np.zeros((Rb, Bb), dtype=bool)
    rows = np.empty((a.R, a.B, N))
    weakest = np.inf
    for k, cl in enumerate(clusters):
        pool = sorted(((imp[r, b], r, b) for r in cl for b in range(Bb)), reverse=True)
        for b, (v, r, bb) in enumerate(pool[:a.B]):
            rows[k, b] = Un[r, bb]
            used[r, bb] = True
            weakest = min(weakest, v)
        if len(pool) < a.B:  # backfill with strongest unused rows anywhere
            extra = sorted(((imp[r, b], r, b) for r in range(Rb) for b in range(Bb) if not used[r, b]),
                           reverse=True)
            for b, (_, r, bb) in zip(range(len(pool), a.B), extra):
                rows[k, b] = Un[r, bb]
                used[r, bb] = True
    sizes = [len(c) for c in clusters]
    np.savez(a.out, U=rows)
    print(f'clusters {sizes} -> {a.R} groups x {a.B} rows; weakest kept imp {weakest:.4f}')


if __name__ == '__main__':
    main()
