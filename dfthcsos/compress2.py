"""Group-merge compression: merge pairs of groups (lossless, B/C double) then
truncate the merged groups' rows/copies and refit — the public 'merged-groups'
recipe to reach small R (8-13) at B~129.

    (32,120,60) -> merge -> (16,240,120) -> rtrunc 240->129 -> ctrunc 120->60
    -> refit -> [merge -> (8,258,120) -> rtrunc -> ctrunc -> refit] -> target

Row truncation drops the rows with smallest leaf weight sum_c w_b,c^2 (their
contribution to H across all copies), then L-BFGS refits U and W jointly.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from target import target_matrix  # noqa: E402
from fit_als import error2, refine_joint  # noqa: E402
from compress import affinity_pairs, leaf_gram, merge_exact, ctrunc, contrib, save  # noqa: E402


def rtrunc(U, W, B):
    """Keep the B highest-contribution rows per group (score = sum_c w_bc^2)."""
    keep = np.argsort(-(W ** 2).sum(axis=2), axis=1)[:, :B]  # (R, B)
    return (np.take_along_axis(U, keep[..., None], axis=1),
            np.take_along_axis(W, keep[..., None], axis=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('init', type=Path, help='checkpoint npz (U, W)')
    ap.add_argument('--rounds', type=int, default=2, help='merge rounds')
    ap.add_argument('--b-target', type=int, default=129)
    ap.add_argument('--c-target', type=int, default=60)
    ap.add_argument('--refit', type=int, default=300)
    ap.add_argument('--e-stop', type=float, default=0.026)
    ap.add_argument('--out', type=Path, default=Path('runs/merge'))
    a = ap.parse_args()

    T = jnp.asarray(target_matrix())
    T2 = float(jnp.sum(T * T))
    err = lambda U, W: error2(U, W, T, T2)  # noqa: E731

    z = np.load(a.init)
    U, W = np.asarray(z['U'], dtype=np.float64), np.asarray(z['W'], dtype=np.float64)
    print(f'start {U.shape}x{W.shape[2]}: {err(U, W) * 1e3:.2f} mHa', flush=True)

    for rd in range(a.rounds):
        pairs, left = affinity_pairs(leaf_gram(U, W))
        U, W = merge_exact(U, W, pairs, left)
        print(f'round {rd}: merged -> ({U.shape[0]},{U.shape[1]},{W.shape[2]}): '
              f'{err(U, W) * 1e3:.2f} mHa', flush=True)

        if U.shape[1] > a.b_target:
            U, W = rtrunc(U, W, a.b_target)
            print(f'  rtrunc -> B={U.shape[1]}: {err(U, W) * 1e3:.2f} mHa (raw)', flush=True)
        if W.shape[2] > a.c_target:
            W = ctrunc(U, W, a.c_target)
            print(f'  ctrunc -> C={W.shape[2]}: {err(U, W) * 1e3:.2f} mHa (raw)', flush=True)

        e, U_j, W_j = refine_joint(U, W, a.refit, every=100)
        U, W = U_j, W_j
        print(f'  refit -> {e * 1e3:.2f} mHa', flush=True)
        save(a.out / f'mrg_R{U.shape[0]}_B{U.shape[1]}_C{W.shape[2]}_{e * 1e3:.1f}mHa.npz', U, W)
        if e > a.e_stop:
            print(f'stop: {e * 1e3:.2f} mHa > {a.e_stop * 1e3:.1f} mHa')
            break
    print('final', U.shape, W.shape)


if __name__ == '__main__':
    main()
