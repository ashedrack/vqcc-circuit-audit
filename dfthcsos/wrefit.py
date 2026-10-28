"""Quantization-aware refit: fix U, encode rows to brot-bit angle words, decode back to Uq,
then refit W (and only W) against Uq so error_2 is evaluated on the vectors the circuits
will actually reconstruct.

Usage: python dfthcsos/wrefit.py FACT.npz OUT.npz --brot 8 --steps 800 [--tree heap]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

jax.config.update('jax_enable_x64', True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from target import N, target_matrix  # noqa: E402
from fit_als import loss_from_UW  # noqa: E402
from pipeline import vectors_to_angles, angles_to_words, words_to_vectors, TREES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data' / 'verifiers'))
from dfthcsos.scoreinfra import reference as REF  # noqa: E402


def quantize_U(U, edges, brot):
    R, B, _ = U.shape
    w = angles_to_words(vectors_to_angles(U.reshape(R * B, N), edges), brot)
    return words_to_vectors(w, edges, brot).reshape(R, B, N), w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('fact', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--brot', type=int, default=8)
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--tree', choices=sorted(TREES), default='heap')
    a = ap.parse_args()

    z = np.load(a.fact)
    U = np.asarray(z['U'], dtype=np.float64)
    W = np.asarray(z['W'], dtype=np.float64)
    edges = TREES[a.tree]()
    T = jnp.asarray(target_matrix())
    T2 = float(jnp.sum(T * T))

    Uq, words2 = quantize_U(U, edges, a.brot)
    Uq = jnp.asarray(Uq)
    f = jax.jit(lambda W: loss_from_UW(Uq, W, T, T2))
    e0 = float(f(jnp.asarray(W)))
    e_c = float(loss_from_UW(jnp.asarray(U), jnp.asarray(W), T, T2))
    print(f'continuous {e_c * 1e3:.2f} mHa -> quantized {e0 * 1e3:.2f} mHa; refitting W', flush=True)

    opt = optax.lbfgs()
    Wj = jnp.asarray(W)
    st = opt.init(Wj)

    @jax.jit
    def step(Wj, st):
        val, g = jax.value_and_grad(f)(Wj)
        upd, st = opt.update(g, st, Wj, value=val, grad=g, value_fn=f)
        return optax.apply_updates(Wj, upd), st, val

    best = (e0, W)
    for i in range(a.steps):
        Wj, st, val = step(Wj, st)
        if float(val) < best[0]:
            best = (float(val), np.asarray(Wj))
        if i % 50 == 0:
            print(f'  lbfgs {i:5d} error_2 {float(val) * 1e3:.3f} mHa', flush=True)
    e, W = best
    print(f'best quantized error_2 {e * 1e3:.3f} mHa')

    shifts = REF.load_fixed_shifts()
    H2_I = np.asarray(z['H2_I'], dtype=np.float64) if 'H2_I' in z.files else np.sum(W, axis=1)
    np.savez(a.out, U=Uq, W=W, H2_I=H2_I,
             H1_I=np.asarray(shifts['H1_I']), Hsym_lower=np.asarray(shifts['Hsym_lower']),
             angle_words_2body=words2, brot=a.brot, tree_edges=edges)
    print(f'wrote {a.out} (U stored = decoded vectors; re-encoding reproduces identical words)')


if __name__ == '__main__':
    main()
