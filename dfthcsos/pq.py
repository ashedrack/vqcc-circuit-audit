"""Progressive quantization of Givens angle words (our own implementation).

Parametrize the two-body rows by their Givens angles (theta/pi units). Each stage:
  1. rank FREE angles by |d error_2 / d angle|,
  2. snap the lowest-impact fraction to the brot-bit grid and freeze them,
  3. Adam-refit the still-free angles only (clipped, gentle),
  4. solve W exactly for the decoded vectors (CG G-step + rank-C projection).

The W solve is exact least squares, so it can only improve the stage's error;
the angle moves supply the compensation. Track best fully-specified iterate.

Usage: python dfthcsos/pq.py FACT.npz OUT.npz --brot 8 --stages 16 --steps 150
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

jax.config.update('jax_enable_x64', True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from target import N, target_matrix  # noqa: E402
from fit_als import loss_from_UW, solve_G, project_psd  # noqa: E402
from pipeline import vectors_to_angles, TREES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data' / 'verifiers'))
from dfthcsos.scoreinfra import reference as REF  # noqa: E402


def decode_angles(theta, edges):
    """(V, 75) theta/pi -> (V, 76) unit vectors; jax version of DM.reconstruct_u_vectors."""
    v = jnp.zeros((theta.shape[0], N)).at[:, 0].set(1.0)

    def body(v, t_edge):
        th, (p, c) = t_edge
        co, si = jnp.cos(jnp.pi * th), jnp.sin(jnp.pi * th)
        vp, vc = v[:, p], v[:, c]
        v = v.at[:, p].set(co * vp - si * vc)
        v = v.at[:, c].set(si * vp + co * vc)
        return v, None

    v, _ = jax.lax.scan(body, v, (theta.T, edges))
    return v


def snap(theta, brot):
    return jnp.floor(jnp.mod(theta, 2.0) * (1 << (brot - 1)) + 0.5) / (1 << (brot - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('fact', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--brot', type=int, default=8)
    ap.add_argument('--stages', type=int, default=16)
    ap.add_argument('--steps', type=int, default=150, help='Adam steps per stage')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--tree', choices=sorted(TREES), default='heap')
    a = ap.parse_args()

    z = np.load(a.fact)
    U = np.asarray(z['U'], dtype=np.float64)
    R, B, _ = U.shape
    C = z['W'].shape[2]
    W = np.asarray(z['W'], dtype=np.float64)
    edges = TREES[a.tree]()
    edges_j = jnp.asarray(edges)
    T = jnp.asarray(target_matrix())
    T2 = float(jnp.sum(T * T))
    V = R * B

    theta = jnp.asarray(vectors_to_angles(U.reshape(V, N), edges))
    frozen = np.zeros(theta.shape, dtype=bool)

    @jax.jit
    def loss_and_grad(theta, W):
        f = lambda th: loss_from_UW(decode_angles(th, edges_j).reshape(R, B, N), W, T, T2)  # noqa: E731
        return jax.value_and_grad(f)(theta)

    def err(th, W):
        u = decode_angles(th, edges_j).reshape(R, B, N)
        return float(loss_from_UW(u, W, T, T2))

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    best = (np.inf, None, None)
    t0 = time.time()
    Wj = jnp.asarray(W)
    for stage in range(a.stages + 1):
        if stage < a.stages:
            _, g = loss_and_grad(theta, Wj)
            imp = np.asarray(jnp.abs(g))
            frac = 1.0 / (a.stages - stage)
            thr = np.quantile(imp[~frozen], frac)
            newly = (~frozen) & (imp <= thr)
            frozen |= newly
            th_new = np.array(theta, copy=True)
            th_new[newly] = np.asarray(snap(theta, a.brot))[newly]
            theta = jnp.asarray(th_new)
            opt_state = opt.init(theta)
            e = err(theta, Wj)
            print(f'stage {stage}: froze {int(newly.sum())} ({frozen.sum()}/{frozen.size}), '
                  f'error_2 {e * 1e3:.2f} mHa ({time.time() - t0:.0f}s)', flush=True)
        else:
            opt_state = opt.init(theta)
        for _ in range(a.steps):
            val, g = loss_and_grad(theta, Wj)
            g = g * jnp.asarray(~frozen)
            upd, opt_state = opt.update(g, opt_state)
            theta = optax.apply_updates(theta, upd)
        # exact W solve for the current decoded vectors
        Uq = decode_angles(theta, edges_j).reshape(R, B, N)
        G = solve_G(Uq, T, jnp.einsum('rbc,rdc->rbd', Wj, Wj))
        Wj = jnp.asarray(project_psd(np.asarray(G), C))
        e = err(theta, Wj)
        if e < best[0]:
            best = (e, np.asarray(theta), np.asarray(Wj))
        print(f'  refit -> {e * 1e3:.3f} mHa (best {best[0] * 1e3:.3f})', flush=True)

    e, th, W = best
    th[~frozen] = np.asarray(snap(jnp.asarray(th[~frozen]), a.brot))
    # final exact W solve on the fully snapped vectors
    Uq = np.asarray(decode_angles(jnp.asarray(th), edges_j)).reshape(R, B, N)
    G = solve_G(jnp.asarray(Uq), T, jnp.einsum('rbc,rdc->rbd', W, W))
    W = project_psd(np.asarray(G), C)
    e_final = float(loss_from_UW(jnp.asarray(Uq), jnp.asarray(W), T, T2))
    words = np.floor(np.mod(th, 2.0) * (1 << (a.brot - 1)) + 0.5).astype(np.int64) % (1 << a.brot)
    shifts = REF.load_fixed_shifts()
    H2_I = np.asarray(z['H2_I'], dtype=np.float64) if 'H2_I' in z.files else np.sum(W, axis=1)
    np.savez(a.out, U=Uq, W=W, H2_I=H2_I,
             H1_I=np.asarray(shifts['H1_I']), Hsym_lower=np.asarray(shifts['Hsym_lower']),
             angle_words_2body=words, tree_edges=edges, brot=a.brot)
    print(f'final fully-quantized error_2 {e_final * 1e3:.3f} mHa -> {a.out}')


if __name__ == '__main__':
    main()
