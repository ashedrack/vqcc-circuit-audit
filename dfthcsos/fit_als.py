"""Alternating fit of the DF-THC two-body factorization.

Model (weighted packed form, see target.py):  H = sum_rho M_rho G_rho M_rho^T,
  M_rho[:, b] = pack(u_b u_b^T) for the unit vectors u_b of group rho,  G_rho = W_rho W_rho^T (rank <= C).
error_2 = ||H - T||_F.

G-step: with U fixed, min ||H - T|| over symmetric G is linear least squares; its normal equations
  sum_sigma K_rs G_s K_sr = M_r^T T M_r,  K_rs[b, b'] = (u_b . u'_b')^2
are solved by conjugate gradients on the B x B blocks, then each G_rho is projected to PSD rank C.
U-step: Adam on U with W fixed (exact gradient via jax).

Usage: python dfthcsos/fit_als.py --R 8 --B 143 --C 62 --outer 20 [--init FILE.npz] --out runs/x.npz
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
from target import N, ROWS, COLS, PAIR_W, target_matrix  # noqa: E402

T_NP = None


def unit(U):
    return U / jnp.linalg.norm(U, axis=-1, keepdims=True)


def pack_rank1(U):
    """(R, B, N) unit vectors -> (R, P, B) packed u u^T."""
    return jnp.transpose(U[..., ROWS] * U[..., COLS] * PAIR_W, (0, 2, 1))


@jax.jit
def gram_blocks(U):
    Un = unit(U)
    D = jnp.einsum('rbn,smn->rsbm', Un, Un)
    return D * D  # K[r, s] = (u_rb . u_sm)^2


def apply_normal(K, G):
    return jnp.einsum('rsab,sbc,rscd->rad', K, G, K.transpose(0, 1, 3, 2))


@jax.jit
def rhs_blocks(U, T):
    M = pack_rank1(unit(U))
    return jnp.einsum('rpb,pq,rqc->rbc', M, T, M)


def solve_G(U, T, G0, iters=60):
    K, Y = gram_blocks(U), rhs_blocks(U, T)
    A = jax.jit(lambda G: apply_normal(K, G))
    G = G0
    r = Y - A(G)
    p = r
    rs = jnp.vdot(r, r)
    for _ in range(iters):
        Ap = A(p)
        alpha = rs / jnp.vdot(p, Ap)
        G = G + alpha * p
        r = r - alpha * Ap
        rs_new = jnp.vdot(r, r)
        if float(rs_new) < 1e-28:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return G


def project_psd(G, C):
    w, V = np.linalg.eigh(np.asarray(G))
    idx = np.argsort(w)[::-1][:C]
    wc = np.clip(w[..., None, :][..., 0, :], 0, None)
    out = []
    for r in range(G.shape[0]):
        i = np.argsort(w[r])[::-1][:C]
        out.append(V[r][:, i] * np.sqrt(np.clip(w[r][i], 0, None)))
    return np.stack(out)  # W: (R, B, C)


def loss_from_UW(U, W, T, T2):
    M = pack_rank1(unit(U))
    X = jnp.einsum('rpb,rbc->rcp', M, W).reshape(-1, M.shape[1])  # leaves x (R*C, P)
    XX = X @ X.T
    ht = jnp.einsum('lp,pq,lq->', X, T, X)
    return jnp.sqrt(jnp.maximum(jnp.sum(XX * XX) - 2 * ht + T2, 0.0))


def error2(U, W, T, T2):
    return float(loss_from_UW(jnp.asarray(U), jnp.asarray(W), T, T2))


def fit(R, B, C, outer, u_steps, lr, seed, init=None, log=print):
    T = jnp.asarray(target_matrix())
    T2 = float(jnp.sum(T * T))
    rng = np.random.default_rng(seed)
    if init is not None:
        U = jnp.asarray(init['U'][:R, :B])
        W = np.asarray(init['W'][:R, :B, :C]) if 'W' in init else None
    else:
        U, W = jnp.asarray(rng.standard_normal((R, B, N))), None
    G = jnp.zeros((R, B, B)) if W is None else jnp.asarray(np.einsum('rbc,rdc->rbd', W, W))
    loss_fn = jax.jit(lambda U, W: loss_from_UW(U, W, T, T2))
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    best = (np.inf, None, None)
    for it in range(outer):
        t0 = time.time()
        G = solve_G(U, T, G)
        W = jnp.asarray(project_psd(G, C))
        e_g = float(loss_fn(U, W))
        opt = optax.adam(lr)
        st = opt.init(U)
        for _ in range(u_steps):
            val, g = grad_fn(U, W)
            upd, st = opt.update(g, st)
            U = unit(optax.apply_updates(U, upd))
        e_u = float(loss_fn(U, W))
        G = jnp.einsum('rbc,rdc->rbd', W, W)
        if e_u < best[0]:
            best = (e_u, np.asarray(U), np.asarray(W))
        log(f'[{it:3d}] error_2 after G-step {e_g * 1e3:8.3f} mHa, after U-step {e_u * 1e3:8.3f} mHa ({time.time() - t0:.1f}s)')
    return best


def refine_joint(U, W, steps=500, log=print, every=50, f32=False):
    """Joint L-BFGS on (U, W) for error_2 (W unconstrained, so rank C is kept).
    f32=True optimizes in float32 (~6x faster); returned metric is still f64."""
    T = jnp.asarray(target_matrix())
    T2 = float(jnp.sum(T * T))
    dt = jnp.float32 if f32 else jnp.float64
    Tl = jnp.asarray(np.asarray(target_matrix()), dtype=dt)
    T2l = jnp.sum(Tl * Tl)
    f = lambda p: loss_from_UW(p['U'], p['W'], Tl, T2l)  # noqa: E731
    opt = optax.lbfgs()
    p = dict(U=jnp.asarray(U, dtype=dt), W=jnp.asarray(W, dtype=dt))
    st = opt.init(p)

    @jax.jit
    def step(p, st):
        val, g = jax.value_and_grad(f)(p)
        upd, st = opt.update(g, st, p, value=val, grad=g, value_fn=f)
        return optax.apply_updates(p, upd), st, val

    best = (float(f(p)), p)
    for i in range(steps):
        p, st, val = step(p, st)
        if float(val) < best[0]:
            best = (float(val), p)
        if i % every == 0:
            log(f'  lbfgs {i:5d} error_2 {float(val) * 1e3:.3f} mHa')
    Uo, Wo = unit(best[1]['U']), best[1]['W']
    e = error2(Uo, Wo, T, T2) if f32 else best[0]
    return e, np.asarray(Uo, dtype=np.float64), np.asarray(Wo, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--R', type=int, required=True)
    ap.add_argument('--B', type=int, required=True)
    ap.add_argument('--C', type=int, required=True)
    ap.add_argument('--outer', type=int, default=20)
    ap.add_argument('--u-steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--lbfgs', type=int, default=0, help='joint L-BFGS steps after the alternating stage')
    a = ap.parse_args()
    init = dict(np.load(a.init)) if a.init else None
    if a.outer > 0:
        e, U, W = fit(a.R, a.B, a.C, a.outer, a.u_steps, a.lr, a.seed, init)
        print(f'best error_2 {e * 1e3:.3f} mHa', flush=True)
    else:
        U, W = init['U'][:a.R, :a.B], init['W'][:a.R, :a.B, :a.C]
    if a.lbfgs:
        e, U, W = refine_joint(U, W, a.lbfgs)
        print(f'after L-BFGS error_2 {e * 1e3:.3f} mHa')
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out, U=U, W=W, error_2=e)


if __name__ == '__main__':
    main()
