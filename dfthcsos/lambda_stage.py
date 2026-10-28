"""Stage 2: with the two-body vectors U and weights w fitted, choose the identity offsets H2_I (= w_B)
and a per-group orthogonal gauge of w (which leaves error_2 unchanged) to minimise lambda_eff,
subject to E_gap > floor. Formulas follow dfthcsos/scoreinfra/dfthc_math.py exactly.

Usage: python dfthcsos/lambda_stage.py runs/fit.npz --out runs/fit_lam.npz [--steps 3000]
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
from target import N, reference  # noqa: E402

E_UPPER = -22140.27335
GAP_FLOOR = 3.914898
LAMBDA_FLOOR = 49.0


def consts():
    ref, S, beta_1 = reference()
    oei, tei = np.asarray(ref['oei']), np.asarray(ref['tei'])
    h1_thc = oei - 0.5 * np.einsum('prrq->pq', tei)
    eta = ref['electrons']
    base = h1_thc - 0.5 * eta * S + beta_1 * np.eye(N)
    return dict(base=jnp.asarray(base), ecore=float(ref['ecore']), eta=eta, beta_1=beta_1)


def metrics(U, w, H2_I, c):
    Un = U / jnp.linalg.norm(U, axis=-1, keepdims=True)
    X = jnp.einsum('rbp,rbq,rbc->rcpq', Un, Un, w)
    W_off = H2_I - jnp.sum(w, axis=1)
    h1p = c['base'] - jnp.einsum('rc,rcpq->pq', W_off, X)
    ev = jnp.linalg.eigvalsh(h1p)
    lam1 = jnp.sum(jnp.abs(ev))
    lam2 = 0.25 * jnp.sum(jnp.square(jnp.sum(jnp.abs(w), axis=1) + jnp.abs(H2_I)))
    Lam = lam1 + lam2
    e_sos = c['ecore'] - c['eta'] * c['beta_1'] - (lam1 - jnp.sum(ev)) - 0.5 * jnp.sum(W_off ** 2)
    gap = E_UPPER - e_sos
    lam_eff = jnp.sqrt(jnp.clip(gap, 1e-9) * (2 * Lam - gap))
    return dict(lambda_eff=lam_eff, Lambda=Lam, lambda_h1=lam1, lambda_h2=lam2, E_gap=gap)


def cayley(A):
    """Orthogonal matrices from skew-symmetric parts of A (batched)."""
    S = A - jnp.swapaxes(A, -1, -2)
    I = jnp.eye(S.shape[-1])
    return jnp.linalg.solve(I + S, I - S)


def optimise(U, w, H2_I=None, steps=3000, lr=1e-2, margin=2e-6, log=print):
    c = consts()
    U, w = jnp.asarray(U), jnp.asarray(w)
    R, B, C = w.shape
    H2_I = jnp.sum(w, axis=1) if H2_I is None else jnp.asarray(H2_I)  # W_off = 0 start
    params = dict(H=H2_I, A=jnp.zeros((R, C, C)))
    floor = GAP_FLOOR + margin

    def loss(p):
        wg = jnp.einsum('rbc,rcd->rbd', w, cayley(p['A']))
        m = metrics(U, wg, p['H'], c)
        return m['lambda_eff'] + 1e3 * jax.nn.relu(floor - m['E_gap']) + 1e3 * jax.nn.relu(LAMBDA_FLOOR + 1e-3 - m['Lambda'])

    opt = optax.adam(lr)
    st = opt.init(params)
    step = jax.jit(lambda p, s: _step(p, s, loss, opt))
    best = (np.inf, params)
    for i in range(steps):
        params, st, val = step(params, st)
        wg = jnp.einsum('rbc,rcd->rbd', w, cayley(params['A']))
        m = metrics(U, wg, params['H'], c)
        ok = float(m['E_gap']) > GAP_FLOOR and float(m['Lambda']) > LAMBDA_FLOOR
        if ok and float(m['lambda_eff']) < best[0]:
            best = (float(m['lambda_eff']), params)
        if i % max(1, steps // 10) == 0:
            log(f'  step {i:5d} lambda_eff {float(m["lambda_eff"]):.4f} Lambda {float(m["Lambda"]):.3f} '
                f'gap {float(m["E_gap"]):.6f} (h1 {float(m["lambda_h1"]):.2f}, h2 {float(m["lambda_h2"]):.2f})')
    p = best[1]
    return best[0], np.asarray(jnp.einsum('rbc,rcd->rbd', w, cayley(p['A']))), np.asarray(p['H'])


def _step(p, s, loss, opt):
    val, g = jax.value_and_grad(loss)(p)
    upd, s = opt.update(g, s)
    return optax.apply_updates(p, upd), s, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('fit', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--lr', type=float, default=1e-2)
    a = ap.parse_args()
    z = np.load(a.fit)
    lam, w, H = optimise(z['U'], z['W'], z['H2_I'] if 'H2_I' in z.files else None, a.steps, a.lr)
    print(f'best feasible lambda_eff {lam:.4f}')
    shifts = np.load(Path(__file__).resolve().parents[1] / 'data/verifiers/dfthcsos/FeMoCo76o113e.fixed_shifts.npz')
    np.savez(a.out, U=z['U'], W=w, H2_I=H, H1_I=shifts['H1_I'], Hsym_lower=shifts['Hsym_lower'])


if __name__ == '__main__':
    main()
