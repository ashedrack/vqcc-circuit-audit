"""Seed a smaller (R, B, C) fit from the official baseline factorization.

Ranks all R_b*B_b baseline vectors by importance ||W[r,b,:]||_F, keeps the top R*B,
and deals them round-robin into R groups so each group gets a mix of strong and weak rows.
W is left to the exact G-step (fit_als solves G_r = W_r W_r^T exactly given U).

Usage: python dfthcsos/seed_from_baseline.py --R 8 --B 143 --out runs/seed_8_143.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BASELINE = Path(__file__).resolve().parents[1] / 'data/starter_kit/starter_kit/starter/task1_factorization/FeMoco76_65_60_60.npz'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--R', type=int, required=True)
    ap.add_argument('--B', type=int, required=True)
    ap.add_argument('--baseline', type=Path, default=BASELINE)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    z = np.load(a.baseline)
    U, W = z['U'], z['W']  # (Rb, Bb, N), (Rb, Bb, Cb)
    Rb, Bb, N = U.shape
    imp = np.linalg.norm(W, axis=-1).ravel()  # (Rb*Bb,)
    need = a.R * a.B
    top = np.argsort(imp)[::-1][:need]
    rows = U.reshape(Rb * Bb, N)[top]
    # round-robin deal into R groups so importance is balanced across groups
    groups = [rows[r::a.R] for r in range(a.R)]
    Useed = np.stack(groups)  # (R, B, N)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, U=Useed)
    print(f'seed {a.R}x{a.B} from {Rb}x{Bb}: kept imp {imp[top].min():.4f}..{imp[top].max():.4f} '
          f'(dropped {imp.size - need} rows, weakest kept rank {need})')
    print('group imp sums:', np.array([imp[top][r::a.R].sum() for r in range(a.R)]).round(1))


if __name__ == '__main__':
    main()
