"""Convert a fit_als.py output npz (U, W) into the verifier's factorization format.

Adds H2_I (defaults to sum_b W, i.e. zero W offset), and the fixed H1_I/Hsym_lower
shifts (the verifier replaces them anyway, but the trainer wants them present).

Usage: python dfthcsos/to_official.py runs/als_seed_8_143_62.npz runs/fact_8_143_62.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'data' / 'verifiers'))
from dfthcsos.scoreinfra import reference as REF  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('fit', type=Path)
    ap.add_argument('out', type=Path)
    a = ap.parse_args()
    z = np.load(a.fit)
    U = np.asarray(z['U'], dtype=np.float64)
    U /= np.linalg.norm(U, axis=-1, keepdims=True)
    W = np.asarray(z['W'], dtype=np.float64)
    H2_I = np.asarray(z['H2_I'], dtype=np.float64) if 'H2_I' in z.files else np.sum(W, axis=1)
    shifts = REF.load_fixed_shifts()
    np.savez(a.out, U=U, W=W, H2_I=H2_I,
             H1_I=np.asarray(shifts['H1_I']), Hsym_lower=np.asarray(shifts['Hsym_lower']))
    print(f'wrote {a.out}: U{U.shape} W{W.shape} H2_I{H2_I.shape}')


if __name__ == '__main__':
    main()
