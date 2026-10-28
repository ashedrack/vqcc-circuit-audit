"""Sweep invsqrt design parameters; prints model error and quick-check score for each.

Usage: python invsqrt/sweep.py  (edit GRID below)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from build import Params, make, TOL  # noqa: E402
from quick_invsqrt import check  # noqa: E402

GRID = dict(k=[9], frac=[27, 28, 29], degree=[2], guard=[0, 1, 2],
            fdelta=[(a, b) for a in (-2, -1, 0) for b in (-4, -3, -2, -1, 0)])
MAX_ERR = 0.97

if __name__ == '__main__':
    results = []
    for values in itertools.product(*GRID.values()):
        p = Params(**dict(zip(GRID, values)))
        try:
            _, err, c = make(p, verbose=False)
        except AssertionError as e:
            continue
        if err > MAX_ERR * TOL:
            continue
        r = check(c, n_random=8192, shots=1)
        results.append((r.get('score', float('inf')), p, err / TOL, r))
        print(f'{r.get("improvement")}x {p.frac} {p.fdelta} g={p.guard}: err {err / TOL:.3f} Q={r.get("Q")} C={r.get("C")}',
              flush=True)
    print('\nBest:')
    for s, p, e, r in sorted(results, key=lambda t: t[0])[:8]:
        print(f'  {r.get("improvement")}x  {p}  err {e:.3f}  Q={r.get("Q")} C={r.get("C")}')
