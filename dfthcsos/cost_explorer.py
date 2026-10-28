"""Cost landscape of dfthcsos shapes using the verifier's own heuristic cost model.

score = lambda_eff * T * Q with
  T = 2 * K * angle_toffoli_per_circuit + heuristic Toffolis,  K = ceil(75 / J)
  Q = heuristic persistent + J * brot + max(heuristic temp, angle temp)
The angle circuit cost is modelled as `frac * A` Toffolis per circuit (A = 76 + R*B addresses):
frac = 1.0 is a plain unary-iteration QROM; public compilers reach ~0.24.

Usage: python dfthcsos/cost_explorer.py [--frac 1.0] [--lam 48.5] [--top 25]
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data' / 'verifiers'))
from dfthcsos.verify import heuristic_costs, QUBIT_CAP  # noqa: E402

N = 76
BASELINE = 3355599588.3155365


def shape_cost(R: int, B: int, C: int, brot: int, J: int, frac: float, lam: float) -> dict | None:
    A = N + R * B
    sel = math.ceil(math.log2(A))
    angle_temp = sel  # unary iteration ancillas, roughly
    if sel + J * brot + angle_temp > QUBIT_CAP:
        return None
    try:
        h = heuristic_costs(N, R, B, C, brot)
    except ValueError:
        return None
    K = math.ceil((N - 1) / J)
    angles_T = 2 * K * frac * A
    T = angles_T + h['toffoli']
    Q = h['persistent_qubits'] + J * brot + max(h['temporary_qubits'], angle_temp)
    score = lam * T * Q
    return dict(R=R, B=B, C=C, brot=brot, J=J, A=A, K=K, angles_T=round(angles_T), heur_T=h['toffoli'],
                T=round(T), Q=Q, temp=h['temporary_qubits'], score=score, ratio=BASELINE / score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--frac', type=float, default=1.0)
    ap.add_argument('--lam', type=float, default=48.5)
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--brot', type=int, nargs='*', default=[7, 8, 9, 10])
    a = ap.parse_args()
    rows = []
    for R, B, C, brot, J in itertools.product(range(4, 17, 2), range(64, 257, 8), (40, 46, 52, 60, 62),
                                              a.brot, range(3, 14)):
        r = shape_cost(R, B, C, brot, J, a.frac, a.lam)
        if r:
            rows.append(r)
    rows.sort(key=lambda r: r['score'])
    print(f'frac={a.frac} lam={a.lam}: {len(rows)} feasible shapes (accuracy NOT checked)')
    for r in rows[:a.top]:
        print('  ' + ' '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items()))
    for ref in ((65, 60, 60, 12, 8), (8, 143, 62, 8, 5)):
        print('ref', shape_cost(*ref, a.frac, a.lam))


if __name__ == '__main__':
    main()
