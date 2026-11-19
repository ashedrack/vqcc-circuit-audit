"""Evaluate chem cost-model shapes (R,B,C,brot) without fitting.

The verifier's cost = lambda_eff * T_total * Q_total, where
  T_total = angles_toffoli + outer_prep + inner_prep + givens_rotations
  Q_total = persistent + max temporary
and lambda_eff comes from the fit. This tool computes the shape-dependent
parts exactly (heuristic_costs from the official scoreinfra) and estimates
the loader part proportional to angle-word count (R*B + N).

Deployed reference: R=8, B=143, C=44, brot=8 -> 1220 words,
heuristic T = 6662, angles T = 8044, T_total = 14706, Q_total = ~427ish.

Usage:
    ../.venv/bin/python dfthcsos/shape_scan.py
"""
import sys, itertools, math
sys.path.insert(0, 'data/verifiers/dfthcsos')
from scoreinfra.costs_prepare import SpinFreeData_V1

N = 76
KEEP_BITS = 10
QUBIT_CAP = 128

# deployed reference numbers for calibration
REF = dict(R=8, B=143, C=44, brot=8, heuristic_T=6662, angles_T=8044)
WORDS_PER = lambda R, B: R * B + N          # angle-word count


def heuristic(R, B, C, brot):
    m = SpinFreeData_V1(R=R, B=B, C=C, N=N, chi=KEEP_BITS, beta=brot)
    oq = m.outer_state_prep_qubit_cost()
    iq = m.inner_state_prep_qubit_cost()
    T = m.outer_state_prep() + m.inner_state_prep() + m.givens_rotations()
    persist = oq['persistent'] + iq['persistent']
    temp = max(oq['temporary'], iq['temporary'])
    return T, persist + temp, persist, temp


def estimate_angles_T(R, B):
    """Loader Toffolis scale with word count; calibrated on deployed 4022/pass."""
    return REF['angles_T'] * WORDS_PER(R, B) / WORDS_PER(REF['R'], REF['B'])


if __name__ == '__main__':
    print('%-4s %-4s %-4s %-5s | %-7s %-7s %-7s %-7s | %-8s' %
          ('R', 'B', 'C', 'brot', 'heur_T', 'ang_T*', 'T_tot*', 'Q_heur', 'words'))
    rows = []
    for R, B, C, brot in itertools.product(
            range(5, 13), [64, 80, 96, 112, 128, 129, 143, 160, 180, 200],
            range(30, 60, 6), [7, 8]):
        try:
            T, Q, p, tmp = heuristic(R, B, C, brot)
        except Exception:
            continue
        ang = estimate_angles_T(R, B)
        rows.append((R, B, C, brot, T, ang, T + ang, Q, WORDS_PER(R, B)))
    # rank by estimated T_total (lambda_eff roughly fixed near cap)
    rows.sort(key=lambda r: r[6] * r[7])     # proxy: T * Q (lambda fixed)
    for r in rows[:25]:
        print('%-4d %-4d %-4d %-5d | %-7d %-7.0f %-7.0f %-7d | %-8d' % r[:9])
    print('\ndeployed: R8 B143 C44 brot8 -> T 14706 Q 427ish')
    print('* angles T estimated ~ word count; lambda_eff not modeled (fit-dependent)')
