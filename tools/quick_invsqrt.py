"""Fast invsqrt check on a subset of x (edges, powers of 4, random), using the verifier's math.

Usage: python tools/quick_invsqrt.py circuit.kmb [--random 65536] [--shots 2]
Prints PASS/FAIL, worst error/tol, mean Toffolis, rotation count, qubits and the estimated score.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import kickmix as km

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data' / 'verifiers' / 'invsqrt'))
import verify_submission_invsqrt as v  # noqa: E402

BASELINE_SCORE = 2076774.233715282


def sample_xs(max_x: int, n_random: int, seed: int) -> np.ndarray:
    xs = set(range(1, 4097)) | set(range(max_x - 4096, max_x + 1))
    for e in range(23):
        for dx in range(-3, 4):
            if 1 <= (1 << e) + dx <= max_x:
                xs.add((1 << e) + dx)
    xs |= set(np.random.default_rng(seed).integers(1, max_x + 1, n_random).tolist())
    return np.array(sorted(xs), dtype=np.uint64)


def check(circuit: km.Circuit, n_random: int = 65536, shots: int = 2, seed: int = 0) -> dict:
    spec = v.ProblemSpecInvSqrt.for_n_p(v.N_P)
    x_reg, scratch = v.validate_registers(circuit, spec)
    xs = np.repeat(sample_xs(spec.max_x, n_random, seed), shots)
    batch = 256
    xs = np.concatenate([xs, np.ones((-len(xs)) % batch, dtype=np.uint64)])
    target, tol = v.reference_values(spec, xs)
    sim = km.Simulator(batch_size=batch, counting_operations=True, seed=seed)
    worst, min_margin = 0.0, 1 << 64
    for s in range(0, len(xs), batch):
        bx = xs[s:s + batch]
        sim.clear_for_shot()
        sim.write_across_shots(x_reg, v.to_bits(bx, spec.width))
        sim.do(circuit)
        if not np.array_equal(v.from_bits(sim.read_across_shots(x_reg)), bx):
            return {'status': 'FAIL', 'error': 'x not restored'}
        if len(scratch) and sim.read_across_shots(scratch).any():
            return {'status': 'FAIL', 'error': 'dirty scratch'}
        theta = np.array([v.half_turns_to_units(sim.read_shot_phase(i)) for i in range(batch)], dtype=np.uint64)
        err = np.abs((theta - target[s:s + batch]).view(np.int64)).view(np.uint64)
        allowed = tol[s:s + batch]
        bad = np.flatnonzero(err > allowed)
        if len(bad):
            i = bad[0]
            return {'status': 'FAIL', 'error': f'x={bx[i]} err/tol={err[i] / allowed[i]:.3f}'}
        worst = max(worst, float((err / allowed).max()))
        min_margin = min(min_margin, int((allowed - err).min()))
    st = sim.operation_statistics()
    n = len(xs)
    tof, t_base = st.num_toffoli_like / n, st.num_t_like / n
    n_rot = st.num_z_pow(min_clifford_hierarchy_level=4) / n
    eps = math.pi * min_margin / 2**64 / n_rot if n_rot else None
    c = tof + (t_base + (n_rot * v.rotation_t_count(eps) if n_rot else 0)) / v.T_PER_TOFFOLI
    score = circuit.num_qubits * c
    return {'status': 'PASS', 'worst_err_over_tol': round(worst, 4), 'toffoli': tof, 'rotations': n_rot,
            'C': round(c, 1), 'Q': circuit.num_qubits, 'score': round(score), 'improvement': round(BASELINE_SCORE / score, 3)}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('circuit')
    ap.add_argument('--random', type=int, default=65536)
    ap.add_argument('--shots', type=int, default=2)
    a = ap.parse_args()
    print(check(km.Circuit.from_file(a.circuit), a.random, a.shots))
