"""Exhaustive single-op deletion audit for invsqrt.

For each CCX op: emit circuit without it, simulate a probe set of x values,
check x-restoration, scratch cleanliness, and phase tolerance. Survivors are
reported for official verification.
"""
import sys, os, time
sys.path.insert(0, 'tools')
sys.path.insert(0, 'data/verifiers/invsqrt')
import numpy as np
import kickmix as km
import verify_submission_invsqrt as V
from kmb_edit import parse_full, write_kmb

SRC = 'public/ozer28/data/invsqrt.kmb'
TMP = '/tmp/prune_cand.kmb'
BATCH = 32

p = parse_full(SRC)
spec = V.ProblemSpecInvSqrt.for_n_p(10, shots_per_x=1)
circuit = km.Circuit.from_file(SRC)
x_reg, scratch = V.validate_registers(circuit, spec)
target, tol = V.reference_values(spec, np.arange(1, spec.max_x + 1, dtype=np.uint64))

# probe x values: edges, powers, midpoints, and near max
xs_probe = np.array(sorted(set(
    [1, 2, 3, 4, 5, 8, 16, 33, 100, 255, 1024, 4096, 65536,
     spec.max_x // 4, spec.max_x // 2, 3 * spec.max_x // 4,
     spec.max_x - 2, spec.max_x - 1, spec.max_x]
    + list(np.linspace(1, spec.max_x, 13, dtype=np.int64))))[:BATCH], dtype=np.uint64)
xs_probe = np.pad(xs_probe, (0, BATCH - len(xs_probe)), constant_values=7)

def quick_ok():
    sim = km.Simulator(batch_size=BATCH, debug_print=False,
                       counting_operations=False, seed=12345)
    sim.clear_for_shot()
    sim.write_across_shots(x_reg, V.to_bits(xs_probe, spec.width))
    try:
        sim.do(km.Circuit.from_file(TMP))
    except Exception as e:
        return False, f'sim error {e}'
    if not np.array_equal(V.from_bits(sim.read_across_shots(x_reg)), xs_probe):
        return False, 'x not restored'
    if len(scratch) and sim.read_across_shots(scratch).any(axis=1).any():
        return False, 'dirty scratch'
    theta = np.array([V.half_turns_to_units(sim.read_shot_phase(i))
                      for i in range(BATCH)], dtype=np.uint64)
    err = (theta - target[xs_probe - 1]).view(np.int64)
    if (np.abs(err).view(np.uint64) > tol[xs_probe - 1]).any():
        return False, 'phase out of tol'
    return True, 'ok'

# baseline sanity
ccx_idx = [i for i, o in enumerate(p['ops']) if (o['t'] & 0x7F) == 0x31]
print(f'{len(ccx_idx)} CCX ops; probe on {len(np.unique(xs_probe))} x values')
t0 = time.time()
write_kmb(p, TMP)
ok, msg = quick_ok()
print(f'baseline probe: {ok} ({msg}) in {time.time()-t0:.1f}s')
assert ok

survivors = []
for n, k in enumerate(ccx_idx):
    ops = p['ops'][:k] + p['ops'][k + 1:]
    write_kmb(p, TMP, ops=ops)
    ok, msg = quick_ok()
    if ok:
        survivors.append(k)
        print(f'  op {k}: DELETABLE  (CCX q{ops[k]["q"]})', flush=True)
    if (n + 1) % 100 == 0:
        print(f'  ...{n+1}/{len(ccx_idx)} ({time.time()-t0:.0f}s, {len(survivors)} found)', flush=True)
print('survivors:', survivors)
