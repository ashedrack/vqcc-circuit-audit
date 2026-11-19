"""Angle refit for invsqrt: extract Z_POW basis, LP-minimax the phase error.

The tracked phase theta(shot) = sum_j (a_j/2)*s_j(shot) + phi_fixed(shot) mod 1,
where s_j(shot) is the basis-state of the rotated qubit (times its condition
bits) at that op. Z_POW never flips qubits, so s_j is angle-independent: theta
is EXACTLY LINEAR in the angles. Units: 2^-64 turns.

Workflow:
  1. extract angle packet (ptype 10): n_angles x 16 bytes, in op order.
  2. write_angles: rewrite just those bytes.
  3. sim_errors: replicate the verifier's shot loop for chosen xs.
  4. extract_basis: finite-difference each angle to get s_ij.
  5. LP: min t s.t. |resid_i(a)| <= t for all probed shots.
"""
import io, struct, hashlib, math
import numpy as np
import kickmix as km

MASK64 = (1 << 64) - 1


def angle_packet_offset(path):
    """Returns (data, offset, n_angles) for the ptype-10 packet payload."""
    data = np.fromfile(path, dtype=np.uint8)
    off = 16 + 16 + 72
    while off < len(data):
        ptype, plen = struct.unpack_from('<QQ', data, off)
        off += 16
        if ptype == 10:
            return data, off, plen // 16
        off += plen
    raise ValueError('no angle packet')


def decode_angles(path):
    data, off, n = angle_packet_offset(path)
    out = []
    for i in range(n):
        w0, w1 = struct.unpack_from('<QQ', data, off + 16 * i)
        ht = w0 / 2**128 + w1 / 2**64
        if ht > 1: ht -= 2
        out.append(ht)
    return np.array(out)


def encode_angle(ht):
    """half-turns in (-1,1) -> two uint64 words."""
    v = round((ht % 2) * 2**128) % 2**128
    return v % 2**64, v >> 64


def write_angles(src, dst, angles):
    data = np.fromfile(src, dtype=np.uint8).copy()
    _, off, n = angle_packet_offset(src)
    assert len(angles) == n, (len(angles), n)
    for i, a in enumerate(angles):
        w0, w1 = encode_angle(float(a))
        struct.pack_into('<QQ', data, off + 16 * i, w0, w1)
    data.tofile(dst)


def reference(xs):
    """target, tol in 2^-64 turns for xs (beta=1, precision 24)."""
    t, o = [], []
    for x in xs.tolist():
        r = math.isqrt((1 << 130) // x)
        t.append(((r + 1) >> 1) & MASK64)
        o.append(min(r >> 25, 1 << 63))
    return np.array(t, dtype=np.uint64), np.array(o, dtype=np.uint64)


def half_turns_to_units(ht):
    n, d = ht.as_integer_ratio()
    return (((n << 64) // d + 1) >> 1) & MASK64


def sim_errors(kmb_path, xs, shots_per_x=1, batch_size=256, seed=0):
    """Simulate `shots_per_x` shots of each x. Returns (err_units signed int64,
    tol_units, theta_units) aligned per shot. theta wraps; err in (-.5,.5) turns."""
    circuit = km.Circuit.from_file(kmb_path)
    x_reg = km.array(list(circuit.register_data['x']))
    n = len(xs) * shots_per_x
    reps = np.repeat(xs, shots_per_x)
    pad = (-n) % batch_size
    if pad:
        reps = np.concatenate([reps, np.repeat(xs[:1], pad)])
    sim = km.Simulator(batch_size=batch_size, debug_print=False,
                       counting_operations=False, seed=seed)
    errs, tols, thetas = [], [], []
    tgt, tol = reference(reps.astype(np.uint64))
    for s in range(0, len(reps), batch_size):
        chunk = reps[s:s + batch_size]
        sim.clear_for_shot()
        bits = ((chunk.astype(np.uint64)[None, :] >> np.arange(22, dtype=np.uint64)[:, None]) & 1) == 1
        sim.write_across_shots(x_reg, bits)
        sim.do(circuit)
        th = np.array([half_turns_to_units(sim.read_shot_phase(i))
                       for i in range(batch_size)], dtype=np.uint64)
        e = (th - tgt[s:s + batch_size]).view(np.int64)
        errs.append(e); tols.append(tol[s:s + batch_size]); thetas.append(th)
    e = np.concatenate(errs)[:n]
    t = np.concatenate(tols)[:n]
    return e, t, np.concatenate(thetas)[:n]


def extract_basis(kmb_path, xs, delta=2**-16, shots_per_x=1, seed=0):
    """Returns s (n_shots x n_angles): coefficient of each angle in theta/2 units,
    plus base errors at current angles."""
    base = decode_angles(kmb_path)
    e0, tol, _ = sim_errors(kmb_path, xs, shots_per_x=shots_per_x, seed=seed)
    n_shots, n_ang = len(e0), len(base)
    S = np.zeros((n_shots, n_ang))
    import tempfile, os
    for j in range(n_ang):
        mod = base.copy(); mod[j] += delta
        with tempfile.NamedTemporaryFile(suffix='.kmb', delete=False) as f:
            write_angles(kmb_path, f.name, mod)
            ej, _, _ = sim_errors(f.name, xs, shots_per_x=shots_per_x, seed=seed)
        os.unlink(f.name)
        # d(err) = s_ij * delta/2 turns = s_ij * delta * 2^63 units
        d = (ej - e0).astype(np.float64)
        S[:, j] = d / (delta * 2**63)
    return S, e0, tol, base
