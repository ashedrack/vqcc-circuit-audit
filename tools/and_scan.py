"""AND-retention scanner for .kmb circuits.

Detects two patterns around repeated CCX(a,b) computations:

Pattern A (park):  CCX(a,b,t1)@i ... CCX(a,b,t2)@j, a,b undisturbed, and a
    spare qubit w provably |0> throughout -> compute once into w, fan out
    with CX, erase w by HMR. Saves (uses-1) Toffolis.

Pattern B (keep):  CCX(a,b,t)@i ... [t only read / HMR-erased] ... CCX(a,b,t)@j
    with a,b undisturbed -> the AND is already parked in t; drop the HMR and
    the recomputation. Saves 1 Toffoli per skipped recompute.

Disturbance model (conservative): a qubit is "disturbed" when it is a WRITE
target - 1Q writes (X, R, Z_POW conservatively, HMR), either operand of
SWAP/CSWAP, CX target, or any operand of CCX/CCZ where it can be written.
Control reads do not disturb. Z/CZ/Z_POW are phase-only and never disturb
basis values; we still treat Z_POW conservatively as disturbing.

Usage: python and_scan.py FILE.kmb [--top N]
"""
import sys, struct
import numpy as np

OP_Z, OP_X, OP_R, OP_ZPOW = 0x10, 0x11, 0x12, 0x13
OP_HMR = 0x50
OP_CZ, OP_CX, OP_SWAP = 0x20, 0x21, 0x22
OP_CCZ, OP_CCX, OP_CSWAP = 0x30, 0x31, 0x32

WRITE_1Q = {OP_X, OP_R, OP_HMR}          # Z/Z_POW/CZ/CCZ are phase-only


def parse_kmb(path):
    """Parse a .kmb file. Returns ops array (num_ops,5): type,a,b,tgt,cond."""
    data = np.fromfile(path, dtype=np.uint8)
    assert bytes(data[:16]) == bytes.fromhex('d750c7d5c329d326e3cc9f6834f2b8bf')
    off = 16
    ver, nq, nb, nreg = struct.unpack_from('<4I', data, off); off += 16
    counts = struct.unpack_from('<9Q', data, off)
    off += 72
    num_ops = counts[0]
    arrs = {}
    types = None
    while off < len(data):
        ptype, plen = struct.unpack_from('<QQ', data, off); off += 16
        payload = data[off:off + plen]
        if ptype == 1:
            types = np.frombuffer(payload, dtype=np.uint8).copy()
        elif 2 <= ptype <= 9:
            arrs[ptype] = np.frombuffer(payload, dtype=np.uint32).copy()
        off += plen
    assert len(types) == num_ops

    # Vectorized SOA -> AOS reconstruction.
    has3 = (types & 0x30) == 0x30
    has2 = (types & 0x30) == 0x20
    has1 = (types & 0x10) != 0
    hasc = (types & 0x80) != 0
    ops = np.full((num_ops, 5), -1, dtype=np.int64)
    ops[:, 0] = types.astype(np.int64)
    ops[has3, 1] = arrs[2]; ops[has3, 2] = arrs[3]; ops[has3, 3] = arrs[4]
    ops[has2, 1] = arrs[5]; ops[has2, 3] = arrs[6]
    ops[has1 & ~has3 & ~has2, 3] = arrs[7]
    ops[hasc, 4] = arrs[9]
    return ops, nq, nb


def build_disturbances(ops, nq):
    """Per-qubit sorted arrays of op indices that write the qubit."""
    fam = ops[:, 0] & 0x7F
    a, b, tgt = ops[:, 1], ops[:, 2], ops[:, 3]
    qs, ks = [], []
    k = np.arange(len(ops))
    # 3-qubit gates: CCX writes only its target; CCZ is phase-only;
    # CSWAP swaps its two non-control operands.
    m = fam == OP_CCX
    qs.append(tgt[m & (tgt >= 0)]); ks.append(k[m & (tgt >= 0)])
    m = fam == OP_CSWAP
    for col in (b, tgt):
        mm = m & (col >= 0)
        qs.append(col[mm]); ks.append(k[mm])
    # 2-qubit: SWAP writes both; CX writes tgt; CZ phase-only -> skip
    m = fam == OP_SWAP
    for col in (a, tgt):
        mm = m & (col >= 0)
        qs.append(col[mm]); ks.append(k[mm])
    m = fam == OP_CX
    mm = m & (tgt >= 0)
    qs.append(tgt[mm]); ks.append(k[mm])
    # 1-qubit writes
    m = np.isin(ops[:, 0], list(WRITE_1Q) + [w | 0x80 for w in WRITE_1Q])
    mm = m & (tgt >= 0)
    qs.append(tgt[mm]); ks.append(k[mm])

    allq = np.concatenate(qs); allk = np.concatenate(ks)
    order = np.lexsort((allk, allq))
    allq, allk = allq[order], allk[order]
    bounds = np.searchsorted(allq, np.arange(nq + 1))
    return [allk[bounds[q]:bounds[q + 1]] for q in range(nq)]


def scan(ops, nq):
    disturb = build_disturbances(ops, nq)
    fam = ops[:, 0] & 0x7F
    is_ccx = (fam == OP_CCX) & (ops[:, 4] < 0)          # unconditional CCX
    is_reset = np.isin(ops[:, 0], [OP_R, OP_HMR])       # unconditional only
    resets = np.where(is_reset & (ops[:, 3] >= 0), ops[:, 3], -1)

    # qubit -> sorted indices of unconditional HMR/R
    reset_of = [[] for _ in range(nq)]
    for k in np.where(is_reset)[0]:
        t = ops[k, 3]
        if t >= 0:
            reset_of[t].append(k)

    def first_disturb_in(qq, lo, hi):
        d = disturb[qq]
        i = np.searchsorted(d, lo)
        return d[i] if i < len(d) and d[i] < hi else None

    def last_disturb_before(qq, k):
        d = disturb[qq]
        i = np.searchsorted(d, k)
        return d[i - 1] if i > 0 else None

    def provably_zero_at(qq, k):
        """True if qq is provably |0> just after op k."""
        d = disturb[qq]
        i = np.searchsorted(d, k, 'right')
        if i == 0:
            return True                     # never written before k
        prev = d[i - 1]
        return ops[prev, 0] in (OP_R, OP_HMR)

    def free_qubit(a_, b_, t_, lo, hi):
        """qubit provably |0> and untouched on (lo,hi), not in {a,b,t}."""
        for w in range(nq):
            if w in (a_, b_, t_):
                continue
            d = disturb[w]
            i = np.searchsorted(d, lo, 'right')
            if i < len(d) and d[i] < hi:
                continue                    # written inside window
            if i == 0:
                return w                    # never written -> |0>
            if ops[d[i - 1], 0] in (OP_R, OP_HMR):
                return w                    # reset before window
        return None

    hits_a, hits_b = [], []
    diag = {'ccx': int(is_ccx.sum()), 'ccx_if': int(((fam == OP_CCX) & (ops[:, 4] >= 0)).sum()),
            'repeat_pairs': 0, 'windows': 0, 'ctrl_ok': 0, 'tgt_dist': 0, 'no_free': 0}
    # group unconditional CCX by (a,b) pair
    idx = np.where(is_ccx)[0]
    pair_key = ops[idx, 1] * nq + ops[idx, 2]
    order = np.argsort(pair_key, kind='stable')
    idx, pair_key = idx[order], pair_key[order]
    bounds = np.searchsorted(pair_key, np.unique(pair_key))
    groups = np.split(idx, bounds[1:])

    for locs in groups:
        if len(locs) < 2:
            continue
        diag['repeat_pairs'] += 1
        a_, b_ = int(ops[locs[0], 1]), int(ops[locs[0], 2])
        for i, j in zip(locs[:-1], locs[1:]):
            diag['windows'] += 1
            if first_disturb_in(a_, i + 1, j) is not None: continue
            if first_disturb_in(b_, i + 1, j) is not None: continue
            diag['ctrl_ok'] += 1
            t1, t2 = int(ops[i, 3]), int(ops[j, 3])
            # disturbances of t1 / t2 inside the window
            dt = first_disturb_in(t1, i + 1, j)
            if t1 == t2:
                # Pattern B: parked AND erased then recomputed?
                d_all = disturb[t1]
                lo_i = np.searchsorted(d_all, i + 1)
                inside = d_all[lo_i:np.searchsorted(d_all, j)]
                if len(inside) == 1 and ops[inside[0], 0] == OP_HMR:
                    hits_b.append((i, inside[0], j, a_, b_, t1))
            elif dt is None:
                w = free_qubit(a_, b_, t2, i, j)
                if w is not None:
                    hits_a.append((i, j, a_, b_, t1, t2, w))
                else:
                    diag['no_free'] += 1
            else:
                diag['tgt_dist'] += 1
    return hits_a, hits_b, diag


if __name__ == '__main__':
    path = sys.argv[1]
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    print(f'parsing {path} ...', file=sys.stderr)
    ops, nq, nb = parse_kmb(path)
    print(f'{len(ops)} ops, {nq} qubits', file=sys.stderr)
    ha, hb, diag = scan(ops, nq)
    print('diag:', diag)
    print(f'Pattern B (keep parked AND, drop HMR+recompute): {len(hb)}')
    for i, h, j, a, b, t in hb[:top]:
        print(f'  CCX(q{a},q{b})->q{t} @{i}, HMR@{h}, recompute @{j}')
    print(f'Pattern A (park in spare qubit): {len(ha)}')
    for i, j, a, b, t1, t2, w in ha[:top]:
        print(f'  ops {i}..{j}  CCX(q{a},q{b}) q{t1}->q{t2}   free q{w}')
