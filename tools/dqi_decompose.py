"""Decompose the dqi circuit's executed-Toffoli cost by region.

Goal: locate where the ~366k executed Toffolis are spent, and whether the
GF(2^12) multiplier blocks use schoolbook (144 AND/mul) or Karatsuba
(~108 AND/mul) structure. Phase-1 measurement for plan_31.md.

Outputs:
  - op histograms per 250k-op window (CCX/CCZ/CSWAP density)
  - register-name activity map (which register group is busiest where)
  - detection of repeated multiplier motifs: runs of dense CCX confined
    to a small qubit support (GF(2^12) arithmetic signature)

Usage:
    ../.venv/bin/python tools/dqi_decompose.py public/ozer28/data/dqi.kmb
"""
import sys, struct
import numpy as np
from and_scan import parse_kmb, OP_CCX, OP_CCZ, OP_CSWAP


def read_registers(path):
    """Read only register-name packets (type 0) — avoids the slow
    full op-record parse on 57M-op files."""
    regs = {}
    with open(path, 'rb') as f:
        f.seek(104)                       # magic + ver/nq/nb/nreg + 72B counts
        while True:
            hdr = f.read(16)
            if len(hdr) < 16:
                break
            ptype, plen = struct.unpack('<QQ', hdr)
            if ptype == 0:
                payload = f.read(plen)
                nlen, rlen = struct.unpack_from('<II', payload, 0)
                name = payload[8:8 + nlen].decode(errors='replace')
                ids = struct.unpack_from(f'<{rlen}I', payload, 8 + nlen)
                regs[name] = list(ids)
            elif ptype == 1:
                break                     # registers precede op packets
            else:
                f.seek(plen, 1)
    return regs


def windows(ops, width=250_000):
    fam = ops[:, 0] & 0x7F
    w = np.arange(len(ops)) // width
    K = w.max() + 1
    return {name: np.bincount(w[fam == code], minlength=K)
            for code, name in [(OP_CCX, 'CCX'), (OP_CCZ, 'CCZ'), (OP_CSWAP, 'CSWAP')]}


def register_map(path, ops, nq):
    regs = read_registers(path)
    fam = ops[:, 0] & 0x7F
    ccx = np.where(fam == OP_CCX)[0]
    targets = np.clip(ops[ccx, 3], 0, nq - 1)
    counts = np.bincount(targets, minlength=nq)
    table = sorted(((name, int(counts[np.array([q & 0x7FFFFFFF for q in qs if (q & 0x7FFFFFFF) < nq], dtype=int)].sum()) if any((q & 0x7FFFFFFF) < nq for q in qs) else 0, len(qs))
                    for name, qs in regs.items()), key=lambda t: -t[1])
    return table


def multiplier_motifs(ops, nq, min_ccx=80, max_qubits=18):
    """Runs of consecutive CCX ops confined to <=max_qubits distinct
    qubits — signature of a GF(2^12) multiplier cell."""
    fam = ops[:, 0] & 0x7F
    ccx = np.where(fam == OP_CCX)[0]
    res, i = [], 0
    while i < len(ccx):
        j, qs = i, set()
        while j < len(ccx):
            block = ops[ccx[j], 1:4]
            new_qs = set(int(x) for x in block if x >= 0)
            if len(qs | new_qs) > max_qubits:
                break
            qs |= new_qs
            j += 1
        if j - i >= min_ccx:
            res.append((int(ccx[i]), int(ccx[j - 1]), sorted(qs), j - i))
        i = max(j, i + 1)
    return res


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'public/ozer28/data/dqi.kmb'
    print('parsing', path)
    ops, nq, nb = parse_kmb(path)
    print('%s ops, %d qubits, %d bits' % (f'{len(ops):,}', nq, nb))

    w = windows(ops)
    K = len(w['CCX'])
    print('\nTop CCX-dense 250k-op windows:')
    for k in np.argsort(w['CCX'])[::-1][:12]:
        print('  window %4d: CCX %6d  CCZ %6d  CSWAP %5d'
              % (k, w['CCX'][k], w['CCZ'][k], w['CSWAP'][k]))
    print('  total static CCX ops:', w['CCX'].sum())

    print('\nTop registers by CCX target activity:')
    for name, c, size in register_map(path, ops, nq)[:20]:
        print('  %-8s %8d CCX (%d qubits)' % (name, c, size))

    print('\nMultiplier-motif scan (dense CCX on <=18 qubits):')
    motifs = multiplier_motifs(ops, nq)
    for s, e, qs, c in motifs[:20]:
        print('  ops %8d..%8d  %3d CCX on %d qubits %s' % (s, e, c, len(qs), qs[:18]))
    print('  total candidate cells:', len(motifs))
