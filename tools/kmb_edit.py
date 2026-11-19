"""Read-modify-write for .kmb circuits.

parse_full() -> dict with header fields, raw register payloads, and op records:
    op = {'t': type_byte, 'q': (q0,q1,q2) tuple (None-padded),
          'bit': bit_operand or None, 'cond': cond_bit or None,
          'angle': 16-byte raw angle or None, 'tag': str or None}

write_kmb(recs...) re-emits a valid .kmb with recomputed header counts and
rebuilt SOA packets. Register packets are re-emitted verbatim.
"""
import struct
import numpy as np

MAGIC = bytes.fromhex('d750c7d5c329d326e3cc9f6834f2b8bf')


def parse_full(path):
    data = np.fromfile(path, dtype=np.uint8)
    assert bytes(data[:16]) == MAGIC
    off = 16
    ver, nq, nb, nreg = struct.unpack_from('<4I', data, off); off += 16
    (num_ops, n3, n2, n1, n1b, ncond, nang, ntags, ntagc) = struct.unpack_from('<9Q', data, off)
    off += 72
    types = None
    arrs = {}
    regs = []          # raw payloads for register packets
    angles = []        # raw 16-byte angle payloads
    tags = []          # decoded info-tag strings
    while off < len(data):
        ptype, plen = struct.unpack_from('<QQ', data, off); off += 16
        payload = data[off:off + plen]
        if ptype == 0:
            nlen, rlen = struct.unpack_from('<II', payload, 0)
            name = bytes(payload[8:8 + nlen]).decode(errors='replace')
            ids = struct.unpack_from(f'<{rlen}I', payload, 8 + nlen)
            regs.append((name, list(ids)))
        elif ptype == 1:
            types = np.frombuffer(payload, dtype=np.uint8).copy()
        elif 2 <= ptype <= 9:
            arrs[ptype] = np.frombuffer(payload, dtype=np.uint32).copy()
        elif ptype == 10:
            for i in range(plen // 16):
                angles.append(bytes(payload[16 * i:16 * i + 16]))
        elif ptype == 11:
            nt, = struct.unpack_from('<Q', payload, 0)
            ends = struct.unpack_from(f'<{nt}Q', payload, 8)
            chars = bytes(payload[8 + 8 * nt:])
            p = 0
            for e in ends:
                tags.append(chars[p:e].decode(errors='replace'))
                p = e
        off += plen

    i3 = i2 = i1 = ib = ic = ia = it = 0
    ops = []
    for k in range(num_ops):
        t = int(types[k])
        f3 = t & 0x30 == 0x30
        f2 = t & 0x30 == 0x20
        q = [None, None, None]
        if f3:
            q = [int(arrs[2][i3]), int(arrs[3][i3]), int(arrs[4][i3])]; i3 += 1
        elif f2:
            q = [int(arrs[5][i2]), int(arrs[6][i2]), None]; i2 += 1
        elif t & 0x10:
            q = [None, None, int(arrs[7][i1])]; i1 += 1
        bit = int(arrs[8][ib]) if t & 0x40 else None
        if t & 0x40: ib += 1
        cond = int(arrs[9][ic]) if t & 0x80 else None
        if t & 0x80: ic += 1
        ang = None
        if (t & 0x7F) == 0x13:                     # Z_POW / Z_POW_IF
            ang = angles[ia]; ia += 1
        tag = None
        if t == 0x06:
            tag = tags[it]; it += 1
        ops.append({'t': t, 'q': tuple(q), 'bit': bit, 'cond': cond,
                    'angle': ang, 'tag': tag})
    assert i3 == n3 and i2 == n2 and i1 == n1
    assert ib == n1b and ic == ncond and ia == nang and it == ntags
    return {'ver': ver, 'nq': nq, 'nb': nb, 'regs': regs, 'ops': ops}


def write_kmb(parsed, path, ops=None):
    ops = parsed['ops'] if ops is None else ops
    types = np.array([o['t'] for o in ops], dtype=np.uint8)
    f3 = (types & 0x30) == 0x30
    f2 = (types & 0x30) == 0x20
    f1 = (types & 0x10) != 0
    f1_only = f1 & ~f3 & ~f2
    fb = (types & 0x40) != 0
    fc = (types & 0x80) != 0

    def u32(vals):
        return np.array(vals, dtype=np.uint32).tobytes()

    packets = []
    for name, ids in parsed['regs']:
        nb_ = name.encode()
        pl = struct.pack('<II', len(nb_), len(ids)) + nb_ + u32(ids)
        packets.append((0, pl))
    packets.append((1, types.tobytes()))
    if f3.any():
        packets.append((2, u32([o['q'][0] for o, m in zip(ops, f3) if m])))
        packets.append((3, u32([o['q'][1] for o, m in zip(ops, f3) if m])))
        packets.append((4, u32([o['q'][2] for o, m in zip(ops, f3) if m])))
    if f2.any():
        packets.append((5, u32([o['q'][0] for o, m in zip(ops, f2) if m])))
        packets.append((6, u32([o['q'][1] for o, m in zip(ops, f2) if m])))
    if f1_only.any():
        packets.append((7, u32([o['q'][2] for o, m in zip(ops, f1_only) if m])))
    if fb.any():
        packets.append((8, u32([o['bit'] for o, m in zip(ops, fb) if m])))
    if fc.any():
        packets.append((9, u32([o['cond'] for o, m in zip(ops, fc) if m])))
    ang = [o['angle'] for o in ops if o['angle'] is not None]
    if ang:
        packets.append((10, b''.join(ang)))
    tlist = [o['tag'] for o in ops if o['tag'] is not None]
    chars = ''.join(tlist).encode()
    ends = np.cumsum([len(t.encode()) for t in tlist]).astype(np.uint64)
    pl = struct.pack('<Q', len(tlist)) + ends.tobytes() + chars
    packets.append((11, pl))

    hdr = MAGIC + struct.pack('<4I', parsed['ver'], parsed['nq'], parsed['nb'],
                              len(parsed['regs']))
    hdr += struct.pack('<9Q', len(ops), int(f3.sum()), int(f2.sum()),
                       int(f1_only.sum()), int(fb.sum()), int(fc.sum()),
                       len(ang), len(tlist),
                       sum(len(t.encode()) for t in tlist))
    with open(path, 'wb') as f:
        f.write(hdr)
        for pt, pl in packets:
            f.write(struct.pack('<QQ', pt, len(pl)))
            f.write(pl)


if __name__ == '__main__':
    import sys
    p = parse_full(sys.argv[1])
    print(len(p['ops']), 'ops,', p['nq'], 'qubits,', p['nb'], 'bits,',
          len(p['regs']), 'registers')
