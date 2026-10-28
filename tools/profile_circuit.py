"""Per-stage Toffoli/rotation/qubit profile of a kickmix circuit, using its @enter/@exit INFO_TAGs.

Usage: python tools/profile_circuit.py circuit.kmb [--depth 2]
"""
from __future__ import annotations

import argparse
import collections
import io
import re

import kickmix as km

TOFFOLI_LIKE = {'CCX', 'CCZ', 'CSWAP'}


def profile(circuit: km.Circuit, depth: int) -> None:
    buf = io.StringIO()
    circuit.to_file(buf, format='kmx')
    stack: list[str] = []
    tof = collections.Counter()
    rot = collections.Counter()
    for line in buf.getvalue().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        name = line.split()[0]
        if name == 'INFO_TAG':
            m = re.search(r'"@(enter|exit):([^"]*)"', line)
            if m:
                if m.group(1) == 'enter':
                    stack.append(m.group(2))
                else:
                    stack.pop()
            continue
        key = '/'.join(stack[:depth]) or '(top)'
        if name in TOFFOLI_LIKE:
            tof[key] += 1
        elif name == 'Z_POW':
            rot[key] += 1
    total = sum(tof.values())
    print(f'qubits={circuit.num_qubits} toffoli_like={total} z_pow={sum(rot.values())}')
    for key in sorted(set(tof) | set(rot), key=lambda k: -tof[k]):
        print(f'  {tof[key]:7d} ({100 * tof[key] / max(total, 1):5.1f}%)  zpow={rot[key]:3d}  {key}')
    for name, reg in circuit.register_data.items():
        print(f'  register {name}: {len(reg)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('circuit')
    ap.add_argument('--depth', type=int, default=2)
    a = ap.parse_args()
    profile(km.Circuit.from_file(a.circuit), a.depth)
