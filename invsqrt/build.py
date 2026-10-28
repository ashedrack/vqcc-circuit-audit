"""invsqrt: lookup + piecewise-polynomial circuit for phasing by 1/sqrt(x).

Pipeline (each stage is uncomputed in reverse after phasing):
1. Normalize by binary search: x_norm = x 4^s, M = x_norm / 2^20 in [1, 4), s in 4 bits.
2. Lookup: address a = top k bits of x_norm -> C_0..C_n (per-interval degree-n polynomial,
   coefficient magnitudes; the signs of 1/sqrt alternate).
3. Horner on the low d = 22 - k bits D of x_norm (delta = D / 2^d):
     V_n = C_n,  V_j = C_j - T(delta V_{j+1}),  Y = V_0,  Y / 2^F ~ 1 / sqrt(M).
   Each accumulator is split (see `Split`) so the row subtractions only touch its low part.
4. Denormalize: Y 2^s by a binary barrel shift, then one Z_POW per bit.

Usage: python invsqrt/build.py out.kmb [--k 9] [--frac 28] [--degree 2]
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import kickmix as km

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'data' / 'verifiers' / 'invsqrt'))
import nr_reference_circuit as ref  # noqa: E402

N_P = 10
WIDTH = 2 * N_P + 2
TOL = 2.0**-24
SHIFT_STAGES = (8, 4, 2, 1)  # Leading zero pairs s = 8 s3 + 4 s2 + 2 s1 + s0.


@dataclasses.dataclass(frozen=True)
class Params:
    k: int = 9  # Lookup address bits (top bits of x_norm).
    frac: int = 28  # Fractional bits F of Y (C_0, V_0).
    degree: int = 2
    guard: int = 0  # Low bits of Y that are computed but not phased (nor denormalized).
    fdelta: tuple[int, ...] = ()  # F_j - F for j = 1..n (default 0): per-register precision.

    @property
    def d(self) -> int:
        return WIDTH - self.k

    @property
    def fracs(self) -> list[int]:
        fd = list(self.fdelta) + [0] * (self.degree - len(self.fdelta))
        return [self.frac] + [self.frac + e for e in fd]


def _row_shift(fracs: list[int], j: int, m: int) -> int:
    """Index offset of mult bit -> acc bit for the row of weight 2^-m in step j (V_j -= delta V_{j+1})."""
    return fracs[j] - fracs[j + 1] - m


def _rows(op: np.ndarray, D: np.ndarray, d: int, fracs: list[int], j: int) -> np.ndarray:
    """T(delta op) aligned to V_j: sum_i D_i (op shifted, low bits dropped), as the circuit computes."""
    acc = np.zeros(np.broadcast(op, D).shape, dtype=np.int64)
    for i in range(d):
        e = _row_shift(fracs, j, d - i)
        acc += ((D >> i) & 1) * (op << e if e >= 0 else op >> -e)
    return acc


def _horner(cs: list, D: np.ndarray, d: int, fracs: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Returns Y and the row sums P_j (V_j = C_j - P_j) for j = n-1..0."""
    v, ps = cs[-1], []
    for j in reversed(range(len(cs) - 1)):
        ps.append(_rows(v, D, d, fracs, j))
        v = cs[j] - ps[-1]
    return v, ps[::-1]


@dataclasses.dataclass
class Split:
    """Accumulator `V = C - P` stored as `high 2^wl + low` with `low = Lo - P` in `wl + 1` bits.

    `Lo = C mod 2^wl` (plus `2^wl` when needed) is at least `max P` of its interval, so the row
    subtractions never borrow out of `low`; one controlled increment then folds `low[wl]` into
    `high`, leaving `V = high 2^wl + low[:wl]` canonical (`low[wl]` is left as garbage).
    """
    wl: int
    wh: int
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def make(cls, c: np.ndarray, pmax: np.ndarray, used: np.ndarray) -> 'Split':
        wl = int(pmax[used].max()).bit_length()
        lo = c % (1 << wl)
        lo = np.where(lo < pmax, lo + (1 << wl), lo)
        lo[~used] = 0
        hi = (c - lo) >> wl
        assert (hi[used] >= 0).all()
        return cls(wl, int(hi.max() + 1).bit_length(), lo, hi)

    @property
    def width(self) -> int:
        return self.wl + 1 + self.wh

    def table(self) -> list[int]:
        return [int(l) | int(h) << (self.wl + 1) for l, h in zip(self.lo, self.hi)]


@dataclasses.dataclass
class Design:
    params: Params
    cs: list[np.ndarray]  # cs[j][a] = |coefficient of delta^j| 2^F (0 for unused a < 2^(k-2)).
    splits: list[Split] = dataclasses.field(default_factory=list)  # One per accumulator V_0..V_{n-1}.

    @property
    def top_width(self) -> int:
        return int(self.cs[-1].max()).bit_length()


def fit(p: Params) -> Design:
    k, d, F, n, fr = p.k, p.d, p.frac, p.degree, p.fracs
    na = 1 << k
    cs = [np.zeros(na, dtype=np.int64) for _ in range(n + 1)]
    D = np.arange(1 << d, dtype=np.int64)
    t = D / 2.0**d
    nodes = (1 - np.cos(np.pi * (2 * np.arange(n + 1) + 1) / (2 * n + 2))) / 2  # Chebyshev on [0, 1].
    used = np.arange(na) >= na // 4
    pmax = [np.zeros(na, dtype=np.int64) for _ in range(n)]
    for a in range(na // 4, na):
        f = lambda s: ((a + s) / 2.0**(k - 2))**-0.5  # noqa: E731
        q = np.polyfit(nodes, f(nodes), n)[::-1]  # q[j] multiplies t^j.
        assert all((q[j] > 0) == (j % 2 == 0) for j in range(n + 1))
        for j in range(n + 1):
            cs[j][a] = round(abs(q[j]) * 2**fr[j])
        exact = f(t) * 2**F
        for _ in range(3):  # Center the absolute error of the phased bits by shifting C_0.
            y, _ = _horner([c[a] for c in cs], D, d, fr)
            err = (y >> p.guard << p.guard) - exact
            cs[0][a] -= int(round((err.max() + err.min()) / 2))
        _, ps = _horner([c[a] for c in cs], D, d, fr)
        for j in range(n):
            pmax[j][a] = ps[j].max()
    return Design(p, cs, [Split.make(cs[j], pmax[j], used) for j in range(n)])


def model_error(des: Design) -> float:
    p = des.params
    a = np.repeat(np.arange(1 << (p.k - 2), 1 << p.k, dtype=np.int64), 1 << p.d)
    D = np.tile(np.arange(1 << p.d, dtype=np.int64), 3 << (p.k - 2))
    y, ps = _horner([c[a] for c in des.cs], D, p.d, p.fracs)
    for j, (pj, sp) in enumerate(zip(ps, des.splits)):
        low = sp.lo[a] - pj
        assert (low >= 0).all() and (low < 2 << sp.wl).all(), f'V_{j} low overflow'
    m = (a.astype(np.float64) * 2**p.d + D) / 2.0**(WIDTH - 2)
    return float(np.abs((y >> p.guard << p.guard) / 2.0**p.frac * np.sqrt(m) - 1).max())


def _normalize(b: km.CircuitBuilder, inverse: bool, x: km.array, sbits: list) -> None:
    """x := x 4^s with s the leading zero pairs, by binary search; sbits[i] is the bit of weight 2^i."""
    stages = list(zip(SHIFT_STAGES, reversed(sbits)))
    for pairs, flag in (reversed(stages) if inverse else stages):
        L = 2 * pairs
        top = x[WIDTH - L:]
        moves = range(WIDTH - L - 1, -1, -1)  # Top L bits are zero when flag is set.
        if inverse:
            for j in reversed(moves):
                b.cswap(flag, x[j], x[j + L])
        b.x(top)
        b.multi_controlled_x(top, flag)
        b.x(top)
        if not inverse:
            for j in moves:
                b.cswap(flag, x[j], x[j + L])


def _denormalize(b: km.CircuitBuilder, inverse: bool, sbits: list, y: km.array, occupied: int) -> None:
    """y := y 2^s, smallest shift first so each stage only moves the occupied bits."""
    stages, o = [], occupied
    for i, flag in enumerate(sbits):
        stages.append((1 << i, flag, o))
        o += 1 << i
    for shift, flag, o in (reversed(stages) if inverse else stages):
        moves = range(o - 1, -1, -1)
        for j in (reversed(moves) if inverse else moves):
            b.cswap(flag, y[j], y[j + shift])


def _lookup(b: km.CircuitBuilder, table: list[int], addr: km.array, target: km.array, lo: int) -> None:
    """target ^= table[addr] for addr in [lo, len(table)); other addresses must hold 0."""
    with b.unary_iteration(addr, range(lo, len(table))) as it:
        for a, match in it:
            bits = [target[i] for i in range(len(target)) if table[a] >> i & 1]
            if bits:
                b.cx(km.array([match] * len(bits)), km.array(bits))


def build(des: Design) -> km.Circuit:
    p = des.params
    k, d, F, n = p.k, p.d, p.frac, p.degree
    max_s = sum(SHIFT_STAGES)
    b = km.CircuitBuilder()
    x = b.create_quantum_register(WIDTH, name='x')
    sbits = list(b.alloc_qubits(len(SHIFT_STAGES)))
    accs = [b.alloc_qubits(sp.width) for sp in des.splits]
    top = b.alloc_qubits(des.top_width)
    lows = [acc[:sp.wl + 1] for acc, sp in zip(accs, des.splits)]
    highs = [acc[sp.wl + 1:] for acc, sp in zip(accs, des.splits)]
    canon = [km.array(list(lw[:sp.wl]) + list(h)) for lw, h, sp in zip(lows, highs, des.splits)] + [top]
    y = canon[0][p.guard:]  # Phased bits; bit j has weight 2^(j + guard - F) of 1/sqrt(M).
    addr, D = x[d:], [x[i] for i in range(d)]
    target = km.array([q for acc in accs for q in acc] + list(top))
    table = [0] * (1 << k)
    shift = 0
    for sp in des.splits:
        table = [t | v << shift for t, v in zip(table, sp.table())]
        shift += sp.width
    table = [t | int(v) << shift for t, v in zip(table, des.cs[-1])]
    lo = 1 << (k - 2)
    ytop = None

    def horner(inverse: bool) -> None:
        scratch = b.alloc_qubits(max(sp.wl for sp in des.splits) + 1)
        steps = [(j, lows[j], highs[j], canon[j + 1]) for j in reversed(range(n))]
        for step, acc, high, mult in (reversed(steps) if inverse else steps):
            if inverse:
                b.isub(1, target=high, control=acc[-1])
            rows = [(D[i], d - i) for i in range(d)]
            for control, m in (reversed(rows) if inverse else rows):
                e = _row_shift(p.fracs, step, m)
                terms = [(mult[jj], jj + e) for jj in range(len(mult)) if jj + e >= 0]
                if terms:
                    ref._add_row(b, inverse, control, terms, acc, scratch, subtract=True)
            if not inverse:
                b.iadd(1, target=high, control=acc[-1])
        b.free(scratch)

    def compute(inverse: bool) -> km.array:
        nonlocal ytop
        if not inverse:
            with b.mark('normalize'):
                _normalize(b, False, x, sbits)
            with b.mark('lookup'):
                _lookup(b, table, addr, target, lo)
            with b.mark('horner'):
                horner(False)
            ytop = b.alloc_qubits(max_s)
            y1 = km.array(list(y) + list(ytop))
            with b.mark('denormalize'):
                _denormalize(b, False, sbits, y1, len(y))
            return y1
        y1 = km.array(list(y) + list(ytop))
        with b.mark('denormalize'):
            _denormalize(b, True, sbits, y1, len(y))
        b.free(ytop)
        with b.mark('horner'):
            horner(True)
        with b.mark('lookup'):
            b.del_lookup(table=table, address=addr, target=target)
        with b.mark('normalize'):
            _normalize(b, True, x, sbits)
        return y1

    with b.mark('compute'):
        y1 = compute(False)
    with b.mark('phase'):
        fp = F - p.guard
        for j in range(min(len(y1), fp + N_P)):  # Bit j has weight 2^(j - fp - N_P) turns.
            b.z_pow(y1[j], Fraction(2) ** (j - fp - N_P + 1))
    with b.mark('uncompute'):
        compute(True)
    for reg in (km.array(sbits), *accs, top):
        b.free(reg)
    return b.finish_circuit()


def make(p: Params, verbose: bool = True) -> tuple[Design, float, km.Circuit]:
    des = fit(p)
    err = model_error(des)
    circuit = build(des)
    if verbose:
        widths = '/'.join(f'{sp.wl + 1}+{sp.wh}' for sp in des.splits) + f'/{des.top_width}'
        print(f'k={p.k} F={p.fracs} n={p.degree} g={p.guard} widths {widths}: err 2^{math.log2(err):.2f} '
              f'({err / TOL:.3f} tol), {circuit.operation_statistics().num_toffoli_like} Toffolis, '
              f'{circuit.num_qubits} qubits')
    return des, err, circuit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('output')
    ap.add_argument('--k', type=int, default=Params.k)
    ap.add_argument('--frac', type=int, default=Params.frac)
    ap.add_argument('--degree', type=int, default=Params.degree)
    ap.add_argument('--guard', type=int, default=Params.guard)
    a = ap.parse_args()
    _, _, circuit = make(Params(k=a.k, frac=a.frac, degree=a.degree, guard=a.guard))
    circuit.to_file(a.output)


if __name__ == '__main__':
    main()
