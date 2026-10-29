"""Own DQI Reed-Solomon MBUC syndrome decoder (kickmix circuit) - v1.

Contract: named registers S0..S{n-1} (quantum, b bits) hold the syndrome
S_i = sum_j e_j alpha^{(i+1)j}; named classical E0..E{m-1} hold MBUC bits M_j.
Must restore S, return all other qubits to |0>, and apply phase
(-1)^{sum_j popcount(e_j & M_j)}. No Z_POW.

Reversible BM (data-oblivious, role-indirection via flag-controlled CSWAP):
  state: C bank (36 x b), Bx bank (36 x b), L (7q), b (bq), D bank (n x b),
         F bank (n flags), coef/ib scratch.
  per iter r:
    D[r]  ^= S_r ^ sum_i C_i S_{r-i}            (fresh slice: clean write)
    F[r]  ^= (d!=0) && (2L<=r)                (fresh flag slice)
    coef  ^= d * inv(b)                        (init inverse scratch, mul, del)
    C_{i+1} ^= coef * Bx_i                     (XOR into locator, in-place)
    if flag: swap roles via CSWAP(C bank <-> Bx bank)
             then the 'Bx' formula reads must select which bank held Bx_old.
    -> implemented as: C-update BEFORE swap; swap handles role change; then a
       conditional shift applies x*Bx when NOT flag.
    Correct per-iter sequence (mirrors bm_oblivious):
      a) d, flag, coef
      b) Bx_new(i) goes to a fresh target: BxT_i ^= flag*C_i ^ (1-flag)*Bx_{i-1}
         -- OUT-OF-PLACE write into the OTHER of two alternating banks (double
         buffer); the stale bank is cleaned by backward replay at BM end.
      c) L := flag ? r+1-L : L  via CSWAP(L, r+1-L scratch)
      d) b := flag ? d : b via swap(b, D[r]) ... or controlled copy into scratch.
      e) C_{i+1} ^= coef * Bx_i  (Bx_i = OLD bank, still live)
  BACKWARD replay r=n-1..0 to clean D/F/L-scratch/stale-Bx banks.

Chien+Forney: 3 rotating banks (sig, Om, sig') rotated by const alpha^{-i}
  maps (CNOTs only); per position: eval->acc, is_root, e=Om*inv(sig'),
  Z(parity(e&M_j) & is_root), inline uncompute, then rotate.

Classical oracle: dqi/model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import kickmix as km

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model  # noqa: E402

FIELD = km.GF2Field(model.B)
PRIM = FIELD.primitive_element
N, M, B, LMAX = model.N, model.M, model.B, 35
NC = LMAX + 1


def _nonzero(builder, reg, target):
    """target ^= (reg != 0) using flip_if_equal then X."""
    builder.flip_if_equal(reg, 0, target=target)
    builder.x([target])


def _lt_const(builder, reg, c, target):
    """target ^= (reg < c) for small-int reg vs constant c."""
    if c <= 0:
        return
    builder.flip_if_less_than(reg, c, target=target)


def _const_pow_map(builder, reg, pow_exp, field=FIELD):
    """reg *= alpha^{pow_exp} in place: a constant GF multiply = linear CNOT map.

    Implemented via ixor_gf2_mul into a fresh buffer then copy back? Simpler:
    use gf2_imul (constant factor variant) if available, else explicit matrix.
    """
    # Constant multiply is a GF(2)-linear map on the 12 bits: build the matrix.
    el = 0
    for i, bit in enumerate(reg):
        if (pow_exp >> i) & 1:
            el |= 1 << i
    # Compute matrix M s.t. out = M * in over GF(2); apply CNOTs per set bit.
    n = len(reg)
    M = np.zeros((n, n), dtype=np.uint8)
    for col in range(n):
        v = field.mul(1 << col, el) if hasattr(field, 'mul') else 0
        for row in range(n):
            M[row, col] = (v >> row) & 1
    # Apply in-place: out_bit_r ^= in_bit_c for M[r,c]=1 -- need in-place order care.
    # Use a scratch-free method: since map is a permutation matrix? Not always.
    # Simplest correct: compute into temp then swap back.
    tmp = builder.alloc_qubits(n)
    for r in range(n):
        for c in range(n):
            if M[r, c]:
                builder.cx(reg[c], tmp[r])
    builder.swap(reg, tmp)
    builder.free(tmp)


def build_decoder(*, n=None, m=None) -> km.Circuit:
    n = n or N
    m = m or M
    builder = km.CircuitBuilder()
    S = [builder.create_quantum_register(B, name=f'S{i}') for i in range(n)]
    E = [builder.create_classical_register(B, name=f'E{j}') for j in range(m)]

    # BM state
    C = [builder.alloc_qubits(B) for _ in range(NC)]
    BxA = [builder.alloc_qubits(B) for _ in range(NC)]
    BxB = [builder.alloc_qubits(B) for _ in range(NC)]   # alternating target bank
    L = builder.alloc_qubits(7)
    breg = builder.alloc_qubits(B)
    D = [builder.alloc_qubits(B) for _ in range(n)]
    F = [builder.alloc_qubits(1) for _ in range(n)]
    Lscr = [builder.alloc_qubits(7) for _ in range(n)]   # per-iter L scratch for CSWAP
    builder.x([C[0][0], BxA[0][0], breg[0]])

    Bx_cur, Bx_nxt = BxA, BxB
    for r in range(n):
        d = D[r]
        builder.cx(S[r], d)
        for i in range(1, min(NC, r + 1)):
            builder.ixor_gf2_mul(C[i], S[r - i], target=d, field=FIELD)

        # flag = (d!=0) && (2L<=r)
        flag = F[r]
        _nonzero(builder, d, flag[0])
        t2 = builder.alloc_qubits(1)[0]
        _lt_const(builder, L, (r + 1) // 2 + 1, t2)
        builder.init_and(flag[0], t2, flag[0])
        builder.flip_if_less_than(L, (r + 1) // 2 + 1, target=t2)
        builder.free(t2)

        # coef = d * inv(b)
        binv = builder.alloc_qubits(B)
        builder.ixor_gf2_inverse(breg, target=binv, field=FIELD)
        coef = builder.alloc_qubits(B)
        builder.ixor_gf2_mul(d, binv, target=coef, field=FIELD)
        builder.del_gf2_inverse(breg, binv, field=FIELD)
        builder.free(binv)

        # BxT_i ^= flag*C_i ^ (1-flag)*Bx_{i-1}   (fresh write into Bx_nxt)
        nflag = builder.alloc_qubits(1)[0]
        builder.x([nflag])
        builder.cx(flag[0], nflag)  # nflag = ~flag
        for i in range(NC):
            for bit in range(B):
                builder.ccx(flag[0], C[i][bit], Bx_nxt[i][bit])
                if i > 0:
                    builder.ccx(nflag, Bx_cur[i - 1][bit], Bx_nxt[i][bit])
        builder.cx(flag[0], nflag)
        builder.x([nflag])
        builder.free(nflag)

        # C_{i+1} ^= coef * Bx_i  (old Bx bank)
        for i in range(NC - 1):
            builder.ixor_gf2_mul(coef, Bx_cur[i], target=C[i + 1], field=FIELD)

        # uncompute coef (self-inverse re-apply)
        for i in range(NC - 1):
            pass
        builder.ixor_gf2_mul(d, coef, target=km.array([]), field=FIELD)  # placeholder
        # (uncompute coef by multiplying d*inv(b) again after C update)
        builder.free(coef)

        # L := flag ? r+1-L : L  via CSWAP with scratch
        ls = Lscr[r]
        # ls := r+1-L   (XOR const r+1 then subtract L? compute into scratch)
        builder.x([ls[k] for k in range(7) if ((r + 1) >> k) & 1])
        builder.isub(km.array(L), target=km.array(ls))  # ls = r+1 - L
        for k in range(7):
            builder.cswap(flag[0], L[k], ls[k])
        # uncompute ls: ls now holds flag?L_old:(r+1-L); recompute to clear via
        # storing for backward replay -> leave as garbage for replay.
        # b := flag ? d : b  via CSWAP(b, D[r]) -- moves d into b, old b into D[r]
        # (D[r] still referenced for C update already done; swap is fine)
        for k in range(B):
            builder.cswap(flag[0], breg[k], d[k])

        Bx_cur, Bx_nxt = Bx_nxt, Bx_cur
    # Locator now in C (sigma coefficients). CHIEN + FORNEY + cleanup: TODO.
    return builder.finish_circuit()


if __name__ == '__main__':
    c = build_decoder(n=4, m=8)
    print(c.num_qubits, 'qubits')
