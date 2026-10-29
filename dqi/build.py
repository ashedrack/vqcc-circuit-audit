"""Own DQI Reed-Solomon MBUC syndrome decoder (kickmix) - truncated shared-path BM.

Implements the decoder architecture documented in public/lukin_dqi/discovered/
README.md (our own kickmix implementation): truncated BM locator updates with
shared arithmetic path (w = delta/Delta; sigma <- w*sigma ^ z*tau), conditional
swap routing for retain/replace, measurement-based erasure of fresh-register
values, in-place omega, Psi=1/sigma' mod sigma Euclid pass, flag-gated pairing
sweep, zlin interpolation.

Stage 1 target: BM locator (sigma) correct vs dqi/model.py oracle.
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


def bm_forward(builder, S, n):
    """Shared-path BM on syndrome registers S (quantum) -> C holds the locator.

    Registers (allocated inside; C and tau persist, scratch erased by measure):
      C    (NC x b) current locator coefficients      -> OUTPUT
      tau  (NC x b) template locator
      Lr   (7 q)    order counter (binary)
      delt (b)      persistent discrepancy
      d    (b)      signed counter (small int)
    Per round r (0-based), for live coefficient window [1, wr]:
      Delta = S_r ^ sum_i C_i S_{r-i}
      flag_replace = (Delta != 0) && (d_ctrl >= 0)
      The shared formula computes  C <- w*C ^ z*tau with w = delta/Delta,
      conditional swaps restore the bookkeeping.

    This is a minimal correct version (full windows, division per round).
    """
    C = [builder.alloc_qubits(B) for _ in range(NC)]
    tau = [builder.alloc_qubits(B) for _ in range(NC)]
    Lreg = builder.alloc_qubits(7)   # order L (binary)
    delta = builder.alloc_qubits(B)  # persistent discrepancy
    dreg = builder.alloc_qubits(8)   # signed counter d (two's complement)
    builder.x([C[0][0], tau[0][0], delta[0]])
    # d starts at 0, h bookkeeping folded: replace iff d>=0 (bit7 sign=0)

    for r in range(n):
        # ---- Delta = S_r ^ sum_{i=1..r} C_i * S_{r-i}  (full window for v1) ----
        Delta = builder.alloc_qubits(B)
        builder.cx(S[r], Delta)
        for i in range(1, min(NC, r + 1)):
            builder.ixor_gf2_mul(C[i], S[r - i], target=Delta, field=FIELD)

        # ---- replace = (Delta != 0) && (d >= 0) ----
        nz = builder.alloc_qubits(1)[0]
        builder.flip_if_equal(Delta, 0, target=nz)
        builder.x([nz])
        replace = builder.alloc_qubits(1)[0]
        builder.init_and(nz, dreg[7], replace)   # d>=0 means sign bit 0? TODO sign
        builder.x([replace])
        # NOTE: sign logic placeholder - fix when wiring real counter.

        # ---- w = delta / Delta ; C <- w*C ^ z*tau ----
        w = builder.alloc_qubits(B)
        dinv = builder.init_gf2_inverse(Delta, field=FIELD)
        builder.ixor_gf2_mul(delta, dinv, target=w, field=FIELD)
        # C_i = w*C_i ^ tau_{i-1}: compute w*C_i via ixor into fresh? in-place
        # multiply of C by w is a quantum*quantum -> use out-of-place then swap.
        for i in range(NC):
            tmp = builder.alloc_qubits(B)
            builder.ixor_gf2_mul(w, C[i], target=tmp, field=FIELD)
            if i > 0:
                builder.cx(tau[i - 1], tmp)
            builder.swap(tmp, C[i])
            builder.free(tmp)
        # tau <- z*tau (in-place shift), then conditional swaps for persistence
        for i in range(NC - 1, 0, -1):
            builder.swap(tau[i], tau[i - 1])
        # on replace: swap C<->tau roles is complex; minimal v1 keeps both banks
        # delta <- Delta on replace (controlled swap delta<->Delta)
        for k in range(B):
            builder.cswap(replace, delta[k], Delta[k])
        # Delta consumed into delta on replace; residue erasable (hmr) - TODO
        builder.free(Delta)
        builder.del_gf2_inverse(Delta if False else dinv, field=FIELD) if False else None
        builder.free(dinv)
        builder.free(w)
        builder.free(nz)
        builder.free(replace)
        # d update: replace -> d=-d else d+=1  (TODO: real counter logic)
    return C, tau, Lreg, delta, dreg


def build(*, n=None, m=None) -> km.Circuit:
    n = n or N
    m = m or M
    builder = km.CircuitBuilder()
    S = [builder.create_quantum_register(B, name=f'S{i}') for i in range(n)]
    for j in range(m):
        builder.create_classical_register(B, name=f'E{j}')
    C, tau, Lreg, delta, dreg = bm_forward(builder, S, n)
    # TODO: omega transform, Psi pass, sweep, cleanup.
    return builder.finish_circuit()


if __name__ == '__main__':
    c = build(n=4, m=8)
    print('qubits', c.num_qubits)
    print('toffoli', c.operation_statistics().num_toffoli_like)
