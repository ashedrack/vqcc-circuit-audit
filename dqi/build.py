"""Own DQI Reed-Solomon MBUC syndrome decoder (kickmix) - shared-path BM.

Architecture (our own kickmix implementation of the truncated-BM + Psi-route
decoder described in public/lukin_dqi/discovered/README.md):

  BM stage (per round r, all reversible):
    Delta = sum_{i=0}^{35} sig_i * S_{r-i}          [fresh reg, i=0 included]
    flag  = (Delta != 0) && (2L <= r)
    on !flag: tau <- z*tau (physical coeff shift)
    on  flag: swap(sig<->tau), swap(DP<->DC), swap(L<->(r+1-L) scratch)
    ratio = DC / DP   [DC=Delta on !flag, delta_old on flag -> auto-consistent]
    sig  += ratio * tau                              [XOR-accumulate; Delta=0
              gives ratio=0 via inverse(0)=0 -> no-op for free]
    (L,d metadata + garbage -> replay bank)

  Chien+Forney stage: see model.py notes / discovered README (Psi route).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import kickmix as km

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model  # noqa: E402

FIELD = km.GF2Field(model.B)
N, M, B, LMAX = model.N, model.M, model.B, 35
NC = LMAX + 1


def _is_nonzero(builder, reg, target):
    builder.flip_if_equal(reg, 0, target=target)
    builder.x([target])


def _shift_poly_left(builder, bank, control=None):
    """bank <- z*bank (coeffs shift up: new[i]=old[i-1], new[0]=0, top drops)."""
    if control is None:
        for i in range(len(bank) - 1, 0, -1):
            builder.swap(bank[i], bank[i - 1])
        builder.reset(bank[0])
        return
    # conditional: rotate then conditionally clear bank[0]
    for i in range(len(bank) - 1, 0, -1):
        for q_i, q_j in zip(bank[i], bank[i - 1]):
            builder.cswap(control, q_i, q_j)
    zr = builder.alloc_qubits(len(bank[0]))
    for k in range(len(bank[0])):
        builder.cswap(control, bank[0][k], zr[k])
    builder.reset(zr)
    builder.free(zr)


def bm_forward(builder, S, n, nc=NC):
    """Shared-path BM. Returns (sig, tau, L, DP) banks and garbage list."""
    sig = [builder.alloc_qubits(B) for _ in range(nc)]
    tau = [builder.alloc_qubits(B) for _ in range(nc)]
    L = builder.alloc_qubits(7)
    DP = builder.alloc_qubits(B)      # persistent discrepancy
    builder.x([sig[0][0], tau[0][0], DP[0]])
    garbage = []

    W = [builder.alloc_qubits(B) for _ in range(n)]   # stored ratios for replay
    D = [builder.alloc_qubits(B) for _ in range(n)]   # stored pre-swap DP on flag
    F = [builder.alloc_qubits(1) for _ in range(n)]
    tmp = [builder.alloc_qubits(B) for _ in range(nc)]  # reusable scratch bank

    for r in range(n):
        DC = builder.alloc_qubits(B)
        for i in range(0, min(nc, r + 1)):
            builder.ixor_gf2_mul(sig[i], S[r - i], target=DC, field=FIELD)
        flag = F[r]
        nz = builder.alloc_qubits(1)[0]
        _is_nonzero(builder, DC, nz)
        aux = builder.alloc_qubits(1)[0]
        builder.flip_if_less_than(L, r // 2 + 1, target=aux)  # L <= r/2
        builder.init_and(nz, aux, flag[0])
        builder.flip_if_less_than(L, r // 2 + 1, target=aux)
        _is_nonzero(builder, DC, nz)
        builder.free(nz); builder.free(aux)

        # ratio = delta_old / Delta  (computed BEFORE swapping DP<->DC)
        dinv = builder.init_gf2_inverse(DC, field=FIELD)
        builder.ixor_gf2_mul(DP, dinv, target=W[r], field=FIELD)
        builder.del_gf2_inverse(DC, target=dinv, field=FIELD)
        builder.free(dinv)

        # tmp_i = ratio*sig_i + tau_{i-1}   (sig_new = w*sig + z*tau)
        # (on Delta==0 ratio=0 -> tmp=z*tau; the controlled swap below skips
        # the update entirely so sig stays unchanged, matching bm_shared)
        nzu = builder.alloc_qubits(1)[0]
        _is_nonzero(builder, DC, nzu)
        for i in range(nc):
            builder.ixor_gf2_mul(W[r], sig[i], target=tmp[i], field=FIELD)
            if i > 0:
                builder.cx(tau[i - 1], tmp[i])
        for i in range(nc):
            for k in range(B):
                builder.cswap(nzu, sig[i][k], tmp[i][k])
        builder.reset(nzu)
        builder.free(nzu)

        # on flag: tau <- sig_old (in tmp), DC slot <- old DP; L <- r+1-L
        for i in range(nc):
            for k in range(B):
                builder.cswap(flag[0], tmp[i][k], tau[i][k])
        for k in range(B):
            builder.cswap(flag[0], DP[k], DC[k])
        lscr = builder.alloc_qubits(7)
        builder.x([lscr[k] for k in range(7) if ((r + 1) >> k) & 1])
        builder.isub(km.array(L), target=km.array(lscr))  # lscr = r+1-L
        for k in range(7):
            builder.cswap(flag[0], L[k], lscr[k])
        builder.reset(lscr)
        builder.free(lscr)

        # on !flag: tau <- z*tau (physical shift up)
        nflag = builder.alloc_qubits(1)[0]
        builder.x([nflag]); builder.cx(flag[0], nflag)
        _shift_poly_left(builder, tau, control=nflag)
        builder.reset(nflag)
        builder.free(nflag)

        # clear tmp (holds sig_old/tau_old) and DC (holds Delta/old DP) via reset
        for i in range(nc):
            builder.reset(tmp[i])
        builder.reset(DC)
        builder.free(DC)
    garbage.extend([('W', W), ('D', D), ('F', F)])
    return sig, tau, L, DP, garbage


def omega_in_place(builder, sig, S, n, nc=NC):
    """Triangular in-place transform: S[i] <- omega_i = S[i] + sum_j sig_j*S[i-j].

    Processes i descending so every source word is still pre-transform. The
    same call in ascending order restores S exactly (XOR-inverse). Works for
    scaled sigma (Forney's omega/sigma' ratio is scale-invariant).
    """
    for i in range(n - 1, -1, -1):
        for j in range(1, min(nc, i + 1)):
            builder.ixor_gf2_mul(sig[j], S[i - j], target=S[i], field=FIELD)


def omega_restore(builder, sig, S, n, nc=NC):
    """Inverse of omega_in_place: same XOR terms applied ascending, restoring
    low words first so later restorations see original sources."""
    for i in range(0, n):
        for j in range(1, min(nc, i + 1)):
            builder.ixor_gf2_mul(sig[j], S[i - j], target=S[i], field=FIELD)


_MULT_MATRIX_CACHE = {}


def _const_gf_mul(builder, reg, pow_exp, nq):
    """reg <- reg * gamma^{pow_exp} in place (constant GF multiply = linear map).

    Computes the 12x12 GF(2) matrix once, applies via CNOTs into a scratch
    register, swaps back, resets scratch (free Toffolis).
    """
    key = pow_exp % 4095
    if key not in _MULT_MATRIX_CACHE:
        M = np.zeros((nq, nq), dtype=np.uint8)
        el = FIELD.pow(PRIM, key)
        for col in range(nq):
            v = FIELD.mul(1 << col, el)
            for row in range(nq):
                M[row, col] = (v >> row) & 1
        _MULT_MATRIX_CACHE[key] = M
    M = _MULT_MATRIX_CACHE[key]
    tmp = builder.alloc_qubits(nq)
    for r in range(nq):
        for c in range(nq):
            if M[r, c]:
                builder.cx(reg[c], tmp[r])
    for k in range(nq):
        builder.swap(reg[k], tmp[k])
    builder.reset(tmp)
    builder.free(tmp)


def _rotate_bank(builder, bank, nc):
    """Advance Chien counters: bank[i] *= gamma^{-i} for all i (constant muls)."""
    for i in range(nc):
        if i:
            _const_gf_mul(builder, bank[i], -i, B)


def _eval_bank(builder, bank, acc, nc):
    """acc ^= sum_i bank[i]  (Chien eval = XOR of rotated coeffs)."""
    for i in range(nc):
        builder.cx(bank[i], acc)


def chien_sweep(builder, sig_rot, om_rot, sgp_rot, E, m, nc=NC, stride=1):
    """Per-position body: eval sig -> root test, Forney e = om/sgp', phase."""
    for j in range(m):
        # eval accs
        a_s = builder.alloc_qubits(B)   # sigma eval
        a_o = builder.alloc_qubits(B)   # omega eval
        a_p = builder.alloc_qubits(B)   # sigma' eval
        _eval_bank(builder, sig_rot, a_s, nc)
        _eval_bank(builder, om_rot, a_o, nc)
        _eval_bank(builder, sgp_rot, a_p, nc)
        is_root = builder.alloc_qubits(1)[0]
        builder.flip_if_equal(a_s, 0, target=is_root)
        # e = om_eval * inv(sgp_eval)
        inv_p = builder.init_gf2_inverse(a_p, field=FIELD)
        e_reg = builder.alloc_qubits(B)
        builder.ixor_gf2_mul(a_o, inv_p, target=e_reg, field=FIELD)
        # phase: parity(e & M_j) gated by is_root
        par = builder.alloc_qubits(1)[0]
        for k in range(B):
            builder.init_and(E[j][k], e_reg[k], par)
        gate = builder.alloc_qubits(1)[0]
        builder.init_and(par, is_root, gate)
        builder.z([gate])
        builder.del_and(par, is_root, gate)
        builder.free(gate)
        for k in range(B - 1, -1, -1):
            builder.del_and(E[j][k], e_reg[k], par)
        builder.free(par)
        # uncompute e, evals
        builder.del_gf2_mul(a_o, inv_p, target=e_reg, field=FIELD)
        builder.del_gf2_inverse(a_p, target=inv_p, field=FIELD)
        builder.free(e_reg); builder.free(inv_p)
        builder.flip_if_equal(a_s, 0, target=is_root)
        builder.free(is_root)
        _eval_bank(builder, sig_rot, a_s, nc)
        _eval_bank(builder, om_rot, a_o, nc)
        _eval_bank(builder, sgp_rot, a_p, nc)
        builder.reset(a_s); builder.reset(a_o); builder.reset(a_p)
        builder.free(a_s); builder.free(a_o); builder.free(a_p)
        # rotate banks for next position
        _rotate_bank(builder, sig_rot, nc)
        _rotate_bank(builder, om_rot, nc)
        _rotate_bank(builder, sgp_rot, nc)


def build(*, n=None, m=None) -> km.Circuit:
    n = n or N
    m = m or M
    builder = km.CircuitBuilder()
    S = [builder.create_quantum_register(B, name=f'S{i}') for i in range(n)]
    for j in range(m):
        builder.create_classical_register(B, name=f'E{j}')
    sig, tau, L, DP, garbage = bm_forward(builder, S, n)
    omega_in_place(builder, sig, S, n)
    # rotating Chien banks: sig evals; om = first 35 S words; sgp' = odd sig
    # TODO: Psi pass, sweep wiring (sig/om/sig' rotating banks), cleanup.
    omega_restore(builder, sig, S, n)
    return builder.finish_circuit()


if __name__ == '__main__':
    c = build(n=6, m=8)
    print('qubits', c.num_qubits, 'toffoli', c.operation_statistics().num_toffoli_like)
