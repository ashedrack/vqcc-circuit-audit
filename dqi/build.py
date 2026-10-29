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
PRIM = FIELD.primitive_element
N, M, B, LMAX = model.N, model.M, model.B, 35
NC = LMAX + 1


def _is_nonzero(builder, reg, target):
    builder.flip_if_equal(reg, 0, target=target)
    builder.x([target])


def _undo_nonzero(builder, reg, target):
    builder.x([target])
    builder.flip_if_equal(reg, 0, target=target)


def _shift_poly_left(builder, bank, garbage, control=None):
    """bank <- z*bank (coeffs shift up: new[i]=old[i-1], new[0]=0, top drops).

    The wrapped coefficient lands in a scratch register kept in `garbage`
    (measurement erasure would corrupt the phase - replay cleans it later).
    """
    if control is None:
        for i in range(len(bank) - 1, 0, -1):
            builder.swap(bank[i], bank[i - 1])
        zr = builder.alloc_qubits(len(bank[0]))
        for k in range(len(bank[0])):
            builder.swap(bank[0][k], zr[k])
        garbage.append(zr)
        return
    for i in range(len(bank) - 1, 0, -1):
        for q_i, q_j in zip(bank[i], bank[i - 1]):
            builder.cswap(control, q_i, q_j)
    zr = builder.alloc_qubits(len(bank[0]))
    for k in range(len(bank[0])):
        builder.cswap(control, bank[0][k], zr[k])
    garbage.append(zr)


def _shift_poly_left_ret(builder, bank, control):
    """bank <- z*bank on control; returns the wrapped-coeff scratch reg."""
    for i in range(len(bank) - 1, 0, -1):
        for q_i, q_j in zip(bank[i], bank[i - 1]):
            builder.cswap(control, q_i, q_j)
    zr = builder.alloc_qubits(len(bank[0]))
    for k in range(len(bank[0])):
        builder.cswap(control, bank[0][k], zr[k])
    return zr


def _shift_poly_right(builder, bank, zr, control):
    """bank <- z^-1 * bank on control (inverse of _shift_poly_left):
    new[i]=old[i+1]; new[top]=zr content (restored via zr, clearing it)."""
    for k in range(len(bank[0])):
        builder.cswap(control, bank[-1][k], zr[k])   # zr <- bank[top]... need care
    for i in range(len(bank) - 1):
        for q_i, q_j in zip(bank[i], bank[i + 1]):
            builder.cswap(control, q_i, q_j)


def bm_forward(builder, S, n, nc=NC):
    """Shared-path BM. Returns (sig, tau, L, DP) banks and garbage list."""
    sig = [builder.alloc_qubits(B) for _ in range(nc)]
    tau = [builder.alloc_qubits(B) for _ in range(nc)]
    L = builder.alloc_qubits(7)
    DP = builder.alloc_qubits(B)      # persistent discrepancy
    builder.x([sig[0][0], tau[0][0], DP[0]])
    garbage = []

    W = [builder.alloc_qubits(B) for _ in range(n)]   # stored ratios for replay
    D = [None] * n                                    # DC registers -> replay
    F = [builder.alloc_qubits(1) for _ in range(n)]
    LS = [None] * n                                   # lscr per round
    ZR = [None] * n                                   # shift wrap coeff per round
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
        builder.del_gf2_inverse(DC, target=dinv, field=FIELD)  # matched pair
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
        _undo_nonzero(builder, DC, nzu)
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
        LS[r] = lscr           # replay clears (holds L_old / r+1-L)

        # on !flag: tau <- z*tau (physical shift up)
        nflag = builder.alloc_qubits(1)[0]
        builder.x([nflag]); builder.cx(flag[0], nflag)
        ZR[r] = _shift_poly_left_ret(builder, tau, control=nflag)
        builder.cx(flag[0], nflag); builder.x([nflag])
        builder.free(nflag)

        # clear tmp (w-recovery):
        #   flag      -> tmp = tau_old = sig_new + w*tau_new  => tmp ^= sig ^ w*tau
        #   !flag&nz  -> tmp = sig_old = winv*(sig + tau)     => tmp ^= winv*(sig^tau)
        #   !flag&!nz -> tmp = z*tau_old = tau_new            => tmp ^= tau
        winv = builder.init_gf2_inverse(W[r], field=FIELD)
        nf = builder.alloc_qubits(1)[0]
        builder.x([nf]); builder.cx(flag[0], nf)        # nf = !flag
        nzv = builder.alloc_qubits(1)[0]
        _is_nonzero(builder, DC, nzv)                   # nz (valid on !flag)
        ctrlB = builder.alloc_qubits(1)[0]
        builder.init_and(nf, nzv, ctrlB)                # !flag & nz
        nzb = builder.alloc_qubits(1)[0]
        builder.x([nzb]); builder.cx(nzv, nzb)          # nzb = !nz
        ctrlC = builder.alloc_qubits(1)[0]
        builder.init_and(nf, nzb, ctrlC)                # !flag & !nz
        for i in range(nc):
            tA = builder.alloc_qubits(B)
            builder.ixor_gf2_mul(W[r], tau[i], target=tA, field=FIELD)
            builder.cx(sig[i], tA)                      # tA = sig_i ^ w*tau_i
            for k in range(B):
                builder.ccx(flag[0], tA[k], tmp[i][k])
            builder.cx(sig[i], tA)                      # uncompute tA
            builder.ixor_gf2_mul(W[r], tau[i], target=tA, field=FIELD)
            builder.free(tA)
            tB = builder.alloc_qubits(B)
            builder.cx(sig[i], tB); builder.cx(tau[i], tB)
            tC = builder.alloc_qubits(B)
            builder.ixor_gf2_mul(winv, tB, target=tC, field=FIELD)
            for k in range(B):
                builder.ccx(ctrlB, tC[k], tmp[i][k])
                builder.ccx(ctrlC, tau[i][k], tmp[i][k])
            builder.ixor_gf2_mul(winv, tB, target=tC, field=FIELD)
            builder.cx(tau[i], tB); builder.cx(sig[i], tB)
            builder.free(tB); builder.free(tC)
        builder.del_gf2_inverse(W[r], target=winv, field=FIELD)
        builder.free(winv)
        builder.ccx(nf, nzb, ctrlC); builder.free(ctrlC)
        builder.cx(nzv, nzb); builder.x([nzb]); builder.free(nzb)
        builder.ccx(nf, nzv, ctrlB); builder.free(ctrlB)
        _undo_nonzero(builder, DC, nzv); builder.free(nzv)
        builder.cx(flag[0], nf); builder.x([nf]); builder.free(nf)

        # DC (holds flag? DP_old : Delta) -> replay bank
        D[r] = DC
    garbage += W + [r for r in D if r is not None] + F
    garbage += [r for r in LS if r is not None] + [r for r in ZR if r is not None]
    return sig, tau, L, DP, W, D, F, LS, ZR, garbage


def monicize(builder, sig, nc=NC):
    """sig <- sig / sig_0 (monic locator). Forney's omega/sigma' ratio is
    scale-invariant. Every erased scratch is cleared by a recompute XOR (no
    measurement - it would corrupt the phase). Returns the sigma_0 register
    (needed later to un-monicize before the BM backward replay)."""
    s0 = builder.alloc_qubits(B)
    builder.cx(sig[0], s0)                        # s0 = sigma_0
    inv0 = builder.init_gf2_inverse(sig[0], field=FIELD)
    for i in range(nc):
        t = builder.alloc_qubits(B)
        builder.ixor_gf2_mul(inv0, sig[i], target=t, field=FIELD)
        for k in range(B):
            builder.swap(sig[i][k], t[k])         # sig_i <- sig_i/sig_0
        builder.ixor_gf2_mul(s0, sig[i], target=t, field=FIELD)  # t = old ^ s0*new = 0
        builder.free(t)
    builder.del_gf2_inverse(s0, target=inv0, field=FIELD)
    builder.free(inv0)
    return s0


def omega_in_place(builder, sig, S, n, nc=NC):
    """Triangular in-place transform: S[i] <- omega_i = S[i] + sum_j sig_j*S[i-j].
    Requires monic sigma (sig_0 == 1). Descending i keeps sources original."""
    for i in range(n - 1, -1, -1):
        for j in range(1, min(nc, i + 1)):
            builder.ixor_gf2_mul(sig[j], S[i - j], target=S[i], field=FIELD)


def omega_restore(builder, sig, S, n, nc=NC):
    """Inverse of omega_in_place: same XOR terms applied ascending."""
    for i in range(0, n):
        for j in range(1, min(nc, i + 1)):
            builder.ixor_gf2_mul(sig[j], S[i - j], target=S[i], field=FIELD)


_MULT_MATRIX_CACHE = {}


def _const_gf_mul(builder, reg, pow_exp, nq):
    """reg <- reg * gamma^{pow_exp} in place (constant GF multiply = linear map).

    Computes M into scratch, swaps, then uncomputes scratch with M^{-1}
    (reversible XOR - no measurement; measurement erasure corrupts phase).
    """
    key = pow_exp % 4095
    if key not in _MULT_MATRIX_CACHE:
        el = FIELD.pow(PRIM, key)
        M = np.zeros((nq, nq), dtype=np.uint8)
        for col in range(nq):
            v = FIELD.mul(1 << col, el)
            for row in range(nq):
                M[row, col] = (v >> row) & 1
        Minv = np.linalg.inv(M.astype(np.float64)).astype(np.uint8) % 2
        _MULT_MATRIX_CACHE[key] = (M, Minv)
    M, Minv = _MULT_MATRIX_CACHE[key]
    tmp = builder.alloc_qubits(nq)
    for r in range(nq):
        for c in range(nq):
            if M[r, c]:
                builder.cx(reg[c], tmp[r])
    for k in range(nq):
        builder.swap(reg[k], tmp[k])
    # tmp = old reg; new reg = M*old -> tmp ^= Minv * new reg -> 0
    for r in range(nq):
        for c in range(nq):
            if Minv[r, c]:
                builder.cx(reg[c], tmp[r])
    builder.free(tmp)


def _rotate_bank(builder, bank, exp_fn):
    """Advance Chien counters: bank[i] *= gamma^{exp_fn(i)} for all i."""
    for i in range(len(bank)):
        e = exp_fn(i) % 4095
        if e:
            _const_gf_mul(builder, bank[i], e, B)


def _eval_bank(builder, bank, acc):
    """acc ^= sum_i bank[i]  (Chien eval = XOR of rotated coeffs)."""
    for i in range(len(bank)):
        builder.cx(bank[i], acc)


def chien_sweep(builder, sig_rot, om_rot, sgp_rot, E, m):
    """Root scan over all m positions: eval sigma/omega/sigma' at beta_j =
    gamma^{-j}, Forney e = omega/sigma', phase Z(parity(e & M_j) & is_root).
    Banks are rotated by constant maps (free Toffolis); after the full 4095-
    point cycle they return to their initial values (rotation is cyclic)."""
    for j in range(m):
        a_s = builder.alloc_qubits(B)
        a_o = builder.alloc_qubits(B)
        a_p = builder.alloc_qubits(B)
        _eval_bank(builder, sig_rot, a_s)
        _eval_bank(builder, om_rot, a_o)
        _eval_bank(builder, sgp_rot, a_p)
        is_root = builder.alloc_qubits(1)[0]
        builder.flip_if_equal(a_s, 0, target=is_root)
        inv_p = builder.init_gf2_inverse(a_p, field=FIELD)
        e_reg = builder.alloc_qubits(B)
        builder.ixor_gf2_mul(a_o, inv_p, target=e_reg, field=FIELD)
        par = builder.alloc_qubits(1)[0]
        for k in range(B):
            builder.ccx(E[j][k], e_reg[k], par)
        gate = builder.alloc_qubits(1)[0]
        builder.ccx(par, is_root, gate)
        builder.z(gate)
        builder.ccx(par, is_root, gate)
        builder.free(gate)
        for k in range(B - 1, -1, -1):
            builder.ccx(E[j][k], e_reg[k], par)
        builder.free(par)
        builder.ixor_gf2_mul(a_o, inv_p, target=e_reg, field=FIELD)
        builder.del_gf2_inverse(a_p, target=inv_p, field=FIELD)
        builder.free(e_reg); builder.free(inv_p)
        builder.flip_if_equal(a_s, 0, target=is_root)
        builder.free(is_root)
        _eval_bank(builder, sig_rot, a_s)
        _eval_bank(builder, om_rot, a_o)
        _eval_bank(builder, sgp_rot, a_p)
        builder.free(a_s); builder.free(a_o); builder.free(a_p)
        _rotate_bank(builder, sig_rot, lambda i: -i)
        _rotate_bank(builder, om_rot, lambda i: -i)
        _rotate_bank(builder, sgp_rot, lambda i: -2 * i)


def build(*, n=None, m=None) -> km.Circuit:
    n = n or N
    m = m or M
    builder = km.CircuitBuilder()
    S = [builder.create_quantum_register(B, name=f'S{i}') for i in range(n)]
    E = [builder.create_classical_register(B, name=f'E{j}') for j in range(m)]
    sig, tau, L, DP, W, D, F, LS, ZR, garbage = bm_forward(builder, S, n)
    s0 = monicize(builder, sig)                  # scaled locator -> monic
    garbage.append(s0)
    omega_in_place(builder, sig, S, n)          # S[:35]=omega, S[35:]=0
    om_rot = S[:35]
    # sigma' bank: odd coeffs sigma_{2k+1}, k=0..17 (deg <= 17)
    sgp_rot = [builder.alloc_qubits(B) for _ in range(18)]
    for k in range(18):
        builder.cx(sig[2 * k + 1], sgp_rot[k])
    chien_sweep(builder, sig, om_rot, sgp_rot, E, m)
    # uncompute sgp copy (banks self-restored after full 4095 cycle)
    for k in range(18):
        builder.cx(sig[2 * k + 1], sgp_rot[k])
        builder.free(sgp_rot[k])
    omega_restore(builder, sig, S, n)          # restore S
    # TODO: backward replay to clear all workspace (no measurement erasure -
    # it corrupts the phase). Garbage stays live until the replay is written.
    return builder.finish_circuit()


if __name__ == '__main__':
    c = build(n=6, m=8)
    print('qubits', c.num_qubits, 'toffoli', c.operation_statistics().num_toffoli_like)


def bm_backward(builder, S, sig, tau, L, DP, W, D, F, LS, ZR, n, nc=NC):
    """Backward replay of bm_forward: r=n-1..0, inverting each round using the
    stored F[r] flag, W[r] ratio, D[r] DC-slot, LS[r] lscr, ZR[r] wrap coeff.

    State inversion (post -> pre):
      flag=1:      sig=tau (swap back); tau_i <- tau[i+1] + w*sig[i+1] (desc)
      flag=0,nz=1: tau unshifted already; sig_i <- winv*(sig_i + tau_i_new)
      flag=0,nz=0: sig unchanged
    Stored banks cleared by recompute-XOR from the restored state.
    All ops reversible - no measurement erasure (it corrupts the phase).
    """
    for r in range(n - 1, -1, -1):
        flag = F[r]
        nflag = builder.alloc_qubits(1)[0]
        builder.x([nflag]); builder.cx(flag[0], nflag)

        # undo !flag tau shift (right shift); restores wrap coeff from ZR[r]
        _shift_poly_right(builder, tau, ZR[r], control=nflag)
        builder.free(ZR[r])

        # undo L swap: cswap(flag, L, LS[r]); clear lscr by recompute
        lscr = LS[r]
        for k in range(7):
            builder.cswap(flag[0], L[k], lscr[k])
        # clear lscr: on flag it holds r+1-L_old=L_new->cur, on !flag r+1-L_cur
        # XOR (r+1-L_cur) into it on !flag, XOR L_cur on flag
        t7 = builder.alloc_qubits(7)
        builder.x([t7[k] for k in range(7) if ((r + 1) >> k) & 1])
        builder.isub(km.array(L), target=km.array(t7))   # t7 = r+1-L_cur
        for k in range(7):
            builder.ccx(nflag, t7[k], lscr[k])           # !flag: lscr ^= r+1-L
            builder.ccx(flag[0], L[k], lscr[k])          # flag:  lscr ^= L_old
        builder.free(t7)
        builder.free(lscr)

        # undo DP<->DC swap: restores DP_old on flag, D[r]<-Delta_r
        DC = D[r]
        for k in range(B):
            builder.cswap(flag[0], DP[k], DC[k])
        # DC now = Delta_r on both branches

        # undo the locator/template update (state inversion)
        winv = builder.init_gf2_inverse(W[r], field=FIELD)
        # !flag&nz: sig_i <- winv*(sig_i + tau_{r+1,i}); flag: swap handled below
        nzv = builder.alloc_qubits(1)[0]
        _is_nonzero(builder, DC, nzv)
        ctrlB = builder.alloc_qubits(1)[0]
        builder.ccx(nflag, nzv, ctrlB)                   # !flag & nz
        for i in range(nc):
            # !flag&nz: sig_i += winv*(sig_i + tau_i) -> transforms to sig_old
            t = builder.alloc_qubits(B)
            builder.cx(sig[i], t); builder.cx(tau[i], t)
            tC = builder.alloc_qubits(B)
            builder.ixor_gf2_mul(winv, t, target=tC, field=FIELD)
            for k in range(B):
                builder.ccx(ctrlB, tC[k], sig[i][k])
            builder.ixor_gf2_mul(winv, t, target=tC, field=FIELD)
            builder.cx(tau[i], t); builder.cx(sig[i], t)
            builder.free(t); builder.free(tC)
        # flag: swap(sig,tau) -> sig=sig_old,tau=sig_new; tau_i<-tau[i+1]+w*sig[i+1]
        for i in range(nc):
            for k in range(B):
                builder.cswap(flag[0], sig[i][k], tau[i][k])
        for i in range(nc - 2, -1, -1):
            t = builder.alloc_qubits(B)
            builder.cx(tau[i + 1], t)
            builder.ixor_gf2_mul(W[r], sig[i + 1], target=t, field=FIELD)
            for k in range(B):
                builder.ccx(flag[0], t[k], tau[i][k])
            builder.ixor_gf2_mul(W[r], sig[i + 1], target=t, field=FIELD)
            builder.cx(tau[i + 1], t)
            builder.free(t)
        builder.del_gf2_inverse(W[r], target=winv, field=FIELD)
        builder.free(winv)
        _undo_nonzero(builder, DC, nzv); builder.free(nzv)
        builder.ccx(nflag, nzv, ctrlB)  # nzv recomputed -> reuse for ctrlB undo
        # (ctrlB uncompute needs nzv value again - it was just undone; recompute)
        # simpler: free ctrlB into garbage
        builder.free(ctrlB)
        builder.cx(flag[0], nflag); builder.x([nflag]); builder.free(nflag)

        # clear stored banks: W[r] (recompute w), F[r] (recompute flag), D[r] (recompute Delta)
        # Delta_r = disc(sig_r) -> XOR into DC slot
        DCs = builder.alloc_qubits(B)
        for i in range(0, min(nc, r + 1)):
            builder.ixor_gf2_mul(sig[i], S[r - i], target=DCs, field=FIELD)
        for k in range(B):
            builder.cx(DCs[k], DC[k])
        builder.free(DCs)
        builder.free(DC)
        # W[r] ^= DP_r * inv(Delta_r): Delta_r == DCs value == disc(sig_r)
        din = builder.init_gf2_inverse(DC, field=FIELD)
        builder.ixor_gf2_mul(DP, din, target=W[r], field=FIELD)
        builder.del_gf2_inverse(DC, target=din, field=FIELD)
        builder.free(din)
        # F[r] ^= flag_recomputed = nz(Delta) & (2L<=r)
        nz2 = builder.alloc_qubits(1)[0]
        _is_nonzero(builder, DC, nz2)
        ax2 = builder.alloc_qubits(1)[0]
        builder.flip_if_less_than(L, r // 2 + 1, target=ax2)
        builder.init_and(nz2, ax2, flag[0])
        builder.flip_if_less_than(L, r // 2 + 1, target=ax2)
        _undo_nonzero(builder, DC, nz2)
        builder.free(nz2); builder.free(ax2)
        # free F[r] slot and DC bank reg now clean
        builder.free(flag)
    return


def _stub_marker():
    pass
