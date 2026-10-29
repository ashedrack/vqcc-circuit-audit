"""Classical reference model for the DQI syndrome decoder.

Given syndromes S_i = sum_j e_j * alpha^{(i+1) j} (i=0..n-1, GF(2^12)), recover
the error vector e (weight <= n//2), then the MBUC phase parity
    sum_j popcount(e_j & M_j) mod 2.
Algorithm: Berlekamp-Massey -> Chien search -> Forney, all in GF(2^12).
This is the algorithmic oracle for our own kickmix decoder.
"""
from __future__ import annotations

import numpy as np
import kickmix as km

FIELD = km.GF2Field(12)
PRIM = FIELD.primitive_element
N, M, B = 70, 4095, 12
ORDER = M  # 2^12 - 1
LOG = {FIELD.pow(PRIM, k): k for k in range(ORDER)}
ANTILOG = np.array([FIELD.pow(PRIM, k) for k in range(ORDER)])


def gmul(a, b):
    return FIELD.mul(a, b)


def ginv(a):
    return FIELD.invert(a)


def gpow(a, k):
    return FIELD.pow(a, k)


def syndrome(symbols: list[int]) -> list[int]:
    """S_i = sum_j e_j alpha^{(i+1)j} for error dict {pos: value}."""
    out = np.zeros(N, dtype=np.int64)
    for i in range(N):
        acc = 0
        for j, e in symbols.items():
            acc ^= gmul(e, gpow(PRIM, (i + 1) * j))
        out[i] = acc
    return out.tolist()


def berlekamp_massey(S: list[int]) -> list[int]:
    """Returns sigma = [s0, s1, ...] with sigma(z) = prod(1 - alpha^j z) over
    error positions (sigma[0] = 1), the shortest linear recurrence of S."""
    C = [0] * (N + 1)
    Bx = [0] * (N + 1)
    C[0] = Bx[0] = 1
    L, m, bb = 0, 1, 1
    for n in range(N):
        d = S[n]
        for i in range(1, L + 1):
            d ^= gmul(C[i], S[n - i])
        if d == 0:
            m += 1
        elif 2 * L <= n:
            T = C[:]
            c = gmul(d, ginv(bb))
            for i in range(len(C) - m):
                C[i + m] ^= gmul(c, Bx[i])
            L, Bx, bb, m = n + 1 - L, T, d, 1
        else:
            c = gmul(d, ginv(bb))
            for i in range(len(C) - m):
                C[i + m] ^= gmul(c, Bx[i])
            m += 1
    return C[:L + 1]


def chien(sigma: list[int]) -> list[int]:
    """Error positions: roots z = alpha^{-j} of sigma."""
    out = []
    for j in range(M):
        x = gpow(PRIM, -j)
        acc, xp = sigma[0], x
        for c in sigma[1:]:
            acc ^= gmul(c, xp)
            xp = gmul(xp, x)
        if acc == 0:
            out.append(j)
    return out


def forney(sigma: list[int], omega: list[int], j: int) -> int:
    """e_j = omega( alpha^{-j} ) / sigma'( alpha^{-j} )  (char 2)."""
    x = gpow(PRIM, -j)
    num, xp = omega[0], x
    for c in omega[1:]:
        num ^= gmul(c, xp)
        xp = gmul(xp, x)
    # formal derivative keeps odd powers: sigma'(x) = sum_{i odd} sigma_i x^{i-1}
    den, xp = 0, 1
    for i in range(1, len(sigma), 2):
        den ^= gmul(sigma[i], xp)
        xp = gmul(xp, gmul(x, x))
    return gmul(num, ginv(den))


def key_equation(S: list[int], sigma: list[int]) -> list[int]:
    """Omega(z) = sigma(z) S(z) mod z^N; only terms of degree < len(sigma) used."""
    n_s = len(sigma)
    om = [0] * max(n_s - 1, 1)
    for k in range(len(om)):
        for i in range(k + 1):
            if i < len(sigma):
                om[k] ^= gmul(sigma[i], S[k - i])
    return om


def decode(syn: list[int]) -> dict[int, int]:
    """Full decode -> {position: error value}."""
    sigma = berlekamp_massey(syn)
    if len(sigma) <= 1:
        return {}
    pos = chien(sigma)
    omega = key_equation(syn, sigma)
    return {j: forney(sigma, omega, j) for j in pos}


def phase(err: dict[int, int], mbuc: np.ndarray) -> int:
    return sum(int(e & mbuc[j]).bit_count() for j, e in err.items()) & 1


def bm_oblivious(S: list[int]) -> tuple[list[int], int]:
    """Data-oblivious BM (validated vs berlekamp_massey): implicit shift via Bx,
    single flag branch, division-based coef. Circuit port target."""
    C = [0] * 72; C[0] = 1
    Bx = [0] * 72; Bx[0] = 1
    L = 0; b = 1
    for r in range(N):
        d = S[r]
        for i in range(1, L + 1):
            d ^= gmul(C[i], S[r - i])
        coef = gmul(d, ginv(b))
        U = [0] + [gmul(coef, Bx[i]) for i in range(71)]
        Cold = C[:]
        C = [C[i] ^ U[i] for i in range(72)]
        if (d != 0) and (2 * L <= r):
            Bx, L, b = Cold, r + 1 - L, d
        else:
            Bx = [0] + Bx[:-1]
    return C[:L + 1], L


# CIRCUIT DESIGN NOTES (v1, unoptimized ~4M Toff / ~3000q target)
# Registers: S0..S69 (named 840q), E0..E4094 (named classical), C/Bx 36x12 each,
#   D bank 70x12, F bank 70 flags, L 7q, b 12q, Chien banks sig/Om/sig' 36x12 each,
#   eval accs + e + parity ancilla + inv scaffold.
# BM: per iter r -> d into D[r] (fresh slice, clean); flag_r; C ^= coef*(x*Bx);
#   Bx mux in-place (Bx_i ^= flag*(Bx_i^C_i) then ^= ~flag*(Bx_i^Bx_{i-1}), i desc);
#   L/b flag-controlled updates; store d_r, flag_r for the later reverse replay.
# Chien: 3 rotating banks (sig, Om=sig*S mod z^70, sig'=deriv). Per position:
#   rotate coeffs by const alpha^-i (CNOTs only); eval sig->acc (XOR fanout);
#   is_root=(acc==0); e=Om*inv(sig'); phase=Z(parity(e&M) & is_root); uncompute all
#   eval regs while inputs live (re-XOR), THEN rotate -> next position (clean).
# Cleanup: after Chien, replay BM backwards r=69..0 using stored d_r/flag_r to
#   regenerate and clear C/Bx/L/b state and the D/F banks.
# Phase trick: gf2_phase_by_product gives the c*bin(e) phase for free once e_j is
#   decoded; alternatively Z on parity(e_j & M_j) computed by CNOT tree.


# ============================================================
# REVERSIBLE-CIRCUIT DESIGN v2 (resolves conditional-overwrite irreversibility)
# ============================================================
# Key insight: a conditional OVERWRITE (Bx_i := flag?A:B) is irreversible.
# Solutions used everywhere below:
#   * CSWAP banks/swaps are self-inverse: swap(C,Bx) on flag; swap(L,scratch) etc.
#   * XOR-accumulate ops (ixor_gf2_mul, cx) are self-inverse given inputs live.
#   * Per-iteration garbage goes into D[r]/F[r]/scratch slices, cleaned by a
#     full BACKWARD replay at the end (Bennett-style compute-store-replay).
#
# FORWARD BM, per iteration r (all steps reversible):
#   d = D[r] := S_r ^ sum_i C_i*S_{r-i}        [fresh slice = clean]
#   flag = F[r] := (d!=0) && (2L<=r)         [fresh slice = clean]
#   binv = init_gf2_inverse(b); coef ^= d*binv (ixor_gf2_mul); del binv later
#   C_{i+1} ^= coef * Bx_i                    [uses OLD Bx; in-place, self-inv]
#   Bx := x*Bx  (in-place cyclic shift chain, unconditional, reversible)
#   if flag: CSWAP entire C bank <-> Bx bank  (Bx <- C_old; C <- x*Bx_old... 
#       wait: swap AFTER shift gives Bx=C_old, C=x*Bx_old -- C loses C_old!
#       -> ORDER: save Cold? NO: do C-update BEFORE shift/swap:
#       correct order: (a) C ^= coef*x*Bx_old  (b) Bx := x*Bx_old  (c) if flag CSWAP(C,Bx)
#       => flag: Bx=C_old, C=(x*Bx_old)_new... WRONG -- C was already updated!
#   REVISED ORDER for update C ^= coef*x*Bx with Bx role-swap:
#     1) compute coef; 2) Bx := x*Bx (shift); 3) if flag CSWAP(C,Bx)
#        [Bx now = flag?C_r : x*Bx_r ; C now = flag?x*Bx_r : C_r]
#     4) C ^= coef * Bx_r  -- PROBLEM: Bx_r gone if flag (it IS C now = x*Bx_r).
#   FINAL CORRECT ORDER:
#     1) compute coef (inv + mul)
#     2) for i: C_{i+1} ^= coef*Bx_i        [Bx_r still live]
#     3) if flag: CSWAP(C,Bx) [BANKS swap -> Bx=C_r_preupdate? NO C already updated!]
#   => must snapshot C_old? NO -- reorder differently:
#     1) compute coef
#     2) if flag: CSWAP(C,Bx) FIRST [Bx=C_old_preupdate, C=Bx_old]
#     3) Bx_new = x*Bx: but which bank? After swap Bx holds C_old (the swap target).
#        Non-flag path needs Bx := x*Bx_old. If flag, Bx should be C_old UNSHIFTED.
#   SO: swap FIRST, then SHIFT only if !flag (conditional shift = controlled swaps):
#     1) compute coef
#     2) if flag: CSWAP(C,Bx)                 [Bx=C_old, C=Bx_old on flag]
#     3) if !flag: shift Bx <- x*Bx           [~432 controlled swaps]
#     4) C ^= coef * Bx_old                   [PROBLEM: Bx_old is in C bank now on flag]
#        FIX: compute U_i=coef*Bx_i into C BEFORE any Bx mutation, using step order:
#   =>> SIMPLEST WORKING ORDER (verify in classical first):
#     a) d,flag,coef computed
#     b) for i: C_{i+1} ^= coef*Bx_i              [Bx_r live; in-place]
#     c) save Cold is NOT needed: on flag we want Bx=C_old_preupdate BUT C is updated.
#        => instead do the flag swap of Bx with C BEFORE C update? C update needs coef*Bx_old:
#     a) d,flag,coef
#     b) if flag: CSWAP(C,Bx)                   [Bx <- C_old(pre-upd), C <- Bx_old]
#     c) for i: (C role)_{i+1} ^= coef*(Bx role)_i ?? indices drift. SOLUTION:
#        keep ROLE indirection via bank pointer parity p (0/1 stored):
#        C-role bank = bank[p], Bx-role bank = bank[1-p]; flag toggles p.
#        Then physical banks never move: updates read bank[p]/bank[1-p] -- but the
#        READ addressing becomes p-dependent => mux both banks' reads (~2x cost, still cheap).
#        Bx shift: applied to bank[1-p] physically.
#   DECISION: implement role-indirection with parity bit p (in F tracking); mux
#   reads cost ~2x on mults -> BM ~700k T. ALTERNATIVE (simpler, chosen):
#   physical CSWAP of the two 36xb banks on flag (~432T) THEN unconditional ops
#   always reference the SAME physical registers -- role = location, zero muxing.
#   Order: a) d,flag,coef b) CSWAP(C,Bx) if flag c) C-update via coef*Bx needs
#   OLD Bx which on flag is now in C bank -> does the update use C-role or Bx-role?
#   In classical BM: C_new = C - coef*x^m*Bx where Bx is the OLD B (pre-swap).
#   With physical swap first, old B is in C bank -> update C_role := Bx_role_bank?
#   => maintain formula with swapped reads: when flag, U = coef*x*(C bank); when
#   !flag, U = coef*x*(Bx bank) -- needs the flag-controlled read selection anyway.
#
#   PRAGMATIC FINAL: accept ~2x read muxing (flag-controlled bank select for the
#   multiplier source). BM total ~700-800k Toffolis. Chien ~3.3M. Total ~4M, Q~3k.
#
# Chien+Forney (clean inline uncompute per position):
#   banks sig(36x12), Om(36x12 = sigma*S mod z^70 via 36 multiplications once),
#   sig'(36x12 = formal deriv coeffs, init once).
#   Per position j:
#     acc1 ^= XOR(sig_i)        ; is_root = (acc1==0) into flag ancilla
#     acc2 ^= XOR(Om_i); acc3 ^= XOR(sig'_i)
#     e = init_gf2_inverse? no: e = acc2 * inv(acc3): ib=init_gf2_inverse(acc3);
#       e ^= acc2*ib (ixor_gf2_mul)
#     p = parity(e & M_j) ancilla; Z(p & is_root): phase applied iff root
#     uncompute p, e, ib(del), is_root, acc1,acc2,acc3  [inputs all live]
#     rotate the 3 banks by const alpha^{-j}: coef_i *= alpha^{-i} = LINEAR CNOT map
#   Per-position cost ~ inverse 315 + del ~90 + mul 63 + evals/control ~300 ~ 800T
#   => 4095 * ~800 ~ 3.3M Toffolis.
#
# Backward replay for BM scratch cleanup (D,F,Lscratch banks): r=69..0 invert.
# Total estimate: BM fwd ~0.4M + BM bwd ~0.4M + Chien ~3.3M ~ 4.1M Toffolis, ~3000q.
# Score ~ R(0) * 3000^2 * 4.1e6 vs baseline 1893^2*5.7e6 -> ~0.8-1.1x. GOOD v1.
