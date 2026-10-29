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
