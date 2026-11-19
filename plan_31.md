# Path to GM ~31 (+3 points)

Current: **28.23498** (v77). Target: **~31** = product ×1.331.

## The arithmetic

| Task | Now | −9.3% (distributed path) | What it means |
|---|---|---|---|
| invsqrt | 74,383.7 (27.92x) | ~67,500 (30.8x) | −6,900 cost |
| dqi | 345.236e9 (70.4x) | ~313e9 (77.6x) | −32e9 = T_exec 366k → ~331k |
| chem | 294.64e6 (11.39x) | ~267.5e6 (12.5x) | −27e6 |

Single-task alternatives are all harder: dqi alone needs T_exec 366k→270k
(−26%) or Q 890→764 (impossible — 840 syndrome qubits are irreducible).
The realistic route is **modest gains on all three tasks at once**.

## Cost decomposition of the dqi file (the big lever)

`score = rep_factor × Q² × T_exec` = 1.19126 × 890² × 365,872.

Three multiplicative attack surfaces:

1. **T_exec = 365,872** — dominated by GF(2¹²) arithmetic in the
   Berlekamp–Massey rounds. Schoolbook GF(2^12) multiply = 12² = 144
   ANDs ≈ 144 Toffolis. Karatsuba (3-way split, char-2) cuts this to
   ~3×6² = 108 ANDs (−25% per multiplication) before even counting the
   additive savings from the binary-tree decomposition. If ~half the
   executed Toffolis are GF-multiplier ANDs, a Karatsuba/Toom multiplier
   pass is worth ~−15–45k T_exec alone. **This is concretely measurable:**
   instrument the op stream, isolate the multiplier blocks, replace with
   a cheaper multiplier circuit. It is also genuinely own work — a better
   GF(2^12) inverter/multiplier cell is a publishable artifact.
2. **rep_factor = 1.19126** — priced by effective error rate 8.67e-4
   (truncation budget). The file deliberately spends ~1.4e-4-9e-4 failure
   probability to save Toffolis. A different point on this tradeoff may
   dominate: e.g. rep 1.19→~1.10 (−7.7% cost) if error-rate ~4e-4 is
   reachable at +few k T_exec. Needs understanding exactly which
   truncations generate the residual failure mass (bm_windows cutoffs +
   reversal-stack depth + Euclid class coverage).
3. **Q² = 792,100** — 840 syndrome + 50 workspace; proven irreducible.
   Ignore.

Composite example to ~31: dqi T_exec −15% (366k→311k → cost ~295e9)
× chem −10% (→265e6) × invsqrt −3% (→72.2k) gives product ×1.33.

## Chem path (−10%)

T_total = 14,706: loaders 8,044, inner-prep 4,095, rotations 2,250.
λ_eff saturated at caps. Levers:

- **Encoding/ANF-mass search**: loaders are optimal *for this encoding*;
  a different coded-word scheme changes the Boolean functions' algebraic
  normal form. Small changes to word coding could cut the 189 product
  terms. Requires a refit (fit_als.py + wrefit.py chain exists) plus a
  coding search — the highest-uncertainty item.
- **Factorization shape**: R/B/C tradeoffs change loader count and
  inner-prep T; the cost model has cliffs (B=129 temp-qubit cliff).
  Cheapest exploration: evaluate alternative R×B×C shapes under the
  error caps using existing pipeline.

## Invsqrt path (−3%)

- 44→43 qubit rebuild = −1,700 cost instantly (1 qubit ≈ 38 T-equiv).
  Requires a borrow scheme for the correction-stage working set — the
  frontier didn't find one; fresh eyes may.
- New rotation basis terms: proven useless as pure x-bit functions;
  computed parities (free CX) on the normalized value might give the LP
  a working column. Low probability, low cost to test.

## Execution order

1. **dqi GF-arithmetic audit** (this week): parse the op stream, locate
   the multiplier structure, measure AND-density per BM round. Output:
   exact T_exec decomposition → tells us how much a Karatsuba/Toom
   multiplier circuit saves. Concrete, bounded, own work.
2. **rep-factor tradeoff mapping**: enumerate which approximations
   generate the 8.7e-4 error rate; re-solve the tradeoff for min cost.
3. **Chem encoding search**: background — run the existing fit chain at
   alternative word codings; measure ANF mass of candidates.
4. **invsqrt 43q study**: sketch the correction stage's working set;
   identify a borrow candidate.
5. **Watch the public frontier**: manifest diff weekly — free gains have
   historically arrived ~weekly (+0.003 to +0.25).

## Deliverables along the way

Each stage produces a citable artifact: GF-multiplier notebook, decoder
build, factorization results, audit updates. Commit + push to
ashedrack/vqcc-circuit-audit per stage; publish the strongest as Kaggle
notebooks for community visibility.
