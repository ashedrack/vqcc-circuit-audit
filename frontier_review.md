# Frontier Review — how the best public circuits work and where the headroom is

Compiled 2026-11-19 from the frontier notebooks (boristown gap-3, maoxin123 loaders/v27,
ozertuu fast-runner, asterios 28.15) and the manifest at manwithacat/vqcc-apache-public-circuits.
Purpose: understand each winning technique before attempting our own improvements.

Cost models:
- invsqrt: C = Q x (Toffoli + T/4 + rotations/4-ish). 1 qubit ~= 38 Toffolis.
- dqi:     C = R(eps) x Q^2 x Toffoli. 1 qubit ~= 830 Toffolis. R=1.191 at zero failures (floor).
- chem:    C = lambda_eff x Toffoli x Q.

Current best-of (verified locally, v76 = 28.18862):
- invsqrt d0227f3c  44q   74,405.59    27.91x   (public best ecf908fe: 74,383.70)
- dqi     e208d350  890q  345.25e9     ~70.8x   (WE hold the manifest's #1 file)
- chem    56ef2562  427q  294.64e6     11.39x

## Task 1 — invsqrt (phase e^{2pi i / sqrt x} on 22-bit x)

### Technique stack (the frontier line)
1. Normalize: even bit-shift of x into a fixed range; 1/sqrt(4^k n) = 2^-k / sqrt n, so
   shifting only rescales the output. (Syed Asad Ali generator line.)
2. Coarse estimate: integer sqrt d ~ sqrt(n) plus a 64-row slope table (Ali Haydar
   Ozdag) giving a first-order guess.
3. Reversible refine: a division/correction stage converts residual n - d^2 into a
   high-precision Taylor/Newton correction term.
4. Phase write: Z_POW rotations per output bit (phase kickback); angles LP-fitted so
   worst-x error stays inside tolerance (margin 38.86M at the current fit).
5. Uncompute: replay the arithmetic backwards; division clean-up now hands
   precomputed AND terms into the second table load (saves Toffolis).

### How the gains actually came (micro-surgery, not redesign)
- AND-merge cuts at wires q28/q27 (Emerson, Tang): -1 T each.
- boristown "keep the AND": when CCX(a,b,t) recurs with a,b unchanged, park a&b in a
  provably-|0> spare qubit, reuse via CX, erase by HMR measurement = 0 Toffolis.
  Their scanner found 281 repeated pairs in dqi; exactly one window had a free qubit.
- boristown "gap-3 residue repair + C43 native compiler": ecf908fe = 74,383.70.
- LP refit of the 33 rotation angles: shaves error margin, not cost directly.

### Headroom assessment
- Q is king: 44 -> 43 qubits is worth ~1,600+ cost alone (~2%). Requires restructuring
  the arithmetic to reuse x-register bits — hard but the single biggest lever.
- T: the AND-parking scanner idea generalizes — a systematic window scanner on the
  invsqrt circuit (not just dqi) may find more free-qubit reuse sites. Cheap to try.
- Rotations: 33 x ~0.25 = ~8 cost. Negligible.
- Realistic private gain: 1-3% (74.4k -> 72.5-73.5k) via AND-parking + tight refits.
  30x (68.3k) needs a structural redesign of the correction stage.

## Task 2 — dqi (RS decode over GF(2^12), syndrome 70 x 12 = 840q)

### Technique stack (Lukin-group lineage, dieted)
1. Truncated shared-path Berlekamp-Massey: 70 rounds, w = delta/Delta, one GF
   inversion per round; coefficient windows measured per round on the error-weight
   distribution (mass < 4e-5 -> approximate decode is allowed).
2. Measurement erasure: values computed into FRESH registers are erased by
   measurement for zero Toffolis (HMR/reset + del_and); no full replay needed.
   This is why the verifier accepts "dirty" measurement garbage.
3. In-place omega: triangular transform puts omega into syndrome words 1-35,
   workspace in 36-70; the reverse pass restores S exactly.
4. Psi route: sigma split as E^2 + zO^2 (squaring linear in char 2); half-degree
   Euclid computes Psi = 1/sigma' mod sigma once per shot (~44kT); sweep per point
   = root test + flag-gated pairing (~23T x 4024).
5. zlin: phase mask interpolated into deg-70 poly U; interpolated part is a fixed
   syndrome parity -> pure Z gates, kills 71 of 4095 sweep points.
6. Phase contract: Tr(nu_j . omega/sigma') applied directly; the error word never
   materializes.
7. Recent merges: zero-extra product retention (Hikawa), multi-target C-merge
   (boristown), HW-27 patches -> e208d350 at 365,890T / 890q / 0 errors.

### Headroom assessment
- Q^2 dominates the cost: 890^2 = 792,100 vs T = 365,890. One qubit ~= 830 T.
  Syndrome input alone needs 840q, so the decoder overhead is only ~50 qubits —
  but dropping 20 qubits saves ~35,000 cost units (~4.5%). THE biggest dqi lever;
  needs registers to overlap the input in ways the current architecture doesn't.
- T: retention/merge scanners keep finding 0.3-1% per release (each merged pair
  is ~-40T to -1,000T). Diminishing but real: our own scanner on the op stream is
  worth building (boristown's window-scan idea, generalized).
- R(eps): already 0 errors — nothing to gain.
- Private gain realistic: 2-5% via qubit-diet + T-squeeze; top teams' ~85x implies
  ~290-310e9 total, i.e. ~10-15% below public — consistent with BOTH a q-diet
  (870q?) and T-cuts.

## Task 3 — chem / dfthcsos (FeMoCo qubitization)

### Technique stack
1. Factorization: DF-THC on the public coded-word table 906fa220 (R x B words);
   all accuracy slack is spent — e1 = e2 = 0.02 (caps), E_gap at floor,
   lambda_eff 46.9216 via corank-2 reduced-gradient weight refit (fixed words).
2. Loaders (the compiler win): 15 sequential ANGLE loaders rebuilt on XOR
   differences of consecutive word batches; cached address monomials, shared
   parities, don't-care padding rows, CNOT-layer compression ~64%;
   k0 = 292T, k1 = 287T, all = 4,022 Toffolis per pass.
3. Cost breakdown (T = 14,706): angle loaders 8,044 (55%), inner state-prep 4,095
   (28%), Givens rotations 2,250 (15%), outer prep 317 (2%).

### Headroom assessment
- lambda_eff is DONE — at the caps. Only T and Q remain.
- T levers, ranked:
  a) Loader compiler: 4,022/pass is the biggest single term. Beating it needs a
     better QROM synthesis (windowed synthesis, cross-loader monomial sharing,
     HMR-eligible AND cleanup inside loaders). ~5% T cut = chem 12.0x alone.
  b) Fewer words: inner-prep ~4,095T and word count both scale with R*B. A fit
     with fewer total words under the caps = own-factorization work targeting
     word count rather than e2. Longer timeline.
  c) Q: 427 = 124 persistent + 303 temporary (brot<=8 floor). brot>=9 -> temp 607.
     Q is near its floor unless inner-prep is redesigned.
- Realistic private gain: 3-8% (294.6M -> 275-285M) via loader/compiler work;
  this is where tensorqrom.py (marculera) vs maoxin's compiler diverges — study both.

## Cross-cutting observations

- All three circuits converge to the same meta-technique: find reversible-AND
  structure that repeats or can be parked, erase by measurement. The general tool
  = an op-stream scanner finding "free qubit windows" and "reused AND terms".
  Building OUR OWN scanner (on any of the 3 circuits) is the natural first
  original contribution: it is publishable, and any hit is a circuit we own.
- The dqi verifier redraws shots per file hash -> a changed file gets a new draw.
  Tail-risk: a circuit that fails rarely can score worse than a slower clean one.
- Frontier movement: +0.25 GM/week from public releases. Private top ~31.7 implies
  ~10-15% below public on at least two tasks — consistent with private q-diets
  (dqi ~870q or invsqrt ~42-43q) plus compiler wins on chem.

## Candidate research directions (ranked by headroom/effort)

1. AND/retention scanner (generalized boristown): scan .kmb op streams for
   repeated CCX(a,b,t) windows with provably-|0> spares; also repeated *products*
   (C-merge opportunities). Target: dqi T and invsqrt T. Effort: days. Payoff: -1%
   per few hits; publishable tool.
2. Chem loader compiler study: reproduce tensorqrom vs maoxin 4,022 loaders; find
   a synthesis that cuts the per-pass cost (windowed tables, shared monomials
   across the 15 loaders, HMR inside loaders). Effort: 1-2 weeks. Payoff: -3-5% T.
3. invsqrt 43-qubit feasibility: map which scratch registers could borrow x-bits;
   the correction stage's working set is the blocker. Effort: research. Payoff:
   ~2% if a borrow scheme exists.
4. dqi qubit diet: overlap decode workspace with syndrome registers after each
   syndrome word's last use. Effort: high (needs the full decoder structure).
   Payoff: ~4.5% per -20q. This is probably what the top teams did.
5. Own chem fit for word count: ladder2 paused at (9,156,60) e2 13.13 mHa —
   viable only if the end shape has fewer R*B words than the public table.
   Otherwise redirect effort to (2).

## Op-stream audit results (2026-11-20, tools/and_scan.py)

Built a .kmb parser + AND-retention scanner implementing the generalized
boristown rule (repeated CCX(a,b) windows, control-stability, provably-|0>
spare detection, condition-stack awareness). Results on the v76 frontier files:

- invsqrt (20,526 ops, 44q): ONE candidate window at ops 20455-20483
  (CCX(q11,q21)->q18, HMR-erase, recompute). REJECTED on safety analysis:
  the erase runs under condition bit b205 while the recompute runs under
  b205 AND b475. Parking the AND unconditionally leaves a dirty ancilla in
  the b205&~b475 trajectory, and SWAP(21,18) @20522 would smear it into a
  live register. The frontier authors already handled this conditional case.
- dqi (56.85M ops, 890q): 349,407 CCX ops, 20,579 repeated (a,b) pairs,
  324,687 windows; 330 pass control-stability. ALL 281 same-target windows
  have t read as a control inside the window = genuine compute/use/uncompute
  blocks, already tight. 49 diff-target windows all have t1 disturbed.
  Zero usable slack.
- chem angle loaders (15 circuits, ~265 CCX each): zero candidate windows.

CONCLUSION: the public frontier is measurement-uncomputation SATURATED.
Op-stream redundancy tricks are exhausted; remaining headroom is structural
(qubit count, loader synthesis, decoder architecture), per the ranked list
above. The scanner itself is validated (it did isolate the one real invsqrt
window and correctly classified it) and is reusable/publishable tooling.

## Chem loader decomposition (2026-11-20)

Structure of each `angle_k` loader (128q: sel[11] + word[5x8] + 74 ancilla):
- 74-node shared-monomial AND-tree (ancilla bank q51-124, R-init / HMR-erased
  with CZ_IF fixups - textbook MBUC).
- ~189 terminal applications CCX(f,g->w_j) where f,g are recomputed linear
  forms on q126/q127. Symbolic linear-form simulation proves ALL 189 products
  are DISTINCT - zero sharing headroom at the application layer.
- Truth-table extraction (bit-parallel GF(2) sim over all 2048 sel inputs):
  every address yields ~20-bit output; each monomial covers ~216 rows avg.
  Output functions are sparse ANF (~4.7 terms/bit vs ~500 for random fn) -
  the coded-word table was designed for low multiplicative complexity.

CONCLUSION: the loader is at/near the algebraic optimum FOR ITS ENCODING.
Beating it means changing the table encoding (different word coding -> lower
ANF mass) or a full AND-DAG resynthesis search. Both are multi-week research
projects, not scan-and-patch wins.

## Where this leaves the 29-30 push

Verified exhausted: op-stream AND-retention (all 3 tasks), loader
application-layer sharing, per-row flag schemes. Remaining real levers:
1. dqi qubit diet (890q, ~830T-equiv/qubit) - needs decoder workspace overlap.
2. invsqrt 44->43q (borrow scheme for correction-stage working set).
3. chem encoding-space search (refit+coding search for lower ANF mass).
4. External: manifest watch for fresh public artifacts.

## Addendum — rotation-layer refit audit (negative result)

Extracted the full linear structure of the invsqrt phase layer (tools/angle_refit.py):
- theta(x) is exactly linear in the 48 Z_POW angles; per-shot coefficient rows s_j(x) ∈ {0,2}
  extracted by finite-difference probing through km.Simulator.
- Phase error is deterministic per x (identical across seeds/shots at current angles).
- Full-x margin scan (3.13M shots, ~8s): true floor = 38,858,846 units at x=219317;
  ~70 erratic binding x's total (231, 882, 219317, 232688, 465819, 764711, 860500,
  877268, 879153, 894210, 901791/2, ...).
- Cutting-plane LP (scipy HiGHS) to maximize min-margin: every solution overfits the
  sampled constraints and catastrophically violates unprobed x (half-turn errors).
  The 48-term basis is at/near its compiler-set optimum — no refit slack.
- Cost-model note: rotation synthesis contributes ~239.6 T-equiv/shot (33 rots @
  eps=2e-13); margin -> eps -> rotT = 0.56*log2(1/eps)+5.3. Doubling margin saves
  only ~4.6 T-equiv (~200 cost); removing a rotation saves ~7.3 T-equiv but the
  error floor can't absorb it.
- CONCLUSION: invsqrt rotation layer is exhausted like the op-stream layer.
  Remaining headroom requires NEW basis terms (rotations on new functions of x) =
  compiler-level rebuild, not patching.
