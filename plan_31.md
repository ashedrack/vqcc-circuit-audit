# Path to GM ~31 (+3 points) — preparation plan

Current: **28.23498** (v77). Target: **~31** = product ×1.331.
Status: PREPARATION PHASE — tooling and measurements staged below; no
execution started. This document is the single source of truth.

## The arithmetic (verified against deployed v77)

| Task | Now | −9.3% (distributed path) | What it means |
|---|---|---|---|
| invsqrt | 74,383.7 (27.92x) | ~67,500 (30.8x) | −6,900 cost |
| dqi | 345.236e9 (70.4x) | ~313e9 (77.6x) | T_exec 366k → ~331k |
| chem | 294.64e6 (11.39x) | ~267.5e6 (12.5x) | −27e6 |

Single-task alternatives: dqi alone needs T_exec 366k→270k (−26%) or
Q 890→764 (impossible — 840 syndrome qubits irreducible). The realistic
route is modest gains on all three at once.

---

## Phase 0 — Measurement infrastructure (DONE / staged)

| Deliverable | Status | Purpose |
|---|---|---|
| `tools/dqi_decompose.py` | written | op histograms per 250k window; register-activity map; dense-CCX multiplier-motif detection |
| `tools/manifest_watch.py` | written | diffs public manifest vs deployed portfolio; flags better artifacts |
| `tools/pack_submission.py` | written | one-command submission.zip assembly |
| `tools/and_scan.py` | done | op-stream AND-window scanner |
| `tools/kmb_edit.py` | done | binary editor, byte-identical round-trip |
| `tools/angle_refit.py` | done | rotation-layer linear-basis extraction + LP |
| `tools/margin_scan` (formalize) | TODO | extract the all-x margin scan into a reusable module |
| `tools/lifetime.py` (formalize) | TODO | qubit lifetime/occupancy analysis (was ad-hoc) |

## Phase 1 — dqi T_exec decomposition (start here)

`score = rep_factor × Q² × T_exec` = 1.19126 × 890² × 365,872.

**Gate G1 (first measurement — DONE, informative negative):**
`dqi_decompose.py` on the 890q file shows:
- 349,407 static CCX concentrated in windows 71–182 (~ops 18M–45M) —
  the Berlekamp–Massey decode region.
- Hottest registers: E70/E73/E72 (~50–58k CCX each, 12 qubits) — the
  GF-arithmetic workhorses; S/E register pairs share activity (in-place
  omega workspace confirmed again).
- **Zero localized multiplier cells**: no dense CCX run stays within
  ≤18 qubits — the compiler interleaves GF arithmetic across the whole
  register set. A block-level Karatsuba swap is NOT available.

Revised Phase-1 consequence: surgical subroutine replacement is off the
table. GF-arithmetic improvement requires either (a) algebraic
decompilation of the multiplier structure across the interleaved stream
(hard), or (b) our own decoder with cheaper arithmetic (Phase 4). The
rep_factor tradeoff (Phase 2) is now the more promising dqi lever —
probe it early.

**1b. Block-catalog:** even without swap, catalog the executed-op
signature per region (BM rounds, omega transform, Psi pass, sweep) —
this is the publishable "anatomy of a 366k-Toffoli decoder" notebook.

## Phase 2 — rep_factor tradeoff (CLOSED: statistical floor)

`eps_plus` is the one-sided Clopper-Pearson bound on OBSERVED failures
at confidence 1−2^-128 over the fixed 102,400 shots. Our circuit had
0 errors → eps_plus ≈ 8.67e-4 → rep ≈ 1.19. This bound is identical
for every zero-error submission — a floor set by shot count and the
verifier's confidence parameter, NOT a design tradeoff. No circuit
change reduces it. Dead lever, closed 2026-11-19.

## Phase 3 — chem shape/coding search

T_total = 14,706 (loaders 8,044 / inner 4,095 / rot 2,250); λ_eff at cap.

- **3a. Shape scan (DONE):** `dfthcsos/shape_scan.py` built on the
  verifier's own `SpinFreeData_V1`. Findings: loader T scales ~3.3
  Toffolis per angle word; deployed uses 1,220 words (8·143+76).
  Heuristic T responds weakly to R/B/C — the real dial is **word count**.
  Since the deployed fit sits at ALL error caps (e1=e2=0.02, gap at
  floor), a fit with ~10% fewer words may still pass → −800 loader T
  ≈ −5% T_total. Gate: run our fit chain at ~1,100 words and check caps.
- **3b. brot=7:** heuristic T drops ~300 (Givens adders get cheaper) but
  needs quantization-aware refit (qat.py) — moderate payoff, known
  risk profile from earlier notes.
- **3c. Coding search:** alternative word-codings → different ANF mass.
  Highest uncertainty — only after 3a's fit lands.

## Phase 4 — invsqrt 43q + own decoder (long-horizon)

- 44→43q = −1,700 cost: needs a borrow scheme for the Newton-correction
  working set. Prep: lifetime map of ops 6760–7230 region
  (correction stage) from and_scan disturbance data.
- `dqi/build.py`: own decoder verified end-to-end on small-m; remaining
  is backward replay + m=4095 compile. Original work, months-scale —
  pursue only if Phase 1 shows the public file is unbeatable at the
  multiplier level (then own-decoder novelty is the only dqi lever).

## Phase 5 — Cadence

- Weekly: `manifest_watch.py` → adopt improvements (free gains).
- Per stage: commit + push to `ashedrack/vqcc-circuit-audit`;
  publish the strongest artifacts as Kaggle notebooks.
- Evidence: log each stage's outputs in frontier_review.md + the
  Global Talent evidence file (section 6).

## Decision summary

| If Phase-1 finds… | Then |
|---|---|
| Schoolbook GF multipliers | Karatsuba swap → likely −10-15% dqi T_exec → GM ~28.7-29; then chem shape scan for the rest |
| Already-optimal multipliers | dqi lever = rep_factor probe → if dead, own decoder is the only 31-path; accept ~29 as realistic ceiling without breakthrough |
