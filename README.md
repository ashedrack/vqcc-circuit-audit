# Verifiable Quantum Circuit Compilation — Research Workspace

Research and tooling for the Kaggle competition
`google-verifiable-quantum-circuit-compilation` (Google Quantum AI).
Score = geometric mean of (baseline cost / submitted cost) across three
circuits: `invsqrt`, `dqi`, `dfthcsos`. Current submission: **28.23498**
(v77, Nov 2026) — the strongest verified public portfolio.

## What's original here

`tools/` — a `.kmb` binary reverse-engineering and audit suite:

- **`and_scan.py`** — vectorized op-stream scanner detecting reusable-AND /
  measurement-based-uncomputation windows across multi-million-op circuits.
- **`kmb_edit.py`** — a `.kmb` binary editor (parse → transform → re-emit;
  byte-identical round-trip verified).
- **`prune_sweep.py`** — exhaustive single-gate deletion audit with
  simulator-backed correctness probes.
- **`angle_refit.py`** — rotation-synthesis cost analysis: extracts the exact
  linear phase basis of a circuit's `Z_POW` layer and runs minimax-LP
  margin optimization against all 3.13M verification inputs.

`frontier_review.md` — the audit writeup: decomposition of the three
frontier circuits' architectures (invsqrt sqrt+Newton+LP-rotation pipeline,
dqi truncated-BM decoder with measurement erasure, chem shared-monomial
QROM loaders) and proofs that each is saturated at the op-stream and
rotation layers. Conclusion: further gains require new circuit generation
(compiler-level work), not patching.

`dqi/` — an original Reed–Solomon decoder implementation in kickmix
(Berlekamp–Massey + omega transform + Chien sweep + phase application,
verified end-to-end on small-m tests; backward replay in progress).

`dfthcsos/` — an original DF-THC factorization fit pipeline
(ALS + L-BFGS, quantization-aware refits, lambda-stage gauge optimizer).

## Status

| Task | Deployed | Mechanism |
|---|---|---|
| invsqrt | 74,383.66 (27.9x) | boristown `ecf908fe` — #1 public |
| dqi | 345,236,065,276 (~70.8x) | boristown `4471b49` — #1 public |
| dfthcsos | 294,642,171 (11.4x) | ozer28 set — #1 public |

The audit establishes that these artifacts are op-stream-optimal; the
28→31 gap is architectural and requires original circuit generation.

## Layout

- `data/verifiers/` — official verifiers (competition data, git-excluded)
- `public/` — reference artifacts from public notebooks/datasets (git-excluded)
- `AGENTS.md` — running research log (verifier commands, baselines, findings)
- `frontier_review.md` — audit findings and remaining headroom analysis
- `tools/` — the audit suite described above
