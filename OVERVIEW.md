# The Big Picture — what this project actually is

For someone who has never seen this before. Five minutes, no quantum
physics required.

## 1. The competition in one picture

Google Quantum AI posed three puzzles on Kaggle. Each puzzle says:
"here is a quantum computation that real applications need — write the
cheapest possible circuit that provably does it." A circuit is a list of
instructions for a quantum computer, and "cheap" means few expensive
operations and few working bits (qubits).

```
                 ┌────────────────────────────────────────────┐
                 │   Google Quantum AI — VQCC competition     │
                 │   score = geometric mean of 3 task scores  │
                 └────────────────────────────────────────────┘
                        │              │              │
                 ┌──────┴───┐   ┌──────┴───┐   ┌──────┴─────┐
                 │  invsqrt │   │   dqi    │   │  dfthcsos  │
                 │ apply a  │   │ decode a │   │ load a     │
                 │ 1/√x     │   │ Reed–    │   │ chemistry  │
                 │ phase    │   │ Solomon  │   │ molecule's │
                 │          │   │ code     │   │ angles     │
                 └──────┬───┘   └──────┬───┘   └──────┬─────┘
                        │              │              │
                 44 qubits        890 qubits      427 qubits
                 ~74.4k cost     ~345e9 cost      ~295M cost
                        └──────────────┴──────────────┘
                                       │
                              submission.zip → Kaggle
                              our score: 28.23× better
                              than the sample baseline
```

Cheaper circuit → higher score. The leaderboard top (~31.7) is held by
private circuits nobody has published.

## 2. What a `.kmb` file is

Each submission contains circuit files in a custom binary format called
`.kmb` (kickmix binary). Think of it as a *compiled executable for a
quantum chip*: packed instructions like "multiply these two bits into
this scratch bit" (a Toffoli — the expensive instruction), "flip this
bit", "measure and erase". The verifier simulates the file on millions of
inputs and checks: correct answer, all scratch bits returned to zero,
phase within tolerance — then prices it.

## 3. The two kinds of work in this repo

```
        ADOPT (the assembly line)          ORIGINATE (the lab)
   ┌──────────────────────────┐     ┌──────────────────────────┐
   │  public circuits from    │     │  our analysis tooling    │
   │  Kaggle notebooks are    │     │  (tools/)                │
   │  Apache-2.0; we track    │     │                          │
   │  every release, verify   │     │  • and_scan.py — finds   │
   │  locally, and submit the │     │    reusable structure    │
   │  best-of combination ────┼────►│  • kmb_edit.py — binary  │
   │                          │     │    editor for .kmb       │
   │  24.94 → 28.23 in weeks  │     │  • prune_sweep.py — gate │
   │                          │     │    deletion auditor      │
   │                          │     │  • angle_refit.py — LP   │
   │                          │     │    refit of rotations    │
   └──────────────────────────┘     └───────────┬──────────────┘
                                               │
                              ┌────────────────┴────────────────┐
                              │  THE AUDIT VERDICT              │
                              │  every patchable layer of the   │
                              │  public frontier is saturated:  │
                              │  no dead gates, no unused AND   │
                              │  windows, rotation basis at its │
                              │  LP optimum, qubit allocation   │
                              │  provably tight                 │
                              └────────────────┬────────────────┘
                                               │
                              frontier_review.md → plan_31.md
                              (what must be BUILT new to reach 31)
```

## 4. How the pieces connect

```
public artifacts (.kmb) ──► tools/and_scan.py  ──► "where is the slack?"
        │                                          verdict: none left
        │                  tools/kmb_edit.py  ──► edit & re-emit circuits
        │                                          (byte-identical round-trip)
        │                  tools/prune_sweep.py ──► delete each gate,
        │                                           simulate, count survivors
        │                                           verdict: 0 of 1,549
        │                  tools/angle_refit.py ──► extract phase-error
        │                                           linear basis, LP-refit
        │                                           verdict: at optimum
        │
        ├──► dqi/build.py  (our own Reed–Solomon decoder — long-horizon)
        ├──► dfthcsos/*.py (our own chemistry factorization pipeline)
        └──► plan_31.md    (the quantified road map to ~31)
```

## 5. Why this matters beyond the leaderboard

Every artifact doubles as evidence of original engineering: a
reverse-engineered binary format, an exhaustive negative-result audit,
verifier-backed proofs, and an original decoder implementation. The repo
(`ashedrack/vqcc-circuit-audit`) is the public record;
`frontier_review.md` is the writeup; the Global Talent evidence file
points at all of it.

## 6. One-paragraph summary for a non-specialist

> Quantum computers run programs called circuits; running them is
> expensive, so Google ran a public competition to shrink three important
> ones. We assembled the best publicly available circuits into a
> submission scoring 28.23× better than baseline — top of the public
> leaderboard. Then, instead of just resubmitting others' work, we built
> our own tools to dissect those circuits at the binary level and proved
> — exhaustively — that every easy optimization is already taken. That
> audit both documents what the frontier got right and maps exactly where
> a genuinely new circuit could still win. The tools and findings are
> published as original research artifacts.
