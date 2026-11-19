# Frontier Circuit Audit Tools

Analysis and binary-editing tooling for `.kmb` quantum circuits from the
Google Quantum AI "Verifiable Quantum Circuit Compilation" competition.
Built to answer one question: *where does optimization headroom remain in
the strongest public circuits?* The answer, established exhaustively, is
that every op-stream and rotation-layer slack is already extracted — the
frontier is compiler-saturated.

## `.kmb` binary format

Kickmix SOA layout: 16-byte magic + version/qubit/bit/register/op-count
header, then typed packets (0=registers, 1=op types, 2-9=SOA target arrays,
10=angles, 11=info tags). See `Qualtran/docs/kickmix/` for the spec.

## Tools

| Tool | Purpose |
|---|---|
| `and_scan.py` | Parse the op stream (vectorized, ~25s on 57M ops) and detect reusable-AND / measurement-based-uncomputation windows: repeated `CCX(a,b)` pairs, control stability, target disturbance (phase-only gates excluded), provably-`|0>` spare qubits, HMR-erase patterns. |
| `kmb_edit.py` | Read-modify-write `.kmb` files: parse to rich op records, filter/transform, re-emit valid SOA binary (byte-identical round-trip, including the empty info-tag packet). |
| `prune_sweep.py` | Single-gate deletion audit: remove each CCX, re-emit, and test with a kickmix-simulator probe (x-restoration, scratch cleanliness, phase tolerance). |
| `angle_refit.py` | Rotation-layer analysis: decode/encode the 16-byte fixed-point angle packet; simulate phase errors per x; extract the exact linear basis `theta(x) = sum_j a_j*s_j(x) + phi(x)` via finite differences; minimax-LP margin refit (`scipy.linprog`). |

## Key results (see `../frontier_review.md`)

- **invsqrt** (`d0227f3c`, 44q, 74.4k cost): 1,549 CCX exhaustively deleted-and-probed — zero dead gates. 48-angle linear basis is at its compiler-set LP optimum; true margin floor 38,858,846 units binds at x=219317 and ~70 erratic x's; every refit fails out-of-sample.
- **dqi** (`e208d350`, 890q, 345e9 cost): 330 stable AND windows — all genuine parked-AND blocks. Lifetime analysis: all 890 qubits occupied at peak (840 syndrome payload + 50 workspace recycled ~49,140 logical slots). Allocation-tight.
- **chem loaders** (15 × ~265 CCX): symbolic simulation proves all 189 terminal `f*g` products distinct; ~4.7 ANF terms/output bit vs ~500 random — the word-coding was designed for minimum multiplicative complexity.

## Usage

```bash
../.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, '.')
from and_scan import parse_kmb
ops, nq, nb = parse_kmb('../public/ozer28/data/invsqrt.kmb')
print(ops.shape, nq, nb)
EOF
```
