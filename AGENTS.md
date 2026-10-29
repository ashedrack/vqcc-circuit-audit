# Google Quantum AI - Verifiable Quantum Circuit Compilation (Kaggle)

Competition: `google-verifiable-quantum-circuit-compilation`. Score = geometric mean over the 3 tasks of
(baseline cost / our cost). Higher is better; the sample submission scores 1.0.

## Layout
- `.venv/` Python 3.12 with `kickmix` 1.0.0 (PyPI), numpy, scipy, kaggle CLI.
- `data/` competition data (git-excluded via `.git/info/exclude`): `verifiers/`, `sample_submission/`, paper PDF.
- `Qualtran/` upstream repo clone (kickmix docs in `Qualtran/docs/kickmix/`).
- Kaggle auth: `~/.kaggle/access_token`. Never commit tokens.

## Verify (run from `data/verifiers/`)
- invsqrt: `../../.venv/bin/python invsqrt/verify_submission_invsqrt.py FILE.kmb --shots-per-x 16 --batch-size 64`
  (~3 min on 8 cores; full = 64 shots/x, batch 256). num_shots must be a multiple of batch_size (max_x = 4 * 783363).
- dqi: `../../.venv/bin/python dqi/verify_submission_dqi.py FILE.kmb`
- dfthcsos: `../../.venv/bin/python dfthcsos/verify.py RUN_DIR/` (~80 s)

## Baselines (lower cost is better)
- invsqrt: Q=260, C=7987.6 (7735 Toffoli + 37 rotations), cost Q*C = 2,076,774
- dfthcsos: lambda_eff=49.30, T_total=93,237, Q_total=730, cost = 3.3556e9
- dqi: see `/tmp/dqi_baseline.json` once computed

## Our invsqrt work (invsqrt/build.py, invsqrt/sweep.py, tools/quick_invsqrt.py)
- Lookup (k=9, 384 used entries) + quadratic Horner with split accumulators, binary normalize/denormalize.
- v2 officially verified: Q=141, C=2731, 385,095 (5.39x). Best so far (quick check): Q=127, C=2440 (6.7x).
- `tools/quick_invsqrt.py` (~1 s) under-reports worst error; trust `model_error` from build.py, then run the full verifier.

## Public state (2026-10-28, from public notebooks; pull with `kaggle kernels pull <ref> -p public/<dir>`)
- Public portfolio GM ~22.3-22.6; LB leader 26.14.
- invsqrt public best: 63 qubits, C=1334.5, cost 102,664 (20.2x) - busyaprime / mrmeows line.
- dqi public best: 889 qubits, 4.1478e11 (58.9x) - heliosli (Schur decoder, jday96314 line).
- dfthcsos public best: 3.4771e8 (~9.65x) - somejoint R8 B143 + Nilanjan column gauge. Weakest task;
  open lever: brot <= 7 halves cost-model temp qubits but needs a quantization-aware refit under the 20 mHa gates.
- Pitfall: `Circuit.to_file(path)` writes text kmx unless `format='kmb'`; the LB rejects text named `.kmb`.

## Submit
Zip `invsqrt.kmb`, `dqi.kmb`, `dfthcsos/` at the archive root:
`.venv/bin/kaggle competitions submit -c google-verifiable-quantum-circuit-compilation -f submission.zip -m "msg"`

## dfthcsos original-factorization effort (2026-10-28)

Goal: an end-to-end factorization+circuits pipeline that is entirely our own work.

- `dfthcsos/target.py` — packed-pair target T; `error_2` == verifier epsilon_fro (float64).
- `dfthcsos/fit_als.py` — alternating fit (exact CG G-step + Adam U-step + L-BFGS joint refine).
- `dfthcsos/seed_from_baseline.py` — top-rows-by-||W|| seed; `dfthcsos/seed_cluster.py` — cluster-then-truncate seed (better: 1610 vs 2883 mHa at 10x129x60).
- `dfthcsos/compress.py` — OUR compression chain: drop weakest -> exact pair-merge (lossless)
  -> copy-truncate (eig of G_r) -> L-BFGS refit -> drop+refit loop to target R.
- `dfthcsos/pad_b.py` — pad group rows to B=129+ (donor baseline rows, then noise) + refit.
- `dfthcsos/wrefit.py` — quantization-aware W-refit vs decoded angle words (our protected refit).
- `dfthcsos/lambda_stage.py` — H2_I/gauge lambda_eff optimizer (verifier-exact metrics).
- `dfthcsos/to_official.py` — fit npz -> verifier factorization npz.

Cost-model facts (verifier heuristic_costs, brot<=8):
- inner-prep temp qubits: B<=128 -> ~575, B>=129 -> 304 (cliff at 129; flat to ~256).
- brot>=9 doubles temp to ~607. C flat ~303-305 through C=90; inner T ~95/copy.
- baseline score 3.3556e9. Public best chem ~3.5e8 (9.58x): R10/B129/C44, lambda_eff 47.55,
  T 17k via optimized loaders (our plain QROM is ~3x more Toffolis -> compiler is next lever).
- Recipe insight (public README): merge->drop+refit->progressive-quantize->lamopt chain keeps
  error low throughout; fitting at target shape from scratch does not converge.

### Gotchas learned 2026-10-28 (pm)
- decode->encode of angle words is NOT idempotent (~68% of words change). Always carry
  quantized state as angle_words, never re-encode decoded vectors for submission.
- init_lookup iterates the full 2^nbits table (~2^k Toffolis) - our manual unary_iteration
  over exactly A addresses is cheaper per circuit (A-1 Toffolis).
- optax.lbfgs in float32 hangs (zoom line search). Adam+f32 fine (~6x faster than f64).
- pq works: freeze lowest-|dE/dtheta| angles in stages + Adam on free angles + exact CG
  G-step for W; early stages improve error below continuous start.
- Naive rounding to brot=8 destroys ANY factorization (+140 mHa on baseline and ours).

## dqi own decoder (ORIGINALITY-MAX path, started 2026-10-29)
- `dqi/model.py`: classical BM+Chien+Forney over GF(2^12), validated 6/6 on random errors.
  `bm_oblivious` = data-oblivious variant (implicit shift via Bx, one flag branch) - validated.
- `dqi/build.py`: kickmix port SKELETON (incomplete - muxes/L-updates are TODO stubs).
- Full reversible design notes are at the END of model.py (v2): key trick = conditional
  overwrites via CSWAP into per-iter scratch slots (self-inverse); garbage -> D/F banks,
  cleaned by backward replay. Est ~4M Toff, ~3000q (~0.8-1.1x). Chien per-position ~800T.
- gf2_phase_by_product applies the MBUC phase for FREE once e_j decoded.
- chain3 grinding R30->R13 in bg: R30@16.2 -> R28@20.9 -> R27 in refit (~22-23 proj).
  Drop cost ~+2.5/drop; e-stop ~30 -> compression ceiling ~R24-26. That gives own chem ~2.5-3x.
- LB: 22.68 placeholder (v10 public portfolio, rank ~422). Top 26.14 needs private work.
