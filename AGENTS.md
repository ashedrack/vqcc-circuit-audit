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

## Lukin-group decoder blueprint (public/lukin_dqi/, from andigu/uncompute repo, public)
Their DQI circuit: 442,397 Toffolis / 1,837 qubits (~12.7x Kaggle score).
Architecture (see public/lukin_dqi/discovered/README.md - full math):
- TRUNCATED shared-path BM: 70 rounds, w=delta/Delta, sig_new=w*sig ^ tau_shifted,
  ONE inversion/round; coefficient windows MEASURED per round on the weight
  distribution (bm_windows in constants.json; cutoff mass <4e-5 -> approx decode).
- Measurement erasure: values computed into FRESH registers get erased by
  measurement (zero Toffolis); reverse pass only recomputes those. kickmix has
  hmr/reset + del_and(measurement uncompute) for this - NO full replay needed.
- In-place omega: triangular transform puts omega in syndrome words 1-35,
  zeros in 36-70 (workspace for euclid cofactors). Reverse restores S exactly.
- Psi route: split sigma=E^2+zO^2 (squaring is LINEAR in char2); sigma'=O^2.
  Half-degree Euclid pass computes Psi=1/sigma' mod sigma ONCE per shot (~44kT);
  sweep per point = root test + flag-gated pairing (~23T x 4024 = 92.5k).
- zlin: interpolate phase mask into deg-70 poly U; interpolated part is a fixed
  syndrome parity -> pure Z gates, no Toffolis; kills 71 of 4095 sweep points.
- Phase contract: Tr(nu_j * omega/sigma') applied directly; error word NEVER exists.
- Failure budget ~1.4e-4 (7x under 1e-3 allowance) = coefficient cutoffs +
  1-entry reversal stack + one Euclidean class. In Kaggle: R(1.4e-4)~1.07.
- Kaggle Q^2*T score punishes their 1837q: public heliosli 889q/58.9x is the
  same method qubit-dieted. Our v1 target ~1900q/~450kT (~12x), then qubit-diet.
- Reference: circuits/discovered/circuit.py (their uncompute DSL) + constants.json
  (searched windows) + gen_constants.py (classical models for the searches).

## dqi decoder status (2026-10-29, WORKING CORE)
- `dqi/build.py`: BM forward VERIFIED in-circuit (4/4); omega transform+restore
  verified; end-to-end phase applied correctly on small-m test.
- Key: sweep banks rotate by gamma^{-i} per position; after full m=4095 cycle
  rotation = identity (self-restoring). Small-m tests show S-restore=False only
  because 8 != 4095.
- Remaining for submission: BM backward replay to clean sig/tau/L/DP/W/D/F
  (~1300q garbage), then real m=4095 compile+verify (~2-3M Toffolis, ~3500q ->
  ~2-4x dqi v1; Psi route + truncation -> ~450kT ~12x v2).
- chain3: R26 refit ~29.8 landing ~29 (ceiling ~R24-26 @ 26-30mHa).

## dqi decoder status (2026-10-29, PHASE WORKING)
- `dqi/build.py`: BM forward + omega + Chien sweep + phase — END-TO-END works.
- Single-error case VERIFIED (phase correct). Multi-error (2+) has a phase bug
  (extra -1 applied; likely spurious root or e-parity issue) — debug next.
- KEY FIX: del_gf2_mul/del_gf2_inverse use X-measure+CZ fixups that inject
  random global phase -> use reset() (Z-measure) for all classical-basis
  garbage; use ccx (not init_and) for reversible AND uncompute.
- Sweep: rotating banks self-restore over full m=4095 cycle (gamma^{-i*4095}=1).
- Remaining: fix multi-error phase, then real m=4095 compile + verify.

## dqi status (2026-10-29, MULTI-ERROR PHASE WORKING)
- `dqi/build.py`: BM+omega+Chien sweep+phase verified on multi-error (2/3 cases).
  Remaining: single-error {0:1} edge case fails (phase not applied - debug next).
- CRITICAL FINDINGS for future work:
  * `reset()`/`hmr`/`del_gf2_*` UNMATCHED inject random -1 phase (X-measure
    w/o fixup). The verifier counts them - NO measurement erasure anywhere.
  * del_gf2_inverse/del_gf2_mul as MATCHED pairs (own init -> own del) are
    phase-safe. ccx (not init_and) for reversible AND uncompute.
  * _const_gf_mul: M^{-1} XOR uncompute (reversible, no measurement).
  * BM per-round garbage (DC, lscr, zr) stays in replay banks; tmp cleared by
    w-recovery (3 case-distinct controlled-XORs). Backward replay still TODO.

## dqi status (2026-10-29, PHASE CORRECT for all in-range cases)
- `dqi/build.py`: BM+monicize+omega+Chien sweep+phase — verified on all cases
  within BM capability (n syndromes -> ~n/2 errors). The 6-error n=8 case
  correctly fails (underdetermined, not a bug).
- Phase-erasure rules learned:
  * reset/hmr/del_*-UNMATCHED inject random -1 -> NEVER use on nonzero regs.
  * del_gf2_inverse/del_gf2_mul as matched pairs (own init -> own del) clean.
  * init_and is NOT self-inverse (uses measurement internally) -> uncompute
    with ccx, never a second init_and.
  * _const_gf_mul: M^{-1} XOR uncompute (reversible, clean).
  * All other scratch: XOR-out/recompute-XOR or keep in replay banks.
- Remaining: backward replay for BM workspace cleanup (sig/tau/L/DP/W/D/F),
  then real n=70/m=4095 compile + verifier.

## dqi bm_backward (IN PROGRESS - has bugs to fix)
- `dqi/build.py`: bm_backward written but has ordering bugs:
  * DC is freed then reused for W[r]/flag recompute - fix order: clear W[r]
    FIRST (needs Delta via disc-recompute), then flag, then DC last.
  * ctrlB uncompute line is mangled (comment says simpler: free into garbage).
  * _shift_poly_right's zr restore direction needs verifying.
- Phase verified end-to-end before replay: BM+monicize+omega+sweep all correct
  for in-range cases (n syndromes -> ~n/2 errors). Replay is for ANCILLA
  cleanup only - does not affect phase correctness, just verifier's |0> check.
- Per-round named banks now: W[r](ratio), D[r](DC), F[r](flag), LS[r](lscr),
  ZR[r](wrap coeff). bm_forward returns them; build() captures.
- Remaining: fix bm_backward ordering, test state restoration (sig/tau/L/DP
  back to init + all banks |0>), then n=70/m=4095 compile + verify.

## dqi status (2026-10-30, replay build - WIP)
- `dqi/build.py`: bm_backward written and VERIFIED standalone: 4/4 state-restore
  cases (sig/tau/L/DP -> |0>) on n=4,{0:1}/{1,3,7,9} n=6,{0,1,3,5} n=8,{1,3} n=8.
- Fixed during session: init_and on nonzero target -> ccx (line ~494); !flag&nz
  stale-ref bug (compute t pre-swap, cswap, recompute-clear); flag-tau inversion
  via tt scratch + recompute-XOR (both flag/nflag branches).

## CRITICAL DEBUGGING LESSONS (verified empirically):
1. alloc_qubits() returns POSSIBLY-DIRTY qubits after prior free()s. Any
   CNOT-into-fresh-register snapshot debug is WRONG unless you builder.reset(dbg)
   FIRST. Several hours of debugging chased phantom corruption that was dirty
   debug regs, not the circuit.
   - `builder.reset(dbg)` on a value-check is fine (phase irrelevant for reads).
2. ixor_gf2_mul is ancilla-clean (compute + immediate uncompute = no leak, probe
   reads 0). The earlier "dirty" probe was from freeing a NONZERO product (our bug).
3. Phase is preserved by: fwd+replay alone, fwd+mon+unmon+replay, fwd+mon+omega+
   restore+unmon. FAILS when all combined (fwd+mon+omega+restore+unmon+z+replay ->
   phase=False). Untraced interaction, likely allocator/dirty-scratch interaction.
   Native phase read (sim.read_phase_flipped) is reliable; CNOT snapshots are not.

## NEXT: trace the fwd+mon+omega+restore+unmon+replay composition. Candidates:
(a) some register in bm_backward/all code is freed non-clean in rare branches;
(b) kickmix allocator hands a dirty qubit to an assumed-zero scratch only under
    heavy churn from omega's ~35-call ixor chains; (c) subtle interplay of
    unrot(_const_gf_mul) + bm_backward. All preceding steps verified cleanly.

## Score-27 exploration (2026-10-29)
- Current verified portfolio: invsqrt 20.22x (102,664), dqi 58.87x (4.148e11),
  chem 9.65x (3.477e8) -> GM 22.68. Target 27 needs product x1.71.
- `public/gluzdov_femoco/` (kaggle ds dmitriigluzdov/vqcc-femoco-public-factorization):
  open_circuits_v*/dfthcsos fits + `circuit_sources/` incl. SomeJoint's mad-dqi
  build tree. mad-dqi `build.py` works after symlinking
  `somejoint/data/comp/kickmix` -> site-packages and `.../verifiers/dqi` ->
  data/verifiers/dqi, and patching os.sched_getaffinity for macOS.
- open_circuits_v40/dqi.kmb verified PASS: cost 4.1027e11 (59.6x), 889q,
  434,512T -> submitted as submission.zip ref 57777774 (~+0.1 GM).
- mad-dqi variants verified so far are all WORSE than the deployed circuit:
  base 881q/1.12M-execT (1.03e12), hankel_half+cyclic_gold 883q/997k static,
  central_circuit+gold_extraction 883q/1.03M static. Deployed 889q/434k config
  not yet reproduced (static T >> executed T: circuit has conditional paths).
- Other chem fits evaluated and REJECTED: r4_b303 (48.75 lam, ~4.3e8),
  r4_b333 (47.50 lam, ~4.4e8), thc_r5 B496/C384 (47.86 lam but brot>=9 ->
  temp 640, ~5.2e8; also e2=0.0199995 at the cap, E_gap at floor).
- public/ozer_v9: invsqrt 20.06x, dqi 877q/482kT (55.28x), chem 9.45x - all
  worse than current. michaelrustad_r10 v5 C44: 3.504e8 (9.58x), also below
  our 9.65x somejoint chem.
- Chem lever notes: T scales with angle-word count D; temp qubits floor at
  304 once B>=129 (brot<=8). lamopt already pushed fits to the e2=0.02 cap.
  Remaining upside needs either a lower-D fit that still meets error gates
  or a cheaper lookup compiler.

## Mid-comp publication (2026-10-31)
- Dataset: shedrackaji/vqcc-2268-portfolio (21 flat files; private - flip public in UI)
- Notebook: kaggle.com/code/shedrackaji/vqcc-portfolio-22-68 (public, competition-
  linked; hash-verifies all 18 files then builds submission.zip; ran COMPLETE on Kaggle)
- kernel-metadata.json in notebooks/; dataset-metadata.json in midcomp_dataset/.
- Kaggle CLI: `datasets create` SKIPS subfolders - keep dataset files flat.
- Kernel slugs must match title slugification exactly (409/400 errors otherwise).

## Chem improvement chain in flight (2026-10-31)
- lamopt_run/: lamopt on our somejoint fit finished - best leff 48.32 (vs our 48.27; wash).
- cont7: cont.py 30k-step continuous refit on cur.npz reached e2 13.9 mHa (cap 19.998),
  still improving. Next: ctrunc C62->C50/C44 (prep T -950/-1424 at same word count),
  re-refit, pq --brot 8 (or 7), finish->gauge->lamopt->canon->to_official->verify.
- drop_group test: R8->R7 dead (e2 jumps to 1600+ mHa). Word count = R*B + 76, so
  only R or B cuts words; C truncation is free-ish in cost model (cuts prep T only).
- qat.py patched to dev=cpu (torch 2.15 CPU-only; MPS lacks float64).

## Hedge-wizardry portfolio (2026-10-31) - current best submission
- manwithacat/vqcc-hedge-wizardry-notebook (public, Apache-2.0) curates ALL public
  circuits via a manifest + verified tweaks. Their submission.zip (kaggle kernels
  output) verified locally: invsqrt bce2948ff24e (62q, 100,568.5 -> 20.65x),
  dqi fe98677c61ef (888q, 3.9827e11, 0 err -> 61.3x), chem HW-04 02f8f7d9d75a
  (marculera C42 R9/B143 fit + slack-spend refit, lam 46.98, 3.310e8 -> 10.14x).
  GM 23.42 -> submitted ref 57848346.
- Their chem base is a DIFFERENT fit (marculera C42, lam 47.08) than ours
  (somejoint R8/B143/C62, lam 48.27). C42: more words (1363 vs 1220) but much
  lower lam and lower prep T (C42 vs C62).
- HW slack-spend technique == our cont.py+lamopt path: spend e1/e2/E_gap margin
  to buy lambda; alternated with column gauge. They got lam -0.22% only.
- Our cont7 refit (e2 13.68 mHa on R8/B143/C62) is a better continuous fit than
  Lera's C42 base (e2 19.98); pq requantization chain in progress, then
  finish/gauge/lamopt/canon + C62->C44 truncation could push chem past 10.14x.
- pq.py 36-stage progressive quantization ~14min/stage (~8h total on CPU).

## C50 chem chain (2026-11-01, in flight)
- cont7 refit died at pq stage 15/36 overnight (sleep kill, no checkpoint).
- Improved path: ctrunc cont7.npz -> c50.npz (C62->C50, kept 97.9% Gram trace,
  prep T -950 at zero word cost). c44.npz (C62->C44, e2 32 -> too far, dropped).
- cont.py c50.npz -> c50r.npz (30k steps): e2 22.8 -> 17.13 mHa recovered
  under cap with ~2.9 slack. C-truncation SURVIVES.
- pq.py c50r.npz -> q8c50.npz brot=8 now running under caffeinate (pid 43940)
  ~8h. Watch e2: starts 17.13, pq adds ~+2.7 over schedule -> expect ~19.5-20.
  After: finish->gauge->lamopt->canon->to_official->rebuild angle circuits->verify.
  If final cost < 331.0e6 (HW-04 chem) -> swap into portfolio, resubmit (>23.4).
- lamopt_run/ files: cur.npz(ours lam48.27), cont7.npz(e2 13.68), c50r.npz(e2 17.13).
- Env: QCC_VERIFIERS=/Users/shed/Desktop/Kaggle-competition/data/verifiers,
  torch 2.15 CPU, qat.py patched dev=cpu, run caffeinate -i to survive sleep.
- q8.npz landed: e2 18.39 mHa (under cap). finish+gauge done: lam_eff 48.597,
  e1 19.16 e2 18.41 Lambda 291.23 (pre-lamopt; HW-04 was 283.8/46.98).
- lamopt g8.npz -> lam8.npz running under caffeinate pid 83639 (20k steps ~50min,
  recipe hyperparams, cap 0.01998, e2 slack ~1.6 mHa to spend on lambda).
- After: finish.py lam8.npz -> gauge -> canon_gauge -> to_official -> rebuild
  angle circuits -> dfthcsos verify. Beat 3.310e8 (HW-04) -> swap -> resubmit.

## HW-05..HW-15 entries (2026-11-04) - manwithacat shipped 11 new circuits
- Re-pulled manwithacat/vqcc-hedge-wizardry (now 222MB) into public/vqcc-hedge-wizardry/.
- New dqi candidates: HW-06..HW-09; best = HW-09 1698e864a3a6 -> 3.788e11 (~64.5x),
  892q, 399,676T, 0 errors (measurement-based uncompute / conditionally-clean flags).
- New chem candidates: HW-05/HW-10..HW-15; best = HW-15 bf016fb054d5 -> 3.1366e8
  (~10.70x), lam 47.02, T=15,585, Q=428 (constrained weight refit + recompiled
  angle loaders on SomeJoint's 15xx-word fit line).
- Our cont7/pq/lamopt chain landed lam_eff 47.52 (~9.8x) - REJECTED vs HW-04.
  C50/C44 ctrunc dead ends (e2 21+/31+ post-quantize).
- v70 submission (ref 57991702): invsqrt bce2948ff24e + dqi HW-09 + chem HW-15,
  local GM ~24.2. Public LB top now ~28.8 (still private method gap).

## v70/v71 submissions (2026-11-04) - adopted syed3000 + HW components
- v70 (ref 57991702): invsqrt bce2948 + dqi HW-09 (3.788e11, 64.5x) + chem HW-15
  (3.1366e8, 10.70x) -> LB 24.24524, rank ~700.
- syed3000/vqcc-own-circuit-portfolio verified BETTER on all three:
  invsqrt 6eea5d21 98,254.9 (21.14x, same 0.9993 error margin as bce2948);
  dqi fc3ba816 891q/396,922T -> 3.7538e11 (~65.1x, 0 errors);
  chem 3.1255e8 (10.74x, loaders 4,436T/pass, lam 47.0).
- v71 (ref 57993046): full syed portfolio -> local GM ~24.5.
- gluzdov v55 (newest open_circuits): dqi 892q/399,933T (worse than syed);
  chem C40 317.3M (worse). v41 has loader compiler SOURCE in source/.
- LB top now ~28.8 (private methods); public-artifact ceiling ~24.5-25.

## 2026-11-06 - rank dropped to 848, recovered via new releases
- v72 (ref 57995252): alexchilton 24.59 portfolio -> LB 24.59420 (rank fell to ~848
  over 2 days as others improved; LB top now ~29.3).
- v73 (ref 58062265): malachiwmoody's EXACT 24.79280 zip
  (public/malachi24/entry_2026_11_05i.pkg, sha ccf0460e) - byte-identical resubmit.
- v74 (ref 58062337): best-of = malachi invsqrt 89c33409 + HW-22 dqi b99bf233
  (375.32e9, 891q, byte-splice of two 891q decoders, -14T) + HW-24 chem ec58b2f5
  (301.88M, lam 46.894, T 15,076, Q 427). Local GM ~24.85.
- HW-16..HW-24 shipped: chem 310.8->301.88M (HW-24 re-places 11 loaders on syed's
  NEW R8xB143xC44 fit eff6a4863f08, D=1220). Malachi chem = their refit on Atakan's
  sign-gauged syed table (304.18M). invsqrt frontier now = kirilltym 0104d1b4 +
  2 exact malachi edits (-0.75 T-eq).
- marculera published FULL dfthc_refit pipeline (public/marculera_merge/): shrink.py
  (merge+fade+LBFGS ladder), quantize.py, tensorqrom.py loader compiler, refit.
  Env: DFTHC_DATA=$PWD/data DFTHC_STARTER=$PWD/data/starter_kit/starter_kit; jax CPU f64.
- Own-work run: runs/ladder1 FAILED (e2 24.1 vs her 13.5 at 143x46 - undertrained,
  defaults 1.2k/4k iters too short). runs/ladder2 RELAUNCHED with --select impact
  --fit 12000 --final 20000 (her real ballpark ~12k-20k iters/step), pid 89820.
  Seed eigen_9x180x90.npz (public/marculera_compact, e2 6.38 mHa continuous).
