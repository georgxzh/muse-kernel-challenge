# muse_bf16_proj experiments

## Operating constraints

- Work only in this folder on shared `master`; stage and commit only this problem's files.
- User deadline: two days from 2026-11-02. Establish a correct speedup early, request permission for each `okc submit`, then improve.
- AI assistance is allowed. Read the live https://openkernelchallenge.dev/rules and `okc --help` on 2026-11-02. The live rules explicitly discuss agent submissions; the homepage supplies an agent workflow.
- CLI/account: `/home/georg/.okc/bin/okc` in Ubuntu WSL; already logged in as `georgxzh`.
- CLI reported 10 checks and 5 scored runs per day for this problem, resetting at 00:00 UTC. Always retain two checks. At most eight checks may be used from a fresh daily allowance.
- H200 SXM, single Python source, existing image dependencies only, no prebuilt binaries or obfuscated code. Compilation counts toward the 10-minute run limit; prepare has a 300-second limit.
- Preserve BF16 weights and activations. Router FP32 weights may use BF16 hi/lo with FP32 accumulation. Lossless weight repacking is allowed.
- Relative L2 <= 1% (BF16 projections), <= 0.01% (router); per-row <= 5x bound; single-nonzero probe rows <= 0.1%; no NaN/Inf; exact output dtype and shape.
- Graph-capturable timed calls, no host synchronization or value-dependent allocation/host control flow. All activation-dependent computation stays inside the timed call. No cached outputs, harness access, reference imports, networking or child processes in the submission.
- Timed cold-L2 CUDA graph replays alternate with the SGLang baseline; graph fixed cost is subtracted. Score is geometric mean of six weighted-load speedups; use reported confidence intervals.
- No local GPU or local torch/triton installation. Local Python syntax checks cannot establish GPU correctness.
- Windows sandbox launcher fails before process creation. Commands run through the approved elevated execution fallback and use WSL Git, avoiding Windows Git ownership configuration changes.

## Attempt 1 - conservative split-K Triton

- Candidate: `submission.py`; check job `c4232f44cf7e460e`.
- Small projection batches (m <= 32): BF16 tensor-core tiles (16 x 64 x 64), FP32 split-K partial sums and a separate reduction. Split 8 for K >= 4096, otherwise 4.
- Larger BF16 batches: `torch.mm`, matching the baseline's cuBLAS strategy.
- Router m <= 8: FP32 weight GEMV reductions; larger m: fused BF16 hi/lo dot products, FP32 accumulation.
- Rationale: increase GPU occupancy for skinny matrices without precision changes; retain the established larger-batch path.
- Local validation: Python syntax passes. Remote result pending; one check requested, no scored submissions.
- Result: PASS, 9/9 pretests and all 30 timed shapes; no static flags. Overall speedup 1.0472675x (95% CI 1.0315585-1.0629765), score 4612.766 us against 4830.8 us SGLang step time.
- Profile: router improves 1.789x / 2.610x at m=1/8 and 1.305-1.339x at m=16/32/128. The projection split-K path regresses most at m=32 (o 0.693x, unpooler 0.719x); restore cuBLAS there before more aggressive tuning.
- Frozen checked source: `attempts/attempt01.py` (byte-for-byte copy). Recommend early scored submission of this file, pending the user's explicit approval.
- Daily budget after this check: 1/10 checks used, 9 left (2 reserved); 0/5 scored runs used.

## Attempt 1 scored submission

- User explicitly approved submitting attempt 1.
- Scored job `194f20ddef5446e0` submitted from frozen `attempts/attempt01.py`; result pending.

## Attempt 2 - measured dispatch and router occupancy

- Restore cuBLAS for qkv_gate and o, and for larger unpooler/pooler batches where attempt 1 regressed.
- Keep split-K for small pooler and sliced unpooler, which showed modest improvements.
- Router: use hi/lo tensor cores for m >= 2, narrower N tiles, larger M tiles for large batches, and split-K chosen to provide roughly a full GPU wave. Tiny m=1 still uses full-FP32 GEMV.
- Rationale: remove observed projection regressions and improve the underoccupied router (attempt 1 had only 32 CTAs at m=16).
- The exact checked attempt 1 remains untouched in `attempts/attempt01.py`.

## Attempt 3 - lossless BF16 exponent packing

- Independent follow-on based on frozen attempt 2; candidate `attempts/attempt03.py`.
- Store sign + seven mantissa bits in one byte and exponent codes in packed nibbles (12 bits/weight for the common path). Code zero escapes to the original BF16 weight; reconstruction is bit-exact, with no rounding or skipped values.
- Prepare derives all packing only from supplied weights; `.item()` is confined to untimed prepare. Timed calls decode on GPU and still consume fresh activations.
- CPU validation: `validate_packing.py` exhaustively reconstructs all 65,536 BF16 bit patterns at nine exponent bases, and all pairs of exponent nibbles: PASS. This includes zeros, subnormals, infinities and NaN bit patterns via the exact escape representation.
- Larger M tiles avoid loading weights twice at m=32; K=1024 uses a single pass directly into the output, eliminating split reduction.
- Larger projection batches retain cuBLAS. Router path remains attempt 2's independent tuning.
- Local Python syntax passes. Remote check requested (third check); result pending.

### Attempt 1 scored result

- Job `194f20ddef5446e0`: PASS, speedup **1.0388608x**, 95% CI **1.0232779-1.0544437x**. The first correct, faster-than-baseline leaderboard entry is secured.
- Difference from its 1.0473x check is within the reported uncertainty.
- Exactly the frozen source `attempts/attempt01.py` was submitted after explicit user approval.
- One scored submission requested/used; four of five daily scored submissions remain. Two follow-on checks remain queued.

### Attempt 2 result

- Check job `54d88712677b4685`: PASS, all correctness and timed cases; **1.1270414x** SGLang, 95% CI **1.1101358-1.1439471x**.
- Frozen checked source: `attempts/attempt02.py`. Main `submission.py` currently matches this version.
- Removing measured projection regressions and improving router occupancy gives a clear improvement over the first scored 1.0389x entry. Recommend submitting attempt 2 with explicit user permission.
- Third check `faf23be693794770` remains queued. Total checks requested today: 3; seven left, including the two-check reserve. Four scored submissions remain.

### Attempt 2 scored request

- User explicitly approved submitting attempt 2; job `7708134c1ceb40e5` submitted from frozen `attempts/attempt02.py`.
- The attempt 2 commit `3b0c284` was pushed successfully after retrying a transient GitHub connection reset.

## Attempt 4 - contiguous transposed weights for larger cuBLAS batches

- Candidate `attempts/attempt04.py`, based on verified attempt 2.
- Prepare stores a lossless contiguous transpose of each BF16 weight. Batches above 32 use cuBLAS with this layout; small batches and the router retain attempt 2.
- Rationale: large projection batches currently match the baseline; alternate exact weight layout changes cuBLAS algorithm selection without new arithmetic or precision risk.
- This experiment is independent of attempt 3's lossless compression. Local Python syntax passes. Fourth check requested; six checks remain including two reserved.

## Results for attempts 2 (scored), 3 and 4

- Attempt 2 scored `7708134c1ceb40e5`: PASS, **1.1292x**, CI 1.1122-1.1461. Same `content_hash` as check `54d88712677b4685`.
- Attempt 3 check `faf23be693794770`: FAILED 3/9 pretests, no timing. Every test reaching the packed path hit a judge compile error (`ttg.local_alloc` i1 in `TritonGPUReduceDataDuplication`). Local Triton 3.5.0 and 3.5.1 both compile the same kernel, so a local compile pass is not a gate for the judge. Packing is parked: by estimate 68-80% of 128x64 tiles contain an escape.
- Attempt 4 check `9be65792f3f44eb8`: PASS, 1.1357x, CI 1.1187-1.1527, overlapping attempt 2. Per shape against the mean of the two attempt 2 runs: pooler m=128 +12.1%, unpooler m=128 +4.3%, unpooler m=256 +2.4%, pooler m=256 +1.2%, o within 1.3%, qkv_gate m=128 -2.8%, qkv_gate m=256 -6.3%. Not adopted whole.
- Noise between the two attempt 2 runs on unchanged shapes: mean 1.2%, maximum 2.9%.

## Verified-better rule (set by the user, 2026-11-02)

A candidate is verified better when its changed shapes beat noise (over 3%), no other shape drops by more than 3%, and overall speedup is higher, even if confidence intervals overlap. Scored submissions still need explicit user approval each time.

## Attempt 5 - three changes on disjoint shapes

- Candidate `attempts/attempt05.py`, built from attempt 2. `submission.py` stays at attempt 2 until this is verified better.
- pooler/unpooler at m>32: contiguous transposed weights (attempt 4's gain, without its qkv_gate and o changes).
- Router at m>128: BM=128, split 8, 8 warps, so weights are read half as often as with BM=64, split 4.
- qkv_gate at m<=32: `_tiled_mm`, a bit-exact repack into contiguous 64x64 tiles. Row-major control is attempt 1 (same tiles at m<=16): 14.19, 14.59, 14.78, 15.27 us at m=1, 8, 16, 32. m=32 uses BM=32 here, against attempt 1's BM=16.
- Local validation (scratchpad only): real kernel code run on CPU through a mock of `triton.language`, scored by `reference.py`: 39/39 on all 30 scored shapes and 9 pretests, with attempt 2 as control at 39/39. All stored weight copies reconstruct bit-exactly. Offline compile passes on Triton 3.5.0 and 3.5.1. This covers correctness only, not speed, and does not rule out a judge compile failure.
- Check `632584e6061b49a9` requested with user approval. Fifth check of the day; five left, two of them reserved. Three scored runs left.

### Attempt 5 result

- PASS 9/9 and all timed shapes, no flags: **1.1513x**, CI 1.1341-1.1686, overlapping attempt 2's 1.1122-1.1461.
- **Not verified better under the rule; `submission.py` stays at attempt 2.** Per shape against the mean of the two attempt 2 runs:
  - Changed shapes over +3%: router m=256 +35.5% (16.4 to 12.3 us), pooler m=128 +11.8%, unpooler m=128 +5.8%, qkv_gate m=1 +4.3%, qkv_gate m=8 +3.7%.
  - Changed shapes under +3%: qkv_gate m=16 +2.6%, pooler m=256 +2.5%, unpooler m=256 +2.3%.
  - Changed shape regressed: qkv_gate m=32 -5.9%.
  - Unchanged shape over the limit: router m=1 -4.1% (-4.8% against the scored run). Its code path is unchanged, but this exceeds the 2.9% seen between attempt 2 runs, so noise is likely but not established. Against the scored run alone, router m=8 (-3.4%) and unpooler m=1 (-3.3%) also cross 3%.
- Tiled layout against the row-major control (attempt 1, same tile parameters at m<=16): speedup up about 9-10% at m=1, 8, 16. m=32 is not a clean control because BM differs. Still far from the qkv_gate floor (13.1 us against 7.9 us).
- Transposed weights at m=256 were small but positive in both attempt 4 and attempt 5.

## Verified-better rule, group reading (user update, 2026-11-02)

Changed shapes count as one group. A candidate is verified better when the group gains overall, no single shape regresses by more than 3%, and overall speedup is higher, even if intervals overlap. Scored submissions still need explicit user approval.

- Attempt 5 under this rule: **still not verified better.** The group gains +6.4% by geometric mean, but qkv_gate m=32 (-5.9%) and router m=1 (-4.1%) each drop more than 3%.

## Attempt 6 - attempt 5 without qkv_gate m=32, tiling extended

- Candidate `attempts/attempt06.py`, built from attempt 5. `submission.py` stays at attempt 2.
- qkv_gate m=17-32 returns to cuBLAS (attempt 2's path), removing the -5.9% regression.
- Tiled weights extended to pooler m<=16 and unpooler m<=8, replacing their row-major `_split_mm` path with identical tile parameters (BM=16, BN=64, BK=64, split 8).
- Router and transposed-weight changes are unchanged from attempt 5.
- Local validation (scratchpad only): 39/39 under the CPU mock with `reference.py`; all stored weight copies bit-exact; offline compile passes on Triton 3.5.0 and 3.5.1.
- Check `97c95b793b9a4867` requested with user approval. Sixth check of the day; four left, two of them reserved.

### Attempt 6 result

- PASS 9/9 and all timed shapes, no flags: **1.1524x**, CI 1.1351-1.1697, overlapping attempt 2's 1.1122-1.1461. Level with attempt 5's 1.1513x.
- **Not verified better as the rule is written; `submission.py` stays at attempt 2.** The 13-shape group gains +4.7% and overall speedup is higher, but one shape crosses -3% on each baseline: o m=256 -3.3% against the attempt 2 mean, qkv_gate m=128 -3.4% against the attempt 2 scored run. Neither shape's code changed from attempt 2.
- Identical code has spread that much on those shapes: o m=256 ran 1.021, 0.990, 0.976, 0.972 across attempts 2, 2, 5, 6; qkv_gate m=128 ran 1.010, 1.024, 1.014, 0.989.
- Reproduced gains: router m=256 +30-35% (attempts 5, 6); pooler m=128 +10-12% (attempts 4, 5, 6); unpooler m=128 +4-6% (attempts 4, 5, 6).
- Tiling gained less than estimated. pooler m<=16 +2.6-3.3%, unpooler m<=8 +0.6-2.7%, against an expected 10%. qkv_gate m<=16 fell back from attempt 5's +2.6-4.3% to -0.3 to +2.6%, on identical code, so its gain over cuBLAS is not established. Against the row-major control it is about 6-10%.
- Overall result missed the 1.17x estimate.

### Attempt 6 scored run

- The user treated the two -3% shapes as noise, since their code is unchanged, and approved one scored run. The board keeps each team's best run.
- Scored job `c0a7d84f19104a4a`: PASS 9/9, no flags, **1.1466x**, CI 1.1294-1.1638, rank 1363 of 1645. Same `content_hash` as check `97c95b793b9a4867` (`f3a900a0...`).
- Above attempt 2's 1.1292x, so `submission.py` now matches `attempts/attempt06.py`. The intervals overlap.
- Per shape against the attempt 2 mean: group +4.4%. router m=1 -3.7%, the only shape past -3%. Against the attempt 2 scored run alone, router m=1 -4.5%, qkv_gate m=16 -3.5%, m=32 -3.3%, m=128 -3.2%.
- router m=1 uses unchanged code but ran 1.853, 1.881, 1.791, 1.844, 1.797 across attempts 2, 2, 5, 6, 6. Its two lowest readings are in the two runs of this session's later code, so a small real loss is not ruled out.
- Reproduced again: router m=256 +34.9%, pooler m=128 +10.4%, unpooler m=128 +3.6%.
- Budget after this run: 4 checks left (2 above the reserve), 2 scored runs left.

## Continuation after UTC quota reset (2026-11-03 UTC / Nov 2 Toronto)

- Reconciled newer committed attempts 5/6 before editing. Current verified main is attempt 6, scored 1.1466x (commit e629891). Preserve it while checking new candidates.
- Server now advertises 20 checks and 10 scored runs, all unused for this problem since reset. Retain the user's stricter original cap of 10 checks, at most eight used and two reserved; request permission for every scored submission.

## Attempt 7 - one cuBLAS GEMM for the two router products

- Candidate `attempts/attempt07.py`, based on current verified attempt 6.
- For router m>32, concatenate hi/lo weight columns once in prepare as exact BF16 [K,2N]. One BF16-input, FP32-output cuBLAS GEMM computes both products; a small Triton kernel adds the FP32 halves.
- Algebra matches the baseline's two separate hi/lo products. Activations are neither duplicated nor rounded, original FP32 router remains available, and all activation-dependent work is timed.
- Rationale: use optimized cuBLAS for larger router matrices and improve occupancy by doubling output width; avoid two GEMM launches and Triton's repeated tiled weight loads. All other shape paths are byte-for-byte unchanged from attempt 6.
- Local Python syntax passes. First check requested after the UTC reset; source has no new GPU compilation complexity except a masked FP32 add kernel.

### Archival completeness

- Archived our original attempt 3 source and exhaustive bit-level packing validator after reconciling the intervening work; result remains compiler failure, not a correctness or speed result.
- Archived our original attempt 4 source and measured text profile: PASS 1.1356860x, CI 1.1186507-1.1527213; layout gains were already selectively incorporated in the later committed attempts.
- These archival commits do not modify verified `submission.py` or consume new checks.
