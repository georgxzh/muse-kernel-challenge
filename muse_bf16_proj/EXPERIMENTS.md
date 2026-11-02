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
