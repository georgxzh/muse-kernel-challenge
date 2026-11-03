# muse_moe_w4a16 experiments

Work is confined to this problem on `master`. Other chats share the repository.

## Rules and budget

- Official rules read 2026-11-02: https://openkernelchallenge.dev/rules.
  AI-assisted entries are allowed. Entries are Apache-2.0 licensed.
- One Python submission with `kernel_fn`; untimed `prepare` may repack weights
  for up to 300 seconds. Keep NVFP4 weights and BF16 activations; FP32 reductions.
- Every timed operation must support CUDA graph capture and fresh input tensors.
  No input-dependent host synchronization, output caching, or harness manipulation.
- Correctness: BF16 `[M, 4096]`, relative L2 <= 0.015 globally and <= 0.075
  per row with the reference's near-zero floor; finite outputs.
- Scored M: 1, 8, 16, 32, 128, 256. Additional checks: 4, 33, 257,
  skewed 128, and 1. Cold-L2, alternating CUDA graph replays, paired median
  ratios; fixed graph overhead removed. Score aggregates decode/verify loads.
- CLI initially reports 10 checks and 5 scored submissions available today.
  Preserve at least 2 checks. Ask the user before **every** `okc submit`.
- No local GPU. Local validation is static; correctness/timing use remote checks.

## Attempt 1 — local alignment and fused post norm

- Retain supplied SGLang Marlin NVFP4 packing, both W4A16 GEMMs, BF16
  intermediates, and FP32 reduction.
- Align using the 32 local experts rather than 256 global experts; mapped
  nonlocal IDs remain -1, supported by SGLang's alignment contract.
- Replace eager post norm for M < 32 with a single Triton kernel; use the
  same fused implementation at larger M.
- Python syntax compilation: passed.
- Remote check: `2897485f56e34715`: PASS, all 5 tests and 6 timed shapes.
- Speedup: **1.131714x**, 95% CI [1.114739, 1.148690]; score 11883.30 us.
- Per-shape SGLang / candidate (us): M1 76.94 / 54.70; M16 195.98 /
  162.95; M32 204.05 / 206.40; M8 146.64 / 116.77; M128 470.19 /
  469.92; M256 634.35 / 634.55.
- Test relative L2 errors 0.00527–0.00560, safely below 0.015.
- Profile: GEMMs consume 78–96% of GPU time. Next experiment should tune
  Marlin row blocking. Recommend submitting this known-correct improvement;
  user approval required before running submit.
- Snapshot: `attempts/attempt_01.py`; profile saved under `profiles/`.
- Checks spent: 1; remaining: 9. Scored submissions spent: 0.

## Attempt 2 — smaller Marlin row blocks

- Same computation as attempt 1; choose block M=8 for token batches <=32,
  otherwise M=16. Baseline selected 16, 48, 64 at M=32, 128, 256 respectively.
- Rationale: baseline's tile heuristic includes remote routes; smaller tiles
  should reduce padding and improve occupancy for sparse local assignments.
- Python syntax compilation: passed.
- Initial check request failed before creating a job (verified via `okc jobs`).
  Retried with `GODEBUG=http2client=0` to avoid the HTTP/2 protocol failure.
- Remote check: `71abb00c320f4ebb`: PASS, all 5 tests and 6 timed shapes.
- Speedup: **1.245736x**, 95% CI [1.227050, 1.264422]; score 10795.62 us.
- Per-shape SGLang / candidate (us): M1 80.35 / 57.58; M16 184.99 /
  152.74; M32 203.26 / 207.49; M8 141.73 / 112.80; M128 467.42 /
  343.24; M256 623.52 / 472.62.
- Smaller row blocks clearly help large verification batches. M32 does not
  improve; retain this as a tuning target. Snapshot `attempts/attempt_02.py`.
- An isolated Python/Triton 3.5 offline compiler was installed entirely under
  ignored `.tooling/`; `tools/offline_compile.py` compiles for SM90 without a GPU.
- Checks spent: 2; expected remaining: 8. Preserve at least 2.

## Submission of attempt 1

- User explicitly approved one submission of `attempts/attempt_01.py`.
- Scored job: `83fa44c28b254feb`: PASS, **1.130308x**, CI [1.113354, 1.147263],
  score 11898.08 us; all 5 tests and 6 timed shapes passed.
- Scored submissions spent: 1 (4 remaining).
- Attempt 1 commit: `7514239`, pushed to `origin/master`. HTTP 408 on initial
  pushes was resolved by `git -c http.postBuffer=524288000 -c http.version=HTTP/1.1 push`.

## Attempt 3 — fused local route alignment

- Builds on attempt 2's row blocks, without changing arithmetic.
- A Triton CTA per local expert reads the route IDs, computes a masked
  32-bin histogram, derives disjoint padded offsets, and scatters stable
  pair indices. CTA 0 also writes mapped local IDs and the padded total.
- Nonlocal routes have no blocks in the GEMM schedule. The norm and SiLU
  still mask nonlocal slots, so their unwritten intermediates are not read.
- Fill/scatter within a CTA has an explicit barrier. Cross-CTA writes are
  disjoint; downstream kernels wait on stream order.
- Buffer capacity is min(N * BM, N + local_count * (BM - 1)), where N=M*8.
  Both expressions bound the sum of per-expert padded counts for any routing.
- Python syntax compilation: passed. No local GPU runtime available.
- Remote check: `d3779b6e54304b26`: PASS, 1.344003x, CI [1.323843, 1.364163].
- Snapshot: `attempts/attempt_03.py`; attempt 2 also saved separately.
- Checks spent: 3; expected remaining: 7. Scored submissions spent: 1.

## Attempt 3 result and scored submission

- Check `d3779b6e54304b26`: PASS 5/5, 1.3440x, CI 1.3238-1.3642, no flags.
- User explicitly approved one scored submission of `attempts/attempt_03.py`.
- Scored job `e0f116ab0de94c44`: PASS 5/5, **1.3355x**, CI 1.3155-1.3556, no flags. Rank 1040 of 1316.
- Its `content_hash` equals the check's (`9629a886...`), so the scored run is the code that passed the check.
- `submission.py` remains at attempt 2; the scored run used the frozen attempt 3 file. Scored runs left today: 3. Checks left: 7.

## Attempt 3 verification and promotion in this chat

- Our offline Triton 3.5 SM90 compile passed for routing M=1,4,8,16,32,33,
  128,256,257 and the fused post norm; no local GPU was used.
- Check `d3779b6e54304b26`: all 5 tests and 6 timed shapes pass, no flags;
  score 10006.31 us. Per-shape baseline / candidate us: M1 80.50 / 51.02;
  M16 187.92 / 146.39; M32 210.26 / 203.79; M8 153.25 / 114.73;
  M128 459.60 / 313.42; M256 623.73 / 433.19.
- Another process committed the frozen candidate and recorded scored run
  `e0f116ab0de94c44` with user approval in commit `6dcba10`. This chat did
  not launch that submission and will not duplicate it.
- Promoted the checked frozen attempt 3 to `submission.py`.
- Confirmed CLI quota: 7 checks and 3 scored submissions remaining.

## Attempt 4 - use 8-row blocks at all sizes (regression)

- Based on attempt 3, change the large-batch row block from 16 to 8.
- Offline SM90 compilation passed all shapes; Python syntax passed.
- First check request failed DNS lookup without creating a job; retry job
  `cbc1221a72494149` passed all 5 tests and 6 timed shapes.
- Speedup **1.219725x**, CI [1.201429, 1.238021], score 11025.84 us.
- Per-shape baseline / candidate us: M1 77.04 / 48.91; M16 189.10 /
  148.58; M32 206.77 / 198.69; M8 142.27 / 105.17; M128 458.67 /
  389.22; M256 622.70 / 628.86.
- Larger batches clearly regress versus attempt 3. Increased weight reads
  outweigh reduced row padding. Do not promote or submit this candidate.
- Snapshot `attempts/attempt_04.py`; verified active entry remains attempt 3.
- Checks used: 4, remaining: 6. Scored submissions used: 2, remaining: 3.

## Attempt 5 - adaptive sparse/busy expert blocks (prepared)

- Route experts receiving <=16 tokens to 8-row GEMMs and busier experts to
  32-row GEMMs. Small token batches retain attempt 3 exactly.
- One GPU alignment launch builds both schedules. Each expert belongs to
  exactly one group. Four GEMMs write disjoint rows in shared BF16 buffers;
  SiLU runs once after both gate/up GEMMs; norm runs after both down GEMMs.
- All intermediate rounding and FP32 reduction settings remain unchanged.
- Python syntax and offline SM90 compilation passed, including M=33,65,
  128,256,257 adaptive routing. Offline helper now supports both schedules.
- First remote request failed connection reset. `okc jobs` confirmed no
  job was created; successful retry: `a511776d5176408f`.
- PASS: all 5 tests and 6 timed shapes; **1.322713x**, CI [1.302872, 1.342554],
  score 10167.36 us. M128 345.52 us vs 472.22 baseline; M256 445.46 us
  vs 617.81 baseline. This does not beat attempt 3, which stays active.
- Marlin's minimum MMA row tile is 16. The small group wastes weight reads
  for experts with 9-16 tokens; next attempt should use 16 for that group.
- Checks used: 5; remaining: 5. Scored submissions remaining: 3.

## Attempt 6 - adaptive 16/32-row blocks

- Fix attempt 5's small group to use one 16-row tile for 9-16 tokens.
- Experts with <=16 tokens use BM16; other experts use BM32. Token batches
  <=32 keep the already-verified attempt-3 path.
- Python syntax and offline SM90 compilation passed, including nearby M=65.
- Remote check: `845df6a7fadd4c2a`: PASS, all 5 tests and 6 timed shapes.
- Speedup **1.366962x**, CI [1.346457, 1.387466], score 9838.24 us.
- M128 318.28 us vs 470.10 baseline (1.477x); M256 403.19 us vs 618.50
  baseline (1.534x). Overall gain over attempt 3 is modest and CIs overlap.
- Promote this to active `submission.py`; recommend a scored submission
  after asking for explicit approval. Snapshot `attempts/attempt_06.py`.
- Checks used after launch: 6; remaining: 4.

## Attempt 7 - adaptive 16/64-row blocks

- Explore more weight reuse for the very busiest experts: <=32 tokens use
  BM16; >32 use BM64. Small token batches still retain attempt 3.
- NVFP4 weights, BF16 activations, and FP32 reductions are unchanged.
- Python syntax and offline SM90 compilation passed for every target size.
- Remote check: `c2def1a570fd4209`: PASS, all 5 tests and 6 timed shapes.
- Speedup **1.349124x**, CI [1.328887, 1.369361], score 9968.32 us.
- M128 325.76 us vs 467.87 baseline (1.436x); M256 419.49 us vs 617.81
  baseline (1.473x). Slower than attempt 6 on both large scored shapes;
  retain attempt 6. Snapshot `attempts/attempt_07.py`.
- Checks used after launch: 7; remaining: 3. At most one more check may be
  used today; two checks remain reserved. Scored submissions remaining: 3.

## Attempt 8 - 16-row tiles for M=16 and M=32 decode

- Based on attempt 6. Keep BM8 for M<=8, use BM16 for 9<=M<=32,
  and retain adaptive BM16/BM32 for larger token batches.
- Rationale: a local expert receiving 9-16 rows otherwise rereads weights
  in two BM8 tiles despite the hardware's minimum 16-row MMA tile.
- Python syntax and offline SM90 compilation passed for the actual small
  batch row-block choices and all adaptive large-batch sizes.
- Remote check: `95c5b10e2f414f91`: PASS, all 5 tests and 6 timed shapes.
- Speedup **1.362464x**, CI [1.342027, 1.382900], score 9870.72 us.
- M16 151.66 us vs 189.39 baseline (1.249x), below attempt 6's 1.281x;
  M32 204.66 us vs 213.54 baseline (1.043x), similar to attempt 6.
- No overall improvement over attempt 6. Retain attempt 6 as the final entry.
- Checks used: 8; **2 checks reserved. No more checks today.**
- Attempt 6 remains active. User explicitly approved one scored submission
  of frozen `attempts/attempt_06.py`; job `aa4935c3095c450d` passed all 5 correctness tests and 6 timed shapes.
- Final scored speedup: **1.361666x**, CI [1.341241, 1.382091],
  score 9876.50 us. The checked frozen attempt 6 was submitted unchanged.
- Scored submissions used: 3; remaining: 2.

## Final selection

- Active `submission.py` exactly matches frozen `attempts/attempt_06.py`.
- Best check: **1.366962x** SGLang, 95% CI [1.346457, 1.387466].
- Eight remote check attempts all passed 5 correctness tests and 6 timed
  shapes. Tested tile/routing variations also compiled offline for SM90.
- Precision stays NVFP4 weights, BF16 intermediates, FP32 GEMM reduction.
  Only weight packing runs in prepare; fresh routing and activations are
  computed entirely within CUDA-graph-capturable timed calls.
- Daily checks used: 8/10. The final 2 checks are reserved.

## Final scored result

- Approved scored attempt 6: `aa4935c3095c450d`, PASS, **1.361666x** SGLang.
- Check speedup 1.366962x and scored speedup 1.361666x agree within their
  reported confidence intervals. No further tuning or submissions planned today.
- Active entry: `submission.py`, byte-identical to `attempts/attempt_06.py`.
- All eight attempts are logged and committed with result evidence.
- Budget: 8/10 checks used, **2 reserved**; 3/5 scored submissions used,
  **2 remaining**. Every scored submission launched by this chat had explicit
  user approval; the separate attempt-3 submission is documented above.

## Offline profile review (UTC 2026-11-03, no checks spent)

- Board: scored 1.3617x (aa4935c3095c450d, attempt 6), rank 1029/1361,
  leader 2.76x. The user dropped the 2-check reserve (boards close 07:59 UTC Nov 5).
- Marlin is 87-96% of GPU time on every scored shape. Against the task's HBM
  floor, attempt 6 reaches: M1 23%, M16 36%, M32 36%, M8 36%, M128 32%,
  M256 27% of floor bandwidth. Alignment/SiLU/norm are only 3-10 us in total.
- Marlin tuning is exhausted: row blocks 8/16/32/64 and adaptive grouping
  (attempts 2-8) all land at 1.32-1.37x. use_atomic_add is not an option:
  it reduces through BF16 atomics, coarser than the FP32 reduction we keep.
- Only remaining large lever: replace Marlin with a skinny NVFP4 W4A16 MoE
  kernel for small M. Sketch: reuse _align_local_routes BM16 blocks; one CTA
  per (16-row expert block, N tile, K split); load native NVFP4 bytes and FP8
  scales; exact PRMT E2M1 decode with the same FP32 SF*GS product and RN BF16
  rounding as muse_prefill_layer attempt 9 (checked bit-exact path); tl.dot
  BF16 with FP32 accumulation; FP32 split-K reduction; BF16 outputs as now.
  Codex's prefill tiled prototype fits in 168-254 regs at BM=64 without
  spills, so BM=16 should be well within budget.
- Estimate: at ~55-60% of floor on M1/M8/M16/M32 the board would reach about
  1.7-2.0x (geomean over the six loads); the same kernel would also replace
  Marlin in muse_decode_layer (45-52% of its larger shapes). Cost: several
  hours of build plus multiple blind checks (no local GPU).

## Attempt 9 - native NVFP4 skinny kernel for M<=32 (FAILED timing recheck)

- File attempts/attempt_09.py (submission.py untouched; stays attempt 6).
- For M<=32: _align_local with 16-row blocks; one Triton kernel per GEMM,
  grid (16-row expert block, 64-column tile), K loop of 128 values. Native
  NVFP4 bytes (sliced to the local 32 experts in prepare, no precision change)
  are decoded exactly by muse_prefill_layer attempt 9's PRMT decoder, split
  into low/high nibble outputs that feed even-K and odd-K BF16 tl.dot with FP32
  accumulation. Same BF16 intermediates, SiLU and fused post norm. M>32 keeps
  attempt 6's Marlin path unchanged.
- Offline SM90: 108 registers, 0 spills, 12 KB shared, mma.sync generated
  (attempts/offline_skinny_09.txt, tools/compile_skinny.py).
- Check 1a0734c7fca341e7 (UTC 2026-11-03 check 1): all 5 correctness tests
  PASS (bs=1 and bs=4 run the new kernel; rel L2 ~0.0057, same as Marlin), but
  timing FAILED: decode-1 "recheck failed on repetition 2: relative L2 error
  8.621e-01". No speedup reported.
- Root cause: _align_local_routes uses NP=next_pow2(N) lanes. With BM=16 and
  bs=1 (N=8), a padded segment of 16 slots had only 8 filled with the sentinel;
  the other 8 kept stale row indices from earlier calls, so other routes'
  rows were overwritten under graph replay. Previous paths used BM<=N.
