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
- Remote check: `d3779b6e54304b26` (pending).
- Snapshot: `attempts/attempt_03.py`; attempt 2 also saved separately.
- Checks spent: 3; expected remaining: 7. Scored submissions spent: 1.

## Attempt 3 result and scored submission

- Check `d3779b6e54304b26`: PASS 5/5, 1.3440x, CI 1.3238-1.3642, no flags.
- User explicitly approved one scored submission of `attempts/attempt_03.py`.
- Scored job `e0f116ab0de94c44`: PASS 5/5, **1.3355x**, CI 1.3155-1.3556, no flags. Rank 1040 of 1316.
- Its `content_hash` equals the check's (`9629a886...`), so the scored run is the code that passed the check.
- `submission.py` remains at attempt 2; the scored run used the frozen attempt 3 file. Scored runs left today: 3. Checks left: 7.
