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
