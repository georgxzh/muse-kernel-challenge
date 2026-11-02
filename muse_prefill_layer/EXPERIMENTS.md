# muse_prefill_layer experiments

## Scope and rules (2026-11-02)

- Work on master, only under muse_prefill_layer/. Other chats own other problems.
- AI-assisted entries are allowed by https://openkernelchallenge.dev/rules.
- User limits: 10 checks/day; always preserve two unused. Five submissions/day,
  explicit user approval before EVERY okc submit. Reset is 00:00 UTC.
- H200 SXM; lmsysorg/sglang:dev-muse-spark-open; single Python submission.
- prepare(weights): up to 300 untimed seconds. Preserve NVFP4/BF16 weight values
  and precision. Router FP32 weights may use the permitted BF16 hi/lo split.
- Timed kernel_fn must support CUDA graphs: no host sync, no tensor-value-based
  Python control flow or allocations, no caching inputs/outputs between calls.
- Preserve BF16 activations and FP8 KV cache. Queries may use FP8 E4M3. Compute
  every causal key in the full context or entire 2048-token sliding window.
- Correctness: routing constraints; cache 8% relative L2; residual update and
  next attention input have shape-dependent FP8-query bounds (roughly 3-15%).
  Finite outputs and worst-row bounds (5x global) are also checked.
- Timing: alternating CUDA graphs, fresh inputs, cold L2, paired median ratios;
  geometric mean over three weighted loads; 95% interval at least +/-1.5%.
  Total job timeout: 15 minutes, compilation included.
- CLI: /home/georg/.okc/bin/okc (not on noninteractive WSL PATH).
- Read TASK.md, AGENTS.md, starter, baseline, reference, checks and NVFP4 helper.

## Attempt 1 - fused QK/cache and prefill Marlin alignment

- Source: submission.py, SHA256:
  7a91299a10e1a546c40bd6038dec3e486086320b1eb220b2b9d15822e2a7e60d
- Start from supplied SGLang baseline. Retain FA3, dense projections, residual
  norms, router hi/lo split, NVFP4 Marlin weights and expert execution.
- Fuse per-head QK RMSNorm, optional interleaved RoPE, FP8 query conversion,
  and FP8 K/V cache writes in one Triton launch. Preserve intermediate BF16
  rounding and cache scales. Physical cache slot zero remains valid.
- Align using 32 local expert bins instead of 256. Use M=32 expert tiles for
  <=2048 tokens; retain baseline selection for longer prefills.
- Static validation: Python compilation passed. Pinned upstream alignment
  source confirms -1 denotes nonlocal slots and uses an additional bin.
- Check: b29d810082d0432c, queued 2026-11-02 16:38 UTC.
- Result: PASS all five tests and all eight timed shapes; 0.994040x baseline
  (95% interval 0.979129-1.008950). Score time 389726.8 us vs baseline
  387403.9 us. No flags. One check spent, nine left, two reserved.
- Profile: Marlin expert GEMMs take about 10.5-11.6 ms on 16K prefills
  (43-66% of GPU time). Fused QK/cache is about 30 us. This candidate is
  correct but has no demonstrated speedup; do not submit it.
- Submission: none; permission has not been requested.

## Next decisions

- Read check profile before spending another check.
- If correct and faster, recommend immediate submission and ask the user.
- Investigate skipping invalid alignment slots and Marlin block choice first.
- If expert GEMMs dominate, evaluate on-the-fly NVFP4 dequantization with BF16
  grouped GEMMs; never persist expanded BF16 expert weights in prepare state.

## Attempt 2 - grouped BF16 experts for long prefills

- Keep attempt 1's fused QK/cache. Restore baseline Marlin M tiles on short
  prefills, since M=32 did not improve measured short-prefill performance.
- For >=4096 tokens, expand original NVFP4 experts to BF16 INSIDE kernel_fn,
  pack routed activations into expert-aligned rows, and use DeepGEMM BF16
  grouped GEMMs with FP32 accumulation. No expanded weights persist in state.
- Align to 128 rows. Nonlocal blocks are represented by layout -1 and skipped.
  Pack builds an inverse route-slot map; finalize reads grouped results directly,
  applies each expert's post-norm and the original top-8 routing weights.
- No attention, routing, or expert work is approximated or omitted.
- Static validation: Python compilation and all 16 E2M1 nibble values passed.
- Reason for check: profile shows 10.5-11.6 ms in Marlin on 16K prefills; timed
  dequantization plus BF16 WGMMA should reduce this despite expanded-weight I/O.
- Source SHA256: 95aadfaf306a85c41ed419abdd765f6937aa65299def80b7889b74274693629d
- Check: 85ebdf91944c47a4. Result: FAILED at the 4096-token test.
  Triton compilation rejected integer other=0 in the FP8 scale load:
  cannot cast int32[2048] to fp8e4nv. Four smaller tests passed. No timing
  result or flags. Two checks spent, eight left, two reserved.
- Next fix: use a floating-point zero fallback for the FP8 scale load.
- Initial requests failed with transient network errors; using
  GODEBUG=http2client=0 resolved the HTTP/2 protocol failure.
