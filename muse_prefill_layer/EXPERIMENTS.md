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

- Source: submission.py, server content key (not raw-file SHA256):
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

## Attempt 3 - fix FP8 scale load type

- Same grouped BF16 expert candidate as attempt 2, with other=0.0 for the
  masked FP8 scale load instead of integer other=0.
- Added tools/compile_offline.py. All five custom Triton kernels compile for
  SM90 with Triton 3.5.1, including all RoPE/cache-scale variants. Compiler
  outputs and caches stay inside .tools/; interpreter reused read-only with
  bytecode writing disabled. This validates code generation, not correctness.
- Source SHA256: 450ad5ddcc02c26de02923eeb52358c372319f9f28163c0a02ec1d68067c8a80
- Check: c2db90db6bc943ac. Result: PASS five tests and eight timed shapes;
  speedup 0.917684x (95% interval 0.903918-0.931449), score 422154.1 us.
  No flags. Three checks used, seven left, two reserved.
- Profile: grouped GEMMs take ~11.5 + 5.7 ms; expansion only ~1.5 ms.
  This resembles work over padded capacity, including nonlocal rows, rather
  than only active routes. Investigate runtime-count scheduling before another check.
- Not submitted: no demonstrated speedup.

## Attempt 4 - compact routes and GPU prefix-sum scheduling

- Replace the padded per-row group-ID layout with DeepGEMM's PSUM layout:
  a GPU histogram counts all local routes, prefix scan gives each expert's
  aligned begin and actual end, and atomic row allocation packs every route.
- PSUM ends are computed fresh on every call. Tensor capacities remain based
  only on shapes. Every selected local expert is retained, with an inverse
  slot mapping for the same weighted per-expert normalization.
- Zero only per-expert padding rows, including the post-SiLU buffer. No
  input-value-based host branches, synchronization, or persistent data caches.
- All custom kernels passed offline SM90 compilation under Triton 3.5.1.
- Source SHA256: 427bce95be9d9e96d5645bdeec154d60a8883b3477e18f871096b4edec3bdbcf
- Reason for check: attempt 3's ~17 ms grouped GEMM cost matches the padded
  capacity; PSUM scheduling uses actual expert ranges to avoid this work.
- Check: df0174a2efbc4524. Result: PASS five tests and eight timed shapes;
  speedup 1.138786x (95% interval 1.121704-1.155868), no flags.
  Score time 340190.3 us vs baseline 387403.9 us.
  Four checks used, six left, two reserved.
- Archived byte-identical source in attempt4.py; raw SHA256 matches the source
  hash recorded before check 4. OKC server content keys differ from raw-file
  SHA256 for all four attempts; they should not be compared directly.
- User explicitly approved submission of attempt4.py.
- Scored submission: 77805d4b2f504d81, PASS five tests/eight timed shapes.
  Speedup 1.139429x (95% interval 1.122337-1.156520), no flags.
  Score 339998.4 us. Exact approved SHA256 verified before submit.
  One scored submission used; four remain today. No further submission approved.

## Attempt 5 - expansion occupancy and medium-prefill threshold

- Keep submitted attempt4.py immutable. Candidate is attempt5.py.
- Expand NVFP4 using 2048 elements / eight warps (was four warps). Offline
  SM90 compilation reports 32 registers, no spills, versus 56 registers at
  four warps. Larger 4096/four and 8192/eight variants used 127-128 registers.
- Use the correct compact PSUM grouped-expert path above 2048 tokens (was
  4096), testing the 3012-token prefill load. The same compute and precision
  are preserved; no new arithmetic kernels are introduced.
- Reason for check: expansion costs ~1.44 ms on long prefills; medium prefills
  spend ~2.2 ms in Marlin and may benefit from grouped BF16 GEMMs.
- Source SHA256: d2d9ac293c8f1106eb3b3711e0042e066a46180ab0618d29bca896515f11ecad
- Check: e10cdfade5d847af. PASS five tests/eight timed shapes, no flags.
  Speedup 1.110263x (95% interval 1.093609-1.126917), slower than submitted
  attempt 4 (1.139429x). Not promoted or submitted. Five checks used.
  See per-shape/profile results before reusing either change.

## Attempt 6 - contiguous FP8 cache for full-attention prefills

- Independent candidate based on submitted attempt 4, retaining its expert path.
- Gather the paged K/V cache into contiguous logical sequence order inside
  kernel_fn, after writing new entries to the original cache. Copy as uint8
  bytes, preserving every FP8 value exactly; no scaling or rounding changes.
- On full layers, call FA3 with unpaged cache and the original device sequence
  lengths and query offsets. Sliding layers retain the submitted paged path.
- All allocations depend only on shapes. The full context is gathered; FA3's
  original causal mask and scales are retained. No sparse/block skipping.
- Rationale: upstream FA3 uses PagedKVNonTMA for one-token pages; contiguous
  cache enables TMA. Profile full-attention costs reach 4.5-8 ms on long prefills.
- Source: https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_api.cpp
- Offline SM90 compilation passed for gather with page sizes 1 and 64.
- Source SHA256: b32e6e28dfe51c47f9b5186af724ddd2edb3615812bf5789d68fd910511727cc
- Check: c5442121b8454919. PASS five tests/eight timed shapes, no flags.
  Speedup 1.142620x (95% interval 1.125481-1.159759). Gain over the
  submitted 1.139429x is ~0.3%, inside timing uncertainty. Not submitted.
- Gather costs ~10.9 us per request; full-attention kernel remains the dominant
  cost. Retain attempt 4 as the submitted/default version.
- Six checks used, four left, two reserved.
