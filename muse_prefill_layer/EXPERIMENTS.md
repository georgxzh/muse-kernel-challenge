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
  The 3012-token shapes regressed to 0.924x/0.936x baseline, and expansion
  was ~1.57 ms versus ~1.44 ms. Reuse neither change without new evidence.

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

## Attempt 7 - adaptive local Marlin schedules for short prefills

- Independent candidate based on submitted attempt 4. Only <=1024-token
  expert execution changes; larger prefills keep the submitted paths.
- Borrowed the already committed _write_segment/_align_mixed_routes helpers
  from muse_moe_w4a16/attempts/attempt_06.py as reference, copying them into
  this problem only. No other problem files were changed.
- Experts with <=16 local routes use M=16; busier ones use M=32. Schedules
  omit nonlocal slots completely. Each expert is in exactly one group.
- Two gate/up and two down GEMMs write disjoint BF16 rows. Original SiLU,
  per-expert normalization and routing weights are retained; FP32 GEMM
  reduction is unchanged. This is a scheduling/tile change only.
- Offline SM90 compilation passed for 300, 600, 753 and 1024 tokens.
- Source SHA256: 170fc213ad8c826cf4c8ea87710dedef8be15b46583cbe22ed7adb8a7940248a
- Rationale: short-prefill Marlin still consumes ~65% of GPU time; the same
  scheduling pattern has checked improvements at smaller token batches.
- Check: 9d58ad6b10794c17. PASS five tests/eight timed shapes, no flags.
  Speedup 1.123649x (95% interval 1.106794-1.140503), slower than submitted
  attempt 4. Short shapes are 0.969x/0.972x baseline. Four expert GEMMs
  total ~731 us vs ~700 us with baseline row tiles; not promoted/submitted.
- Seven checks used, three left, two reserved. At most one more check today.

## Attempt 8 - preserve row reuse; tune grouped-GEMM expected size

- Independent candidate based on submitted attempt 4. Short prefills keep
  baseline 64-row Marlin tiles and one pair of GEMMs. Only local alignment
  changes: copied the checked _align_local_routes helper from the committed
  expert problem into this folder and exclude nonlocal schedule blocks.
- Long prefills retain every original compacted route and BF16 computation;
  give DeepGEMM expected_m_for_psum_layout=max(128, tokens//16), a shape-only
  performance hint (2x uniform EP route count per expert). This does not cap
  routing or work: the exact GPU PSUM ends still determine all live rows.
- Rationale: attempt 7 showed that smaller tiles/additional GEMMs regress,
  while allocation capacity greatly overstates typical grouped expert size.
- Python syntax and offline SM90 compile passed for all short-prefill shapes.
- Source SHA256: 6fd9822278c7708731fb55856b0e1440351582e94409820fb00286f2b4365802
- Initial requests on UTC 2026-11-02 failed transport; job history confirmed
  no check was created. Accepted after the UTC reset as a58b6669fe854141.
- PASS five tests/eight timed shapes, no flags. Speedup 1.149098x
  (95% interval 1.131862-1.166335), score 337137.3 us.
  Modest ~0.85% gain over submitted attempt 4 is inside timing uncertainty.
  Retain the submitted default pending stronger evidence.
- This is check 1 of the new UTC day, not check 8 of the old day.

## Daily budget ledger

- UTC 2026-11-02: seven accepted checks and one approved scored submission.
  The attempt-8 transport failures consumed no quota.
- UTC 2026-11-03: attempt 8 is accepted check 1; zero scored submissions.
- Service now allows 20 checks / 10 submissions, but user limits remain
  TEN checks / FIVE submissions, with two user-budget checks reserved.
  Therefore at most eight accepted checks per UTC day may be run by this chat.

## Attempt 9 - exact paired NVFP4 expansion

- Independent candidate based on submitted attempt 4, not unverified attempt 8.
- Decode each packed byte into two exact E2M1 values with a PRMT lookup into
  FP16 encodings; convert these exact constants to FP32. No FP16 arithmetic
  is performed. Share the rounded FP32 SF*GS product, then multiply FP32 and
  use cvt.rn.bf16x2.f32 for one packed store. This preserves the submitted
  decoder's FP32 multiplication order and BF16 rounding.
- NVFP4 weights and FP8 scales remain unchanged in prepare. Expansion still
  runs entirely inside the timed call; no expanded weights persist in state.
- CPU emulation verified all 256 packed-byte decodings. NVIDIA PTX docs
  confirm the first input is packed high and second low for bf16x2 conversion.
- Offline SM90 compile passed. ptxas: 32 registers, zero spills versus 56
  registers for the submitted single-value decoder at equal output count/CTA.
- Source SHA256: 6cd90a1c7601a32134694da494de3e27a87fd5859f337dfc1f622bc9090147cc
- Reason for check: reduce repeated scale conversion, decode/register cost,
  and store instructions in the ~1.44 ms expansion stage.
- Check a537a0e611a844a7 accepted after one transport-reset retry.
- This is UTC 2026-11-03 check 2. User limits still 10/5, with two checks
  reserved (at most eight accepted checks per UTC day).
- Result: PASS five tests/eight timed shapes, no flags. Speedup 1.146172x
  (95% interval 1.128979-1.163365), score 337998.0 us. Saved as check_v9.json.
- Profile: _expand_pairs ~1.06 ms per 16K prefill vs ~1.45 ms for
  _expand_nvfp4 (-27%); other kernels unchanged.
- Per-shape paired ratio vs scored attempt 4 (77805d4b2f504d81):
  16K ctx0 full +1.64%, 16K ctx32768 sliding +2.22%, 16K ctx32768 full
  +3.17%, 16K ctx65536 full +2.18%. The four changed (long) shapes gain
  +2.30% by geometric mean (+2.37% vs mean of both attempt 4 runs).
  Unchanged short shapes -0.40% to +0.07%. Attempt 4's own run-to-run
  spread is up to 1.6% (16K ctx0) and below 0.8% elsewhere.
- Verified better under the muse_bf16_proj rule: the changed group gains,
  worst shape -0.40% (limit -3%), and overall speedup 1.1462 > 1.1394.
  Recommend promoting to submission.py; scored run needs user approval.
- User approved the scored run. TASK.md: "Your best run is your row on the
  board", so a worse run could not displace attempt 4's 1.1394x.
- submission.py = attempt9.py (raw SHA256 6cd90a1c..., same as recorded
  before the check). Scored job 9cf1606be94840fd: PASS five tests/eight
  timed shapes, no flags. Speedup 1.150580x (95% interval
  1.133321-1.167839), score 336703.2 us. Server content_hash 9acff9d0...
  equals check a537a0e611a844a7's, confirming identical code.
- Per shape vs scored attempt 4: short +0.02/-0.27/+0.03/-0.01%; long
  +1.27/+3.12/+4.41/+3.20%. Rank 684 -> 672. Scored runs used today: 1.

## Attempt 8 re-read under the verified-better rule

- Not verified better. vs scored attempt 4: 16K ctx0 full -3.86% (-3.08% vs
  attempt 4 mean). The long group with the grouped-GEMM hint is -0.09% overall,
  so the expected_m hint shows no gain.
- Short-prefill local alignment (q753 batch 1): +2.68% sliding, +1.34% full,
  vs attempt 4's run-to-run spread of 0.25-0.36% on those shapes. Batch 4: ~0.
  This is the part of attempt 8 worth keeping.

## Offline exploration - fused NVFP4 WGMMA GEMM (no remote check)

- Prototype saved as prototypes/fused_nvfp4_mm.py; never submitted or GPU-tested.
- It packs routes as attempt 4, dequantizes each weight tile inside BF16
  WGMMA GEMMs with FP32 accumulation, and retains all original expert work.
- Native WGMMA is generated, but initial 64x128 tiles spill; legal small
  variants still use 128-255 registers. Paired reshape hit a Triton shared-
  layout compiler error; a 2D gather compiles but does not fix high pressure.
- Resource reports saved in tiled_*_resources.txt. Do not promote this
  prototype without GPU correctness and CUDA-graph timing evidence.
- No remote quota spent; prioritize the separately checked pipeline and
  the pending exact paired-expansion experiment.

## Prepared combination (not checked)

- candidate_combined.py combines attempt 8's checked short scheduling and
  grouped-GEMM hint with attempt 9's exact paired expansion. Neither original
  candidate nor submitted attempt4.py is modified.
- Prepared while the paired-decoder GPU job waits. No additional check or
  scored submission requested for this combination.
- Source SHA256: 945b89abd1a9c9fd3fd3509468e370c18b925d8641800a252ac3142b5b4a7ffa
