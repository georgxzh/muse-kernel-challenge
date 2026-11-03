# Attention decode experiments

## Current status (evening Nov 2 Toronto; UTC quota cycle 2026-11-03)

- Best scored: **1.38346370x SGLang**, job `a4b548ad9d744089`, PASS 16/16.
- Current kernel: `submission.py`, byte-identical to `candidates/attempt07_hybrid.py`.
- 95% speedup CI [1.36271175, 1.40421566]; equivalent score 2722.370 us.
- Started with a submitted 1.11109588x fused-FA3 entry, improved to a submitted
  1.28034635x custom-sliding entry, then to the current native-FP8/FA3 hybrid.
- New UTC cycle: 5/10 checks used under user cap (5 left, including 2 reserved),
  0/5 scored submissions used. Four checks pending (attempts 9-12).
  Prior UTC cycle used 6 checks and 3 scored runs. Ask before every new submit.
- Current hybrid retains FA3 for batched full verify and uses custom native FP8 QK,
  FP32 online softmax/accumulation and BF16 PV elsewhere. All keys in each context
  or window are included; no value skipping, tensor-derived caching, or host sync.
- Passing check's largest test relative-error/bound ratio was 0.6468; static
  rule checks reported no flags or rejections. Scored rerun also passed all tests.

Scope: only `muse_attention_decode`, on shared `master`. Never stage other problems.
Rules reviewed 2026-11-02: https://openkernelchallenge.dev/rules and local TASK.md/AGENTS.md.
AI assistance is permitted. Every required key participates; no cached input-derived results,
host synchronization, input-value host branches, or harness changes. Preserve BF16 activation
boundaries and FP8 E4M3 KV storage; FP8 E4M3 queries are permitted.

Budget policy: at most 8 of the user's 10 daily attention checks, keeping 2 in reserve.
Ask the user before EACH scored submit (5/day). Reset is 00:00 UTC.

## Attempt 1 — fused preparation + FA3

- File: `submission.py`.
- Change: fuse per-head Q/K RMSNorm, interleaved RoPE, query BF16-to-FP8 conversion,
  and scaled FP8 KV writes in one Triton launch. Retain SGLang FA3 attention and use
  a fused Triton per-head output RMSNorm/sigmoid gate.
- Covers arbitrary input lengths via FA3 metadata, page sizes, causal verify blocks,
  full/sliding attention, and device cache scales. RoPE uses libdevice trig and FP32 math.
- Static validation: Python compilation and `git diff --check` passed.
- Remote check: `906081c5cdb740b1`, OKC content hash
  `392f9d853f43b24a20dbc436b5da85fde8410ab2aef8066047851bd90956dc68`.
- Result: PASS, 16/16 tests. Speedup 1.11613745x, 95% CI [1.09939539, 1.13287951]. Score 3374.405 us versus SGLang 3766.3 us. 1/10 daily checks used, 9 remain.
- Scored submissions: none; no submit authorized yet.

- Exact passing snapshot: candidates/attempt01_fused_fa3.py (same source hash).
- Profile: profiles/submission.906081c5cdb740b1.txt and .json.
- Decision: recommend immediate scored submission, subject to user approval.
- Profile finding: full FA3 takes ~189/375 us at batch 16/32; largest sliding verify preparation takes 6 us and gate 3.6 us. Next changes should target these measured costs.

## Attempt 1 scored submission

- User explicitly approved one submit of the preserved attempt 1.
- Submitted as `5284aaeb6d134d73`; PASS 16/16, 1.11109588x, 95% CI [1.09442944, 1.12776232], score 3389.716 us.
- Local source SHA256: `849b3d102f0fb1e53c6dcf8189edbca7b52d952e1209637ca1435173aa515915`.
  OKC content hashes include problem metadata and differ from raw file SHA256.
- Attempt commit `27163e0` confirmed on remote master after transient push failures.

## Attempt 2 — split-context full attention

- File: `candidates/attempt02_split_full.py`; attempt 1 remains the current submission.py.
- Full layers: Triton paged gathers, BF16 tensor-core QK/PV with FP32 accumulation,
  stable online softmax within 2048-token chunks, exact log-sum-exp merge across
  ALL chunks, fused output BF16 RMSNorm/gate. Sliding layers retain FA3.
- Input FP8 Q/K/V values convert exactly to BF16, and cache scales are applied.
- Offline H200 SM90 compile: decode, 8-token verify, page 64, and 513-split combine
  all pass using problem-local Triton 3.5.1. No local GPU; this is compile-only.
  Fixed an FP8 masked-load fallback type error before spending the remote check.
- Remote check: `a8db51632a9a4133`, PASS 16/16 but 0.86569415x, 95% CI [0.85270874, 0.87867956], score 4350.613 us. 2/10 checks used; 8 remain, including
  2 reserved. No additional scored submit authorized.

- Attempt 2 decision: reject as a universal replacement. Full decode at B=16/32
  improves to 181.91/331.25 us versus FA3 194-195/378-380 us. Full B=1 regresses
  to 64.65/74.65 us (decode/verify), and B=16/32 verify to 331.69/616.32 us.
  Keep FA3 for those shapes. Next candidate uses the custom path on sliding
  layers and on full decode only when batch >= 8; dispatch uses public shapes.
- Initial scored entry is live and correct. 1 of 5 scored runs used today;
  each additional scored submit still requires fresh user approval.

## Attempt 3 — custom sliding plus measured full-decode dispatch

- File: `candidates/attempt03_split_all.py`.
- Reuse the validated stable chunk/combine algorithm for the complete sliding
  window. Include the block's full union of windows, then apply each query's
  exact window and causal masks. Fused combine/gate removes FA3 scheduling and
  separate output launches on this path.
- Use custom full attention only for decode with batch >= 8, where attempt 2's
  profile showed a gain. Full verify and small-batch full decode retain FA3.
- Offline SM90 compilation passes, including sliding verify and long combine.
- Remote check `8527a083c2844c23`: PASS 16/16; 1.27569792x, 95% CI [1.25656246, 1.30073351], score 2952.345 us. Promoted unchanged to submission.py.

## Draft 4 — grouped preparation (not remotely checked yet)

- File: `candidates/attempt04_grouped_prepare.py`.
- Same dispatch as attempt 3, with four Q/K head rows per preparation block to
  reduce the measured 6 us cost on large verify shapes. Offline compilation passes.
- No remote check spent. Do not promote until measured.

- Attempt 3 decision: recommend scored submit now; await fresh user approval.
  Sliding decode B=1/16/32: 10.69/16.00/19.20 us. Sliding verify:
  11.59/22.09/31.76 us. Full batched decode 181.54/327.47 us.

## Attempt 5 — wider tiles and small-batch parallelism

- File: `candidates/attempt05_wider_tiles.py`; remote check `911497e0df3d42bf`.
- Use N=128 instead of 64, full-context chunks of 256 for smaller batches and
  shorter capacities, grouped preparation from draft 4. Retest custom full verify.
- Offline SM90 compilation passes. Resource analysis finds high register use
  (up to 249 registers), so remote timing is essential; no promotion assumed.
- Result PASS 16/16, 1.13039337x, 95% CI [1.11343747, 1.14734927], score 3331.849 us. Do not promote universally: full verify still regresses. 4/10 checks used; 6 remain including 2 reserved.

## Attempt 3 scored submission and Git integration

- User explicitly approved one submit; job `526d2f2d75ef4631` PASS 16/16 at 1.28034635x SGLang.
- Attempt commit `697eb8a`. Remote master diverged at `203f89f` (Create README.md).
  The user explicitly authorized preserving and merging that remote README outside
  the attention folder. The merge completed; no other project's source was edited.
- Scored attempts used today: 2/5. No other submit authorized.

## Attempt 5 decision and next direction

- Keep FA3 for full verify (custom ~326/629 us versus FA3 ~200/386 us at B=16/32).
- Small full decode improved to 21.73 us (FA3 ~27.7 us); this is useful in a hybrid.
- Wider sliding tiles are not a universal improvement: largest sliding verify is
  34.31 us versus attempt 3's 31.76 us. Grouped preparation reduces large-verify
  overhead, but not enough to compensate.
- Offline resource reports show 218-249 registers/thread for the attention body,
  limiting occupancy. Next investigate launch warps/stages before another check.

## Attempt 6 — native FP8 QK tensor cores

- File: `candidates/attempt06_native_fp8_qk.py`; check `10f349ae95f842fd`: PASS 16/16, 1.29488362x, 95% CI [1.27546037, 1.31430688], score 2908.601 us.
- Keep allowed FP8 query and key values directly in the QK dot product instead of
  converting them exactly to BF16 first. FP32 scores/softmax/accumulation and BF16
  probabilities/values for the weighted sum remain as before. No attention sparsity.
- Offline SM90 compilation passes. Several warps/stages/smaller-tile drafts were
  compiled but not remotely checked: resource reductions alone did not establish
  enough expected benefit. All draft compiler files/caches stay in .tools/.
- 5/10 checks used, 5 remain including 2 reserved. 2/5 scored runs used.

- Attempt 6 decision: useful native FP8 gains, but do not submit universally.
  Full decode B=1/16/32 improves to 20.80/162.46/300.10 us, and sliding cases
  improve too. Full verify B=16/32 remains slower at 250.39/488.50 us than FA3
  ~200/386 us. Next combine native FP8 with FA3 for batched full verify.
- Attempt 3 scored run `526d2f2d75ef4631`: PASS 16/16, 1.28034635x,
  95% CI [1.26114116, 1.30460614], score 2941.626 us. This remains the live best.

## Attempt 7 — native FP8 / FA3 hybrid

- File: `candidates/attempt07_hybrid.py`; check `f966398b2a454b3e`: PASS 16/16, speedup 1.37586848x. Promoted unchanged to submission.py.
- Identical native-FP8 attention and grouped preparation to passing attempt 6.
  For full verify with batch * num_kv_heads >= 16, use passing FA3 path instead.
  All dispatch depends solely on public shapes/config, never tensor values.
- Python syntax and diff-whitespace checks pass; unchanged Triton bodies already
  compile and passed attempt 6. This check validates the combined dispatch end to end.
- 6/10 attention checks used, 4 remain, of which 2 are always reserved.
- Best scored remains attempt 3: job `526d2f2d75ef4631`, 1.28034635x.
- No more scored submits are authorized yet (2/5 used).

- Attempt 7 95% CI: [1.35523045187781, 1.4024870769109], score 2737.39827411617 us.
- Decision: recommend immediate scored submission; fresh user approval pending.

## Attempt 7 scored submission

- User explicitly approved one submit of exact commit `5e9e7c7` candidate.
- Job `a4b548ad9d744089`: PASS 16/16, 1.38346370x, 95% CI [1.36271175, 1.40421566], score 2722.370 us. Source commit confirmed pushed to remote master.
- Current `submission.py` is byte-identical to `candidates/attempt07_hybrid.py`.
- Today's usage: 6/10 checks, 3/5 scored submissions. 4 checks remain, with 2
  reserved; 2 scored submissions remain and need fresh approval each time.

## New UTC quota cycle — 2026-11-03 (evening Nov 2 Toronto)

- Server reset observed at 01:03 UTC: 0 checks/submissions used. The server now
  advertises 20 checks and 10 scored runs/day; keep the user's stricter limits of
  10 checks / 5 scored, using at most 8 checks and always reserving 2.
- Existing best remains scored attempt 7, 1.38346370x; no change to submission.py.

## Attempt 8 — FA3-style scaled FP8 probability/value multiply

- File: `candidates/attempt08_fp8_pv.py`; check `0ca845bb169f461d`: PASS 16/16, 1.29608350x, CI [1.27664225, 1.31552475], score 2905.908 us.
- Replace BF16 PV operands with FP8 P and the existing unchanged FP8 V values.
  Compute softmax max/sum and output accumulation in FP32. Quantize P only after
  multiplication by 256, and divide the final weighted output by that same factor.
- Precision basis: FA3's FP8 mainloop defines Element as E4M3, creates its P
  operand with make_tensor_like<Element>, converts softmax scores into it, and
  uses Element for both PV operands. Its softmax.h explicitly applies an FP8
  max offset of 8 (x256) to reduce underflow. This matches baseline-internal
  arithmetic; BF16 input/output boundaries and FP8 cache storage are preserved.
- Source evidence reviewed:
  https://github.com/Dao-AILab/flash-attention/blob/main/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp
  (Is_FP8 / Element; TiledMmaPV; tOrP conversion at lines 1073-1075 / 1197-1199)
  https://github.com/Dao-AILab/flash-attention/blob/main/hopper/softmax.h
  (scale_apply_exp2, Max_offset=8 discussion).
- Every key/value is loaded and processed. No pruning, attention-weight threshold,
  sparse mask, output caching, or host input reads. Keep full causal/window masks.
- Offline H200 compilation passed decode, verify, page 64, and long combine.
- New cycle: 1/10 checks used; 9 left under user cap, including 2 reserved.
  No new-cycle scored submissions made or approved.

- Attempt 8 decision: reject universal promotion. Full decode B=1/16/32 improves
  to 19.97/155.78/284.54 us, but full verify B=16/32 takes 267.00/498.56 us
  versus current FA3 ~199/384 us. Existing 1.383x entry remains current.
- Validation reports no flags/rejections. Largest whole-test relative error was
  0.09713 against 0.15; coherent tests were 0.04112/0.03810 against 0.108/0.111.
- Added source evidence: flash_fwd_kernel_sm90.h line 385 explicitly selects
  Max_offset=8 for Is_FP8, matching the x256 scaling in attempt 8.
- Offline compiler correction: model real CUDA tensor pointer alignment with
  tt.divisibility=16. Previous reports omitted this and were conservative;
  correctness compilation remains valid, but old register/occupancy estimates
  should not guide new tuning. Aligned FP8 decode/verify reports use 120/168
  registers and 22/28 KB shared memory for full attention, with no spills.
- Prepared two independent drafts (not yet measured): attempt09_l2_loads.py uses
  .cg on KV loads; attempt10_dynamic_splits.py fixes total CTA count using H200
  occupancy and computes chunk sizes from live GPU sequence lengths. Both compile.

## Attempts 9-12 — independent follow-ups (pending)

- Attempt 9: `candidates/attempt09_l2_loads.py`, job `c9fb0a3f8f0847d5`.
  Change only K/V load cache policy to .cg, bypassing L1 for cache rows loaded once
  by each CTA. Correctly aligned SM90 compilation passes with no attention spills.
- Attempt 10: `candidates/attempt10_dynamic_splits.py`, job `53ead2c9320442a8`.
  Fix total CTAs to approximately H200 occupancy (4 decode / 3 verify CTAs per
  each of 132 SMs), choose per-request chunks from live GPU seqlens, and retain
  static allocation/grid dimensions. Every required key remains assigned once
  to a split. No GPU-to-host length reads. Offline compilation passes.
- Attempt 11: `candidates/attempt11_query_tiles_fp8.py`, job `ac4f2870058c4d84`.
  Split verify's 64 query rows per KV head into 2 independent 32-row groups;
  output scratch indices remain global query-row indices, with disjoint writes.
  Uses passing attempt 8 FP8 PV. Each group still traverses the complete context.
- Attempt 12: `candidates/attempt12_query_tiles_bf16.py`, job `1fc675802a884460`.
  Same 32-row query tiling with BF16 PV (more precise, as in current hybrid).
- Both query-tiling variants compile offline without attention spills. Attention
  registers estimated at 168 (FP8 PV) / 140 (BF16 PV) for full verify.
- New UTC cycle usage: 5/10 checks; 5 left with 2 reserved. No scored run approved
  or made in this cycle. Preserve submission.py at the scored 1.383x candidate.

- Prepared candidates 9-12 are committed as compile-only PASS / GPU queued, to
  preserve the work during the service queue. Each GPU result will get a separate
  result commit. No draft is promoted or submitted based on compilation alone.
- Aligned full-verify BF16-PV resource comparison: current 64-row body uses
  202 registers/thread; 32-row draft uses 140, potentially raising residency from
  2 to 3 CTAs/SM. This is an estimate, not a measured speedup.
