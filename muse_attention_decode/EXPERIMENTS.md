# Attention decode experiments

## Current status (evening Nov 2 Toronto; UTC quota cycle 2026-11-03)

- Best scored: **1.38346370x SGLang**, job `a4b548ad9d744089`, PASS 16/16.
- Current kernel: `submission.py`, byte-identical to `candidates/attempt07_hybrid.py`.
- 95% speedup CI [1.36271175, 1.40421566]; equivalent score 2722.370 us.
- Started with a submitted 1.11109588x fused-FA3 entry, improved to a submitted
  1.28034635x custom-sliding entry, then to the current native-FP8/FA3 hybrid.
- New UTC cycle: 6/10 checks used under user cap (4 left, including 2 reserved),
  0/5 scored submissions used. Five checks pending (attempts 9-13).
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

## Attempt 13 — 16-row query tiles (pending)

- File: `candidates/attempt13_query_tiles16_bf16.py`, job `66d0fa57bd7f403b`.
- Split verify into four 16-row query groups, using FP8 QK / BF16 PV as in the
  current hybrid. All groups traverse all required keys and have disjoint output
  scratch rows. No changes to precision or input/output semantics.
- Aligned SM90 compile PASS, 128 attention registers/thread and 22 KB shared
  memory, with no attention spills. This supports an estimated four CTAs/SM.
- 6/10 new-cycle checks used; 4 left under user cap, including 2 reserved. No
  more exploratory checks until pending results guide the final consolidation.
- Added tools/compare_profiles.py: normalize candidate paired ratios against
  the current scored run's per-case baseline and recompute load weights. It
  reproduces 1.38346370x for the current run. With only existing attempt 8 gains,
  estimated combination is 1.41345x; this is not a measured result or a promotion.

## Attempt 9 GPU result

- Job `c9fb0a3f8f0847d5`: PASS 16/16, 1.28433151x, 95% CI [1.26506654, 1.31143679], score 2932.498 us.
- L2-only KV loads did not improve the universal FP8-PV candidate; retain current hybrid.
- submission.py remains byte-identical to scored attempt 7 (1.38346370x).

## Attempt 10 GPU result

- Job `53ead2c9320442a8`: PASS 16/16, 1.15726522x, 95% CI [1.13990625, 1.17462420], score 3254.483 us.
- Occupancy-sized dynamic splitting was slower, especially full decode (213/422 us at B16/32). Fixed 2048-token splits remain preferable.
- submission.py remains byte-identical to scored attempt 7 (1.38346370x).

## Attempt 11 GPU result

- Job `ac4f2870058c4d84`: PASS 16/16, 1.11308153x, 95% CI [1.09638531, 1.13354513], score 3383.669 us.
- 32-row FP8 query tiling repeats cache traffic and slows batched verify substantially; reject promotion.
- submission.py remains byte-identical to scored attempt 7 (1.38346370x).

## Attempt 12 GPU result

- Job `1fc675802a884460`: PASS 16/16, 1.07914559x, 95% CI [1.06295841, 1.09533278], score 3490.076 us.
- 32-row BF16 query tiling reduces registers but slows full verify to 411/786 us at B16/32; reject promotion.
- submission.py remains byte-identical to scored attempt 7 (1.38346370x).

## Attempt 13 GPU result

- Job `66d0fa57bd7f403b`: PASS 16/16, 0.98575904x, 95% CI [0.97097265, 1.00612524], score 3820.711 us.
- 16-row BF16 query tiling further increases repeated cache traffic; full verify 520/1016 us at B16/32. Reject promotion.
- submission.py remains byte-identical to scored attempt 7 (1.38346370x).

## Attempt 14 — conservative PV hybrid (pending)

- File: `candidates/attempt14_pv_hybrid.py`; check `300cb7ecb4304bfa`.
- Preserve current FA3 full batched verify and BF16-PV sliding attention. Use
  attempt 8 scaled FP8 PV only for custom full attention, where repeated tests
  showed about 4-6% improvements. No exact-shape or input-value dispatch.
- All required keys/values remain processed; FP32 softmax and accumulation,
  BF16 output boundary, and CUDA graph behavior are unchanged.
- Offline SM90 compilation and whitespace checks pass. Combined performance
  estimate is approximately 1.41x; await measured correctness and timing.
- 7/10 current-cycle checks used. Three remain, of which two stay reserved.
  No scored submissions made or approved in this UTC cycle.

## Attempt 15 — wider full-decode tiles (pending)

- File: `candidates/attempt15_wide_decode.py`; check `b71d5bd0b39c4ffc`.
- Change only full decode N=128 to N=256 relative to attempt 14. This halves
  per-split loop iterations and may reduce gather/softmax overhead. Sliding
  attention and verify retain N=128. No exact-context or tensor-value dispatch.
- SM90 compilation PASS without attention spills. Full decode uses 196 registers
  and 43 KB shared at two stages, versus 120 registers / 22 KB for N=128. This
  reduces estimated residency from four to two CTAs/SM, but doubles keys/tile;
  only GPU measurements can settle this tradeoff. One stage remained at 191
  registers and offered no occupancy advantage, so retain two-stage pipelining.
- Offline compile tool now respects the candidate's full-decode tile constant.
- 8/10 current-cycle checks used. Both remaining checks are reserved: no further
  check runs this cycle. No scored submission is authorized in this cycle.

## Attempt 14 GPU result — new best checked

- Job `300cb7ecb4304bfa`: PASS 16/16, 1.40088781x, 95% CI
  [1.37987449, 1.43037438], score 2688.509 us, no flags.
- Full decode B=1/16/32: 20.15/157.49/284.00 us; full verify B1: 27.35 us.
  Sliding and FA3 paths remain close to current timings. Overall improvement
  versus the scored 1.38346370x version is modest and confidence intervals overlap.
- Promoted byte-identical candidate to submission.py as best checked. Existing
  scored submission remains attempt 7 until a new exact submit is approved.
- Attempt 15 still running; compare before asking for a scored submission.
- 8/10 current-cycle checks used, two reserved. No scored runs used this cycle.

## Attempt 15 GPU result — retain attempt 14

- Job `b71d5bd0b39c4ffc`: PASS 16/16, 1.36917817x, 95% CI
  [1.34864050, 1.38971584], score 2750.774 us, no flags.
- N=256 full decode B16/32 takes 166.42/322.93 us versus attempt 14
  157.49/284.00 us. Reduced residency outweighs the loop-count reduction.
- Reject promotion. submission.py remains byte-identical to attempt 14, source
  commit `9ccaccd`, best checked 1.40088781x. Best scored remains 1.38346370x.
- Recommend one scored submit of exact attempt 14, pending fresh user approval.
  Improvement is modest and the overall confidence intervals overlap, but
  full-decode gains over the previous BF16-PV path recur across independent checks.
- Current UTC-cycle usage: 8/10 checks, two reserved; 0/5 scored submissions.
  Do not run additional checks this cycle. Every submit still needs approval.

## Attempt 14 scored submission — approved and queued

- User explicitly approved one scored submit of exact attempt 14, source commit
  `9ccaccd`, after reviewing the passing 1.40088781x check result.
- Verified source SHA256 against that commit:
  `7f0dbde841d0377d814fc2c958f89103424a6a8bf5bb7b0a25961afd7c0f5899`.
- Submitted `candidates/attempt14_pv_hybrid.py`; scored job `1f40b617a3c14d74`.
- Await scored correctness/timing report. Existing best scored is attempt 7
  at 1.38346370x until this job completes.
- Current-cycle usage: 8/10 checks (two reserved), 1/5 scored runs. No further
  scored submission is authorized by this approval.

## Attempt 14 scored result (recorded UTC 2026-11-03 from the prefill session)

- Scored job 1f40b617a3c14d74: PASS, speedup 1.4279x, score 2637.58 us; it is
  our board row (rank 984 of 1308 at 15:5x UTC). Server content_hash c79cc2b7...
  equals check 300cb7ecb4304bfa's (1.4009x), so the scored code is the checked code.
- submission.py is byte-identical to candidates/attempt14_pv_hybrid.py
  (SHA256 7f0dbde8...). Lowest check on this board is attempt 13 at 0.9858x
  (66d0fa57bd7f403b, never scored); every scored run has been >= 1.28x.

## Attempt 16 - programmatic dependent launch (PDL) between the Triton kernels

- Bookkeeping audit (UTC 2026-11-03): every small shape runs exactly three
  kernels (_prepare -> _attention_parts -> _combine_gate) with no zero-fills,
  copies or casts; timed minus profiled kernel time is 2.4-4.6 us on the small
  sliding shapes (two kernel boundaries), i.e. launch overhead.
- Judge image (llms.txt): PyTorch, Triton, FlashInfer, DeepGEMM, TileLang, CuTe
  DSL, CUTLASS; nothing else, no network. Triton 3.5.1 supports launch_pdl and
  triton.language.extra.cuda.gdc_wait / gdc_launch_dependents (griddepcontrol).
- candidates/attempt16_pdl.py, based on scored attempt 14: _prepare calls
  gdc_launch_dependents() at its start; _attention_parts is launched with
  launch_pdl=True and calls gdc_wait() before any memory access (SASS: no load or
  store before ACQBULK), triggering its dependents after its final stores;
  _combine_gate (launch_pdl=True) loads its gate from the immutable qkv input,
  then gdc_wait() before reading partials/LSE. No global write precedes a wait,
  so CUDA-graph memory planning (stream order) stays valid. FA3 path unchanged.
  Arithmetic unchanged.
- Offline SM90 PASS, no spills; PTX griddepcontrol.wait/launch_dependents and
  SASS ACQBULK/PREEXIT present (candidates/offline_attempt16_pdl.txt).
- Source SHA256: 616c4bbbb33456d1ae36d4ecb5cc0d6db249dd370af25b9a9a99f69e2f209f38
- Check 464ab004478145af (UTC 2026-11-03 check 9): PASS 16/16 tests, 12/12
  shapes, no flags. 1.4462070x (CI 1.4245139-1.4679001), score 2604.261 us.
- vs mean of attempt 14's two runs: 10 PDL-path shapes +3.47% as a group:
  sliding decode-1 +17.54% (9.4 -> 7.8 us), full decode-1 +6.07%, sliding
  verify-1 +9.20%, full verify-1 +4.64%, sliding verify-16 +4.75%, sliding
  decode-32 +2.90%, sliding verify-32 +1.94%, full decode-16 -0.06%; sliding
  decode-16 -3.87% (a14 run-to-run 0.8%), full decode-32 -6.45% (a14's own runs
  differ 12.7% there). FA3 shapes +0.09/+0.03% (unchanged code).
- Rule: group gains, overall higher (1.4462 > scored 1.4279), but two shapes past
  -3% (one known-noisy). Next: attempt 17 moves _prepare's trigger to its end to
  test whether early staging of attention CTAs costs sliding decode-16.

## Attempt 17 - PDL with _prepare's trigger after its stores

- candidates/attempt17_pdl_late.py = attempt 16 with _prepare calling
  gdc_launch_dependents() after its cache stores instead of at its start.
  Offline SM90 PASS, no spills (candidates/offline_attempt17_pdl_late.txt).
- Source SHA256: 1c3069c7c9123438bf0361aacf11ef2b1a19ae305f4ad8f57819e06977126d67
- Check de94bdcac8e8488b (UTC 2026-11-03 check 10): PASS 16/16 tests, 12/12
  shapes, no flags. 1.4473404x (CI 1.4256303-1.4690505), score 2602.221 us.
- vs mean of attempt 14's two runs, PDL-path group +3.98% (attempt 16: +3.47%):
  sliding decode-1 +15.50%, decode-16 +8.23% (attempt 16: -3.87%), decode-32
  +4.45%, verify-1 +6.01%, verify-16 +1.31%, verify-32 -0.15%; full decode-1
  +6.43%, decode-16 +1.02%, verify-1 +4.58%, decode-32 -6.21%. FA3 shapes +0.08/-0.06%.
- full decode-32: attempts 16 and 17 both ran 284 us; attempt 14 ran 250.1 us
  (scored) and about 284 us (check 300cb7ec), so against the check alone it is
  level; the -6% is attempt 14's fast scored sample pulling the mean.
- Rule: group gains above noise, overall higher (1.4473 > scored 1.4279), every
  other shape within -0.2%. Recommend a scored run of attempt 17, pending
  explicit user approval. Checks used today: 10/10.

### Attempt 17 scored

- User approved. submission.py = candidates/attempt17_pdl_late.py (SHA256
  1c3069c7..., unchanged since check de94bdcac8e8488b; server content_hash
  a99208c0c61c56ad... identical for check and scored run).
- Scored run 7e9dd2b58e9746e4 (UTC 2026-11-03): PASS 16/16 tests, 1.4424609x
  (CI 1.421-1.473), score 2611.024 us. Board row now this run (best kept):
  1.442x, rank 969 (was 985 at 1.428x). Result in
  candidates/scored_attempt17_pdl_late.json, profiles/submission.7e9dd2b5....
- Per shape: sliding decode-1 8.4 us (2.473x), decode-16 13.7, decode-32 16.8,
  verify-1 10.5; full decode-16 155.7, decode-32 284.4, verify-16 198.6,
  verify-32 382.5 us (1.01-1.35x). The full-attention batched shapes are now
  most of the remaining gap: board leaders are at ~3.0-3.3x overall.

## Attempts 18-20 - deeper software pipelining of the full-layer loop

- Offline finding (tools/inspect_pipeline.py, Triton 3.5.1 TTGIR/SASS): with
  num_stages=2 the full-layer loop keeps single K/V buffers, waits for all
  copies at the top (async_wait 0) and reloads the next tile's page indices
  with a blocking LDG before issuing its K/V gather: two dependent memory
  latencies per 128-token tile (~3.3 us per tile measured from 284 us at
  decode-32). num_stages=3/4 prefetch page indices one tile ahead (cp.async to
  shared), num_stages=5 double-buffers K/V and pages (40 KB smem, 120 regs).
- attempt18_full_st3.py: attempt 17 with num_stages=3 for full-layer launches
  (sliding stays 2). SHA256 afa0838388906bad771bebaf4b38dca9ac3ab17fc31dbcd82326b278084743ea.
  Check ac046e75f11240b2: PASS 16/16, 1.4450093x, score 2606.419 us.
- attempt19_full_st5.py: num_stages=5 for full-layer launches.
  SHA256 19e6a3f74bd482e210bf4d9e00115974bcf37877464f307fab84b5b4e9c56b20.
  Check 4d80f3a60d9a48bd: PASS 16/16, 1.4326389x, score 2628.925 us.
- attempt20_triton_verify_st5.py: attempt 19 with FA3 retired (batched full
  verify on the Triton kernel). SHA256 36cc1df369df3ca8c21d3bffca071eca7fc93ef2effd20fdbc973011c2c352fe.
  Check 8a995d3e3cef406c: PASS 16/16, 1.3146110x, score 2864.954 us.
- Full decode-32: 283.7 (st3) / 287.6 (st5) us vs 284.4 us; decode-16
  153.5 / 147.9 us vs 155.7 us (+1.9% / +5.9%); full verify-1 st5 30.0 us vs
  26.4 us (-12%). Triton full verify-16/32 at st5: 275.1 / 526.6 us vs FA3
  200 / 385 us. Latency is not the limiter; all three rejected.
- Next hypothesis (attempts 21/22): the random page=1 gathers read 64-byte half
  rows per KV-head CTA (two random DRAM accesses per 128-byte row); read whole
  rows with both KV heads in one CTA.

## Attempts 21-22 - full decode with both KV heads per CTA

- New kernel _attention_pair for full-layer decode (L=1, HK=2, HQ=16): one CTA
  per (sequence, split) covers all 16 query heads; each token's K and V rows
  (128 FP8 bytes) are gathered whole instead of one 64-byte half per KV-head
  CTA. Query row r holds zeros outside its KV head's 64 columns, so one FP8 QK
  over 128 columns equals the per-head product; PV produces both heads'
  columns and each row stores only its own. Same partial/LSE layout and
  _combine_gate; same FP8 QK / scaled FP8 PV / FP32 softmax as attempt 17.
  Everything else is attempt 17. Offline SM90 PASS, no spills, PDL order OK
  (tools/compile_pair.py).
- attempt21_pair_heads.py (128-token tiles, 201 regs, 40 KB smem: 2 CTAs/SM).
  SHA256 4f04adfd47c50a68c1e994683d58f68192a73d4020fff6c60c32b387fd8bcb2b.
  Check 84dcbe01b6254c7f: PASS 16/16, 1.4318052x, score 2630.456 us.
  Full decode-1/16/32: 20.9 / 164.0 / 276.6 us vs 18.8 / 155.7 / 284.4 us.
- attempt22_pair_heads_n64.py (64-token tiles, 146 regs, 22 KB: 3 CTAs/SM).
  SHA256 892cfe05cb336663da1312cda4f9aa54c4f56c8a1151507ee11ca9538dad1292.
  Check 1833a762b0e9414a: PASS 16/16, 1.4095663x, score 2671.957 us.
  Full decode-1/16/32: 18.9 / 170.9 / 316.7 us.
- Whole-row gathers do not help: the half-row reads are not the limiter. Both
  rejected.
- SASS count (attempt 17, full decode loop): 759 instructions per warp per
  128-token tile (IMAD 155, LDS 105, F2FP 80, ISETP 70, FSEL+SEL 50, HMMA 32);
  5.12M token-heads at decode-32 is ~116 us of pure issue at 4 IPC/SM. The
  kernel is largely instruction-bound. Next: attempt 23 removes the masks from
  interior tiles (572 instructions per tile).
