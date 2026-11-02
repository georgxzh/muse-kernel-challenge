# muse_decode_layer experiments

Scope: only this folder; shared repository on master. User requires approval before
EVERY scored submit. Check budget: 10/day, always retain 2; scored budget: 5/day.
Resets: 00:00 UTC. No local PyTorch, Triton, or GPU; syntax validation is not a
correctness or performance result.

## Rules reviewed (2026-11-02)

- https://openkernelchallenge.dev/rules, official terms, TASK.md, AGENTS.md,
  reference.py, checks.py, baseline.py, and `okc --help` reviewed.
- AI assistance is allowed; the site supplies an agent prompt and agent guidance.
- Single readable Python source; use packages supplied by the SGLang image.
- kernel_fn must capture under CUDA graphs: no host synchronization or control
  flow/allocation based on tensor values. Shapes/config branches are permitted.
- No cached input-derived data, skipped keys/values, harness access, or timing
  manipulation. prepare may repack only weights, untimed, within 300 seconds.
- Preserve BF16 weights/activations and NVFP4 experts; cache stays FP8 E4M3.
  Queries may use E4M3; router FP32 may use BF16 hi+lo with FP32 accumulation.
- All 13 correctness cases and timed inputs are checked, including routing,
  FP8 cache writes (8% bound), output/residual relative L2 (query-error-dependent
  3-15% bound), worst-row bounds, and finite values.
- Timing: H200, alternating paired CUDA graph replays, cold L2, median ratios,
  95% interval at least +/-1.5%, fixed graph cost removed. Score is geometric
  mean across six loads, each combining sliding/full times with weights 46/16.
- Runtime limits from `okc problems --json`: 10 checks, 5 scored/day. These
  override older generic website limits. Account already signed in: georgxzh.

## Attempt 1: fuse QK norm / RoPE / FP8 conversion / cache writes

- Starts from the supplied SGLang baseline copied into submission.py, preserving
  FA3 attention, Marlin W4A16 experts, fused boundaries and overlapped routing.
- Adds one readable Triton kernel based on pinned SGLang FusedQKNorm semantics.
  Keeps BF16 norm and RoPE rounding; handles cache scale 1/64 and live slot zero.
  Removes separate K/V casts, cache store, and query cast launches.
- Source of QK rounding semantics:
  https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py
- Local validation: Python ast.parse passed; no local GPU execution possible.
- Remote check: 6de5454c568f45c1, server content_hash
  df800bdfd23d74ad1d7750d895a953f30e5da682622979681758933121085216.
- Result: PASS, 13/13 tests, all 12 timed shapes faster, no flags.
  Speedup 1.0137454x; 95% CI [0.9985392, 1.0289516]; score 22313.5908 us
  versus the normalized 22620.3 us SGLang baseline. Gain overlaps timing noise.
  Local source SHA256: 643e84964ec686c05b4ba4569bd1ec71e6e6b8a22ae5a88b90409c040456c893.
  Profile: profiles/submission.6de5454c568f45c1.txt.
  First decode-layer check today; 9 remain, 2 reserved.
- Profile: Marlin experts dominate verify (up to 75%); full attention dominates
  full decode (up to 60%). Small-batch dense projections also take material time.
- Recommendation: bank the passing candidate early, with user approval, while
  continuing checks for a more decisive improvement.
- At the initial check: no scored submission had been authorized or performed.

## Approved scored entry for attempt 1

- User explicitly approved this one submit in chat.
- Job: ead6ef40d45f4304. PASS, 13/13 tests, no flags.
- Scored speedup 1.0148404x, 95% CI [0.9996178, 1.0300631].
- Profile: profiles/submission.ead6ef40d45f4304.txt.
- Source: submission.py from a7ab50c; first scored submission today (4 remain).

## Attempt 2: BF16 split-K dense projections, up to 32 tokens

- File: candidate_splitk.py, based on attempt 1.
- Four projection calls use BF16 tensor-core dot products with FP32 accumulation
  and FP32 split reduction, followed by the usual BF16 output. Larger batches
  retain SGLang dispatch. Router, attention, and experts unchanged.
- Static tiles M=16/32, N=64, K=128; 8 K splits for K>=4096, otherwise 4.
- Motivation: profile shows small-batch dense kernels take material time.
- Local syntax validation passed. No local GPU measurement.
- Check 52a6ea3eb69443be: PASS, 13/13 tests, no flags.
- Speedup 1.0200123x; 95% CI [1.0047121, 1.0353125]; score 22176.4970 us.
- Source SHA256: 77cc09853a54f6faa8d7de6d4b07ece7248a8a77be569593e9acdf091d551308.
- Profile: profiles/candidate_splitk.52a6ea3eb69443be.txt.
- Shape results: T=1 1.029-1.034x, T=8 1.099-1.106x; T=16/32 regresses
  to 0.980-0.990x. Larger batches unchanged beyond QK fusion.
- Decision: retain as an experiment; future combinations restrict split-K to
  T<=8. Do not promote this all-small-batch version.
- Second check today; 8 remained at dispatch, 2 reserved.

## Attempt 3: smaller Marlin route tiles

- File: candidate_moe_tiles.py, based on attempt 1 independently of attempt 2.
- Use route block M=8 up to 16 tokens, M=16 above that, reducing the baseline's
  M=48/64 padding and exposing more parallel work for large batches.
- The first profile shows expert GEMMs take up to 75% of sliding verify time.
- Same route alignment, expert weights, activations, FP32 reduction, and norms.
- Local syntax validation passed.
- Check 4a095a5b613f4139: PASS, 13/13 tests, no flags, every timed shape faster.
- Speedup 1.0688928x; 95% CI [1.0528594, 1.0849262]; score 21162.3652 us.
- Source SHA256: 350cc8bab0c9c6bf415c1e00daca3bcf71067776a5a9a14f39aadcf4429aec53.
- Profile: profiles/candidate_moe_tiles.4a095a5b613f4139.txt.
- Large verify shapes improve to 1.131-1.216x; other shapes retain QK-fusion gains.
- Decision: promote the exact checked source to submission.py; recommend an
  immediate scored submission with user approval.
- Third check today; 7 remain, 2 reserved. Four scored submissions remain.

## Prepared candidate: fuse pre-expert norm and local-route mapping

- File: candidate_expert_prep.py, currently based on attempt 1 independently.
- Fuse the 4096-channel RMSNorm with mapping global expert IDs to local 0-31
  (or -1 for nonlocal), replacing separate subtraction, modulo, expert-map
  lookup and normalization launches. Retain all routes and their weights.
- Effective gain computed in BF16 in prepare, as SGLang does. RMS arithmetic
  FP32; expert input BF16. Local syntax passed; not checked yet.
- Source reviewed for norm precision:
  https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/muse_spark_layers.py

## Attempt 4: combine measured winners and fuse expert preparation

- File: candidate_combined.py; promoted exact checked source to submission.py.
- QK/cache fusion + attempt 3 expert tiles + split-K projections restricted to
  T<=8, where attempt 2 won. T=16/32 use SGLang dispatch to avoid regressions.
- Adds the prepared pre-expert norm/local-route mapping fusion described above.
- All weight/activation precision, FP32 sums, route weights and keys preserved.
- Local ast.parse passed. Check 4a37347009d24582: PASS, 13/13 tests, no flags.
- Speedup 1.1134628x; 95% CI [1.0967609, 1.1301648]; score 20315.2719 us.
- Source SHA256: d60e7ef0aaf3274f249489dce37611e096adb29908827e2e87f4329fe8e07935.
- Profile: profiles/candidate_combined.4a37347009d24582.txt.
- All 12 timed shapes improve: 1.026-1.236x.
- Decision: promote and recommend immediate submit, pending user approval.
- Fourth check today; 6 remain, including 2 reserved. Three scored runs remain.

## Approved scored entry for attempt 3

- User approved attempt 3, then explicitly approved one retry after HTTP/2
  protocol errors. The failed POST did not appear in jobs or consume quota.
- Accepted job acf0293fff4c4d60: PASS, 13/13 tests, no flags.
- Scored speedup 1.0590949x; 95% CI [1.0432085, 1.0749813].
- Profile: profiles/submission.acf0293fff4c4d60.txt.
- HTTP/2 also failed the first attempt-4 check POST without consuming quota.
  Subsequent CLI calls use per-process GODEBUG=http2client=0 (HTTP/1.1),
  with no global configuration change.
- Completed attempts 1-4 have each been committed with results and pushed.
