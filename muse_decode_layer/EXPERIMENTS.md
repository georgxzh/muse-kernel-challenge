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
- No scored submission authorized or performed.
