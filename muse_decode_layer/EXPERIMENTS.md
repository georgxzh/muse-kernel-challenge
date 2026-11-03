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

- Initially prepared independently as candidate_expert_prep.py; its code was
  folded into attempt 4 and the unused standalone prototype was removed.
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

## Attempt 5: route tiles M=8 throughout

- File: candidate_tiles8.py, based on checked attempt 4.
- Only change: Marlin alignment/route block M=8 at every token count, rather
  than M=16 above 16 tokens. Same FP32 reductions, precision and every route.
- Motivation: attempt 3 measured substantial large-batch gains from smaller
  tiles; M=8 may further reduce route padding and expose parallel blocks.
- Local ast.parse passed. Check 4b019aca11194469: PASS, 13/13 tests, no flags.
- Speedup 1.0304546x; 95% CI [1.0149978, 1.0459115]; score 21951.7670 us.
- Source SHA256: bddbfa0c22105cfe904387caafce1e69b05e1d9194b7395ec4772b6ea3eaddd2.
- Profile: profiles/candidate_tiles8.4b019aca11194469.txt.
- Result: REGRESSION versus attempt 4. T=32 reaches 0.972-0.993x and
  T=256 reaches 0.913-0.919x versus SGLang; T=128 loses its expert-tile gain.
- Decision: reject uniform M=8. Keep M=16 above 16 tokens; smaller tiles lose
  enough weight reuse to outweigh reduced padding at the large loads.
- Fifth check today; 5 remained at dispatch, including 2 reserved.

## Attempt 6: fuse BF16 high/low router projection

- File: candidate_router.py, based on checked attempt 4 (M=16 expert tiles).
- For 2-32 tokens, one split-K Triton kernel computes both BF16 hi/lo matrix
  products with separate FP32 accumulators, followed by an FP32 reduction.
- Use the BF16 FFN branch as the router input, as the permitted hi/lo router
  path does. At T=1 retain the efficient FP32 GEMV; above 32 keep SGLang's pair.
- Motivation: the T=8 profile spends roughly 22 us in FP32 router GEMMs.
  Shape-specific dispatch avoids replacing T=1's roughly 5 us FP32 GEMV.
- Router bias, sigmoid, top-8 selection and normalization retain SGLang's op.
- All original FP32 router weights remain in state, with the allowed BF16
  high/low pair. Both matrix accumulations and their reduction stay FP32.
- Local ast.parse passed. Check 9009425450f04502: PASS, 13/13 tests, no flags.
- Speedup 1.1309949x; 95% CI [1.1140300, 1.1479598]; score 20000.3551 us.
- Source SHA256: 8b4e6e50c49c84543e059bac24b2ec9296564f68a85c238bc6741a34c203a50e.
- Profile: profiles/candidate_router.9009425450f04502.txt.
- All timed shapes faster than SGLang; T=8 reaches 1.220-1.228x and T=32
  improves to 1.038-1.086x. Larger batches retain attempt 4 expert tiles.
- Decision: promote the exact checked source to submission.py and recommend
  a scored update with explicit user approval.
- Sixth check today; 4 remained at dispatch, including 2 reserved.

## Approved scored entry for attempt 4

- User explicitly approved one submit. Accepted job cda1f808b7754b1a.
- PASS, 13/13 tests, no flags. Scored speedup 1.1137551x;
  95% CI [1.0970488, 1.1304615]; score 20309.9398 us.
- Profile: profiles/submission.cda1f808b7754b1a.txt.
- Matches attempt 4 check (1.1134628x), confirming a gain above timing noise.
- Current best checked and scored source: attempt 4 in submission.py (da14c66).
- Runtime limits after dispatching attempt 6: 6 checks used, 4 left (2 reserved);
  3 scored submissions used, 2 left.

## Attempt 7: intermediate M=32 route tiles for large batches

- File: candidate_tiles32.py, independently based on checked attempt 4.
- Expert tiles: M=8 for T<=16, M=16 for 17-32 tokens, M=32 above 32.
- Motivation: measured M=8 loses large-batch weight reuse, while baseline
  M=48/64 wastes padding relative to M=16. Test the intermediate tile.
- Router remains SGLang's; this test isolates route-tile performance from
  the still-running attempt 6 router experiment.
- Same weights, precision, FP32 reductions, norms, and complete routing.
- Local ast.parse passed. Check 877b103cc2b240ec: PASS, 13/13 tests, no flags.
- Speedup 1.1116475x; 95% CI [1.0949728, 1.1283222]; score 20348.4461 us.
- Source SHA256: f11e23c4b31138659062db000985e205f284379327748a0f063336317aca7722.
- Profile: profiles/candidate_tiles32.877b103cc2b240ec.txt.
- M=32 reduces T=128 gains (1.128-1.166x versus attempt 4 1.171-1.236x),
  but T=256 reaches 1.124-1.208x versus attempt 4 1.107-1.148x.
- Decision: do not promote as-is. Test M=16 through T=128 and M=32 above
  that, combined with attempt 6 router fusion, as the final non-reserve check.
- Seventh check today; 3 remain, including 2 reserved.
- submission.py promoted separately to checked attempt 6; this check tests the
  original attempt-4 base independently so the tile effect can be assessed.

## Attempt 8: router fusion with measured shape-based expert tiles

- File: candidate_final.py, based on checked attempt 6.
- QK/cache fusion, BF16 split-K dense projections at T<=8, fused pre-expert
  norm/local mapping, permitted BF16 hi/lo router fusion for 2-32 tokens.
- Expert tiles: M=8 for T<=16, M=16 through T=128, M=32 above T=128.
- Combines measured M=16 advantage at T=128 with measured M=32 advantage
  at T=256. Shape dispatch applies generally, with every route still computed.
- Local ast.parse passed. Only dispatch if live check budget exceeds 2.
- Check 226b8fff3e7c46ec: PASS, 13/13 tests, no flags.
- Speedup 1.1514327x; 95% CI [1.1341612, 1.1687042]; score 19645.3508 us.
- Source SHA256: 25745660cab7273dd135e2ff0c9a1400697413f0451c4c9067e8e40dcbe79243.
- Profile: profiles/candidate_final.226b8fff3e7c46ec.txt.
- Every timed shape faster than SGLang: 1.039-1.261x.
- Decision: promote exact checked candidate to submission.py; recommend
  final scored update with user approval. Gain over scored attempt 6 is
  modest and confidence intervals overlap.
- Eighth and final non-reserve check today; exactly 2 checks remain untouched.
- submission.py now holds this exact checked candidate; previous scored
  source remains in candidate_router.py and commit 5961b8d.

## Approved scored entry for attempt 6

- User explicitly approved one submit. Accepted job db5931ab057149c2.
- PASS, 13/13 tests, no flags. Scored speedup 1.1367534x;
  95% CI [1.1197021, 1.1538047].
- Profile: profiles/submission.db5931ab057149c2.txt.
- Submitted best before final entry: attempt 6, source commit 5961b8d.
- Quota: 8 checks used, 2 reserved; 4 scored runs used, 1 left. No further
  check today before the UTC reset, honoring the user's reserve requirement.
- Generated verbose JSON/log profiles stay on disk but are ignored; readable
  timing profiles for every completed attempt are committed with the source.

## Current status after attempt 8

| Attempt | Main change | Check speedup | Correctness | Decision |
|---|---|---:|---|---|
| 1 | Fused QK/cache writes | 1.0137x | 13/13 | Scored 1.0148x |
| 2 | Split-K through T=32 | 1.0200x | 13/13 | Restrict to T<=8 |
| 3 | Expert route tiles M=16 | 1.0689x | 13/13 | Scored 1.0591x |
| 4 | Combined + expert preparation | 1.1135x | 13/13 | Scored 1.1138x |
| 5 | Uniform expert tiles M=8 | 1.0305x | 13/13 | Rejected regression |
| 6 | Fused BF16 hi/lo router | 1.1310x | 13/13 | Scored 1.1368x |
| 7 | Expert tiles M=32 above T=32 | 1.1116x | 13/13 | Use only above T=128 |
| 8 | Router + measured expert tiles | 1.1514x | 13/13 | Scored 1.1437x; submitted best |

- All work is inside muse_decode_layer on master. Every attempt is committed
  with its result and pushed; only explicit decoder paths are staged.
- The previous UTC day ended with 8 checks used and 2 reserved, untouched.
- Final scored submission was explicitly approved and accepted as
  daa4848cafe04e93; PASS at 1.1436969x, the new submitted best.
- No further check was dispatched; the two reserved checks were untouched.

## Approved scored entry for attempt 8 (completed)

- User explicitly approved one submit, then asked to continue.
- Verified master and approved source SHA256 before dispatch:
  25745660cab7273dd135e2ff0c9a1400697413f0451c4c9067e8e40dcbe79243.
- Accepted job daa4848cafe04e93; source commit 581916f.
- Result: PASS, 13/13 tests, no flags; every timed shape faster than SGLang.
- Scored speedup 1.1436969x; 95% CI [1.1265415, 1.1608524];
  score 19778.2292 us versus normalized baseline 22620.3 us.
- Profile: profiles/submission.daa4848cafe04e93.txt.
- Keep the exact approved source in submission.py; its SHA256 is unchanged.
- New submitted best improves on the previous 1.1367534x scored entry.
- No extra check or scored retry launched. Two check slots remain reserved.

## Quota correction after the UTC reset

- Final job was created 2026-11-03 01:39:13 UTC (still Nov 2 in Toronto), so
  it counts in the new UTC day. Earlier pending text incorrectly assumed the
  previous day's remaining scored slot; that text is corrected above.
- Live limits after final scoring: platform 20 checks/day (0 used, 20 left),
  10 scored/day (1 used, 9 left). The platform increased its limits.
- Continue honoring the user's stricter limits: at most 10 checks and 5 scored
  per UTC day, always retain 2 checks, and ask before every okc submit.
- Therefore the new UTC day's user-capped budget is 10 checks (at most 8 may
  be used while retaining the reserve) and 4 further scored entries with approval.
- No new check or scored run was launched after this final approved submission.
- Only muse_decode_layer files were edited/staged; master retained throughout.

## Session handoff and budget (UTC 2026-11-03)

- This board is now worked from the prefill session (Codex stopped). The user
  dropped the 2-check reserve because the kernel boards close 07:59 UTC Nov 5.
  Cap stays 10 checks / 5 scored per UTC day; ask before every scored run.
- Board at handoff: scored 1.1437x (daa4848cafe04e93, attempt 8), rank 1150/1234.

## Attempt 9: local-only route alignment (candidate_align_local.py)

- Based on scored attempt 8 (submission.py). Only change: replace
  moe_align_block_size(local_ids, bm, 256) with _align_local_routes, ported
  verbatim from muse_prefill_layer attempt 10 (checked there; originally from
  muse_moe_w4a16 attempt 6). It aligns the 32 local experts only; nonlocal
  (-1) routes get no padded Marlin blocks. Every local route, weight and
  precision is unchanged; same route tiles M=8/16/32.
- Offline SM90 compile (tools/compile_align_offline.py, Triton 3.5.1) PASS
  for T=1,2,8,16,17,32,64,128,129,256 (offline_align_local.txt).
- Source SHA256: 3c5c140fb8937550b32096cd783a9be94066d451bcbd714715cd6ebd67db7618
- Check 182f48356e8448fe (UTC 2026-11-03 check 1): PASS 13/13 tests, 12 timed
  shapes, no flags. Speedup 1.1505725x; 95% CI [1.1333139, 1.1678310];
  score 19660.0395 us. content_hash 6aca48e6...
- All 12 shapes changed. vs scored attempt 8: geo +0.56%; vs mean of attempt
  8's two runs (same code, 1.1437x/1.1514x): +0.22%. Per shape vs mean: decode
  32 sliding +3.36%, decode 16 sliding +1.49%, verify 16 sliding/full
  -1.58/-1.97%, others -1.2..+1.1%. Attempt 8 run-to-run is up to 1.5%.
- Profile Marlin time: verify 32 446-449 -> 395-410 us, but single-shot
  profiles also show increases elsewhere; timed medians do not confirm a gain.
- NOT verified better (group gain inside noise). submission.py stays attempt 8.

## Attempt 10: bf16_proj attempt-12 kernels for T<=8 projections and T<=16 router

- File candidate_bf16_kernels.py, based on scored attempt 8 (submission.py).
  The decode layer's projections are exactly bf16_proj's layers (qkv_gate
  [2304,8192], o [8192,1024], pooler [4096,8192], unpooler [8192,4096] sliced
  to 1024 rows at T<=8, router [256,8192] FP32).
- Ported verbatim (AST-checked) from muse_bf16_proj attempt 12 (scored
  f7b3322f650247af): _serial_reduce, _split_mm, _tiled_mm, _router_small.
  T<=8: qkv_gate/pooler/unpooler use contiguous 64x64 weight tiles (exact copies
  made in prepare) with single-kernel split-K; o uses cuBLAS (bf16_proj's
  measured choice). Router T=1: FP32 GEMV (_router_small, FP32 input);
  T=2..16: BF16 hi/lo split, FP32 accumulation, single kernel. Separate counter
  buffers for the main stream and the router's side stream.
- Offline SM90: 18 kernel variants, no spills (offline_bf16_kernels.txt).
- Source SHA256: 893cb9bae65cc0148b8edd2c804053e6a9306c9c4fbad4ef1a8f0338589b03f9
- Check 90c232a634c54bea (UTC 2026-11-03 check 2): PASS 13/13 tests, 12
  shapes, no flags. 1.1422922x (CI 1.1251579-1.1594266), score 19802.551 us.
- vs mean of attempt 8's two runs: changed T<=16 shapes +1.35% (decode-1
  +1.64/+2.23%, verify-1 +1.34/+1.62%, decode-16 +0.76/+0.48%). Unchanged
  verify-16/32 shapes -1.7 to -4.2% on identical code (slower machine/run).
- Profile decode-1: _tiled_mm (3 calls) 39.8 us + cuBLAS o 7.1 us = 46.9 us
  versus 50.3 us before (_dense_partials x4 + _dense_reduce x4): only ~3.4 us
  saved. In-layer the tiled kernels run slower than standalone in bf16_proj
  (33.2 us), and cuBLAS o is 7.1 us here versus 4.7 us standalone.
- Not verified better (overall below scored 1.1437; unchanged shapes past -3%).
  Next: attempt 11 restores the old split-K o path (_gemm) at T<=8.
