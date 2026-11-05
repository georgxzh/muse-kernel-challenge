# Muse Spark Open Kernel Challenge: technical write-up

My entries for the kernel track of the [Open Kernel Challenge](https://openkernelchallenge.dev) (Paradigm with Meta and Andromeda, Oct 22 to Nov 4 2026; the boards closed 2026-11-05 07:59 UTC). Each of the five problems is one piece of Muse Spark Open, a 256-expert MoE model with NVFP4 expert weights, and each submission is a single Python file timed against the SGLang kernel that serves the model today.

Every number below comes from `okc` (leaderboards and run reports), the per-board `EXPERIMENTS.md` logs, or the saved run profiles. Numbers I could not confirm are marked *(unverified)* or explained as estimates.

## Summary

### Hardware and model setup

- The model is served on 8×H200 with tensor parallelism 8 (TP8) for attention and dense layers and expert parallelism 8 (EP8) for the experts. Each problem is one GPU's share of that work: 16 query heads and 2 KV heads per GPU (head dim 64), 32 of the 256 experts local, and all-reduces/all-gathers replaced by the identity.
- Every kernel run uses one H200 SXM inside the `lmsysorg/sglang:dev-muse-spark-open` image (SGLang commit `a55ab7d5`): PyTorch, Triton 3.5.1, FlashInfer, DeepGEMM, TileLang, CuTe DSL and CUTLASS, no network. Native CUDA must be compiled from source inside the file (`load_inline`/NVRTC).
- Each TASK.md lists a per-shape "floor": the time to move the data at 4.8 TB/s or do the math at 989 TFLOP/s, whichever is longer.

### How scoring works

- Each scored shape is captured as a CUDA graph and replayed in the same run, on the same GPU, as SGLang's kernel. Replays alternate between us and SGLang on the same fresh inputs.
- Every timed replay starts with a cold L2: the judge reads a buffer larger than L2 first. A graph replay's fixed cost (about 4 µs) is subtracted from both sides.
- A shape's speedup is the median of paired ratios (SGLang time / our time), taken two repetitions at a time with the order swapped. Our time on the shape is SGLang's median divided by that speedup.
- A load's time is the weighted sum of its shapes (the weight is how many times one step runs the shape, e.g. 46 sliding-window layers and 16 full-attention layers). A board's speedup is the geometric mean over loads of SGLang's load time divided by ours. Each run reports a 95% interval of at least ±1.5%.
- Every test and every timed call is checked against an FP32/FP64 reference. Every scored run listed here passed all tests with no flags.
- The board keeps each team's best scored run. `okc check` runs the same job off the board, with its own daily cap.

### Final results

Leaderboards fetched with `okc leaderboard <problem> --json` on 2026-11-05 around 14:00 UTC, after the close. CIs come from `okc status <job>`. "Ranked" is the number of teams on the board (the SGLang reference row excluded).

| Board | Speedup vs SGLang | 95% CI | Rank / ranked | Best scored job | Our time per step | Board leader |
|---|---:|---|---:|---|---:|---:|
| muse_attention_decode | **1.546×** | 1.523–1.570 | 926 / 1,379 | `7d6c77f92723446c` | 2,435.5 µs | 3.339× |
| muse_moe_w4a16 | **1.362×** | 1.341–1.382 | 1,091 / 1,425 | `aa4935c3095c450d` | 9,876.5 µs | 2.844× |
| muse_decode_layer | **1.327×** | 1.307–1.347 | 1,004 / 1,299 | `444f66ab63174de2` | 17.05 ms | 2.614× |
| muse_bf16_proj | **1.199×** | 1.181–1.217 | 1,137 / 1,746 | `35e70406100a4526` | 4,028.3 µs | 1.728× |
| muse_prefill_layer | **1.157×** | 1.140–1.174 | 733 / 898 | `46d3afde6a5a4ce1` | 334.80 ms | 2.147× |
| **Overall kernel board** | (ranks only) | | **1,075 / 2,014** | | | |

The organizers' Nov 5 update says these standings are not final: submissions are reviewed and final standings are posted before the serving track opens. The top entries are also rerun on nearby shapes. Our dispatch thresholds are on batch or token counts (B ≥ 8, T > 32, T ≤ 1024, T ≥ 4096), not on exact scored shapes.

**Why the overall board lists us on 4 problems, not 5.** The overall board only counts the four decode-step problems. `okc show muse_prefill_layer` says: "Prefill isn't part of a decode step, so this problem has its own board and doesn't count toward the overall board." TASK.md describes the overall board as ranking "by how much your best runs together would speed up a decode step". The leaderboard JSON agrees: nobody on the overall board has more than 4 problems (pjmullin3, rank 1, also shows 4). Counting each team's appearances on the four non-prefill boards reproduces the overall board's problem count for 1,988 of 2,014 rows. Counting all five boards matches only 1,118. I did not track down the remaining 26 rows. The exact overall ranking formula is not published (the API returns ranks only).

## Per-board technical sections

Shape notation: B is the batch, L the query length (1 for decode, 8 for an 8-token speculative verify block), T = B × L the tokens per step, and m/M the row count of a GEMM.

### muse_attention_decode: 1.111× → 1.546×

**Problem.** One decode step or 8-token verify block of gated GQA over a paged FP8 E4M3 KV cache with page size 1: per-head Q/K RMSNorm (q × 5.473), interleaved RoPE on sliding layers, FP8 cache write of the new K/V, softmax(qKᵀ/8)·V over a 2,048-token sliding window or the full ~80K context, then RMSNorm(out) × sigmoid(gate). 12 shapes: B ∈ {1, 16, 32} × {decode, verify} × {sliding (weight 46), full (weight 16)}.

**Where SGLang spends time.** SGLang runs five kernels: FusedQKNorm (Triton), `store_cache`, a Q cast to FP8, FA3 `flash_attn_with_kvcache`, and `rms_norm_gate`. Full-context FA3 dominates. In attempt 1's profile, FA3 took about 189 µs at B=16 and 375 µs at B=32 on full decode, against floors of 63.2 and 126.4 µs. With one-token pages FA3 uses its non-TMA paged path (from reading upstream `flash_api.cpp`). The small sliding shapes have floors of 0.1–3.7 µs, so kernel boundaries matter as much as the math. Even in our three-kernel version, timed minus profiled kernel time was 2.4–4.6 µs on those shapes (attempt 16's audit), when decode-1 sliding ran 9.4 µs.

**What our kernel does.**

- `_prepare` (Triton): Q/K RMSNorm, RoPE in FP32 with libdevice trig, Q to FP8 E4M3, and the scaled FP8 K/V cache store, four head rows per program.
- `_attention_parts` (Triton): split-KV with fixed 2,048-token chunks on full layers. The QK product runs natively in FP8 tensor cores (attempt 6), with FP32 online softmax and accumulation. Full layers use FA3-style P·V: P is scaled ×256, stored in E4M3, and the output is divided by 256 (attempt 8, from FA3's `Max_offset=8`). Sliding layers keep BF16 P·V. Key tiles are N=128, num_stages=2.
- `_combine_gate` (Triton): exact log-sum-exp merge of all splits, then the output RMSNorm × sigmoid(gate).
- PDL (programmatic dependent launch) chains the three kernels. Each consumer calls `gdc_wait()` before any global access, and each producer calls `gdc_launch_dependents()` only after its last store. Triggering after the stores gave +3.98% on the PDL shapes, against +3.47% for triggering at kernel start (attempts 16–17).
- **CUDA full-decode kernel for B ≥ 8** (attempts 36–37/39, `candidates/cuda/attn_full_decode_v3.cu`, built with `load_inline` for sm_90a). One CTA per (sequence, 2,048-token split) covers both KV heads, so each token's K and V are fetched as whole 128-byte rows instead of one 64-byte half per KV-head CTA. Diagnostic 01 measured whole-row gathers 1.55× faster than half-row gathers. Each of the 4 warps runs a 2-stage `cp.async` ring of 32-token tiles. Eight lanes copy one row, so each instruction moves four whole rows. Tiles land in XOR-swizzled shared memory. QKᵀ and P·V use `mma.sync m16n8k32` E4M3: QKᵀ is block-diagonal over the two KV heads, and P·V uses a PRMT byte transpose of V. Softmax is FP32 base-2 online. The kernel uses 168 registers with a 56 B spill, so 3 CTAs fit per SM.
- FA3 stays for full verify at B ≥ 8, where every custom kernel lost (see the negative results).

**Per shape, final scored run `7d6c77f92723446c`** (µs; SGLang is the paired baseline in the same run):

| Shape | Weight | Floor | SGLang | Ours | Speedup | Path |
|---|---:|---:|---:|---:|---:|---|
| decode 1 sliding | 46 | 0.1 | 20.9 | 8.6 | 2.449× | Triton + PDL |
| decode 1 full | 16 | 4.3 | 31.3 | 19.4 | 1.612× | Triton + PDL |
| decode 16 sliding | 46 | 1.6 | 26.6 | 13.4 | 1.986× | Triton + PDL |
| decode 16 full | 16 | 63.2 | 201.1 | 122.2 | 1.645× | CUDA whole-row |
| decode 32 sliding | 46 | 3.3 | 31.7 | 17.3 | 1.830× | Triton + PDL |
| decode 32 full | 16 | 126.4 | 384.1 | 221.5 | 1.734× | CUDA whole-row |
| verify 1 sliding | 46 | 0.1 | 22.9 | 10.0 | 2.286× | Triton + PDL |
| verify 1 full | 16 | 4.3 | 36.6 | 26.5 | 1.381× | Triton + PDL |
| verify 16 sliding | 46 | 1.9 | 29.1 | 19.3 | 1.509× | Triton + PDL |
| verify 16 full | 16 | 63.4 | 205.6 | 199.8 | 1.029× | fused prepare + FA3 |
| verify 32 sliding | 46 | 3.7 | 34.2 | 29.8 | 1.148× | Triton + PDL |
| verify 32 full | 16 | 126.8 | 389.8 | 385.6 | 1.011× | fused prepare + FA3 |

**Full decode at B=16 / B=32 across attempts** (µs, timed shape unless noted):

| Version | B=16 | B=32 |
|---|---:|---:|
| FA3 kernel time, attempt 1 profile | ~189 | ~375 |
| Triton split-KV, BF16 QK/PV (attempt 2) | 181.9 | 331.3 |
| native FP8 QK (attempt 6) | 162.5 | 300.1 |
| + FP8 P·V, PDL (attempt 17, scored 1.4425×) | 155.7 | 284.4 |
| CUDA whole-row kernel (attempt 39, scored 1.5464×) | 122.2 | 221.5 |
| floor | 63.2 | 126.4 |

Scored history: 1.1111× (fused prep + FA3) → 1.2803× (custom sliding) → 1.3835× (FP8 QK / FA3 hybrid) → 1.4279× (FP8 P·V on full) → 1.4425× (PDL) → 1.5464× (CUDA full decode). The remaining gap to the leaders (~3.3×) is mostly the batched full-attention shapes. To reach that, the leaders' full verify-32 must run near its 127 µs floor. My estimate from attempt 23's analysis puts that near 4.9 TB/s of random 128-byte gathers; ours reached 2.3 TB/s and FA3 1.7 TB/s.

### muse_moe_w4a16: 1.132× → 1.362×

**Problem.** The routed experts for one EP8 GPU: top-8 of 256 experts per token, 32 experts local, SwiGLU with NVFP4 weights (`w13` gate/up, `w2` down), BF16 activations, then a per-expert RMSNorm × (1 + w), weighted and summed. M ∈ {1, 16, 32} decode and {8, 128, 256} verify, each weight 60, with skewed routing that makes this GPU the busiest.

**Where SGLang spends time.** SGLang runs `moe_align_block_size` over all 256 global experts, two `moe_wna16_marlin_gemm` calls (W4A16, FP32 reduce), `silu_and_mul`, and a Triton post-expert-norm weighted sum. Marlin takes 78–96% of GPU time (attempt 1 profile). SGLang's block-size heuristic counts remote routes too, so it picks M tiles of 16/48/64 at M=32/128/256 and pads heavily. At the final speedups, Marlin still runs at only 23–36% of the HBM floor bandwidth (attempt 6 profile review).

**What our kernel does.** We keep SGLang's Marlin NVFP4 packing and both Marlin GEMMs with FP32 reduction, and change the scheduling around them:

- **Local alignment** (attempts 1 and 3): one fused Triton kernel, `_align_local_routes`, with one CTA per local expert. It builds a masked 32-bin histogram, computes disjoint padded offsets and scatters stable pair indices. Nonlocal routes (-1) get no Marlin blocks. The buffer capacity is `min(N·BM, N + local·(BM−1))`.
- **Row tiles by token count** (attempt 2): BM=8 up to 32 tokens, 16 above.
- **Adaptive 16/32-row tiles above 32 tokens** (attempt 6): experts with ≤ 16 routes get 16-row Marlin tiles (Marlin's minimum MMA row tile is 16), busier experts get 32-row tiles. Two Marlin calls per GEMM write disjoint rows, SiLU runs once and the norm runs once.
- **Fused post-expert norm** (`_post_norm_sum`, attempt 1): one Triton kernel for RMSNorm × (1 + w) and the FP32 weighted sum at every M.

**Per shape, final scored run `aa4935c3095c450d`** (µs), with attempt 1's check for comparison:

| Shape | Floor | SGLang | Attempt 1 (local align + fused norm) | Final (attempt 6) | Speedup |
|---|---:|---:|---:|---:|---:|
| decode 1 (M=1) | 11.4 | 78.8 | 54.70 | 50.2 | 1.571× |
| decode 16 (M=16) | 52.5 | 188.1 | 162.95 | 147.0 | 1.280× |
| decode 32 (M=32) | 70.8 | 203.7 | 206.40 | 196.2 | 1.038× |
| verify 1 (M=8) | 37.3 | 141.9 | 116.77 | 105.1 | 1.350× |
| verify 16 (M=128) | 101.3 | 459.2 | 469.92 | 313.5 | 1.465× |
| verify 32 (M=256) | 108.9 | 617.0 | 634.55 | 399.6 | 1.544× |

Tile sweep (check speedups): BM 8/16 1.2457×; fused local alignment 1.3440×; uniform BM8 1.2197×; adaptive 8/32 1.3227×; **adaptive 16/32 1.3670×**; adaptive 16/64 1.3491×; BM16 for 9–32 tokens 1.3625×. Every variant landed between 1.22× and 1.37×, so tile choice in Marlin was exhausted. M=32 never improved (196–207 µs in every attempt).

### muse_decode_layer: 1.015× → 1.327×

**Problem.** One whole decoder layer on one GPU at decode shapes: qkv_gate projection, QK-norm/RoPE/FP8 cache write, attention, gated output norm, o projection, post-attention norm and residual, the FP32 router with biased top-8, the pooler, pre-expert norm, the 32 local NVFP4 experts, the unpooler, and the post-FFN norm and residual. 12 shapes with the same B/L/sliding/full grid as attention_decode, at ~80K context.

**Where SGLang spends time.** In attempt 1's profile, Marlin experts take up to 75% of verify time and full-context attention up to 60% of full decode. Small-batch dense projections were the next largest cost. SGLang's post-expert-norm wrapper also launches about 25 µs of PyTorch reduce/elementwise kernels at decode 1, found in attempt 14.

**What our kernel does** (final file `candidate_cuda_full_decode.py` = `submission.py`):

- **Attention block** from muse_attention_decode: `_prepare` → `_attention_parts` → `_combine_gate` with PDL, the CUDA whole-row decode kernel for full decode at B ≥ 8, and FA3 for batched full verify (attempts 13 and 20).
- **Split-K BF16 projections at T ≤ 8**: `_dense_partials` + `_dense_reduce` with FP32 partials and tiles M=16/32, N=64, K=128, split 8 for K ≥ 4096 and 4 otherwise. Above 8 tokens SGLang's cuBLAS dispatch stays, since split-K lost at T=16/32 (0.980–0.990×, attempt 2).
- **Fused hi/lo router for 2–32 tokens** (attempt 6): one split-K Triton kernel (`_router_partials`, BM 16/32, 32-wide N tiles, 16 K-splits) computes both BF16 hi and lo products with separate FP32 accumulators, followed by an FP32 reduction. T=1 keeps the FP32 GEMV, and routing stays SGLang's `FusedMuseSparkRouting`.
- **Fused pre-expert RMSNorm and global-to-local expert-ID mapping** in one kernel (attempt 4).
- **Marlin tiles**: M=8 up to 16 tokens, M=16 for 17–32, and moe_w4a16's adaptive 16/32-row tiles above 32 tokens (attempts 3, 8, 17, 18).
- **moe_w4a16's fused post-expert norm** (`_post_norm_sum`) in place of SGLang's wrapper (attempt 14).

**Per shape, final scored run `444f66ab63174de2`** (µs):

| Shape | Weight | Floor | SGLang | Ours | Speedup |
|---|---:|---:|---:|---:|---:|
| decode 1 sliding | 46 | 40.0 | 171.3 | 121.1 | 1.415× |
| decode 1 full | 16 | 44.2 | 181.9 | 132.2 | 1.376× |
| decode 16 sliding | 46 | 76.7 | 259.3 | 203.8 | 1.272× |
| decode 16 full | 16 | 137.5 | 442.5 | 314.9 | 1.405× |
| decode 32 sliding | 46 | 84.5 | 271.0 | 241.7 | 1.122× |
| decode 32 full | 16 | 208.1 | 623.8 | 436.8 | 1.428× |
| verify 1 sliding | 46 | 62.3 | 266.9 | 180.2 | 1.481× |
| verify 1 full | 16 | 66.4 | 270.4 | 193.2 | 1.399× |
| verify 16 sliding | 46 | 126.6 | 523.5 | 408.0 | 1.283× |
| verify 16 full | 16 | 188.7 | 694.4 | 584.4 | 1.188× |
| verify 32 sliding | 46 | 143.8 | 732.0 | 540.5 | 1.354× |
| verify 32 full | 16 | 267.2 | 1,086.8 | 899.7 | 1.208× |

**Where the gain came from** (board speedup; check / scored):

| Attempt | Change | Check | Scored |
|---|---|---:|---:|
| 1 | fused QK-norm/RoPE/FP8/cache write | 1.0137× | 1.0148× |
| 3 | Marlin route tiles M=8/16 (verify-16/32: 1.131–1.216×) | 1.0689× | 1.0591× |
| 4 | + split-K projections T ≤ 8, fused pre-expert norm/local IDs | 1.1135× | 1.1138× |
| 6 | + fused hi/lo router, 2–32 tokens (T=8: 1.220–1.228×) | 1.1310× | 1.1368× |
| 8 | + tiles M=8/16/32 by T | 1.1514× | 1.1437× |
| 13 | + muse_attention_decode attempt 17 attention block | 1.1850× | 1.1813× |
| 14 | + moe's fused post-expert norm (comment-only re-check) | 1.2743× | not scored |
| 17–18 | + moe's adaptive 16/32 Marlin tiles above 32 tokens | 1.3030× | 1.3023× |
| 20 | + CUDA whole-row full decode at B ≥ 8 | 1.3230× | **1.3266×** |

### muse_bf16_proj: 1.039× → 1.199×

**Problem.** `out = x @ Wᵀ` for the five dense layers: qkv_gate [2304, 8192], o [8192, 1024], pooler [4096, 8192], unpooler [8192, 4096] (only rows 0–1023 up to 8 tokens), all BF16, and the router [256, 8192] in FP32. m ∈ {1, 16, 32, 8, 128, 256}, 30 shapes, all weight 62. Tolerances: 1% relative L2, 0.01% for the router. Rows with a single nonzero entry must be within 0.1%.

**Where SGLang spends time.** SGLang uses cuBLAS (`bf16_gemm_dispatch`) for the BF16 layers. For the router it runs FP32 `F.linear` up to 8 tokens, and above that two BF16 GEMMs on a hi/lo split of the weight with FP32 accumulation. At m ≤ 32 every layer is a weight-streaming GEMV. From the final run's table, cuBLAS reaches 56% of the floor on qkv_gate m=1 (7.9 / 14.0 µs) and 65% on pooler m=1 (14.0 / 21.4 µs). The FP32 router at m=8 takes 26.6 µs against a 1.8 µs floor.

**What our kernel does** (`attempts/attempt16_bf16_partials.py` = `submission.py`):

- **qkv_gate and pooler at m ≤ 16, unpooler at m ≤ 8**: `_tiled_mm` over a bit-exact repack of W into contiguous 64×64 tiles (one 8 KB tile per load), with BM=16, split-K 8 and 3 stages.
- **Single-kernel split-K** (attempts 11–12): each split CTA stores its partial, then takes a GPU-scope `acq_rel` atomic ticket on its output tile. The last CTA sums all partials in fixed order, writes the output and resets the counter (the self-resetting pattern of Marlin's workspace). Since attempt 16 the partials are BF16 (allowed by the Nov 3 update) and are summed in FP32.
- **o at every m, qkv_gate above 16, pooler at m = 17–32 and unpooler at m = 9–32** stay on cuBLAS. A tiled o lost 6.5–17.5%. Pooler/unpooler above 32 use cuBLAS on a contiguous transposed copy of W, which changes cuBLAS's algorithm choice: +10–12% on pooler m=128.
- **Router**: m=1 is a single-kernel FP32 GEMV (`_router_small`, 16 K-splits). m=2–16 is one single-kernel split-K pass that computes the hi and lo BF16 products with separate FP32 accumulators. m=32 is the same kernel plus a separate `_finish` launched with PDL. Above 32 tokens one cuBLAS GEMM runs over the concatenated [K, 2N] hi|lo weight, then a PDL-launched `_join_router` adds the halves. Router partials stay FP32 because of the 0.01% tolerance.

**Per shape, final scored run `35e70406100a4526`** (µs; selected shapes, all 30 are in the run's profile):

| Shape | Floor | SGLang | Ours | Speedup | Path |
|---|---:|---:|---:|---:|---|
| router m=1 | 1.75 | 10.53 | 5.08 | 2.072× | FP32 GEMV, one kernel |
| router m=8 | 1.78 | 26.62 | 6.89 | 3.862× | hi/lo split-K, one kernel |
| router m=16 | 1.81 | 14.98 | 7.16 | 2.091× | hi/lo split-K, one kernel |
| router m=32 | 1.86 | 15.50 | 9.38 | 1.653× | hi/lo split-K + PDL finish |
| router m=128 | 2.21 | 16.37 | 9.19 | 1.780× | one cuBLAS [K,2N] GEMM + PDL join |
| router m=256 | 2.68 | 17.47 | 9.86 | 1.772× | one cuBLAS [K,2N] GEMM + PDL join |
| qkv_gate m=1 | 7.87 | 13.98 | 12.34 | 1.134× | tiled split-K |
| qkv_gate m=16 | 7.93 | 14.30 | 13.04 | 1.097× | tiled split-K |
| pooler m=1 | 13.99 | 21.41 | 18.19 | 1.177× | tiled split-K |
| pooler m=16 | 14.06 | 21.25 | 18.52 | 1.148× | tiled split-K |
| pooler m=128 | 14.64 | 25.07 | 22.70 | 1.105× | cuBLAS, transposed W |
| unpooler m=1 | 1.75 | 7.04 | 6.49 | 1.084× | tiled split-K |
| o m=1 | 3.50 | 6.78 | 6.59 | 1.029× | cuBLAS (as SGLang) |
| qkv_gate m=32 | 8.00 | 14.30 | 14.09 | 1.015× | cuBLAS (as SGLang) |

Scored history: 1.0389× → 1.1292× (measured dispatch, router occupancy) → 1.1466× (tiled weights, transposed W) → 1.1656× (one cuBLAS GEMM for both router halves) → 1.1701× (single-kernel split-K) → 1.1736× (PDL into the router's second pass) → 1.1992× (BF16 partials). The router gives most of the gain: at verify 1 (m=8) it accounts for 19.7 of the 24.7 µs saved across the five layers. o and qkv_gate at m ≥ 32 run cuBLAS like SGLang and land at 1.00–1.04×.

### muse_prefill_layer: 0.994× → 1.157×

**Problem.** The same decoder layer at prefill shapes: 753 new tokens on an 80K-token context (turn 1: B=1, turn 4: B=4, so 3,012 tokens), and 16,384-token first-turn chunks at contexts 0, 32K and 64K. There are three loads (turn 1, turn 4, first turn), with weights 46/16 sliding/full and 138 for the first-turn sliding chunk. This board does not count toward the overall board.

**Where SGLang spends time.** Marlin expert GEMMs take 10.5–11.6 ms per 16K prefill, 43–66% of GPU time (attempt 1 profile). Full attention takes 4.5–8 ms on the long prefills (attempt 6). On the 753/3,012-token turns, Marlin is 64–66% of the time (cross-board review).

**What our kernel does** (`attempt10.py` = `submission.py`):

- **T ≥ 4096: grouped BF16 experts.** The 32 local experts' NVFP4 weights are expanded to BF16 inside the timed call (`_expand_pairs`; about 2.1 GB for w13 and 1.07 GB for w2). Nothing expanded persists in `state`. Routed rows go through DeepGEMM BF16 grouped GEMMs with FP32 accumulation in DeepGEMM's PSUM layout: a GPU histogram counts local routes, a prefix scan gives each expert's 128-aligned begin and actual end, and an atomic allocator packs every route. Attempt 3's padded-capacity layout checked at 0.918×; the PSUM layout took it to 1.139× (attempt 4).
- **Exact paired NVFP4 decode** (attempt 9): each packed byte decodes to two exact E2M1 values through a PRMT lookup into FP16 encodings, converted to FP32. One rounded FP32 SF×GS product is shared by both values, and one `cvt.rn.bf16x2.f32` store writes the pair. The kernel uses 32 registers with no spills, against 56 for the single-value decoder, and expansion time fell from ~1.45 ms to ~1.06 ms per 16K prefill (−27%). All 256 byte decodings were verified on CPU.
- **T ≤ 1024: local 32-bin alignment** (moe_w4a16's `_align_local_routes`) with SGLang's Marlin tiles, so nonlocal routes get no Marlin blocks (attempt 10).
- **Turn 4 (3,012 tokens)** uses SGLang's `moe_align_block_size` over the 32 local expert bins (not 256) and Marlin. Fused Triton QK-norm/RoPE/FP8 cache write and FA3 attention are used at every size.

**Per shape, final scored run `46d3afde6a5a4ce1`:**

| Shape | Tokens | Weight | SGLang | Ours | Speedup |
|---|---:|---:|---:|---:|---:|
| turn 1 sliding | 753 | 46 | 1,057.4 µs | 1,025.8 µs | 1.031× |
| turn 1 full | 753 | 16 | 1,313.8 µs | 1,281.6 µs | 1.025× |
| turn 4 sliding | 3,012 | 46 | 3,217.1 µs | 3,203.5 µs | 1.004× |
| turn 4 full | 3,012 | 16 | 4,626.4 µs | 4,613.1 µs | 1.003× |
| first turn full, ctx 0 | 16,384 | 16 | 18.07 ms | 11.72 ms | 1.541× |
| first turn sliding, ctx 32K | 16,384 | 138 | 17.04 ms | 10.97 ms | 1.553× |
| first turn full, ctx 32K | 16,384 | 16 | 20.63 ms | 15.02 ms | 1.374× |
| first turn full, ctx 64K | 16,384 | 16 | 23.50 ms | 18.08 ms | 1.300× |

Scored history: 1.1394× (PSUM grouped GEMM) → 1.1506× (paired decode; long shapes +1.3 to +4.4%) → 1.1571× (short-prefill local alignment; batch-1 turn +2.35/+1.96%). In attempt 10, a 16K prefill spends ~5.2 ms in the expert path: expansion 1.06 ms plus DeepGEMM 2.7 + 1.4 ms. The fixed ~1,063 µs expansion keeps the grouped path slower than Marlin below roughly 3,000 tokens: moving the threshold to 2,048 tokens made the 3,012-token shapes 0.924×/0.936× (attempt 5).

## Cross-board reuse

The five problems overlap: decode_layer contains attention_decode, bf16_proj and moe_w4a16, and prefill_layer is the same layer at other shapes. Kernels were ported verbatim (AST-checked where noted in the logs) and re-verified on the target board. The verified-better rule applied to ports the same way as to new kernels.

| From | Into | What moved | Measured effect on the target |
|---|---|---|---|
| attention_decode attempt 17 (Triton + PDL) | decode_layer attempt 13 | the whole attention block (prepare, split-KV FP8 attention, combine + gate) | full decode-32 +19.5% (590 → 492 µs), full decode-16 +9.6%, decode-1 +5.2/+4.0%; board 1.1437× → 1.1813× scored (+3.3%) |
| attention_decode attempt 37 (CUDA whole-row decode) | decode_layer attempt 20 | CUDA full decode for B ≥ 8 | full decode-16 +11.0% (354 → 316 µs), decode-32 +14.2% (505 → 432 µs); board 1.3023× → 1.3266× scored |
| moe_w4a16 attempt 6 (`_post_norm_sum`) | decode_layer attempt 14 | fused post-expert norm | decode-1 +20.1/+18.5%, decode-16 +17.2/+9.8%, verify-1 +15.8/+15.7%; +7.5% geomean over all shapes |
| moe_w4a16 attempt 6 (adaptive 16/32 tiles) | decode_layer attempts 17–18 | Marlin tile schedule above 32 tokens | verify-32 sliding +12.1% (608 → 534 µs), full +6.0%; verify-16 sliding +6.5%, full +2.6% |
| moe_w4a16 `_align_local_routes` | prefill_layer attempts 8/10 | local alignment at ≤ 1024 tokens | batch-1 turn +2.35% / +1.96% (scored) |
| prefill_layer attempt 9 (PRMT paired decoder) | moe_w4a16 attempts 9–11 | exact NVFP4 decode inside a skinny GEMM | correct, but 0.58–0.67× (see negative results) |
| bf16_proj attempt 12 (tiled single-kernel split-K, router kernels) | decode_layer attempts 10–11 | projections at T ≤ 8, router at T ≤ 16 | +1.0 to +1.5% on the changed shapes, inside noise; not adopted |
| moe_w4a16 attempt 6 / adaptive tiles, `_post_norm_sum` | prefill_layer attempts 7, 14 | short-prefill Marlin schedule, post-norm | attempt 7 slower (0.969×/0.972× on the short shapes); attempt 14 −0.25 to 0.00% |

The router in decode_layer: the fused hi/lo split-K router that decode_layer scores with (attempt 6, T=8 at 1.220–1.228×) uses the same method as bf16_proj's router, one split-K kernel with separate FP32 hi and lo accumulators. It was written in the decode_layer session, not ported. The verbatim port of bf16_proj's newer single-kernel router and tiled projections (attempts 10–11) gained only 1–1.5% on the changed shapes. In the layer, the tiled kernels ran slower than standalone (39.8 µs for three calls against 33.2 µs in bf16_proj), and cuBLAS o took 7.1 µs against 4.7 µs standalone.

attention_decode's kernels do not apply to prefill. They are built for 1 or 8 query tokens per sequence, while prefill attends 753 to 16,384 query tokens per sequence, so FA3's prefill path stays there.

## Negative results

Each of these was measured on the judge (`okc check`) unless marked offline.

| Experiment | Result | Root cause |
|---|---|---|
| **Fused NVFP4 dequant + grouped GEMM** (prefill attempt 12: 128×128 tiles, 8 warps, 3 stages, wgmma; 127 regs, 0 spills offline) | correct; board 1.0379× vs 1.1571×; 16K shapes 1.059–1.092× vs 1.30–1.55× | The fused GEMMs took 9.3–9.5 ms per 16K prefill against 5.2 ms for expansion + DeepGEMM, 1.8× slower (~215 vs ~490 effective TFLOP/s). Each weight is decoded once per 128-row tile, 5–6 times in total instead of once, and the decode runs in the same warps as the wgmma loop. BM=256 needs 256-aligned segments (~20% padding) and ~200+ registers; by estimate even that plus a cheaper decode only reaches parity. |
| **Custom W4A16 MoE GEMM vs Marlin** (moe attempts 9–11: native NVFP4 bytes, PRMT decode, BF16 `tl.dot`) | correct; 0.5758× (attempt 10), 0.6709× with 2-deep register prefetch (attempt 11). M=1: 220 → 169 µs vs Marlin's 50 µs | Instruction-bound. The SASS inner loop spends ~19 instructions per weight value (LOP3 235, IMAD 204, PRMT 140, byte-wise LDG 120, HADD2 96, FMUL 96, STS 84 for a shared-memory layout change, F2FP 52, HMMA 16). Marlin needs roughly 2: word-level bit-trick dequant, bf16x2 scale multiply, cp.async pipelining, no shared-memory round trip. Prefetching hid latency for only 20–30%. Attempt 9 also exposed an alignment bug: with BM=16 > N=8, stale row indices in a padded segment survived graph replays (rel. L2 0.862); the fix sizes lanes as `next_pow2(max(N, BM))`. |
| **L2 prefetch for whole-row fetches** (attention attempts 29/31/32, full layers) | 1.2875× / 1.3824× / 1.3624× vs 1.4425× scored; full decode-32 386.4 / 356.9 / 346.5 µs vs 284.4 | `cp.async.bulk.prefetch.L2` (wrapped by ptxas in a 32-lane loop), per-thread `prefetch.global.L2` and `ld.global.L2::128B` all add fetch requests that cost more than the whole-line DRAM fetches save. The working fix was the CUDA kernel's real whole-row loads. `.cg` loads (attempt 9) did not help either (1.2843×). |
| **Fused sliding-window decode kernel** (attention attempts 42–44: prepare + attention + combine/gate in one CUDA launch) | attempt 42 failed to build; 43: sliding decode-1 21.3 µs; 44: 16.4 µs vs 8.4 µs for the Triton PDL chain | Attempt 42's host wrapper passed `int64_t*` where the kernel took `const long long*`. NVRTC checks device code only, so the offline check missed it. In 43, the last CTA's combine looped over splits with serialized L2 loads. 44 loaded all splits at once (kernel 13.2 µs) but still lost to three short PDL-chained Triton kernels. |
| **CUDA full decode at batch 1** (attention attempts 45/46: 512-/1024-token splits; attempt 36 at 256) | full decode-1 21.1 / 24.6 µs vs 19.3–19.4 µs on Triton (−8.1 / −21.6%); attempt 36 −5.4% | One sequence has too little parallelism. Longer splits leave fewer CTAs, and with short splits each warp gets 2 tiles, so pipeline fill and the cross-warp merge dominate. Batch 1 stays on Triton. |
| **Second-stream NVFP4 expansion** (prefill attempt 11: expansion forked onto a side stream to overlap QKV/FA3/o/pooler/router) | 1.1594× vs 1.1571×; long-shape group +0.81% (+0.34% vs the mean of both attempt 10 runs), inside the 0.7–2.7% run-to-run spread | GPU-busy time fell only 0.2–0.3 ms, the same overlap attempt 10 already had, and `_expand_pairs` slowed from 1.06 to ~1.15 ms. The 524K-CTA expansion grid and the persistent cuBLAS/DeepGEMM/FA3 kernels cannot co-reside on the SMs, so they run in turn. |
| CUDA full-verify kernel replacing FA3 (attention 38/40, decode_layer 21) | full verify-16/32 331.7/628.1 µs vs FA3 198.6/382.5 (−39/−38%); a 4-stage ring barely helped (324.4/610.9) | Bound by the per-warp softmax/MMA chain: 128 query rows per token. |
| CUDA tensor-core GEMV for BF16 layers at m ≤ 16 (bf16_proj 17, decode_layer 22) | 1.0782× vs 1.1992×; qkv_gate m=1 15.3 vs 12.3 µs | Per-lane 16-byte strided reads split every 128-byte line over two instructions. The tiled layout streams contiguous 8 KB tiles. |
| Deeper Triton pipelining, mask-free tiles, both-heads Triton CTAs (attention 18–23, 25–26) | all slower or flat (e.g. mask-free: full shapes −9.7%) | Diagnostic probes showed the Triton kernel sat at the half-row gather floor. Neither page-table latency, instruction count (759 → 572 SASS per tile bought nothing) nor occupancy was the limit. |
| Query tiling in verify (attention 11–13), occupancy-sized dynamic splits (10), `_prepare` fused into attention (24) | 0.986–1.157× | Repeated cache traffic per query tile; worse full-decode splits; libdevice sin/cos recomputed on every CTA's critical path. |
| Marlin BF16 split-K reduce (`use_fp32_reduce=False`; moe 12, decode_layer 16, prefill 13) | −0.7% to −4.3% | Marlin's BF16 reduce path is slower here. |
| Lossless 12-bit BF16 packing (bf16_proj 3/8) | judge compile error (`ttg.local_alloc` i1) | The same kernel compiled with local Triton 3.5.0 and 3.5.1, so a local compile pass is not proof that the judge will compile it. By estimate 68–80% of 128×64 tiles contain an escape, so the line was parked. |

## Methodology

### The verified-better rule

The rule was set by the user on 2026-11-02 and used on every board. A candidate is better only if:

1. the shapes whose code changed gain as a group beyond run-to-run noise (about 3%);
2. no single shape regresses more than 3% against the mean of the current best's runs; and
3. the overall speedup is higher than the board's current scored best, even if the confidence intervals overlap.

The first version required each changed shape to beat 3%. The same day it was relaxed to the group reading above (bf16_proj attempt 5 was the test case). Two standing additions came on 2026-11-04 and are logged in `SUBMISSION_LOG.md`:

- **Overnight rule** (~05:25 UTC): a candidate that passes the rule and whose file hash matches its check may be submitted without asking. decode_layer attempt 20 was submitted this way.
- **Unchanged-code waiver** (~14:55 UTC): if the only shapes that miss the rule run unchanged code and their own kernel time is unchanged, treat it as noise. attention_decode attempt 39 was submitted under it: decode-1 sliding read −4.2% at an unchanged 8.4 µs, and across 7 checks that shape's ratio against attempt 17's mean ranged +0.7 to −4.2%.

Every other scored run had explicit user approval.

### Hash matching before scored runs

okc's server `content_hash` covers problem metadata, so it differs from the raw file SHA256. For each candidate:

1. Record the raw SHA256 when it goes to `okc check`.
2. Before `okc submit`, confirm the file still has that SHA256, and copy it unchanged to `submission.py`.
3. After the scored run, confirm its server `content_hash` equals the check's.

All scored runs since Nov 3 are logged with both hashes in the EXPERIMENTS.md files and in `SUBMISSION_LOG.md`, and every pair matched. At the close, each board's `submission.py` is byte-identical to the file of its best scored run (SHA256 checked 2026-11-05).

### Offline compilation and resource checks

There was no local GPU, so judge checks were the only correctness and timing signal. Every Triton kernel was first compiled offline for SM90 with the judge's Triton 3.5.1 (`tools/compile_*.py` in each folder, interpreter and caches in the untracked `.tooling/`/`.tools/`). Each compile was checked for:

- **Registers and spills.** For example, full decode at N=256 needed 196 registers against 120 at N=128, halving the estimated residency; attempt 15 then measured the loss (full decode-32 322.9 vs 284.0 µs).
- **The instructions that matter in SASS**: `ACQBULK` before the first global access and `PREEXIT` after the last store for PDL; `ATOMG`/`MEMBAR`/`CCTL` for the single-kernel split-K; HGMMA and cp.async for the fused GEMM.
- **Pointer alignment.** Offline compiles must model real tensor alignment (`tt.divisibility=16`). Without it, register estimates and load widths were wrong: attention found this at attempt 8, and prefill attempt 12's scalar loads were an artifact of the same omission.

CUDA kernels were compiled with NVRTC + ptxas (`tools/nvrtc_compile.py`). Their fragment layouts were verified by lane-level NumPy emulation against direct attention or GEMM, to 1e-15 / 2e-15 / 6e-14 (`tools/emulate_full_decode.py`, `emulate_full_verify.py`, `emulate_gemv.py`). bf16_proj attempt 5 also ran its real kernel code on CPU through a mock of `triton.language` and scored it with `reference.py`. These checks catch most failures, but not all. The judge rejected a kernel that compiled locally (bf16_proj attempt 3), and NVRTC does not see host wrappers (attention attempt 42).

### Noise estimates from repeated runs

- okc returns the earlier result for byte-identical code ("duplicate"). To measure code again we used a comment-only copy (attention attempt 39 re-measured 37; decode_layer `candidate_postnorm_b.py` re-measured attempt 14) or the scored run itself.
- Observed spreads on unchanged code:
  - bf16_proj: mean 1.2%, max 2.9% between attempt 2's two runs. o m=256 read 1.021/0.990/0.976/0.972 across four runs, router m=1 between 1.791 and 1.881.
  - decode_layer: attempt 8's runs differed by up to 1.5%. On identical code, verify-16 read −5.8/−6.4% because SGLang's paired time moved.
  - prefill: up to 1.6% (16K ctx 0) and below 0.8% elsewhere for attempt 4; 0.13–0.26% on the batch-1 turns for attempt 9; 0.7–2.7% on the long shapes for attempt 10.
  - attention: attempt 14's two runs differed 12.7% on full decode-32 (250.1 vs ~284 µs).
- Comparisons are made per shape against the mean of the incumbent's runs, not one run. `muse_attention_decode/tools/compare_profiles.py` normalizes a candidate's paired ratios against the scored run's per-case baseline and recomputes the load-weighted score; it reproduces the scored 1.38346370× exactly.

## Rule and limit changes during the challenge

| When | Change (source) | How it affected this work |
|---|---|---|
| 2026-11-02 08:00 PT | Submissions must be readable source: no prebuilt or embedded binaries, no compressed/obfuscated code (updates page) | Our CUDA kernels are source strings in the `.py` file, compiled at import with `load_inline` (nvcc, sm_90a). |
| 2026-11-03 ~01:03 UTC | Per-problem daily caps rose from 10 checks / 5 scored runs to 20 / 10 (observed in `okc problems`; the docs say limits may change with load) | We kept the user's 10 / 5 cap. The 2-check daily reserve was dropped later on Nov 3 because the boards closed 07:59 UTC Nov 5. |
| **2026-11-03 12:15 PT** (20:15 UTC) | Kernel rules: approximate math accurate to BF16 precision, partial sums (split-K, split-KV) held in BF16 or FP16, and FP16 accumulation of QKᵀ and P·V in attention are allowed (updates page, rules) | See below. |
| current rules *(I could not confirm from the updates page whether this was part of the Nov 3 change)* | P may be stored in FP8 E4M3 at any power-of-two scale, with exp2 approximated | We already used FP8 P ×256 from attempt 8 (Nov 3 UTC, before the update), justified from FA3's own FP8 mainloop and `Max_offset=8`. |
| 2026-11-03 15:45 PT | Serving track moved to Nov 10–19; serving leaderboard private | No effect on the kernel work. |
| 2026-11-05 00:00 PT | Kernel track closed; standings not final pending review | Final numbers above. |

How the Nov 3 precision allowances were used (all measured):

| Board / attempt | Allowance | Result |
|---|---|---|
| bf16_proj 16 | BF16 split-K partials for the tiled BF16 layers (router kept FP32) | all 8 changed shapes faster, +2.83% as a group; **scored 1.1992×, the final entry** |
| attention 27 | FP16 split-KV partials | −1.0% over all shapes, rejected |
| attention 28 / 33 | FP16 P·V with FP16 per-tile accumulation on sliding layers (all / verify only) | 28: sliding verify +1.6 to +6.1% but decode-32 −12.0%; 33: +2.2% on sliding verify, ~0.2–0.3% expected on the board, judged too close to noise to submit |
| decode_layer 15 | BF16 partials in the T ≤ 8 split-K GEMMs | +0.24%, noise, rejected |
| decode_layer 19 | attention 33's FP16 P·V | −0.9% on the changed shapes, rejected |
| moe 12, decode_layer 16, prefill 13 | Marlin's BF16 split-K reduce | −0.7 to −4.3%, rejected |
| prefill (review) | FP8 Q/P, BF16 partials | Not applicable: FA3 already runs FP8 Q/P there, and the GEMMs are cuBLAS/DeepGEMM/Marlin. |

## Repository layout and how to reproduce

```
README.md              this write-up
SUBMISSION_LOG.md      scored runs from Nov 4 with hashes, rule checks and approvals
muse_<problem>/
  TASK.md              problem statement, shapes, weights, floors, correctness bounds, timing rules
  AGENTS.md            okc's notes for coding agents
  submission.py        the final entry (byte-identical to the best scored attempt)
  baseline.py          SGLang's kernel, the timing baseline
  reference.py, checks.py, nvfp4.py   the judge's reference, checks and NVFP4 helper (local copies)
  EXPERIMENTS.md       every attempt: change, hashes, job IDs, per-shape results, decision
  tools/               offline compile, SASS/resource inspection, emulation, profile comparison
  profiles/            okc run reports for some runs
```

Where the attempts live:

| Board | Candidates | Run results | Final file |
|---|---|---|---|
| muse_attention_decode | `candidates/attemptNN_*.py`, CUDA sources in `candidates/cuda/` | `candidates/check_*.json`, `scored_*.json`, `offline_*.txt` | `candidates/attempt39_cuda_full_decode_b8_remeasure.py` |
| muse_bf16_proj | `attempts/attemptNN*.py`, CUDA GEMV in `cuda/` | `attempts/check_NN.json`, `submit_NN.json`, `offline_*.txt` | `attempts/attempt16_bf16_partials.py` |
| muse_decode_layer | `candidate_*.py` in the folder root | `check_*.json`, `submit_*.json`, `offline_*.txt` | `candidate_cuda_full_decode.py` |
| muse_moe_w4a16 | `attempts/attempt_NN.py` | `attempts/check_NN.json`, `offline_skinny_*.txt`, `attempts/profiles/` | `attempts/attempt_06.py` |
| muse_prefill_layer | `attemptN.py`, `attemptNN_*.py`, `candidate_combined.py`, `prototypes/fused_nvfp4_mm.py` | `check_*.json`, `submit_v*.json`, `offline_*.txt`, `*_resources.txt` | `attempt10.py` |

To reproduce (the kernel track is closed, so new runs now return HTTP 409; these are the commands that were used):

```bash
okc check <problem> <file.py>     # tests + timing, kept off the board
okc submit <problem> <file.py>    # scored run; the board keeps your best
okc status <job>                  # per-shape SGLang vs ours, 95% CI, profile path
okc leaderboard <problem> --json  # rank, speedup and job ID of every row
okc leaderboard --json            # overall board (ranks and problem counts only)
```

Re-submitting byte-identical code returns the earlier job ("duplicate"). The offline tools need a local Python with Triton 3.5.1 (and NVRTC for the CUDA tools). That environment and the compiler caches live in the untracked `.tooling/` and `.tools/` folders and are not part of the repository.

## Notes

- Written with AI coding agents (OpenAI Codex, Claude Code and Kiro), as the challenge rules allow. Every scored submission was approved by me or ran under a standing rule I set, logged in `SUBMISSION_LOG.md`.
- Entries are licensed Apache-2.0, as the challenge requires.
