# Muse Spark Open Kernel Challenge

My entries for the [Open Kernel Challenge](https://openkernelchallenge.dev): GPU kernels that run parts of Muse Spark Open faster than SGLang, the inference engine they're timed against. The model is served on 8×H200 with tensor parallelism 8 (TP8) and expert parallelism 8 (EP8).

## Results

Final scored runs, as a speedup over SGLang on an H200 (higher is better). Ranks as of 2026-11-04 evening:

| Problem | Speedup | Rank | Scored job | What made the difference |
|---|---|---|---|---|
| muse_attention_decode | **1.546×** | 912 | 7d6c77f92723446c | CUDA whole-row full-context decode (B ≥ 8), paged-FP8 Triton kernel, PDL |
| muse_moe_w4a16 | **1.362×** | 1084 | aa4935c3095c450d | Local route alignment, adaptive 16/32 Marlin tiles, fused post-norm |
| muse_decode_layer | **1.327×** | 996 | 444f66ab63174de2 | Reused the attention_decode kernels inside the full layer, adaptive Marlin tiles |
| muse_bf16_proj | **1.199×** | 1124 | 35e70406100a4526 | Single-kernel split-K router, BF16 partial sums |
| muse_prefill_layer | **1.157×** | 723 | 46d3afde6a5a4ce1 | Paired FP4 weight decode, short-prefill scheduling |

Overall leaderboard rank: **1066**.

Every scored run passed all correctness tests against an FP32/FP64 reference, with no flags.

## Problems

| Folder | What it computes | Key ideas |
|---|---|---|
| muse_bf16_proj | Dense projections: out = x @ W.T for the qkv_gate, o, pooler, unpooler and router layers | Memory-bound skinny GEMMs, split-K, router FP32 via BF16 hi+lo |
| muse_attention_decode | One decode step (or an 8-token verify block) of gated GQA over a paged FP8 KV cache | QK-norm, RoPE, 2,048-token sliding window or ~80K full context |
| muse_moe_w4a16 | Routed experts: top 8 of 256 per token, 32 local per GPU, SwiGLU with NVFP4 weights | 4-bit weights, BF16 activations, Marlin |
| muse_decode_layer | A whole decoder layer at decode shapes | Fusion across projections, attention, router, experts, norms |
| muse_prefill_layer | The same layer at prefill shapes: 753 tokens on 80K context, and 16K-token chunks | Grouped NVFP4 GEMM, compute-bound attention |

## What worked

- **Removing launch overhead with PDL.** On small decode shapes, 15–30% of runtime was gaps between kernels. Programmatic dependent launch, signalled *after* the producer's writes, gave +4% on the changed shapes.
- **Reusing kernels across problems.** The attention_decode kernel dropped into decode_layer in place of FA3: +19.5% on full decode-32 and about +10% on the board overall.
- **One kernel instead of two.** The router's split-K partial pass plus finish kernel was launch-bound at m ≤ 32; fusing them gave +5–8% per shape.
- **BF16 partial sums** (allowed by the November rule update): half the split-K traffic.
- **Exact paired FP4 decode**: two values per packed byte, 32 instead of 56 registers, no spills.
- **CUDA whole-row decode for full context at B ≥ 8** (attention_decode): +28% on full decode-16/32; ported into decode_layer for +11–14% on those shapes.
- **Adaptive Marlin tile sizes** (16/32 rows by token count) in MoE and decode_layer: +4.5% on the changed verify shapes.

## What didn't

- **Fused dequant + grouped GEMM (prefill):** correct but 1.8× slower than expand-then-DeepGEMM; each weight is decoded 5–6 times, once per row tile.
- **Custom W4A16 MoE GEMM:** about 3× slower than Marlin at 1 token; byte-wise loads cost about 20 instructions per weight.
- **Second-stream weight expansion, L2 prefetch hints, FP16 partials/PV in attention:** within noise.
- **Fused sliding-window decode kernel:** 16.4 µs vs 8.4 µs for the Triton path.
- **CUDA full decode at batch 1:** too little parallelism; -8 to -22% vs Triton.
- **CUDA GEMV for small projections:** slower than Triton split-K (bf16_proj and decode_layer).
- **Marlin BF16 reduce path:** -1 to -4% on MoE, decode_layer and prefill.

## How results were verified

- **A candidate counts as better** only if the changed shapes gain as a group beyond run-to-run noise (about 3%), no shape regresses by more than 3%, and the overall speedup is higher.
- **The content hash of every scored submission is checked** against the passing check run before submitting.
- **Kernels are compiled offline** with the judge's Triton 3.5.1 (and checked for register spills) before spending a GPU check.

## Repository layout

Each problem folder contains:

- `TASK.md`: problem statement, shapes, correctness bounds and timing rules
- `submission.py`: the scored kernel. `kernel_fn` is the timed call; `prepare` does untimed weight repacking
- `attempts/`, `candidates/`: every attempt, checked or not
- `baseline.py`: SGLang's implementation, the timing baseline
- `reference.py`, `checks.py`: reference computation and correctness checks
- `EXPERIMENTS.md`: a log of each attempt, what changed and the result
- `profiles/`: per-run profiles returned by `okc`

## Rules that shape the code

- **CUDA graphs:** `kernel_fn` must be capturable, so no `.item()`, `.cpu()` or syncs, and no allocation or branching on input values. Branching on shapes and config is fine.
- **Precision:** weights keep their precision (BF16 stays BF16, NVFP4 stays NVFP4) but can be repacked losslessly. Activations stay BF16 wherever SGLang keeps BF16, and the KV cache stays FP8 E4M3. Since November 3, BF16/FP16 partial sums, FP16 attention accumulation and FP8 Q/P are allowed.
- **No shortcuts:** attention reads every key, and nothing derived from the inputs is cached between calls.

Full rules: https://openkernelchallenge.dev/rules

## Usage

    okc check submission.py     # tests + timing, kept off the leaderboard
    okc submit submission.py    # scored run
    okc leaderboard <problem>   # standings

## Notes

Written with AI coding agents (OpenAI Codex, then Claude Code), as the challenge rules allow. Entries are licensed Apache-2.0, as the challenge requires.
