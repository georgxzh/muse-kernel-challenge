# Muse Spark Open Kernel Challenge

My entries for the [Open Kernel Challenge](https://openkernelchallenge.dev/). The challenge is to write GPU kernels that run parts of **Muse Spark Open** faster than SGLang, the inference engine they're timed against. The model is served on 8×H200 with tensor parallelism 8 (TP8) and expert parallelism 8 (EP8).

Each problem is scored on an H200 as a speedup over SGLang. Every run must first pass correctness tests against an FP32/FP64 reference. Timing replays CUDA graphs with a cold L2 cache, alternating between my kernel and SGLang's on the same fresh inputs.

## Problems

| Folder | What it computes | Key ideas |
|---|---|---|
| [`muse_bf16_proj`](muse_bf16_proj/) | The dense projections: `out = x @ W.T` for the qkv_gate, o, pooler, unpooler and router layers | BF16 GEMM, memory-bound skinny matmuls, split-K, router FP32 via BF16 hi+lo |
| [`muse_attention_decode`](muse_attention_decode/) | One decode step (or an 8-token verify block) of gated grouped-query attention over a paged FP8 KV cache | QK-norm, RoPE, 2,048-token sliding window or full ~80K context, FA3 |
| [`muse_moe_w4a16`](muse_moe_w4a16/) | Routed experts: each token's top 8 of 256 experts, 32 of them local to this GPU, SwiGLU with NVFP4 weights | 4-bit weights with BF16 activations, Marlin kernels, fused post-norm |
| [`muse_decode_layer`](muse_decode_layer/) | A whole decoder layer at decode shapes: projections, attention, router, experts, norms and residuals | Kernel fusion across the full layer |
| [`muse_prefill_layer`](muse_prefill_layer/) | The same layer at prefill shapes: 753 new tokens on an 80K context, and 16K-token first-turn chunks | Large grouped NVFP4 GEMM, compute-bound attention |

Each folder contains:

- `TASK.md`: the problem statement, shapes, correctness bounds and timing rules
- `submission.py`: my kernel. `kernel_fn` is the timed call, and `prepare` does untimed weight repacking
- `baseline.py`: SGLang's implementation, which is what I'm timed against
- `reference.py` / `checks.py`: the reference computation and correctness checks
- `EXPERIMENTS.md`: a log of each attempt, what changed, and the result
- `profiles/`: per-run profiles returned by `okc`

## Results so far

| Problem | Best result | Notes |
|---|---|---|
| `muse_moe_w4a16` | **1.336×** | Local-expert alignment, fused post-norm |
| `muse_bf16_proj` | **1.127×** (check 1.151×, unconfirmed) | Split-K Triton for small m, cuBLAS for large m, faster router |
| `muse_attention_decode` | **1.111×** | One Triton launch fuses QK-norm, RoPE, FP8 quantization and the cache write; FA3 does the attention |
| `muse_decode_layer` | **1.069×** | QK/cache fusion, split-K, smaller expert tiles |
| `muse_prefill_layer` | 0.994× | Work in progress |

Speedup is the geometric mean over loads of SGLang's time divided by mine. Each run reports a 95% interval of at least ±1.5%.

## Rules that shape the code

- **CUDA graphs:** `kernel_fn` must be capturable. That means no `.item()`, `.cpu()` or syncs, and no allocation or branching that depends on input values. Branching on shapes and config is fine.
- **Precision:** weights stay at their own precision (BF16 stays BF16, NVFP4 stays NVFP4), though they can be repacked losslessly. Activations stay BF16 wherever SGLang keeps BF16. The KV cache stays FP8 E4M3.
- **No shortcuts:** attention reads every key, with no sparse or top-k attention. Nothing derived from the inputs may be cached between calls.
- Full rules: https://openkernelchallenge.dev/rules

## Usage

```bash
okc check submission.py     # tests + timing, kept off the leaderboard
okc submit submission.py    # scored run
okc leaderboard <problem>   # standings
```

## Notes

The code is written with AI coding agents, which the challenge rules allow. Entries are licensed Apache-2.0, as the challenge requires.
