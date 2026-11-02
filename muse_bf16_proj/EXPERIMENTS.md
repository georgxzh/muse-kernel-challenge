# muse_bf16_proj experiments

## Operating constraints

- Work only in this folder on shared `master`; stage and commit only this problem's files.
- User deadline: two days from 2026-11-02. Establish a correct speedup early, request permission for each `okc submit`, then improve.
- AI assistance is allowed. Read the live https://openkernelchallenge.dev/rules and `okc --help` on 2026-11-02. The live rules explicitly discuss agent submissions; the homepage supplies an agent workflow.
- CLI/account: `/home/georg/.okc/bin/okc` in Ubuntu WSL; already logged in as `georgxzh`.
- CLI reported 10 checks and 5 scored runs per day for this problem, resetting at 00:00 UTC. Always retain two checks. At most eight checks may be used from a fresh daily allowance.
- H200 SXM, single Python source, existing image dependencies only, no prebuilt binaries or obfuscated code. Compilation counts toward the 10-minute run limit; prepare has a 300-second limit.
- Preserve BF16 weights and activations. Router FP32 weights may use BF16 hi/lo with FP32 accumulation. Lossless weight repacking is allowed.
- Relative L2 <= 1% (BF16 projections), <= 0.01% (router); per-row <= 5x bound; single-nonzero probe rows <= 0.1%; no NaN/Inf; exact output dtype and shape.
- Graph-capturable timed calls, no host synchronization or value-dependent allocation/host control flow. All activation-dependent computation stays inside the timed call. No cached outputs, harness access, reference imports, networking or child processes in the submission.
- Timed cold-L2 CUDA graph replays alternate with the SGLang baseline; graph fixed cost is subtracted. Score is geometric mean of six weighted-load speedups; use reported confidence intervals.
- No local GPU or local torch/triton installation. Local Python syntax checks cannot establish GPU correctness.
- Windows sandbox launcher fails before process creation. Commands run through the approved elevated execution fallback and use WSL Git, avoiding Windows Git ownership configuration changes.

## Attempt 1 - conservative split-K Triton

- Candidate: `submission.py`; check job `c4232f44cf7e460e`.
- Small projection batches (m <= 32): BF16 tensor-core tiles (16 x 64 x 64), FP32 split-K partial sums and a separate reduction. Split 8 for K >= 4096, otherwise 4.
- Larger BF16 batches: `torch.mm`, matching the baseline's cuBLAS strategy.
- Router m <= 8: FP32 weight GEMV reductions; larger m: fused BF16 hi/lo dot products, FP32 accumulation.
- Rationale: increase GPU occupancy for skinny matrices without precision changes; retain the established larger-batch path.
- Local validation: Python syntax passes. Remote result pending; one check requested, no scored submissions.
- Result: PASS, 9/9 pretests and all 30 timed shapes; no static flags. Overall speedup 1.0472675x (95% CI 1.0315585-1.0629765), score 4612.766 us against 4830.8 us SGLang step time.
- Profile: router improves 1.789x / 2.610x at m=1/8 and 1.305-1.339x at m=16/32/128. The projection split-K path regresses most at m=32 (o 0.693x, unpooler 0.719x); restore cuBLAS there before more aggressive tuning.
- Frozen checked source: `attempts/attempt01.py` (byte-for-byte copy). Recommend early scored submission of this file, pending the user's explicit approval.
- Daily budget after this check: 1/10 checks used, 9 left (2 reserved); 0/5 scored runs used.
