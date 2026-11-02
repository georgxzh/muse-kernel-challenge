# Attention decode experiments

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
