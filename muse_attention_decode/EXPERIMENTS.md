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
- Remote check: `906081c5cdb740b1`, source SHA256
  `392f9d853f43b24a20dbc436b5da85fde8410ab2aef8066047851bd90956dc68`.
- Result: PASS, 16/16 tests. Speedup 1.11613745x, 95% CI [1.09939539, 1.13287951]. Score 3374.405 us versus SGLang 3766.3 us. 1/10 daily checks used, 9 remain.
- Scored submissions: none; no submit authorized yet.

- Exact passing snapshot: candidates/attempt01_fused_fa3.py (same source hash).
- Profile: profiles/submission.906081c5cdb740b1.txt and .json.
- Decision: recommend immediate scored submission, subject to user approval.
- Profile finding: full FA3 takes ~189/375 us at batch 16/32; largest sliding verify preparation takes 6 us and gate 3.6 us. Next changes should target these measured costs.
