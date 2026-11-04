# Scored-run log

Every scored run (`okc submit`), newest last. Overnight standing rule from the user
(2026-11-04 ~05:25 UTC): submit only candidates that pass the verified-better rule
(changed shapes gain as a group above noise, no shape below -3% vs the baseline's run
mean, overall above the board's current scored best) and whose local SHA256 is
unchanged since the check whose server content_hash the scored run must match.
Standing waiver from the user (2026-11-04 ~14:55 UTC): if the only shapes that miss the
rule run unchanged code and their own kernel time is unchanged, treat it as noise,
submit without asking, and log it here.

| UTC | Board | File (SHA256) | Check (content_hash) | Scored job | Result | Rank | Approval |
|---|---|---|---|---|---|---|---|
| 2026-11-04 05:27 | muse_decode_layer | candidate_mixed_tiles33.py, attempt 18 (33d2fc1f9f63cd6f) | 7b1b48545b4340f1 (8e716eed11171122), 1.3030x | 5d42151d048d4a03 (server hash 8e716eed11171122, matches) | PASS 1.3023x (CI 1.283-1.322) | 976 (was 1131) | user, explicit |
| 2026-11-04 06:50 | muse_decode_layer | candidate_cuda_full_decode.py, attempt 20 (8b3b01ee8948298d) | 514225bb47ed4939 (67184bed0c55de5a), 1.3230x | 444f66ab63174de2 (server hash 67184bed0c55de5a, matches) | PASS 1.3266x | 974 (was 976) | autonomous (overnight rule) |
| 2026-11-04 14:56 | muse_attention_decode | candidates/attempt39_cuda_full_decode_b8_remeasure.py (42d97020e63fa40f) | 5e6de4aaeb4f4737 (3cb1d128d24d611c), 1.5415x | 7d6c77f92723446c (server hash 3cb1d128d24d611c, matches) | PASS 1.5464x (CI 1.523-1.570) | 894 (was 997) | user, explicit (waiver) |

## Rule checks behind each submission

- decode_layer attempt 18: vs attempt 17 (1.2990x): verify-16 sliding +6.5%, full +2.6%
  (changed group +4.5%); every other shape within -1.7..+0.9% except verify-32 sliding
  -2.9% (code identical to attempt 17 there). Overall 1.3030x > scored 1.1813x.
  Attempt 14 (1.2743x) was never scored: attempt 18 contains it and is better.
- decode_layer attempt 20: vs attempt 18's two runs: changed shapes (full decode-16/32,
  the only ones using the new kernel) +11.0% / +14.2% (+12.6% as a group); unchanged
  shapes -0.1% on average, worst verify-16 sliding -1.8% (> -3%). Overall 1.3230x >
  scored 1.3023x. Local SHA 8b3b01ee... equals the file checked as 514225bb47ed4939.
- attention_decode attempt 39 (attempt 37 re-measured): full decode-16/32 +28.2/+28.1%
  (CUDA whole-row kernel, B >= 8); overall 1.5415x > scored 1.4425x. decode-1 sliding
  -4.2% on code identical to attempt 17 (time 8.4 us, unchanged; ratio vs attempt 17's
  mean +0.7..-4.2% across 7 checks): waived by the user explicitly.
