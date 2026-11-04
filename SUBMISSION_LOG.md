# Scored-run log

Every scored run (`okc submit`), newest last. Overnight standing rule from the user
(2026-11-04 ~05:25 UTC): submit only candidates that pass the verified-better rule
(changed shapes gain as a group above noise, no shape below -3% vs the baseline's run
mean, overall above the board's current scored best) and whose local SHA256 is
unchanged since the check whose server content_hash the scored run must match.

| UTC | Board | File (SHA256) | Check (content_hash) | Scored job | Result | Rank | Approval |
|---|---|---|---|---|---|---|---|
| 2026-11-04 05:27 | muse_decode_layer | candidate_mixed_tiles33.py, attempt 18 (33d2fc1f9f63cd6f) | 7b1b48545b4340f1 (8e716eed11171122), 1.3030x | 5d42151d048d4a03 (server hash 8e716eed11171122, matches) | PASS 1.3023x (CI 1.283-1.322) | 976 (was 1131) | user, explicit |

## Rule checks behind each submission

- decode_layer attempt 18: vs attempt 17 (1.2990x): verify-16 sliding +6.5%, full +2.6%
  (changed group +4.5%); every other shape within -1.7..+0.9% except verify-32 sliding
  -2.9% (code identical to attempt 17 there). Overall 1.3030x > scored 1.1813x.
  Attempt 14 (1.2743x) was never scored: attempt 18 contains it and is better.
