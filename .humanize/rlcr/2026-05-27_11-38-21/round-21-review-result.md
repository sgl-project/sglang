# Round 21 Code Review

Mainline Progress Verdict: ADVANCED

Round 21 made real mainline progress: the page-vs-token observability bug is fixed in the live DS publication path, and the missing AC-8 smoke harness file now exists. However, the new smoke harness has a blocking gate bug that can reject a perfectly matching DS/DSA run, and the original Loop 4 plan is still far from complete. Do not treat the remaining hardware, benchmark, and quality work as deferred completion.

## Implementation Review

Verified Round 21 claims:
- `python/sglang/srt/models/deepseek_v2.py:1996-2022` now publishes `selected_tokens` and computes `sparsity_rate = 1 - selected_tokens / seq_len_tokens`; the old page denominator is gone from `_publish_ds_request_summary`.
- Error-path summaries in `deepseek_v2.py:2318-2332` now publish `selected_tokens: 0`.
- `python/sglang/srt/layers/attention/double_sparsity/metrics.py:78-151` renamed the metric keys and per-request field to `selected_tokens`.
- `python/sglang/srt/layers/attention/double_sparsity/selection_kernel.py:579-586` calls `record_selection(selected_tokens=..., total_valid_tokens=...)`.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py:4983-5024` covers the token denominator regression.
- `test/manual/test_dsv32_quality_smoke.py` exists and skips cleanly when `DS_BASE_URL` / `DSA_BASE_URL` are unset.

Validation run:
```text
python -m pytest test/manual/test_dsv32_quality_smoke.py -q
1 skipped

PYTHONPATH=python pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py -k 'test_publish_ds_request_summary_uses_token_denominator or test_meta_info_shape or test_real_selector_records_metrics'
3 passed, 199 deselected

PYTHONPATH=python pytest -q test/registered/unit/layers/attention/test_double_sparsity_unit.py
202 passed
```

## Goal Alignment Summary

```text
ACs: 10/15 addressed | Forgotten items: 0 | Unjustified deferrals: 0
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.
Partial: AC-1, AC-4, AC-6, AC-8.
Not met: AC-1b, AC-9, AC-10, AC-11, AC-12.

No original-plan task is missing from Active, Completed, or Deferred after tracker reconciliation. `Explicitly Deferred` remains empty, which is correct. Hardware-gated work is still active/pending, not accepted as deferred. AC-12 remains hard and still blocks loop completion.

## Findings By Lane

### Mainline Gaps

1. The Loop 4 MVP remains incomplete because the remaining original-plan gates have not run.

Round 21 did not complete AC-8; it created the smoke file and fixed a prerequisite observability issue. The following plan tasks are still active and must be executed, not deferred:

1. Fix the AC-8 smoke harness gate bug listed under Blocking Side Issues below.
2. Generate and validate `/models/dsv32-fp8-channel-mask.safetensors` on H200:
   ```bash
   python -m sglang.srt.layers.attention.double_sparsity.calibrate \
     --model /cluster-storage/models/deepseek-ai/DeepSeek-V3.2 \
     --dtype bfloat16 \
     --kv-cache-dtype fp8_e4m3 \
     --output /models/dsv32-fp8-channel-mask.safetensors \
     --label-dim 16 \
     --page-size 64
   ```
3. Run `task-ac1-hwtest`: real H200 `forward_extend`, then assert each `token_label_table.signatures[layer_id, out_cache_loc]` row is non-zero.
4. Run `task-ac6-hwrun`: Option B V3.2 conc=64 full-graph capture, 100 replays, no CUDA launch failure, eager/graph `max_abs_diff <= 1e-6`.
5. Run `task-ac1b-probe`: `chunked_prefill_size=4096`, compare labels for tokens 0..4095 against the non-chunked baseline, and record the pass/fail decision.
6. Run AC-8 server smoke: DSv3.2 FP8 TP=8, locked Option B flags, `bench_serving` conc 16/32/64, ≥64 requests, ISL about 4096, non-trivial `selected_tokens` on ≥90% of decode steps, and dense fallback accounting clean.
7. Run the corrected AC-8 quality smoke against paired DSA/DS servers.
8. Complete AC-12 full NIAH/MMLU quality gate before declaring the loop done.
9. Complete stretch AC-9, AC-10, and AC-11 rather than treating them as complete-by-deferral.

2. AC-12 is still a skip-only scaffold.

Evidence:
- `test/manual/test_double_sparsity_v32.py:62-93` still calls `self.skipTest(...)` for NIAH 4K, NIAH 16K, NIAH 64K, MMLU 5-shot, and both negative sensitivity checks.
- The plan states AC-12 is hard: the loop does not close without NIAH deltas ≤ 5 pp and MMLU delta ≤ 1 pp.

Required implementation plan:
1. Replace the scaffold with a paired-server harness that requires `DS_BASE_URL` and `DSA_BASE_URL`.
2. Implement NIAH datasets at 4K, 16K, and 64K with deterministic seeds and exact-match scoring.
3. Implement MMLU 5-shot evaluation using the repo’s existing evaluation utilities where possible; persist per-subject and aggregate scores.
4. Fail the test when `DSA - DS > 5 pp` for any NIAH length or `DSA - DS > 1.0 pp` for MMLU.
5. Keep the corrupt-mask and zero-signature negative checks as executable fault-injection tests, not skips.

3. AC-8/AC-9/AC-11 run tooling is still not aligned with the locked operating point.

Evidence:
- `development/benchmark_compare.py:21-25` still documents an older SLO/no-op contract around `selected_pages == total_pages`.
- `development/benchmark_compare.py:61-63`, `211-213`, `274-283`, and `315-316` still consume and report `selected_pages_mean` / `total_pages_mean`, which now conflicts with Round 21’s `selected_tokens` rename.
- Prior Round 20 review evidence still applies: launch scripts and benchmark scripts need locked Option B flags and conc 16/32/64 defaults before hardware evidence is valid.

Required implementation plan:
1. Update `development/serve_double_sparsity.sh` and `development/serve_native_nsa.sh` to encode the locked Option B flags: `--kv-cache-dtype fp8_e4m3`, `--dsa-prefill-backend flashmla_kv`, `--dsa-decode-backend flashmla_kv`, `--disable-overlap-schedule`, `--disable-piecewise-cuda-graph`, `--page-size 64`, and matching chunked-prefill choice after AC-1b.
2. Keep `--disable-radix-cache` only on the DS AC-8 path until AC-10 passes.
3. Make benchmark scripts run conc 16/32/64 by default and persist commit/version, server args, seed, warmup, measurement window, and chunked-prefill setting.
4. Update `benchmark_compare.py` to use `selected_tokens_mean` / `total_tokens_mean`, consume three trials per concurrency, take medians, enforce DS TPS within 5% of DSA and P99 TTFT ≤ 1.10x DSA, and remove the old page-named no-op logic.

### Blocking Side Issues

1. `test_dsv32_quality_smoke.py` can fail a perfect paired DS/DSA run because short exact matches are counted as prefix failures.

Evidence:
- `test/manual/test_dsv32_quality_smoke.py:280-285` counts a prefix-match hit only if `ds[:32] == dsa[:32]` and `len(dsa) >= 32`.
- Most of the 20 smoke prompts intentionally request short answers; 12 of them include explicit short-output instructions such as "Output only", "Give just", or "Output just". Exact DS/DSA matches like `Au`, `1969`, `Jupiter`, or `1024` will be counted as prefix-match misses solely because the DSA answer is shorter than 32 chars.
- The AC-8 gate is "first 32 chars match", not "the DSA answer must be at least 32 chars". A shorter identical output should count as a prefix match.

Required fix:
1. Change the prefix hit condition to count exact short outputs, for example:
   ```python
   ds[:PREFIX_MATCH_CHARS] == dsa[:PREFIX_MATCH_CHARS]
   ```
   without the `len(dsa) >= PREFIX_MATCH_CHARS` guard.
2. Add helper-level regressions proving an exact short output counts as a prefix hit and a genuinely different short output does not.

2. `_first_n_tokens_match` claims "any overlap" but only checks same-position equality.

Evidence:
- `test/manual/test_dsv32_quality_smoke.py:208-215` says "Any overlap at all means they're not entirely different", but the implementation uses `zip(a_toks, b_toks)` and therefore only detects overlap at the same token positions.
- A shifted overlap such as `alpha beta gamma` vs `beta gamma alpha` returns `False` even though the first 3-token windows share every token.

Required fix:
1. Implement the intended gate as set intersection over the first `n` normalized tokens:
   ```python
   return bool(set(a_toks) & set(b_toks))
   ```
2. Add a regression for shifted overlap and one for no overlap.

### Queued Side Issues

- `deepseek_v2.py` still has stale comments/docstrings saying `max_tokens = req_to_token_pool.size` even though Round 2 corrected the authority to the physical KV slot address space.
- `token_label_table.py` still says reused slots are safe because the write hook overwrites before read, but Round 6 changed the invariant to invalidate before selection and rewrite later.

These should stay queued behind AC-8 harness correctness, hardware gates, AC-12, and Option B run tooling.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Plan Version now says `Updated: Round 21 Review`.
- Added a Round 21 Review plan-evolution row for the verified token-denominator fix and the smoke-harness gate bug.
- Updated `task-ac8-quality` notes to require fixing the short-output prefix gate and first-8 overlap semantics before the paired H200 smoke run.
- Added the smoke-harness gate bug to Blocking Side Issues.
- Removed the old observability page-vs-token item from Queued Side Issues because Round 21 fixed that specific issue.

NOT COMPLETE
