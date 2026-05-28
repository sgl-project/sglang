# Round 4 Review Result

Mainline Progress Verdict: ADVANCED

Round 4 materially advanced AC-1. The prior blocking gap was production call-site coverage, and the new `TestAC1CallSites` class now exercises the three `_write_token_labels` call sites with `save_kv_cache=True`: `forward_extend`, `forward_decode`, and `_forward_trtllm`. I verified the hook sites are still present at `python/sglang/srt/layers/attention/dsa_backend.py:1510`, `:1709`, and `:2233`, and the local suite passes:

```text
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
160 passed, 24 warnings in 11.58s
```

AC-1 is still partial, not met. The plan still requires the H200 real `forward_extend` population test, and AC-1's integration smoke is tied to AC-8 `bench_serving` not crashing on selector reads.

## Part 1: Goal Tracker Audit

### Acceptance Criteria Status

| AC | Status | Evidence / Blocker |
|----|--------|--------------------|
| AC-0 | MET | Public import succeeds; `page_table_adapter.py` is 71 LOC; `TestAC0RealSlotRegression` covers non-contiguous physical slots and the corrected physical KV slot sizing. Note: the literal immutable text saying `req_to_token_pool.size` was superseded by Round 2 because it contradicted slot-indexing by `out_cache_loc`; current authority is `kv_pool.size + kv_pool.page_size`. |
| AC-1 | PARTIAL | Unit call-site coverage is now verified: tests at `test/registered/unit/layers/attention/test_double_sparsity_unit.py:4601` cover extend, decode, TRT-LLM FP8, and `save_kv_cache=False`. Blockers: `task-ac1-hwtest` on H200 and AC-8 selector-read smoke. |
| AC-1b | NOT MET | Chunked-prefill probe has not run. |
| AC-2 | NOT MET | Lifetime/stale-slot work is pending. No boot-time GB/rank log or stale-slot negative test evidence yet. |
| AC-3 | NOT MET | Per-request ownership mask and boundary negative test are pending. |
| AC-4 | NOT MET | Method 1 Q+K calibration code and H200 mask generation are pending. Design doc `development/past_implementations/study/07-mvp-proposed-architecture.md` sections 9-10 keep real calibration and quality smoke in MVP scope. |
| AC-5 | NOT MET | TP=2 multiprocess harness is pending. |
| AC-6 | NOT MET | CUDA graph capture work and H200 replay are pending; queued graph helper issue remains. |
| AC-7 | NOT MET | Short-seq MHA bypass verification is pending. |
| AC-8 | NOT MET | 8xH200 `bench_serving` and lightweight quality smoke are pending. |
| AC-9 | NOT MET | Stretch DSA baseline JSON is pending, not deferred. |
| AC-10 | NOT MET | Stretch radix-cache fixture and config flip are pending, not deferred. |
| AC-11 | NOT MET | Stretch comparator row is pending, not deferred. |
| AC-12 | NOT MET | Hard NIAH/MMLU quality gate is pending. The loop cannot close without this. |
| AC-13 | MET | Regression suite is green after shape migration. Current suite has grown to 160 tests and passes. |

### Forgotten Items Detection

No actionable forgotten plan item found after tracker correction. Remaining original tasks are in Active, Completed and Verified, or represented by queued side issues with explicit revisit triggers. The AC-0 task granularity is collapsed under verified AC-0 rows from earlier reviews; the stale `capture_decode_step` work is tracked as queued until `task-ac6-cuda-graph`.

One tracker drift item was corrected: `task-m1-hook` was duplicated between Active and Completed with `pending-codex`. I moved it out of Active and marked it verified in Round 4. I also added a queued cleanup for stale `deepseek_v2.py` comments that still say `max_tokens = req_to_token_pool.size`.

### Deferred Items Audit

There are no explicitly deferred items in the tracker. AC-9 through AC-11 are stretch tasks but still active/pending, not accepted deferrals. No deferral currently contradicts the Ultimate Goal.

### Goal Completion Summary

```text
Acceptance Criteria: 2/15 met (0 deferred, 1 partial)
Active Tasks: 17 remaining
Estimated remaining rounds: at least 8-10, hardware-window dependent
Critical blockers: H200 access for AC-1/AC-4/AC-6/AC-8/AC-12 hardware gates; generated V3.2 channel mask not available yet
```

## Part 2: Mainline Drift Audit

The current round's objective was clear and singular: close the AC-1 call-site verification gap from Round 3. Claude advanced the mainline by adding targeted production-path tests rather than clearing unrelated side issues.

The repeated AC-1 reopening over Rounds 2-4 is not stagnation because each review found a different concrete blocker and the next round fixed it: Round 2 fixed slot authority and wired hooks, Round 3 fixed projection/input correctness, and Round 4 added call-site tests.

```text
Mainline Progress Verdict: ADVANCED
Blocking Side Issues: 0
Queued Side Issues: 3
```

## Part 3: Implementation Review

No high-signal correctness issue found in the Round 4 test implementation.

Verified claims:
- `test_forward_extend_writes_token_labels` calls `forward_extend` with `save_kv_cache=True` and asserts writes at slots 5 and 10.
- `test_forward_decode_writes_token_labels` calls `forward_decode` with `save_kv_cache=True` and asserts writes at slots 7 and 15.
- `test_trtllm_hook_receives_pre_quantized_k` calls `_forward_trtllm`, patches FP8 quantization, spies on `_write_token_labels`, and verifies the hook receives the original float latent K rather than FP8 output.
- `test_no_labels_when_save_kv_cache_false` proves the hook is inside the KV-write guard.

Important limit: these are CPU/unit call-site tests with patched attention kernels. They prove the Python production calls are wired and guarded, but they do not replace `task-ac1-hwtest` against real V3.2/H200 execution.

Queued implementation cleanup:
- `python/sglang/srt/models/deepseek_v2.py:1541-1544` and `:1836-1840` still contain stale text saying table sizing comes from `req_to_token_pool.size`. The actual code at `:1906-1917` correctly uses physical KV slot capacity, so this does not reopen AC-0. It should be cleaned when DS bind/runtime comments are next touched.

## Part 4: Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Added a Round 4 Review plan-evolution row.
- Moved `task-m1-hook` from Active to Completed and Verified with Verified Round `4`.
- Left `task-ac1-hwtest` active, so AC-1 remains partial.
- Added the stale `deepseek_v2.py` sizing-comments issue to Queued Side Issues.

## Part 5: Progress Stagnation Check

No stagnation trigger. The loop had one regression in Round 0, then steady mainline recovery and advancement in Rounds 1-4. Feedback has not been circular: each repeated AC-1 review item was narrowed and addressed in the next round. The loop is incomplete, but not stalled.

## Action Items

### Mainline Gaps

1. Run `task-ac1-hwtest` on H200: real V3.2 `forward_extend`, assert `token_label_table.signatures[layer_id, out_cache_loc]` is non-zero for each written slot.
2. Continue dependency order after AC-1 hardware evidence: AC-2 lifetime/stale-slot, AC-3 range mask and boundary test, AC-7 short-seq bypass, AC-4 Method 1 Q+K calibration and H200 mask generation, AC-5 TP harness, AC-6 graph capture, AC-1b probe, AC-8 bench/quality smoke, AC-12 full quality.

### Blocking Side Issues

None verified in this round.

### Queued Side Issues

1. Before `task-ac6-cuda-graph`, update `capture_decode_step` to use the logical-domain selector path with `req_to_token`.
2. Before AC-8 server/quality smoke, fix DS observability metrics that still report token selections through page-named fields and page-count denominators.
3. Clean stale `deepseek_v2.py` comments/docstrings that still point to the old `req_to_token_pool.size` sizing authority.

NOT COMPLETE
