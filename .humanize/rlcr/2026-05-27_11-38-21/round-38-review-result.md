# Round 38 Code Review

Mainline Progress Verdict: STALLED

Round 38 fixed part of the Round 37 drift: the verdict helper now accepts the dict-shaped `meta_info` value, the cached-prefix comparison is per-position, and the FP8 fixture now has a fallback branch. However, the recovered mainline objective was producer→transport→consumer evidence. The producer side still does not publish the response-routed capture at all, so AC-10 remains code-tier incomplete.

## Goal Alignment Summary

```text
ACs: 13/15 addressed (6 met, 7 partial, 2 not met) | Forgotten items: 0 | Unjustified deferrals: 1
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.

Partial: AC-1, AC-4, AC-6, AC-8, AC-10, AC-11, AC-12.

Not met: AC-1b, AC-9.

The tracker still covers the original-plan pending work in Active Tasks, but it had drifted by marking the Round 38 AC-10 code tier as complete and the AC-10 blocking issue as resolved. I updated only the mutable section: Plan Version now says Round 38 Review, the plan log reopens AC-10 code-tier completeness, `task-ac10-radix` now records the producer-side gap, and the AC-10 blocking issue is open again.

## Mainline Gaps

1. **The new post-write capture publisher is unreachable in production.**

Evidence:
- `_ds_radix_publish_extend_snapshot(...)` is the only new producer that writes `summary["double_sparsity_radix_capture"]` for the response transport (`python/sglang/srt/layers/attention/dsa_backend.py:322-369`).
- `_write_token_labels` still has signature `(self, layer, cache_loc, k)` with no `forward_batch` parameter (`python/sglang/srt/layers/attention/dsa_backend.py:1501-1506`).
- The capture branch references `forward_batch` anyway (`python/sglang/srt/layers/attention/dsa_backend.py:1581-1593`). Because that lookup sits inside `except Exception`, the resulting `NameError` is swallowed, `is_extend` becomes `False`, and `_ds_radix_publish_extend_snapshot(...)` is never called.
- All three production hook sites still call `_write_token_labels(layer, cache_loc, k...)` without `forward_batch` (`python/sglang/srt/layers/attention/dsa_backend.py:1664`, `:1863`, `:2387`).
- A capture-enabled smoke reproducer with a spy on `_ds_radix_publish_extend_snapshot` produced `{'publish_calls': 0, 'write_records': 1, 'written': True}`. The old module write log records, but nothing is published into `forward_batch.ds_per_request_summary`, so `/generate` still lacks `meta_info["double_sparsity_radix_capture"]`.

Impact:
- The Round 38 transport regression starts from a synthetic `producer_summary={"double_sparsity_radix_capture": [record]}`. It does not prove the real producer can create that summary.
- The manual H200 label-capture fixture will fail as missing evidence, or worse, future edits may keep testing only helper-level shims while the server response remains empty.
- AC-10 cannot be closed and AC-11 cannot run honestly.

Required implementation plan:
1. Change `_write_token_labels` to accept `forward_batch: Optional[ForwardBatch] = None`.
2. Pass the live `forward_batch` from all three production call sites: forward extend, forward decode, and TRT-LLM MLA.
3. Keep `token_label_write(...)` first, then under `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` call `_ds_radix_publish_extend_snapshot(...)` only when `forward_batch` is present and `forward_batch.forward_mode.is_extend()` is true.
4. Do not hide a missing `forward_batch` behind the same broad best-effort exception. The best-effort catch belongs inside `_ds_radix_publish_extend_snapshot` for snapshot failures, not around the producer's required context lookup.
5. Add a registered producer-side regression: enable `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, call the real `_write_token_labels` path with a minimal extend `forward_batch` carrying `req_to_token_pool.req_to_token`, `req_pool_indices`, `seq_lens`, and `forward_mode`, then assert `forward_batch.ds_per_request_summary["double_sparsity_radix_capture"]` is a non-empty per-batch list. The test must fail if the call sites stop passing `forward_batch`.
6. Add a decode-mode regression proving decode writes do not overwrite an existing extend snapshot.
7. Keep the Round 38 verdict/transport tests; they are useful consumer-side coverage, but they are not a substitute for the producer-side regression.

2. **The rest of AC-10 remains unfinished original-plan work.**

This must not be done before the producer bug above is fixed and the corrected fixtures pass, but it is still part of AC-10:

1. Run `test/manual/test_dsv32_fp8_scale_stability.py` on H200 with `SGLANG_DS_FP8_SCALE_PROOF=1` and preserve the artifact.
2. Boot the DS server with radix cache on and `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, then run `test/manual/test_dsv32_radix_label_capture_fixture.py`; verify warm `cached_tokens > 0` and verdict PASS.
3. Wire `record_radix_fixture_passed(server_args, artifact_path=...)` into launcher/server-args initialization before `validate_double_sparsity`.
4. Remove `--disable-radix-cache` from `development/serve_double_sparsity.sh`.
5. Update `test_ds_server_does_disable_radix_cache_until_ac10` to the post-AC-10 expectation.
6. Only then unblock the AC-11 H200 comparator sweep.

## Blocking Side Issues

None separate from the AC-10 mainline gaps. The unreachable producer-side capture directly blocks AC-10 and therefore blocks AC-11.

## Queued Side Issues

- Preserve the existing queued cleanup items: AC-8 prefix-match helper regression coverage, stale `deepseek_v2.py` slot-authority comments, and stale `token_label_table.py` lifetime docs.
- Do not let those cleanup items displace the AC-10 producer fix and H200 fixture execution.

## Validation

Local checks run:

```text
PYTHONPATH=python pytest test/registered/unit/manual/test_m3b_label_capture_verdict.py -q
13 passed

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -k TestBuildRequestCapture -q
10 passed, 227 deselected
```

I also ran a capture-enabled smoke reproducer against `_write_token_labels`; it proved the real producer writes the old capture log but never calls `_ds_radix_publish_extend_snapshot` (`publish_calls=0`). I did not run H200 hardware gates.

## Tracker Update

I updated only the mutable section of `goal-tracker.md`:
- Plan Version: `Updated: Round 38 Review`.
- Added a Round 38 Review Plan Evolution row reopening AC-10 code-tier completeness.
- Rewrote `task-ac10-radix` active-task notes to reflect the missing producer-side publish.
- Reopened the AC-10 blocking side issue instead of leaving it marked resolved.

No immutable Ultimate Goal or Acceptance Criteria text was modified.

NOT COMPLETE
