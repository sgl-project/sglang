# Round 31 Code Review

Mainline Progress Verdict: ADVANCED

Round 31 fixed the two exact Round 30 reproducers: missing sidecars now refuse instead of publishing a mismatched PASS, and malformed `*_c64.jsonl` inputs now exit 2 without traceback. The targeted suites pass:

```text
PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
38 passed

PYTHONPATH=python pytest test/registered/unit/development/test_option_b_scripts.py -q
16 passed

PYTHONPATH=python pytest test/registered/unit/development -q
65 passed

bash -n development/benchmark.sh
bash -n development/benchmark_baseline.sh
OK
```

AC-11 is still not acceptance-ready. The current implementation can still publish a PASS without actual 120s/600s evidence, can accept sidecars missing required reproducibility fields, and deliberately allows the radix-cache mismatch that AC-11 is supposed to run only after AC-10 removes.

## Goal Alignment Summary

```text
ACs: 13/15 addressed | Forgotten items: 0 | Unjustified deferrals: 2
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.

Partial: AC-1, AC-4, AC-6, AC-8, AC-11, AC-12.

Not met: AC-1b, AC-9, AC-10.

The tracker still covers the remaining original-plan tasks. The unjustified deferrals are: time-based AC-11 warmup/measurement enforcement was explicitly pushed out of Round 31, and AC-11 currently permits DS-vs-DSA radix mismatch even though the refined plan makes AC-11 depend on AC-10.

## Mainline Gaps

1. AC-11 timing floors are metadata-only; the scripts do not run or enforce the required 120s warmup and 600s measurement window.

Evidence:
- Plan §AC-11 requires a “minimum 600s measurement window after a 120s warmup” (`development/loop4/refined_plan_v1.md:123-125`).
- `development/benchmark.sh:45-47` and `development/benchmark_baseline.sh:47-49` only set `WARMUP_SECONDS=120` and `MEASUREMENT_WINDOW_S=600`.
- The measured `bench_serving` command in `development/benchmark.sh:63-77` has no timing or warmup flag beyond bench_serving’s default one-request warmup; same in `benchmark_baseline.sh:63-77`.
- `_bench_meta_writer.py:106-108` writes those env values to the sidecar, but `benchmark_compare.py:522-536` validates only the sidecar values. `_read_bench_jsonl` does not parse or check JSONL `duration`.
- Reproducer: three DSA + three DS JSONLs with `"duration": 5` and sidecars claiming `warmup_seconds=120` / `measurement_window_seconds=600` return exit 0 and `AC-11 verdict: PASS`.

Required implementation plan:
1. Add `--warmup-seconds` and `--measurement-window-seconds` to `python/sglang/bench_serving.py`.
2. Implement a warmup phase that uses the same dataset shape, seed family, backend, max concurrency, and request scheduling, discards its outputs, and runs until wall time is at least `warmup_seconds`.
3. Implement measured execution so the result row’s actual `duration` is at least `measurement_window_seconds`; repeat deterministic dataset epochs under the same request generator until the duration floor is reached.
4. Have `benchmark.sh` and `benchmark_baseline.sh` pass these CLI flags, write the actual measured duration into the JSONL, and fail immediately if the final row duration is below `MEASUREMENT_WINDOW_S`.
5. Extend `RunMetrics` with `duration_s` and make `benchmark_compare.py --ac11` refuse if any trial duration is below 600s or below the sidecar window.
6. Add regressions for: short JSONL duration + valid sidecar refuses; scripts pass the new CLI flags; sidecar timing and JSONL duration disagreement refuses.

2. AC-11 sidecar validation treats missing required fields as “agreement.”

Evidence:
- `_validate_per_side_agreement` and `_validate_cross_side_agreement` use `m.get(field) != first.get(field)` / `dsa_meta.get(field) != ds_meta.get(field)` (`development/benchmark_compare.py:546-575`). If both sides omit `seed`, `commit_sha`, `chunked_prefill_size`, or workload fields, `None == None` passes.
- `_normalize_ac11_server_args` returns `{}` for missing or non-dict `server_args` (`development/benchmark_compare.py:497-505`), so both sidecars can omit the full server args and still compare equal.
- Reproducer: sidecars containing only timing floors plus `server_args_error=null`, with `seed`, `commit_sha`, `chunked_prefill_size`, workload fields, and `server_args` removed, return exit 0 and `AC-11 verdict: PASS`.

Required implementation plan:
1. Add a required-field validator called from `_read_ac11_meta` before returning metadata.
2. Require `seed` to be an int, `commit_sha` to be a non-empty non-`unknown` string, `chunked_prefill_size` to be present, `num_prompts` / `isl_total_tokens` / `osl_tokens` to be positive ints, and `server_args` to be a non-empty object.
3. Require the sidecar `concurrency` to match the JSONL concurrency group.
4. Where JSONL contains workload fields, require sidecar workload fields to agree with the JSONL. Add the missing workload fields to `bench_serving` JSONL output for the generated-shared-prefix path so this check is not best-effort.
5. Add one registered negative regression for each missing required field and for non-object/empty `server_args`.

3. AC-11 still permits the DS/DSA radix-cache mismatch.

Evidence:
- Refined plan §AC-11 says only `--enable-double-sparsity` and `--double-sparsity-config` differ between columns (`development/loop4/refined_plan_v1.md:123-125`).
- The task table makes `task-ac11-compare` depend on `task-ac10-radix` (`development/loop4/refined_plan_v1.md:303-304`), and AC-10 requires removing `--disable-radix-cache` from the DS launcher (`development/loop4/refined_plan_v1.md:120-121`).
- Round 31 adds `"disable_radix_cache"` to `_DS_ONLY_SERVER_ARG_KEYS` (`development/benchmark_compare.py:445-449`) and then drops radix mismatch reasons from `_match_or_refuse` (`development/benchmark_compare.py:810-816`).
- The registered test `test_allowed_ds_only_differences_still_pass` now locks in the wrong contract by asserting DS radix off vs DSA radix on returns exit 0 (`test/registered/unit/development/test_ac11_comparator.py:535-543`).

Required implementation plan:
1. Remove `disable_radix_cache` from `_DS_ONLY_SERVER_ARG_KEYS`.
2. Stop filtering `disable_radix_cache` mismatch reasons after `_match_or_refuse`.
3. Change the current allowed-difference regression to expect exit 2 for radix mismatch, and add a passing test where both DS and DSA have radix parity after DS-only enablement keys are normalized.
4. Complete AC-10 before AC-11 hardware execution: run the M3-B fixture, verify FP8 cold/warm scale stability, set the operator flag only after evidence, and remove `--disable-radix-cache` from `serve_double_sparsity.sh`.

## Blocking Side Issues

1. Raw `/get_server_info` comparison can make script-generated trial sets self-refuse.

Evidence:
- `/server_info` returns server args plus `internal_states` (`python/sglang/srt/entrypoints/http_server.py:641-645`).
- Scheduler internal state includes dynamic values such as `last_gen_throughput`, memory usage, optional spec metrics, and step-time dictionaries (`python/sglang/srt/managers/scheduler.py:3320-3343`).
- `_normalize_ac11_server_args` compares the raw sidecar `server_args` object after dropping only DS keys (`development/benchmark_compare.py:490-505`), and `_validate_per_side_agreement` requires that normalized object to match across trials (`development/benchmark_compare.py:554-558`).
- Reproducer: sidecars that differ only in `server_args.internal_states[0].last_gen_throughput` return exit 2 with “normalized server_args disagrees.” This can happen naturally because the scripts call `/get_server_info` before every sequential trial.

Required fix:
1. In `_bench_meta_writer.py`, store the raw endpoint payload separately as `server_info_raw` if needed for audit, but store `server_args` as a stable launch-argument subset.
2. In `_normalize_ac11_server_args`, strip non-launch fields from existing sidecars: `internal_states`, `kv_events`, runtime metric dictionaries, profiler/step-time fields, and any other non-argument endpoint fields.
3. Keep comparing stable launch fields that matter for AC-11: TP size, page size, backend flags, dtype, model path, chunked prefill, scheduler flags, CUDA graph flags, and DS enablement/config after DS-only normalization.
4. Add a regression where only `internal_states.last_gen_throughput` differs across trials and the comparator still accepts.

## Queued Side Issues

- `_filename_concurrency` still only recognizes `_c<N>.jsonl`, not the new `_c<N>_t<M>.jsonl` filenames. This is non-blocking for current script output because `bench_serving` writes `max_concurrency`, but the fallback and its regression should be updated.
- AC-8 prefix-match helper regressions still manually replicate the slicing expression instead of exercising the actual smoke-harness gate.
- `deepseek_v2.py` still has stale slot-authority comments.
- `token_label_table.py` still has stale lifetime docs.

## Goal Tracker Update

I updated only the mutable section of `goal-tracker.md`:
- Plan Version now says `Updated: Round 31 Review`.
- Added a Round 31 Review evolution row reopening AC-11 code-tier completeness.
- Updated `task-ac11-compare` notes to require timing enforcement, required sidecar validation, stable server-args normalization, and radix parity.
- Added four AC-11 blocking side issues for metadata-only timing, missing required sidecar fields, radix mismatch allowance, and raw dynamic `/get_server_info` comparison.

No immutable Ultimate Goal or Acceptance Criteria text was modified.

NOT COMPLETE
