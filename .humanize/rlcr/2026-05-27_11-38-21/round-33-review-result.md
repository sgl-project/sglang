# Round 33 Code Review

Mainline Progress Verdict: STALLED

Round 33 fixed the specific `disable_cuda_graph` under-comparison reproducer and added the requested timing flags, but AC-11 code-tier completeness is still not safe. The new time-window producer path can crash on the intended multi-epoch workload, script-produced artifacts will not pass the new workload cross-check after multiple epochs, and the comparator can still publish a PASS without a DS-on column.

Targeted tests claimed by Claude do pass:

```text
PYTHONPATH=python pytest test/registered/unit/development/test_bench_serving_timing.py -q
7 passed

PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
51 passed, 26 subtests passed
```

Those tests miss the blockers below.

## Goal Alignment Summary

```text
ACs: 13/15 addressed | Forgotten items: 0 | Unjustified deferrals: 1
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.

Partial: AC-1, AC-4, AC-6, AC-8, AC-11, AC-12.

Not met: AC-1b, AC-9, AC-10.

The tracker still covers the original-plan pending tasks. I reject Claude's wording that AC-10 is merely "queued for future rounds": AC-10 is an active original-plan task and blocks the AC-11 H200 sweep because AC-11 depends on radix-cache parity.

## Mainline Gaps

1. **AC-11 time-based `bench_serving` crashes once the measurement window needs more than one epoch.**

Evidence:
- The new loop appends every epoch's outputs to one list (`python/sglang/bench_serving.py:1486-1503`).
- Metrics still receive only the original single-epoch `input_requests` (`python/sglang/bench_serving.py:1560-1562`).
- `calculate_metrics` indexes `input_requests[i]` for every successful output (`python/sglang/bench_serving.py:991-1002`). With `len(outputs) > len(input_requests)`, the second epoch runs off the end.
- I reproduced this with a fake backend: 2 input requests, `measurement_window_seconds=0.01`, no `calculate_metrics` monkeypatch. After repeated epochs, `bench_serving` raised `IndexError: list index out of range`.
- The new timing tests patch `calculate_metrics` (`test/registered/unit/development/test_bench_serving_timing.py:292-299`), so they prove dispatch accumulation but not the real metrics path.

Required fix:
1. Track the measured request rows alongside accumulated outputs. For non-multi-turn runs, pass a repeated measured-input list to `calculate_metrics` whose length matches `outputs`; for multi-turn keep the existing `input_requests=None` behavior.
2. Add a regression that exercises `measurement_window_seconds > 0` without monkeypatching `calculate_metrics`, forces `measured_epochs > 1`, and asserts the JSONL is written without `IndexError`.
3. Assert `completed`, output detail arrays, and duration are internally consistent after repeated epochs.

2. **The new JSONL/sidecar workload cross-check rejects the benchmark scripts' own multi-epoch artifacts.**

Evidence:
- `bench_serving` writes `num_prompts = args.num_prompts * measured_epochs` (`python/sglang/bench_serving.py:1699-1700`).
- `development/benchmark.sh` and `development/benchmark_baseline.sh` still pass sidecar `NUM_PROMPTS="${NUM_PROMPTS}"`, the per-epoch workload size (`development/benchmark.sh:100-112`, `development/benchmark_baseline.sh:98-110`).
- `_bench_meta_writer.py` writes that value directly as sidecar `num_prompts` (`development/_bench_meta_writer.py:95-102`).
- The comparator now requires JSONL `num_prompts` to equal sidecar `num_prompts` (`development/benchmark_compare.py:676-702`).
- Therefore any valid 600s run that needs `measured_epochs > 1` will emit JSONL `num_prompts > sidecar num_prompts` and be refused.

Required fix:
1. Define `num_prompts` consistently as the per-epoch workload shape for the sidecar cross-check. Keep total measured attempts/completions in a separate field such as `measured_num_prompts` or rely on `completed`.
2. Update the comparator to cross-check sidecar `num_prompts` against the per-epoch JSONL workload field, not against total repeated-epoch attempts.
3. Add a regression that builds script-shaped artifacts with `num_prompts=320`, `measured_epochs=2`, `completed=640`, sidecar `num_prompts=320`, and verifies `--ac11` accepts the workload fields when the rest of the operating point matches.

3. **The comparator still does not prove the DS column is actually DS-on.**

Evidence:
- `_normalize_ac11_server_args` drops `enable_double_sparsity` and `double_sparsity_config` before comparison (`development/benchmark_compare.py:621-624`).
- No later validation checks `mode`, `enable_double_sparsity`, or non-empty `double_sparsity_config` for the expected side (`development/benchmark_compare.py:970-1037`).
- Reproducer: I generated three DSA files and three "DS" files whose sidecars all had `mode="native_nsa"` and no DS enablement fields. `benchmark_compare.py --ac11` exited `0` and printed `## AC-11 verdict: PASS`.
- Plan AC-11 is explicitly DS vs DSA and allows only the DS enablement pair to differ (`development/loop4/refined_plan_v1.md:123-125`); dropping those fields from equality is correct only if side identity is separately enforced.

Required fix:
1. Add `_validate_ac11_side_identity(meta, expected_side, path)` and call it for every DSA and DS sidecar in `_run_ac11_mode`.
2. For DSA: require `mode == "native_nsa"` and `server_args.enable_double_sparsity` absent or false; reject any non-empty `double_sparsity_config`.
3. For DS: require `mode == "double_sparsity"`, `server_args.enable_double_sparsity is True`, and non-empty `double_sparsity_config`.
4. Add negative regressions for "both sides native", "DS side missing enable flag", "DS side missing config", and "baseline side has DS enabled".

4. **AC-10 remains incomplete and blocks AC-11 execution.**

Claude lists AC-10 as future work, but the original plan makes `task-ac11-compare` depend on `task-ac10-radix` (`development/loop4/refined_plan_v1.md:262-264`, `development/loop4/refined_plan_v1.md:302-304`). This is not optional if the loop is to reach COMPLETE.

Required implementation plan:
1. Implement or run the M3-B radix-cache hardware fixture against real V3.2 plus the generated mask.
2. Verify cold-prefix vs warm-prefix labels are bit-stable and explicitly check FP8 block scale-factor stability for cold singleton vs packed-block writes.
3. On pass, flip the operator guard (`_double_sparsity_radix_fixture_passed = True`) and remove `--disable-radix-cache` from `development/serve_double_sparsity.sh`.
4. Add/update script-contract tests so DS and DSA launchers both run radix-cache ON once AC-10 passes.
5. Only then run the 3-trial AC-11 H200 DSA+DS sweep and comparator.

## Blocking Side Issues

No separate non-mainline blockers. The blockers are the AC-11/AC-10 mainline gaps above.

## Queued Side Issues

- The Round 33 implementation added plan markers such as "Round 33" and "AC-11" in production comments, print strings, and CLI help in `python/sglang/bench_serving.py` and `development/benchmark_compare.py`, despite `development/loop4/refined_plan_v1.md:361-364` forbidding plan-specific terminology in implementation code. Clean this while touching the same files for the blocking AC-11 fixes; it should not displace the crash/identity fixes.
- Existing queued tracker items remain valid: AC-8 prefix-match helper coverage, stale `deepseek_v2.py` slot-authority comments, and stale `token_label_table.py` lifetime docs.

## Goal Tracker Update

I updated only the mutable section of `goal-tracker.md`:
- Plan Version now says `Updated: Round 33 Review`.
- Added a Round 33 Review evolution row reopening AC-11 code-tier completeness.
- Updated `task-ac11-compare` notes to reflect the three new AC-11 blockers.
- Added blocking side issue rows for the multi-epoch `bench_serving` crash, JSONL/sidecar `num_prompts` mismatch, and missing DS/DSA identity validation.

No immutable Ultimate Goal or Acceptance Criteria text was modified.

NOT COMPLETE
