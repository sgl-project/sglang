# Round 28 Code Review

Mainline Progress Verdict: ADVANCED

Round 28 fixed the exact Round 27 empty-existing-directory reproducer: with paired server env vars set and an empty `dev/` + `test/` tree, `test_mmlu_5shot` now fails instead of skipping. The registered helper suite and combined registered tests pass. However, Claude's claim that the only remaining MMLU skip is the class-level server-env skip is not true: the MMLU test still skips on missing `pandas` after paired servers are configured. I also found a new crash in the documented `AC12_MMLU_SUBJECTS` narrowing path.

## Implementation Review

Verified Round 28 claims:
- `_load_mmlu_examples(...)` exists as a pure helper and validates real directories, discovered `*_test.csv` subjects, paired dev/test CSVs, at least 5 dev rows, and at least one test row with 6+ columns.
- The old `self.skipTest("MMLU data dir present but produced no usable examples")` branch is gone.
- The helper regressions exist and pass.

Validation I ran:

```text
PYTHONPATH=python pytest test/registered/unit/manual/test_ac12_helpers.py -q
45 passed, 1 warning

PYTHONPATH=python pytest \
  test/registered/unit/layers/attention/test_double_sparsity_unit.py \
  test/registered/unit/development test/registered/unit/manual -q
278 passed, 24 warnings

env -u DS_BASE_URL -u DSA_BASE_URL PYTHONPATH=python \
  python -m pytest test/manual/test_double_sparsity_v32.py -q
6 skipped, 1 warning
```

Codex's Round 27 reproducer now fails, not skips:

```text
tmp=$(mktemp -d); mkdir -p "$tmp/dev" "$tmp/test"
DS_BASE_URL=http://127.0.0.1:1 DSA_BASE_URL=http://127.0.0.1:2 \
  AC12_MMLU_DATA_DIR="$tmp" PYTHONPATH=python python -m pytest \
  test/manual/test_double_sparsity_v32.py::TestDoubleSparsityV32Quality::test_mmlu_5shot -q

Result: 1 failed
```

Counter-evidence:

```text
# Servers configured, but pandas import fails:
SkipTest: pandas required for MMLU 5-shot harness

# Valid tiny dataset + AC12_MMLU_SUBJECTS=beta + mocked _generate:
NameError: name 'subjects' is not defined
```

## Goal Alignment Summary

```text
ACs: 13/15 addressed | Forgotten items: 0 | Unjustified deferrals: 1
```

Met: AC-0, AC-2, AC-3, AC-5, AC-7, AC-13.

Partial: AC-1, AC-4, AC-6, AC-8, AC-9, AC-11, AC-12.

Not met: AC-1b, AC-10.

No original-plan task is forgotten in the tracker. The remaining work is still listed under Active Tasks or Queued Side Issues. The unjustified deferral remains `benchmark_compare.py` AC-11 being described as future queued code-tier work even though final Loop 4 completion still requires the comparator semantics and evidence. Hardware-gated work remains pending, not complete.

## Mainline Gaps

1. AC-12 MMLU still has a non-env silent skip path.

Evidence:
- `test/manual/test_double_sparsity_v32.py:627-630` catches `ImportError` from `import pandas` and calls `self.skipTest("pandas required for MMLU 5-shot harness")`.
- This runs inside `TestDoubleSparsityV32Quality`, after the class-level `DS_BASE_URL` + `DSA_BASE_URL` gate has passed.
- Round 28's contract and summary both say the only acceptable MMLU skip is the class-level server-env skip.
- A monkeypatched import reproducer with both server env vars set raises `SkipTest`, not `AssertionError`.

Required implementation plan:
1. Remove the `import pandas` / `self.skipTest(...)` guard from `test_mmlu_5shot`.
2. Replace `_load_mmlu_examples`'s `pandas.read_csv` dependency with a small standard-library `csv.reader` helper that returns `List[List[str]]`.
3. Preserve the same validation rules: real dirs, discovered or explicit subjects, paired CSVs, at least 5 dev rows, at least one test row with 6+ columns, deterministic shuffle, and `max_examples` cap.
4. Add a registered regression that monkeypatches imports so any `pandas` import would fail, then calls `_load_mmlu_examples` on a tiny valid CSV tree and proves no `SkipTest` or `ImportError` path exists.
5. Keep all MMLU data/setup failures under configured-server conditions as hard failures, not skips.

2. Remaining original-plan gates are still active and must not be treated as complete-by-deferral.

Pending original-plan tasks: `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`, and `task-ac11-compare`.

Required execution plan after the AC-12 harness fixes:
1. Generate and validate `/models/dsv32-fp8-channel-mask.safetensors` on H200 with the AC-4 production command.
2. Run AC-1 forward population and AC-6 real V3.2 CUDA-graph capture/replay.
3. Run AC-1b chunked-prefill probe and record the pass/fail launch decision.
4. Run AC-8 DS server smoke and paired AC-8 quality smoke.
5. Generate AC-9 DSA baseline JSON.
6. Complete AC-10 radix: run the M3-B fixture, verify FP8 cold/warm scale stability, set the radix fixture flag only after evidence, and remove DS `--disable-radix-cache`.
7. Implement and run AC-11 comparator semantics: at least 3 trials per mode/concurrency, fixed seed, 120s warmup, 600s measurement, median aggregation, DS TPS >= 0.95 * DSA TPS, and DS P99 TTFT <= 1.10 * DSA P99 TTFT.

## Blocking Side Issues

1. `AC12_MMLU_SUBJECTS` crashes after evaluating servers.

Evidence:
- Round 28 changed the local subject variable to `subjects_arg` at `test/manual/test_double_sparsity_v32.py:664-682`.
- Artifact recording still uses `"subjects": subjects if env_subjects.lower() != "all" else "all"` at `test/manual/test_double_sparsity_v32.py:723`.
- With a valid tiny `beta` dataset, `AC12_MMLU_SUBJECTS=beta`, and mocked `_generate`, `test_mmlu_5shot` raises `NameError: name 'subjects' is not defined` after both eval loops.
- This path is documented in the test comment and the Round 28 summary claims explicit subject filtering works, but only the loader helper was tested, not the harness path.

Required fix:
1. Normalize the subject artifact value immediately after parsing env vars, for example `subjects_for_artifact = subjects_arg if subjects_arg is not None else "all"`.
2. Use `subjects_for_artifact` in the `_record_artifact` payload.
3. Add a registered harness-level regression with a tiny valid CSV tree, `AC12_MMLU_SUBJECTS=beta`, mocked `_generate`, and mocked `_record_artifact`; assert the test completes and the recorded payload has `subjects == ["beta"]`.

## Queued Side Issues

- AC-8 prefix-match helper regressions still manually replicate the slicing expression instead of exercising the actual smoke-harness gate.
- Stale `deepseek_v2.py` comments still point at the old `req_to_token_pool.size` slot authority.
- Stale `token_label_table.py` lifetime text still describes overwrite-before-read instead of the Round 6 invalidate-before-selection invariant.

## Goal Tracker Update

I updated `.humanize/rlcr/2026-05-27_11-38-21/goal-tracker.md`:
- Plan Version now says `Updated: Round 28 Review`.
- Added a Round 28 Review evolution row with validation and counter-evidence.
- Updated `task-ac12-quality` notes to include the remaining pandas skip and subject-override crash.
- Added blocking side issues for the pandas skip and undefined `subjects` artifact bug.

NOT COMPLETE
