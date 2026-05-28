# Round 29 Summary

## Work Completed

Codex Round 28 review caught two more AC-12 harness defects:

1. `test_mmlu_5shot` opened with
   `try: import pandas as pd except ImportError: self.skipTest(...)`.
   That runs INSIDE the test, after the class-level `skipUnless` has
   already passed — so when servers are configured but pandas isn't
   installed, the gate silently skips. Round 28's contract had
   explicitly promised "the only acceptable MMLU skip is the
   class-level server-env skip", but the pandas guard slipped through.
2. Round 28 renamed `subjects → subjects_arg` but left
   `"subjects": subjects` in the `_record_artifact` payload, so
   `AC12_MMLU_SUBJECTS=beta` crashed with
   `NameError: name 'subjects' is not defined` AFTER both eval
   loops. The Round 28 registered regression tested
   `_load_mmlu_examples` directly with the `subjects=["beta"]` kwarg
   but never exercised the full harness path.

### Fix 1 — drop pandas; use stdlib csv

`test/manual/test_double_sparsity_v32.py`:

- Removed the `try: import pandas / self.skipTest(...)` guard from
  `test_mmlu_5shot`. The harness now has no non-env silent-skip
  paths.
- `_load_mmlu_examples` no longer imports pandas. Added an inline
  `_read_csv_rows(path)` helper using `csv.reader(open(path,
  newline=""))` that returns `List[List[str]]`. Drop-in replacement
  for the prior `pd.read_csv(...).values.tolist()` semantics.

### Fix 2 — `subjects` NameError

Defined `subjects_for_artifact = subjects_arg if subjects_arg is
not None else "all"` immediately after env parsing. The
`_record_artifact` payload now uses `subjects_for_artifact`. The
undefined `subjects` reference is gone.

### Fix 3 — Registered regressions (+2)

`test/registered/unit/manual/test_ac12_helpers.py`:

- `test_load_mmlu_examples_works_without_pandas`:
  - Pops `sys.modules["pandas"]` if present.
  - Monkeypatches `builtins.__import__` to raise
    `ImportError("simulated absence of pandas")` for any `pandas`
    import.
  - Builds a tiny valid CSV tree with one subject.
  - Calls `_load_mmlu_examples(...)` and asserts 2 examples
    returned with the right subject totals. No `SkipTest` or
    `ImportError` propagates.
  - Restores the original `__import__` and `sys.modules["pandas"]`
    in `finally` to avoid polluting other tests.

- `test_mmlu_5shot_subjects_filter_does_not_crash`:
  - Builds alpha + beta subjects under a temp `AC12_MMLU_DATA_DIR`.
  - Sets `AC12_MMLU_SUBJECTS=beta` + `DS_BASE_URL` + `DSA_BASE_URL`.
  - Mocks `_generate` to return the gold answer ("A") for every
    prompt so the |DSA - DS| gate passes.
  - Mocks `_record_artifact` to capture payloads.
  - Crucial: temporarily clears the class-level
    `__unittest_skip__` + `__unittest_skip_why__` attributes — the
    `@skipUnless` decorator evaluates its condition at module
    import time, so `patch.dict(os.environ)` alone doesn't unskip
    the class. Restores both in `finally`.
  - Runs the test via `unittest.TextTestRunner`. Asserts:
    - No errors and no failures (the gate path completed without
      `NameError`).
    - The recorded MMLU artifact has `subjects == ["beta"]`.
    - `alpha` does NOT appear in `dsa_per_subject` totals (the
      explicit filter actually narrowed evaluation).

## Files Changed

- `test/manual/test_double_sparsity_v32.py`:
  - `_load_mmlu_examples`: removed `import pandas`; inlined
    `_read_csv_rows(path)` stdlib-csv helper; updated CSV reads
    accordingly.
  - `test_mmlu_5shot`: removed the pandas import-guard skipTest
    block; defined `subjects_for_artifact`; replaced the undefined
    `subjects` reference in the recorder.
- `test/registered/unit/manual/test_ac12_helpers.py`:
  - +2 new regressions described above.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/manual/test_ac12_helpers.py -q
47 passed, 0 failed (was 45)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
280 passed, 0 failed (was 278)

env -u DS_BASE_URL -u DSA_BASE_URL python -m pytest test/manual/test_double_sparsity_v32.py -v
6 skipped — clean skip when env vars unset (the ONLY remaining skip path)
```

Round 28's empty-dir Codex reproducer still reports `1 failed`
(unchanged regression from Round 28). Round 28's subjects-filter
NameError is fixed; Round 28's pandas SkipTest is fixed.

Commit: `90942a402` — [AC-12] Drop pandas + fix subjects NameError;
harness fully gate-tight.

## Remaining Items

Code-tier items queued for future rounds:

- `benchmark_compare.py` AC-11 directional gate (3-trial median,
  DS TPS ≥ 95% of DSA, P99 TTFT ≤ 1.10× DSA).
- Shallow AC-8 prefix-match regression coverage cleanup.
- Stale DS bind/runtime comments + token-label lifetime docs.

Hardware-gated tasks unchanged: `task-ac1-hwtest`, `task-ac4-hwrun`,
`task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
`task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`,
`task-ac11-compare`, `task-ac12-quality` (harness fully gate-tight;
hardware execution pending).

## Push-to-remote Status

Branch is 30 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 29 applied the existing
`BL-20260527-conservative-llm-output-parser` principle ("fail loud
in eval harnesses, never silently skip") to two more sub-bugs in
the same harness. The general principle is already captured. The
specific `@unittest.skipUnless` class-decorator caching pattern
that bit Codex's subjects-filter integration test (forcing the
`__unittest_skip__` attribute clear in the regression) is too
narrow to generalize — it shows up whenever someone needs to
exercise an env-gated test class under monkeypatched env vars.
