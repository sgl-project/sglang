# Round 29 Contract

## Mainline Objective

Close the last two AC-12 harness defects Codex flagged in the
Round 28 review:

1. **`pandas` import still silently skips MMLU.** Round 28's
   `test_mmlu_5shot` opens with
   `try: import pandas except ImportError: self.skipTest(...)`.
   That runs INSIDE the test (after class-level `skipUnless` already
   passed), so when servers are configured but pandas is unavailable,
   the gate skips silently. Remove the pandas dependency — `csv` from
   stdlib is enough.
2. **`AC12_MMLU_SUBJECTS` crashes with `NameError`.** Round 28
   renamed `subjects → subjects_arg` but the artifact-recorder line
   still references the undefined name `subjects`. Codex's reproducer
   (`AC12_MMLU_SUBJECTS=beta`, valid tiny dataset, mocked `_generate`)
   raises `NameError: name 'subjects' is not defined` after the eval
   loops.

## Target ACs

- **AC-12** — `test_mmlu_5shot` has no silent-skip paths beyond the
  class-level server-env gate; the `AC12_MMLU_SUBJECTS` filter
  works end-to-end.

## Required Implementation

### Fix 1: Drop pandas; use stdlib `csv`

`test/manual/test_double_sparsity_v32.py`:

- Remove the `import pandas as pd` + `self.skipTest(...)` guard
  block at the top of `test_mmlu_5shot`.
- In `_load_mmlu_examples`, replace `pd.read_csv(path, header=None)`
  with a small `_read_mmlu_csv(path)` helper using
  `csv.reader(open(path, newline=''))`. Return `List[List[str]]`.
- All downstream code already operates on `dev_df.iloc[:5].values.tolist()`
  i.e. nested-list semantics — switching to plain `List[List[str]]`
  is a drop-in replacement.

### Fix 2: Fix `subjects` NameError in artifact recorder

- After parsing `env_subjects`, store the artifact form once:
  `subjects_for_artifact = subjects_arg if subjects_arg is not None
  else "all"`.
- Reference `subjects_for_artifact` in `_record_artifact(...)`, not
  the undefined `subjects`.

### Fix 3: Two new registered regressions

`test/registered/unit/manual/test_ac12_helpers.py`:

- `test_load_mmlu_examples_works_without_pandas` — monkeypatch
  `sys.modules["pandas"] = None` and `__import__` to raise
  `ImportError("pandas")` for any pandas import; then call
  `_load_mmlu_examples` against a tiny valid CSV tree. Proves the
  loader needs no pandas.
- `test_mmlu_5shot_subjects_filter_does_not_crash` — full harness
  integration test: build a tiny `AC12_MMLU_DATA_DIR` with a `beta`
  subject; set `AC12_MMLU_SUBJECTS=beta`; mock `_generate` to return
  the gold answer (so the gate passes); mock `_record_artifact` to
  capture the payload; drive `test_mmlu_5shot` via
  `unittest.TextTestRunner`. Assert no `NameError`; assert recorded
  payload has `subjects == ["beta"]`.

## Tests

- Existing 278 tests must still pass.
- 2 new registered regressions for the pandas-free loader + harness
  subjects-filter integration.
- Expect ≥ 280 passed.

## Success Criteria

1. `import test.manual.test_double_sparsity_v32` succeeds without
   `pandas` installed.
2. `_load_mmlu_examples` runs against a tiny valid CSV tree under
   monkeypatched `pandas` ImportError.
3. `test_mmlu_5shot` with `AC12_MMLU_SUBJECTS=beta` + mocked
   `_generate` returning gold answers reaches the recorder and writes
   `subjects == ["beta"]` — no `NameError`.
4. `pytest test/registered -q` ≥ 280 passed.

## Blocking Issues

None.

## Queued (out of scope for Round 29)

- `benchmark_compare.py` AC-11 directional gate.
- Shallow AC-8 prefix-match regression coverage.
- Stale DS bind/runtime comments + token-label lifetime docs.
- All hardware-gated execution.
