# Round 28 Summary

## Work Completed

Codex Round 27 review confirmed the parser + missing-directory
data-prep fixes from Round 27, but caught one remaining silent-skip
path:

```
tmp=$(mktemp -d); mkdir -p "$tmp/dev" "$tmp/test"
DS_BASE_URL=... DSA_BASE_URL=... AC12_MMLU_DATA_DIR="$tmp" \
  pytest test_mmlu_5shot
→ 1 skipped (should be 1 failed)
```

When `_ensure_mmlu_data_dir` saw existing `dev/`+`test/` directories,
it returned them as ready — even when the trees contained no usable
CSVs. The test loop then produced an empty `examples` list and fell
through to `self.skipTest("MMLU data dir present but produced no
usable examples")`, silently bypassing the hard AC-12 MMLU gate.

### Fix — `_load_mmlu_examples` pure validator

`test/manual/test_double_sparsity_v32.py`:

- New module-level helper `_load_mmlu_examples(dev_dir, test_dir,
  *, subjects=None, max_examples=200, seed=0xAC12)`:
  - Requires `dev_dir` and `test_dir` to be real directories.
  - Discovers subjects via `*_test.csv`.
  - Per subject: requires paired `{subject}_dev.csv` +
    `{subject}_test.csv`; requires ≥ 5 dev rows; requires test rows
    with ≥ 6 columns.
  - Rejects unusable subjects with a per-subject reason, collected
    into a single composite error message.
  - On no usable examples: raises `ValueError` with the resolved
    paths + rejection reasons + the expected layout.
  - Otherwise shuffles deterministically (`random.Random(seed)`),
    caps at `max_examples`, returns `(examples, per_subject_totals)`.
- Pure function — no unittest dependency, easy to CI-test.

`test_mmlu_5shot` now wraps the call in `try/except ValueError →
self.fail(...)`. The Round 27 silent
`skipTest("MMLU data dir present but produced no usable examples")`
branch is gone. The only acceptable skip remains the class-level
`@unittest.skipUnless` on `DS_BASE_URL` + `DSA_BASE_URL`.

### Verification of Codex's reproducer

```
tmp=$(mktemp -d); mkdir -p "$tmp/dev" "$tmp/test"
DS_BASE_URL=http://127.0.0.1:1 DSA_BASE_URL=http://127.0.0.1:2 \
  AC12_MMLU_DATA_DIR="$tmp" PYTHONPATH=python python -m pytest \
  test/manual/test_double_sparsity_v32.py::TestDoubleSparsityV32Quality::test_mmlu_5shot -v
```

→ `1 failed` with clear remediation message naming
`benchmark/mmlu/bench_sglang.py` + `AC12_MMLU_DATA_DIR`.

### Registered regressions (+7)

`test/registered/unit/manual/test_ac12_helpers.py`:

- `test_load_mmlu_examples_happy_path` — 5 dev rows + 3 test rows →
  3 examples, totals populated correctly.
- `test_load_mmlu_examples_raises_on_empty_test_dir` — Codex's
  reproducer scenario; ValueError "no subjects found".
- `test_load_mmlu_examples_raises_on_too_few_dev_rows` — 3 dev rows
  → ValueError "dev rows, need 5".
- `test_load_mmlu_examples_raises_on_malformed_test_rows` — test
  rows with 4 columns instead of 6 → ValueError "≥6 columns".
- `test_load_mmlu_examples_raises_on_missing_dev_csv` — test CSV
  present, dev CSV absent → ValueError "missing dev or test CSV".
- `test_load_mmlu_examples_deterministic_seed_and_cap` — same seed
  → same order; `max_examples=3` caps to exactly 3 examples.
- `test_load_mmlu_examples_explicit_subjects_filter` — explicit
  `subjects=["beta"]` returns only beta's examples; alpha excluded
  from `per_subject_totals`.

## Files Changed

- `test/manual/test_double_sparsity_v32.py`:
  - Added module-level `_load_mmlu_examples(...)`.
  - Replaced the inline subject-discovery loop in `test_mmlu_5shot`
    with a single call to the helper; converted the silent skip to
    `self.fail(...)`.
- `test/registered/unit/manual/test_ac12_helpers.py`:
  - +7 helper regressions exercising the loader's success path,
    every ValueError trigger, deterministic shuffle + cap,
    explicit subjects filter.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/manual/test_ac12_helpers.py -q
45 passed, 0 failed (was 38)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
278 passed, 0 failed (was 271)

env -u DS_BASE_URL -u DSA_BASE_URL python -m pytest test/manual/test_double_sparsity_v32.py -v
6 skipped — clean skip when env vars unset

# Codex's empty-dir reproducer now FAILS instead of SKIPS:
tmp=$(mktemp -d); mkdir -p "$tmp/dev" "$tmp/test"
DS_BASE_URL=http://127.0.0.1:1 DSA_BASE_URL=http://127.0.0.1:2 \
    AC12_MMLU_DATA_DIR="$tmp" python -m pytest \
    test/manual/test_double_sparsity_v32.py::TestDoubleSparsityV32Quality::test_mmlu_5shot -v
→ 1 failed (was 1 skipped)
```

Commit: `9d39f544e` — [AC-12] Close last MMLU silent-skip path via
`_load_mmlu_examples`.

## Remaining Items

Code-tier items queued for future rounds:

- `benchmark_compare.py` AC-11 directional gate (3-trial median,
  DS TPS ≥ 95% of DSA, P99 TTFT ≤ 1.10× DSA).
- Shallow AC-8 prefix-match regression coverage cleanup.
- Stale DS bind/runtime comments + token-label lifetime docs.

Hardware-gated tasks unchanged: `task-ac1-hwtest`, `task-ac4-hwrun`,
`task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
`task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`,
`task-ac11-compare`, `task-ac12-quality` (harness silent-skip paths
fully closed; hardware execution pending).

## Push-to-remote Status

Branch is 29 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: This round applied the existing
`BL-20260527-conservative-llm-output-parser` principle — "fail
loudly in eval harnesses, never silently skip" — to the data
preflight side of the same harness. The pattern is already
captured. No new entry warranted.
