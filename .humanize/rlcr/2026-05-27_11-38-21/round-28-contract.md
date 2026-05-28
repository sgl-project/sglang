# Round 28 Contract

## Mainline Objective

Close the last remaining silent-skip path in the AC-12 MMLU harness:
when `dev/` and `test/` directories exist but contain no usable
examples (empty, malformed, or missing paired CSVs), the test still
calls `self.skipTest(...)`. With paired server env vars set
(operator running the gate), this is a silent bypass of the hard
AC-12 MMLU gate — exactly the failure mode Round 27 was supposed
to close.

Codex Round-27-review evidence:
```
tmp=$(mktemp -d); mkdir -p "$tmp/dev" "$tmp/test"
DS_BASE_URL=... DSA_BASE_URL=... AC12_MMLU_DATA_DIR="$tmp" \
  pytest test_mmlu_5shot
→ 1 skipped (should be 1 failed)
```

## Target ACs

- **AC-12** — when `DS_BASE_URL` + `DSA_BASE_URL` are set,
  `test_mmlu_5shot` either succeeds (real evaluation against paired
  servers) or fails loudly. No silent-skip path remains.

## Required Implementation

### Fix 1: Extract `_load_mmlu_examples` helper

`test/manual/test_double_sparsity_v32.py`:

- New helper `_load_mmlu_examples(dev_dir, test_dir, *, subjects=None,
  max_examples=200, seed=0xAC12) -> Tuple[List[Dict[str, Any]],
  Dict[str, int]]`:
  - Discover subjects: if `subjects is None or subjects == "all"`, use
    `sorted(f[:-9] for f in os.listdir(test_dir) if f.endswith("_test.csv"))`.
    Otherwise use the explicit list.
  - For each subject, require:
    - `{dev_dir}/{subject}_dev.csv` exists and is readable.
    - `{test_dir}/{subject}_test.csv` exists and is readable.
    - The dev CSV has ≥5 rows.
    - Each test row has ≥6 columns (question + 4 choices + gold).
  - Build the example list (the same code currently in
    `test_mmlu_5shot`), shuffle deterministically with
    `random.Random(seed)`, cap at `max_examples`.
  - Return `(examples, per_subject_totals)`.
- Pure-function — no `unittest` dependencies, easy to CI-test.

### Fix 2: Replace silent skip with `self.fail` in `test_mmlu_5shot`

- After `_ensure_mmlu_data_dir(...)`, call
  `_load_mmlu_examples(dev_dir, test_dir, subjects, max_examples)`.
- Wrap the call in `try/except ValueError`. On failure (no subjects,
  no usable examples) → `self.fail(...)` with a clear message naming
  the resolved `data_dir`, the expected layout, and the
  `AC12_MMLU_DATA_DIR` override.
- The old `if not examples: self.skipTest(...)` branch is gone.
- The only acceptable skip remains the class-level
  `@unittest.skipUnless(_env("DS_BASE_URL") and _env("DSA_BASE_URL"))`.

### Fix 3: Registered regressions

`test/registered/unit/manual/test_ac12_helpers.py`:

- `_load_mmlu_examples` happy path: build a tiny pair of CSVs
  (1 subject, 5 dev rows + 1 test row); assert returns one example
  with the right shape.
- `_load_mmlu_examples` raises `ValueError` when:
  - The test dir is empty (no `*_test.csv`).
  - A subject has fewer than 5 dev rows.
  - A subject's test CSV row has fewer than 6 columns.
- Deterministic shuffle: same `seed` → same example order;
  `max_examples` cap honored.

`test/manual` integration coverage stays minimal — the registered
regressions exercise the helper logic, and the manual harness still
skips cleanly when env vars are unset.

## Tests

- Existing 271 tests must still pass.
- ~6 new registered regressions for `_load_mmlu_examples`.
- Expect ≥ 277 passed.

## Success Criteria

1. `_load_mmlu_examples(empty_dev, empty_test)` raises `ValueError`
   with a clear message (no subjects found).
2. `_load_mmlu_examples(dev, test_with_subject_lacking_5_dev_rows)`
   raises `ValueError`.
3. `test_mmlu_5shot` with `DS_BASE_URL`+`DSA_BASE_URL` set AND a
   missing-or-empty data dir reports `failed`, not `skipped`.
4. Harness still skips cleanly when env vars are unset.
5. `pytest test/registered -q` ≥ 277 passed.

## Blocking Issues

None.

## Queued (out of scope for Round 28)

- `benchmark_compare.py` AC-11 directional gate (3-trial median,
  TPS ≥ 95% of DSA, P99 TTFT ≤ 1.10× DSA).
- Shallow AC-8 prefix-match regression coverage.
- Stale DS bind/runtime comments + token-label lifetime docs.
- Hardware-gated execution of AC-12, AC-1, AC-1b, AC-4, AC-6, AC-8,
  AC-9, AC-10, AC-11.
