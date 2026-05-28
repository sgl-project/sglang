# Round 22 Summary

## Work Completed

Codex Round-21-review verified the token-denominator observability
fix but reopened the AC-8 smoke harness as blocking because two gate
bugs would reject a perfect paired DS/DSA run. Round 22 fixes both
and finishes the residual page→token rename in `benchmark_compare.py`.

### Fix 1 — prefix-match gate accepts short exact matches

`test/manual/test_dsv32_quality_smoke.py`:
- Removed the `len(dsa) >= self.PREFIX_MATCH_CHARS` guard from the
  prefix-match-hit condition. The AC-8 contract is "first 32 chars
  match", not "DSA answer must be at least 32 chars long".
- ~12 of the 20 smoke prompts deliberately request short outputs
  ("Output only X", "Give just Y"). Under the old guard, exact
  short DS/DSA matches like `Au == Au`, `1969 == 1969`, or
  `Jupiter == Jupiter` were counted as misses, breaking the 80%
  prefix-match gate even with perfect quality.

### Fix 2 — `_first_n_tokens_match` uses set intersection

`test/manual/test_dsv32_quality_smoke.py`:
- Replaced the zipped same-position scan
  (`any(at == bt for at, bt in zip(a_toks, b_toks))`) with set
  intersection (`bool(set(a_toks) & set(b_toks))`). The docstring
  always said "any overlap"; the implementation only detected
  positionally-aligned overlap. Shifted overlap like
  `"alpha beta gamma"` vs `"beta gamma alpha"` returned False even
  though every token is shared.

### Fix 3 — Helper-level regression tests

`test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
new class `TestDSv32SmokeHelpers` loads the manual smoke module via
`importlib.util.spec_from_file_location` (no `__init__.py` under
`test/manual/`) and exercises the helpers in CI:

- `test_prefix_match_accepts_short_exact_outputs` — `"Au" == "Au"`
  prefix hit.
- `test_prefix_match_rejects_short_different_outputs` — `"Au" != "Ag"`.
- `test_first_n_tokens_match_shifted_overlap_is_true` —
  `("alpha beta gamma", "beta gamma alpha", n=3)` → True.
- `test_first_n_tokens_match_no_overlap_is_false` — `("a b c", "x y z")` → False.

### Fix 4 — `benchmark_compare.py` page→token rename

Residual from Round 21's per-request rename. After Round 21 the
per-request publication used `selected_tokens`, but the comparator
still consumed/reported `selected_pages_mean` / `total_pages_mean` —
a stale naming conflict.

`development/benchmark_compare.py`:
- `RunMetrics.selected_pages_mean` → `selected_tokens_mean`.
- `RunMetrics.total_pages_mean` → `total_tokens_mean`.
- JSON consumer reads `selected_tokens_mean` / `total_tokens_mean`.
- Missing-field reporter, no-op-detector message
  (`selected_pages == total_pages` → `selected_tokens == total_tokens`),
  and report row labels updated.
- Existing unit tests at lines 2577/2579/2584/2585/2593/2595/2754/2756
  in `test_double_sparsity_unit.py` updated to pass / assert the new
  field names.

The bigger AC-11 work (3-trial median, DS TPS within 5% of DSA, P99
TTFT ≤ 1.10× DSA) is explicitly out of Round 22 scope and stays
queued for a future round.

## Files Changed

- `test/manual/test_dsv32_quality_smoke.py`: prefix-match condition +
  `_first_n_tokens_match` semantics + clarifying comments.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  new `TestDSv32SmokeHelpers` class (4 tests); renamed
  `selected_pages_mean` / `total_pages_mean` references in existing
  benchmark_compare tests.
- `development/benchmark_compare.py`: page→token rename across
  RunMetrics, JSON consumer, no-op detector, report rows.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q
206 passed, 0 failed (was 202 before this round)

env -u DS_BASE_URL -u DSA_BASE_URL python -m pytest test/manual/test_dsv32_quality_smoke.py -v
1 skipped — clean skip when env vars unset

pytest -v -k TestDSv32SmokeHelpers
4 passed
```

Manual helper sanity:
```
_first_n_tokens_match("alpha beta gamma", "beta gamma alpha", n=3) = True
_first_n_tokens_match("a b c", "x y z", n=3) = False
_rouge_l_f("Au", "Au") = 1.0
_rouge_l_f("Au", "Ag") = 0.0
```

Branch state: 23 commits ahead of `jimmy/dev/double-sparsity-standalone`.
Commit `931949f99` — [AC-8] Fix two AC-8 smoke gate bugs + finish R21 token rename.

## Remaining Items

Mainline AC items still requiring hardware execution:
- `task-ac8-server` + `task-ac8-quality` (harness is now correct; needs
  paired DS+DSA H200 servers + same-session reference run).
- `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`,
  `task-ac1b-probe`, `task-ac12-quality`.

Code-tier items still queued for future rounds:
- Replace `test/manual/test_double_sparsity_v32.py` skip-only scaffolds
  with real NIAH 4K/16K/64K + MMLU 5-shot execution against paired
  servers (AC-12).
- `serve_double_sparsity.sh` + `serve_native_nsa.sh` Option B flag
  alignment.
- `benchmark.sh` + `benchmark_baseline.sh` conc 16/32/64 sweep.
- `benchmark_compare.py` 3-trial median + AC-11 directional gate
  enforcement.
- Stale DS bind/runtime comments + token-label lifetime docs.

## Push-to-remote Status

Branch is 23 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`; commits remain local only. To enable per-round pushing,
re-launch with `--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Both gate bugs were standard documentation-vs-implementation
mismatches in test fixtures — narrow enough that a specific BitLesson
would not generalize usefully. The token rename was mechanical cleanup
after Round 21's primary rename, also too narrow. The existing
`BL-20260527-reshape-before-slice-mla` covers the general "re-check
derived names after rotation" framing.
