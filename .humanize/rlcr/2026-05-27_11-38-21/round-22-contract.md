# Round 22 Contract

## Mainline Objective

Make the Round 21 AC-8 smoke harness usable on real hardware by fixing
the two gate bugs Codex flagged as blocking, and finish the Round 21
token-rename by updating `benchmark_compare.py`'s remaining
`selected_pages_mean` / `total_pages_mean` references so the
comparator does not conflict with the new metric names.

Codex Round-21-review findings (blocking side issues):

1. **prefix-match gate rejects valid short answers.** The Round 21
   harness counts a prefix-match hit only when both
   `ds[:32] == dsa[:32]` AND `len(dsa) >= 32`. The AC-8 contract is
   "first 32 chars match" — a 4-char identical answer like `1969` or
   `Au` should count as a hit, not a miss. With the current code, ~12
   of the 20 smoke prompts (the ones whose prompt says "Output only X")
   would fail the gate for a perfectly-matching DS run.
2. **`_first_n_tokens_match` documented "any overlap" but only checks
   same-position equality.** Shifted overlap (`alpha beta gamma` vs
   `beta gamma alpha`) returns False even though every token is shared.
   Required: set intersection over the first n tokens.

Residual from Round 21's token rename:

3. `development/benchmark_compare.py` still consumes and reports
   `selected_pages_mean` / `total_pages_mean`. After Round 21 these are
   stale field names that mismatch the per-request `selected_tokens`
   publication; clean up the rename so the comparator and its existing
   unit tests are consistent. (The 3-trial-median + AC-11 directional
   gate enforcement is a separate, larger round — not in Round 22 scope.)

## Target ACs

- **AC-8** — the quality smoke harness's four gates accept a perfect
  DS/DSA run (no false-negative rejections from the bugs above), and
  the comparator code does not reference the removed
  `selected_pages_mean` field names.

## Required Implementation

### Fix 1: Prefix-match gate accepts short exact matches

`test/manual/test_dsv32_quality_smoke.py`:
- Change the prefix hit condition from
  `ds[:32] == dsa[:32] and len(dsa) >= 32` to
  `ds[:PREFIX_MATCH_CHARS] == dsa[:PREFIX_MATCH_CHARS]`.

### Fix 2: `_first_n_tokens_match` uses set intersection

`test/manual/test_dsv32_quality_smoke.py`:
- Replace the zipped same-position scan with
  `bool(set(a_toks[:n]) & set(b_toks[:n]))`.

### Fix 3: Helper-level regression tests

Add a registered (CPU) unit test file
`test/registered/unit/manual/test_dsv32_quality_smoke_helpers.py` (or
extend an existing file) that imports the helpers and asserts:
- Exact short match (e.g. `"Au"` vs `"Au"`) counts as a prefix hit.
- Genuinely different short outputs (`"Au"` vs `"Ag"`) do not count.
- `_first_n_tokens_match("alpha beta gamma", "beta gamma alpha", n=3)`
  is True (shifted overlap).
- `_first_n_tokens_match("a b c", "x y z", n=3)` is False.

### Fix 4: `benchmark_compare.py` page→token rename

- `development/benchmark_compare.py`: rename `selected_pages_mean` →
  `selected_tokens_mean`, `total_pages_mean` → `total_tokens_mean` in
  the `RunMetrics` dataclass, all consumers, all reporting strings, and
  the SLO/no-op message text.
- Update existing tests in
  `test/registered/unit/layers/attention/test_double_sparsity_unit.py`
  that pass these as kw-args or assert on the message text.

## Tests

- All existing 202 tests must still pass.
- New helper regression tests (~4) must pass.
- Renamed `benchmark_compare.py` tests still pass.

## Success Criteria

1. `_first_n_tokens_match` returns True on shifted-overlap fixture.
2. Prefix-match gate counts a 2-character identical answer (`"Au"`) as
   a hit.
3. `benchmark_compare.py` has 0 references to `selected_pages_mean`
   or `total_pages_mean` (verified by grep).
4. `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q`
   ≥ 206 passed, 0 failed (existing 202 + 4 new helper tests).
5. `pytest test/manual/test_dsv32_quality_smoke.py` still skips
   cleanly with no env vars.

## Blocking Issues

None.

## Queued (out of scope for Round 22)

- AC-12 scaffold replacement (`test/manual/test_double_sparsity_v32.py`
  full NIAH/MMLU implementation) — separate round.
- `benchmark_compare.py` 3-trial median + AC-11 directional gate
  enforcement — separate round.
- `serve_double_sparsity.sh` + `serve_native_nsa.sh` Option B flag
  alignment.
- `benchmark.sh` + `benchmark_baseline.sh` conc 16/32/64 sweep.
- Stale DS bind/runtime comments + token-label lifetime docs.
- Hardware-gated tasks (`task-ac1-hwtest`, `task-ac4-hwrun`,
  `task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
  `task-ac8-quality`, `task-ac12-quality`).
