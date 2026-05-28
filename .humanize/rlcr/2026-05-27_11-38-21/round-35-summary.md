# Round 35 Summary

## Work Completed

Codex Round 34 review verdict was **ADVANCED** with no blocking
implementation bugs in AC-11. Action items moved the mainline back to
active original-plan execution, and the only original-plan task with
non-hardware code work still open was `task-ac10-radix`. Round 35
lands every piece of AC-10 that can be done without running V3.2.

### Fix 1 — CPU unit-level proof of DS label bit-stability

`test/registered/unit/layers/attention/test_double_sparsity_unit.py`
+ `TestAC10RadixCacheLabelBitStability` (4 tests). The hardware
fixture's cold-vs-warm label equality reduces, at the labeling level,
to the deterministic property tested here: given the same projected
K-noPE input, `token_label_write` at the same `cache_loc` produces
bit-equal `signatures` rows even when the slot was just invalidated
(the radix-cache reuse semantic).

- `test_token_label_write_is_deterministic_for_same_kv_input` —
  identical input twice → bit-equal rows.
- `test_invalidate_then_rewrite_same_input_yields_equal_labels` —
  write `x` at slot S; `invalidate_token_label_slots(S)`; re-write
  `x` at S → bit-equal to the cold write. This is the labeling-side
  equivalent of the hardware fixture's cold-vs-warm equality
  property.
- `test_different_kv_input_yields_different_labels` (negative
  counterpart): different K-noPE → different label rows, so the
  bit-equality test above is real determinism, not a constant.
- `test_invalidate_does_not_clear_signature_bytes` — `invalidate`
  only clears the `written` flag; signature bytes are preserved.
  Matters because the "no stale picks" property is governed by
  `written` alone — a partial re-write must not be able to mix old
  and new bytes.

### Fix 2 — record_radix_fixture_passed helper

`python/sglang/srt/layers/attention/double_sparsity/validator.py`:
new `record_radix_fixture_passed(server_args)` sets
`_double_sparsity_radix_fixture_passed = True` on the args object and
emits a WARNING-level audit log line. The existing DEC-2 guard inside
`validate_double_sparsity` already reads this attribute (lines
211-223); the helper makes the operator flip explicit + grep-able.

New validator test `test_radix_on_refused_until_fixture_recorded`
proves the two-state guard:

1. `disable_radix_cache=False` + no helper call → validator raises
   `ValueError` with "M3-B page-stability fixture" in the message.
2. `disable_radix_cache=False` + `record_radix_fixture_passed(args)`
   → validator passes cleanly without `SGLANG_DS_RADIX_OVERRIDE=1`.

### Fix 3 — Hardware-gated M3-B fixture harness

`test/manual/test_dsv32_radix_cache_fixture.py` (NEW): mirrors the
AC-12 manual pattern (`@unittest.skipUnless(DS_BASE_URL, ...)`).
Issues a paired cold-prefix / warm-prefix request against a DS server
with radix cache ENABLED:

- **Cold request**: unique shared-prefix payload the server has never
  seen → fresh KV-slot allocation + DS label writes from a single-pass
  FP8 dequant of the just-written K-noPE.
- **Warm request**: same shared-prefix payload re-sent immediately.
  With radix cache ON, the shared-prefix slots are reused.

At `temperature=0` with `max_new_tokens=128`, equal continuations are
the operator-observable proxy for label bit-stability. Records
artifact at
`development/results/dsv32_radix_fixture_cold_warm_<ts>.json` with
`commit_sha`, `server_args`, prompts, continuations, and verdict.
`setUpClass` re-skips when the server reports
`disable_radix_cache=True` so the fixture cannot accidentally run
against the gated configuration it is meant to verify.

### Fix 4 — Launcher AC-10-FIXTURE-MARKER + contract test

`development/serve_double_sparsity.sh`: inline marker

```
`# AC-10-FIXTURE-MARKER: remove the next line after M3-B fixture` \
`# pass + record_radix_fixture_passed(server_args) flip (DEC-2).` \
--disable-radix-cache \
```

placed above the `--disable-radix-cache` flag. The post-AC-10
launcher edit is now a mechanical one-line deletion; the marker
persists for audit. The trailing comment now points operators at the
new `record_radix_fixture_passed` helper.

New `test_ds_server_has_ac10_fixture_marker` contract test in
`test_option_b_scripts.py` asserts (a) the marker is present, and
(b) it sits ABOVE the `--disable-radix-cache` flag so the edit-point
context is visible.

### Fix 5 (queued cleanup) — Strip Round 33 markers from benchmark scripts

`development/benchmark.sh` + `development/benchmark_baseline.sh`:
replaced

```
# Round 33 (AC-11): refuse the run if the observed JSONL `duration`
# is below MEASUREMENT_WINDOW_S — guards against bench_serving
# bailing out early before the time-based loop met its threshold.
```

with neutral wording that names the property, not the round (plan
§361-364). Closes the Codex Round-34 review queued item.

## Files Changed

- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  +`TestAC10RadixCacheLabelBitStability` (4 tests),
  +`test_radix_on_refused_until_fixture_recorded` (1 test).
  +236 lines.
- `python/sglang/srt/layers/attention/double_sparsity/validator.py`:
  +`record_radix_fixture_passed(server_args)` helper.
  +33 lines.
- `test/manual/test_dsv32_radix_cache_fixture.py` (NEW): manual
  hardware fixture + operator runbook + artifact recorder.
  +224 lines.
- `development/serve_double_sparsity.sh`: AC-10-FIXTURE-MARKER above
  `--disable-radix-cache` + updated trailing comment pointing at the
  helper. +9 / -3 lines.
- `test/registered/unit/development/test_option_b_scripts.py`:
  +`test_ds_server_has_ac10_fixture_marker`. +26 lines.
- `development/benchmark.sh` + `development/benchmark_baseline.sh`:
  plan-marker cleanup. +6 / -6 lines.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
362 passed, 26 subtests passed (was 356 + 26; +6 named)

env -u DS_BASE_URL PYTHONPATH=python pytest test/manual/test_dsv32_radix_cache_fixture.py -q
1 skipped (correct: harness skips when env var unset)

bash -n development/serve_double_sparsity.sh   # OK
bash -n development/benchmark.sh               # OK
bash -n development/benchmark_baseline.sh      # OK

grep -nE 'Round 3[0-9]|Codex Round' \
  development/benchmark.sh development/benchmark_baseline.sh
(no output — plan markers fully stripped)
```

Commit: `2d0336a81` — [AC-10] M3-B radix-cache fixture harness +
record_radix_fixture_passed helper.

## Remaining Items

AC-10 hardware execution (operator-driven, not in this loop):

1. Boot DS server with radix cache ON. The launcher still passes
   `--disable-radix-cache`; for this one-shot fixture run set
   `SGLANG_DS_RADIX_OVERRIDE=1` and edit the launcher to drop the
   flag (use the AC-10-FIXTURE-MARKER line as the edit point).
2. Run `DS_BASE_URL=http://...:30000 pytest
   test/manual/test_dsv32_radix_cache_fixture.py -v`. On pass, the
   artifact at `development/results/dsv32_radix_fixture_cold_warm_
   <ts>.json` records the verdict.
3. On pass, permanently remove the `--disable-radix-cache` line in
   `serve_double_sparsity.sh` and update the launcher (or downstream
   server-args parser) to call
   `record_radix_fixture_passed(server_args)` before
   `validate_double_sparsity` runs.
4. Update `test_ds_server_does_disable_radix_cache_until_ac10` to
   the post-AC-10 expectation (e.g., assert the flag is absent +
   the helper is invoked from the launcher).

After AC-10 closes, `task-ac11-compare` H200 3-trial DSA + DS sweep
+ comparator invocation runs.

Other hardware-gated tasks (unchanged): `task-ac1-hwtest`,
`task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
`task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
`task-ac12-quality`.

Queued cleanup (unchanged):
- AC-8 prefix-match helper regression coverage cleanup.
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

## Push-to-remote Status

Branch is 36 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 35 applied existing
`BL-20260527-ds-metadata-via-forward-context` (DS metadata flow),
`BL-20260527-importlib-dataclass-sys-modules` (manual-harness
loader pattern — the new harness uses standard import, no dataclass
loader issue), and `BL-20260527-shell-json-into-python-source`
(launcher's JSON-builder path remains via env-var helper). The
M3-B fixture pattern (CPU unit-level determinism proof + hardware-
gated harness + operator-helper-flip + launcher-marker) is the
AC-10-specific embodiment of a more general "fixture-gated launch
flag" pattern, but it is grounded in this AC's specifics and is
better captured as code + tests than as a generalized BitLesson. No
new entry warranted.
