# Round 34 Summary

## Work Completed

Codex Round 33 review identified three real bugs in Round 33's
bench_serving timing + comparator code. None could be deferred without
leaving AC-11 unsafe.

### Fix 1 — Multi-epoch metrics IndexError

`python/sglang/bench_serving.py`: when `measurement_window_seconds > 0`
ran more than one epoch, `outputs` accumulated across epochs but
`calculate_metrics(input_requests=input_requests, outputs=outputs)`
indexed `input_requests[i]` for every successful output. As soon as
`len(outputs) > len(input_requests)`, the metric path raised
`IndexError: list index out of range` and the JSONL was never written.

Fix: replicate the per-epoch input row list `measured_epochs` times
before passing it to `calculate_metrics` for non-multi-turn runs. The
workload is deterministic per epoch (seed reset between warmup and
measured phases), so replication mirrors the dispatch order. Multi-
turn keeps the existing `input_requests=None` behavior. Result:

```python
if is_multi_turn:
    metrics_input_requests = None
elif measured_epochs > 1:
    metrics_input_requests = input_requests * measured_epochs
else:
    metrics_input_requests = input_requests
```

### Fix 2 — Per-epoch `num_prompts` in JSONL, total in `measured_num_prompts`

Round 33 emitted `num_prompts = args.num_prompts * measured_epochs`
in the JSONL, but the benchmark scripts wrote the per-epoch
`NUM_PROMPTS` into the sidecar. The Round 33 workload cross-check
then refused every legitimate multi-epoch artifact. The cross-check
itself was correct given matching semantics; only the producer was
off-by-multiplier.

Fix (`python/sglang/bench_serving.py`):

- `num_prompts` in the JSONL is now the PER-EPOCH workload shape
  (= `args.num_prompts`), matching the sidecar's `num_prompts`.
- New JSONL field `measured_num_prompts = per_epoch * measured_epochs`
  carries the total measured attempts across the multi-epoch window.
- `completed` (number of successful outputs) is unchanged.

### Fix 3 — DS/DSA side-identity validation

Because `_normalize_ac11_server_args` strips
`enable_double_sparsity` + `double_sparsity_config` from cross-side
comparison (those are the ONLY sanctioned differences per plan §AC-11),
the comparator could publish PASS when both columns were actually
DSA-on. Codex's reproducer: 3 DSA files + 3 fake "DS" files whose
sidecars also said `mode="native_nsa"` and lacked the DS flags →
exit 0 + `AC-11 verdict: PASS`.

New helper `_validate_ac11_side_identity(meta, *, expected_side, path)`
runs inside `_run_ac11_mode` before cross-side normalization:

- **DSA** column requires `mode == "native_nsa"`, no
  `enable_double_sparsity` (or it is False/absent), and no non-empty
  `double_sparsity_config`.
- **DS** column requires `mode == "double_sparsity"`,
  `server_args.enable_double_sparsity is True`, and a non-empty
  `double_sparsity_config` string.

### Cleanup — Plan markers removed from production code

Per plan §361-364, implementation code must not carry plan-specific
terminology. Round 33 introduced "Round 33" / "Codex Round NN review"
markers; Round 34 stripped them:

- `python/sglang/bench_serving.py`: 4 comments + 2 CLI help strings.
- `development/benchmark_compare.py`: 5 comments.

Kept references that are load-bearing for users: the `--ac11` CLI
flag, AC-11 in error messages (operator needs to know which gate
they're hitting), and AC-11 in the module docstring.

### Regressions (+8 named)

`test/registered/unit/development/test_bench_serving_timing.py` (+2):

- `test_multi_epoch_real_metrics_path_no_index_error` — drives
  bench_serving with 2 input rows and a 30ms window, WITHOUT
  monkeypatching `calculate_metrics`. The real metric path is
  exercised; asserts `measured_epochs >= 2` and
  `completed == num_input_rows * measured_epochs`. Round 33's code
  would have raised `IndexError` here.
- `test_multi_epoch_jsonl_consistency` — asserts `num_prompts` is
  per-epoch, `measured_num_prompts == num_prompts * measured_epochs`,
  and `duration >= window`.

`test/registered/unit/development/test_ac11_comparator.py` (+6):

- `test_side_identity_dsa_with_ds_flag_refused` — DSA sidecar has
  `enable_double_sparsity=True` → exit 2.
- `test_side_identity_ds_missing_enable_flag_refused` — DS sidecar
  lacks the enable flag → exit 2.
- `test_side_identity_ds_missing_config_refused` — DS sidecar has
  empty `double_sparsity_config` → exit 2.
- `test_side_identity_both_sides_native_refused` — closes Codex's
  reproducer: 3 DSA + 3 fake-DS all `mode='native_nsa'` → exit 2.
- `test_side_identity_mode_field_mismatch_refused` — DSA sidecar
  declares `mode='double_sparsity'` (copy-paste error) → exit 2.
- `test_jsonl_num_prompts_per_epoch_matches_sidecar` — legitimate
  multi-epoch artifact (`num_prompts=320`, `measured_epochs=2`,
  `measured_num_prompts=640`) matching sidecar `num_prompts=320`
  passes the comparator.

## Files Changed

- `python/sglang/bench_serving.py`: replicate per-epoch input rows
  for `calculate_metrics`; emit per-epoch `num_prompts` + new
  `measured_num_prompts` field; strip plan markers. +56 / -48 lines.
- `development/benchmark_compare.py`: `_validate_ac11_side_identity`
  helper + call from `_run_ac11_mode` (DSA + DS); plan-marker
  cleanup. +99 / -19 lines.
- `test/registered/unit/development/test_ac11_comparator.py`: +6
  identity / per-epoch regressions. +103 lines.
- `test/registered/unit/development/test_bench_serving_timing.py`:
  +2 multi-epoch real-metrics regressions. +147 lines.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
57 passed, 26 subtests passed (was 51 + 26)

PYTHONPATH=python pytest test/registered/unit/development/test_bench_serving_timing.py -q
9 passed (was 7)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
356 passed, 26 subtests passed (was 348 + 26)
```

Verified Codex's "both-sides-native" reproducer (3 DSA + 3 fake-DS
sidecars, all `mode='native_nsa'`, no DS flags, all other context
matching):

```
python development/benchmark_compare.py --ac11 \
    --ac11-baseline-results dsa_1.jsonl dsa_2.jsonl dsa_3.jsonl \
    --ac11-ds-results ds_1.jsonl ds_2.jsonl ds_3.jsonl
→ exit 2 with
  "AC-11 input refusal at conc=64: AC-11 DS trial ...: sidecar
   mode='native_nsa' but expected 'double_sparsity' for the DS-on
   column."
(was exit 0 + "AC-11 verdict: PASS")
```

Verified the multi-epoch real-metrics path: 2 input rows + 30ms
window completes without `IndexError` and writes a coherent JSONL.

```
grep -nE 'Round 3[0-9]|Codex Round' \
  python/sglang/bench_serving.py development/benchmark_compare.py
(no output — plan markers stripped)
```

Commit: `04e811aba` — [AC-11] fix multi-epoch metrics crash,
per-epoch num_prompts, side-identity gate.

## Remaining Items

Hardware-gated tasks (unchanged):
- `task-ac1-hwtest`, `task-ac4-hwrun`, `task-ac6-hwrun`,
  `task-ac1b-probe`, `task-ac8-server`, `task-ac8-quality`,
  `task-ac9-baseline`, `task-ac12-quality`.

Active original-plan task blocking AC-11 H200 execution:
- `task-ac10-radix` — M3-B radix-cache hardware fixture; FP8 scale
  stability check; flip `_double_sparsity_radix_fixture_passed =
  True`; remove `--disable-radix-cache` from
  `serve_double_sparsity.sh`; update launcher contract tests. This
  is an active original-plan task (corrected the tracker wording
  per the previous-round review feedback). After AC-10 passes,
  `task-ac11-compare` runs the H200 3-trial sweep + comparator.

Queued side issues (unchanged):
- AC-8 prefix-match helper regression coverage cleanup.
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

## Push-to-remote Status

Branch is 35 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 34 applied existing
`BL-20260527-shell-json-into-python-source` (sidecar/JSONL JSON
hand-off) and `BL-20260527-ds-metadata-via-forward-context` (DS path
identification — applied conceptually to the new side-identity
validator). The multi-epoch input-replication fix and the
identity-gate pattern are problem-specific defenses better captured
as code + regressions than as generalized lessons. No new entry
warranted.
