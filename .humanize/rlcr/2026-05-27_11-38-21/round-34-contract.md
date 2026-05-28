# Round 34 Contract

## Mainline Objective

Close the three AC-11 code-tier gaps Codex flagged in the Round 33
review. All three are real bugs in Round 33's bench_serving timing
path + comparator; none can be deferred without leaving AC-11 unsafe.

1. **bench_serving multi-epoch crash.** Round 33 accumulates outputs
   across epochs but still passes the single-epoch `input_requests`
   to `calculate_metrics`, which indexes `input_requests[i]` per
   output. Any measurement window that needs more than one epoch
   raises `IndexError: list index out of range` and the JSONL is
   never written.

2. **JSONL/sidecar `num_prompts` mismatch.** Round 33 emits
   `num_prompts = args.num_prompts * measured_epochs` in the JSONL,
   while `benchmark.sh` / `_bench_meta_writer.py` write the per-epoch
   `NUM_PROMPTS` into the sidecar. Round 33's own workload
   cross-check then refuses every legitimate multi-epoch artifact.

3. **No DS/DSA side-identity validation.** The comparator strips
   `enable_double_sparsity` + `double_sparsity_config` before
   comparison (DS-only allowance). No later check ensures the DS
   column is actually DS-on or the DSA column is actually DSA.
   Codex's reproducer: three DSA files + three fake "DS" files whose
   sidecars also say `mode="native_nsa"` and lack the DS flags →
   exit 0 + `AC-11 verdict: PASS`.

## Target ACs

- **AC-11** — multi-epoch bench_serving runs publish complete JSONLs,
  per-epoch `num_prompts` round-trips the producer/consumer
  contract, and the comparator refuses any pair where the side
  identity (mode + DS flag presence) does not match the column.

## Required Implementation

### Fix 1: bench_serving multi-epoch metrics path

`python/sglang/bench_serving.py`:

- Track the per-epoch input request rows alongside accumulated
  outputs. For non-multi-turn runs, build a `measured_inputs` list
  whose length equals `len(outputs)` by repeating `input_requests`
  `measured_epochs` times (the workload is deterministic per epoch
  given the seed reset; replication is the right semantic). Pass
  `measured_inputs` to `calculate_metrics(input_requests=...)`.
- For multi-turn runs, keep the existing `input_requests=None`
  behavior.
- Ensure `completed`, output detail arrays, and `duration` are
  internally consistent after repeated epochs (i.e.
  `len(input_lens) == len(outputs) == len(output_lens)`).

### Fix 2: per-epoch `num_prompts` semantics

`python/sglang/bench_serving.py`:

- `num_prompts` in the JSONL is the PER-EPOCH workload shape, equal
  to `args.num_prompts` (matches the sidecar).
- Total measured request attempts go into a NEW field
  `measured_num_prompts = args.num_prompts * measured_epochs`.
- Keep `completed` (= number of successful outputs) and
  `measured_epochs` as-is.

`development/benchmark_compare.py`:

- `_validate_jsonl_workload_matches_sidecar` continues to compare
  JSONL `num_prompts` (per-epoch) with sidecar `num_prompts`
  (per-epoch). Round 33's logic was correct given matching
  semantics; only the bench_serving producer was off-by-multiplier.

### Fix 3: DS/DSA side-identity validation

`development/benchmark_compare.py`:

- New helper `_validate_ac11_side_identity(meta, *, expected_side,
  path)`:
  - DSA: `meta["mode"] == "native_nsa"`; `server_args` either
    lacks `enable_double_sparsity` or has it `False`/`None`; if
    `double_sparsity_config` is present it must be empty/None.
  - DS: `meta["mode"] == "double_sparsity"`;
    `server_args["enable_double_sparsity"] is True`;
    `server_args["double_sparsity_config"]` is a non-empty string.
- Called inside `_run_ac11_mode` for every DSA trial sidecar
  (`expected_side="DSA"`) and every DS trial sidecar
  (`expected_side="DS"`) — runs BEFORE the cross-side normalized
  server_args check (which strips DS-only keys).

### Fix 4: Plan-marker cleanup (queued, do while in the file)

Plan §361-364 forbids plan-specific terminology in implementation
code. Round 33 introduced "Round 33" / "AC-11" / "Codex Round NN
review" in `python/sglang/bench_serving.py` comments + CLI help and
in `development/benchmark_compare.py` comments. Clean these while
fixing the blocking issues above. Keep references in tests, plan
docs, and round summaries.

### Fix 5: Test regressions

`test/registered/unit/development/test_bench_serving_timing.py`:

- `test_multi_epoch_real_metrics_path_no_index_error` — drive
  `bench_serving` with `measurement_window_seconds=0.01`, 2 input
  requests, no `calculate_metrics` monkeypatch. After 2+ epochs the
  JSONL must be written without `IndexError`.
- `test_multi_epoch_jsonl_consistency` — assert
  `len(input_lens) == len(output_lens) == row["completed"]` and
  `row["measured_epochs"] >= 2`.

`test/registered/unit/development/test_ac11_comparator.py`:

- `test_jsonl_num_prompts_per_epoch_matches_sidecar` — JSONL
  `num_prompts=320`, `measured_epochs=2`, `completed=640`, sidecar
  `num_prompts=320` → exit 0 (comparator accepts the legitimate
  multi-epoch artifact).
- `test_jsonl_measured_num_prompts_field_present` — JSONL surfaces
  `measured_num_prompts = num_prompts * measured_epochs`.
- `test_side_identity_dsa_with_ds_flag_refused` — DSA sidecar with
  `enable_double_sparsity=True` → exit 2.
- `test_side_identity_ds_missing_enable_flag_refused` — DS sidecar
  with `enable_double_sparsity` absent or False → exit 2.
- `test_side_identity_ds_missing_config_refused` — DS sidecar with
  empty `double_sparsity_config` → exit 2.
- `test_side_identity_both_sides_native_refused` — both columns
  declare `mode="native_nsa"` and lack DS flags → exit 2 (Codex's
  reproducer).
- `test_side_identity_mode_field_mismatch_refused` — DSA sidecar
  declares `mode="double_sparsity"` (operator copy-paste error) →
  exit 2.

## Tests

- Existing 348 tests + 26 subtests must still pass.
- ~9 new named regressions + new identity subTests.
- Expect ≥ 357 passed.

## Success Criteria

1. Codex's "both-sides-native" reproducer (3 DSA + 3 fake-DS, all
   sidecars `mode="native_nsa"`, no DS flags) now exits 2 with a
   clear "DS side missing enable_double_sparsity" message.
2. `bench_serving --measurement-window-seconds=0.01` with a 2-row
   workload completes without `IndexError` and writes the JSONL.
3. A script-shaped artifact (`num_prompts=320`, `measured_epochs=2`,
   `completed=640`, sidecar `num_prompts=320`) passes
   `--ac11` when other context matches.
4. `grep -nE '\\bRound [0-9]+\\b|AC-[0-9]+' python/sglang/bench_serving.py
   development/benchmark_compare.py` returns no production-code
   matches (test files / docstring AC-11 reference allowed).
5. `pytest test/registered -q` ≥ 357 passed.

## Blocking Issues

The three Round-33-review gaps above. No separate non-mainline
blockers.

## Queued (out of scope for Round 34)

- AC-10 radix-cache hardware fixture (`task-ac10-radix`). Hardware-
  gated; the AC-11 H200 sweep depends on it but the round
  cannot land AC-10 evidence without hardware. The tracker now
  calls AC-10 an "active original-plan task" (not "queued").
- Shallow AC-8 prefix-match helper regression coverage cleanup.
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.
