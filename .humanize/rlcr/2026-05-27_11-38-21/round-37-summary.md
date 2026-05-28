# Round 37 Summary

## Work Completed

Codex Round 36 review found three real defects: (1) the M3-B
label-capture fixture had a false-pass path because the capture log
lived in the SERVER process but the fixture imported it CLIENT-side
(different memory; empty list ⇒ `not mismatches and cached_tokens
> 0` PASSes); (2) even with a real server log, radix-cache HITS
skip the write path entirely so write-record overlap can never
prove warm-prefix label stability — `record_table_snapshot` existed
but was unused; (3) the FP8 fixture called
`sglang_per_token_group_quant_fp8` (row-wise quantizer) rather than
exercising the production `fused_store_index_k_cache` path. Round 37
closes all three.

### Fix 1 — Server-side per-request capture via `meta_info`

`python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py`:

- New pure helper `build_request_capture(*, signatures, written,
  req_to_token, req_pool_indices, seq_lens) -> list[dict]`.
- For each request in the batch, computes
  `slots = req_to_token[req_pool_indices[b], :seq_lens[b]]` and
  returns a dict with `prompt_len`, `slots_sha`,
  `per_layer_label_sha`, `per_layer_written_sha`,
  `per_layer_written_all_true`.
- Pure function: safe to call from the production per-request
  finalization site; no IO, no globals, no env reads.

`python/sglang/srt/models/deepseek_v2.py::_publish_ds_request_summary`:

- When `radix_fixture_capture.is_capture_enabled()` is True, looks
  up `self.double_sparsity_selector.token_label_table` and
  `forward_batch.req_to_token_pool.req_to_token`, calls
  `build_request_capture(...)`, and attaches the per-request
  snapshot records to
  `summary["double_sparsity_radix_capture"]`.
- Wrapped in try/except so a capture-path bug can never break
  production; the WARNING-style `summary[
  "double_sparsity_radix_capture_error"]` records the failure
  shape.
- Default (env unset) path pays exactly one `os.environ.get`
  lookup.

Snapshots (not write records) are the right primitive: on a
radix-cache HIT the warm pass skips the write path entirely, so
write-record-overlap comparison would find nothing. The snapshot
of `signatures[L, prompt_slots]` after the request captures the
slot state regardless of whether labels were re-written.

### Fix 2 — Pure verdict helper + 11 CPU regressions

`test/manual/_m3b_label_capture_verdict.py` (NEW): pure helper
`evaluate_m3b_label_capture_verdict(*, cold_capture, warm_capture,
cached_tokens) -> {verdict, reasons}`. PASS only when ALL of:

- both captures present and non-empty;
- `cached_tokens > 0` on warm pass;
- `slots_sha` matches between cold and warm;
- `per_layer_label_sha` matches;
- `per_layer_written_all_true` is True on both sides.

`test/registered/unit/manual/test_m3b_label_capture_verdict.py`
(NEW, 11 tests):

- `test_all_conditions_met_is_pass`
- `test_empty_cold_capture_fails_false_pass_guard` ← closes the
  Codex Round-36 false-pass class
- `test_empty_warm_capture_fails`
- `test_none_capture_fails`
- `test_zero_cached_tokens_fails`
- `test_slots_sha_mismatch_fails`
- `test_layer_label_sha_mismatch_fails` (with first-mismatch hint)
- `test_layer_label_length_mismatch_fails`
- `test_unwritten_slot_on_cold_fails`
- `test_unwritten_slot_on_warm_fails`
- `test_non_list_capture_treated_as_missing`

### Fix 3 — Capture-aware fixture reads `meta_info`

`test/manual/test_dsv32_radix_label_capture_fixture.py` rewritten:

- Dropped the broken client-process in-process import path AND
  the optional HTTP endpoint scaffolding.
- Reads `response["meta_info"]["double_sparsity_radix_capture"]`
  for each pass; calls the pure verdict helper; asserts PASS.
- Works against remote and local servers identically (capture
  travels with the response). Class-level skip remains on
  `DS_BASE_URL`; the helper reports if
  `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` was not set on the server
  (capture list empty → verdict FAILs the run with a clear
  message rather than silently scoring it as a pass).

### Fix 4 — FP8 fixture via production fused-store path

`test/manual/test_dsv32_fp8_scale_stability.py` rewritten:

- Directly calls `sglang.jit_kernel.fused_store_index_cache.
  fused_store_index_k_cache` — the JIT kernel that the production
  `DSAIndexer._store_index_k_cache` uses on the fused-store path.
- Allocates a real `index_k_with_scale_buffer`-shaped tensor of
  shape `(num_pages, page_size * (128 + 4))` (the production
  layout for DSv4SingleKVPool).
- Writes K0 alone into page 0 of `buf_singleton`; writes K0 + 63
  deterministic neighbours into page 0 of `buf_packed`.
- Reads back K0's FP8 bytes (`buf[0, 0:128]`) and per-token scale
  bytes (`buf[0, page_size*128 : +4]`) at the production byte
  offsets. Asserts bit-equality of both.
- Hardware-gated via `SGLANG_DS_FP8_SCALE_PROOF=1` + CUDA +
  `can_use_dsa_fused_store(...)` returning True.

### Fix 5 — `build_request_capture` CPU unit tests (+4)

`TestBuildRequestCapture` in `test_double_sparsity_unit.py`:

- `test_single_request_snapshot_matches_manual_hash` — proves the
  per-layer SHA equals a manual `hashlib.sha256(...).hexdigest()`
  computed on the same slot bytes.
- `test_identical_calls_produce_identical_records` — foundation
  of the cold/warm equality.
- `test_two_request_batch_per_request_records_independent` —
  per-request records use the correct `req_to_token` row.
- `test_unwritten_slots_flag_not_all_true` — when prompt slots are
  not all written, `written_all_true` is False so the fixture
  refuses the side.

### Fix 6 — Launcher comment names the M3-B fixtures

`development/serve_double_sparsity.sh` trailing comment rewritten
to name BOTH the label-capture fixture
(`SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`) AND the FP8 production-path
fixture (`SGLANG_DS_FP8_SCALE_PROOF=1`) as the M3-B evidence
required before any guard flip — the continuation smoke is now
correctly identified as pre-flight only.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py`:
  `build_request_capture(...)` pure helper.
- `python/sglang/srt/models/deepseek_v2.py`: env-gated capture
  attachment in `_publish_ds_request_summary`.
- `development/serve_double_sparsity.sh`: trailing comment now
  names the M3-B fixtures.
- `test/manual/_m3b_label_capture_verdict.py` (NEW): pure verdict
  helper.
- `test/manual/test_dsv32_radix_label_capture_fixture.py`:
  rewritten to read `meta_info` + use the pure verdict helper.
- `test/manual/test_dsv32_fp8_scale_stability.py`: rewritten to
  call `fused_store_index_k_cache` on a real
  `index_k_with_scale_buffer`-shaped buffer.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  +`TestBuildRequestCapture` (4 tests).
- `test/registered/unit/manual/test_m3b_label_capture_verdict.py`
  (NEW): +11 verdict regressions.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
389 passed, 26 subtests passed (was 374 + 26; +15 named)

PYTHONPATH=python pytest test/registered/unit/manual/test_m3b_label_capture_verdict.py -v
11 passed (every false-pass class Codex Round 36 named now FAILs)

env -u DS_BASE_URL -u SGLANG_DS_RADIX_FIXTURE_CAPTURE -u SGLANG_DS_FP8_SCALE_PROOF \
  PYTHONPATH=python pytest \
   test/manual/test_dsv32_radix_cache_fixture.py \
   test/manual/test_dsv32_radix_label_capture_fixture.py \
   test/manual/test_dsv32_fp8_scale_stability.py -q
3 skipped (all 3 manual fixtures skip cleanly when env unset)

bash -n development/serve_double_sparsity.sh   # OK
```

The verdict helper unit suite reproduces the exact false-pass
scenarios Codex Round-36 review identified and confirms they now
FAIL — the same helper is used by the manual hardware fixture, so
the hardware-side path is guarded by the same logic.

Commit: `b7ab80fc0` — [AC-10] M3-B direct evidence via meta_info +
production FP8 store-path proof.

## Remaining Items

AC-10 hardware execution (operator-driven, not in this loop):

1. Boot DS server with `SGLANG_DS_RADIX_OVERRIDE=1` +
   `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` + remove
   `--disable-radix-cache` for this one-shot run.
2. Run `test_dsv32_fp8_scale_stability.py` with
   `SGLANG_DS_FP8_SCALE_PROOF=1`. On PASS, save the artifact.
3. Run `test_dsv32_radix_label_capture_fixture.py`. On PASS, save
   the artifact.
4. Wire `record_radix_fixture_passed(server_args, artifact_path=
   "<label-capture-artifact.json>")` into a launcher init module
   BEFORE `validate_double_sparsity` runs.
5. Delete the `--disable-radix-cache` line in
   `serve_double_sparsity.sh` (the AC-10-FIXTURE-MARKER comment
   names the exact line).
6. Update `test_ds_server_does_disable_radix_cache_until_ac10`
   for the post-AC-10 expectation.

After AC-10 closes, `task-ac11-compare` H200 3-trial DSA + DS
sweep + comparator invocation runs.

Other hardware-gated tasks (unchanged): `task-ac1-hwtest`,
`task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
`task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
`task-ac12-quality`.

Queued cleanup (unchanged):
- AC-8 prefix-match helper regression coverage cleanup.
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

## Push-to-remote Status

Branch is 38 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 37 applied `BL-20260527-ds-metadata-via-forward-context`
to the new server-side capture attachment in
`_publish_ds_request_summary` — the DS metadata access lives on
`self.double_sparsity_selector.token_label_table` +
`forward_batch.req_to_token_pool`, not on
`forward_batch.attn_backend` (which production never sets). The
"capture-via-per-request-summary" pattern is one-shot
infrastructure for AC-10 specifically rather than a general
guideline. No new entry warranted.
