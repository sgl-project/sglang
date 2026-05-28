# Round 38 Summary (Drift Recovery)

## Drift Cause + Recovery Anchor

Rounds 35 → 37 iteratively patched the M3-B label-capture fixture
but never verified the full producer→transport→consumer contract
in CI. Each round Codex review found a NEW seam:

* R35 — continuation-only proxy (not direct label evidence).
* R36 — capture log lived in the client process (different memory
  from the server's `_LOG`).
* R37 — capture published as a per-batch list under
  `summary["double_sparsity_radix_capture"]`, but the existing
  per-request summary transport in
  `_maybe_collect_per_request_summary` (`batch_result_processor.py`)
  unwraps `v[i]` per request — the client receives a **dict per
  request** in `meta_info`. My verdict helper rejected dicts as
  missing. Also: capture fired from `_publish_ds_request_summary`
  (called per-layer at the START of selection, BEFORE writes), so
  the current layer was stale. Also: only the fused FP8 path was
  implemented; fallback was `skipTest`.

**Recovery anchor**: a CPU integration test that reproduces the
actual transport unwrap and asserts the verdict helper accepts the
dict-shaped per-request meta_info. Future shape/timing drift on
either side now fails locally in CI, not in remote review.

## Mainline Objective + Target ACs

- AC-10 M3-B label-capture fixture delivers honest direct evidence
  on H200 with the correct response shape, post-write extend-only
  timing, cached-prefix-only comparison, AND both fused + fallback
  FP8 production-store paths.

## Work Completed

### Fix 1 — Verdict helper accepts the real transported shape

`test/manual/_m3b_label_capture_verdict.py::_records`:

- `dict` → `[dict]` (production transport: scheduler unwraps `v[i]`
  per request → tokenizer surfaces a single dict in meta_info).
- `list` → `list` (legacy / direct helper test).
- None / missing / other → None (treated as missing evidence).

### Fix 2 — Capture moved to `_write_token_labels` post-write, extend-only

`python/sglang/srt/layers/attention/dsa_backend.py`:

- New module-level helper `_ds_radix_publish_extend_snapshot(*,
  backend, forward_batch)` calls `build_request_capture` against
  the live table state and stashes the per-batch list on
  `forward_batch.ds_per_request_summary["double_sparsity_radix_capture"]`.
- Called from `_write_token_labels` AFTER `token_label_write(...)`
  returns AND only when `forward_batch.forward_mode.is_extend()`.
- Each layer's publish overwrites the previous → the LAST DS
  layer's call wins; at that point every layer's prompt-slot
  labels are fresh.
- Extend-only restriction prevents decode steps from clobbering
  the prefill snapshot.
- Wrapped in try/except; failure records the shape in
  `summary["double_sparsity_radix_capture_error"]` but never
  raises (capture must not break production).

`python/sglang/srt/models/deepseek_v2.py::_publish_ds_request_summary`:

- Removed the Round-37 capture branch (fired pre-write, per layer,
  every mode). Comment left in place naming the new emission site.

### Fix 3 — Per-token API in `build_request_capture`

`radix_fixture_capture.py`:

- Each record now includes `per_token_slot_sha: list[str]`
  (per-prompt-position slot SHA) and
  `per_layer_per_token_label_sha: list[list[str]]` (per layer,
  per position).
- `slots_sha` retained for back-compat (aggregate over the full
  range).
- New `compare_cached_prefix(*, cold, warm, cached_tokens) ->
  {ok, first_diverging_position, divergence_kind, reason}` —
  compares only the first `cached_tokens` positions. Clamps to
  the shorter capture so extra decode-allocated positions on
  either side don't force a false mismatch.

### Fix 4 — Producer→transport→consumer CPU integration test

`test/registered/unit/manual/test_m3b_label_capture_verdict.py`:

- `test_meta_info_transport_dict_shape_passes_verdict` — the
  drift-recovery anchor. Mirrors the scheduler's
  `_maybe_collect_per_request_summary` unwrap shim, asserts the
  per-request meta_info value is a DICT, then feeds it to the
  verdict helper → PASS.
- `test_meta_info_transport_dict_shape_with_extra_decode_slots` —
  warm has more positions than cold; `cached_tokens=5` → PASS
  (positions ≥5 ignored).
- `test_meta_info_transport_dict_shape_with_cached_prefix_diverging`
  — position 2 differs in the cached prefix → FAIL with
  `first_diverging_position=2`, `kind='slot'`.
- `test_meta_info_transport_label_divergence_within_cached_prefix`
  — slots match but layer-1 label SHA differs at position 3 →
  FAIL with `kind='label'`, `layer=1`.
- 9 verdict negatives: empty/None/non-list-non-dict capture,
  zero cached_tokens, written_all_true=False on either side,
  direct dict path PASS, direct list path PASS.

### Fix 5 — FP8 fixture with fused + fallback production paths

`test/manual/test_dsv32_fp8_scale_stability.py`:

- `_read_back_k0_bytes(buf, *, page_idx, position)` — shared
  helper using production byte offsets (`buf[page,
  position*128:(position+1)*128]` for FP8 and
  `buf[page, PAGE_SIZE*128 + position*4 : +4]` for the scale).
- `_try_fused_path` — `fused_store_index_k_cache(K0, buf, loc,
  page_size)` against a real `index_k_with_scale_buffer`-shaped
  buffer. Skips if `can_use_dsa_fused_store(...)` returns False
  or the JIT module import fails.
- `_try_fallback_path` — `act_quant(K, block_size=128)` →
  `SetKAndS.execute(pool=SimpleNamespace(page_size=64), buf=...,
  loc=..., index_k=fp8, index_k_scale=scale)` against the same
  buffer layout. Same readback helper.
- Runs every path that successfully executes; `skipTest` only
  when NEITHER runs. Artifact records `path_used` per branch +
  `per_path_verdict` + combined verdict.

### Fix 6 — `build_request_capture` per-token regressions (+6)

`TestBuildRequestCapture` in `test_double_sparsity_unit.py`:

- `test_per_token_slot_sha_lengths_match_prompt_len`.
- `test_compare_cached_prefix_first_position_diff`.
- `test_compare_cached_prefix_zero_cached_tokens_no_overlap`.
- `test_per_token_slot_sha_deterministic_across_calls`.
- `test_compare_cached_prefix_label_divergence_named_layer`.
- `test_compare_cached_prefix_clamps_to_shorter_capture`.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py`:
  per-token SHAs + `compare_cached_prefix`.
- `python/sglang/srt/layers/attention/dsa_backend.py`:
  `_ds_radix_publish_extend_snapshot` helper + call from
  `_write_token_labels`.
- `python/sglang/srt/models/deepseek_v2.py`: removed Round-37
  capture branch.
- `test/manual/_m3b_label_capture_verdict.py`: dict/list dual
  shape acceptance, uses `compare_cached_prefix`.
- `test/manual/test_dsv32_fp8_scale_stability.py`: fused +
  fallback production paths; shared readback helper.
- `test/registered/unit/manual/test_m3b_label_capture_verdict.py`:
  4 transport integration + 9 verdict negatives.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  +6 per-token / compare_cached_prefix regressions.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
397 passed, 26 subtests passed (was 389 + 26; +8 named)

PYTHONPATH=python pytest test/registered/unit/manual/test_m3b_label_capture_verdict.py -v
13 passed (4 transport integration + 9 verdict negatives)

env -u DS_BASE_URL -u SGLANG_DS_RADIX_FIXTURE_CAPTURE -u SGLANG_DS_FP8_SCALE_PROOF \
  PYTHONPATH=python pytest \
    test/manual/test_dsv32_radix_cache_fixture.py \
    test/manual/test_dsv32_radix_label_capture_fixture.py \
    test/manual/test_dsv32_fp8_scale_stability.py -q
3 skipped (all manual fixtures skip cleanly when env unset)
```

The drift-recovery integration test (`test_meta_info_transport_
dict_shape_passes_verdict`) reproduces the EXACT scheduler-side
unwrap that converted the producer's per-batch list into a
per-request dict — and asserts the verdict helper now accepts that
dict shape. The same helper is used by the manual hardware
fixture, so the dict-shape transport path is guarded under CI.

Commit: `ccbdac7c5` — [AC-10] M3-B drift recovery: transport
shape, post-write timing, per-token API, FP8 fallback.

## Recovery Verdict

Recovery is in CI: every Round-37-review claim is now exercised
either as a direct regression (slot divergence at position 2; layer
SHA divergence at layer 1 position 3) or as an integration test
(dict-shape transport unwrap → verdict). Failure modes are loud:

* Empty / wrong-shape capture → FAIL with "capture missing".
* `cached_tokens == 0` → FAIL with "radix cache was not exercised".
* Slot divergence within `cached_tokens` → FAIL naming the
  position.
* Per-layer per-token label divergence → FAIL naming layer +
  position.
* Either side's `written_all_true` False → FAIL naming the bad
  layers.

The FP8 fixture exercises BOTH production paths that
`_store_index_k_cache` uses on real hardware; CPU-only runs that
cannot execute the production kernel still skip cleanly.

## Remaining Items

AC-10 hardware execution (operator-driven):

1. Boot DS server with `SGLANG_DS_RADIX_OVERRIDE=1` +
   `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`, removing
   `--disable-radix-cache` for this one-shot run.
2. Run `test_dsv32_fp8_scale_stability.py` with
   `SGLANG_DS_FP8_SCALE_PROOF=1`. Verify BOTH paths PASS.
3. Run `test_dsv32_radix_label_capture_fixture.py`. Verify the
   `meta_info["double_sparsity_radix_capture"]` records non-zero
   `cached_tokens` AND verdict PASS.
4. Wire `record_radix_fixture_passed(server_args, artifact_path=
   "<label-capture-artifact.json>")` into a launcher init module
   BEFORE `validate_double_sparsity` runs.
5. Delete the `--disable-radix-cache` line in
   `serve_double_sparsity.sh` (`# AC-10-FIXTURE-MARKER` names the
   line).
6. Update `test_ds_server_does_disable_radix_cache_until_ac10`
   for the post-AC-10 expectation.

After AC-10 closes, `task-ac11-compare` H200 sweep runs.

Other hardware-gated tasks (unchanged): `task-ac1-hwtest`,
`task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
`task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
`task-ac12-quality`.

Queued cleanup (unchanged):
- AC-8 prefix-match helper regression coverage cleanup.
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

## Push-to-remote Status

Branch is 39 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: The drift-cause this round (producer/transport/consumer
mismatch from incremental patching) is a candidate for a future
generalized lesson ("when adding a server→client side-channel,
ship the CPU integration test FIRST and the producer side
SECOND"), but the pattern is too broad to phrase as a single
problem→solution entry. The specific failure modes are already
captured as registered regressions, which are the more durable
guardrail.
