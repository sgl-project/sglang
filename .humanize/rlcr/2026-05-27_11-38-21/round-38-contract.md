# Round 38 Recovery Contract

## Drift / Stagnation Root Cause

Rounds 35 → 37 iteratively patched the M3-B label-capture fixture but
never verified the full producer→transport→consumer contract
end-to-end. Each round Codex review found a NEW seam that broke the
chain:

* Round 35 — continuation-only proxy.
* Round 36 — capture log lived in a separate client process.
* Round 37 — capture published as a list under
  `summary["double_sparsity_radix_capture"]`, but the existing
  per-request summary transport in
  `_maybe_collect_per_request_summary` unwraps `v[i]` per request, so
  the client receives a **dict per request** in `meta_info`. My
  verdict helper treats anything non-list as missing → false fails
  on the actual response shape. Additionally the capture fires from
  `_publish_ds_request_summary` (called per layer at the START of
  selection, BEFORE writes), so it snapshots stale labels for the
  current layer, and slices `req_to_token[:seq_lens[b]]` which
  includes decode-generated tokens, not just the cached prefix.
  Finally, the FP8 fixture only implemented the fused store path
  and `skipTest`d when `can_use_dsa_fused_store(...)` returned False
  instead of running the production fallback (`act_quant` +
  `set_index_k_scale_buffer`).

The drift is process-level: I patched what each review named without
running a transport integration test in CI. Round 38 fixes the chain
**and adds a CPU integration test that exercises the full path**, so
future shape/timing mismatches are caught locally instead of by
remote review.

## Mainline Objective

Make the AC-10 M3-B label-capture fixture deliver honest direct
evidence on H200: correct response shape, correct snapshot timing
(after writes), correct slot range (cached prefix only), AND the FP8
fallback path. Lock the producer→transport→consumer chain with a CPU
integration test that simulates the actual transport.

## Target ACs

- **AC-10** — radix-cache stability fixture passes against a real
  H200 server with both fused and fallback FP8 store paths; the
  capture survives the production per-request summary transport.

## Required Implementation

### Fix 1: Verdict helper accepts the real transported shape (dict)

`test/manual/_m3b_label_capture_verdict.py`:

- `_records(capture)` returns `[capture]` when `capture` is a dict
  (the actual transported per-request shape) and `capture` itself
  when it is a list (legacy shape for direct unit tests). Treat
  None and other shapes as missing.
- Add a docstring explaining the dual shape: dict ← production
  transport (per-request unwrap); list ← direct helper tests.

### Fix 2: Move capture publishing to post-write, extend-only

`python/sglang/srt/layers/attention/dsa_backend.py::_write_token_labels`:

- After `token_label_write(...)` returns, when capture is enabled
  AND `forward_batch.forward_mode.is_extend()` (skipping decode),
  publish a per-request capture into
  `forward_batch.ds_per_request_summary["double_sparsity_radix_capture"]`
  using `radix_fixture_capture.build_request_capture(...)`. Each
  call overwrites the previous — the LAST DS layer's call wins, so
  the published capture reflects the table state after ALL DS
  layers have written this forward's labels.
- Read `req_to_token`, `req_pool_indices`, and `seq_lens` from
  `forward_batch`. For prefill extend, `seq_lens[b]` is the full
  prompt length (cached prefix + suffix), which is the slot range
  the cold/warm fixture needs to compare.

`python/sglang/srt/models/deepseek_v2.py::_publish_ds_request_summary`:

- **Remove** the Round-37 radix-capture branch. It fires at the
  wrong point (before writes) and on the wrong mode (every layer,
  including decode).

### Fix 3: `build_request_capture` exposes per-token slot SHAs

`python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py`:

- Extend the per-request record with a NEW field
  `per_token_slot_sha: list[str]` — SHA256 of each prompt
  position's slot index. This lets the fixture compare just the
  first `cached_tokens` positions across cold and warm (the cached
  prefix), instead of comparing one aggregate `slots_sha` over the
  full prompt range. Extra generated-token slots in either capture
  no longer force a false mismatch.
- Keep `slots_sha` for back-compat (it equals the SHA over the full
  slice).
- New helper `compare_cached_prefix(cold_rec, warm_rec,
  cached_tokens) -> dict` returns `{first_diverging_position,
  ok}` for the first `cached_tokens` positions. The verdict helper
  uses this for the cold/warm comparison.

### Fix 4: Verdict helper uses cached-prefix comparison

`test/manual/_m3b_label_capture_verdict.py`:

- Replace the `slots_sha` equality check with a call to
  `compare_cached_prefix(cold, warm, cached_tokens)`. PASS requires
  the cached prefix slots AND label hashes to match position-by-
  position for the first `cached_tokens` positions. Extra
  generated-token slots in either capture are ignored.

### Fix 5: Producer→transport→consumer CPU integration test

`test/registered/unit/manual/test_m3b_label_capture_verdict.py`:

- New `test_meta_info_transport_dict_shape_passes_verdict`:
  - Build a per-batch list `[record]` representing the producer
    output for a single-request batch.
  - Simulate `_maybe_collect_per_request_summary` by extracting
    `v[0]` — yields a dict.
  - Wrap in `meta_info = {"double_sparsity_radix_capture": dict,
    "cached_tokens": 10}`.
  - Feed straight to the verdict helper through the same accessor
    the manual fixture uses. Assert PASS.
- New `test_meta_info_transport_dict_shape_with_extra_decode_slots`:
  - Cold capture: 5 prompt positions, label SHAs match warm's.
  - Warm capture: 5 prompt positions + 3 decode-generated slot
    positions (different per-token slot SHAs at positions 5..7).
  - cached_tokens=5 → PASS (the first 5 positions agree; later
    positions are ignored).
- New `test_meta_info_transport_dict_shape_with_cached_prefix_diverging`:
  - Cold and warm capture differ at position 2 within the first
    `cached_tokens=5` → FAIL naming position 2.

### Fix 6: FP8 fallback path

`test/manual/test_dsv32_fp8_scale_stability.py`:

- Add a `_run_fallback_path(K_packed, K_singleton)` branch that
  performs `act_quant(key, block_size, scale_fmt)` + writes via
  `set_index_k_scale_buffer` / `SetKAndS.execute` against a real
  `index_k_with_scale_buffer`-shaped pool buffer. Reads back via
  the production accessor (`GetKAndS.execute`).
- Try `fused_store_index_k_cache` first. If `can_use_dsa_fused_store`
  returns False or the JIT module fails to compile, fall back to
  the `act_quant`+`set_index_k_scale_buffer` path. `skipTest` only
  when neither path can run.
- Record `path_used` in the artifact: `fused_store_index_k_cache`
  or `fallback_act_quant_set_index_k_scale_buffer`.
- Factor a `_read_back_k0_bytes(buf, page_idx, page_size,
  index_head_dim)` helper that uses the same byte offsets as the
  production accessor.

### Fix 7: CPU regressions for `build_request_capture` per-token API

`test/registered/unit/layers/attention/test_double_sparsity_unit.py`:

- `test_per_token_slot_sha_lengths_match_prompt_len`.
- `test_compare_cached_prefix_position_zero_diff` (first-mismatch
  hint names the position).
- `test_compare_cached_prefix_zero_cached_tokens_treated_as_no_overlap`.

## Tests

- Existing 389 tests + 26 subtests must still pass.
- ~8-10 new regressions covering transport, per-token API, and
  fallback behavior.
- Expect ≥ 397 passed.

## Success Criteria (would change verdict to ADVANCED)

1. CPU integration test simulates the actual transport
   (`per_request_summary={"k": [record]}` → `meta_info["k"] =
   record` dict) and the verdict helper PASSes it. The test
   reproduces the exact transport unwrap from
   `_maybe_collect_per_request_summary`.
2. The capture is emitted from `_write_token_labels` (post-write,
   extend-only), not from `_publish_ds_request_summary`.
3. `build_request_capture` exposes `per_token_slot_sha`; the
   verdict compares only the first `cached_tokens` positions; extra
   decode-generated slot positions do NOT cause a false mismatch.
4. FP8 fixture has both fused and fallback branches; skips only
   when neither can run.
5. `pytest test/registered -q` ≥ 397 passed.

## Blocking Issues

None separate from the AC-10 mainline gaps above. The transport-
shape and snapshot-timing defects directly block AC-10 and AC-11.

## Queued (out of scope for Round 38)

- Hardware execution of the M3-B fixtures on H200.
- Post-pass guard flip (`record_radix_fixture_passed` before
  `validate_double_sparsity`, launcher edit). Waits on hardware
  pass.
- All other hardware-gated original-plan tasks.
- AC-8 prefix-match helper regression cleanup, stale DS comments,
  stale `token_label_table.py` lifetime docs.
