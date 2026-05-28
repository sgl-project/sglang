# Round 37 Contract

## Mainline Objective

Close the three Round-36-review AC-10 mainline gaps:

1. The M3-B label-capture fixture has a false-pass path: the
   client-process `radix_fixture_capture._LOG` import returns an
   empty list because the SERVER process holds a different copy of
   the module state. The fixture's `verdict = (not mismatches and
   cached_tokens > 0)` therefore PASSes on empty evidence.
2. Even with a real server-side log, a radix-cache HIT skips the
   write path entirely for the warm prefix slots, so write-record
   overlap can never prove warm-prefix label stability. The fixture
   must compare **post-forward snapshots** of the slot ranges (cold
   pass populates; warm pass reuses without rewriting; the snapshot
   bytes must match).
3. The FP8 fixture invokes `sglang_per_token_group_quant_fp8`
   directly. That helper per-row-quantizes — comparing K0 as a
   1-row input vs row 0 of a 64-row input does NOT prove anything
   about the production DSA index-cache page-fill behavior. The
   AC-10 property must be measured through the production
   `_store_index_k_cache` path (`fused_store_index_k_cache` or the
   `act_quant` + `set_index_k_scale_buffer` fallback) writing into
   a real `index_k_with_scale_buffer`-shaped page.

Additionally, the launcher comments still name the smoke fixture as
M3-B evidence; that is operator-visible documentation drift.

## Target ACs

- **AC-10** — direct cold/warm label snapshot equality via
  server-side capture ferried through `meta_info`; FP8 scale
  equality through the production DSA index-cache store path.

## Required Implementation

### Fix 1: Server-side per-request capture via `meta_info`

`python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py`:

- Extend with `build_request_capture(signatures, written,
  req_to_token, req_pool_indices, seq_lens) -> list[dict]` that
  returns, for each request `b`, a dict with `prompt_len`,
  `slots_sha`, `per_layer_label_sha[L]`, `per_layer_written_sha[L]`,
  `per_layer_written_all_true[L]`. Pure, no IO, CPU-testable.

`python/sglang/srt/models/deepseek_v2.py::_publish_ds_request_summary`:

- When `radix_fixture_capture.is_capture_enabled()` is True, look
  up `self.double_sparsity_selector.token_label_table`, derive
  `req_to_token` + `req_pool_indices` + `seq_lens` from
  `forward_batch`, call `build_request_capture(...)`, and attach
  the per-request records to
  `summary["double_sparsity_radix_capture"]` aligned with the
  existing `summary["double_sparsity"]` records.
- Default path (env unset) is unchanged.

The scheduler's existing per-request summary plumbing (already used
for `double_sparsity` stats) ferries the new key to the client as
`meta_info["double_sparsity_radix_capture"]`.

### Fix 2: Capture-aware fixture reads `meta_info`

`test/manual/test_dsv32_radix_label_capture_fixture.py`:

- Drop the in-process / HTTP capture-log paths (client-process
  imports cannot observe server state).
- Read `response["meta_info"]["double_sparsity_radix_capture"]`
  for the cold and warm passes.
- Extract verdict logic into a pure helper
  `_evaluate_m3b_label_capture_verdict(cold_capture, warm_capture,
  cached_tokens) -> {"verdict": "PASS"|"FAIL", "reasons": [...]}`
  in the new client-side helper module
  (`test/manual/_m3b_label_capture_verdict.py`).
- Assertion path: PASS iff
  (a) both captures present and non-empty;
  (b) cached_tokens > 0 on warm;
  (c) cold and warm `slots_sha` match (radix cache reused the
      same physical slots);
  (d) per-layer `label_sha` lists are bit-equal;
  (e) `written_all_true` is True for both.

### Fix 3: False-pass CPU regressions

`test/registered/unit/manual/test_m3b_label_capture_verdict.py`
(NEW):

- 6+ tests of the verdict helper:
  - Both empty captures + cached_tokens=10 → FAIL with "no evidence".
  - Cold empty, warm populated → FAIL.
  - Both populated, cached_tokens=0 → FAIL with "no radix reuse".
  - Both populated, cached_tokens=10, slots_sha mismatch → FAIL.
  - Both populated, slots match, layer hash mismatch on any layer
    → FAIL listing the mismatched layers.
  - All conditions met → PASS.
  - `written_all_true=False` on either side → FAIL.

### Fix 4: FP8 production-path fixture

`test/manual/test_dsv32_fp8_scale_stability.py` (REWRITE):

- Replace direct `sglang_per_token_group_quant_fp8` calls with
  the production `_store_index_k_cache` path:
  - Allocate a real `index_k_with_scale_buffer`-shaped page buffer.
  - Write `K0` at a singleton location (block-start, one token
    only).
  - Write `K0` at a position inside a fully populated page
    (deterministic neighbors filling the rest of the page).
  - Read back the stored scale bytes for K0 via the production
    `index_buf_accessor.py` accessor.
  - Try fused + fallback paths; skip only when neither runs on
    the target hardware.
- Artifact records the path used, page size, cache locations,
  scale bytes (hex), commit SHAs, verdict.

### Fix 5: Launcher comment correctness

`development/serve_double_sparsity.sh`:

- Trailing comment must name the M3-B label-capture fixture +
  the FP8 production-path fixture as the evidence required
  before any guard flip — NOT the continuation smoke.

### Fix 6: CPU unit tests for `build_request_capture`

`test/registered/unit/layers/attention/test_double_sparsity_unit.py`:

- 3+ tests for the new pure helper:
  - Single-request batch: snapshot equals manual
    `signatures[L, slots]` hash.
  - Two-request batch: per-request records align with
    `req_to_token` rows.
  - Identical inputs across two calls produce identical SHAs
    (foundation of the cold/warm equality assertion).

## Tests

- Existing 374 tests + 26 subtests must still pass.
- ~10 new regressions (verdict logic + capture builder).
- Expect ≥ 384 passed.

## Success Criteria

1. CPU regression where capture-log payload is empty + cached_tokens
   > 0 → verdict FAIL (closes the Round-36 false-pass path).
2. Server-side capture builder produces snapshots even when no
   writes happened (warm radix-hit case): the fixture uses
   snapshots, not write-record overlap.
3. FP8 fixture writes through `_store_index_k_cache` and reads
   scale bytes via `index_buf_accessor.py`.
4. Launcher comment names the two M3-B fixtures, not the smoke.
5. `pytest test/registered -q` ≥ 384 passed.

## Blocking Issues

None separate from the AC-10 mainline gaps. The false-pass path IS
the blocker.

## Queued (out of scope for Round 37)

- AC-10 hardware execution (operator-driven).
- Removal of `--disable-radix-cache` + post-pass guard flip wiring
  (waits on hardware verification).
- All other hardware-gated tasks.
- AC-8 prefix-match helper regression cleanup; stale `deepseek_v2.py`
  slot-authority comments; stale `token_label_table.py` lifetime
  docs.
