# Round 36 Contract

## Mainline Objective

Close Codex Round 35's AC-10 mainline gap: the manual M3-B fixture
from Round 35 is scaffolding, not the M3-B-conformant evidence plan
§AC-10 / §303 requires. Five concrete defects must close, and the
operator-facing capture path must actually produce direct label
evidence rather than a continuation-only proxy.

Hardware execution of the fixture + the post-pass guard flip remain
operator-driven (out of scope for any CPU-only loop round). What
this round produces is the *infrastructure* that lets the operator
collect honest evidence on H200.

## Target ACs

- **AC-10** (STRETCH) — radix cache ON under DS, conditioned on the
  M3-B page-stability fixture having recorded a pass with **direct
  label evidence** for the running configuration.

## Required Implementation

### Fix 1: Identical prompts + honest naming + commit_sha (existing fixture)

`test/manual/test_dsv32_radix_cache_fixture.py`:

- Remove the `pass_id="cold"` / `pass_id="warm"` template
  substitution. Both passes use the SAME prompt text; the
  cold/warm distinction lives in the artifact metadata (and
  optionally in the HTTP request `id`/`stream_options` payload),
  NOT the prompt itself. Otherwise the radix cache cannot reuse
  slots and the continuation comparison conflates prompt change
  with label change.
- Rename the test from
  `test_cold_then_warm_continuation_is_bit_equal` to
  `test_cold_warm_continuation_smoke` and mark its assertion as a
  necessary-but-not-sufficient pre-flight check (passing
  continuations imply the high-temperature divergence smell is
  absent, but bit-equal labels require the capture-based fixture
  added below).
- Capture `commit_sha` from `/get_server_info`'s `server_args`
  (operator records it on the server side) and from the local
  `git rev-parse HEAD`; record both in the artifact.
- Module docstring is rewritten to make the two-fixture layout
  explicit: this file is the *continuation smoke* fixture; the
  full M3-B-conformant capture fixture lives in the new module
  added in Fix 4.

### Fix 2: Server-side capture primitive

`python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py`
(NEW):

- Module-level `_CAPTURE_LOG: list[dict]` holding per-write
  records: `{layer_id, cache_loc_sha, k_nope_sha, written_after}`.
- `is_capture_enabled() -> bool` — reads
  `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`.
- `record_write(layer_id, cache_loc, k_nope, written)` —
  no-op when disabled. When enabled, computes
  `hashlib.sha256(cache_loc_int_bytes)` and
  `hashlib.sha256(k_nope_fp32_bytes)` per write and appends a
  record.
- `record_table_snapshot(table, slots, *, label="snapshot")` —
  computes `[L]` per-layer SHA256 of `signatures[L, slots, :, :]`
  + `written[L, slots]` and records as one entry. This is the
  "after-forward" snapshot the fixture reads to compare cold vs
  warm.
- `get_log() -> list[dict]` and `clear_log()` — accessors.
- Pure Python; no torch dependency beyond `.cpu().numpy().tobytes()`
  in the snapshot path. Fully CPU-testable.

### Fix 3: Wire capture into `_write_token_labels`

`python/sglang/srt/layers/attention/dsa_backend.py`:

- After `token_label_write(...)` in `_write_token_labels`, when
  `radix_fixture_capture.is_capture_enabled()` is True, call
  `record_write(layer_id, cache_loc, k_nope, written_after=
  written[layer_id, cache_loc].clone())`.
- Default (env unset) path is unchanged — no overhead.

### Fix 4: Capture-aware manual fixture

`test/manual/test_dsv32_radix_label_capture_fixture.py` (NEW):

- Class-level skip on unset `DS_BASE_URL` (matches AC-12 / Round
  35 pattern).
- Additional class-level skip on unset
  `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` — the test is meaningless
  without the server-side capture wiring active.
- One paired-request test:
  - Best-effort `/flush_cache`.
  - Send the SAME prompt twice (`request.id` or trailing
    `metadata` field differentiates cold/warm for the artifact).
  - Between requests, query an operator-provided helper endpoint
    or read the capture log directly via in-process import (the
    fixture's runbook shows both paths).
  - Compute the set of layers + slots that were written for the
    shared-prefix range; assert the per-layer SHA256 hashes are
    bit-equal across cold and warm. Assert `written_after` is
    True for every slot in both passes.
  - Assert the second request's `cached_tokens` (from server
    meta_info) is > 0 — the radix cache actually reused slots.
    Otherwise the test only proves "two identical writes produce
    identical labels", which the CPU unit test already establishes.
- Artifact at
  `development/results/dsv32_radix_label_capture_<ts>.json`:
  `commit_sha`, `server_args`, `prompt`, `cached_tokens`,
  per-layer label hashes, `written_after` summary, verdict.

### Fix 5: FP8 block scale-factor stability hardware-gated proof

`test/manual/test_dsv32_fp8_scale_stability.py` (NEW):

- Class-level skip when neither (a) CUDA available with FP8
  support, nor (b) the user has set `SGLANG_DS_FP8_SCALE_PROOF=1`.
- Imports the production FP8 quantization kernel sglang uses for
  KV cache and exercises it for:
  - The same input K row written as a singleton (1-token batch),
    versus
  - The same input K row as one token in a packed block
    (e.g. 64-token batch with deterministic neighbors).
- Compares the resulting per-block scale factor bytes for that
  token's block. Asserts equality (or documents the failure
  mode explicitly so the operator keeps `--disable-radix-cache`
  in the launcher).
- Artifact recorder + clear pass/fail reporting.

### Fix 6: CPU unit tests for the capture primitive

`test/registered/unit/layers/attention/test_double_sparsity_unit.py`:

- `TestRadixFixtureCapture` (5+ tests):
  - `record_write` no-op when env unset.
  - `record_write` appends one record per call when env set.
  - `cache_loc_sha` + `k_nope_sha` deterministic for identical
    inputs (foundation of the bit-equality check).
  - `cache_loc_sha` differs across different cache_loc.
  - `k_nope_sha` differs across different K-noPE.
  - `record_table_snapshot` per-layer hash differs across
    differently-written tables and equals across identically-
    written tables.
  - `clear_log` resets the module state.

### Fix 7: Extend `record_radix_fixture_passed` to capture fixture artifact

`python/sglang/srt/layers/attention/double_sparsity/validator.py`:

- Extend the helper signature to
  `record_radix_fixture_passed(server_args, *, artifact_path:
  Optional[str] = None)`.
- When `artifact_path` is provided, the WARNING log line includes
  the artifact path + SHA256 of the artifact contents. Provides
  the audit trail Codex Round 35 review asked for.
- Update the validator regression test to exercise both code paths.

### Fix 8: Document operator runbook for the post-pass flip

Inline in `test/manual/test_dsv32_radix_label_capture_fixture.py`'s
docstring + `serve_double_sparsity.sh` trailing comment: the
operator-facing flip sequence post-AC-10 passage is:

1. Run the FP8 scale proof + the capture fixture; verify both
   pass and persist the artifacts.
2. In a launcher wrapper or a small init module, call
   `record_radix_fixture_passed(server_args, artifact_path=
   "...")` BEFORE `validate_double_sparsity` runs.
3. Delete the `--disable-radix-cache` line in
   `serve_double_sparsity.sh` (AC-10-FIXTURE-MARKER comment
   points at it).
4. Update `test_ds_server_does_disable_radix_cache_until_ac10`
   to the post-flip expectation.

Steps 2-4 still require the fixture to have actually passed on
real H200; this round provides the runnable + auditable
infrastructure to make that happen.

## Tests

- Existing 362 tests + 26 subtests must still pass.
- ~6-8 new regressions (capture primitive unit tests + updated
  validator helper test).
- Expect ≥ 370 passed.

## Success Criteria

1. The continuation-smoke fixture sends IDENTICAL prompts (grep
   for `pass_id` returns nothing in the prompt template).
2. The capture primitive produces a non-empty log when enabled and
   an empty log when disabled; CPU unit tests exercise both paths.
3. The capture-aware fixture skips cleanly when
   `DS_BASE_URL` and/or `SGLANG_DS_RADIX_FIXTURE_CAPTURE` is
   unset.
4. The FP8 scale-stability fixture skips cleanly when CUDA / FP8
   is unavailable.
5. `record_radix_fixture_passed(args, artifact_path="...")` logs
   the artifact path + SHA in the WARNING line; the existing
   validator regression test still passes.
6. `pytest test/registered -q` ≥ 370 passed.

## Blocking Issues

None separate from the AC-10 mainline gap. The fixture
inadequacy *is* the mainline; AC-11 cannot run honestly until
the M3-B fixture produces direct label + FP8 scale evidence.

## Queued (out of scope for Round 36)

- AC-10 hardware execution itself (operator action on H200; the
  whole point of this round is to make that execution honest).
- AC-11 H200 3-trial DSA + DS sweep (waits on AC-10 pass).
- All other hardware-gated tasks (`task-ac1-hwtest`,
  `task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
  `task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
  `task-ac12-quality`).
- AC-8 prefix-match helper regression cleanup, stale DS comments
  (queued tracker items).
