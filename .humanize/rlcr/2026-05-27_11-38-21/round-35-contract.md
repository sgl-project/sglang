# Round 35 Contract

## Mainline Objective

Land the AC-10 (`task-ac10-radix`) code-tier work: the M3-B
radix-cache fixture harness + the operator-guard handshake that
lets a verified pass flip `--disable-radix-cache` off in the DS
launcher. Hardware execution remains gated by H200 availability;
this round produces a runnable fixture (matching the AC-12 pattern)
+ a CPU unit-level proof of the label bit-stability property AC-10
depends on.

## Target ACs

- **AC-10** (STRETCH) — radix cache ON under DS, conditioned on the
  M3-B page-stability fixture having recorded a pass for the running
  configuration.

## Required Implementation

### Fix 1: CPU unit-level proof of DS label determinism

`test/registered/unit/layers/attention/test_double_sparsity_unit.py`:

- New `TestAC10RadixCacheLabelBitStability` class with 3 tests:
  - `test_token_label_write_is_deterministic_for_same_kv_input` —
    write at a fixed `cache_loc` with K-noPE `x`; capture
    `signatures` row. Write again with the same K-noPE; assert
    the second row is bit-equal to the first.
  - `test_invalidate_then_rewrite_same_input_yields_equal_labels` —
    write `x` at slot `S`; `invalidate_token_label_slots(S)`;
    re-write `x` at `S`; assert the new signatures row is bit-equal
    to the original (simulates radix-cache reuse: same prefix
    token re-allocated to the same physical slot must produce the
    same label).
  - `test_different_kv_input_yields_different_labels` (negative
    counterpart): writing different K-noPE at the same slot must
    produce a different label, so the bit-equality proof above is
    actually testing label-derivation determinism rather than a
    constant.

These are the CPU-runnable proof that the DS label-write code is
deterministic given input — the property AC-10's hardware fixture
must verify end-to-end against real FP8 dequant.

### Fix 2: Hardware-gated M3-B fixture harness

`test/manual/test_dsv32_radix_cache_fixture.py` (NEW):

- Class-level skip if `DS_BASE_URL` is unset (same pattern as
  `test/manual/test_double_sparsity_v32.py`).
- One paired request fixture:
  - Request A (cold prefix): unique shared-prefix payload that the
    server has never seen. Triggers fresh KV-slot allocation + DS
    label writes.
  - Request B (warm prefix): same shared-prefix payload re-sent
    after Request A. With radix cache ON, the server reuses the
    shared-prefix KV slots; without radix cache (current state),
    new slots are allocated and labels are re-written from scratch.
  - Assert response equality on the generated continuation (proxy
    for label equality — if labels diverged, DS selection diverges
    and continuations diverge under temperature=0).
- Records artifact at
  `development/results/dsv32_radix_fixture_<ts>.json` carrying
  commit_sha, server `disable_radix_cache` value, request payloads,
  continuations, and pass/fail verdict.
- Operator runbook in module docstring: after pass, set
  `server_args._double_sparsity_radix_fixture_passed = True`
  via the helper (Fix 3) and the DS validator will accept
  radix-on at boot.

### Fix 3: Operator-facing helper to record fixture passage

`python/sglang/srt/layers/attention/double_sparsity/validator.py`:

- New `record_radix_fixture_passed(server_args)` helper that sets
  the `_double_sparsity_radix_fixture_passed = True` attribute and
  logs a one-line WARNING (audit trail: "DS radix-cache fixture
  recorded as passed; ServerArgs guard set"). The existing
  validator already reads this attribute and refuses boot if False;
  the helper just makes the operator call explicit and grep-able.
- Add a registered unit test that asserts the validator now boots
  with `disable_radix_cache=False` after the helper is called.

### Fix 4: Launcher contract test parameterized over the guard state

`test/registered/unit/development/test_option_b_scripts.py`:

- The current `test_ds_server_does_disable_radix_cache_until_ac10`
  passes; keep it (pre-AC-10 state).
- New `test_ds_server_has_post_ac10_marker_comment` — asserts
  `serve_double_sparsity.sh` contains a marker comment
  (`# AC-10-FIXTURE-MARKER`) above the `--disable-radix-cache`
  line so the post-AC-10 launcher edit is mechanical (operator
  removes the flag line; the marker stays for future audits).

### Fix 5 (queued cleanup): script plan-marker leftovers

`development/benchmark.sh` + `development/benchmark_baseline.sh`:

- Replace the `Round 33 (AC-11): refuse the run if the observed
  JSONL ``duration`` is below MEASUREMENT_WINDOW_S` comment with a
  neutral version that names the property, not the round.

### Fix 6: Marker comment in serve_double_sparsity.sh

`development/serve_double_sparsity.sh`:

- Add `# AC-10-FIXTURE-MARKER: remove --disable-radix-cache after
  M3-B fixture pass + ServerArgs guard flip.` immediately above the
  `--disable-radix-cache` flag.

## Tests

- Existing 356 tests + 26 subtests must still pass.
- ~5-6 new regressions across unit + script contract suites.
- Expect ≥ 362 passed.

## Success Criteria

1. `pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py
   -k TestAC10RadixCacheLabelBitStability` passes (CPU-runnable
   AC-10 property proof).
2. `pytest test/manual/test_dsv32_radix_cache_fixture.py` skips
   cleanly when `DS_BASE_URL` is unset (matches AC-12 pattern).
3. After `record_radix_fixture_passed(server_args)` is called, the
   validator boots with `disable_radix_cache=False` without
   `SGLANG_DS_RADIX_OVERRIDE=1`.
4. `grep -nE 'Round 3[0-9]|Codex Round' development/benchmark.sh
   development/benchmark_baseline.sh` returns no matches.
5. `pytest test/registered -q` ≥ 362 passed.

## Blocking Issues

None. AC-10 hardware execution itself blocks the FINAL passage of
AC-10 (and thus AC-11 H200 execution), but the code-tier work for
this round is well-defined and does not depend on hardware.

## Queued (out of scope for Round 35)

- All hardware-gated execution (`task-ac1-hwtest`,
  `task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
  `task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
  `task-ac11-compare`, `task-ac12-quality`).
- AC-8 prefix-match helper regression coverage cleanup.
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.
