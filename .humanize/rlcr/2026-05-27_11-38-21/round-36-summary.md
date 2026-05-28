# Round 36 Summary

## Work Completed

Codex Round 35 review surfaced five concrete defects in the manual
M3-B fixture: it (a) sent DIFFERENT prompts in its cold and warm
passes while asserting continuation equality, (b) asserted only
continuation byte-equality as a proxy for label bit-equality (the
plan requires direct label evidence), (c) did not record
`commit_sha`, (d) had no FP8 scale-stability proof, and (e) gave the
operator no way to actually capture DS label table state. Round 36
closes all five.

### Fix 1 — Server-side capture primitive

New module
`python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py`:

- Module-level `_LOG: list[dict]` guarded by a `threading.Lock`.
- `is_capture_enabled()` reads
  `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`.
- `record_write(layer_id, cache_loc, k_nope, written_after=None)`
  computes SHA256 of `cache_loc.long().contiguous()` bytes and
  `k_nope.float().contiguous()` bytes per write and appends a
  per-write record. No-op early exit when the env flag is off — the
  production hot path pays one `os.environ.get` lookup.
- `record_table_snapshot(signatures, written, slots, label=...)`
  emits per-layer SHA256 of `signatures[L, slots]` and
  `written[L, slots]` so cold/warm comparisons are by-slot,
  by-layer.
- `get_log()` / `clear_log()` accessors.

`python/sglang/srt/layers/attention/dsa_backend.py`: hooked into
`_write_token_labels` after `token_label_write(...)` returns. The
default path (env unset) is unchanged — zero overhead.

### Fix 2 — Identical-prompt continuation-smoke fixture

`test/manual/test_dsv32_radix_cache_fixture.py`:

- Replaced the `pass_id="cold"` / `pass_id="warm"` template with a
  single `_SHARED_PREFIX_PROMPT` constant; both passes use the
  identical string. The radix cache can now actually reuse slots
  across the two requests.
- Renamed the test
  `test_cold_then_warm_continuation_is_bit_equal` →
  `test_cold_warm_continuation_smoke`. Module docstring marks this
  fixture as necessary-but-not-sufficient pre-flight; the
  AC-10-conformant M3-B evidence comes from the new capture
  fixture + FP8 scale fixture.
- Artifact records `local_commit_sha` (from `git rev-parse HEAD`)
  + `server_commit_sha` (from `/get_server_info`'s `server_args`
  when present) per the audit-trail requirement.

### Fix 3 — Capture-aware M3-B label-equality fixture

New `test/manual/test_dsv32_radix_label_capture_fixture.py`:

- Class-level skip unless `DS_BASE_URL` set AND
  `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` set. setUpClass also re-skips
  when the server reports `disable_radix_cache=True` so the test
  cannot accidentally run against the gated configuration it is
  designed to verify.
- Reads the capture log via in-process import (operator co-locates
  fixture with the server process) or via an optional HTTP endpoint
  at `SGLANG_DS_RADIX_CAPTURE_LOG_URL` for remote-server setups.
- Issues two requests with IDENTICAL prompts at temperature=0;
  asserts:
  - `meta_info.cached_tokens > 0` on the warm pass — proves the
    radix cache actually reused slots (otherwise the test only
    re-proves the CPU unit determinism property).
  - Every overlap between cold-pass and warm-pass writes at the
    same `(layer_id, cache_loc_sha)` has matching `k_nope_sha`.
    This is the direct M3-B label bit-equality proof.
- Artifact records commit_sha (local + server), prompt,
  cached_tokens, write counts, mismatches (first 32 logged).

### Fix 4 — Hardware-gated FP8 scale-stability proof

New `test/manual/test_dsv32_fp8_scale_stability.py`:

- Class-level skip unless `SGLANG_DS_FP8_SCALE_PROOF=1` AND CUDA
  with FP8 support is available. Deliberate opt-in — a CPU-only
  run cannot be misread as a passing M3-B check.
- Invokes the production `sglang_per_token_group_quant_fp8` kernel
  for the same K0 row as:
  - Singleton input (1×128, just K0).
  - Packed input (64×128, K0 alongside 63 deterministic
    neighbours).
- Asserts `scale_single[0] == scale_packed[0]` and
  `fp8_single[0] == fp8_packed[0]`. If the kernel's per-block scale
  depends on neighbour tokens, the DS label-write hook would see
  different dequantized K-noPE in cold vs warm paths and the AC-10
  guard MUST stay in place.

### Fix 5 — `record_radix_fixture_passed` artifact audit trail

`python/sglang/srt/layers/attention/double_sparsity/validator.py`:

- Signature extended to
  `record_radix_fixture_passed(server_args, *, artifact_path:
  Optional[str] = None)`.
- When `artifact_path` is supplied, the audit WARNING line records
  the path + SHA256 of its contents. A grep over server logs
  surfaces both the flip event AND the evidence that authorized
  it.
- Three new validator tests cover (a) artifact path + SHA appear in
  the log, (b) no-artifact-path back-compat, (c) unreadable path
  still flips the guard but marks the artifact as
  `<unreadable:...>`.

### Fix 6 — CPU unit tests for the capture primitive (+9)

`TestRadixFixtureCapture` (in the existing DS unit test file):

- `test_record_write_noop_when_env_unset` — production hot path
  pays zero overhead.
- `test_record_write_appends_when_env_set` — full record shape.
- `test_identical_inputs_produce_identical_hashes` — the
  foundational determinism the capture fixture relies on.
- `test_different_cache_loc_produces_different_hash` and
  `test_different_k_nope_produces_different_hash` — sensitivity
  proofs.
- `test_int32_vs_int64_cache_loc_hashes_equal` — dtype stability
  so int32 cold vs int64 warm cannot spuriously disagree.
- `test_snapshot_equals_across_identical_label_writes` and
  `test_snapshot_differs_when_a_layer_row_changes` — per-layer
  hash sensitivity.
- `test_clear_log_resets_state`.

## Files Changed

- `python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py`
  (NEW): capture primitive. ~150 lines.
- `python/sglang/srt/layers/attention/double_sparsity/validator.py`:
  `record_radix_fixture_passed` gains `artifact_path` + SHA audit.
- `python/sglang/srt/layers/attention/dsa_backend.py`: env-gated
  capture hook in `_write_token_labels`.
- `test/manual/test_dsv32_radix_cache_fixture.py`: identical-
  prompt fix, honest naming, commit_sha, docstring rewritten to
  point at the M3-B fixtures.
- `test/manual/test_dsv32_radix_label_capture_fixture.py` (NEW):
  M3-B label-equality fixture.
- `test/manual/test_dsv32_fp8_scale_stability.py` (NEW): FP8
  scale-stability fixture.
- `test/registered/unit/layers/attention/test_double_sparsity_unit.py`:
  +`TestRadixFixtureCapture` (9 tests) +3 validator artifact-path
  tests.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
374 passed, 26 subtests passed (was 362 + 26; +12 named)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
   -k 'TestRadixFixtureCapture or test_record_radix_fixture' -q
12 passed (9 capture primitive + 3 record helper)

env -u DS_BASE_URL -u SGLANG_DS_RADIX_FIXTURE_CAPTURE \
    -u SGLANG_DS_FP8_SCALE_PROOF \
  PYTHONPATH=python pytest \
   test/manual/test_dsv32_radix_cache_fixture.py \
   test/manual/test_dsv32_radix_label_capture_fixture.py \
   test/manual/test_dsv32_fp8_scale_stability.py -q
3 skipped (all three manual fixtures skip cleanly when env unset)

bash -n development/serve_double_sparsity.sh   # OK
```

Commit: `a41b1d952` — [AC-10] M3-B capture primitive + direct-
evidence fixtures + identical-prompt smoke.

## Remaining Items

AC-10 hardware execution (operator-driven, not in this loop):

1. Boot DS server with `SGLANG_DS_RADIX_OVERRIDE=1` +
   `SGLANG_DS_RADIX_FIXTURE_CAPTURE=1` and radix cache ON.
2. Run `test_dsv32_fp8_scale_stability.py` with
   `SGLANG_DS_FP8_SCALE_PROOF=1`. On pass, save the artifact.
3. Run `test_dsv32_radix_label_capture_fixture.py`. On pass, save
   the artifact.
4. (Optional) Run the continuation smoke first as a sanity check.
5. Wire `record_radix_fixture_passed(server_args, artifact_path=
   "<label-capture-artifact.json>")` into a launcher init module
   BEFORE `validate_double_sparsity` runs.
6. Delete the `--disable-radix-cache` line in
   `serve_double_sparsity.sh` (the AC-10-FIXTURE-MARKER comment
   names the exact line).
7. Update `test_ds_server_does_disable_radix_cache_until_ac10`
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

Branch is 37 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 36 followed `BL-20260527-shell-json-into-python-source`
implicitly (the new capture fixture passes JSON-shaped payloads
through env vars + HTTP, not through Python source splicing).
Bitlesson-selector returned `NONE`: the existing entries do not
cover the "env-gated zero-cost capture hook + SHA audit trail"
pattern this round introduces, but the pattern is one-shot
infrastructure for AC-10 specifically rather than a general
guideline. No new entry warranted.
