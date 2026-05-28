# Round 32 Contract

## Mainline Objective

Close the three remaining AC-11 comparator defects Codex flagged in the
Round 31 review and the one blocking side issue:

1. **Timing floors are sidecar-only.** A JSONL with `duration: 5` and
   a sidecar claiming `measurement_window_seconds: 600` still passes.
   The comparator must read the JSONL `duration` and refuse runs whose
   real measurement window is below the AC-11 floor.
2. **Sidecar validation treats missing fields as agreement.** Two
   sidecars that both omit `seed`/`commit_sha`/`server_args` compare
   equal under `None == None`. Required fields must be enforced
   present-and-well-typed in `_read_ac11_meta`.
3. **Radix mismatch allowance is wrong.** Round 31 added
   `disable_radix_cache` to `_DS_ONLY_SERVER_ARG_KEYS` and filtered
   the radix mismatch out of `_match_or_refuse`. Plan §AC-11 says only
   `--enable-double-sparsity` + `--double-sparsity-config` may differ,
   and AC-11 depends on AC-10 removing `--disable-radix-cache` from
   the DS launcher. Tighten back to refuse the radix mismatch.
4. **Dynamic `/server_info` fields self-refuse.** `internal_states.
   last_gen_throughput`, memory usage, step times, etc. drift between
   trials. Round 31's normalization keeps them; sidecars from two
   sequential trials end up disagreeing on inert metrics. Normalize
   to a stable launch-args subset.

## Target ACs

- **AC-11** — comparator enforces real 600s measurement window from
  the JSONL `duration` field; rejects sidecars missing required
  reproducibility fields; refuses radix-cache mismatch; tolerates
  dynamic `/server_info` fields that drift between sequential trials.

## Required Implementation

### Fix 1: Extract + validate JSONL `duration` (Mainline Gap 1, subset)

`development/benchmark_compare.py`:

- Extend `RunMetrics` with `duration_s: Optional[float]`.
- `_read_bench_jsonl` reads `summary.get("duration")` (bench_serving
  emits this as the wall-clock of the measured phase).
- New validator `_validate_jsonl_duration(metrics, side, path)`:
  refuses `duration_s is None` or
  `duration_s < AC11_MIN_MEASUREMENT_WINDOW_SECONDS` with a clear
  message.
- Wired into `_run_ac11_mode` per trial, alongside
  `_validate_trial_metrics` / `_validate_meta_floors`.

Note: bench_serving CLI changes (the actual `--warmup-seconds` /
`--measurement-window-seconds` flags + warmup loop) are out of
Round 32's scope — they require upstream sglang changes. The
comparator-side floor on JSONL `duration` is the safety net that
catches operator mistakes where `num_prompts` was too small.

### Fix 2: Required-field sidecar validator (Mainline Gap 2)

`development/benchmark_compare.py`:

- New helper `_require_sidecar_fields(meta, side, path)`:
  - `seed`: must be `int`.
  - `commit_sha`: must be a non-empty `str`, not `"unknown"` or empty.
  - `chunked_prefill_size`: must be present (int OR `"unknown"`
    explicitly — but `None`/missing refuses).
  - `num_prompts`, `isl_total_tokens`, `osl_tokens`: positive ints.
  - `server_args`: non-empty `dict`.
  - `warmup_seconds`, `measurement_window_seconds`: floats already
    enforced ≥ floors by `_validate_meta_floors`; just require the
    fields are present (covered there).
- Called inside `_read_ac11_meta` before returning the dict.
- Sidecar concurrency must match the JSONL concurrency group:
  `meta["concurrency"] == grouping_concurrency`.

### Fix 3: Remove radix-cache allowance (Mainline Gap 3)

`development/benchmark_compare.py`:

- Remove `"disable_radix_cache"` from `_DS_ONLY_SERVER_ARG_KEYS`.
- Stop filtering `disable_radix_cache` mismatch reasons from
  `_match_or_refuse` output (`hw_reasons = [r for r in hw_reasons
  if "disable_radix_cache" not in r]` → just `hw_reasons`).
- Update Round 31's `test_allowed_ds_only_differences_still_pass`
  fixture: BOTH sides must have radix off (or both on). Add a
  separate negative regression
  `test_radix_mismatch_refused` that asserts exit 2 when DSA radix
  on vs DS radix off.

### Fix 4: Stable server-args normalization (Blocking Side Issue)

`development/benchmark_compare.py`:

- Extend `_normalize_ac11_server_args(meta)` to strip dynamic
  endpoint fields that aren't launch args:
  - Top-level keys: `internal_states`, `kv_events`, `step_time`,
    `last_gen_throughput`, `gpu_memory_used_bytes`,
    `gpu_memory_free_bytes`, `mem_usage`, `request_count`,
    `cumulative_*`, anything matching `*_throughput*`, `*_count`,
    `*_bytes`, `*_seconds`, `*_ms`, `last_*`, `cumulative_*`.
  - Use a strict whitelist instead: define the stable launch-arg
    keys we DO compare (`tp_size`, `page_size`, `model_path`,
    `chunked_prefill_size`, `dsa_prefill_backend`,
    `dsa_decode_backend`, `disable_overlap_schedule`,
    `disable_piecewise_cuda_graph`, `kv_cache_dtype`,
    `enable_double_sparsity`, `double_sparsity_config`,
    `disable_radix_cache`). Reduce comparison surface to these
    fields modulo `_DS_ONLY_SERVER_ARG_KEYS`.
- Whitelist approach is more robust than blocklist as
  `internal_states` schema evolves across sglang versions.

### Fix 5: Update Round 31 fixture + add new regressions

`test/registered/unit/development/test_ac11_comparator.py`:

- Default `_write_bench_jsonl` sidecar gains a default
  `duration` of `600.0` (above floor) so existing tests still pass.
  Tests that exercise the duration validator pass `duration=5.0`
  via `extra=`.
- Existing `test_allowed_ds_only_differences_still_pass` updated to
  use radix-OFF on BOTH sides (the legitimate AC-10-passed state)
  → still exit 0.
- New regressions:
  - `test_short_jsonl_duration_refused` — JSONL duration=5 + sidecar
    timing fine → exit 2 ("duration < AC11_MIN_MEASUREMENT_WINDOW").
  - `test_missing_jsonl_duration_refused` — JSONL has no duration
    field → exit 2.
  - `test_radix_mismatch_refused` — DSA radix on / DS radix off
    → exit 2.
  - `test_missing_required_sidecar_field_refused` (parametrized over
    seed / commit_sha / chunked_prefill_size / num_prompts /
    isl_total_tokens / osl_tokens / server_args) → each missing field
    → exit 2.
  - `test_server_args_empty_dict_refused` — `server_args == {}`
    → exit 2.
  - `test_dynamic_server_info_drift_does_not_self_refuse` — two
    trials' sidecars differ only in
    `server_args.internal_states[0].last_gen_throughput` → exit 0.
  - `test_chunked_prefill_size_unknown_string_allowed_when_consistent` —
    both sides have `chunked_prefill_size: "unknown"` (the
    `_bench_meta_writer` fallback) → exit 0.

### Fix 6: `_filename_concurrency` recognizes `_c<N>_t<M>.jsonl` (Queued Side Issue)

`development/benchmark_compare.py`:

- Update the regex from `r"_c(\d+)\.jsonl$"` to
  `r"_c(\d+)(?:_t\d+)?\.jsonl$"` so the Round 31 trial-suffix
  filenames are handled.

## Tests

- Existing 322 tests must still pass.
- ~10 new regressions for the validation tightening + tolerance.
- Expect ≥ 332 passed total.

## Success Criteria

1. Codex's short-duration reproducer (JSONL `duration: 5` + sidecar
   `measurement_window_seconds: 600`) now exits 2 with a "duration
   below AC-11 floor" message.
2. Codex's missing-fields reproducer (sidecars with only timing +
   `server_args_error: null`, no `seed`/`commit_sha`/etc.) now
   exits 2 naming the first missing required field.
3. Two sidecars differing ONLY in
   `server_args.internal_states[0].last_gen_throughput` produce exit 0.
4. `pytest test/registered -q` ≥ 332 passed.

## Blocking Issues

None.

## Queued (out of scope for Round 32)

- bench_serving CLI changes (`--warmup-seconds` flag + warmup loop +
  measurement-window enforcement) — requires upstream sglang
  edits, separate round.
- Shallow AC-8 prefix-match regression coverage.
- Stale DS bind/runtime comments + token-label lifetime docs.
- All hardware-gated execution.
