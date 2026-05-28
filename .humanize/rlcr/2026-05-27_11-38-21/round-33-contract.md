# Round 33 Contract

## Mainline Objective

Close the two AC-11 code-tier gaps Codex flagged in the Round 32
review:

1. **Producer-side timing is still metadata-only.** The plan requires
   "minimum 600s measurement window after a 120s warmup"
   (`development/loop4/refined_plan_v1.md:123-125`). Round 31's
   benchmark scripts only write `WARMUP_SECONDS=120` /
   `MEASUREMENT_WINDOW_S=600` into the sidecar; `bench_serving.py`
   has no `--warmup-seconds` / `--measurement-window-seconds` flag
   and the measured run consumes the finite request generator once.
   AC-11 evidence cannot be produced from these scripts.

2. **Server-args comparison silently under-compares launch flags.**
   Round 32's `_AC11_STABLE_LAUNCH_ARG_KEYS` has only 12 hand-listed
   keys; everything outside it is dropped. Codex's reproducer:
   `disable_cuda_graph=false` on DSA vs `true` on DS still exits 0
   with `AC-11 verdict: PASS`. Plan §AC-11 requires that only
   `--enable-double-sparsity` + `--double-sparsity-config` differ
   between the two columns.

## Target ACs

- **AC-11** — producer scripts run a real 120s warmup + 600s
  measured window; comparator refuses on ANY non-DS `ServerArgs`
  launch-flag mismatch.

## Required Implementation

### Fix 1: bench_serving seconds-based warmup + measured window

`python/sglang/bench_serving.py`:

- New CLI flags:
  - `--warmup-seconds <float>` (default 0 → legacy
    `--warmup-requests` path).
  - `--measurement-window-seconds <float>` (default 0 → legacy
    single-pass behavior).
- Extract the main request-execution loop
  (`python/sglang/bench_serving.py:1348-1415` per Codex's pointer)
  into one reusable `_run_measured_epoch(...)` helper that takes the
  prepared inputs, request-rate scheduling, concurrency semaphore,
  LoRA selection, extra request body, and progress-bar controls,
  and returns `(outputs, elapsed_wallclock_s)`.
- Time-based warmup: while `elapsed < warmup_seconds`, run full
  workload epochs and discard their outputs. Before the first
  measured epoch, reset `random.seed(args.seed)` and
  `np.random.seed(args.seed)` so warmup does not perturb the
  measured request-arrival process.
- Time-based measurement: while `accumulated_measured < window`,
  run measured epochs and accumulate outputs. Compute metrics over
  the accumulated outputs; `duration` is the accumulated measured
  wall-clock.
- Always emit the workload fields the comparator now requires:
  `num_prompts` (count after any repeated epochs), `input_len`,
  `output_len`. Generated-shared-prefix runs must surface the
  effective ISL/OSL even when bench_serving currently derives those
  from `gsp_*` knobs.
- Keep `--warmup-requests` working when `--warmup-seconds` is unset
  (back-compat for non-AC-11 callers).

### Fix 2: `benchmark.sh` + `benchmark_baseline.sh`: wire timing flags

`development/benchmark.sh` and `development/benchmark_baseline.sh`:

- Pass `--warmup-seconds "${WARMUP_SECONDS}"` and
  `--measurement-window-seconds "${MEASUREMENT_WINDOW_S}"` to
  `bench_serving` (kept defaults: 120 / 600).
- After each measured trial, if the JSONL's observed `duration` is
  below `MEASUREMENT_WINDOW_S`, exit non-zero with a clear message
  (catches the case where bench_serving's helper itself bails out
  early; the comparator side already catches this for downstream
  consumers).

### Fix 3: Comparator uses full stable `ServerArgs` projection

`development/benchmark_compare.py`:

- Replace the hand-curated 12-key `_AC11_STABLE_LAUNCH_ARG_KEYS`
  with the full set of `ServerArgs` dataclass field names. Prefer
  `dataclasses.fields(sglang.srt.server_args.ServerArgs)` at module
  load time; fall back to a frozen explicit snapshot of all
  current ServerArgs field names if the import fails (operator
  environment without `PYTHONPATH=python`).
- `_normalize_ac11_server_args` projects onto the full set and
  excludes `_DS_ONLY_SERVER_ARG_KEYS` (still
  `{enable_double_sparsity, double_sparsity_config}`) and any
  `SGLANG_DS_FAULT_INJECT_*` env key.
- Dynamic `/server_info` additions (not `ServerArgs` fields:
  `internal_states`, `kv_events`, scheduler capacity, telemetry,
  step times, etc.) are still dropped — they're not in the
  ServerArgs field set, so they're already excluded by the
  whitelist projection.
- Require the normalized projection to be non-empty AND to include
  the Option B locked fields (`model_path`, `tp_size`,
  `kv_cache_dtype`, `dsa_prefill_backend`, `dsa_decode_backend`,
  `disable_overlap_schedule`, `disable_piecewise_cuda_graph`,
  `page_size`, `disable_radix_cache`, `disable_cuda_graph`) per
  plan §13. Empty / locked-field-missing → exit 2.

### Fix 4: Comparator validates JSONL workload fields

`development/benchmark_compare.py`:

- When `num_prompts` / `input_len` / `output_len` are present in
  the JSONL `summary`, cross-check them against the sidecar's
  `num_prompts` / `isl_total_tokens` / `osl_tokens`. Mismatch →
  refuse (the sidecar must not lie about the workload that
  actually ran).

### Fix 5: Test regressions

`test/registered/unit/development/test_ac11_comparator.py`:

- `test_server_args_non_ds_launch_flag_mismatch_refused` — parametrized
  subTest matrix over `disable_cuda_graph`, `trust_remote_code`,
  `dtype`, `max_total_tokens`, `attention_backend`,
  `mem_fraction_static` → each → exit 2.
- `test_server_args_locked_option_b_field_missing_refused` — drop a
  required Option B field (e.g. `kv_cache_dtype`) from both sides'
  `server_args` → exit 2 ("required Option B field missing").
- `test_jsonl_workload_lies_about_sidecar_refused` — JSONL
  `num_prompts=160` vs sidecar `num_prompts=320` → exit 2.
- Updated `test_dynamic_server_info_drift_does_not_self_refuse`:
  prove that BOTH (a) dynamic non-ServerArgs fields are ignored
  AND (b) real ServerArgs launch-flag mismatch still refuses, by
  using the full ServerArgs projection.

`test/registered/unit/development/test_option_b_scripts.py`:

- `test_ds_bench_passes_warmup_seconds_flag` /
  `test_dsa_bench_passes_warmup_seconds_flag`.
- `test_ds_bench_passes_measurement_window_seconds_flag` /
  `test_dsa_bench_passes_measurement_window_seconds_flag`.
- `test_ds_bench_fails_on_short_observed_duration` /
  `test_dsa_bench_fails_on_short_observed_duration` (script-text
  assertion that the duration guard exists).

`test/registered/unit/development/test_bench_serving_timing.py` (NEW):

- CLI exposes `--warmup-seconds` and
  `--measurement-window-seconds`.
- Seconds-based warmup runs at least 2 discarded epochs when
  `warmup_seconds` is large and per-epoch wall-time is small
  (mock-driven; no real model).
- Seed reset between warmup and first measured epoch: capture
  `random.seed` / `np.random.seed` call counts.
- Measured-window loop accumulates outputs until elapsed >= window.
- JSONL gains `num_prompts` / `input_len` / `output_len` reflecting
  accumulated counts.
- `--warmup-requests` legacy path still works when
  `--warmup-seconds` is unset.

## Tests

- Existing 332 tests must still pass after the comparator widens
  its projection (existing fixtures use a full set of ServerArgs
  fields in their sidecar; verify by adding the missing keys to
  the default `_write_bench_jsonl` sidecar).
- ~15-25 new regressions for the bench_serving timing path,
  comparator widening, and JSONL workload cross-check.
- Expect ≥ 350 passed total.

## Success Criteria

1. `python -m sglang.bench_serving --help` shows
   `--warmup-seconds` and `--measurement-window-seconds`.
2. A driven bench_serving run with
   `--warmup-seconds 0.05 --measurement-window-seconds 0.5 --num-prompts 4`
   against a mock backend writes a JSONL with `duration >= 0.5`
   and includes `num_prompts`, `input_len`, `output_len`.
3. Codex's `disable_cuda_graph=false` vs `true` reproducer now
   exits 2 with "normalized launch-args server_args differ".
4. `pytest test/registered -q` ≥ 350 passed; no Round-32
   regressions.

## Blocking Issues

None — Round 32 closed the radix-cache / required-field /
dynamic-drift / duration-floor gaps. Round 33 builds on that
foundation.

## Queued (out of scope for Round 33)

- Shallow AC-8 prefix-match helper regression coverage cleanup.
- Stale DS bind/runtime comments + token-label lifetime docs.
- All hardware-gated execution (`task-ac1-hwtest`,
  `task-ac4-hwrun`, etc.).
- AC-10 radix-cache fixture — required before AC-11 can run with
  DS radix-cache parity-ON on H200, but does not gate Round-33
  code-tier work.
