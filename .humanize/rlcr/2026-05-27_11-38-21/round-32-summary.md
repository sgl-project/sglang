# Round 32 Summary

## Work Completed

Codex Round 31 review verified that the validation gauntlet (sidecar
reads, per-side / cross-side agreement, JSONL malformed-input crash fix,
3-trial scripts) all landed correctly, but reopened AC-11 code-tier
completeness with four remaining defects:

1. **Mainline Gap 1** — Timing floors were sidecar-only. A JSONL with
   `duration: 5` and a sidecar claiming `measurement_window_seconds:
   600` still passed.
2. **Mainline Gap 2** — Sidecar validation treated missing required
   fields as agreement under `None == None`. Two sidecars omitting
   `seed` / `commit_sha` / `server_args` compared equal.
3. **Mainline Gap 3** — Radix mismatch was allowed via
   `_DS_ONLY_SERVER_ARG_KEYS`, but plan §AC-11 says only the DS
   enablement pair may differ; AC-11 depends on AC-10.
4. **Blocking Side Issue** — The raw `/get_server_info` payload (with
   `internal_states.last_gen_throughput` and other dynamic telemetry)
   was compared per-trial, so script-generated sequential trials could
   self-refuse on inert drift.

Round 32 closes all four with hard CI regressions.

### Fix 1 — JSONL `duration` floor (`development/benchmark_compare.py`)

`RunMetrics` gained a `duration_s: Optional[float]` field;
`_read_bench_jsonl` extracts `summary["duration"]` (bench_serving emits
this as the wall-clock duration of the measured phase).

New validator `_validate_jsonl_duration(metrics, side, path)`:

- refuses `duration_s is None` ("JSONL is missing the `duration` field;
  cannot verify the AC-11 measurement window"),
- refuses `duration_s < AC11_MIN_MEASUREMENT_WINDOW_SECONDS` ("wall-clock
  duration=Xs is below the AC-11 measurement-window floor of 600.0s").

Wired into `_run_ac11_mode` per trial, alongside `_validate_trial_metrics`
and `_validate_meta_floors`. The Codex reproducer (`duration=5` + sidecar
`measurement_window_seconds=600`) now exits 2 with a clear message.

### Fix 2 — Required-field sidecar validator

New helper `_require_sidecar_fields(meta, side, path)` enforces presence
and well-typedness of:

- `seed`: must be an `int` (and not a `bool`).
- `commit_sha`: must be a non-empty `str`, not `"unknown"` or `""`.
- `chunked_prefill_size`: must be a positive `int`, OR the string
  `"unknown"` (the `_bench_meta_writer.py` fallback when
  `/server_info` doesn't expose the knob).
- `num_prompts`, `isl_total_tokens`, `osl_tokens`: positive `int`.
- `server_args`: non-empty `dict`.

Called inside `_read_ac11_meta` before returning, so every later
agreement check operates on real values. `_read_ac11_meta` now takes
`side: str = "?"` so refusal messages can name "DSA" vs "DS".

`_run_ac11_mode` also asserts `meta["concurrency"] == grouping_conc`
per trial (catches sidecar/file-rename mishaps).

### Fix 3 — Remove radix-cache allowance

- `_DS_ONLY_SERVER_ARG_KEYS` shrunk to `{enable_double_sparsity,
  double_sparsity_config}` (was `{... , disable_radix_cache}`).
- The `hw_reasons = [r for r in hw_reasons if "disable_radix_cache"
  not in r]` post-filter in `_run_ac11_mode` is gone.

AC-11 now refuses radix mismatch on BOTH the launch-args server_args
path (via `_validate_cross_side_agreement` → `_normalize_ac11_server_args`)
AND the JSONL `RunContext` path (via `_match_or_refuse`).

Production note: AC-11 still requires AC-10 to close before the DS
launcher can drop `--disable-radix-cache`. The Round 31 launchers
(DSA: radix on, DS: radix off) now correctly refuse AC-11.

### Fix 4 — Stable launch-args whitelist

`_normalize_ac11_server_args` was a blocklist (`{k: v for k, v in sa.items()
if k not in _DS_ONLY_SERVER_ARG_KEYS and not k.startswith(
"SGLANG_DS_FAULT_INJECT_")}`). It now projects onto a fixed launch-args
whitelist:

```python
_AC11_STABLE_LAUNCH_ARG_KEYS = frozenset({
    "tp_size", "page_size", "model_path",
    "chunked_prefill_size", "dsa_prefill_backend", "dsa_decode_backend",
    "disable_overlap_schedule", "disable_piecewise_cuda_graph",
    "kv_cache_dtype",
    "enable_double_sparsity", "double_sparsity_config",
    "disable_radix_cache",
})
```

Dynamic `/get_server_info` telemetry (`internal_states`, `kv_events`,
`last_gen_throughput`, `gpu_memory_used_bytes`, `step_time`, …) is
dropped — sequential trials no longer self-refuse on inert drift. The
whitelist is schema-safe across future sglang versions, where a
blocklist would re-leak every new dynamic field.

### Fix 5 — `_filename_concurrency` recognizes `_c<N>_t<M>.jsonl`

Regex updated from `r"_c(\d+)\.jsonl$"` to
`r"_c(\d+)(?:_t\d+)?\.jsonl$"`. Round 31's three-trial sweep filenames
(e.g. `dsa_c64_t2.jsonl`) now resolve via the filename fallback when
the JSONL row lacks `max_concurrency`/`concurrency`. The legacy
`_c64.jsonl` form still works.

### Fix 6 — Test fixtures + 10 new validation regressions

`test/registered/unit/development/test_ac11_comparator.py`:

- `_write_bench_jsonl` gains `duration: float = 600.0` (AC-11 floor), so
  existing tests pass the JSONL duration validator. Tests that exercise
  it pass `duration=5.0` or `extra={"duration": None}` (writer pops the
  key for `None`).
- Sidecar `server_args` now always carries `disable_radix_cache`
  (matching real `/get_server_info`). The DS-mode block only adds
  `enable_double_sparsity` + `double_sparsity_config`.
- New `_OMIT = object()` sentinel: `sidecar_overrides={"seed": _OMIT}`
  removes the field entirely so missing-field validation can be
  exercised.
- `_make_trials` defaults both sides to `disable_radix=True` (radix-OFF
  parity, the pre-AC-10 state where the DS launcher passes
  `--disable-radix-cache`). The Round 32 comparator requires radix-cache
  parity, so the default fixture must keep it. Tests that exercise the
  mismatch refusal pass `disable_radix` explicitly.
- `test_workload_num_prompts_mismatch_exit_2` updated to keep radix
  parity so the refusal reason is the intended `num_prompts` mismatch,
  not a side-effect radix mismatch.
- `test_allowed_ds_only_differences_still_pass` doc-comment updated to
  the Round-32 semantics ("only `enable_double_sparsity` +
  `double_sparsity_config` may differ").

New regressions (10 named + 7 parameterized subTests = +17 cases):

- `test_short_jsonl_duration_refused` — JSONL `duration=5` + sidecar
  `measurement_window_seconds=600` → exit 2 (Codex Mainline Gap 1).
- `test_missing_jsonl_duration_refused` — JSONL has no `duration` key
  → exit 2.
- `test_radix_mismatch_refused` — DSA radix ON / DS radix OFF
  → exit 2 (Codex Mainline Gap 3).
- `test_missing_required_sidecar_field_refused` — subTest matrix over
  `seed` / `commit_sha` / `chunked_prefill_size` / `num_prompts` /
  `isl_total_tokens` / `osl_tokens` / `server_args` → each → exit 2
  (Codex Mainline Gap 2).
- `test_server_args_empty_dict_refused` — `server_args = {}` → exit 2.
- `test_commit_sha_unknown_refused` — `commit_sha = "unknown"` → exit 2.
- `test_commit_sha_empty_string_refused` — `commit_sha = ""` → exit 2.
- `test_sidecar_concurrency_mismatch_refused` — sidecar `concurrency=32`
  while JSONL/filename concurrency=64 → exit 2.
- `test_dynamic_server_info_drift_does_not_self_refuse` — 3 trials per
  side whose only difference is per-trial dynamic telemetry
  (`internal_states[0].last_gen_throughput`, `kv_events`,
  `last_gen_throughput`, `gpu_memory_used_bytes`, `step_time`) →
  exit 0 (blocking side issue from Codex Round 31).
- `test_chunked_prefill_size_unknown_string_allowed_when_consistent` —
  both sides have sidecar `chunked_prefill_size = "unknown"` (the
  `_bench_meta_writer.py` fallback) → exit 0.

## Files Changed

- `development/benchmark_compare.py`: + `_AC11_STABLE_LAUNCH_ARG_KEYS`
  whitelist, `_require_sidecar_fields`, `_validate_jsonl_duration`;
  updated `_read_ac11_meta` / `_normalize_ac11_server_args` /
  `_validate_cross_side_agreement` / `_run_ac11_mode`;
  `RunMetrics.duration_s`; updated `_filename_concurrency` regex;
  dropped `disable_radix_cache` from `_DS_ONLY_SERVER_ARG_KEYS`.
- `test/registered/unit/development/test_ac11_comparator.py`: fixture
  updates (`duration=600.0` default, `_OMIT` sentinel,
  `disable_radix_cache` always in sidecar server_args,
  `extra_summary` knob, `_make_trials` defaults radix-parity); existing
  Round 31 regressions adjusted for parity; +10 new tests + 7
  subTests.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
48 passed, 7 subtests passed (was 38; +10 + 7 subtests)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
332 passed, 7 subtests passed (was 322)
```

Verified both Codex Round-31 review reproducers:

```
# Mainline Gap 1: JSONL duration=5 + sidecar measurement_window_seconds=600
python development/benchmark_compare.py --ac11 \
    --ac11-baseline-results dsa_1.jsonl dsa_2.jsonl dsa_3.jsonl \
    --ac11-ds-results ds_1.jsonl ds_2.jsonl ds_3.jsonl
→ exit 2 with
  "AC-11 trial DS=...: bench_serving wall-clock duration=5.0s is below
   the AC-11 measurement-window floor of 600.0s"
  (was exit 0 + "AC-11 verdict: PASS")

# Mainline Gap 2: Sidecars with only timing + server_args_error: null
→ exit 2 with
  "AC-11 sidecar DSA=...: seed must be an int, got None"
  (was exit 0)
```

Commit: `48d6497b1` — [AC-11] tighten comparator: JSONL duration floor,
required-field sidecars, radix parity, launch-args whitelist.

## Remaining Items

Code-tier items queued for future rounds:

- bench_serving CLI changes (`--warmup-seconds` flag + warmup loop +
  measurement-window enforcement at the script/CLI layer). The
  comparator-side `duration` floor catches operator mistakes today;
  enforcing the floor at the bench_serving side will catch them earlier
  (before the trial JSONL is even written).
- Shallow AC-8 prefix-match regression coverage cleanup
  (Codex Round 22 queued).
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

Hardware-gated execution unchanged: `task-ac1-hwtest`,
`task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
`task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
`task-ac10-radix`, `task-ac11-compare` (comparator + scripts now
plan-conformant against all four Round-31 review reproducers; AC-10
must close before the H200 3-trial sweep can run with radix parity on),
`task-ac12-quality`.

## Push-to-remote Status

Branch is 33 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 32 applied the existing
`BL-20260527-shell-json-into-python-source` lesson on the comparator
side — the whitelist normalization is the consumer-side complement of
the writer-side lesson, ensuring dynamic `/get_server_info` payload
shapes don't break sidecar comparison. The validation tightening
itself is a "defense-in-depth" pattern that's better captured as code
+ regressions than as a generalized BitLesson. No new entry warranted.
