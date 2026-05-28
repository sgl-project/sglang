# Round 24 Contract

## Mainline Objective

Fix the Round 23 benchmark-sidecar writer so it survives real
`/get_server_info` JSON (which contains `true`/`false`/`null` — invalid
Python identifiers) and so the sidecar carries the
`chunked_prefill_size` field the Round 23 contract promised but the
Round 23 implementation forgot. Add the AC-11 reproducibility fields
Codex called out (`warmup_requests`, `measurement_window_seconds`,
`trial_id`). Lock the contract with a registered regression that feeds
problematic JSON through the writer.

Codex Round-23-review blocking finding:
- `development/benchmark.sh:52` captures `/get_server_info` raw JSON.
- `development/benchmark.sh:71-83` splices that JSON into a Python
  heredoc as source code. JSON `true` / `false` / `null` are NOT valid
  Python identifiers, so the sidecar generation raises `NameError`
  after a real H200 benchmark run completes. The benchmark JSONL is
  fine; the sidecar — and therefore the AC-11 reproducibility audit —
  is missing.
- Same bug in `development/benchmark_baseline.sh:72-84`.
- The Round 23 contract listed `chunked_prefill_size` as a required
  sidecar field; the implementation did not include it.

## Target ACs

- **AC-8 / AC-9** — the benchmark sidecar emits valid JSON for every
  realistic `/get_server_info` response and includes
  `chunked_prefill_size` plus the AC-11 reproducibility fields
  (`warmup_requests`, `measurement_window_seconds`, `trial_id`).

## Required Implementation

### Fix 1: Extract sidecar writer into a standalone helper

New file `development/_bench_meta_writer.py`:
- Reads all metadata from environment variables (no string interpolation
  into Python source — the existing heredoc approach is the bug).
- Reads `SERVER_ARGS_JSON` env var and `json.loads`-es it inside Python.
  On parse failure, records `{}` for `server_args` plus
  `"server_args_error": <reason>` so the operator can diagnose offline.
- Extracts `chunked_prefill_size` from `server_args` directly (fallback
  `"unknown"` if absent).
- Writes the sidecar JSON to stdout (bash redirects to file).
- Schema:
  ```
  {
    "commit_sha": "...",
    "mode": "double_sparsity" | "native_nsa",
    "concurrency": int,
    "seed": int,
    "num_prompts": int,
    "isl_total_tokens": int,
    "osl_tokens": int,
    "timestamp_utc": "...",
    "chunked_prefill_size": int | "unknown",
    "warmup_requests": int | null,
    "measurement_window_seconds": float | null,
    "trial_id": str (defaults to "1"; operator increments for 3-trial sweep),
    "server_args": {...} | {},
    "server_args_error": null | "...",
  }
  ```

### Fix 2: Update both benchmark scripts to call the helper

- Replace the inline `python3 - <<PYEOF` block in `benchmark.sh` and
  `benchmark_baseline.sh` with:
  ```bash
  COMMIT_SHA="${COMMIT_SHA}" \
  MODE="${MODE}" \
  CONCURRENCY="${CONCURRENCY}" \
  SEED="${SEED}" \
  NUM_PROMPTS="${NUM_PROMPTS}" \
  ISL_TOTAL_TOKENS="$(( SYS_LEN + Q_LEN ))" \
  OSL_TOKENS="${OUT_LEN}" \
  TIMESTAMP_UTC="${TIMESTAMP_UTC}" \
  SERVER_ARGS_JSON="${SERVER_ARGS_JSON}" \
  TRIAL_ID="${TRIAL_ID:-1}" \
  WARMUP_REQUESTS="${WARMUP_REQUESTS:-}" \
  MEASUREMENT_WINDOW_S="${MEASUREMENT_WINDOW_S:-}" \
  python3 "$(dirname "$0")/_bench_meta_writer.py" > "${META_FILE}"
  ```
- This passes JSON via environment, not as Python source — the Round 23
  bug is structurally impossible to recur.

### Fix 3: Registered regression for the sidecar writer

`test/registered/unit/development/test_bench_meta_writer.py`:
- Invoke `_bench_meta_writer.py` as a subprocess with env vars set.
- Cases:
  1. `SERVER_ARGS_JSON` is a realistic shape with `true`/`false`/`null`
     values (e.g. `'{"disable_radix_cache": true, "kv_events": null,
     "chunked_prefill_size": 4096}'`). Asserts the output is valid JSON
     parseable by `json.loads`, `chunked_prefill_size == 4096`, and
     `server_args["disable_radix_cache"] is True`.
  2. `SERVER_ARGS_JSON` is empty (server unreachable). Asserts
     `server_args == {}` and `chunked_prefill_size == "unknown"`.
  3. `SERVER_ARGS_JSON` is malformed (`"{not json"`). Asserts
     `server_args == {}` and `server_args_error` is a non-empty string.
  4. `TRIAL_ID` defaults to `"1"` when env var is unset.
  5. All AC-11 reproducibility fields are present (even if null).

### Fix 4: Strengthen Round 23 script test

`test/registered/unit/development/test_option_b_scripts.py`:
- Add an assertion that both benchmark scripts call
  `_bench_meta_writer.py` (so a future refactor that reintroduces the
  inline heredoc fails the test).
- Add an assertion that both scripts include `chunked_prefill_size` in
  the sidecar contract (verified via the helper's schema, not by
  string match on the bash file).

## Tests

- Existing 216 tests must still pass.
- ≥ 5 new sidecar-writer tests + 1 reinforcement on the script
  contract test.
- Expect ≥ 222 passed.

## Success Criteria

1. `python3 development/_bench_meta_writer.py` with
   `SERVER_ARGS_JSON='{"disable_radix_cache": true, "kv_events": null,
   "chunked_prefill_size": 4096}'` produces valid JSON parseable by
   `json.loads`, no `NameError`.
2. Output JSON contains `chunked_prefill_size: 4096`,
   `server_args.disable_radix_cache: true`, all AC-11 reproducibility
   keys present (even if null).
3. Both `benchmark.sh` and `benchmark_baseline.sh` call
   `_bench_meta_writer.py` (no embedded `<<PYEOF` JSON-spliced heredoc).
4. `bash -n` still passes on both benchmark scripts.
5. `PYTHONPATH=python pytest test/registered/unit/development -q`
   ≥ 16 passed (10 existing + 5 new writer tests + 1 reinforcement).

## Blocking Issues

None.

## Queued (out of scope for Round 24)

- AC-12 scaffold replacement (NIAH 4K/16K/64K + MMLU 5-shot real
  harness) — separate larger round.
- `benchmark_compare.py` 3-trial median + AC-11 directional gate
  enforcement — separate round.
- Shallow prefix-match regression coverage cleanup.
- Stale DS bind/runtime comments + token-label lifetime docs.
- All hardware-gated tasks.
