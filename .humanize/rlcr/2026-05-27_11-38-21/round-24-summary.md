# Round 24 Summary

## Work Completed

Codex Round 23 review verified the locked Option B flags and the
16/32/64 concurrency sweep, but caught a blocking bug in the new
benchmark sidecar writer:

> `development/benchmark.sh:71-83` splices raw `/get_server_info` JSON
> into a Python heredoc as source code. `/get_server_info` legitimately
> returns booleans (`true` / `false`) and nulls (`null`) — valid JSON
> tokens but invalid Python identifiers. After a successful H200
> benchmark run, sidecar generation raises `NameError`. The Round 23
> contract also required `chunked_prefill_size`; the implementation
> did not include it.

Same bug in `benchmark_baseline.sh`. Without sidecars the AC-11
reproducibility audit cannot run — both AC-8 and AC-9 hardware
artifacts would be missing the operating-point fingerprint.

### Fix 1 — `development/_bench_meta_writer.py` (new helper)

Pure-Python standalone script. Reads every metadata field from
environment variables. Parses `SERVER_ARGS_JSON` with `json.loads`
inside Python — never splices JSON as source. On parse failure
(empty / malformed / non-object) records `server_args = {}` and a
short `server_args_error` string so the operator can diagnose offline.

Schema (every field always present; missing values are `null`):

- `commit_sha`, `mode`, `concurrency`, `seed`, `num_prompts`,
  `isl_total_tokens`, `osl_tokens`, `timestamp_utc`.
- `chunked_prefill_size` — extracted from `server_args` (was missing
  in Round 23 despite the contract).
- `warmup_requests`, `measurement_window_seconds`, `trial_id` —
  AC-11 reproducibility fields Codex called out.
- `server_args` — parsed JSON object (or `{}` on error).
- `server_args_error` — `null` or a short reason.

The Round 23 JSON-as-Python-source splice is now structurally
impossible.

### Fix 2 — Both benchmark scripts call the helper

`development/benchmark.sh` and `benchmark_baseline.sh` replace the
inline `python3 - <<PYEOF` heredoc with:

```bash
COMMIT_SHA="${COMMIT_SHA}" MODE="${MODE}" ... \
SERVER_ARGS_JSON="${SERVER_ARGS_JSON}" \
TRIAL_ID="${TRIAL_ID:-1}" \
WARMUP_REQUESTS="${WARMUP_REQUESTS:-}" \
MEASUREMENT_WINDOW_S="${MEASUREMENT_WINDOW_S:-}" \
python3 "$(dirname "$0")/_bench_meta_writer.py" > "${META_FILE}"
```

All JSON now travels via env-var data, not Python source. Operators
who want to record 3-trial AC-11 evidence can override
`TRIAL_ID=1/2/3`, `WARMUP_REQUESTS=120`, `MEASUREMENT_WINDOW_S=600`.

### Fix 3 — `test/registered/unit/development/test_bench_meta_writer.py`

10 registered tests invoke `_bench_meta_writer.py` as a subprocess
with controlled env vars:

- Realistic JSON with `true` / `false` / `null` / nested dicts
  produces valid JSON, preserves types (`is True`, `is None`,
  nested object identity), and extracts `chunked_prefill_size: 4096`.
- Empty `SERVER_ARGS_JSON` → `server_args = {}`, error contains
  `"empty"`.
- Malformed `SERVER_ARGS_JSON` (`"{not json"`) → error contains
  `"parse_error"`.
- Non-object JSON (`"[1, 2, 3]"`) → error contains `"not_object"`.
- `TRIAL_ID` defaults to `"1"`; env override works.
- AC-11 reproducibility fields always present (even when null);
  numeric overrides parse correctly.
- Output is multi-line pretty-printed JSON, re-parseable.

### Fix 4 — `test_option_b_scripts.py` reinforced

Replaced the shallow `commit_sha` / `server_args` string assertions
(Codex called them "shallow" in the review) with explicit
no-go assertions:

- Both benchmark scripts MUST reference `_bench_meta_writer.py`.
- Both benchmark scripts MUST NOT contain `PYEOF` (the unsafe
  heredoc terminator).
- Both benchmark scripts MUST NOT contain
  `"server_args": ${SERVER_ARGS_JSON` (the JSON-as-source splice).
- Both benchmark scripts MUST pass `SERVER_ARGS_JSON="${SERVER_ARGS_JSON}"`
  as an env-var assignment to the helper.

So any future "small cleanup" that reintroduces the Round 23 bug
fails this test.

## Files Changed

- `development/_bench_meta_writer.py`: NEW — standalone, env-var-driven,
  json.loads-safe sidecar writer.
- `development/benchmark.sh`, `development/benchmark_baseline.sh`:
  replaced embedded heredoc with subprocess call to the helper;
  added `TRIAL_ID`, `WARMUP_REQUESTS`, `MEASUREMENT_WINDOW_S` env
  pass-throughs.
- `test/registered/unit/development/test_bench_meta_writer.py`: NEW —
  10 tests against the writer.
- `test/registered/unit/development/test_option_b_scripts.py`:
  reinforced 2 sidecar-emit tests with helper-reference + heredoc-forbid
  assertions.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/development -q
23 passed, 0 failed (was 10)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development -q
229 passed, 0 failed (was 216 before this round)

bash -n development/benchmark.sh         # OK
bash -n development/benchmark_baseline.sh # OK
```

Manual writer smoke (the realistic JSON shape Codex named):

```
SERVER_ARGS_JSON='{"disable_radix_cache": true, "kv_events": null,
                   "chunked_prefill_size": 4096, "tp_size": 8}' \
python3 development/_bench_meta_writer.py
# → valid JSON, server_args["disable_radix_cache"]=true (round-trip),
#   chunked_prefill_size=4096, server_args_error=null.
```

Commit: `856ab7356` — [AC-8/9] Fix Round 23 sidecar JSON injection
bug; add chunked_prefill_size + AC-11 fields.

## Remaining Items

Code-tier items queued for future rounds:

- **AC-12 scaffold replacement** (`test/manual/test_double_sparsity_v32.py`
  still skip-only NIAH 4K/16K/64K + MMLU 5-shot). HARD loop-closure
  gate.
- **`benchmark_compare.py` AC-11 directional gate**: 3-trial median
  per concurrency, DS TPS within 5% of DSA, P99 TTFT ≤ 1.10× DSA;
  currently still the absolute-SLO single-trial framing.
- Shallow AC-8 prefix-match regression coverage (extract `_prefix_match`
  or mock through `test_quality_smoke`).
- Stale DS bind/runtime comments + token-label lifetime docs.

Hardware-gated tasks unchanged: `task-ac1-hwtest`, `task-ac4-hwrun`,
`task-ac6-hwrun`, `task-ac1b-probe`, `task-ac8-server`,
`task-ac8-quality`, `task-ac9-baseline`, `task-ac10-radix`,
`task-ac11-compare`, `task-ac12-quality`.

## Push-to-remote Status

Branch is 25 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: add
Lesson ID(s): BL-20260527-shell-json-into-python-source
Notes: This was a clean, generalizable failure mode: shell scripts that
inject a captured JSON string into a Python heredoc as source code
work fine on test fixtures (objects of strings + ints) but fail on
real JSON containing `true` / `false` / `null` (valid JSON tokens,
invalid Python identifiers). Worth recording so future scripts use
env-var + `json.loads` from the start.
