# Round 31 Summary

## Work Completed

Codex Round 30 review caught three real AC-11 defects:

1. The comparator's gate evaluation ran on ratios alone — no
   sidecar reads, no cross-side workload/server-args checks, no
   warmup/window enforcement. A run with DSA `tp_size=8` /
   DS `tp_size=1`, mismatched `num_prompts`, missing `.meta.json`
   sidecars, etc. still published `AC-11 verdict: PASS`.
2. `_group_by_concurrency` swallowed JSON parse errors and let
   `_run_ac11_mode` re-trip them as an uncaught
   `JSONDecodeError` + exit 1 (instead of clean exit 2).
3. `benchmark.sh` / `benchmark_baseline.sh` overwrote
   `..._c${CONC}.jsonl` on every run — three trials produced one
   file, not three.

Round 31 closes all three with hard CI regressions.

### Fix 1 — Validation gauntlet (`development/benchmark_compare.py`)

New constants:

- `AC11_MIN_WARMUP_SECONDS = 120.0`
- `AC11_MIN_MEASUREMENT_WINDOW_SECONDS = 600.0`
- `_DS_ONLY_SERVER_ARG_KEYS = {enable_double_sparsity,
  double_sparsity_config, disable_radix_cache}` plus any
  `SGLANG_DS_FAULT_INJECT_*` flag — the only legitimate
  differences between DSA and DS server args.

New helpers:

- `_sidecar_path(p)` → `p + ".meta.json"`.
- `_read_ac11_meta(p)` raises `ValueError` on missing /
  malformed / non-object sidecar or non-null
  `server_args_error`.
- `_normalize_ac11_server_args(meta)` filters DS-only keys before
  cross-side comparison.
- `_validate_trial_metrics(metrics, side, path)` refuses trials
  missing `output_tps_p50` or `ttft_p99_s` (so `_median` can't
  turn 1 valid sample + 2 Nones into a passing 3-trial median).
- `_validate_meta_floors(meta, side, path)` refuses
  `warmup_seconds < 120` or `measurement_window_seconds < 600`.
- `_validate_per_side_agreement(metas, paths, side)` refuses
  any within-side seed / commit_sha / chunked_prefill_size /
  num_prompts / ISL / OSL / normalized_server_args disagreement.
- `_validate_cross_side_agreement(dsa_meta, ds_meta, conc)`
  refuses any cross-side seed / commit_sha / chunked / workload /
  server_args (after normalization) disagreement.
- Reuses existing `_match_or_refuse` for GPU/TP/page/concurrency on
  the JSONL context. Drops only the `disable_radix_cache`
  mismatch reason (AC-10 gap).

`_run_ac11_mode` now wraps the read passes in try/except so any
parse/refusal raises become clean exit 2 + log message naming the
side, trial path, and failing field. No tracebacks reach the
operator.

### Fix 2 — `_group_by_concurrency` crash fix

Stop swallowing `json.JSONDecodeError`; only fall back to filename
when the JSONL parsed cleanly but the parsed context lacks
concurrency. The second-read pass in `_run_ac11_mode` is also
wrapped so any parse / FileNotFound / refusal returns clean exit 2.

### Fix 3 — `_bench_meta_writer.py`: emit `warmup_seconds`

Added `warmup_seconds` field (float) from `WARMUP_SECONDS` env var.
`warmup_requests` retained for back-compat.

### Fix 4 — Benchmark scripts: 3-trial loop + non-overwriting filenames

`benchmark.sh` + `benchmark_baseline.sh`:

- `TRIALS="${TRIALS:-3}"`, `WARMUP_SECONDS="${WARMUP_SECONDS:-120}"`,
  `MEASUREMENT_WINDOW_S="${MEASUREMENT_WINDOW_S:-600}"`.
- Outer trial loop `for TRIAL_ID in $(seq 1 "${TRIALS}")` writes
  `${MODE}_..._c${CONCURRENCY}_t${TRIAL_ID}.jsonl` (and matching
  `.meta.json`).
- Three trial runs produce three distinct files.
- Timing knobs reach `_bench_meta_writer.py` via env vars, so the
  comparator can refuse runs that don't meet AC-11 floors.

### Fix 5 — Registered regressions (+18 new)

`test_ac11_comparator.py` (+14):
- missing sidecar → exit 2;
- malformed JSON sidecar → exit 2;
- non-null `server_args_error` → exit 2;
- `warmup_seconds < 120` → exit 2;
- `measurement_window_seconds < 600` → exit 2;
- within-side seed mismatch → exit 2;
- cross-side seed mismatch → exit 2;
- cross-side commit_sha mismatch → exit 2;
- chunked_prefill_size mismatch → exit 2;
- num_prompts workload mismatch → exit 2;
- server_args tp_size mismatch → exit 2;
- DS-only differences (enable_double_sparsity + disable_radix_cache)
  still produce exit 0 (allowed via `_DS_ONLY_SERVER_ARG_KEYS`);
- missing `output_tps_p50` metric → exit 2;
- malformed `*_c64.jsonl` JSONL → exit 2 (no traceback — Codex's
  Round 30 review reproducer).

`test_option_b_scripts.py` (+4):
- both bench scripts default `TRIALS="${TRIALS:-3}"`;
- both loop `for TRIAL_ID in $(seq 1 "${TRIALS}")`;
- both filename include `_c${CONCURRENCY}_t${TRIAL_ID}.jsonl`;
- both default `WARMUP_SECONDS=120` + `MEASUREMENT_WINDOW_S=600`.

## Files Changed

- `development/benchmark_compare.py`: +AC-11 constants + 8 new
  validation helpers; `_group_by_concurrency` crash fix; full
  validation gauntlet wired into `_run_ac11_mode`.
- `development/_bench_meta_writer.py`: +`warmup_seconds` field.
- `development/benchmark.sh` + `development/benchmark_baseline.sh`:
  `TRIALS=3` outer loop, `_t${TRIAL_ID}.jsonl` filenames,
  `WARMUP_SECONDS`/`MEASUREMENT_WINDOW_S` defaults + pass-through.
- `test/registered/unit/development/test_ac11_comparator.py`:
  fixture helper grows `sidecar=` / `sidecar_overrides=` / `mode=`
  / `tp_size=` / `disable_radix_cache=` knobs; +14 validation
  regressions; existing 24 tests still pass.
- `test/registered/unit/development/test_option_b_scripts.py`:
  +4 script-contract regressions.

## Validation

```
PYTHONPATH=python pytest test/registered/unit/development/test_ac11_comparator.py -q
38 passed, 0 failed (was 24; +14)

PYTHONPATH=python pytest test/registered/unit/development -q
65 passed, 0 failed (was 47; +18)

PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py \
                            test/registered/unit/development \
                            test/registered/unit/manual -q
322 passed, 0 failed (was 304)

bash -n development/benchmark.sh         # OK
bash -n development/benchmark_baseline.sh # OK
```

Verified both Codex Round-30 reproducers:

```
# Mismatched DSA tp=8 / DS tp=1, no sidecars:
python development/benchmark_compare.py --ac11 \
    --ac11-baseline-results dsa_1.jsonl dsa_2.jsonl dsa_3.jsonl \
    --ac11-ds-results ds_1.jsonl ds_2.jsonl ds_3.jsonl
→ exit 2 with "AC-11 sidecar missing" + "AC-11 input refusal"
  (was exit 0 + "AC-11 verdict: PASS")

# Malformed *_c64.jsonl files:
python development/benchmark_compare.py --ac11 \
    --ac11-baseline-results malformed_t1_c64.jsonl ...
→ exit 2 + clean log (was uncaught JSONDecodeError + exit 1)
```

Commit: `732929181` — [AC-11] Comparator validation gauntlet
+ 3-trial scripts + crash fix.

## Remaining Items

Code-tier items queued for future rounds:

- Time-based warmup/measurement enforcement at the bench_serving CLI
  level (requires upstream CLI changes; the AC-11 comparator side
  enforces the floor via sidecar metadata).
- Shallow AC-8 prefix-match regression coverage cleanup
  (Codex Round 22 queued).
- Stale `deepseek_v2.py` slot-authority comments.
- Stale `token_label_table.py` lifetime docs.

Hardware-gated execution unchanged: `task-ac1-hwtest`,
`task-ac4-hwrun`, `task-ac6-hwrun`, `task-ac1b-probe`,
`task-ac8-server`, `task-ac8-quality`, `task-ac9-baseline`,
`task-ac10-radix`, `task-ac11-compare` (comparator + scripts now
plan-conformant; only the 3-trial H200 sweep + comparator
invocation remain), `task-ac12-quality` (harness fully gate-tight).

## Push-to-remote Status

Branch is 32 commits ahead of `jimmy/dev/double-sparsity-standalone`.
The RLCR loop's `loop-bash-validator.sh` hook continues to block
`git push`. Per-round pushing requires re-launching with
`--push-every-round`.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: Round 31 applied the existing
`BL-20260527-shell-json-into-python-source` lesson on the
benchmark-meta side (already implemented in Round 24) — the
comparator side just consumes the safe JSON the writer already
emits. The validation gauntlet itself is the kind of
"defense-in-depth" pattern that's better captured as code +
regressions than as a generalized BitLesson. No new entry warranted.
