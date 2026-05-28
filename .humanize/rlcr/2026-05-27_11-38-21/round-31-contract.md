# Round 31 Contract

## Mainline Objective

Close the three AC-11 defects Codex flagged in the Round 30 review:

1. **AC-11 comparator does not enforce reproducibility / apples-to-apples.**
   Round 30 only checks medians + concurrency. A run with mismatched
   `num_prompts`, `gpu_id`, `disable_radix_cache`, missing `.meta.json`
   sidecars, etc. still publishes `AC-11 verdict: PASS`. The plan
   requires fixed seed + ≥120s warmup + ≥600s measurement window + commit
   SHA + server args + chunked-prefill + cross-side workload agreement.
2. **`_group_by_concurrency` crash path.** It swallows JSON parse
   errors and falls back to filename, but `_run_ac11_mode` re-reads
   the same file → uncaught `JSONDecodeError` + exit 1 (should be
   clean exit 2).
3. **Benchmark scripts cannot produce 3 independent AC-11 trials.**
   Repeated runs overwrite `${MODE}_..._c${CONC}.jsonl`; `WARMUP_*`
   and `MEASUREMENT_WINDOW_S` env knobs are exposed but never enforced.

## Target ACs

- **AC-11** — comparator refuses publication unless every input has a
  validated sidecar with `seed`, `warmup_seconds ≥ 120`,
  `measurement_window_seconds ≥ 600`, matching `commit_sha`,
  `chunked_prefill_size`, and normalized server-args across DSA and
  DS modulo DS-only flags. Benchmark scripts emit
  `_t${TRIAL_ID}` filenames so 3 trials don't overwrite each other.

## Required Implementation

### Fix 1: AC-11 comparator validation gauntlet (`development/benchmark_compare.py`)

New helpers:

- `_sidecar_path(result_path)` → result_path + ".meta.json".
- `_read_ac11_meta(result_path)` → reads sidecar; raises `ValueError`
  on missing file / malformed JSON / non-object root.
- `_normalize_ac11_server_args(meta)` → returns `server_args` dict
  filtered to remove keys that legitimately differ between DSA and
  DS: `enable_double_sparsity`, `double_sparsity_config`,
  `disable_radix_cache` (until AC-10), and any
  `SGLANG_DS_FAULT_INJECT_*` flags.

New per-trial validation in `_run_ac11_mode`:

- Require `output_tps_p50 is not None` and `ttft_p99_s is not None`
  on every trial (no median over None-mixed columns).
- Require `_read_ac11_meta` succeeds for every trial (sidecar
  exists, parses, has `server_args_error == null`).
- Per side (DSA / DS): all trial sidecars share the same `seed`,
  `commit_sha`, `chunked_prefill_size`,
  `normalized_server_args`, and JSONL `num_prompts` / `input_len`
  /`output_len`.
- Per concurrency, across sides: `seed`, `commit_sha`,
  `chunked_prefill_size`, `num_prompts`, `input_len`, `output_len`,
  `normalized_server_args` must all match (DS may differ only in
  the DS-only keys named above). Use `_match_or_refuse` semantics
  to refuse on hardware mismatch.
- Per trial: `meta["warmup_seconds"]` must be present and `>= 120`;
  `meta["measurement_window_seconds"]` must be present and `>= 600`.

Each refusal → exit 2 with a clear log message naming the failing
field, side, trial, and resolved path.

### Fix 2: `_group_by_concurrency` crash fix

- Only fall back to filename when the JSONL parsed successfully but
  the parsed context has no concurrency. Real
  `json.JSONDecodeError` / `FileNotFoundError` should refuse the
  whole AC-11 run with exit 2 (wrapped in `_run_ac11_mode`).
- Wrap the second `_read_bench_jsonl` pass in `_run_ac11_mode` with
  a try/except that returns exit 2 on parse errors.

### Fix 3: Benchmark scripts (`development/benchmark{,_baseline}.sh`)

- Add `TRIALS="${TRIALS:-3}"` outer loop.
- Inside, set `TRIAL_ID=1..TRIALS`; output file becomes
  `${MODE}_..._c${CONCURRENCY}_t${TRIAL_ID}.jsonl` (matching
  `.meta.json` sidecar).
- Add defaults: `WARMUP_SECONDS="${WARMUP_SECONDS:-120}"`,
  `MEASUREMENT_WINDOW_S="${MEASUREMENT_WINDOW_S:-600}"`.
- Pass `WARMUP_SECONDS`, `MEASUREMENT_WINDOW_S`, `TRIAL_ID`,
  `SEED` to `_bench_meta_writer.py`. The writer already records
  these env-var fields.
- The actual time-based warmup workload execution + duration
  enforcement requires bench_serving CLI changes that are outside
  Round 31's scope — the comparator-side validation in Fix 1 will
  refuse trials whose sidecar reports `warmup_seconds < 120` or
  `measurement_window_seconds < 600`. The operator running the
  hardware sweep must therefore set the env knobs honestly.

### Fix 4: Registered regressions

`test/registered/unit/development/test_ac11_comparator.py`:

- Sidecar absent on any trial → exit 2.
- Sidecar malformed JSON → exit 2.
- Sidecar `server_args_error != null` → exit 2.
- Sidecar `warmup_seconds < 120` → exit 2.
- Sidecar `measurement_window_seconds < 600` → exit 2.
- Sidecar `seed` mismatch across trials (same side) → exit 2.
- Sidecar `seed` mismatch across DSA/DS → exit 2.
- Sidecar `commit_sha` mismatch across DSA/DS → exit 2.
- Sidecar `chunked_prefill_size` mismatch → exit 2.
- Workload (`num_prompts` / `input_len` / `output_len`) mismatch
  across DSA/DS → exit 2.
- Normalized `server_args` mismatch (e.g. different `tp_size`) →
  exit 2.
- Allowed DS-only differences (DS has `enable_double_sparsity`,
  DSA doesn't) → still PASS.
- Missing `output_tps_p50` on any trial → exit 2.
- Malformed JSONL with `_c64.jsonl` filename → exit 2 (no traceback).
- Happy path with full valid sidecars + matching workload + correct
  ratios → exit 0.

`test/registered/unit/development/test_option_b_scripts.py`:

- Both benchmark scripts include `TRIALS=${TRIALS:-3}` loop.
- Both write `_t${TRIAL_ID}.jsonl` filenames (regex on the
  parameter expansion line).
- Both default `WARMUP_SECONDS=120` and `MEASUREMENT_WINDOW_S=600`.

## Tests

- Existing 304 tests must still pass.
- ~15 new regressions for the validation gauntlet + crash fix.
- ~4 new script regressions.
- Expect ≥ 323 passed total.

## Success Criteria

1. Codex's PASS-on-mismatch reproducer (DSA tp=8 / DS tp=1, no
   sidecars) now reports exit 2 with a clear refusal message.
2. Codex's malformed `_c64.jsonl` reproducer reports exit 2 with no
   traceback.
3. Three sequential runs of `benchmark.sh` produce
   `_t1.jsonl` / `_t2.jsonl` / `_t3.jsonl` distinct files.
4. `pytest test/registered -q` ≥ 323 passed.

## Blocking Issues

None.

## Queued (out of scope for Round 31)

- Time-based warmup/measurement enforcement at the bench_serving
  CLI level (requires upstream CLI changes).
- Shallow AC-8 prefix-match regression coverage.
- Stale DS bind/runtime comments + token-label lifetime docs.
- All hardware-gated execution.
