# Round 30 Contract

## Mainline Objective

Implement the AC-11 directional comparator semantics in
`development/benchmark_compare.py`. With Round 29 closing the last
AC-12 harness defect Codex's review explicitly called out, and the
hardware gates outside my reach, the next code-tier work the plan
needs is the AC-11 comparator. Codex's queued list and the plan §
AC-11 both spec the same gate:

> Fixed seed, 600s window, 120s warmup, 3 trials, median.
> DS TPS within 5% of DSA TPS at conc=64 (directional gate).
> P99 TTFT ≤ DSA-on P99 TTFT × 1.10.

The current single-trial / absolute-SLO comparator is preserved for
backward compatibility (AC-7 / AC-8 reports already consume it); the
new AC-11 mode is a parallel CLI surface that operators invoke on
the three-trial sweep.

## Target ACs

- **AC-11** — `development/benchmark_compare.py` accepts ≥ 3
  trial JSONLs per mode per concurrency, computes per-concurrency
  medians, enforces the two directional gates, and exits non-zero
  with a profiling-obligation message on failure.

## Required Implementation

### Fix 1: AC-11 trial-set CLI + median aggregation

`development/benchmark_compare.py`:

- New CLI mode invoked via the presence of `--ac11`:
  - `--ac11-baseline-results path1 path2 path3 [more...]` —
    DSA trial JSONLs (≥ 3 per concurrency).
  - `--ac11-ds-results path1 path2 path3 [more...]` —
    DS trial JSONLs (≥ 3 per concurrency).
  - Both arg lists are flat globs; concurrency is parsed from
    each file's per-row JSON (or filename suffix `_c<N>.jsonl`).
- New helpers (pure, CI-testable):
  - `_group_by_concurrency(paths) -> Dict[int, List[str]]` —
    group input paths by their resolved concurrency.
  - `_median(values) -> Optional[float]` — median of a non-empty
    list (returns `None` on empty / all-None).
  - `_median_metrics(metrics: List[RunMetrics]) -> RunMetrics` —
    take medians of `output_tps_p50`, `output_tps_p99`,
    `ttft_p50_s`, `ttft_p99_s`, `tpot_p50_ms`, `tpot_p99_ms`,
    `goodput_under_slo`, `selected_tokens_mean`,
    `total_tokens_mean`; `dense_fallback_total` becomes the sum
    (semantically a counter); `concurrency`, `num_prompts`,
    `isl`, `osl` are taken from the first trial (must agree
    across trials — refuse otherwise).
  - `_evaluate_ac11_gates(dsa_median, ds_median) -> Dict[str, Any]`:
    - `tps_ratio = ds_median.output_tps_p50 / dsa_median.output_tps_p50`
    - `ttft_ratio = ds_median.ttft_p99_s / dsa_median.ttft_p99_s`
    - `tps_pass = tps_ratio >= 0.95`
    - `ttft_pass = ttft_ratio <= 1.10`
    - Returns dict with the ratios + pass/fail booleans + a
      human-readable reason on failure.
- AC-11 mode enforces:
  - At least 3 trials per concurrency on each side (else refuse).
  - Per-concurrency baseline and DS sets have the same
    concurrency keys (else refuse).
  - The first-trial context fields match across trials within
    each mode (uses existing `_match_or_refuse`).
  - Operating-point fingerprint check via the `.meta.json`
    sidecar's `commit_sha` + `chunked_prefill_size` —
    optional but logged.
- Output:
  - Markdown table with per-concurrency `DSA median TPS / DS
    median TPS / TPS ratio / TPS gate (pass/fail) / DSA P99 TTFT
    / DS P99 TTFT / TTFT ratio / TTFT gate` rows.
  - Profiling-obligation message + exit code 3 when ANY
    concurrency fails either gate.
  - Exit code 0 when ALL concurrencies pass both gates.

### Fix 2: Registered regressions

`test/registered/unit/development/test_ac11_comparator.py` (new file):

- `_median` happy path + odd / even counts + None handling.
- `_median_metrics`: 3 RunMetrics → medians where expected,
  `dense_fallback_total` is summed.
- `_group_by_concurrency`: groups paths correctly by parsed
  concurrency; rejects path with no resolvable concurrency.
- `_evaluate_ac11_gates`:
  - TPS pass (DS == DSA, ratio == 1.0).
  - TPS pass (DS = 0.96 × DSA, ratio = 0.96 ≥ 0.95).
  - TPS fail (DS = 0.90 × DSA, ratio = 0.90 < 0.95).
  - TTFT pass (DS = 1.05 × DSA).
  - TTFT fail (DS = 1.20 × DSA).
- AC-11 mode end-to-end:
  - 3 baseline + 3 DS trial fixtures at the same concurrency
    (fabricated as in-memory bench_serving JSONLs); call
    `main(["--ac11", ...])` via `argparse`; assert exit 0 +
    Markdown output names both gates.
  - Same fixture but DS TPS = 0.5 × DSA → exit 3 + obligation
    message present.
  - 2-trial input on either side → exit 2 (refuse, < 3 trials).
  - Concurrency-set mismatch → exit 2.

### Fix 3: Documentation update

Module docstring updated to enumerate the two CLI modes:

1. Single-trial AC-7/AC-8 report (legacy, unchanged).
2. Three-trial AC-11 directional report (new, invoked via
   `--ac11`).

## Tests

- Existing 280 tests must still pass.
- ~12 new registered regressions for AC-11 helpers + CLI.
- Expect ≥ 292 passed total.

## Success Criteria

1. `python development/benchmark_compare.py --ac11 \
   --ac11-baseline-results dsa_c16_t1 dsa_c16_t2 dsa_c16_t3 \
   --ac11-ds-results ds_c16_t1 ds_c16_t2 ds_c16_t3` runs to
   completion against fabricated fixtures with passing gates →
   exit 0.
2. Same invocation with one fixture deliberately failing the TPS
   gate → exit 3 + Markdown output contains "AC-11 TPS gate
   failed" plus a profiling obligation.
3. Same invocation with only 2 trials per side → exit 2.
4. `pytest test/registered -q` ≥ 292 passed.

## Blocking Issues

None.

## Queued (out of scope for Round 30)

- Shallow AC-8 prefix-match regression coverage.
- Stale DS bind/runtime comments + token-label lifetime docs.
- All hardware-gated execution (AC-1, AC-1b, AC-4, AC-6, AC-8,
  AC-9, AC-10, AC-11 hardware runs, AC-12).
