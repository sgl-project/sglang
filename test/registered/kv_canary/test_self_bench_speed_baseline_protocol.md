# test_self_bench_speed.py — baseline accumulation protocol

This file documents the two-week warn-only baseline collection mechanism that
precedes the hard-gate flip referenced in testing.md SOT §3.3 + plan
`04-canary-self-e2e.md` step 7.

## State machine

```
[warn-only ship date]  ───▶  [≥ 14 nightly runs accumulated]  ───▶  [hard gate]
        │                                  │                              │
        │                                  │                              │
        ▼                                  ▼                              ▼
test_self_bench_speed.py            recompute P50/P90/P99 from        assert overhead_pct
appends each run's                  samples[] in baseline.json,       <= baseline_pct * 1.5
overhead_pct to                     write back, commit                or env var override
samples[] (per
scenario_key)
```

## Schema

`test_self_bench_speed.baseline.json` is a 3-level dict:

```
{
  "<runner_config>": {
    "<scenario_key>": {
      "samples": [
        {
          "ts_unix": <float>,
          "latency_off_s": <float>,
          "latency_on_s": <float>,
          "overhead_pct": <float>
        },
        ...
      ],
      "p50_ms": <float | null>,
      "p90_ms": <float | null>,
      "p99_ms": <float | null>
    }
  }
}
```

- `runner_config` mirrors the `runner_config=` kwarg in `register_cuda_ci`
  (`1-gpu-large` is the only one defined here today). Adding a runner means
  adding a new top-level key.
- `scenario_key` is the same string emitted by `_measure_overhead` (e.g.
  `qwen3-0.6b/prefill_bs32_isl16384_osl1`). Adding a case means appending a
  new key under each runner.
- `samples[]` is append-only by the bench file; never edit by hand.
- `p50_ms` / `p90_ms` / `p99_ms` start `null` and are populated by the
  recompute step below once `len(samples) >= 14`.

## Recompute step (out of scope for phase 04, in scope for the flip)

Implement a helper script (suggested location:
`scripts/canary/recompute_baseline_percentiles.py`) that:

1. Loads the JSON.
2. For each `(runner_config, scenario_key)` pair, sorts `samples` by `ts_unix`,
   keeps the most recent 14 entries, and computes P50/P90/P99 over their
   `overhead_pct` field.
3. Writes the percentiles back to the same JSON.
4. Commits with message `Refresh canary self-bench baseline percentiles`.

Hook it into the nightly suite tail (after `test_self_bench_speed.py` runs) so
the recompute is automatic.

## Hard-gate flip checklist (plan step 7, two weeks out)

1. Verify each `(runner_config, scenario_key)` has `len(samples) >= 14` and the
   percentiles fields are populated.
2. Modify `_measure_overhead` to read `p50_ms` from the JSON and assert
   `overhead_pct <= p50_ms * float(os.environ.get("SGLANG_KV_CANARY_BENCH_OVERHEAD_THRESHOLD_RATIO", "1.5"))`.
3. Update testing.md SOT §3.3 to remove the "warn-only" qualifier.
4. Update plan 04 `验收标准` to reflect the live gate.

## Out of scope (phase 04 ship)

- Computing percentiles. The bench file appends raw samples only.
- Editing thresholds. The default 1.5x is hard-coded but env-var overridable
  once step 7 lands.
- Adding runner configs other than `1-gpu-large`. Multi-runner work happens
  alongside any future cross-runner CI expansion.
