# Continuous online stress suite

`sglang.benchmark.stress_suite` creates one timestamped workload and runs it
through the canonical `sglang.benchmark.serving` entrypoint. All phases share
one client process and one time line, so QPS and token lengths change without a
warmup or process gap between corner cases.

```bash
python -m sglang.benchmark.stress_suite \
  --base-url http://127.0.0.1:30000 \
  --profile standard \
  --baseline-qps 2 \
  --peak-qps 8 \
  --output-dir stress_results/standard
```

Profiles:

- `quick`: smoke, steady, burst, and recovery.
- `standard`: ramp, length boundaries, and generated shared-prefix traffic.
- `edge`: zigzag, microburst, connection churn, tool schema, 32K/80K context
  boundaries, and high concurrency.
- `all`: every phase once.
- `soak`: continuously repeat all non-smoke phases for 20 hours. Smoke runs
  only once at the beginning.

Use repeated `--scenario` options to select phases. `--total-duration-sec`
repeats the selected workload to an exact planned duration and truncates only
the final phase. For example, a 20-hour custom run is:

```bash
python -m sglang.benchmark.stress_suite \
  --base-url http://127.0.0.1:30000 \
  --profile standard \
  --total-duration-sec 72000 \
  --baseline-qps 2 \
  --peak-qps 8 \
  --output-dir stress_results/standard-20h
```

`--arrival-pattern constant` gives deterministic spacing. Use
`--arrival-pattern poisson` for reproducible Poisson arrivals. Every run writes
`workload.json`, the normal serving `result.jsonl`, `benchmark.log`,
`phases.jsonl`, `health.jsonl`, `progress.json`, `summary.json`, and
`summary.md`. `phases.jsonl` is flushed as each phase completes, so a partial
run retains completed-phase evidence. `progress.json` stays small and contains
only current progress plus an aggregate health status.

While the serving process runs, the suite checks `/health` every 60 seconds and
atomically updates `progress.json`. The final health check is always recorded
and any failure fails the suite by default. Use `--health-check-interval-sec`
to change the interval, or `--max-health-failures` to tolerate a bounded number
of transient failures. `--max-pending-requests` bounds live asyncio tasks; a
positive scheduling backlog is reported as phase-level mean/p99 schedule lag
and actual dispatch QPS instead of being hidden.

The suite queries `/server_info` and `/model_info` when available. Phases that
exceed the advertised context length or require an explicitly unavailable tool
parser are recorded as `SKIP`. Use `--context-length` when a compatible server
does not expose these endpoints, or `--disable-capability-check` to exercise
server-side rejection behavior deliberately. Skipped intervals remain in the
timeline, so a 72,000-second soak does not silently become shorter.

Add `--cache-report` for phase-level cached-token totals and cache-hit ratios.
Add `--prometheus` to snapshot selected `/metrics` series at phase completion;
repeat `--prometheus-metric` to add series. HTTP status, 429/4xx/5xx, timeout,
connection/protocol failures, finish reasons, missing usage, and incomplete SSE
streams are summarized per phase. Headers, authorization fields, cookies, and
token-like fields are redacted from printed and persisted commands.

Generating a 20-hour plan or completing a short validation is not a substitute
for a 20-hour acceptance run. Run `soak` against a dedicated service, retain the
entire output directory, and verify the final health, phase count, failure
taxonomy, server log, and post-run inference before declaring long-run PASS.

## Test data

The default `--dataset-source random-ids` generates synthetic prompts locally
with the existing SGLang random dataset generator. No dataset is downloaded.
Eight prompts are generated for each length/source combination and reused
across the timestamped request sequence. Change this with `--prompt-pool-size`.

Two existing open-source benchmark generators are also supported:

- `--dataset-source sharegpt` uses the community ShareGPT sampler. Pass
  `--dataset-path` to use a local copy; otherwise the existing serving dataset
  loader downloads its public source at runtime.
- `generated-shared-prefix` is used automatically by the `shared_prefix`
  phase and generates synthetic prefix-sharing requests locally.

The repository contains only six hand-authored examples in
`data/synthetic_trace.jsonl`. Never commit production requests, transformed
production data, customer data, or private dataset metadata.

For a short validation against an existing service:

```bash
python -m sglang.benchmark.stress_suite \
  --base-url http://127.0.0.1:8000 \
  --profile quick \
  --duration-scale 0.1 \
  --context-length 4096 \
  --max-pending-requests 256 \
  --cache-report \
  --prometheus \
  --output-dir /tmp/sglang-stress-quick
```
