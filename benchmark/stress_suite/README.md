# Continuous online stress suite

`sglang.benchmark.stress_suite` creates one timestamped workload and runs it
through `sglang.benchmark.serving`. All phases share one client process and
one timeline, so QPS and token lengths change without a warmup or process gap
between phases.

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
- `edge`: zigzag, microburst, tool schema, 32K/80K context boundaries, and high
  concurrency.
- `all`: every phase once.
- `soak`: repeat all non-smoke phases for 20 hours. Smoke runs once at the start.

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

`--arrival-pattern constant` gives fixed spacing. Use `--arrival-pattern poisson`
for Poisson arrivals, reproducible with `--seed`. Every run writes `workload.json`,
the normal serving `result.jsonl`, `benchmark.log`, `phases.jsonl`, `health.jsonl`,
`progress.json`, `summary.json`, and `summary.md`. Completed phases are flushed to
`phases.jsonl` as their last request finishes, including while later phases are
still dispatching. Records follow completion order; the final summary follows
planned phase order. `progress.json` contains current progress and health counts.

While the serving process runs, the suite probes `/ready` every 60 seconds and
updates `progress.json`. SGLang's `/ready` checks readiness without generating a
token, so a long prefill does not require output to pass the probe. The default
probe timeout is 30 seconds and the suite tolerates three failed probes. The
final probe is always recorded. Configure these with `--health-check-path`,
`--health-check-interval-sec`, `--health-check-timeout-sec`, and
`--max-health-failures`. Use a suitable readiness endpoint for another compatible
server. A successful readiness probe does not establish generation correctness;
request completion and latency are checked separately.

`--max-pending-requests` bounds pending request tasks. Scheduling backlog is
reported as phase-level mean/p99 schedule lag and actual dispatch QPS.

The suite queries `/server_info` and `/model_info` when available. Phases that
exceed the advertised context length or require an explicitly unavailable tool
parser are recorded as `SKIP`. Use `--context-length` when a compatible server
does not expose these endpoints, or `--disable-capability-check` to exercise
server-side rejection. Skipped intervals remain in the planned timeline.

## Latency SLOs and diagnostics

Use the serving benchmark's `--goodput` option for per-request latency SLOs in
milliseconds. A request meets the SLO only if it succeeds and meets every given
threshold. Supported metrics are `ttft`, `tpot`, and `e2el`. For example:

```bash
python -m sglang.benchmark.stress_suite \
  --base-url http://127.0.0.1:30000 \
  --profile quick \
  --goodput ttft:500 tpot:50 e2el:5000 \
  --min-slo-attainment 0.99
```

The suite uses the same goodput calculation as `sglang.benchmark.serving`, both
for the full run and each phase. Phase goodput divides qualifying requests by the
planned phase duration; full-run goodput uses the measured benchmark duration.
With `--goodput`, a phase must reach
`--min-slo-attainment` (default `1`, meaning every request) to pass. All planned
requests must still complete successfully and streaming responses must include
a completion marker. Without `--goodput`, latency percentiles are reported but
are not pass/fail thresholds.

Add `--cache-report` for phase-level cached-token totals and cache-hit ratios.
Add `--prometheus` to asynchronously snapshot selected `/metrics` series after
phase completion; repeat `--prometheus-metric` to add series. Snapshots and explicit
fetch errors are written separately to `phases.metrics.jsonl` and included in the
final summary. A slow or unavailable metrics endpoint does not delay request
handling or the completed-phase record.

HTTP status, 429/4xx/5xx, timeout, connection/protocol failures, finish reasons,
missing usage, and incomplete SSE streams are summarized per phase. Chat
streaming requests ask for usage so supported servers can return it. Headers,
authorization fields, cookies, and token-like fields are redacted from printed
and persisted commands.

## Test data and cache behavior

The default `--dataset-source random-ids` generates synthetic prompts locally;
no dataset is downloaded. Ordinary phases use distinct prompts per request,
including across repeated phases. Each run receives a fresh `prompt_seed`, saved
in `workload.json`, so repeating a command with the same traffic seed does not
reuse the previous run's prompts. To replay prompt generation intentionally, use
`--prompt-seed` with the recorded value and retain the same tokenizer, traffic
seed, and workload settings. Replaying that seed against an uncleared cache can
produce cache hits.

The suite defaults to `--warmup-requests 0`; profiles already start with a smoke
phase. Explicit extra warmups reuse the first measured prompt through the serving
benchmark, which can prime its cache. Add `--flush-cache` to clear the target's
cache after warmup when that is appropriate for your test service.

Only the `shared_prefix` phase intentionally reuses a prompt pool, controlled by
`--prompt-pool-size` (default eight). It uses the existing generated shared-prefix
dataset. All serving backends currently create a client session per request, so
there is no separate connection-churn scenario.

`--dataset-source sharegpt` uses the community ShareGPT sampler. Pass
`--dataset-path` for a local copy; otherwise the existing dataset loader downloads
its public source. Selecting `--dataset-source generated-shared-prefix` explicitly
extends intentional prefix reuse to other phases. Private datasets should remain
outside the repository. The suite does not consume a bundled trace fixture.

For a short validation against an existing service:

```bash
python -m sglang.benchmark.stress_suite \
  --base-url http://127.0.0.1:8000 \
  --profile quick \
  --duration-scale 0.1 \
  --context-length 4096 \
  --max-pending-requests 256 \
  --cache-report \
  --output-dir /tmp/sglang-stress-quick
```

Generating a 20-hour plan or completing a short validation does not establish
20-hour stability. Run `soak` against a dedicated service, retain the output
directory, and inspect the final health, phase count, failure categories, server
log, and post-run inference before declaring long-run acceptance.
