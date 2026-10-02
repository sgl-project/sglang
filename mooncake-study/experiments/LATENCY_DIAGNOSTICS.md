# Request And Pause Clock Correlation

The [metadata comparison](CONTEXT_METADATA.md) retained a roughly 600ms TTFT
spike without enough information to identify the paused process. Request
records used `perf_counter`, while scheduler GC logs used wall time. The new
opt-in `--latency-diagnostics` mode adds a common time mapping and client pause
observations. It changes only experiment/test modules. Serving, capture, GC
policy, transfer defaults and publication validation are unchanged.

## Reproduction

Use `h100-runtime-lock.json` and the resident worker, with
`PYTHONDONTWRITEBYTECODE=1` and the matching source `PYTHONPATH`. The option
requires `--request-details` and the existing single-rank, same-host benchmark.

```bash
PYTHONPATH=python:test/registered/unit/training_capture:test/registered/unit \
  python -m unittest test_latency_diagnostics test_benchmark_latency \
  test_bench_sglang_streaming -v

python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /models/Qwen3-0.6B --source-revision CHECKOUT_REVISION \
  --output-dir NEW_DIAGNOSTIC_DIRECTORY --num-prompts 1024 \
  --input-len 16 --output-len 32 --concurrency 8 --capture-slots 16 \
  --ratios 0.1 --repeats 2 --request-details --latency-diagnostics
```

Omit `--latency-diagnostics` for the ordinary measurement path. Instrumented
results are diagnostic evidence, not directly interchangeable with earlier
performance comparisons. The probe allocates event records, schedules callbacks
and enables scheduler GC logging; this can affect allocation/GC timing and
request latency. It neither disables/freezes GC nor discards slow requests.

## Artifacts And Interpretation

Each `requests.json` retains native request timings and adds a `diagnostics`
object only when enabled. The object includes process/host identity, start/end
clock anchors, GC events, delayed event-loop timer observations and truncation
counts. Each anchor brackets a wall-clock read with two `perf_counter` reads;
its midpoint and half-width describe that sample's alignment uncertainty.

Client GC callbacks use the same `perf_counter` as requests. An event-loop timer
runs every 10ms from the first recorded request. Delays of at least 20ms are
recorded as the interval from the missed deadline to actual callback execution.
This is not the exact start/end of blocking code, nor proof of why the callback
was late. Observations may occur before or after all in-flight requests; those
events are retained with an empty overlap list.

At most 10,000 client events are retained. Truncation, incomplete client GC,
invalid intervals and requests outside the anchors reject correlation rather
than silently claiming complete coverage. Hooks and timers are removed on
normal and exceptional client exit. GC enablement and thresholds are preserved.

The driver enables the existing `SGLANG_LOG_GC` switch only for the local server.
Its test-only entrypoint replaces the tokenizer's optional GC warning callback
with a structured observer. It logs completed tokenizer collections of at least
20ms with process identity, generation, exact wall timestamps and monotonic
duration, without changing the GC thresholds or triggering collection. An
installation record is mandatory; a missing observer is not interpreted as an
absence of pauses.
It captures stdout and stderr separately, waits for both log-reader threads to
reach EOF after process exit, and then hashes/parses the completed logs.
Scheduler GC intervals come from paired start/end timestamps, not rounded
duration text. Unmatched log entries remain explicit in the parser result;
overlapping starts for one generation reject an ambiguous multi-rank log.

`measurement.json` and `report.json` retain both log hashes, parsed scheduler
events and `latency_diagnostics`. Structured tokenizer events are parsed as JSON
and joined through the same clock mapping. The correlator uses the first anchor's wall
offset and checks its change at the second anchor, including both read
uncertainties. If that observed bound exceeds 5ms, scheduler mapping is omitted
and request wall timestamps are null; client observations remain usable.
Two anchors cannot exclude an intervening clock jump that later reverses.

Long events retain request indices whose TTFT or post-first-token interval
overlaps. The worst TTFT/TPOT requests include their publication membership and
the corresponding event IDs and overlap durations. Client GC and timer lag can
describe the same pause, and one event can affect a whole batch. These durations
must not be added as independent costs or interpreted as causal latency.
The full publication identity set and snapshot readback checks remain present.
Non-GC tokenizer pauses, detokenizer and OS scheduling pauses are not directly
observed. The maximum TTFT and number of requests exceeding 500ms accompany the
worst-request lists; the 500ms count is a diagnostic threshold, not a production
SLO.

## Intermediate Findings

The v1 512-request smoke completes an off/on/off bracket and validates 50
snapshots. In its final capture-off phase a **103.524ms client timer delay**
overlaps TTFT for requests 88-95, with no simultaneous client GC of at least
20ms. Request 92 has 148.725ms TTFT followed by a 0.00785ms TPOT, consistent
with buffered delivery. This identifies a late client event loop, not the
operation that delayed it. Other long client GC events occur outside requests.

V2 completes two 1,024-request off/on/off rounds: **6,144 requests and 184
post-exit snapshots**. Looking only at p99 and long observed pause events would
miss the main tail: every phase has **eight requests above 500ms TTFT**. Their
maximum TTFT spans **578.894-603.620ms**, while p99 spans **58.031-88.592ms**.
The affected fraction is 8/1,024, below 1%. Slow requests are 624-631 in off
phases and 632-639 in enabled phases. No client GC, client timer lag or scheduler
GC of at least 20ms overlaps those requests. This motivates the additional
tokenizer observation in v3; it does not establish that the stalls disappeared.

A separate v2 run with diagnostics disabled completes 192 requests and validates
64 snapshots. Its request files retain the original schema without diagnostic
fields. An offline replay of all three intermediate reports reproduces all
12 phase summaries, verifies request/log hashes and publication identities,
and reproduces the nine instrumented phase correlations without inference.

## Tokenizer GC Finding

V3 completes another 1,024-request off/on/off bracket. Each phase observes one
generation-two tokenizer GC that overlaps TTFT for requests 168-175. The
tokenizer entrypoint changes import/allocation history, so the affected indices
differ from v2. Client timer lag and scheduler GC do not account for these
in-flight pauses.

| Capture Ratio | Requests/s | p99 TTFT ms | Maximum TTFT ms | Requests Above 500ms | Tokenizer GC ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Off | 88.166 | 114.634 | 545.043 | 8 | 500.057 |
| 10% | 75.050 | 61.398 | 534.256 | 7 | 487.729 |
| Off | 88.987 | 62.315 | 537.393 | 8 | 492.640 |

The GC spans lie within each affected request's TTFT interval, across both
capture modes. The wall-clock intervals agree with separately recorded
monotonic GC durations within approximately one microsecond; the client anchor
bounds are 2.727-3.083 microseconds. This provides direct process/collection
evidence for the reproduced pauses, rather than inferring them from publication
membership. It does not retroactively prove the cause of the original
uninstrumented metadata comparison, or measure the effect of changing GC policy.

The final run validates **92 snapshots after producer exit**, with matching
Prometheus stage metrics and no measured writer error, capture failure or
quarantine. All four serving experiments together complete **10,944 requests /
350,208 output tokens** and validate **390 snapshots**, including intermediates.
Offline replays reproduce all 15 phase summaries and 12 diagnostic correlations
with verified request/log hashes and no inference. The diagnostic-disabled
smoke is from v2, before the test-only tokenizer entrypoint was added.

The current pod's visible cgroup has `cpu.max = max 100000`, zero throttling
counters and no pod CPU limit. This excludes an observed quota throttle in that
cgroup, not all host scheduling effects. Detokenizer, non-GC tokenizer work and
other client stalls remain outside the observed GC spans.

## Verification Scope

The unit suite includes real GC callback installation/removal, failure cleanup,
a known 50ms event-loop block, bounded records, incomplete log handling, clock
steps and exact synthetic overlap checks. Native empty-text streaming timing
and publication identity replay tests remain included. There are **24 passing
methods**, including structured tokenizer GC parsing and a sub-1% tail case.

Baseline production is `b091ab1be267f9b9bd30e143d5cacfccf034540d`.
Frozen sources v1/v2/v3 have identical production code and native streaming
client. V2 separates the server log streams and waits for log drain; it also
adds the live timer-block test. V3 adds tokenizer observations, tail extrema
and their tests. The v1 smoke result is retained as an intermediate observation,
not a final log-completeness check. Source,
job and artifact identities are recorded in [latency-diagnostics.json](latency-diagnostics.json).

All seven worker jobs finish with exit code zero. Five Python files pass Black
and full Ruff. The resident H100 returns to idle load with no extra allocation.
The next performance experiment should exercise existing warmup/GC controls
under explicit policy and check memory/lifecycle behavior before changing a
serving default. The implementation here adds observation, not a GC-policy fix.

These tests use synthetic token IDs, Qwen3-0.6B BF16, one H100, decode graphs and
overlap, local TCP Store, a test HTTP Catalog and GPFS publication journals.
They do not establish a production SLO, production Catalog retention, new RDMA
coverage or trained draft quality. The original metadata-run spike cannot be
retroactively attributed because its client pause observations were not saved.
