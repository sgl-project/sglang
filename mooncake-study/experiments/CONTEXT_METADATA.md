# Deferred Teacher Position Metadata

`RequestCaptureContext.record_teacher_range` validates contiguous teacher rows
against the computed KV prefix. It previously also allocated an `arange` tensor
and copied it into the Host slot on every append. The aux owner now materializes
`logits_positions = prompt_length ... total_length - 1` directly into the
existing Host buffer once at `seal`, after speculative/lookahead trimming.

Actual model `position_ids` still come from forward execution. Teacher values,
IDs, full-vocabulary LSE, CUDA completion fences, masks, snapshot validation and
Mooncake publication are unchanged. Collecting slots are incomplete snapshots;
only the final committed position prefix is published. Non-aux partitions do
not materialize this field. Direct D2H remains the default.

## Reproduction

Use `h100-runtime-lock.json` and submit through the resident worker. Set
`PYTHONDONTWRITEBYTECODE=1` to avoid filling the shared filesystem with caches
for frozen source copies. Each output directory must be new.

```bash
PYTHONPATH=python:test/registered/unit/training_capture python -m unittest \
  test_context test_partition_context test_teacher_staging test_cuda_snapshot \
  test_coordinator test_pd_capture test_snapshot_writer test_trace_attribution -v

python test/registered/storage/test_training_capture_runtime.py \
  --model-path /models/Qwen3-0.6B
TRAINING_CAPTURE_TEST_MODEL=/models/Qwen3-0.6B \
  python test/registered/storage/test_training_capture_pd.py

python mooncake-study/experiments/profile_training_capture.py \
  --model-path /models/Qwen3-0.6B --source-revision CHECKOUT_REVISION \
  --output-dir NEW_NODE_LOCAL_PROFILE_DIRECTORY

python mooncake-study/experiments/summarize_capture_trace.py \
  NEW_NODE_LOCAL_PROFILE_DIRECTORY/on/prefill/*.trace.json.gz \
  NEW_NODE_LOCAL_PROFILE_DIRECTORY/on/decode/*.trace.json.gz \
  --require-capture --output NEW_SUMMARY_JSON

python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /models/Qwen3-0.6B --source-revision CHECKOUT_REVISION \
  --output-dir NEW_SERVING_DIRECTORY --num-prompts 512 \
  --input-len 16 --output-len 32 --concurrency 8 --capture-slots 16 \
  --ratios 0.1 --repeats 1 --request-details
```

Baseline production is `f8c0e1616eb3760292eab038c9dbc51b81f29495`, frozen at
`$LAB/sglang-context-metadata-before`. Candidate runtime uses
`$LAB/sglang-context-metadata-v1`; v2 adds only the offline trace analyzer and its
test. The [evidence index](context-metadata.json) binds final source hashes,
commands, terminal results, logs and artifacts. No additional GPU is allocated.

## CPU Operator Evidence

The analyzer now attributes complete `cpu_op` events contained in a capture
scope on the same process/thread. Nested operator durations are inclusive and
overlap; event counts are not allocation counts. CPU events never establish
CUDA correlation bindings. Device work retains the runtime/driver correlation
path, including asynchronous work that finishes after its CPU scope.

Both sources use Qwen3-0.6B BF16, concurrency eight, ten warmup batches and five
profiled batches per workload. Each enabled workload publishes its 40 measured
requests. The profile driver checks READY counts but does not perform numerical
Store readback; the separate runtime and serving tests do that.

| Input / Output Tokens | Teacher Append Scopes | CPU Scope Before / After ms | `aten::arange` Events Before / After | `aten::empty` Events Before / After | `aten::copy_` Events Before / After |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 / 1 | 45 / 45 | 11.325 / 9.562 | 90 / 0 | 45 / 0 | 180 / 135 |
| 1 / 32 | 1,320 / 1,320 | 283.948 / 246.773 | 2,640 / 0 | 1,320 / 0 | 5,280 / 3,960 |

Teacher append CPU scope time falls **15.57%** and **13.09%** in these traces.
The `arange` counts include nested dispatch. Materialization at seal is outside
these append scopes, so this does not imply zero position-generation work in
the complete request. Profiler scope timings do not measure serving latency.

All four capture groups retain their GPU kernel counts, D2H/D2D counts and
transferred bytes. Teacher D2H remains 135 calls / 46,260 bytes for prefill and
3,960 calls / 1,356,960 bytes for decode. There is no teacher-append GPU kernel
or D2D copy in either source. Two original reports and eight traces are archived
outside the shared quota; all ten archive hashes match the node originals.

## Serving And Tail Investigation

Each run uses an off/on/off bracket, 512 requests per phase, 16 input and 32
output tokens, concurrency eight and fixed 10% sampling. It uses the corrected
token-based streaming client, decode graphs/overlap, a local TCP Store, an HTTP
test Catalog and a GPFS publication journal. Startup, warmup, drain and readback
are outside request timing. Profiler runs use a node-local journal and are not
substitutes for this serving measurement.

The ordinary baseline/candidate pair is measured in that order. A second,
separate diagnostic pair reverses the order and sets `SGLANG_LOG_GC=1`; its
logging changes the environment, so it must not be pooled silently with the
ordinary pair.

| Measurement | Source | Requests/s | Throughput / Off Bracket | p99 TTFT ms | p99 TPOT ms |
| --- | --- | ---: | ---: | ---: | ---: |
| Ordinary | Baseline | 76.889 | 82.75% | 58.787 | 3.409 |
| Ordinary | Candidate | 68.765 | 74.26% | 601.336 | 5.008 |
| Scheduler GC diagnostic | Candidate | 77.318 | 84.32% | 72.560 | 3.329 |
| Scheduler GC diagnostic | Baseline | 76.582 | 84.74% | 59.417 | 3.458 |

The first candidate run has a real observed regression: throughput falls from
76.889 to 68.765 requests/s and p99 TTFT rises from 58.787 to 601.336ms. Requests
488-495 start together and include both published and unpublished requests.
Request 488 has a 565.185ms TTFT followed by only 3.963ms for the remaining
response, consistent with buffered delivery. That signature alone does not
identify which process paused. The other requests also have approximately
110ms inter-token gaps. Writer aggregate timers cannot assign causality to an
individual request. The original run did not record GC events.

The reversed diagnostic does not reproduce that large spike, but still has a
higher candidate p99 TTFT. Its candidate throughput is slightly higher in raw
requests/s and slightly lower relative to its own off bracket. This is not a
consistent end-to-end speedup. The baseline's final capture-off phase itself
has a 4.278ms p99 TPOT, and its off-bracket throughput drifts by 3.32%.

Both diagnostic logs contain 71 scheduler GC completions. The longest candidate
and baseline events are 260.7ms and 332.4ms; both occur before the client's
`/model_info` request and therefore before benchmark request timing. These
events cannot explain the original run's in-flight stall. The switch only
instruments scheduler GC, not the client/tokenizer event loops. Request records
currently lack a wall-clock anchor for exact cross-process event joins. Add
that clock binding and client/event-loop pause observations before assigning
the original tail to a specific operation. No GC behavior was changed.

All four runs finish **6,144 timed requests / 196,608 output tokens** and
validate **200 snapshots** after producer exit. Each enabled phase publishes
50 samples with identical payload byte counts and capture forward counts.
Four actual Prometheus scrapes match status; there is no measured capture
failure, writer stage error, Catalog error or quarantine. All request-file
hashes and publication membership counts are verified. This validates content
and accounting, not production latency acceptance.

## Correctness

The focused suite passes **114 test methods in 53.306s**. It includes two new
cases for trimming speculative/lookahead teacher suffixes and reusing the same
Host slot with different prompt/response lengths. Existing owner partitions,
staging, CUDA fences, cancellation, recovery and publication tests are included.
The analyzer passes **five methods in 0.015s**, including same-thread containment,
nested inclusive timing and exclusion of unrelated GPU work.

Actual inference passes **two methods in 774.144s**, including AR, static/ragged
DSpark, graph/overlap, prefix reuse, memory retraction and admission recovery.
Independent P/D passes **two methods in 163.401s**, including eager and
graph/overlap modes, handoff failures/cancellation and ten complete snapshots
read after both producer trees exit. In total **123 methods** pass. These are
single-rank checks, not new distributed topology or RDMA coverage. Production
SLO, production Catalog retention and trained draft quality remain open.

All ten submitted jobs finish with exit code zero. Four changed Python files
pass Black and full Ruff, and `git diff --check` is clean. The worker returns
to its idle task after the queue drains. The change is retained for its verified
reduction of repeated metadata operations; overall performance acceptance is
still open, including the ordinary candidate regression above.
