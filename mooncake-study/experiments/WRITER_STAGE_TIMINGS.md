# Background Writer Stage Timings

The producer exposes bounded cumulative `stage_timings` in its existing status
and four Prometheus families through the existing metrics thread. The snapshot
writer, single-rank coordinator and cohort writer share the same local counters.
No request identity or tensor contents enter the timing state. No CUDA event,
extra GPU synchronization or inference-thread Prometheus work is added.

## Measurement Contract

Each stage records `calls`, `errors`, `seconds` and `max_seconds`. An attempt is
recorded when it returns or raises, and the original return value or exception
is preserved. Retries are new attempts. The maximum is process-lifetime, not an
interval maximum or latency quantile. Timing is host wall time, including
thread scheduling and GIL waits; it does not isolate CPU execution time.

| Stage | Included work |
| --- | --- |
| `queue_wait` | Enqueue to first background writer processing |
| `copy_wait` | Existing wait for outstanding capture copies to finish |
| `snapshot_build` | Context snapshot or owner partition preparation |
| `validation` | Snapshot writer tensor/manifest validation calls |
| `catalog_register` | REGISTERED descriptor requests |
| `store_payload` | Registered payload batch adapter, including local validation and retry reads |
| `catalog_written` | Payload and manifest WRITTEN receipt requests |
| `journal_save` | Existing metadata journal save, file and directory durability operations |
| `catalog_seal` | Catalog prepare/seal request and receipt check |
| `store_manifest` | Registered manifest adapter write or immutable retry |
| `catalog_publish` | Catalog publish request and receipt check |
| `journal_complete` | Metadata unlink and directory durability operation |
| `recovery_read` | Readback of all payloads for one publication recovery attempt |

These stages do not cover all capture costs. GPU top-k/LSE, D2H transfer time,
serving interference, allocation/reservation RPCs, distributed peer waits,
collectives and some metadata work need separate measurements. `copy_wait`
is the remaining wait when the writer reaches the sample, not the complete
copy duration. A stage with no calls has no completed observations. Pending
work remains visible through writer age and queue/ownership states.

All rank-local timing snapshots are detached and protected by a dedicated
short-held lock. The lock is never held during the measured operation. Cohort
admission and pressure checks request stats without copying timing dictionaries.
Prometheus receives cumulative deltas once per metrics update, so repeated
snapshots do not double-count. Labels are restricted to the 13 known stages.

The dashboard adds mean completed-attempt duration and failure-rate panels.
They aggregate local operation counts, not global sample counts. Do not add
stage time across ranks or include queue wait to infer end-to-end latency.
Live Grafana rendering and deployment acceptance remain separate checks.

## Reproduction

Use the resident H100 and [pinned runtime](h100-runtime-lock.json). Baseline is
`4673ddb2c7aced317caddf4012f9daa3535a170d`, which already includes native Store
payload batching. Frozen sources are:

- `/gpfs/users/fuxuanwei-1/dspark-maas-lab/sglang-timings-before`: baseline producer
  with the new benchmark driver only.
- `/gpfs/users/fuxuanwei-1/dspark-maas-lab/sglang-timings-v1`: final producer and
  unit/Store tests, before the driver's actual `/metrics` verification was added.
- `/gpfs/users/fuxuanwei-1/dspark-maas-lab/sglang-timings-v2`: same producer and tests,
  with the final driver. Source is never overwritten while jobs use it.

Set `PYTHONPATH` to the source's `python` directory and use the lab capture venv.
The worker suspends the idle load while serializing these jobs:

```bash
CUDA_VISIBLE_DEVICES=999 python test/registered/unit/training_capture/test_metrics.py -f
CUDA_VISIBLE_DEVICES=999 python -m unittest discover \
  -s test/registered/unit/training_capture -p 'test_*writer.py' -f
CUDA_VISIBLE_DEVICES=999 python -m unittest discover \
  -s test/registered/unit/training_capture -p 'test_*coordinator.py' -f
python test/registered/storage/test_training_snapshot_mooncake.py -f
python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  --source-revision SOURCE_REVISION --output-dir NEW_OUTPUT_DIRECTORY \
  --num-prompts 512 --input-len 16 --output-len 2 \
  --concurrency 4 --capture-slots 4 --ratios 1.0 --repeats 2
python test/registered/storage/test_training_capture_runtime.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  TestTrainingCaptureRuntime.test_chunk_prefix_single_token_and_raw_teacher_reference -f
```

The before and after benchmarks use identical driver bytes and runtime. Each
capture-on phase is bracketed by capture-off phases, with warmup excluded from
client timing and stage deltas. Every measured READY snapshot is read and
validated after producer exit. The new driver checks all four timing families
against the final capture status through the actual multiprocess `/metrics`
endpoint after client timing stops, retaining each successful `metrics.prom`.
The baseline reports timing metrics unavailable; this is expected, not a failed
metrics check. The instrumented producer must pass the endpoint check.

Tests cover exact deterministic durations, exception/result preservation,
concurrent updates, detached snapshots, bounded labels, repeated export,
publication failure, journal recovery and cohort timing propagation. Existing
writer/ownership, real Store and inference checks retain their original data
and lifecycle assertions.

## Regression Results

All 87 test methods pass with the final producer implementation:

| Suite | Methods | Seconds | Job |
| --- | --- | --- | --- |
| Timings and Prometheus | 8 | 0.034 | `01790944299544410488-7daeaa9f798b` |
| Snapshot/cohort writers | 23 | 5.711 | `01790944299871611741-78c066567902` |
| Direct/staged/cohort coordinators | 49 | 37.008 | `01790944300246853476-25bd73da3706` |
| Actual Store and multiprocess publication | 6 | 175.354 | `01790944300563078422-0177dd3d67a4` |
| Actual inference, retraction and admission | 1 | 505.947 | `01790944503551940825-fd652280850c` |

The inference method exercises AR eager/graph, overlap, target-KV DSpark,
actual memory retraction and adaptive/latency recovery with online source
comparison and post-exit Store reads. These are functional regressions, not
the client-latency performance gate. Ten edited/new Python files pass Black
and Ruff I/F, and full Ruff adds no warnings beyond the coordinator's nine
pre-existing BLE001 diagnostics. Dashboard JSON parses with unique panel IDs;
no live Grafana rendering or production dashboard acceptance is claimed.

## Short-Request Results

Both sources completed two off/on/off rounds. Each capture-on phase has 512
requests, 16 input/two output tokens, concurrency four and four Host slots.
The sources use identical driver bytes and pinned runtime; the changed producer
modules are precisely the six timing/integration modules. Both allocate
5,155,584 registered Host bytes with no device staging.

| Source | Round | READY | READY/s | Requests/s | Throughput vs off | p99 TTFT ms | p99 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1 | 295 | 40.26 | 69.87 | 75.45% | 62.63 | 59.61 |
| Baseline | 2 | 320 | 42.87 | 68.59 | 70.30% | 67.34 | 59.19 |
| Timed | 1 | 295 | 38.52 | 66.85 | 71.22% | 118.64 | 58.11 |
| Timed | 2 | 324 | 41.75 | 65.97 | 67.84% | 126.68 | 63.18 |

Every admitted measured request reached READY, and all 1,234 snapshots passed
post-exit readback. There were no stage errors or recovery reads in the measured
timed phases. All 13 stage counters match publication counts: normal stages have
one attempt per sample, `catalog_written` has two, and `recovery_read` has zero.
Both timed phases pass the actual `/metrics` comparison for all four fields.

Across both capture-on phases, baseline yields 41.57 samples/s and 69.22 requests/s;
timed yields 40.14 samples/s and 66.41 requests/s. Observed request throughput is
4.07% lower, with higher p99 TTFT in both timed phases. This experiment does not
establish negligible instrumentation overhead or pass a serving SLO. Two short
sequential repetitions cannot separate instrumentation from run-order/system
effects or explain the latency tail; these costs need follow-up investigation.

The 619 timed snapshots give the following weighted wall-time means. Fractions
are of the measured execution stages only, excluding queue wait and unmeasured
work; they are not fractions of serving latency or CPU utilization.

| Measured work | ms/sample | Share of measured execution |
| --- | --- | --- |
| Snapshot construction and writer validation | 6.695 | 38.74% |
| Catalog registration, receipts, seal and publish | 4.624 | 26.76% |
| Journal save and completion | 4.058 | 23.48% |
| Store payload and manifest adapters | 1.811 | 10.48% |
| Outstanding copy wait | 0.094 | 0.54% |

Store payload alone averages 1.466 ms/sample. Queue wait separately averages
21.471 ms/sample and can overlap another sample's execution. The stage breakdown
supports examining construction/validation and control/durability work before
assuming additional native Store batching will improve this workload.
`build_snapshot()` validates payloads before returning, and `SnapshotWriter`
validates them again at its publication boundary. A future optimization must
retain correctness and immutable-source checks at that boundary; the timing
change itself does not remove any validation or durability operation.

## Longer-Sequence Diagnostic

A further off/on/off run uses the timed source with `--num-prompts 64
--input-len 512 --output-len 128 --repeats 1`; all other options are identical.
It publishes and validates 52 of 64 selected requests after producer exit,
covering 3,536 payload objects and 416,279,552 tensor bytes. All admitted requests
reach READY without stage errors or quarantine. The real metrics check passes
again, and normal stage attempt counts match the 52 publications.

The capture-on phase serves 4.214 requests/s and produces 3.424 samples/s,
retaining 59.26% of its bracketing capture-off throughput. Its p99 TTFT is
984.57 ms and p99 TPOT 11.11 ms. This single diagnostic is not a representative
production workload or an instrumentation-overhead comparison: there is no
long-sequence baseline producer run. Host allocation is 38,329,344 bytes, with
no device staging.

| Measured work | ms/sample | Share of measured execution |
| --- | --- | --- |
| Snapshot construction and writer validation | 64.202 | 66.46% |
| Catalog calls | 14.431 | 14.94% |
| Journal operations | 4.805 | 4.97% |
| Store adapters | 12.906 | 13.36% |
| Outstanding copy wait | 0.260 | 0.27% |

Covered execution totals 96.604 ms/sample; queue wait is separately 86.537 ms.
Store payload alone is 12.514 ms/sample. The increased construction/validation
cost strengthens the case for profiling that work. These host stages still do
not explain GPU capture cost or predict the serving benefit of an optimization.

## Retained Evidence

The [evidence index](writer-stage-timings.json) binds all eight successful,
terminal jobs; logs/results; three raw reports; three actual metrics scrapes;
the pinned runtime; and 15 executable/dashboard source files. The final frozen
tree matches all listed sources. The initial tree differs only in its benchmark
driver, which was not used by the unit/Store tests. Documentation was added
after freezing.

All 1,286 measured benchmark snapshots pass post-exit readback, including 671
from the timed source. No additional GPU was allocated. The resident H100 has
resumed its idle workload with an empty queue. The original checkout's staged
index remains unchanged. Performance, RDMA, distributed timing at serving scale,
production Catalog behavior and live dashboard acceptance remain open gates.
