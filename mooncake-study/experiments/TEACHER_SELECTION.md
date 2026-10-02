# Host-Known Teacher Row Selection

Ordinary capture already knows selected batch rows on the scheduler CPU. P/D
prefill likewise knows the one row that predicts the first response token.
Previously both paths uploaded an index tensor and gathered complete vocabulary
rows before extracting compact teacher outputs, even for a single contiguous
range. `capture_teacher` now accepts a list of host indices and uses a narrow
view for consecutive rows. Empty selections produce empty outputs. Out-of-range,
negative and non-integer host indices fail before CUDA work.

Nonconsecutive, reordered and duplicate host indices keep the existing tensor
upload and gather. Tensor indices, including speculative verify mappings, retain
their existing path. Top-128 IDs/raw values and full-vocabulary LSE own their
storage; no view into serving logits escapes. Padding exclusion, stream order,
teacher D2H, full publication validation and the Store contract are unchanged.

## Reproduction

Use the pinned environment in `h100-runtime-lock.json`, with `PYTHONPATH=python`
and the matching Mooncake SDK/master on PATH. The serving comparison uses a
local TCP Store and HTTP test Catalog on the resident H100.

```bash
PYTHONPATH=python:test/registered/unit/training_capture python -m unittest \
  test_teacher_cuda test_buffers test_cuda_snapshot test_coordinator test_pd_capture

python mooncake-study/experiments/benchmark_teacher_selection.py \
  --source-revision CHECKOUT_REVISION --output NEW_MICROBENCHMARK_JSON

python test/registered/storage/test_training_capture_runtime.py \
  --model-path /models/Qwen3-0.6B
TRAINING_CAPTURE_TEST_MODEL=/models/Qwen3-0.6B \
  python test/registered/storage/test_training_capture_pd.py

python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /models/Qwen3-0.6B --source-revision CHECKOUT_REVISION \
  --output-dir NEW_SERVING_DIRECTORY --num-prompts 512 \
  --input-len 16 --output-len 32 --concurrency 8 --capture-slots 16 \
  --ratios 0.1 --repeats 2
```

The microbenchmark compares the previous caller's tensor-index construction plus
the current extraction algorithm against host-list selection. Both use the same
single-pass LSE. It alternates measurement order, checks exact IDs, raw scores
and LSE, and includes index preparation and a final synchronization in eager
wall time. Separate profiler probes count index uploads, stream synchronization
and gather calls. They are excluded from timings.

The serving driver brackets each enabled phase with capture-off servers, excludes
startup/warmup/drain/readback from request timing, and validates every measured
publication after the producer exits. This comparison does not establish trained
draft quality, production Catalog retention, RDMA or a production SLO.

## Serving Results

These historical measurements use the earlier text-gated native benchmark
client. A later [request-level measurement](D2H_LATENCY.md) found that empty
decoded text could suppress token timing. The original numbers below remain
unchanged; they are not corrected latency estimates.

The baseline is `44645fea3b7ebe44a240e96a052b61e7d6be0269`, frozen at
`$LAB/sglang-teacher-selection-before`; candidate serving uses
`$LAB/sglang-teacher-selection-v2`. Only teacher selection and its AR/P-prefill
call sites change production behavior. Qwen3-0.6B BF16 runs overlap scheduling,
decode graphs and eager prefill. Both sources use two off/on/off rounds, 512
requests per phase, 16 input/32 output tokens, concurrency eight, 16 Host slots
and fixed 10% sampling. Device staging remains disabled.

| Source / Round | Requests/s | Throughput / Off Bracket | Median TPOT ms | p99 TTFT ms | p99 TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline / 0 | 64.28 | 73.13% | 3.090 | 61.97 | 3.458 |
| Baseline / 1 | 64.70 | 70.30% | 3.061 | 61.97 | 3.404 |
| Candidate / 0 | 75.83 | 85.93% | 1.942 | 58.34 | 5.503 |
| Candidate / 1 | 77.18 | 82.85% | 1.943 | 58.12 | 3.404 |

Mean capture-on throughput rises from 64.49 to 76.51 requests/s (18.63%) for
this workload. Each enabled phase selects and publishes exactly 50 samples,
with identical token counts, payload sizes and eager/graph capture-forward
counts. All **200 snapshots** pass post-exit readback, and all four actual
Prometheus scrapes agree with stage status. No measured capture failure,
admission backpressure, quarantine, Catalog error or stage error occurs.

The first candidate round's p99 TPOT increases to 5.503ms, although the second
round returns to baseline. Both sources' first off phase also has a roughly
543-547ms p99 TTFT, compared with roughly 54-63ms in their second round's off
phases. Within-bracket throughput drift reaches 10.92%. These observations do
not establish uniformly improved tails or production acceptance. Request timing
does not identify which requests cause the tail; the test does not attribute it
to the row-selection change, Catalog or filesystem activity.

## Correctness And Measurement Controls

The focused CUDA/CPU suite passes **104 methods in 53.938s**, including host
selections on strided FP16/BF16/FP32 inputs, empty/duplicate/reordered rows,
invalid host indices, source overwrite and CUDA graph replay. Existing capture
ownership, coordinator cancellation/expiry/publication and P/D handoff failure
tests remain included. An initial invocation failed during import because the
installed Python `test` package shadowed the repository namespace; the explicit
test-directory `PYTHONPATH` above fixes test loading without code changes.

The first microbenchmark's CPU-only profile could observe gather calls but
could not certify CUDA runtime call counts. Enabling CUDA profiling between
cases then produced a persistent timing step after the first probe. The final
driver initializes CUDA profiling only after all timing cases have finished.
These intermediate runs remain in the evidence index and are not the final
timing result. Frozen mirrors v1/v2/v3 have identical production and test code;
only the microbenchmark's profiling schedule changes.

The final 16-case microbenchmark covers FP32/BF16, vocabularies 32,768/151,936,
eight input rows and single/three/all-eight contiguous or three disjoint rows.
IDs, raw values and LSE match the old tensor-selection path bitwise in every
case. Contiguous extraction reduces eager wall time by **8.14%-15.04%**.
Disjoint selection adds **1.95%-5.36%** from host validation/dispatch; it retains
the original upload/gather work. At FP32/vocabulary 151,936:

| Selected Rows | Tensor Indices us | Host Indices us |
| --- | ---: | ---: |
| `[3]` | 158.83 | 136.47 |
| `[2, 3, 4]` | 160.36 | 146.12 |
| `[0, ..., 7]` | 162.94 | 149.14 |
| `[0, 3, 7]` | 160.89 | 165.04 |

Separate CPU+CUDA probes observe one `cudaMemcpyAsync`, one
`cudaStreamSynchronize` and one `aten::index_select` per tensor-index call.
All three disappear for contiguous host selection, and remain for disjoint
selection. These counts describe extraction only, not the rest of serving.

Actual inference passes two methods in **621.946s**, including AR,
static/ragged DSpark, overlap/graphs, prefix reuse, memory retraction and
admission/latency recovery. Independent P/D passes eager and graph+overlap in
**133.555s**, including handoff faults/cancellation and ten complete post-exit
snapshots. Together with the focused suite, **108 methods** pass. These new
runs are single-rank; they do not extend distributed-topology coverage.

All ten submitted jobs are terminal: nine succeed and the initial test-loader
invocation fails before collection. No additional GPU is allocated, and the
resident worker resumes idle load. Five Python files pass Black; four pass full
Ruff, and the coordinator retains its nine baseline BLE001 findings with clean
I/F checks. The [evidence index](teacher-selection.json) binds commands, sources,
reports, metrics and logs. Production target/SLO and Catalog retention gates
remain open.
