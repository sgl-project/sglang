# Optional Teacher Top-128 Backend

The capture config accepts `"teacher_topk_backend": "flashinfer"`. The default
is `"torch"`; existing configs keep their previous behavior. Startup warms the
selected FP32 implementation inside the binding vote, before request admission
and before a prefill-only coordinator returns. The backend is included in the
policy fingerprint and distributed startup agreement.

The optional path uses the pinned FlashInfer 0.6.17 top-k dispatch for
nonempty CUDA FP32 inputs with at least 32,768 unpadded vocabulary entries.
Other dtypes, CPU inputs, empty selections and smaller vocabularies use Torch.
The preliminary H100 probe found a BF16 regression at vocabulary 151,936, so
the option does not change the BF16 implementation.

## Ownership And Numerical Contract

FlashInfer's public wrapper caches a device-wide scratch buffer. Capture calls
its pinned internal `get_topk_module().radix_topk` interface with a separately
allocated, zeroed 1 MiB workspace for each invocation. Inputs with padding or
noncontiguous strides are made contiguous for that implementation. These
allocations and any input copy are included in complete-extraction timing.
No persistent shared scratch or host synchronization is added to the hot path.
Upgrading FlashInfer requires checking this internal interface and rerunning
the numerical, graph and concurrent-stream tests.

Outputs retain raw score values, unique int32 vocabulary IDs, sorted top-128
values and the existing FP32 full-vocabulary logsumexp. They own storage that
survives sampler mutations and graph-buffer reuse. Equal-valued boundary tokens
may differ from Torch's IDs, as permitted by the protocol; the optional backend
deterministically prefers lower IDs. Tests compare sorted values and gather the
original scores by the returned IDs. Nonfinite scores are not repaired into
apparently valid training data; existing publication validation remains active.

AR, speculative verification and the P/D first-teacher handoff use the same
config selection. Token alignment, KV selection, loss masks, publication and
Mooncake object formats are unchanged.

## Reproduction

Use the resident H100 queue and the runtime lock in this directory, with a
frozen source checkout selected through `PYTHONPATH`. Do not run alongside the
idle load. Use the same source and instrumentation for both serving variants.

```bash
python -m unittest test_teacher_cuda test_context test_coordinator \
  test_pd_capture test_config test_cohort_startup test_startup \
  test_trace_attribution -v
python mooncake-study/experiments/benchmark_teacher_capture.py \
  --output /results/teacher-topk-micro.json --source-revision CHECKOUT_REVISION \
  --topk-backend flashinfer --reference-backend torch \
  --rows 1 8 32 --iterations 20 --repeats 3
python test/registered/storage/test_training_capture_runtime.py \
  --model-path /models/Qwen3-0.6B --teacher-topk-backend flashinfer -v
env TRAINING_CAPTURE_TEST_MODEL=/models/Qwen3-0.6B \
  python test/registered/storage/test_training_capture_pd.py \
  --teacher-topk-backend flashinfer -v
python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /models/Qwen3-0.6B --source-revision CHECKOUT_REVISION \
  --output-dir /results/teacher-topk-flashinfer \
  --num-prompts 1024 --input-len 16 --output-len 32 --concurrency 8 \
  --capture-slots 16 --ratios 0.1 --repeats 1 --request-details \
  --latency-diagnostics --gc-lifecycle --gc-policy freeze-after-warmup \
  --teacher-topk-backend flashinfer
python mooncake-study/experiments/profile_training_capture.py \
  --model-path /models/Qwen3-0.6B --source-revision CHECKOUT_REVISION \
  --output-dir /tmp/teacher-topk-profile-flashinfer \
  --steps 2 --warmup-steps 3 --teacher-topk-backend flashinfer
```

Repeat the serving/profile commands with `--teacher-topk-backend torch` and
fresh output directories. Serving measurement keeps post-producer-exit Store
readback and exact request/publication accounting. Profiling verifies READY
counts and kernel attribution; it is not an independent tensor-content oracle.

The trace summarizer records complete CUDA runtime and driver API events
contained in each disjoint capture scope, including calls without associated
device activity. API durations are inclusive CPU time and nested calls overlap;
they are neither GPU time nor an estimate of removable wall-clock latency.

## Measurement Scope

The 24-case complete-extraction run covers FP32/BF16, vocabularies 32,768 and
151,936, 1/8/32 output rows, and full versus selected input rows. Sorted score
values match exactly and maximum LSE difference is zero against the same-LSE
Torch reference. Selected FP32 medians in microseconds, for the full-row cases:

| Vocabulary | Rows | Torch Eager | FlashInfer Eager | Torch Graph | FlashInfer Graph |
| --- | --- | --- | --- | --- | --- |
| 32,768 | 1 | 137.287 | 82.995 | 56.906 | 27.966 |
| 32,768 | 8 | 144.695 | 92.516 | 63.020 | 30.543 |
| 151,936 | 1 | 133.874 | 82.905 | 82.508 | 67.267 |
| 151,936 | 8 | 143.309 | 92.660 | 99.142 | 71.368 |

The CUDA test matrix separately covers four dtypes, vocabulary padding and
strides, empty/contiguous/repeated/reordered selections, equal-valued boundaries,
large offsets, nonfinite inputs, independent concurrent streams and independent
graph captures/replays. Overwriting the raw source must not change captured
values. The first 111-method run had one stale startup test fixture lacking
the new config field; all nine teacher CUDA methods passed. After correcting
that fixture, its focused regression passed and confirms that the configured
backend reaches warmup inside the startup failure vote.

The full real-model runtime suite passes both methods in 774.041 seconds with
the optional backend: ordinary AR, static target-KV DSpark, confidence-scheduled
`cap-accept`/`compact` verification, eager/graph and overlap modes, raw source
comparison, prefix reuse, cancellation, adaptive admission and memory lifecycle.
The separate P/D suite also passes both methods and validates ten snapshots
after both producers exit, including eager and decode-graph/overlap handoff,
handoff fault exclusion and cancellation. These are actual SGLang and Mooncake
processes, with synthetic draft weights and a test Catalog. The separate HF KV
diagnostic still reports the previously documented cross-engine differences;
the capture oracle is the exact tensor from the online forward.

The two serving brackets run in Torch-then-FlashInfer order, each with 1,024
requests per off/on/off phase, 16 input tokens, 32 output tokens, concurrency
eight and sample ratio 0.1. Both use the same explicit post-warmup GC freeze
and diagnostics. Each on phase publishes and validates 92 samples after
producer exit; all six phases total 6,144 requests and 196,608 output tokens.

| Backend | Off Before RPS | Capture RPS | Off After RPS | Capture / Off Mean | Capture p99 TTFT ms | Capture p99 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- |
| Torch | 92.293 | 78.818 | 92.509 | 85.30% | 58.740 | 3.394 |
| FlashInfer | 91.656 | 77.655 | 92.520 | 84.33% | 56.364 | 3.458 |

This pair establishes no serving throughput improvement. It is one short
workload in one order, not a significance test or SLO acceptance. Sampling uses
the same seeded producer RNG; random client request IDs are correlation IDs,
not the sampling decision. Warmup and steady-state comparisons use warmed JIT
caches. Cold compilation, additional models, mixed/distributed topologies and
long-duration behavior are not covered by this serving comparison. The backend
therefore remains opt-in and Torch remains the default.

Both profiler runs complete 16 measured requests per prefill/decode workload
in each off/on mode. Their on modes reach READY for all 64 measured samples in
total. Named teacher scopes cover 18 prefill calls and 66 decode calls for each
backend. In decode, total teacher kernel calls fall from 1,518 to 396, or
23 to six per call; summed teacher GPU work falls from 5.913 ms to 4.950 ms.
Teacher CPU scope time under profiler falls from 32.587 ms to 24.864 ms.
The actual `radix_topk` entry point dispatches `FilteredTopKUnifiedKernel` for
this workload, followed by index/value sorting, plus scratch fill and the
unchanged LSE kernels. These are observed kernel names, not an assumption that
every input uses a radix kernel. CPU/API/device durations are not additive
serving latency; the separate serving comparison above remains authoritative
for this workload's throughput result.

All ten experiment jobs and the offline audit are terminal. The initial stale
fixture failure is retained; its corrected regression and all other jobs pass.
The audit replays all six phases' request summaries and pause correlations,
checks source and artifact hashes, and matches the 184 publications to complete
post-exit reads (114,727,680 payload bytes). No observed >=20 ms diagnostic
event overlaps these measured requests. Original traces, per-scope summaries,
commands, results and limitations are bound in [the evidence](teacher-topk.json).

The complete teacher microbenchmark compares the current Torch top-k path with
the optional backend, keeping the LSE implementation identical. It alternates
measurement order and reports eager wall time and batched CUDA-graph time.
It does not establish serving throughput or a production SLO. The resident
single-H100 experiments use Qwen3-0.6B, synthetic inputs, a local TCP Store and
an HTTP Catalog test double. Production retention, deployed model coverage,
SpecForge integration and trained-draft quality require separate evidence.
