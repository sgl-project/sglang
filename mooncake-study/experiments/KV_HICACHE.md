# Optional HiCache KV Export

Capture accepts `"kv_export_backend": "hicache"` with a positive
`max_device_bytes` budget. The default remains `"torch"`. This option reuses
SGLang's existing HiCache JIT byte-row transfer kernel for selected K/V buffers.
It does not enable the serving L2 cache or move training data into its LRU pool.
Capture retains its own registered Host slots, completion events, publication
queue and Mooncake object contract.

## Storage And Lifetime

Each active KV-owning Host slot owns device pointer tables for source, Host and
optional staging buffers, plus int32/int64 destination positions. These tensors
share the slot's bounded device arena with KV and teacher staging. The complete
allocation across slots must fit `max_device_bytes` before Host registration.
Aux-only partitions need no KV metadata. Binding checks layouts and initializes
the JIT and pointer tables inside resource startup, before capture admission.

Sources must be unquantized NHD CUDA rows, with contiguous head/dimension
storage, 16-byte-aligned addresses/row strides and row sizes divisible by 128.
Different K/V dimensions and padded source-row strides form separate kernel
groups. Unavailable JIT support or incompatible layouts reject startup for this
explicit backend. The implementation uses the existing JIT kernel; its
`all_layer_mla` function name describes a pointer-table byte-copy interface,
not the training snapshot's codec or model architecture.

The exporter validates position shapes and destination bounds and checks source
index bounds asynchronously before launching the raw-pointer kernel. Calls keep
the producer stream's ordering and protect metadata/index storage across streams.
Direct mode writes mapped pinned Host storage from the kernel. Batched KV mode
can instead gather to owned device staging and use the existing full/tail D2H
flush. Large prefill ranges can bypass staging. Completion uncertainty keeps
both Host storage and GPU metadata quarantined; the writer waits for completion
before inspecting or publishing data.

`host_pool.kv_export_host_enqueued_bytes` and
`host_pool.kv_export_device_enqueued_bytes` count bytes queued by the bound
HiCache exporter. They are zero for the Torch backend. They are not completed
DMA, RDMA, wire or published-payload counters. With staging, the latter counts
gather traffic before the existing D2H flush. Python counters count enqueue
calls, not standalone CUDA graph replays of a captured exporter. The serving
hooks execute these calls outside the model graph.

The profiler driver records measured byte-counter deltas after warmup. Mapped
Host writes appear as kernel activity and need not create a `Memcpy DtoH`
event; a zero memcpy count therefore does not establish zero Host traffic.

## Reproduction

Use the locked H100 environment and a frozen checkout through the resident queue.

```bash
python -m unittest test_kv_hicache test_buffers test_context test_kv_staging \
  test_teacher_staging test_resources test_config test_partition_context \
  test_coordinator test_cohort_startup test_pd_capture test_trace_attribution -v
python mooncake-study/experiments/benchmark_kv_export.py \
  --output /results/kv-export.json --source-revision CHECKOUT_REVISION
python test/registered/storage/test_training_capture_runtime.py \
  --model-path /models/Qwen3-0.6B --kv-export-backend hicache -v
env TRAINING_CAPTURE_TEST_MODEL=/models/Qwen3-0.6B \
  python test/registered/storage/test_training_capture_pd.py \
  --kv-export-backend hicache -v
python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /models/Qwen3-0.6B --source-revision CHECKOUT_REVISION \
  --output-dir /results/kv-hicache-serving \
  --num-prompts 1024 --input-len 16 --output-len 32 --concurrency 8 \
  --capture-slots 16 --ratios 0.1 --repeats 1 --request-details \
  --latency-diagnostics --gc-lifecycle --gc-policy freeze-after-warmup \
  --kv-export-backend hicache --device-mib 16
python mooncake-study/experiments/profile_training_capture.py \
  --model-path /models/Qwen3-0.6B --source-revision CHECKOUT_REVISION \
  --output-dir /tmp/kv-hicache-profile --steps 2 --warmup-steps 3 \
  --kv-export-backend hicache --device-mib 16
```

Repeat serving/profiling with `--kv-export-backend torch`, the same positive
budget and fresh output directories. Teacher top-k and teacher staging remain
at their defaults in this comparison. Serving brackets validate every published
sample after producer exit; profiles check activity and READY counts, not tensor
content independently.

The 24-case complete-export microbenchmark includes Python validation, async
bounds checks, index handling and metadata views. It alternates backend order
and reports eager wall/stream time plus batched graph replay time for BF16/FP16,
int32/int64 indices, 1/8/128 rows and Host/device destinations. Both backends
must preserve fresh source values after overwrite and leave guard rows intact.
Graph timings exclude Python enqueue work and are not serving latency.

The earlier primitive-only probe prebinds metadata and omits the full exporter
checks. Its overwrite check is strong only for the first case, because later
cases reuse a source already filled with the overwrite value. Retain that probe
as exploratory evidence; use the fresh-source unit tests and complete-export
benchmark for source-reuse correctness and production-path timing.

## Correctness Evidence

The initial focused suite passes 140 methods in 130.466 seconds. The new CUDA
cases include BF16/FP16, different K/V widths, padded source strides, duplicate
and noncontiguous indices, independent concurrent graphs/streams, staged tails,
budget rejection before registration, invalid destination ranges and completion
failure quarantine. Out-of-range GPU indices run in isolated subprocesses so
device assertions cannot poison the remaining tests.

The first FP16 unit/micro fixtures used a BF16 codec label while exercising FP16
storage. The final fixtures set the matching codec and call `validate_kv_spec`
before allocation. The original frozen jobs are retained and the corrected
fixtures are rerun; production exporter code is identical across the versions.

Both complete real-model runtime methods pass in 754.566 seconds, including
AR, static target-KV DSpark, confidence-scheduled verification, overlap/graphs,
source-tensor comparison, cancellation, adaptive admission and memory pressure.
Both P/D methods pass in 174.873 seconds, including ten complete Store snapshots
read after the two producers exit. Handoff faults and cancelled requests are
excluded from publication. Runtime suites enable KV/teacher staging; the serving
and profile comparisons use direct KV/teacher copies instead.

This evidence uses a single H100, Qwen3-0.6B, synthetic draft weights, local TCP
Mooncake Store and an HTTP Catalog test double. This backend still needs separate
validation for additional models, distributed topology/RDMA combinations,
production retention, consumer integration and representative service SLOs.

## Serving Comparison

Each backend runs one off/on/off bracket with 1,024 requests per phase, 16 input
tokens, 32 output tokens, concurrency eight and capture ratio 0.1. Torch runs
first; both use the same source, instrumentation, device budget and explicit
post-warmup GC freeze. The producer uses seeded RNG sampling; random client
request IDs serve only to correlate measurements with published manifests.
Both on phases publish and independently validate 92 samples after producer
exit, totaling 6,144 requests, 196,608 output tokens and 184 snapshots across
the six phases. Payload validation covers 114,727,680 tensor bytes.

| Backend | Off Before RPS | Capture RPS | Off After RPS | Capture / Off Mean | Capture p99 TTFT ms | Capture p99 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- |
| Torch | 92.817 | 76.960 | 91.978 | 83.29% | 55.788 | 3.441 |
| HiCache | 91.525 | 78.209 | 92.628 | 84.94% | 58.583 | 3.497 |

Capture throughput is 1.62% higher in this pair, while both reported p99
latencies are slightly worse. The test is one short workload in one order,
with warm JIT caches; it does not establish statistical significance, sustained
service improvement or production SLO acceptance. Torch remains the default.

## Complete-Export Timing

The corrected six-method CUDA suite passes in 23.991 seconds, and all 24
complete-export cases pass. Across the twelve Host-destination cases, eager
wall time falls by 51.6-57.7%. Selected BF16/int32 median times in microseconds:

| Rows | Torch Host Eager | HiCache Host Eager | Torch Host Graph | HiCache Host Graph |
| --- | --- | --- | --- | --- |
| 1 | 139.370 | 67.159 | 79.990 | 9.042 |
| 8 | 140.103 | 66.310 | 91.227 | 11.645 |
| 128 | 155.871 | 66.487 | 104.242 | 37.703 |

The device-destination path has a counterexample: at 128 BF16/int32 rows,
batched graph time rises from 8.805 to 24.712 microseconds, even though eager
wall time falls from 96.949 to 65.913. Graph replay omits the Python/dispatch
costs included in eager export. Neither result alone predicts serving latency.

The initial offline auditor incorrectly assumed every request exports exactly
`prompt + response - 1` KV rows. In the measured overlap workload, prefill
exports two additional terminal rows across 16 requests, and decode exports
one additional row per request. Torch's attributed D2H trace independently
reports 25,190,400 prefill bytes and 6,488,064 decode bytes, exactly matching
the HiCache Host enqueue counters. The corrected auditor checks this equality,
the number of export calls and the bounded lookahead range. It preserves the
original failed audit; production code and experimental outputs are unchanged.

## Trace Evidence

Both profilers complete 16 measured requests per off/on prefill/decode workload;
their on modes reach READY for all 64 measured samples in total. KV scopes
contain 18 prefill exports and 528 decode exports for each backend. Under the
profiler, summed decode KV CPU scope time falls from 223.052 to 123.286 ms.
Torch records 3,168 gather kernels (6.161 ms) and 3,168 D2H copies (8.263 ms).
HiCache records 2,640 kernels (6.184 ms), including index checks and mapped Host
stores, and zero KV-scope D2H memcpy events. The Host byte volume is identical,
as checked above. Prefill KV CPU scope time falls from 10.436 to 6.168 ms;
kernel work rises from 0.199 to 0.690 ms while 0.815 ms of D2H activity disappears.
These are summed scoped work measurements, not additive serving wall time.

All eleven experiment jobs pass. The original audit fails on the documented
lookahead accounting assumption; the corrected audit passes and replays all
six phases' request summaries, publication joins and pause diagnostics. Final
source hashes match the frozen checkout, and the eight raw traces plus two
profile reports are independently hash-checked in the persistent local archive.
Commands, job outcomes, source identities, artifact hashes, trace summaries and
limitations are in [the evidence](kv-hicache.json). The resident H100 returns
to its idle task after the experiment queue drains.
