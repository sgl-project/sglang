# Repeated Capture Within One Serving Lifetime

`soak_training_capture.py` extends the existing benchmark helpers to multiple
measured batches in the same real SGLang process. It uses ordinary AR with
overlap, decode CUDA graphs, a real local TCP Mooncake Store and the HTTP test
Catalog. The Catalog is not a production retention or consumer implementation.

Each batch uses a distinct random-input seed, four concurrent streaming
requests, 512 input tokens and 128 output tokens in the retained experiment.
Warmup and the initial cache flush happen once. No server restart, cache flush
or forced GC occurs between measured batches. Each native benchmark client is
a separate process, so there are client startup and observation gaps between
batches. This is repeated workload coverage, not uninterrupted saturation.

## Invariants And Observations

- The same serving process IDs and creation times must survive from the first
  measured batch to the last. A restart cannot reset the measured memory history.
- Each batch must complete every request and generate the configured token count.
  The producer's considered count must match the batch; every admitted capture
  must reach READY and match the Catalog. Admission backpressure is reported,
  not counted as a completed capture.
- Host/device arena sizes must remain equal to their post-warmup allocation and
  within configured budgets. No quarantined slot, pending journal entry, queued
  publication or disabled producer is allowed at a drained batch boundary.
- Actual stage metrics must agree with producer status after each batch.
- Per-process RSS, USS, PSS, threads and open file descriptors are recorded at
  drained boundaries. The predeclared RSS gate permits at most 128 MiB growth
  per process after the first measured batch. These observations do not measure
  transient allocation peaks or prove absence of all leaks.
- A test-only weak-reference probe tracks every tokenizer request state. Final
  collection checks complete request accounting, unchanged GC ownership, zero
  active requests and release of completed states. A new cycle must remain
  collectible when explicit serving GC freeze is enabled.
- After producer exit, the independent reader validates every measured READY
  manifest and all tensor bytes, including semantic checks. Publication trace
  IDs must join to that batch's native client records without duplicates.

Store segments and the test Catalog intentionally retain the entire dataset in
the experiment driver. Their growth is outside the serving process measurements
and is not evidence about production Catalog GC. The driver records exact
payload sizes and sample counts separately.

## Reproduction

Use the retained H100 worker with the pinned capture venv, a frozen source tree,
`PYTHONPATH` pointing to its `python` directory and `OMP_NUM_THREADS=1`:

```bash
python mooncake-study/experiments/soak_training_capture.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  --source-revision SOURCE_REVISION --output-dir NEW_OUTPUT_DIRECTORY \
  --batches 16 --num-prompts 128 --input-len 512 --output-len 128 \
  --concurrency 4 --capture-slots 16 --segment-mib 32768 \
  --max-rss-growth-mib 128 --gc-policy freeze-after-warmup --capture
```

Run a separate lifetime without `--capture` for the no-capture control. Both
runs must use the same source, workload, GC policy and resource budgets. The
default policy is `unchanged`; the experiment above explicitly exercises the
post-warmup freeze policy. A two-batch/eight-request capture smoke run checks
the harness before the full pair.

The report records runtime package versions, source hashes, all batch results,
per-process observations, final request-lifetime checks and post-exit readback.
A completed report means the declared finite-run checks passed. It does not
certify production SLOs, distributed/RDMA sustained load, days of memory
stability, production retention or trained draft quality.

## Verified Results

The smoke and both full lifetimes completed on the resident H100. Each full
lifetime served 2,048 measured requests across 16 batches. The repeated-load
interval, including client startup and boundary observations, was 504.408
seconds without capture and 603.085 seconds with capture. Total experiment
durations including server startup and final readback were 551.158 and 711.230
seconds respectively. These are approximately eight- and ten-minute workload
windows, not hours or days of operation.

Every one of the 2,048 selected capture requests was admitted, sealed and
published. Post-exit readback validated 139,264 tensor objects containing
16,395,010,048 payload bytes plus 68,802,560 manifest bytes. The separate smoke
validated another 16 samples / 1,088 objects / 128,086,016 payload bytes.
No writer stage failed and no admission backpressure was observed.

The registered Host arena stayed at 153,317,376 bytes (146.215 MiB), with zero
device staging and zero quarantined slots. Its 16 filling slots at each drained
boundary belong to available reservations for future requests; they are not
16 unfinished samples. The publication queue and metadata journal drained.
All 32 measured batch boundaries had zero active/live tokenizer request states,
with complete weak-reference coverage. Final explicit collection also passed.

Peak RSS growth after the first measured batch was:

| Process | Capture Off MiB | Capture On MiB |
| --- | ---: | ---: |
| Tokenizer/HTTP | 0.797 | 0.063 |
| Scheduler | 0.000 | 0.258 |
| Detokenizer | 0.871 | 0.254 |
| Multiprocessing resource tracker | 0.000 | 0.000 |

All remain within the predeclared 128 MiB per-process budget. Thread counts
remain fixed; file descriptor counts never increase from the settled baseline
(the capture scheduler falls from 108 to 107). These are boundary measurements
and include the test-only request lifetime probe.

Aggregate native-client throughput is 7.236 requests/s without capture and
5.383 with 100% capture, approximately 74.4% of the control. This single pair
shows residual capture overhead; it does not establish a serving speedup or
production SLO acceptance. Client timing excludes inter-batch startup and
observation gaps as well as post-producer readback.

[The evidence JSON](capture-soak.json) records all three terminal worker jobs,
source hashes, full memory observations and cleanup. All 5,046 Python files
match the frozen full-run source. The new driver passes full Ruff and format
checks. The initial hardlink copy could not link root-owned bytecode caches;
the full runs use a separately completed, source-audited rsync snapshot with
caches excluded. This preparation issue did not require changing assertions.

The archive contains 151 artifacts at
`/gpfs/user/fuxuanwei/mooncake-lab-archive/capture-soak-20261003`, with manifest
SHA-256 `db97b6974ff4ba5874e3cb6a45fae35be32d1a0e1e427bd92c917b72995af275`.
Live-process and GPU inspection confirm that experiment/model/Store processes
have exited and only the resident worker's idle CUDA load remains. No additional
GPU was allocated.
