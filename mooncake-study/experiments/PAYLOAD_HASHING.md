# Optional Parallel Payload Hashing

SHA-256 is a substantial part of the remaining background capture work. The
snapshot builder creates per-object digests, the writer independently validates
contents before registration, and the Store adapter checks source bytes before
transfer or validates received bytes. Each boundary remains mandatory.

`payload_hash_workers` is an optional capture setting in `[1, 8]`, defaulting to
1. The Store owns one lazy `PayloadHasher`; the active owner shares it across
snapshot construction, writer validation, batch writes, recovery and batch reads.
At least two objects and 1 MiB of aggregate payload are required to use threads.
Byte-balanced groups limit submissions to the configured worker count rather
than creating one future per object. No payload copy or cached digest is added.
Single-thread construction retains the original immediate descriptor path.

The caller keeps source views immutable throughout each synchronous hash call.
If submission or hashing fails, the method waits for every submitted task before
propagating the exception. This prevents Host slot reuse and receive-buffer
unregistration while another thread still reads them. Store close joins hashing
before closing the native client and releasing registered storage. Workers only
use SHA-256 on Host byte views; all SDK calls remain on the original owner thread.

The setting participates in capture configuration/startup agreement. Manual
connected-resource callers must match the Store and capture worker counts.
Inactive ranks allocate no hashing pool. CPU budgets should account for all
active TP/PP owners, and the default remains single-threaded.

## Checks

Focused tests cover exact digest order, sliced and empty byte views, bounded
submissions, lazy startup, shutdown, and both worker/submission failures with
another source reader held behind an event. They compare full serial/parallel
descriptors, validate TP owner partitions and reject changed or nonfinite
payloads. Batch Store read/write checks retain digest mismatch rejection and
registration cleanup. The native TCP suite adds a multi-megabyte sample with
four hash workers, then closes the original producer and recovers through a
fresh connection using Store payloads.

The complete capture unit directory passes **337 methods in 282.321s**, including
the seven focused hashing tests. The complete native Mooncake TCP suite passes
**10 methods in 243.400s**, including producer-close recovery with parallel
hashing. All 14 changed Python files pass formatting and introduce no Ruff
diagnostics relative to `1b12e036c` (12 pre-existing findings remain).

## CPU Pipeline Probe

The initial isolated thread-pool probe hashes 256 MiB in 128 KiB chunks in
185.162 ms with one worker and 61.235 ms with four (median of ten trials).
Its CPU time rises from 185.166 to 232.377 ms. The implementation groups work
into only four tasks; this probe motivated the implementation but does not
measure its complete producer behavior or serving latency.

The production builder/validator probe uses two layers, eight KV heads, 128
dimensions, 64-token chunks and a 128-token response. Every parallel descriptor
is compared with the serial reference, and complete content validation remains
enabled. Two warmups precede ten measured trials per operation; these are
synthetic Host tensors, without inference or Store transfer.

| Prompt | Payload | Stage | 1 worker, ms | 4 workers, ms | CPU ms, 1 / 4 |
| --- | --- | --- | ---: | ---: | ---: |
| 512 | 5,375,744 bytes / 48 objects | Construction | 4.891 | 2.221 | 4.892 / 5.093 |
| 512 | Same | Validation | 5.999 | 3.403 | 6.000 / 6.267 |
| 32,768 | 270,068,480 bytes / 2,064 objects | Construction | 227.016 | 87.815 | 226.933 / 229.477 |
| 32,768 | Same | Validation | 261.773 | 122.131 | 261.771 / 263.831 |

Values are medians. Long-context maximums include a 300.561 ms parallel
construction and 509.917 ms serial validation; the evidence retains those
outliers. The resident process has 192 CPUs in its affinity and no container CPU
limit. This is not a CPU entitlement assumption for production MaaS owners.

## Serving Comparison

Use the pinned environment and one frozen source for both settings:

```bash
python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  --source-revision FROZEN_SOURCE --output-dir NEW_OUTPUT_DIRECTORY \
  --num-prompts 64 --input-len 512 --output-len 128 --concurrency 4 \
  --capture-slots 4 --host-mib 256 --segment-mib 2048 \
  --ratios 1.0 --repeats 1 --payload-hash-workers 4
```

Repeat with one worker and separate output. The driver uses an off/on/off bracket
for each setting, verifies the live stage metrics and checks every measured
READY sample after producer exit. Report admission and READY counts together with
throughput: faster slot reuse may change how much capture work is admitted.

Four sequential Qwen3-0.6B brackets ran with worker counts **1, 4, 4, 1** on
the same frozen source. Each phase serves 64 requests with 512 input tokens,
128 output tokens, concurrency four and four capture slots. The selected layers
are 0/14/27; all 256 capture-on requests completed inference. Every admitted
sample published and passed complete Store readback after its producer exited.

| Run | Workers | READY / considered | Requests/s | Fraction of own off bracket | Captured samples/s | p99 TTFT / TPOT, ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1 | 64 / 64 | 5.3153 | 74.04% | 5.3153 | 710.074 / 3.748 |
| parallel | 4 | 62 / 64 | 5.4255 | 74.30% | 5.2560 | 700.844 / 3.409 |
| parallel-repeat | 4 | 61 / 64 | 5.4232 | 75.78% | 5.1690 | 678.226 / 3.456 |
| baseline-repeat | 1 | 62 / 64 | 5.2910 | 73.50% | 5.1257 | 694.068 / 3.642 |

The seven exclusions are `admission_backpressure` under the four-slot budget.
There are no stage errors, Catalog errors, quarantined slots, disabled producers
or failed admitted samples. Live metrics match status counters. Post-exit
readback validates **249 samples / 16,932 objects / 1,993,338,624 payload bytes**.
All four journal directories contain only their lock file, with no pending
publication after cleanup.

| Background stage | 1 worker, ms/sample | 4 workers, ms/sample |
| --- | ---: | ---: |
| Snapshot construction | 10.195 | 5.964 |
| Mandatory content validation | 13.811 | 9.359 |
| Store payload write, including source hashes | 12.763 | 8.341 |

These stage values divide total stage seconds by READY count across both runs
of each setting. They are elapsed stage time, not CPU time or wire-only timing.
The background improvement is repeatable in these four runs, but aggregate
captured throughput is **5.220 samples/s with one worker versus 5.212 with four**.
Small serving-rate differences, differing admission counts and only two runs
per setting do not establish an end-to-end throughput or tail-latency benefit.
Default workers remain one. This experiment does not contain 32K model serving;
the long-context numbers above come from the Host-only probe.

## Evidence

[payload-hashing.json](payload-hashing.json) binds nine successful terminal jobs,
the 5,626-file frozen source audit and a 109-artifact archive at
`/gpfs/user/fuxuanwei/mooncake-lab-archive/hash-scaling-20261003`. The archive keeps
job specifications/results/logs, both probes, the audit and submission scripts,
the changed Python source and all benchmark artifacts except service-owned
journal directories. Those directories were checked inside the container.
The resident has no live experiment processes and its idle load has resumed;
no additional GPU was allocated.

These experiments use a resident H100, native TCP Mooncake and a test Catalog.
They do not certify production retention, RDMA, multi-GPU CPU contention, MaaS
SLOs or trained draft-model quality.
