# Batched Snapshot Store Writes

`SnapshotWriter.write()` and `write_partition()` now submit their tensor payloads
through `MooncakeSnapshotStore.put_registered_batch()`. On the pinned SDK, a
fresh snapshot uses one native existence query and one native registered-buffer
write for its payloads. The prior implementation made these calls per object.
For the 16-input/two-output-token workload below, this changes 14 existence and
14 write calls to one of each. The manifest remains a separate final write.

## Ownership and Failure

Before calling the SDK, the adapter validates every source registration and
digest and rejects duplicate keys. An existing key is read back and checked
for exact byte count and digest; its existence alone cannot acknowledge a retry.
Only missing keys are passed to `batch_put_from`, with the same mandatory
hard-pin and replica configuration used by scalar writes.

Each returned status must be an integer, and zero is the only success value.
Known nonzero entries quarantine their enclosing registered allocation, including
other views sharing that allocation. Missing/malformed statuses or an exception
leave the whole submitted batch uncertain, so all submitted registrations are
retained. No uncertain write is retried on a reusable Host buffer. Sources whose
remote contents were already verified are not part of that submitted batch.

The writer reports no WRITTEN receipt and publishes no manifest after any failed
payload batch. Fenced Catalog cleanup remains responsible for partial orphan
objects. Manifest journaling, seal/publication retries and producer-exit retention
are unchanged. The adapter uses the scalar path only when either optional native
batch method is absent; an actual batch transfer error never triggers fallback.

## Reproduction

Use [h100-runtime-lock.json](h100-runtime-lock.json), the resident H100 and a
local Qwen3-0.6B model. The idle workload is suspended during each job. Source
directories are frozen before running:

- Baseline `5e13be5919423b169b4057454b148a4753132e0b`:
  `/gpfs/users/fuxuanwei-1/dspark-maas-lab/sglang-batch-put-before`.
- Implementation and initial tests:
  `/gpfs/users/fuxuanwei-1/dspark-maas-lab/sglang-batch-put-v1`.
- Final tests, differing only in four test context-manager formatting changes:
  `/gpfs/users/fuxuanwei-1/dspark-maas-lab/sglang-batch-put-v2`.

Set `PYTHONPATH` to the frozen source's `python` directory. From each benchmark
tree, use a new output directory:

```bash
python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  --source-revision SOURCE_REVISION --output-dir NEW_OUTPUT_DIRECTORY \
  --num-prompts 512 --input-len 16 --output-len 2 \
  --concurrency 4 --capture-slots 4 --ratios 1.0 --repeats 2
```

The unchanged driver brackets each capture-on phase with capture-off phases,
excludes warmup, flushes the prefix cache and uses seeded random-token streaming
requests. It runs actual Qwen3 BF16 with Triton, overlap and Full decode graphs.
Each phase owns a real TCP Mooncake master/data node and HTTP test Catalog.
Every measured publication is read and validated after producer exit. Exact
online KV/teacher parity is exercised by the separate inference regression,
without adding observer overhead to the benchmark.

Run the focused and existing regressions from the final test tree:

```bash
CUDA_VISIBLE_DEVICES=999 python test/registered/unit/training_capture/test_buffers.py -f
CUDA_VISIBLE_DEVICES=999 python -m unittest discover \
  -s test/registered/unit/training_capture -p 'test_*writer.py' -f
CUDA_VISIBLE_DEVICES=999 python -m unittest discover \
  -s test/registered/unit/training_capture -p 'test_*coordinator.py' -f
python test/registered/storage/test_training_snapshot_mooncake.py -f
python test/registered/storage/test_training_capture_runtime.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  TestTrainingCaptureRuntime.test_chunk_prefix_single_token_and_raw_teacher_reference -f
TRAINING_CAPTURE_TEST_MODEL=/gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  python test/registered/storage/test_training_capture_pd.py -f
```

The Store suite includes CUDA-backed cohort collectors and must keep the GPU
visible. Its native SDK spy verifies the batch calls, hard-pin configuration,
mixed existing/missing keys and an identical retry without a second write.
Other cases cover independent owners and post-exit readers. Unit faults cover
partial success, ambiguous completion, arena aliases, corrupt retries and SDKs
without batching. The inference suite covers AR, overlap, target-KV DSpark,
real retraction and adaptive/latency admission; P/D runs eager and graph/overlap.

## Regression Results

All 99 distinct test methods below pass on the final production implementation.
The buffer and writer suites were rerun after test-only formatting; their first
41 passing executions are retained but excluded from this count.

| Suite | Methods | Seconds | Final job |
| --- | --- | --- | --- |
| Buffer ownership and Store contract | 18 | 0.013 | `01790941547249557533-3c92f5afa110` |
| Snapshot/cohort writers | 23 | 5.653 | `01790941547561959205-50a243e0dd72` |
| Direct/staged/cohort coordinators | 49 | 36.146 | `01790941185326471663-c1ca22bbfc91` |
| Actual Store and multiprocess publication | 6 | 173.405 | `01790941547901828816-46da8e273aeb` |
| Actual inference, retraction and admission | 1 | 505.639 | `01790941366398817775-3bcf7595e77b` |
| P/D eager and graph/overlap | 2 | 176.418 | `01790941366715507668-c38a4ab88acc` |

The first Store invocation, `01790941185691925587-97fbcc898e41`, incorrectly
set `CUDA_VISIBLE_DEVICES=999`. It passed the first method, then failed when a
cohort collector called `torch.cuda.set_device(0)`. The GPU-visible rerun passes
all six methods; this failed environment attempt is retained in the evidence.
All 11 submitted jobs are terminal: ten succeeded and one failed as described.
No additional GPU was allocated; the resident H100 resumed its idle workload
with an empty queue.

All six edited Python files pass Black and Ruff I/F. Full Ruff introduces no
diagnostic beyond the three pre-existing warnings in two touched test files.
The [evidence index](batch-store-writes.json) binds source hashes, runtime,
commands, terminal results, logs and raw benchmark reports. Its 12 executable
source hashes match the final frozen test tree; documentation was added after
the freeze. The original checkout's staged index remains unchanged.

## Serving Results

Both fresh benchmarks completed on 2026-10-02. Each row is a capture-on phase;
the off comparison uses the mean of that phase's two adjacent capture-off runs.

| Source | Round | READY / 512 | READY/s | Requests/s | Throughput vs off | p99 TTFT ms | p99 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Scalar | 1 | 301 | 40.70 | 69.22 | 74.87% | 59.81 | 57.98 |
| Scalar | 2 | 308 | 41.37 | 68.77 | 72.76% | 59.14 | 60.00 |
| Batch | 1 | 303 | 40.59 | 68.59 | 73.26% | 65.80 | 61.91 |
| Batch | 2 | 298 | 40.42 | 69.44 | 69.88% | 60.64 | 59.74 |

All 1,210 measured snapshots passed post-exit Store readback, including all 601
from the new batch path. Every admitted measured request reached READY, with no
Catalog errors, permanent disable or quarantine. Both versions use four Host
slots, allocate 5,155,584 registered Host bytes and use no device staging.

Across both capture-on phases, the scalar version produces 41.03 samples/s and
68.99 requests/s; batch produces 40.51 samples/s and 69.01 requests/s. These
results do not demonstrate an end-to-end speedup. Capture yield remains bounded
below the configured sampling ratio of one. The off baselines drift, and two
short repetitions do not establish an SLO or explain the remaining bottleneck.
The verified implementation benefit is fewer SDK calls while preserving the
publication and ownership contract. Further performance work must measure the
remaining writer, validation and serving costs rather than infer improvement
from call count.

Raw phase reports and logs are retained under
`/gpfs/users/fuxuanwei-1/dspark-maas-lab/experiments/batch-put-before` and
`batch-put-after`.

## Limits

This is a same-node TCP measurement with a test Catalog and two short benchmark
repetitions. Reduced SDK call count is not proof of a serving speedup. Capture
yield, service throughput and latency must be compared together. RDMA batching,
production retention, representative traffic SLOs and trained-draft quality need
separate validation.
