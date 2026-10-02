# Publication-Boundary Snapshot Validation

The live single-rank producer prepares descriptors and then validates tensor
contents once, in `SnapshotWriter.write()`, before any Catalog object registration
or Store write. Previously, context construction and the writer each performed
the same full validation. Descriptor preparation still checks structure and
coverage and hashes source bytes; Store adapter checks remain in place.

## Interface and Lifecycle

| API | Result and required caller action |
| --- | --- |
| `prepare_snapshot(metadata, buffers, ...)` | Describes completed Host views; caller must validate contents before consuming or publishing |
| `build_snapshot(metadata, buffers, ...)` | Prepares and fully validates the snapshot |
| `RequestCaptureContext.prepare_snapshot(...)` | Checks sealed state, waits for copies, prepares views and rechecks sealed state |
| `RequestCaptureContext.snapshot(...)` | Calls preparation, validates contents and rechecks sealed state |
| `SnapshotWriter.write(..., check_current=...)` | Always validates contents, then invokes the optional state guard before registering objects |

The coordinator supplies a guard which checks invalidation, sealed state, local
lease deadline and maximum capture age under its state lock. Cancellation or
expiry during full validation therefore prevents registration. The callback
cannot skip or substitute for validation, does not hold the lock during tensor
scans or network calls, and does not replace authoritative Catalog fences. A
later cancellation still uses existing fencing and cleanup rules. A prepared
descriptor is not a certificate that its bytes are valid training data.

Journal recovery continues to verify durable manifests and Store payloads
against Catalog state. It does not depend on a live request callback. Distributed
owner preparation and cohort publication are unchanged. Buffer ownership,
uncertain-transfer quarantine, hard pinning, exact tensor values, manifests and
retention behavior retain their existing contracts.

`snapshot_built` now explicitly means descriptor/manifest preparation completed,
not that content validation passed. The `snapshot_build` timing includes
preparation and hashing; `validation` includes the mandatory full scan and the
optional state guard. These host wall times include scheduling/GIL waits. They
are not isolated CPU/GPU execution times or wire latency. Queue wait can overlap
other work and is excluded from covered execution totals below.

## Reproduction

Baseline is `4d412e4154a058a35813dd1dc68ca49048c4c6cc`, including native batch
payload writes and writer stage timings. Both sources use identical benchmark
driver bytes and the [pinned H100 runtime](h100-runtime-lock.json):

- `/gpfs/users/fuxuanwei-1/dspark-maas-lab/sglang-validation-before`
- `/gpfs/users/fuxuanwei-1/dspark-maas-lab/sglang-validation-v1`

Set `PYTHONPATH` to the respective source's `python` directory. Use the lab
capture venv and `OMP_NUM_THREADS=1`; no frozen source is overwritten during
experiments. The resident worker pauses its idle workload and serializes jobs.

```bash
CUDA_VISIBLE_DEVICES=999 python -m unittest discover \
  -s test/registered/unit/training_capture -p 'test_*context.py' -f
CUDA_VISIBLE_DEVICES=999 python test/registered/unit/training_capture/test_protocol.py -f
CUDA_VISIBLE_DEVICES=999 python -m unittest discover \
  -s test/registered/unit/training_capture -p 'test_*coordinator.py' -f
CUDA_VISIBLE_DEVICES=999 python -m unittest discover \
  -s test/registered/unit/training_capture -p 'test_*writer.py' -f
python test/registered/storage/test_training_snapshot_mooncake.py -f
python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  --source-revision SOURCE_REVISION --output-dir NEW_OUTPUT_DIRECTORY \
  --num-prompts 64 --input-len 512 --output-len 128 \
  --concurrency 4 --capture-slots 4 --ratios 1.0 --repeats 1
python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  --source-revision SOURCE_REVISION --output-dir NEW_OUTPUT_DIRECTORY \
  --num-prompts 512 --input-len 16 --output-len 2 \
  --concurrency 4 --capture-slots 4 --ratios 1.0 --repeats 2
python test/registered/storage/test_training_capture_runtime.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  TestTrainingCaptureRuntime.test_chunk_prefix_single_token_and_raw_teacher_reference -f
TRAINING_CAPTURE_TEST_MODEL=/gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  python test/registered/storage/test_training_capture_pd.py -f
```

Each benchmark round uses off/on/off phases with warmup excluded. Serving timing
excludes launch, Store readback and final writer drain. All measured READY samples
are read and validated after producer exit. The actual multiprocess `/metrics`
scrape must agree with status for all four fields of all 13 stages; the scrape
runs after client timing. Fixed random token workloads, a local TCP Store and
an HTTP Catalog test double do not establish production serving SLOs.

## Correctness Coverage

The direct and staged coordinator tests prove a successful publication invokes
the full writer validator exactly once. Eight corrupt prepared payload cases
(mask, token IDs, positions, top-k IDs, top-k order, LSE mass, KV validity and
non-finite KV) all fail before registration, Store calls, journal creation or
publication. Source digests are constructed from those corrupt bytes, so the
tests exercise semantic checks independently of checksum rejection.

Four blocked-validation cases invalidate the context, invalidate the record,
expire the local lease or exceed capture age. Each prevents registration and
returns the slot without quarantine. Context tests cover cancellation during
copy waits for checked and prepared APIs, and during checked content validation.
Existing writer/recovery, cohort, real Store and inference tests exercise the
unchanged publication, ownership and data contracts.

## Longer-Sequence Results

Each source completed one off/on/off round with 64 requests, 512 input/128 output
tokens, concurrency four and four Host slots. The four changed production
modules are exactly `context`, `coordinator`, `snapshot` and `snapshot_writer`.
Both allocate 38,329,344 registered Host bytes with no device staging.

| Source | READY | READY/s | Requests/s | Throughput vs off | p99 TTFT ms | p99 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 53 | 3.360 | 4.057 | 56.42% | 1012.57 | 11.38 |
| Single validation | 62 | 4.048 | 4.179 | 57.96% | 973.58 | 11.19 |

All admitted requests publish and all 115 snapshots pass post-exit readback,
covering 920,618,240 payload bytes and 7,820 tensor objects. Both actual metrics
scrapes pass. Every normal stage runs once per publication, `catalog_written`
runs twice, and `recovery_read` has zero calls. No measured stage errors,
Catalog failures or quarantine occur.

| Covered host work | Baseline ms/sample | Single validation ms/sample |
| --- | --- | --- |
| Snapshot preparation plus content validation | 66.232 | 37.161 |
| Catalog calls | 13.938 | 12.213 |
| Journal operations | 6.013 | 4.343 |
| Store adapters | 12.516 | 13.046 |
| Remaining copy wait | 0.255 | 0.200 |

Preparation alone falls from 37.351 to 10.683 ms/sample; mandatory writer
validation remains, at 28.881 versus 26.478 ms/sample. Combined preparation and
validation is 43.89% lower. Covered execution falls from 98.953 to 66.963
ms/sample. Queue wait, reported separately, is 88.789 versus 71.501 ms/sample.

Observed serving throughput is only 3.01% higher, while sample throughput is
20.50% higher. Admission is 53/64 versus 62/64 with the same bounded pool. The
capture-on throughput still remains below 60% of bracketing capture-off, so this
does not pass a serving overhead gate. One round per source cannot establish
stable latency tails or separate all run-order/system effects.

## Short-Request Results

Each source completed two off/on/off rounds with 512 requests, 16 input/two
output tokens per capture-on phase, concurrency four and four Host slots. Both
allocate 5,155,584 registered Host bytes with no device staging.

| Source | Round | READY | READY/s | Requests/s | Throughput vs off | p99 TTFT ms | p99 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1 | 282 | 38.68 | 70.23 | 71.85% | 61.97 | 58.91 |
| Baseline | 2 | 316 | 42.01 | 68.07 | 68.62% | 60.95 | 56.98 |
| Single validation | 1 | 319 | 43.99 | 70.60 | 71.80% | 59.52 | 60.17 |
| Single validation | 2 | 345 | 46.10 | 68.41 | 70.21% | 65.16 | 61.35 |

All 598 baseline and 664 optimized snapshots pass post-exit readback. Each
admitted request publishes, with no measured stage error, Catalog failure or
quarantine. All four actual metrics scrapes and stage-count checks pass.

Weighted preparation plus validation falls from 6.713 to 4.411 ms/sample, a
34.29% reduction. Covered execution falls from 18.695 to 15.260 ms/sample.
Across the two rounds, sample throughput rises from 40.37 to 45.06 samples/s
(11.62%), while serving throughput is nearly unchanged at 69.13 versus 69.49
requests/s (0.52%). The bounded producer admits 598/1024 versus 664/1024
requests. More accepted captures also mean more total collection work, so this
comparison does not isolate constant-capture-rate serving overhead.

The earlier timing experiment's 118-127 ms short-request p99 TTFT is not
reproduced here, including in the unchanged timed baseline. This does not prove
that removing duplicate validation fixes that tail or establish the cause of
the earlier regression. More representative workloads, balanced repetitions,
production Catalog/RDMA measurements and explicit serving SLOs remain required.

## Regression Results and Evidence

All 103 test methods pass on the final frozen implementation:

| Suite | Methods | Seconds | Job |
| --- | --- | --- | --- |
| Checked/prepared/partitioned contexts | 12 | 0.055 | `01790947518644702369-c9895161ec7b` |
| Tensor protocol | 4 | 0.060 | `01790947518955897329-e0f2b395f0e4` |
| Direct/staged/cohort coordinators | 55 | 40.724 | `01790947519283970028-3d2d90297672` |
| Snapshot/cohort writers | 23 | 5.650 | `01790947519611830629-e36edce22f24` |
| Actual Store/multiprocess publication | 6 | 172.422 | `01790947734135863072-e0f08d919349` |
| Actual inference and lifecycle | 1 | 505.648 | `01790947735815151737-e4502c1aac30` |
| Actual P/D handoff and failures | 2 | 175.265 | `01790947736146183575-8365af36f441` |

The inference method includes AR eager/graph/overlap, target-KV DSpark, actual
retraction, adaptive admission, latency recovery and post-exit reads. P/D covers
colocated single-GPU eager and graph/overlap execution, online source comparison,
handoff faults and cancellation. These functional checks do not measure a
serving SLO or validate new distributed/RDMA topologies.

All 1,377 measured benchmark snapshots pass readback, covering 1,202,675,240
payload bytes and 25,488 tensor objects. The [evidence index](publication-validation.json)
binds all 11 successful terminal jobs, logs/results, four reports, six actual
metrics scrapes, 43 executable/test source files and the pinned runtime.
Six edited Python files pass Black and Ruff I/F; full Ruff retains only the
coordinator's nine pre-existing BLE001 findings. The outer checkout's staged
index is unchanged. No additional GPU was allocated, and the resident H100 has
resumed its idle workload with an empty queue.
