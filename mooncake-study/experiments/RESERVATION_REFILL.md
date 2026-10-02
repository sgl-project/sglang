# Single-Rank Reservation Refill

The single-rank producer previously waited 100 ms after every Catalog admission,
including successful reservations. This limited replenishment to at most ten
slots per second, before Catalog latency. Selected requests arriving without a
reservation skipped capture even when the writer had already freed Host slots.

The coordinator now fills free capacity without delaying successful admissions.
Each reservation still takes one background Catalog call. The loop checks
renewals, expiry, disable state and adaptive cooldown between these calls.
Writer retirement signals the loop after pool release; pending-publication
recovery and shutdown also signal it. Clearing the event before inspecting
capacity preserves releases between a failed acquire and the wait.

The 100 ms idle maintenance poll remains. A separate monotonic retry deadline
prevents retirement signals from accelerating failed Catalog admissions. Pool
size, registered-buffer ownership, quarantine, writer serialization and serving
admission behavior remain bounded by their existing contracts. Distributed
cohort reservations use a separate collective loop and are outside this change.

## Reproduction

Use the dependency overlay pinned in [h100-runtime-lock.json](h100-runtime-lock.json).
Both versions run on the same resident H100 with its idle load suspended during
the experiment. Each source tree is frozen before submission:

- Before: `113f0e3fd1ee6d135553b950e9d52f6fbf7fe92b`, retained at
  `/gpfs/users/fuxuanwei-1/dspark-maas-lab/sglang-refill-before`.
- After: the same source plus the coordinator refill change and focused tests,
  retained at `/gpfs/users/fuxuanwei-1/dspark-maas-lab/sglang-refill-v1`.

From each frozen tree, use its own `python` directory on `PYTHONPATH` and run:

```bash
python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  --source-revision SOURCE_REVISION \
  --output-dir NEW_OUTPUT_DIRECTORY \
  --num-prompts 512 --input-len 16 --output-len 2 \
  --concurrency 4 --capture-slots 4 --ratios 1.0 --repeats 2
```

The unchanged benchmark driver uses Qwen3-0.6B BF16, Triton attention, overlap,
Full decode graphs and eager prefill. Each repetition brackets capture-on with
two capture-off server runs. Every phase has its own real TCP Mooncake Store
and HTTP test Catalog. Warmup is excluded, the prefix cache is flushed, and
the native streaming client generates 512 seeded random-token requests.

Each capture request stores selected layers 0/14/27, two raw top-128 teacher
rows with vocabulary IDs and full-vocabulary LSE, token IDs, masks, positions
and terminal KV validity. After producer exit, an independent Store client
reads every published manifest and object, checks SHA256 and validates tensor
content/coverage. Exact online KV/logit parity is tested separately by the
runtime regression; this benchmark has no tensor-observer hooks.

For CPU and actual inference regressions:

```bash
CUDA_VISIBLE_DEVICES=999 python -m unittest discover \
  -s test/registered/unit/training_capture -p 'test_*coordinator.py' -f
CUDA_VISIBLE_DEVICES=999 python \
  test/registered/unit/training_capture/test_reservation_refill.py -f
CUDA_VISIBLE_DEVICES=999 python \
  test/registered/unit/training_capture/test_pd_capture.py -f
python test/registered/storage/test_training_capture_runtime.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B -f
TRAINING_CAPTURE_TEST_MODEL=/gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  python test/registered/storage/test_training_capture_pd.py -f
```

The nine focused CPU tests cover bounded initial fill, writer-driven refill,
release between failed acquire and wait, heartbeat fairness, error retry fencing
despite wakeups, cooldown, disable during an admission, quarantine and shutdown.
They hold the maintenance timeout so refill must progress through the event;
they do not assert machine-dependent sub-100-ms timing.

## Results

Both versions completed two off/on/off repetitions on 2026-10-02. Each row
below describes one capture-on phase; its serving baseline is the mean of the
adjacent capture-off phases.

| Source | Round | READY / 512 | Capture rate | READY/s | Requests/s | Throughput vs off | p99 TTFT ms | p99 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Before | 1 | 60 | 11.72% | 10.30 | 87.86 | 92.77% | 57.49 | 44.87 |
| Before | 2 | 60 | 11.72% | 10.46 | 89.29 | 91.59% | 50.46 | 44.34 |
| After | 1 | 292 | 57.03% | 39.29 | 68.90 | 76.58% | 62.52 | 60.07 |
| After | 2 | 312 | 60.94% | 40.10 | 65.81 | 68.37% | 85.14 | 61.03 |

READY/s divides final measured-request READY count by client measurement
duration, excluding the subsequent writer drain. Startup spares can make the
old version slightly exceed ten samples per second during this short window.

The new version produced 604 complete samples, versus 120 before; all 724
snapshots passed post-exit Store readback. Every admitted measured request
published, with no Catalog errors, quarantine or permanent disable reason.
Backpressure skips fell from 452/452 to 220/200. Both versions allocated exactly
5,155,584 registered Host bytes and no device staging bytes, with four slots.
The old version had one/two free slots at the final drained observations; the
new version had replenished all four slots at both observations.

This raises measured sample throughput by about 3.8 times, while increasing the
work performed per incoming request. It does not improve serving throughput:
the new capture-on runs retain only 68-77% of their adjacent off baselines.
The short two-token responses make per-request capture costs prominent. There
are only two repetitions, and the off baselines drift, so these numbers do not
establish a production latency bound or identify the next bottleneck. The
existing ratio/adaptive controls remain necessary for service budgets.

Raw reports, phase measurements and logs are retained under
`/gpfs/users/fuxuanwei-1/dspark-maas-lab/experiments/reservation-refill-before`
and `reservation-refill-after`. The [evidence index](reservation-refill.json)
binds reports, worker results and sources by hash.

## Regression Results

| Suite | Tests | Seconds |
| --- | --- | --- |
| Direct/staged/cohort coordinator | 49 | 37.168 |
| Reservation refill | 9 | 4.796 |
| P/D collector unit tests | 12 | 9.004 |
| P/D eager and graph/overlap inference | 2 | 176.081 |
| AR, DSpark, pressure, admission and ragged inference | 2 | 707.379 |

All 74 tests pass. The runtime suites include exact online selected-KV/teacher
comparison, prefix reuse, token/mask alignment, cancellation, real KV-capacity
retraction, adaptive and latency cooldown/recovery, and producer-exit Store
reads. The P/D suite validates ten complete snapshots and excludes missing/stale
handoffs. The four ragged cases cover cap-accept and compact execution with and
without graph/overlap.

The first runtime invocation passed `TRAINING_CAPTURE_TEST_MODEL`, which that
older test does not consume. Its setup attempted a Hub download and failed with
`Network is unreachable`, before running any tests. The corrected command above
uses `--model-path`; it passed unchanged code. Both attempt logs/results are
retained. Eight submitted jobs are terminal: seven successful and this one
failed setup. The resident H100 resumed its idle workload with an empty queue.

Both touched Python files pass Black and Ruff I/F; the new test passes full
Ruff. The coordinator retains nine pre-existing BLE001 diagnostics and adds no
new diagnostics. The original outer checkout's staged index remains unchanged.

## Scope

This is a short-request saturation diagnostic, not production traffic or an SLO
acceptance test. Increasing successful collection adds capture and writer work;
compare both sample yield and serving cost. The fixed ratio is one and adaptive
admission is disabled in this experiment. Full production Catalog retention,
distributed refill throughput, RDMA saturation, longer workloads and training
quality require their own evidence.
