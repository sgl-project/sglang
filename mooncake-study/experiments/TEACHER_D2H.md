# Bounded Teacher D2H Batching

Compact teacher IDs, raw top-128 logits and full-vocabulary LSE can now batch
their D2H transfers independently of selected KV. Configure:

```json
{
  "kv_d2h_batch_tokens": 16,
  "teacher_d2h_batch_tokens": 16,
  "max_device_bytes": 16777216
}
```

Both batch sizes default to one. The existing device budget covers aligned
KV plus teacher staging tensors across every local slot. It is checked before
Host registration; it does not bound model memory, transient raw logits or
CUDA allocator reservations. Teacher staging exists only on the aux owner,
including a PP stage with no selected KV layers. Server pool statistics and the
historical `kv_staging_*` Prometheus gauges report this combined allocation.

Short ranges are copied immediately into request-owned device storage. Full
batches and the sealed tail transfer into the existing registered Host tensors.
Large ranges bypass staging. P/D handoff rows already on CPU stay there and
advance the same row ledger. Source mutation after enqueue, producer stream
changes, full-arena reuse and sealing outside the forward stream are fenced.
Abort discards unpublished tails; uncertain completion quarantines both arenas.
Lookahead rows may be transferred, but only the accepted prefix enters the
manifest. The raw Store format, checksums and CE/TV128 alignment are unchanged.

## Reproduction

Use the pinned environment in `h100-runtime-lock.json`. All commands below run
from the implementation checkout with `PYTHONPATH=python`, a local Qwen3-0.6B
target and Mooncake SDK/master on PATH. The Catalog is an HTTP test double.

```bash
python -m unittest discover -s test/registered/unit/training_capture -p 'test_*.py'
python test/registered/storage/test_training_capture_runtime.py \
  --model-path /models/Qwen3-0.6B
TRAINING_CAPTURE_TEST_MODEL=/models/Qwen3-0.6B \
  python test/registered/storage/test_training_capture_pd.py

python mooncake-study/experiments/profile_training_capture.py \
  --model-path /models/Qwen3-0.6B --source-revision CHECKOUT_REVISION \
  --output-dir /experiments/teacher-profile \
  --kv-d2h-batch-tokens 16 --teacher-d2h-batch-tokens 16 --device-mib 16

python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /models/Qwen3-0.6B --source-revision CHECKOUT_REVISION \
  --output-dir /experiments/teacher-benchmark \
  --kv-d2h-batch-tokens 16 --teacher-d2h-batch-tokens 16 --device-mib 16 \
  --ratios 0.1 --repeats 1
```

For direct-teacher comparison use `--teacher-d2h-batch-tokens 1` in a fresh
output directory. The retained baseline was frozen before this change and
uses the old driver without that flag; KV batching remains 16 on both sides.
The benchmark brackets its capture phase with capture-off servers and validates
measured Store snapshots after the producer exits. It uses the existing
streaming client without tensor observer hooks. Profiler runs are separate:
their annotated CPU scopes cover append and final tail flush without overlapping
scope ranges, and trace analysis attributes DMA by CUDA correlation IDs.

## Profiler Results

Both runs use 16-token KV staging, 40 measured requests per workload, normal
overlap and decode graphs. All measured requests reach READY. Teacher staging
adds 263,168 bytes across sixteen slots, bringing total staging to 3,408,896
bytes within the 16MiB budget. The table includes full and final partial batches.

| Workload | Teacher D2H Calls, Before / After | D2H Work, Before / After | Added D2D Calls / Work | Teacher D2H Bytes, Both |
| --- | ---: | ---: | ---: | ---: |
| 128-to-1 | 135 / 120 | 0.320 / 0.332ms | 135 / 0.136ms | 46,260 |
| 1-to-32 | 3,960 / 360 | 9.863 / 0.980ms | 3,960 / 3.956ms | 1,356,960 |

For the decode workload, teacher D2H calls fall 90.9%; D2H plus added D2D work
is 4.936ms instead of 9.863ms. Total capture D2H calls fall from 6,000 to 2,400,
with identical 17,587,680 transferred bytes. KV gather and teacher extraction
kernel counts are unchanged. Teacher append/tail CPU scope time increases from
263.658ms to 276.240ms; these annotated times are not client latency.

The one-output-token workload adds device copying and almost doubles the
annotated teacher CPU scope (11.192ms to 21.690ms). Its lookahead rows explain
why D2H calls still decrease slightly. Batching does not reduce the captured
values or change the accepted publication prefix. The trace analyzer now
records D2D separately; missing byte fields remain unknown rather than zero.

## Serving Results

The direct-teacher and batched-teacher runs each complete an off/on/off round of
2,048 requests per phase: 12,288 timed requests and 393,216 output tokens in
total. Both use 128 input / 32 output tokens, concurrency eight and 10% selection.
Each capture phase admits, publishes and validates 190 measured snapshots after
stopping its producer. There is no capture backpressure, quarantined slot,
Catalog error or cached prompt token. Each phase stores exactly 373,555,200 KV
bytes and 6,700,160 auxiliary bytes.

| Teacher Policy | Off Bracket Mean, Tokens/s | Capture Tokens/s | Throughput Loss | TPOT p95 Increase |
| --- | ---: | ---: | ---: | ---: |
| Direct | 1,310.43 | 1,118.35 | 14.7% | 13.1% |
| 16-row staging | 1,333.22 | 1,115.33 | 16.3% | 15.3% |

There is no demonstrated end-to-end speedup: capture throughput changes by
-0.27%, and its relative latency/throughput cost is worse against its own
bracket. One round per policy cannot establish statistical significance, and
the baseline TTFT p99 drifts within each bracket. Staging remains disabled by
default. The measured reduction in device transfer work does not resolve the
producer's Python/launch overhead or satisfy P10's service SLO.

## Scope

This is an opt-in transfer policy, not an assertion that fewer DMA calls always
improve serving latency. Device copies and Python bookkeeping add work; a
deployment must measure its workload. Synthetic local TCP measurements do not
certify production SLOs, distributed teacher-batching performance, cross-node
RDMA, Catalog retention or trained draft quality. The full design goal remains
active.

## Validation

All ten worker jobs completed successfully on the resident H100:

| Check | Tests | Seconds |
| --- | ---: | ---: |
| Complete capture unit discovery | 192 | 180.733 |
| Final CUDA ownership/fence file | 7 | 0.421 |
| Final teacher CPU file | 6 | 0.008 |
| Final trace attribution file | 4 | 0.011 |
| Complete single-GPU runtime | 2 | 641.671 |
| Real P/D eager and graph/overlap | 2 | 134.279 |

The focused files overlap the discovery suite; do not sum these as distinct
test cases. Discovery preceded the extra teacher tail-fence and D2D attribution
tests and a dict-literal lint correction; each final affected file was then
verified. Production sources are the same throughout these runs. The four other
jobs are the direct/staged profiler and serving benchmarks.

Runtime tests cover exact online KV/teacher preservation, accepted speculative
paths, arbitrary source-slot reuse, cached/chunked prefixes, ragged verification,
lookahead trim, abort, retract/rebuild, writer pressure and latency feedback.
The real P/D tests verify ten complete snapshots and exclude four failed
handoffs plus two aborted captures. New unit tests cover combined budgets before
registration, aux-only ownership, CPU handoff followed by GPU rows, cross-stream
reuse, pending tails across slot reuse and completion-failure quarantine.

No extra GPU allocation was used. All experiment processes exited and the
resident worker resumed its 60% idle load. The original worktree's staged index
is unchanged. Source, trace, log and report hashes, commands and per-job results
are retained in [teacher-d2h.json](teacher-d2h.json).
