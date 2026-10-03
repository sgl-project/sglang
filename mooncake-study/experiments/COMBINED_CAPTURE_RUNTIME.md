# Combined TP/PP Capture Runtime

This matrix covers the intersection of tensor and pipeline parallelism:
`TP=2, PP=2`, four distinct CUDA devices and four capture owners. Separate TP2
and PP2 runs do not prove this combination. Tests use the real Qwen3-0.6B
model, Mooncake TCP transport/Store and the existing HTTP test Catalog.

## Prefill Graphs

The colocated and P/D suites each run ordinary AR and static target-KV DSpark
with Full, Breakable and torch.compile piecewise prefill backends. Pipeline
execution is synchronous. The piecewise fixture uses the repository's eager
compile debug mode; it does not certify arbitrary inductor compilation.
Full prefill graphs remain experimental under the existing server contract.

Each graph family is compared with real eager generation at the same topology.
Requests exercise chunked prefill, prefix reuse, token padding, repeated graph
buffers, different prompt/response lengths, batches and speculative acceptance
or rejection. Independent pool/logits observations compare all selected KV
shards, raw top-128 IDs/values and full-vocabulary log-sum-exp with Store
snapshots. Graph buffers are destroyed before the final Store reads.

The shared graph checks now require padded replay and repeated buffer use on
**every** rank. Graph buffer identity includes both TP and PP rank: a virtual
address alone does not identify a buffer across worker processes. The
colocated report records the exact ranks with observed buffer reuse.

P/D uses two independent serving groups on the same four GPUs. P executes
prefill graphs and transfers the first teacher row with KV; D publishes the
complete sample. The fixture also injects missing/stale teacher handoffs and
request cancellation. These failed requests must execute prefill on all four
P ranks and must never create a partial READY snapshot. P/D placement is a
correctness fixture, not cross-node or disaggregated throughput validation.

## Live Controls

The combined P/D control suite runs AR and static target-KV DSpark in eager
and decode graph modes. PP graph scheduling stays synchronous. It exercises:

- Pause/resume of each endpoint independently while inference continues.
- Partial prefill capture paused after every P rank has entered the request.
- Abort followed by resume before an old first-teacher handoff arrives.
- Active decode capture abort and unbound-ticket retirement.
- State and control-counter agreement on all four ranks of each endpoint.
- Exact manifest/tensor readback after both serving groups have exited.

The control API remains asynchronous with respect to background readiness and
in-flight publication. These tests do not establish a distributed drain barrier
or permit a late, fenced-out handoff to start a new capture.

## Reproduction

Use four H100s with `h100-runtime-lock.json` and an isolated checkout. The local
test model path can be selected through `TRAINING_CAPTURE_TEST_MODEL`. Keep
the runtime venv's `mooncake_master` on `PATH` and run the suites sequentially:

```bash
export TRAINING_CAPTURE_TEST_MODEL=/models/Qwen3-0.6B
export PYTHONPATH=python
export OMP_NUM_THREADS=1
python test/registered/storage/test_training_capture_tp_pp_prefill_graph.py -v -f
python test/registered/storage/test_training_capture_pd_tp_pp_prefill_graph.py -v -f
python test/registered/storage/test_training_capture_pd_tp_pp_control.py -v -f
```

The three files register with the existing `4-gpu-h100` CI runner. They require
four actual visible devices and skip on smaller allocations. No rank is
emulated and no production capability guard is disabled. Cold FlashInfer JIT
compilation can delay the first health request on a fresh temporary node; retain
the authoritative test outcome instead of inferring failure from a health poll.

The test Catalog does not establish production retention or consumer checkpoint
behavior. Synthetic draft weights establish serving/capture compatibility, not
trained draft quality. Asymmetric P/D prefill graphs, asynchronous PP,
cross-node distributed RDMA, additional models and production SLOs remain
separate requirements.

## Colocated Results

The frozen `sglang-prefill-tp-pp-v1` checkout passes all six colocated methods
in 635.511 seconds on four H100 80GB GPUs on node073. Including the shared eager
baseline, seven cells validate 70 post-exit snapshots, 2,408 tensor objects and
95,762,660 tensor bytes. They observe 264 prefill replay rank-frames and 204
speculative verify replay rank-frames. All four ranks exercise padding and
buffer reuse in each graph cell; rank-frames are observations, not request counts.

The initial cold health polls time out during live FlashInfer JIT compilation.
The same job proceeds after compilation and completes successfully; no test is
restarted or omitted. This elapsed time is a correctness-suite runtime, not a
serving performance measurement.

## P/D Graph Results

The same frozen checkout passes all six P/D methods in 990.107 seconds.
Including the eager baseline, seven cells validate another 70 post-exit
snapshots and observe 192 prefill replay rank-frames. Both serving groups use
TP2/PP2. The six graph cells cover all three backends with AR and static
target-KV DSpark, with eager generation parity and exact selected KV and
top-128 Store readback.

Each cell also rejects two captures with missing/stale handoffs and one
cancelled request. These deliberately failed captures do not publish; their
failure counters are expected. All four P ranks must exercise padding,
buffer reuse and the failed-request prefill paths.

## Control Results And Evidence

The frozen `sglang-prefill-tp-pp-v2` checkout passes all four control methods
in 610.333 seconds. AR and static target-KV DSpark each pass eager and decode
graph execution. Each cell validates eight post-exit snapshots, the five
intended failed request captures, and nine retired unbound tickets without
payload writes. All four ranks on each endpoint agree on control state and
counters, with no permanent disable or D-side quarantine/backpressure.

Across the three suites, **16 methods pass**, with **172 post-exit snapshots**
and **456 prefill replay rank-frames**. Synthetic draft-contract seed samples
are excluded from these totals. The [evidence index](combined-capture-runtime.json)
records commands, per-cell results, source identities and archived log hashes.
All tracked Python source and storage test files match the frozen snapshots:
3,355 files in v1 and 3,356 in v2. This change adds tests and stronger oracles;
the production implementation remains at base commit `d4940b115`.

Before releasing temporary Northjob `job-8e90d73789c0-20261003091401`, all jobs
had successful terminal results, the queue was empty, and the supervisor was
idle. No live serving/Mooncake process or GPU compute process remained.
The supervisor exited normally after SIGTERM; Northjob deletion succeeded and
both the Pod and job resources were confirmed absent. The original resident
H100 remains allocated with its idle workload running.
