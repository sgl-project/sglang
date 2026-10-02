# Colocated DSpark Pipeline Serving

`event_loop_pp_dspark` connects the stage coordinator to real request scheduling,
target-KV draft initialization, CUDA graph verification and distributed Mooncake
training capture. The public path supports colocated static target-KV drafts at
PP>1, DP1/CP1, without overlap scheduling or asynchronous host-tier cache. PP
speculation with hidden-input drafts remains rejected. The P/D variant has
separate readiness and first-teacher coordination; see [its runbook](PIPELINE_PD.md).

## Reproduce

The runtime test needs two CUDA GPUs, a local Qwen3-0.6B checkpoint, the matching
Mooncake SDK/master binary and the SpecForge reference exporter on PYTHONPATH.
It starts an actual Mooncake TCP Store and the repository's Catalog test double.
It captures a seed from an AR service, exports a synthetic KV-input draft, and
starts a real two-stage speculative service. Each test owns and stops its model
servers. It does not require a preexisting MaaS endpoint.

```bash
PYTHONPATH=python:/path/to/specforge-reference \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_dspark_pp.py -v -f
```

Run the shared graph and PP1/PP2 regressions as complete files:

```bash
PYTHONPATH=python python test/registered/unit/spec/test_dspark_pp_scheduler.py -v -f
PYTHONPATH=python python test/registered/unit/managers/test_scheduler_timeouts.py -v -f
PYTHONPATH=python python test/registered/unit/model_executor/test_cuda_graph_buffer_registry.py -v -f
PYTHONPATH=python python test/registered/unit/spec/test_draft_per_runner_config.py -v -f
PYTHONPATH=python python test/registered/unit/spec/test_dspark_pp_coordinator.py -v -f
PYTHONPATH=python python test/registered/unit/training_capture/test_config.py -v -f
PYTHONPATH=python python test/registered/unit/server_args/test_server_args.py -v -f
```

With the same runtime environment and local model/exporter configuration, run
`test_training_capture_pd_dspark.py`, `test_training_capture_pd_dspark_hidden.py`
and `test_training_capture_pd_pp.py` under `test/registered/storage`, each with
`-v -f`. The PP2 AR P/D file requires two GPUs; the two speculative P/D files use
the resident single H100 sequentially.

## Scheduling And Graphs

Each turn forwards received requests to the next stage before invoking request
handlers. Handlers and projection can enter world collectives, so delaying the
relay until afterward would leave peers waiting for requests. The loop retains
ordinary batch planning and result processing, and enters the coordinator inside
the existing scheduler field-isolation scope. Idle turns also agree across ranks.

Abort and idle inspection use the PP batch arrays' slot zero, which mirrors the
current running/last batch. Queue and running timeouts broadcast world rank zero's
expired request IDs. Existing grammar readiness propagation and PP minimum KV
capacity synchronization are reused. The stage coordinator validates batch and
proposal agreement, passes target activations, broadcasts final acceptance, then
commits through each stage's physical slots.

Real graph execution exposed three shared-buffer requirements:

1. Allocate hidden/residual PP buffers by token capacity, including the full
   speculative verify width per request, in both dummy and regular graph buffers.
2. Slice activation output by the actual token count, not the request count.
3. When verify metadata was prepared before predecessor activations arrived,
   copy those activations into the captured backing storage before replay.

The coordinator rejects activation frames with the wrong token count before
transport. Request signatures normalize production `array.array` input IDs to
the canonical sequence representation before hashing.

## Evidence Scope

Each eager/graph test validates seven ordinary request captures: greedy output
against an independently launched AR service, forced acceptance with a prefix
hit, a two-request mixed-acceptance batch, grammar, stochastic sampling with
penalties, and explicit stop-token handling. Selected layers 0, 14 and 27 span
both pipeline stages. Independent online observers supply physical target KV
and full raw vocabulary scores for snapshot comparison.

Cancellation must finish a failed distributed Catalog cohort without publishing
a sample. Four disjoint 208-token requests then exceed the 512-token KV pool,
causing automatic retraction without the debug retract flag. Every request must
still generate its complete 192-token output. Per-stage retraction metrics must
match reported request retractions; retired requests must rebuild their draft
context on every stage. Only complete surviving captures publish. A fresh request
then demonstrates recovered admission/capture capacity.

Snapshots are checked for selected KV, exact token IDs, response/loss mask,
teacher position alignment, raw top-128 logits and matching vocabulary IDs,
and logsumexp. They are also read after producer exit while the Store stays alive.
The distributed writer's local `stored` counter and Catalog publications verify
completion; the PP1 runtime's `ready` counter does not track cohort publication.

This proves Qwen3-0.6B TP1/PP2 mechanics with a synthetic draft, two H100s, NCCL
between model stages, Mooncake TCP storage and a Catalog test double. It does not
prove trained-checkpoint quality, SpecForge production retention/consumption,
host-tier cache integration,
cross-node pipeline RDMA, additional model families or latency/throughput SLOs.
Only one speculative batch is outstanding; this is not asynchronous pipeline
microbatch scheduling. Failed internal model collectives or crashed ranks still
require process-group timeout/restart.

The colocated fixture does not exercise P/D. Separate P2/D2 and P2/D1 tests cover
static target-KV speculation with the existing Mooncake transport.
Combined TP2/PP2 colocated and matching P/D serving now have separate real-model
eager/graph and natural-pressure coverage in [the combined suite](COMBINED_TP_PP.md).

Exact results, source hashes and retained log locations are recorded in
`pipeline-dspark-serving.json`.
