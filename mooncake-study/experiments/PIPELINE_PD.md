# Pipeline DSpark P/D Capture

The synchronous target-KV pipeline loop can drive prefill and decode services
using their existing disaggregated batch planners. P sends target KV and the
first response teacher row. D projects the received target prefix into its local
draft cache, performs speculative verification, and publishes the complete
training snapshot through the existing Mooncake Store writer and Catalog client.
No target prefill is recomputed on D for draft initialization.

## Reproduce

Use two CUDA GPUs, a local Qwen3-0.6B model, the matching Mooncake SDK/master and
the SpecForge reference exporter. The fixture colocates P and D processes on the
two GPUs with bounded memory fractions. Model-stage communication uses NCCL;
the local P/D and Store transport is Mooncake TCP. Catalog is the test double.

```bash
PYTHONPATH=python:/path/to/specforge-reference \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_dspark_pp.py -v -f
```

The complete file covers P2/D2 with a target-KV draft on D alone or on both P and
D, plus P2/D1 with drafts on both sides, each in eager and decode graph modes.
It reuses the P/D runtime assertions for single-token and chunked prompts, prefix
reuse, rejected drafts, mixed batches, invalid first-teacher handoffs, live abort,
and complete sample reads after model producer exit. Independent observers read
online target KV and raw vocabulary scores. Each case checks six snapshots
against those sources while serving, stops both model process trees with a
20-second bounded wait for their descendants to exit, and then reads
all objects again with digest and content validation. The case summary reports
`post_exit_snapshots` only after this second read succeeds.

Run the queue and scheduling checks as complete files:

```bash
PYTHONPATH=python python test/registered/unit/spec/test_dspark_pd_queue.py -v -f
PYTHONPATH=python python test/registered/unit/spec/test_dspark_pp_scheduler.py -v -f
PYTHONPATH=python python test/registered/unit/disaggregation/test_decode_queue_cleanup.py -v -f
PYTHONPATH=python python test/registered/unit/disaggregation/test_pp_transfer_readiness.py -v -f
PYTHONPATH=python python test/registered/unit/training_capture/test_pd_capture.py -v -f
PYTHONPATH=python python test/registered/unit/training_capture/test_config.py -v -f
PYTHONPATH=python python test/registered/unit/server_args/test_server_args.py -v -f
```

## Queue Agreement

`DSparkPDQueueCoordinator` is installed only for the synchronous speculative PP
P/D loop. All stages receive the same requests before entering queue processing.
The existing PP1 and asynchronous AR PP paths retain their existing polling.

`poll(label, entries, is_send=..., metadata_buffers=...)` exchanges phase labels,
ordered request IDs/bootstrap rooms and local poll results across the PP/TP world.
Missing decode metadata holds every rank at `Transferring`. A mismatched nonzero
bootstrap room, request abort or backend failure becomes a common `Failed` state.
Bootstrap readiness is the minimum poll state across all participants. Transfer
queues additionally require every stage to be terminal before consuming either
success or failure: one failed stage cannot release another stage's live buffers.
While any stage is nonterminal, the common result remains `Transferring`.
An abort marker does not bypass this rule while its backend is still transferring.
Poll exceptions, phase mismatch or request-order divergence raise on every
participant after the same exchange, including empty-queue participants.

The ordinary queues consume these decisions for P bootstrap, P waiting-request
validation, P transfer release, D handshake/preallocation and D transfer commit.
They do not independently poll again between common readiness and mutation.
Metadata buffer capacity is reduced before P bootstrap admission. D preallocation
and retracted-request resume use common minimum request-slot/token budgets;
preallocation also limits metadata slots. Physical KV and metadata indices remain
local to each stage. Queue-state disagreement is an error, not an instruction to
guess which local request should be committed.

The four-process test uses actual Gloo exchanges. It checks delayed metadata,
successful completion, corrupt room IDs, one-rank failure/abort, in-flight abort,
delayed failure completion on other stages, sender bootstrap, different capacities,
empty queues, identity/phase/shape mismatches and a
backend poll exception. A separate test calls the actual bootstrap queue to show
that a rank with eight free metadata slots admits only the common budget of one.

## First Teacher

The last P stage's TP0 packs the existing bounded `PrefillTeacherHandoff` after
the sampled-token D2H event completes. A world broadcast supplies the same row
to each P stage before ordinary result processing can send the final KV chunk.
Each stage validates the capture context and sampled token with
`accept_pp_handoffs`. Missing or invalid teacher rows fail capture while ordinary
serving follows its existing result path. Mooncake's existing final-chunk metadata
then transports the row to D; no wire-format extension is introduced here.

P may use AR without a draft. If it loads a target-KV draft, prefill commit skips
draft-context projection because D reconstructs that context from received KV.
The D stages use the existing PP source-KV assembly and accepted-token commit.
Only owner-local captured payloads go to Store; the inference projection
collectives do not replace distributed snapshot ownership.

## Verified Results

The final six-case file passed in **548.107 seconds**, checking **36 snapshots**
against online sources and rereading them after the complete producer process
trees exited. The ordinary AR PP2 regression passed two tests in **139.549
seconds**, including ten more post-exit reads. The final queue file passed both
tests, including four actual Gloo processes, in **12.114 seconds**.

Including configuration, queue, scheduler, colocated PP2 and PP1 draft
regressions, the recorded total is **204 tests / 126 checked snapshots**.
This total excludes earlier iterations. PP1 regressions checked their snapshots
while producers were live; they are not included in the 46 strict P/D post-exit
reads above. Source hashes, log hashes, earlier attempts, environment details and
resource cleanup are retained in [the evidence JSON](pipeline-dspark-pd.json).

## Scope

This path requires DP1/CP1, static target-KV drafts, non-overlap PP scheduling,
dense device KV pools and Mooncake transport. Optimistic prefill, P/D KV offload
and transfer staging require additional integration and are rejected. The
underlying transport supports matching P/D PP or reduction to D=PP1; it does not
support expanding P=PP1 to D=PP2. Hidden-input and confidence-scheduled PP drafts
remain unsupported.

These fixtures do not establish trained-draft quality, production Catalog
retention, real combined TP2/PP2, cross-node pipeline RDMA, asynchronous PP
microbatch throughput, distributed prefill graphs or performance SLOs. Runtime
retraction under P/D memory pressure needs its own evidence beyond queue-budget
checks. Failed internal model collectives or crashed processes still require
process-group failure handling and restart.
