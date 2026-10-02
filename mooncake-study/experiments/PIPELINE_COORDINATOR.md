# DSpark Pipeline Stage Coordinator

The coordinator consumes an already agreed static target-KV batch and orders
projection, proposal agreement, target activation transport, final acceptance
and local commit. The colocated synchronous PP scheduler now instantiates it;
see [the serving runbook](PIPELINE_SERVING.md) for real-model integration and
current capability limits. The evidence in this file describes the isolated
coordinator prerequisite.

Run the complete unit files:

```bash
PYTHONPATH=python python test/registered/unit/spec/test_dspark_pp_coordinator.py -v -f
PYTHONPATH=python python test/registered/unit/spec/test_dspark_execution_phases.py -v -f
PYTHONPATH=python python test/registered/unit/spec/test_dspark_target_kv.py -v -f
```

Run the same coordination scenarios over two CUDA devices:

```bash
PYTHONPATH=python python test/registered/storage/test_dspark_pp_coordinator_nccl.py -v -f
```

Run the existing single-GPU target-KV and hidden-input serving regressions:

```bash
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_dspark.py -v -f
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_dspark_hidden.py -v -f
```

## Protocol

The coordinator requires all world ranks, including idle ranks, in a DP1/CP1
PP-major/TP-minor layout. One coordinator owns one outstanding batch and its PP
communication groups. The caller resolves inputs and isolates scheduler fields;
after return it installs the common next-draft state and processes each local
result. Existing asynchronous PP loops cannot call it while other stages are
still waiting to receive that turn's requests.

1. Fence capability and batch validation, then compare logical request/step
   identities. Ordered IDs, full token-history digests, committed/output lengths,
   prefix lengths, input/anchor digest, prefill ranges and static dimensions must
   match. Physical pool indices are excluded.
2. Compare local projected-prefix state and draft weight identity. For a request
   with divergent cached ends or stale state, invalidate all local replicas so
   their next projection traverses the same full prefix. Reject different weights.
   For decode, complete this projection before preparing proposals.
3. Prepare all local proposals. Broadcast the last PP stage's TP0 token block.
   Keep corrected logits only on that owner, which performs acceptance. Replace
   static verify IDs and rebuild the grammar tree before target execution.
4. Run each target stage and pass its activation frame to the next stage. The
   frame is bound to the agreed step and source stage. No accepted-KV projection
   occurs while another target stage is waiting for its activation.
5. After all stages finish, broadcast the final prefill sample or acceptance.
   Validate accepted lengths, prefix arithmetic, bonus and accepted draft tokens
   on every rank before any commit. Ignore padded rejected suffixes.
6. Commit with each stage's own physical slots and capture ticket. Increment the
   coordinator sequence only after all commits return. A failed step requires
   worker restart and cannot be retried through the coordinator.

## Evidence Scope

The distributed fixture uses real Gloo collectives for TP2/PP2 and TP1/PP4, and
real two-H100 NCCL transport for TP1/PP2. It calls the production worker's proposal
installation, prefill/verify forward, acceptance and commit phase methods, plus
the production target-verify executor. Model execution, proposal creation,
acceptance kernels and KV projection are deterministic test boundaries. The
projection fixture uses collectives only for missing prefixes, exposing divergent
cached-state ordering rather than always entering a fixed collective sequence.

Successful cases include idle, prefill, two sequential decode steps, different
local proposals, greedy and mixed sampling metadata, grammar-tree rebuilding,
different physical request/KV slots and excluded rejected suffixes. Only the
canonical owner performs acceptance with its matching corrected logits. Raw
capture precedes accepted capture, which precedes local context commit. Missing,
shorter, stale-version and retracted projected prefixes cause a common rebuild.

Failure cases cover request order/history, idle/busy disagreement, CPU/device
prefix mismatch, proposal preparation, bad anchors and tokens, target forward,
stale activation frames, wrong accepted tokens/lengths, invalid prefill samples,
receiver allocation and different draft weights. All ranks reject these cases
without accepted-KV commit; failed coordinators reject another call. A fresh
fixture then runs on the same healthy process groups. This is not recovery of
partially mutated real workers or failed internal model collectives.

The existing real-model regressions exercise the ordinary PP1 worker with actual
Mooncake TCP P/D transport and Store, plus a Catalog test double. They do not
execute the new coordinator. They check target-KV drafts on D alone and on P/D,
and hidden-input static/cap-accept/compact paths with eager and graph/overlap
execution. Snapshot readback uses independent online KV/logit observations.

The separate serving test now covers TP1/PP2 model initialization, scheduling,
cancellation/retraction, actual PP capture and source-KV NCCL projection.
P/D readiness has separate coverage in [the P/D runbook](PIPELINE_PD.md).
Latency/throughput validation remains required. Boundary
fences cannot recover a crashed process, failed device communication or a rank
stranded inside an internal model/injector collective. Process-group failure
handling and restart still apply. The synchronous ordering, metadata validation
and rebuild-on-divergence policy have not been benchmarked.

Exact source/log hashes and run results are retained in
`pipeline-dspark-coordinator.json`.
