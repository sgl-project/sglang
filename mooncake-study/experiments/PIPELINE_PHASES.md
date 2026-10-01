# DSpark Stage Execution Phases

This change separates the worker's local target forward from acceptance and
target-context commit. It is a dependency for PP DSpark serving; the existing
capability gates remain closed. The current single-stage worker uses the same
phase APIs for its normal serving path.

Run the complete registered files:

```bash
PYTHONPATH=python python test/registered/unit/spec/test_dspark_execution_phases.py -v -f
PYTHONPATH=python python test/registered/unit/spec/test_dspark_target_kv.py -v -f
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_dspark.py -v -f
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_dspark_hidden.py -v -f
```

The unit phase tests call the production worker and target-verify executor with
deterministic target/model-boundary fixtures. They do not initialize a model or
a process group. Two stages use different physical request/KV slots. The first
returns an activation without logits, the last consumes it and accepts, and
both explicitly commit using the same accepted path. Assertions cover:

- No local acceptance or target-KV commit while a non-final stage is yielding
  its activation; remote acceptance still commits local slots.
- Raw capture precedes logit adjustments and grammar masking; accepted capture
  precedes publication and draft context projection.
- Temporary CPU verify lengths are restored, including preparation failure.
- Accepted lengths, block lengths, bonus tokens and next-draft state agree
  between stages, including requests with different accepted lengths.
- Foreign/stale steps, duplicate execution, missing forward/acceptance and
  replacing a local acceptance are rejected.
- Pending or failed prefill/decode, including a failed decode preparation,
  prevents the next forward from reusing live graph buffers; a failed commit
  cannot be retried.
- Prefill waits for the final sample before projecting target KV and accepts
  only the matching batch/result pair.

Real-model regressions run on one H100 with Qwen3-0.6B. The target-KV suite
covers drafts on D alone and on both P and D, each in eager and graph/overlap
modes. The hidden-input suite covers static, cap-accept and compact verification,
each in eager and graph/overlap modes. They use actual Mooncake TCP P/D and Store
with the HTTP Catalog test double, and compare complete stored snapshots against
online source KV/logits. Missing/stale teacher handoffs and cancellation must
remain excluded.

The worker permits one outstanding step. Future PP scheduling must communicate
ordered request/proposal agreement and final acceptance, relay activations before
entering collective KV projection, and provide the draft's shared embedding/head
on every stage. Existing compact graph epilogues may execute acceptance/commit
inside forward and remain restricted to the existing non-PP path. These tests do
not establish full PP serving, source-KV NCCL assembly, distributed scheduling
under cancellation, trained checkpoint quality or latency/throughput SLOs.

Retained run IDs, source hashes and results are in `pipeline-dspark-phases.json`.
