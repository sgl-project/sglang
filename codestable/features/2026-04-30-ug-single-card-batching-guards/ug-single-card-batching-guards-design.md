---
doc_type: feature-design
feature: 2026-04-30-ug-single-card-batching-guards
slug: ug-single-card-batching-guards
status: approved
created: 2026-04-30
roadmap: ug-official-alignment
roadmap_item: ug-single-card-batching-guards
related_architecture: [ug-runtime]
---

# UG Single Card Batching Guards Design

## 1. Decisions And Constraints

This feature is Phase 6 hardening after official VLM, T2I, edit, interleaved, entrypoint, and native-model smoke have already passed. The goal is correctness on one GPU: multiple UG sessions must stay isolated, UG non-causal G/image requests must not contaminate ordinary causal SRT requests, and session release must not free another active session.

Mount points:

- `python/sglang/srt/managers/schedule_batch.py`: defines UG batch compatibility keys, batch-level non-causal flag propagation, and mixed-batch validation.
- `python/sglang/srt/managers/scheduler.py`: uses compatibility keys while selecting prefill/running batches.
- `python/sglang/srt/managers/schedule_policy.py`: keeps UG non-causal requests out of chunked prefill paths that would split an image/G block.
- `python/sglang/srt/model_executor/forward_batch_info.py`: transfers the guarded batch flag into `ForwardBatch`.
- `python/sglang/srt/ug/runtime.py`: owns UG session records, sidecar sessions, close semantics, and debug counters.
- `python/sglang/multimodal_gen/runtime/pipelines/ug.py`: keeps experimental interleaved batch API session ids and results separated.
- Existing UG unit tests and registered scheduler tests are the regression surface.

Hard constraints:

- Single GPU only. Do not add multi-GPU, disaggregation, CFG parallel, or throughput-oriented batching.
- Do not rewrite the ordinary SRT scheduler. Add guards around the existing queue/batch formation points.
- Do not expose KV allocator/page/slot outside SRT.
- Do not import official BAGEL/seed code into runtime.
- If non-causal attention remains batch-global, mixed ordinary causal requests must be rejected before `ForwardBatch` is built.

This feature has no new public API requirement. It productizes the correctness envelope of the experimental UG entrypoints already added by earlier roadmap items.

## 2. Flow

Minimal closure:

```text
incoming reqs
-> scheduler computes UG batch compatibility key
-> only compatible reqs enter the same prefill/running batch
-> ScheduleBatch validates the req list before setting batch-level non-causal flag
-> ForwardBatch sees non-causal only for all-UG-non-causal batches
-> UGSessionRuntime closes/release sidecars per session id
```

Two invariants matter more than throughput:

- A batch is either all ordinary causal or all UG non-causal-query. A mixed batch is a correctness bug because the current attention flag is batch-global.
- Closing one UG session releases only that session and its sidecar sessions. Other UG sessions keep their SRT session id, model state, counters, and sidecar mappings.

The implementation should first strengthen guard tests around existing helpers. Only if tests reveal a real gap should code be changed, and the change should stay local to batch compatibility or session cleanup.

Stop signals:

- If this requires changing attention backends to per-request masks, stop and split a new roadmap item.
- If scheduler changes start affecting ordinary text batching beyond a compatibility guard, stop and redesign.
- If multi-session isolation requires diffusion-side access to KV pages or SRT allocator internals, stop and redesign.

## 3. Verification

Minimal automatic checks:

- Unit tests prove ordinary causal and UG non-causal requests cannot be mixed in a `ScheduleBatch` or selected prefill group.
- Unit tests prove two UG non-causal requests can share the non-causal batch key, and ordinary requests remain causal.
- Registered scheduler test proves UG non-causal requests are not chunked.
- UG runtime tests prove two experimental interleaved sessions keep distinct `session_id`, sidecar ids, counters, and model state.
- UG runtime close tests prove closing one session releases only that session and its sidecars.
- Existing fake and real-backend UG CPU tests still pass.

Remote true-weight smoke is optional for this item because no model math should change. If code touches runtime close or scheduler batch formation, rerun a small BAGEL entrypoint smoke on one idle GPU to confirm `image,text` still works and counters still show one prefill plus multiple G velocity calls.
