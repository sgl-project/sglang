# RFC: Active Incremental Prompt Ingestion and Prefill-Only Scheduling

- **Status:** Draft
- **Authors:** Community proposal
- **Target:** SGLang runtime and serving APIs
- **Scope:** Text-only first; multimodal inputs are out of scope for the initial design

## Summary

This RFC proposes a serving primitive that allows an application to append prompt
tokens to one active request over time. SGLang would prefill each newly available
chunk, retain the resulting KV state, and postpone autoregressive decoding until
the producer explicitly marks the prompt complete.

The proposal is intended for workloads in which prompt tokens are produced by
another model, tool, parser, retriever, or streaming upstream service. The
downstream model should be able to consume available context as it arrives,
without waiting for the complete prompt and without resending the complete
prefix for every chunk.

The core behavior is:

```text
open request
  -> append chunk A -> prefill A
  -> append chunk B -> prefill B
  -> append chunk C -> prefill C
  -> finalize prompt
  -> decode response
```

This is not a proposal to change ordinary response streaming, chunked prefill of
a known prompt, or the existing session-aware radix cache.

## Motivation

Many agent and pipeline workloads have a producer-consumer shape:

```text
upstream model / tool / retriever
              │ token chunks
              ▼
        downstream LLM
```

Today an application generally has to choose one of the following behaviors:

1. Wait for the upstream producer to finish and issue one ordinary request. This
   increases time to first token and leaves the downstream GPU idle while the
   prompt is being produced.
2. Issue one downstream request per chunk. This introduces request overhead,
   requires prompt reconstruction, and can repeat tokenization or prefix
   matching work.
3. Start decoding before the producer is finished. This risks producing output
   conditioned on an incomplete prompt and does not provide a clear way to
   continue prefill safely.

The desired primitive should expose the boundary between prompt ingestion and
generation explicitly. It should let the scheduler overlap upstream production
with downstream prefill while preserving normal causal attention semantics.

## Existing SGLang capabilities and the gap

SGLang already has several building blocks that are related to this proposal:

- [Session-aware radix cache](../docs/advanced_features/session_radix_cache.mdx)
  protects reusable KV associated with a long-lived session. Its documented
  contract is reference-only: `session_id` does not append or reconstruct a
  conversation, and each request still supplies the prompt for that turn.
- [`StreamingSession`](../../python/sglang/srt/session/streaming_session.py)
  owns or lends KV records across turns and handles prefix matching, retraction,
  and session lifecycle. Its current streaming-session semantics are
  append-only across completed/retracted turns, rather than token ingestion into
  one request that is still waiting for finalization.
- SGLang supports normal streaming responses and chunked execution of a prompt
  whose contents are already known. Neither changes the request's prompt after
  admission.
- [Incremental chat tokenization](https://github.com/sgl-project/sglang/pull/41149)
  reduces repeated CPU tokenization for multi-turn chat histories. It does not
  incrementally execute GPU prefill or keep a generation request open while new
  prompt tokens arrive.

The gap is therefore an in-flight request lifecycle with an explicit
`append -> prefill -> append -> ... -> finalize -> decode` contract. In
particular, the proposed feature must not be implemented as repeated ordinary
turns using `session_id`: that would retain the existing turn boundary and would
not provide the same request-level semantics or avoid resubmitting the full
prompt.

## Goals

1. Allow a request to receive multiple prompt chunks without resending the
   already committed prefix.
2. Prefill available chunks incrementally and reuse the request's existing KV
   state.
3. Prevent sampling and visible response generation until the producer calls
   `finalize`.
4. Preserve causal equivalence with one ordinary request containing the final
   concatenated prompt, subject to the normal scheduling and numerical behavior
   of SGLang.
5. Reuse SGLang's existing request, scheduler, radix-cache, and session
   lifecycle machinery where possible.
6. Make cancellation, backpressure, memory limits, and client disconnects
   explicit rather than leaving them as transport-specific behavior.

## Non-goals

- Replacing or changing the semantics of ordinary response streaming.
- Replacing session-aware radix caching or multi-turn session reuse.
- Automatically reconstructing chat history from a `session_id`.
- Supporting arbitrary edits to the middle of a prompt in the first version.
- Making multimodal encoder outputs incrementally appendable in the first
  version.
- Performing generation between chunks. The request remains in prompt-ingestion
  mode until finalized.

## Proposed request model

The exact public API is intentionally left open for maintainer feedback. The
minimum logical interface is:

```text
create() -> request_id
append(request_id, input_ids | text_chunk) -> accepted range / status
finalize(request_id) -> generation stream
abort(request_id) -> terminal status
```

The initial implementation should prefer token IDs for the append path. This
avoids making chunk boundaries depend on tokenizer behavior and lets an upstream
component that already has token IDs avoid an unnecessary decode/re-tokenize
round trip. A text-based adapter can be added if its boundary and tokenizer
state semantics are well-defined.

Possible transports include a dedicated extension endpoint, a bidirectional
stream, or an internal API used by an agent runtime. The RFC does not require
that the feature be exposed through the OpenAI-compatible API in its first
iteration.

### Request states

The request should have an explicit state separate from the existing response
stream state:

```text
OPEN
  -> RECEIVING
  -> PREFILLING
  -> RECEIVING       (after a chunk is committed)
  -> FINALIZING
  -> DECODING
  -> FINISHED

Any non-terminal state -> ABORTED
```

`PREFILLING` is an internal scheduler state. The client may append while a
previous chunk is being processed, but chunks must be ordered and committed
atomically. The implementation may apply backpressure instead of accepting an
unbounded pending buffer.

## KV-cache and scheduler semantics

The request needs a clear distinction between prompt tokens that have been
accepted by the control plane and tokens whose KV has been computed:

- `committed_len`: prompt tokens for which the request owns valid KV state;
- `pending_len`: accepted prompt tokens waiting to be prefetched;
- `allocated_len`: KV capacity currently reserved for the request.

For each append operation:

1. Validate request state, token limits, and ordering.
2. Add the new token IDs to the request's logical input buffer.
3. Schedule a prefill for the range beginning at `committed_len`.
4. Commit `committed_len` only after the model forward and KV write complete.
5. Keep the request in prompt-ingestion mode; do not sample a response token.

The scheduler should be able to batch an incremental-prefill request with other
prefill work, subject to the configured token budget. It should not be treated
as an ordinary decode request merely because it has already executed one or
more prefill steps.

The request's cache ownership must remain stable across chunks. A chunk that is
rejected, cancelled, or fails during prefill must not publish partially valid
KV as if it were committed. If SGLang uses a temporary allocation or a
checkpoint/retract path for a chunk, the rollback target should be the last
committed prompt boundary.

The design should integrate with existing session behavior without conflating
the two concepts:

- a streaming-prompt request owns an in-flight request and its unfinalized
  prompt;
- a session-aware radix cache may optionally retain the finalized prefix for
  later requests after the request completes.

## Finalization and decoding

`finalize` is a synchronization barrier. It must wait for or arrange completion
of all accepted prompt chunks before normal generation begins. Once finalization
is acknowledged:

1. no further append is accepted;
2. the final prompt length becomes immutable;
3. the request enters the normal decode path;
4. response streaming follows the existing SGLang behavior.

If the client disconnects before finalization, the default should be to abort
the request and release its uncommitted resources. A configurable retain/close
policy may be considered later for agent runtimes that intentionally reconnect.

## Ordering, concurrency, and backpressure

The first version should define conservative semantics:

- one append stream per request;
- monotonically increasing chunk sequence numbers;
- at most one in-flight prefill mutation for a request;
- append acknowledgement only after the chunk is accepted into the ordered
  pending buffer;
- a bounded pending-token budget with explicit `RESOURCE_EXHAUSTED` or
  backpressure behavior;
- abort is idempotent and wins over a not-yet-started append;
- finalize is idempotent after it has been accepted, but append after finalize
  is rejected.

These rules avoid races between scheduler retraction, client cancellation, and
session cache publication. More aggressive overlap between multiple chunks can
be explored after the single-stream state machine is correct.

## Failure and recovery behavior

The implementation should document and test the following cases:

- an append arrives while the request is queued;
- an append arrives while a previous chunk is prefilling;
- a chunk exceeds the context limit;
- KV allocation fails after the control plane accepted the chunk;
- a client disconnects during prefill;
- abort races with completion of a prefill step;
- finalize races with the last append;
- a session containing a streaming-prompt request is closed;
- a worker or scheduler failure occurs before a chunk is committed.

The safe recovery point is the last committed prompt boundary. Reusing a
partially written chunk must never be required for correctness.

## Compatibility and observability

This feature should be opt-in and should not alter existing requests. Suggested
metrics and tracing fields include:

- number and size of appended chunks;
- time from append acceptance to KV commit;
- pending-token queue depth;
- prefill tokens per scheduler step;
- time from create to finalize;
- time from finalize to first output token;
- aborted requests and rollback counts;
- KV allocation and eviction attributable to incremental requests.

The request status should expose enough information for an upstream producer to
apply backpressure without depending on private scheduler logs.

## Correctness and performance evaluation

The initial test plan should include a small deterministic model and synthetic
token streams so that the state machine can be exercised without requiring a
large GPU:

1. Split a fixed prompt into 1, 2, 4, and many chunks.
2. Compare incremental-prefill logits and greedy output with one ordinary
   request containing the concatenated prompt.
3. Test append during queued, running, and completed prefill states.
4. Test finalize, abort, client disconnect, context overflow, and allocation
   failure paths.
5. Verify that no output token is emitted before finalization.
6. Verify that a failed chunk rolls back to the preceding committed boundary.
7. Measure prompt-token-to-KV-commit latency, finalize-to-first-token latency,
   scheduler overhead, and peak KV usage.
8. Compare against a baseline that resubmits the complete prompt for every
   chunk and against a baseline that waits for the complete prompt.

The most important correctness invariant is:

```text
decode(final_prompt after incremental prefill)
==
decode(final_prompt after one ordinary prefill)
```

for identical model, tokenizer, sampling, and random-seed configuration, within
the numerical tolerance already accepted by the relevant SGLang backend.

## Implementation sketch

An incremental implementation could be staged as follows:

### Phase 1: Runtime state machine

- Add a request mode for prompt ingestion.
- Add ordered append/finalize/abort operations.
- Add `committed_len` and pending-range bookkeeping where existing KV metadata
  is insufficient.
- Keep decoding disabled until finalization.

### Phase 2: Scheduler integration

- Admit incremental requests into prefill scheduling.
- Reuse the existing extend/prefill path for each pending range.
- Add bounded backpressure and explicit rollback on failed prefill.
- Ensure retraction and cancellation return the request to its last committed
  boundary.

### Phase 3: Serving API and tests

- Add one experimental transport/API surface.
- Add CPU/state-machine tests and a small GPU integration test.
- Add metrics and documentation.
- Evaluate whether session-aware cache publication after finalization should be
  enabled by default or remain opt-in.

## Alternatives considered

### Resubmit each turn with `session_id`

This uses existing session KV reuse, but it retains a completed-request turn
boundary, requires the client to submit the intended prompt for every turn, and
does not provide an in-flight prompt-ingestion contract.

### Wait for the complete prompt

This is simple and preserves today's semantics, but it loses the opportunity to
overlap upstream token production and downstream prefill.

### Start ordinary generation and pause it

This complicates sampling, output correctness, and rollback. It also risks
exposing tokens generated from an incomplete prompt. A prefill-only mode with an
explicit finalization barrier is easier to reason about.

### Use incremental tokenization only

Incremental tokenization can reduce CPU work, but it does not compute KV for the
new tokens and therefore does not solve the GPU scheduling problem addressed by
this RFC.

## Open questions for SGLang maintainers

1. Should this be modeled as a new request mode, an extension of
   `StreamingSession`, or a separate serving primitive?
2. Should the first API accept token IDs only, or should SGLang own text chunk
   tokenization and tokenizer-boundary handling?
3. Can the existing extend/prefill and retraction paths provide the required
   commit/rollback semantics without introducing a second KV ownership model?
4. Should unfinalized requests participate in session-aware radix-cache
   publication, or only publish after successful finalization?
5. What pending-token and KV reservation policy best protects other requests
   from a producer that stalls or never finalizes?
6. Is a bidirectional streaming transport preferable to separate append and
   finalize endpoints for the first experimental implementation?
7. Which scheduler metrics and invariants should be required before enabling the
   feature outside an experimental flag?

## Request for feedback

This is a design-only proposal. We would appreciate feedback on whether the
request lifecycle and KV ownership model fit SGLang's scheduler architecture,
which existing abstractions should be reused, and what API shape would be
acceptable before implementation begins.

