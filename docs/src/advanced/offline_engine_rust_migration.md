# Offline Engine Rust migration

This plan covers offline `sglang.Engine` support in the Rust server migration
described by [RFC #23206](https://github.com/sgl-project/sglang/issues/23206).
An offline Engine is constructed in-process and called directly by an
application. The migration moves its request front half to Rust, up to
submission to the existing Python scheduler.

## Scope and ownership boundary

The Rust implementation owns the offline Engine request lifecycle through
validation, preprocessing, tokenization, admission to the frontend queue,
submission to the scheduler, response streaming, detokenization, and API-side
control requests. The existing Python scheduler and GPU worker continue to own
scheduling, model loading and execution, attention backends, KV-cache
internals, and quantization. Rust sends requests across the established
Rust/Python boundary; it does not implement a second scheduler.

The public Python `sglang.Engine` import and API can remain as a compatibility
entry point. Its Rust-backed path must delegate request handling to Rust while
the Python scheduler remains the execution backend. This migration does not
remove Python model execution.

| In scope for this migration | Remains in the Python execution backend |
| --- | --- |
| Offline Engine construction and lifecycle integration with the Rust runtime | Scheduler algorithms, batching decisions, and scheduler state |
| Request conversion, validation, preprocessing, tokenization, and detokenization | Model loading and model-runner orchestration |
| Sync/async request handling, streaming, cancellation, and result conversion | GPU workers, kernels, attention backends, and sampling during execution |
| Engine control operations and their scheduler handoff | KV-cache internals, quantization, and model execution |
| Typed metadata handoff, tensor-buffer ownership, backpressure, and error propagation | Existing distributed scheduler and worker coordination |

The Rust HTTP server already implements parts of the same front half, but HTTP
route parity does not establish the in-process `sglang.Engine` contract. The
Python Engine currently rejects `SGLANG_RUST_SERVER`; offline support is still
missing.

The separate `rust/sglang-engine` crate contains lifecycle foundations and
standalone scheduler/model prototypes from earlier exploration. Those pieces
are not the production offline Engine path and do not move scheduler or model
execution into scope for RFC #23206. Reuse them only where they fit the Rust
server architecture and its existing scheduler boundary.

## Migration phases

### Phase status

| Phase | Status | Evidence / remaining work |
| --- | --- | --- |
| 0. Offline Engine contract inventory | Baseline recorded | Public API groups, startup owners, runtime context, and request/control families are listed below. Expand each row into argument and behavior parity tests as it is implemented. |
| 1. Rust request-front-half foundation | In progress in the Rust server migration | `sglang-server` has a Rust HTTP/tokenizer request pipeline and typed bounded channels to and from the Python scheduler. The RFC reports the single-process framework, tokenizer/detokenizer, and `/generate` streaming path implemented; reconcile that checklist against the current branch before treating each item as complete. |
| 2. Offline Engine integration | Not supported | `python/sglang/srt/entrypoints/engine.py` rejects `SGLANG_RUST_SERVER`. The Rust server does not yet expose the in-process Engine contract. |
| 3. Offline Engine feature parity | Not started | The Python Engine remains the only implementation for offline generation and its related API/control operations. |
| 4. Differential qualification and rollout | Not started | No offline Engine parity suite or rollout gate currently qualifies the Rust path. |

### Public API parity inventory

This inventory is based on `Engine`, `EngineBase`, and `EngineScoreMixin`.
Rust HTTP endpoints that happen to expose similar operations do not count as
offline Engine parity until the in-process API, ownership, and lifecycle
contracts are satisfied.

| API group | Current offline Engine operations | Rust offline Engine status |
| --- | --- | --- |
| Construction and ownership | `Engine(**kwargs)`, `server_args=...`, child-PID reporting, attached tokenizer worker, context manager, shutdown | Not supported |
| Generation | `generate`, `async_generate`; prompts or token IDs; batch/scalar forms; sampling; streaming; logprobs; hidden states; routed experts; multimodal inputs; LoRA; sessions; DP/disaggregation routing; request IDs, priority, tracing, and cache hints | Not supported |
| Embeddings and ranking | `encode`, `async_encode`, `rerank`; text/token IDs, multimodal inputs, dimensions/format, LoRA and embedding overrides | Not supported |
| Scoring | `score`, `async_score`; causal-LM and sequence-classification results, setwise extraction, softmax/calibration, token logprobs, embedding overrides | Not supported |
| Sessions and cache | `open_session`, `close_session`, `flush_cache` | Not supported |
| Profiling and diagnostics | Start/stop profile; start/stop/dump expert-distribution recording; `get_server_info`, `get_model_info`; `collective_rpc` | Not supported |
| Weight-update lifecycle | Initialize/destroy update group; begin/end update | Not supported |
| Weight operations | Updates from tensors, disk, distributed sources, and IPC; get weights by name; tensor serialization/layout handling | Not supported |
| LoRA | Load from path or tensors, unload, async load/unload, pinned adapters | Not supported |
| Runtime controls | Release/resume memory occupation with tags; `freeze_gc`; `save_remote_model`, `save_sharded_model` | Not supported |
| Base interface | Inherited `EngineBase` generate/control methods, including flush, weight updates, LoRA, memory controls, shutdown | Not supported |

Each row needs a detailed contract as work begins: accepted input forms,
defaults and validation, return structures, exceptions, ordering, side effects,
cancellation behavior, and ownership of Python objects and tensor storage.

### Runtime ownership and handoff

| Responsibility | Existing owner | Target for this migration |
| --- | --- | --- |
| Public Python imports and compatibility | `sglang.Engine` in `engine.py`, `EngineBase.py`, and `engine_score_mixin.py` | Preserve public entry points while routing the supported offline path through the Rust request runtime. |
| Offline argument construction | `ServerArgs`, `resolve_once()`, and `check_server_args()` in Python | Define the supported offline argument subset and preserve defaults, validation, and error behavior at the Rust boundary. Do not port unrelated server-only settings without a demonstrated dependency. |
| Rust request pipeline | `sglang-server` Rust runtime and tokenizer manager | Reuse the same request lifecycle and components for offline calls; do not create a parallel Rust Engine pipeline. |
| Scheduler and model execution | Python `scheduler.py` and `model_executor/model_runner.py` | Keep Python as the execution backend and preserve the existing scheduler boundary. |
| Scheduler request/response handoff | Rust `ToScheduler`/`FromScheduler` channels with the Python bridge | Reuse and qualify the typed handoff, serialization, backpressure, error propagation, and shutdown behavior for offline requests. |
| Tensor and multimodal payloads | Python objects, shared memory, and Rust message buffers depending on path | Carry metadata separately from large payloads; define buffer lifetime and transfer ownership through tests for each supported payload type. |
| Engine lifecycle and child processes | Python `Engine` and its managers/workers | Bind offline Engine startup, readiness, cancellation, and shutdown to Rust runtime lifecycle while retaining scheduler/GPU worker cleanup semantics. |

The RFC architecture uses one API-server process, Rust request/tokenizer and
detokenizer workers, and the existing Python scheduler group. Cross-group
communication uses bounded channels; scheduler metadata crosses the
Rust/Python boundary as msgpack, while large tensor payloads use shared-buffer
handles. The offline Engine path must use these same ownership and
backpressure rules. It must not introduce ZMQ or a second scheduler path where
the established in-process boundary already applies.

## Phase 0: Offline Engine contract inventory

- [x] Record public operations from `Engine`, `EngineBase`, and the score
  mixin, grouped by API family.
- [x] Record current owners for Engine startup, tokenizer/detokenizer work,
  request handoff, scheduler execution, and model execution.
- [ ] For every operation, record input forms, defaults, outputs, errors,
  streaming/cancellation behavior, tensor ownership, and side effects.
- [ ] Map each operation to its Python request/control type, Rust message,
  transport, and differential test.
- [ ] Identify which `ServerArgs` and runtime-context fields are required by
  offline Engine support; classify the rest as out of scope with evidence.

**Exit criteria:** each public Engine method is accounted for and has a
reviewable Rust boundary contract and test plan. The current inventory is a
baseline; method-level behavior and argument coverage remain to be expanded.

## Phase 1: Rust request-front-half foundation

Reuse and complete the Rust server migration foundation from RFC #23206:

- Rust request lifecycle and tokenizer/detokenizer managers.
- API ingress and response streaming with ordering and cancellation.
- In-process Python scheduler integration using the established channel and
  buffer boundary.
- Typed control messages, bounded queues, backpressure, and failure propagation.
- Startup/readiness, metrics/logging, and deterministic worker shutdown.

Offline Engine work depends on the components above. It should extend their
configuration and entry points rather than create duplicate queue, tokenizer,
or scheduler state.

**Exit criteria:** the Rust front half can accept, tokenize, hand off, stream,
cancel, and clean up requests against the existing Python scheduler, with
contract tests for the Rust/Python boundary. This phase does not require Rust
model execution.

## Phase 2: Offline Engine integration

Implement the in-process Engine path using the Rust request runtime and the
existing Python scheduler backend:

1. Define the Engine-facing bridge and lifecycle contract. Preserve public
   imports, constructor behavior, context-manager support, child ownership, and
   shutdown semantics.
2. Convert offline Engine inputs to the Rust request/control messages. Preserve
   validation, preprocessing, tokenizer behavior, tensor ownership, and error
   mapping.
3. Route sync and async generation through the Rust request lifecycle and
   established Python scheduler handoff. Preserve batching, stream frame
   ordering, cancellation, and output conversion.
4. Implement offline Engine startup/readiness and resource cleanup across the
   Rust workers and Python scheduler group, including startup failures and
   repeated shutdown.
5. Remove the `SGLANG_RUST_SERVER` rejection only when the Rust-backed offline
   path is usable for its explicitly supported feature set.

**Exit criteria:** a Rust-backed offline Engine completes tokenized generation
requests through the Python scheduler with correct streaming, cancellation,
errors, and lifecycle behavior. Scheduler selection and model execution remain
Python-owned.

## Phase 3: Offline Engine feature parity

Expand coverage across the parity inventory in dependency order:

- Generation variants: token IDs, batches, sampling options, logprobs, hidden
  states, multimodal inputs, LoRA, sessions, and routing metadata.
- Embedding, reranking, and scoring methods and their output contracts.
- Session/cache operations, profiling, diagnostics, RPC, and memory controls.
- Weight-update groups and tensor/disk/distributed/IPC weight operations.
- LoRA path/tensor and asynchronous operations, plus model-save methods.
- All inherited `EngineBase` methods, errors, and shutdown interactions.

For each feature, add it to the matrix with an explicit status and link its
implementation and differential tests. Unsupported modes must have a named
reason and release decision; they must not be silently treated as migrated.

**Exit criteria:** every supported API inventory row has an implementation and
passing contract/parity tests, or a documented exclusion for the target
release.

## Phase 4: Differential qualification and rollout

- Compare Python and Rust-backed offline Engine behavior on shared fixtures:
  request acceptance, tokenization, token IDs, outputs, stream frames,
  logprobs, exceptions, cancellation, and control side effects.
- Exercise concurrency, queue saturation, slow consumers, worker failures,
  startup timeouts, partial startup, signal handling, repeated shutdown, and
  buffer cleanup.
- Run supported CPU/GPU and distributed configurations where they affect the
  front-half contract. GPU model correctness remains covered by the existing
  Python execution backend tests.
- Benchmark request-front-half latency, throughput, memory, and backpressure
  against the Python Engine path; investigate material regressions.
- Roll out behind the existing Rust-server selection mechanism with a defined
  fallback and release policy.

**Exit criteria:** the supported offline Engine matrix passes parity and
reliability gates, performance is qualified, and rollout/fallback behavior is
documented.

## Testing and coverage requirements

- **Unit tests:** argument conversion, defaults, validation, message encoding,
  request-state transitions, stream ordering, buffer ownership, and every
  operation's result/error mapping.
- **Integration tests:** offline Engine startup, single and batched requests,
  streaming, cancellation, concurrency/backpressure, controls, startup
  failures, worker failure, and cleanup through the real Rust/Python boundary.
- **Differential tests:** run equivalent requests through Python Engine and
  Rust-backed Engine and compare inputs accepted, tokenized requests, output
  structures, stream frames, errors, and side effects.
- **Coverage accounting:** link every public method and supported mode to
  tests. Cover success and failure paths for each conditional transport and
  ownership path; register hardware-dependent cases in the appropriate CI
  tier.
- **Documentation:** document the offline API support matrix, configuration
  and limitations, Python compatibility behavior, examples, and rollout
  controls.

## Working rules

- Keep the public API matrix, phase status, and test evidence current with each
  implementation slice.
- Keep scheduler algorithms, model execution, kernels, attention, KV-cache
  internals, and quantization outside this RFC migration.
- Reuse the Rust server request path and Python scheduler boundary. Do not
  count similar HTTP endpoints or standalone scheduler/model prototypes as
  offline Engine parity.
- Do not mark offline Engine support complete until its supported API matrix
  passes differential, lifecycle, and failure-path qualification.
