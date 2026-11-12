# Offline Engine Rust Migration

This document defines the staged migration of SGLang's offline `Engine` from
Python to Rust. “Offline” means an application constructs an engine in-process
and calls its API directly; it does not mean that model execution or scheduler
work is outside the migration. The target is a Rust-owned engine lifecycle and
runtime, including scheduler and model execution. A Python compatibility layer
may remain temporarily, but it must delegate to Rust and must not retain a
second independent implementation at completion.

## Current boundary

The Python implementation is centered on:

- `python/sglang/srt/entrypoints/engine.py`: configuration, process startup,
  public synchronous/asynchronous methods, child process ownership, and shutdown.
- `python/sglang/srt/entrypoints/EngineBase.py` and
  `engine_score_mixin.py`: shared and scoring API behavior.
- `python/sglang/srt/managers/tokenizer_manager.py`,
  `detokenizer_manager.py`, `data_parallel_controller.py`, and `scheduler.py`:
  request processing, process topology, scheduling, and model execution via
  `python/sglang/srt/model_executor/model_runner.py`.

The existing Rust `sglang-server` crate implements a substantial serving
frontend and tokenizer/request pipeline. Its runtime is currently embedded
through a PyO3 module in the Python process, and the scheduler/model execution
path remains Python. It is useful migration infrastructure, but it is not yet
a Rust offline Engine or a Rust scheduler/model runtime.

## Target and completion criteria

The Rust implementation owns configuration resolution, startup/readiness,
request submission and streaming, scheduler/model execution, process/resource
lifecycle, and all public Engine operations. Python remains only as a
compatibility adapter while required; it forwards calls to the Rust
implementation and contains no independent Engine behavior.

Completion requires all of the following:

- Every supported Python `Engine` behavior is represented in the Rust API and
  documented, including inputs, outputs, defaults, errors, streaming, and
  lifecycle semantics.
- Scheduler and model execution run through Rust-owned runtime components;
  Python is not required to start or operate an engine.
- Python compatibility tests demonstrate delegation and parity, or the Python
  surface is removed through an explicitly documented compatibility change.
- Unit, integration, parity, fault-injection, and supported-hardware tests cover
  the behavior and failure paths described below.
- No feature is silently omitted: unsupported hardware, modes, or model
  features are reported in a maintained parity matrix with an explicit plan or
  documented exclusion approved for the target release.

## Migration phases

### Phase status

| Phase | Status | Evidence / remaining work |
| --- | --- | --- |
| 0. Behavioral inventory and contracts | Complete (baseline) | Public method groups, startup owners, runtime-context reads, IPC families, and initial parity gaps are recorded below. Contract detail is tracked at the feature/method-family level; per-model and per-argument parity is expanded with each implementation slice. |
| 1. Rust crate, configuration, and runtime foundation | Complete | `sglang-engine` has typed seed config, lifecycle/error types, and a process supervisor with explicit readiness, startup timeout/early-exit handling, and child cleanup. Scheduler-specific `ServerArgs` and runtime context are Phase 2 inputs because their semantics are implemented by the scheduler/model runtime being migrated there. |
| 2. Rust scheduler and model execution | In progress | Rust has queue ordering policies, bounded pending/running admission, a `SchedulerRuntime`, a typed model-execution contract, and CPU reference sampling for greedy, temperature, top-k, and top-p paths. The only executor is test-only; Python still owns production admission, cache matching, batching, and all model loading/execution. |
| 3. Offline Engine API parity | Not started | Existing Rust server is not an offline Engine API. |
| 4. Python compatibility adapter and migration | Not started | Python `sglang.Engine` remains the implementation. |
| 5. Qualification, rollout, and removal | Not started | No Rust offline Engine qualification or rollout. |

### Public API parity matrix

This is the initial inventory from `Engine`, `EngineBase`, and
`EngineScoreMixin`. “No” means there is no Rust offline Engine implementation;
similar Rust server routes or control handlers do not count as parity because
they do not provide the in-process offline Engine contract.

| API group | Current public operations | Rust offline Engine parity |
| --- | --- | --- |
| Construction and ownership | `Engine(**kwargs)`, `server_args=...`, child-PID reporting, attached tokenizer worker, context manager, shutdown | No |
| Generation | `generate`, `async_generate`; prompts or token IDs; batch/scalar forms; sampling; streaming; logprobs; hidden states; routed experts; multimodal inputs; LoRA; sessions; DP/disaggregation routing; request IDs, priority, tracing and cache hints | No |
| Embeddings and ranking | `encode`, `async_encode`, `rerank`; text/token IDs, multimodal inputs, dimensions/format, LoRA and embedding overrides | No |
| Scoring | `score`, `async_score`; causal-LM and sequence-classification results, setwise extraction, softmax/calibration, token logprobs, embedding overrides | No |
| Sessions and cache | `open_session`, `close_session`, `flush_cache` | No |
| Profiling and diagnostics | start/stop profile; start/stop/dump expert-distribution recording; `get_server_info`, `get_model_info`; `collective_rpc` | No |
| Weight update lifecycle | initialize/destroy update group; begin/end update | No |
| Weight operations | update from tensors, disk, distributed sources, and IPC; get weights by name; tensor serialization/layout handling | No |
| LoRA | load from path or tensors, unload, async load/unload, pinned adapters | No |
| Runtime controls | release/resume memory occupation with tags; `freeze_gc`; model save operations (`save_remote_model`, `save_sharded_model`) | No |
| Base interface | inherited `EngineBase` generate/control methods, including flush, weight updates, LoRA, memory controls, shutdown | No |

The matrix is complete at the method/group level. Contract details remain
implementation acceptance criteria for each slice, including sync/async
behavior, stream iterator semantics, tensor and multimodal input ownership,
and tensor serialization formats.

### Runtime ownership and startup dependency map

| Responsibility | Current implementation | Rust migration work |
| --- | --- | --- |
| Argument construction and validation | `ServerArgs` in `python/sglang/srt/server_args.py`; `resolve_once()` and `check_server_args()` run before launch | Port all fields that affect offline Engine behavior, derived defaults, validation and error parity; keep secrets redacted in logs. |
| Process-global config | `runtime_context.publish()` projects `ServerArgs` into parallel/device/model/exec/schedule/memory/spec/LoRA/MM/disagg/serving/observability namespaces | Replace ambient Python globals with explicit immutable config/context handles and scoped per-worker state. Preserve role/rank provenance and resource leases. |
| Port and IPC endpoint allocation | `PortArgs.init_new()` plus temporary IPC paths in `engine.py` | Typed endpoint allocation, collision handling, cleanup, local vs distributed addressing, and test injection. |
| Scheduler launch and readiness | `Engine._launch_scheduler_processes()` starts TP/PP workers or a DP controller and consumes readiness pipes | Rust-owned process topology, rank/device mapping, readiness protocol, child supervision, crash propagation, and shutdown. |
| Model loading and execution | `scheduler.py` and `model_executor/model_runner.py`, invoked by `run_scheduler_process()` | Port scheduler and model execution; implement backend/kernel interfaces and model-family support matrix. |
| Tokenizer and request admission | Python `TokenizerManager` or Rust `sglang-server` tokenizer manager, selected by launch mode | Make Rust implementation the offline Engine owner; parity for normalization, validation, preprocessing, tokenization, queueing and cancellation. |
| Detokenization | Python `run_detokenizer_process()` workers and optional router | Rust worker implementation and routing, output ordering, stop conditions and cleanup. |
| Auxiliary processes | Weight-cache daemons, expert backup manager, optional follower gRPC/bootstrap services | Rust ownership or explicit component interface for every supported daemon; lifecycle tied to Engine and process tree. |
| Engine request/control transport | Python `TokenizerManager` async APIs plus ZMQ RPC; startup uses multiprocessing pipes; tensor payloads can use shared memory | One versioned protocol with typed schemas, bounded payloads, backpressure, ownership transfer, cancellation, and compatibility tests. |
| Shutdown and signals | `shutdown()`, atexit, watchdog, process joins/termination, follower SIGTERM path | Deterministic drain, idempotent cleanup, timeout/force-kill policy, crash dumps, and no leaked child processes/shared-memory segments. |

### Existing wire-schema inventory

The Python `io_struct.py` request families directly used by the public Engine
include `GenerateReqInput`, `EmbeddingReqInput`, scoring request structures,
session requests, profile requests, RPC, weight-update group requests, weight
updates from disk/tensor/distributed/IPC, weight reads, LoRA path/tensor
requests, and release/resume-memory requests. `msgspec`/msgpack serialization is
used on the internal control paths; ZMQ carries RPC and worker messages,
multiprocessing pipes carry startup readiness, and shared memory can carry
large tensor/multimodal payloads.

The Rust server has typed generate/request/control wire structures and bounded
channels for the HTTP-serving frontend, but that schema is not yet a canonical
offline Engine protocol. Each API implementation slice must map its operation
to the Python request type, Rust type, transport, tensor ownership rule, and
compatibility test before being marked complete.

Engine startup directly reads runtime context for model/chat-template metadata;
node and DP/TP topology; distributed address; device base ID/stride; weight
cache mode and timeout; elastic expert backup and EP join state; memory-saver
behavior; tracing endpoints/service/modules; crash-dump location; serving host,
port, tokenizer/detokenizer worker counts; metrics; disaggregation role; and
remote weight-loader bootstrap. The concrete access sites are
`Engine.__init__`, `_launch_subprocesses`, `_launch_scheduler_processes`,
`_launch_detokenizer_subprocesses`, and `_launch_weight_cache_daemons`.
`RuntimeContext` additionally owns mutable flags, process resources (CUDA
streams and buffers), role provenance, and forward-pass flags; Rust must model
these as explicit owned/scoped state instead of ambient process globals.

### Phase 0: Behavioral inventory and contracts — complete baseline

- Record the public surface from `Engine`, `EngineBase`, and the score mixin:
  initialization and args, context-manager cleanup, child PIDs,
  `generate`/`async_generate`, `encode`/`async_encode`, `rerank`,
  `score`/`async_score`, sessions, cache operations, profiling and expert
  distribution recording, info queries, weight-update groups and all update
  sources, weight reads, LoRA operations, memory occupation, GC controls, RPC,
  model saving, and shutdown.
- For each operation, capture accepted prompt/media/tensor forms, batch and
  stream behavior, return shape, defaulting and validation, error mapping,
  cancellation, and side effects.
- Inventory `ServerArgs` and runtime-context fields used by Engine startup;
  classify each as supported, intentionally unsupported, or not applicable.
- Establish canonical Rust/Python IPC schemas and compatibility/versioning
  rules. Reuse `io_struct.py` definitions where possible; avoid independently
  maintained wire field lists.
- Produce a checked-in feature matrix mapping every public method and runtime
  mode to implementation, tests, and completion status.

**Exit criteria:** reviewed API contract, runtime dependency map, and complete
feature matrix; no public methods are unaccounted for.

**Baseline outcome:** every public method is represented in the parity matrix;
the startup dependency map identifies process/runtime owners and direct
runtime-context dependencies; internal request/control schema families and
current transports are recorded. `ServerArgs` covers the full serving runtime,
so irrelevant fields will be excluded only with evidence during each subsystem
port; no Engine API or startup owner is excluded from this migration.

### Phase 1: Rust crate, configuration, and runtime foundation

- [x] Add a dedicated Rust offline-engine crate with typed seed configuration,
  errors, an engine handle, and a runtime ownership boundary. Keep Rust domain
  types separate from wire types.
- [x] Add a process supervisor that starts a configured worker, waits for its
  explicit stdout readiness line, detects early exit and timeout, and kills and
  reaps the child on startup failure or shutdown.
- [x] Define lifecycle state and idempotent shutdown; test validation before
  launch, startup failure, readiness, timeout, early exit, and worker cleanup.
- [x] Document the Engine runtime ownership map, direct runtime-context reads,
  and process-topology decisions that must be preserved during the scheduler
  port.

The Rust configuration in this phase is intentionally the stable seed needed
to own and test a runtime process. The scheduler-specific `ServerArgs` fields,
ambient runtime-context projections, rank/device topology, and model loading
configuration move with their consuming scheduler/model subsystems in Phase 2;
duplicating them before those semantics have Rust owners would create a second,
temporary config contract.

**Exit criteria: complete.** Rust has a tested Engine lifecycle and a production
process supervisor contract that the Rust scheduler/model runtime can plug
into. Scheduler/model readiness and device cleanup are qualified in Phase 2.

### Phase 2: Rust scheduler and model execution

**Current decomposition finding:** `Scheduler` is a large orchestrator over
many existing `scheduler_components` (request receiver/retirer, batch result
processing, output streaming, DP attention, weight updates, profiling,
invariants, metrics, cache/KV events, and load publication). `ModelRunner`
owns model construction, GPU worker setup, memory/KV pools, attention backends,
graphs, forward execution, sampling, and weight loading. The Rust server's
`to_scheduler`/`from_scheduler` modules currently validate, queue, and forward
typed messages but ultimately hand scheduler requests across the Python
boundary; this is not the scheduler algorithm or model execution.

Port order must preserve a working vertical slice and avoid a second scheduler:

1. Define Rust scheduler request/state and model-executor contracts from the
   existing Python IPC and scheduler component interfaces.
2. Move one end-to-end CPU/control-plane slice (startup config, worker
   readiness, request admission, cancellation, output completion) to Rust and
   run it against a test model backend.
3. Add a real Rust model-executor backend and port model loading, forward,
   sampling, and KV/cache ownership behind that contract.
4. Port scheduling policy and advanced execution features in dependency order;
   each slice removes its Python owner only after parity tests pass.

The detailed component matrix and per-slice exit evidence are maintained below
as work proceeds. No Python scheduler or model execution is considered migrated
until Rust owns and executes it. The first Rust scheduler slice is
`sglang_engine::scheduler::order_waiting`, matching Python's FCFS,
priority-FCFS, HRRN, shortest-prefill-first, longest-output-first,
longest-prefix-match, routing-key, and DFS-weight ordering.
Tests cover priority direction/ties, missing priorities, HRRN aging and
zero-work requests, temporary prefix deprioritization, and output ordering.
This code is not yet wired into scheduler admission, so Python remains
authoritative until the Rust queue lifecycle is ported. The new
`sglang_engine::admission::AdmissionQueue` owns a testable pending/running state
machine with queue bounds, prefill-token admission budgets, duplicate-ID
rejection, completion, and cancellation signaling. It does not yet implement
preemption, KV/cache allocation, production scheduler IPC, or distributed rank
consensus. DFS-weight consumes an explicit prefix-tree topology and matches
Python's subtree-weight traversal; integration with the live KV tree remains
pending. Random ordering is seeded and
reproducible in Rust while Python uses its process RNG, so seed policy still
needs to be specified at the distributed scheduler boundary. The LPM ordering
step is ported, but KV-tree prefix matching is not. Routing-key ordering is
ported as a pure function of active-batch key counts; the Rust scheduler does
not yet own production batch-key collection.

`sglang_engine::model` now defines typed tokenized tasks, sampling inputs,
generation results, a batched `ModelExecutor` trait, and boundary validation
for unique IDs, complete result sets, output limits, and logprob alignment.
`sglang_engine::runtime::SchedulerRuntime` composes that contract with queue
admission and exercises load → submit → schedule → execute → validate → complete
and shutdown using a fake executor. Backend failures release active slots and
surface the error. This is scheduler orchestration only: no production model
loader or accelerator executor implements the trait yet, and production SRT
still executes entirely in Python.

`sglang_engine::sampling` provides a CPU reference for greedy and seeded
temperature/top-k/top-p sampling. Greedy selection preserves the first index
on ties; stochastic filtering retains the token crossing the nucleus threshold.
This is not yet calibrated against SGLang's FlashInfer/platform kernels and is
not wired into a production accelerator executor or Python's sampler.

- Port scheduler request admission, batching, scheduling policy, queueing,
  preemption, cancellation, token/KV accounting, cache interaction, and
  scheduler state transitions.
- Port model-runner orchestration and execution interfaces, including
  attention, sampling, logits/logprobs, embeddings, multimodal inputs, and
  model-family-specific execution hooks. Keep kernels behind typed backend
  interfaces and reuse existing Rust kernels/crates where available.
- Port detokenization and tokenizer/model-file loading needed by the offline
  path; reuse `sglang-processor` for rendering/tokenization where its behavior
  matches Python, and maintain parity fixtures for differences.
- Replace Python multiprocessing/Pipes/ZMQ ownership with Rust process or task
  supervision and versioned typed IPC only where process isolation is needed.
  Preserve backpressure, ordering, shared-memory ownership, and failure
  propagation.
- Port distributed execution and rank coordination. Validate recovery and
  shutdown across worker failures, startup timeouts, signals, and partial rank
  failure.

**Exit criteria:** Rust starts a supported model and performs deterministic
single-request and concurrent inference without Python scheduler/model
execution.

### Phase 3: Offline Engine API parity

- Implement synchronous and asynchronous generation, batching, streaming,
  abort/cancellation, prompt/input-ID inputs, sampling parameters, logprobs,
  hidden states, multimodal requests, and request metadata.
- Implement encode, rerank, and score semantics with the same validation and
  output contracts as Python.
- Implement sessions, cache flush, profiling, expert-distribution recording,
  server/model information, RPC, and memory occupation controls.
- Implement weight-update groups, tensor/disk/distributed/IPC updates,
  weight reads, LoRA load/unload (including tensor-based and async forms), and
  model-save operations.
- Match Engine lifecycle details: attach/worker behavior if retained,
  context-manager behavior, child-process reporting, shutdown ordering,
  idempotency, timeout handling, and error reporting.
- Keep API ergonomics idiomatic for Rust while preserving behavior; expose
  Python-compatible semantics at the Python adapter boundary rather than
  forcing Python-specific conventions into Rust types.

**Exit criteria:** every row in the feature matrix has a Rust implementation
and a passing contract/parity test, or an explicitly approved exclusion.

### Phase 4: Python compatibility adapter and migration

- Keep `sglang.Engine` import paths and signatures as a thin adapter during
  transition. Convert Python values at the boundary and delegate engine work
  to Rust; retain Python-only conveniences only when they do not duplicate
  runtime behavior.
- Preserve subclassing/hooks only if they are part of the supported public
  contract; otherwise identify migration replacements and deprecation timing.
- Add warnings and release notes only for actual incompatibilities. Keep Python
  as a selectable backend during rollout only if it helps parity testing and
  has a defined removal policy.
- Remove the old Python Engine launch/request implementation only after the
  Rust implementation passes the parity gate and the compatibility policy is
  documented.

**Exit criteria:** compatibility callers use Rust-backed behavior and there is
one authoritative Engine implementation.

### Phase 5: Qualification, rollout, and removal

- Run the full matrix across supported model families, accelerators, topology,
  precisions, quantization, multimodal modalities, and feature combinations.
- Compare Rust and Python outputs, token IDs, logprobs, stream frames, errors,
  cancellation, and side effects on equivalent fixtures and hardware runs.
- Exercise fault cases: invalid configuration, model load failure, worker
  crash, startup timeout, queue saturation, client cancellation, distributed
  rank loss, repeated shutdown, and resource cleanup.
- Benchmark startup time, throughput, latency, memory use, and concurrency
  against Python Engine; investigate material regressions before switching the
  default.
- Roll out behind a documented backend selection mechanism, monitor failures
  and regressions, then make Rust the default and remove legacy Python runtime
  code according to the compatibility policy.

**Exit criteria:** supported configurations pass correctness and reliability
gates, performance is qualified, and fallback/removal policy is complete.

## Test and coverage requirements

- **Unit tests:** config normalization, request conversion, sampling/default
  precedence, validation, serialization, state transitions, resource ownership,
  and each public operation's result/error mapping.
- **Runtime integration tests:** startup/readiness/shutdown, single and batched
  inference, streaming, cancellation, concurrency/backpressure, model loading,
  sessions, controls, and failure cleanup.
- **Cross-language parity tests:** compare the Rust API and Python Engine on
  shared fixtures for request acceptance, tokenization/rendering, token IDs,
  output structure, stream ordering, logprobs, errors, and control effects.
- **Distributed/hardware tests:** supported TP/PP/DP/EP and multi-node layouts,
  device mapping, worker failures, and accelerator-specific model execution.
- **Coverage accounting:** every public method and feature-matrix row links to
  at least one test; every conditional runtime path has success and failure
  coverage where feasible. Hardware-gated cases must be registered in the
  appropriate CI tier, not silently omitted.
- **Documentation:** Rust API guide, Python adapter guide, configuration
  reference, feature/support matrix, migration notes, and examples for sync,
  async, streaming, multimodal, and lifecycle usage.

## Working rules

- Keep this feature matrix and phase exit status updated in the same changes
  that add or remove Engine behavior.
- Do not call the migration complete while Python starts or runs scheduler/model
  execution, even if Python API calls already delegate to Rust.
- Prefer small vertical slices that include implementation, parity tests, and
  docs. Avoid porting a broad subsystem without preserving its observable
  contract.
