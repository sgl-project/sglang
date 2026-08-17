# Rust SGLang Server Manual Review Plan

This plan covers every file under `rust/sglang-server/`: 49 files and 20,163
physical lines. The review order moves from public abstractions, interfaces,
launcher code, and wire protocols toward specialized paths and leaf utilities.

Later phases are less architecturally central, not necessarily lower risk.
For example, `utils/regex.rs` is a leaf utility but requires a careful security
review.

## Review conventions

Use this checklist for every file before marking its file-level checkbox complete:

- [ ] Read production code and module documentation.
- [ ] Read all embedded `#[cfg(test)]` tests.
- [ ] Identify callers, callees, channel producers/consumers, and ownership.
- [ ] Record public and wire-level invariants.
- [ ] Check normal, malformed-input, overload, cancellation, and shutdown paths.
- [ ] Check compatibility with the corresponding Python implementation where applicable.
- [ ] Record correctness, compatibility, concurrency, security, performance, and test-gap findings.
- [ ] Add or identify a regression test for every confirmed issue.

For each finding, record:

- File and line number
- Severity and category
- Trigger or failing input
- Expected and actual behavior
- Cross-language compatibility impact
- Suggested fix and regression test

## Phase 1: Public interface, abstractions, and launcher

**Scope:** 8 files, 1,661 LOC.

**Objective:** Understand how Python starts the Rust frontend, how stages are
spawned, and where data crosses the Python/Rust scheduler boundary.

- [ ] **01. `rust/sglang-server/src/lib.rs` — 314 LOC**
  - PyO3 module entry point, `Server`, `IngressBatch`, and `MmHandoff`.
  - Review GIL-held versus detached paths, zero-copy claims, backpressure,
    scheduler-facing methods, MM ownership transfer, logging initialization,
    and shutdown behavior.
- [ ] **02. `rust/sglang-server/src/runtime/runnable.rs` — 10 LOC**
  - Common `Runnable` abstraction for pipeline stages.
  - Verify the lifetime, ownership, blocking, and shutdown expectations imposed
    on every implementation.
- [ ] **03. `rust/sglang-server/src/runtime/config.rs` — 300 LOC**
  - Rust boot configuration and typed view of Python `server_args`.
  - Check JSON field parity, defaults, mandatory validation, worker counts,
    address formatting, model metadata, and disaggregation configuration.
- [ ] **04. `rust/sglang-server/src/runtime/threads.rs` — 142 LOC**
  - CPU-core partitioning, affinity, and stage/pool spawning.
  - Review insufficient-core behavior, indexing, worker counts, thread naming,
    affinity failures, and future extensibility.
- [ ] **05. `rust/sglang-server/src/runtime.rs` — 559 LOC**
  - Main runtime launcher and owner of rings, channels, workers, and shutdown.
  - Trace every channel edge, partial-start failure, tokenizer skip mode, thread
    ownership, shutdown notification, bounded join, and resource cleanup.
- [ ] **06. `rust/sglang-server/src/ring.rs` — 251 LOC**
  - Rust/Python ingress and egress queues and columnar ingress batches.
  - Check ordering, capacity, stash behavior, blocking versus nonblocking paths,
    closure handling, byte accounting, and backpressure.
- [ ] **07. `rust/sglang-server/Cargo.toml` — 69 LOC**
  - Crate type, Python-extension metadata, dependencies, and version pins.
  - Review compatibility-sensitive pins, dependency features, build outputs,
    and security/maintenance implications.
- [ ] **08. `rust/sglang-server/pyproject.toml` — 16 LOC**
  - Maturin build configuration and Python module name.
  - Confirm agreement with `Cargo.toml`, Python imports, supported Python
    versions, and packaging behavior.

### Phase 1 completion

- [ ] Draw the complete thread and channel topology.
- [ ] Document which operations may block and whether they hold the GIL.
- [ ] Document startup ownership and the exact shutdown/drop sequence.
- [ ] Verify that every partially constructed runtime resource is recoverable.

## Phase 2: Internal and scheduler wire protocols

**Scope:** 9 files, 4,665 LOC.

**Objective:** Establish exact request and response contracts before reviewing
their producers and consumers.

- [ ] **09. `rust/sglang-server/src/message.rs` — 96 LOC**
  - Message-module facade plus `Request`, `IngressMsg`, and `DetokMsg`.
  - Map every variant to its producers, consumers, ownership, and terminal path.
- [ ] **10. `rust/sglang-server/src/message/types.rs` — 271 LOC**
  - Shared token types, `OneOrMany`, sealed type allowlist, and wire macros.
  - Review untagged deserialization ambiguity, macro-generated field order,
    message tags, and base-request fields.
- [ ] **11. `rust/sglang-server/src/message/io_struct.rs` — 247 LOC**
  - Positional msgpack structures sent to the Python scheduler.
  - Compare every field and index with
    `python/sglang/srt/managers/io_struct.py`, especially nil fillers and the
    disaggregation block.
- [ ] **12. `rust/sglang-server/src/message/sampling.rs` — 998 LOC**
  - Rust port of Python `SamplingParams` normalization and validation.
  - Compare all fields, defaults, ranges, normalization order, stop handling,
    regex admission, and serialized layout with Python.
- [ ] **13. `rust/sglang-server/src/message/request.rs` — 1,312 LOC**
  - Native request body, batch fan-out, request variants, MM fields, and ingress
    serialization.
  - Check scalar/list broadcasting, batch-size limits, clone budgets, memory
    estimates, ambiguous shapes, RID consistency, and control messages.
- [ ] **14. `rust/sglang-server/src/message/egress.rs` — 1,195 LOC**
  - Scheduler egress frame encoding/decoding and `ChunkEvent` reconstruction.
  - Audit every byte offset and length, ragged columns, optional data, malformed
    frames, RID recovery, numeric conversion, and partial batch behavior.
- [ ] **15. `rust/sglang-server/src/message/finish_reason.rs` — 229 LOC**
  - Python-compatible terminal finish reasons and matched-stop representation.
  - Verify all variants, aliases, malformed values, abort details, and API
    mappings.
- [ ] **16. `rust/sglang-server/src/ids.rs` — 265 LOC**
  - Request identifiers, client-ID uniquification, health IDs, and shard choice.
  - Check collision behavior, maximum lengths, suffix parsing, equality/hash
    agreement, display values, and shard stability.
- [ ] **17. `rust/sglang-server/src/error.rs` — 52 LOC**
  - Shared error taxonomy.
  - Check whether each error retains enough context and is converted to the
    appropriate client, codec, or internal failure.

### Phase 2 completion

- [ ] Produce a field-by-field ingress protocol sheet.
- [ ] Produce a field-by-field egress header/data-column protocol sheet.
- [ ] Compare positional structures against their Python declarations.
- [ ] Enumerate malformed frames that must fail safely without panicking.

## Phase 3: Request lifecycle and pipeline stages

**Scope:** 6 files, 3,169 LOC.

**Objective:** Trace requests through validation, tokenization, scheduling,
detokenization, completion, failure, and cancellation.

- [ ] **18. `rust/sglang-server/src/fsm.rs` — 202 LOC**
  - Request lifecycle states, events, and transitions.
  - Verify every legal edge, illegal transition, terminal state, and guarantee
    that all ingress branches pass through pre-send validation.
- [ ] **19. `rust/sglang-server/src/tokenizer_manager.rs` — 100 LOC**
  - TM events, senders, shutdown-aware receive, and abort-source interface.
  - Review channel roles, abort-lane guarantees, shard selection, and closure
    behavior.
- [ ] **20. `rust/sglang-server/src/tokenizer_manager/ingress.rs` — 1,611 LOC**
  - Central ingress FSM driver and validation/dispatch stage.
  - Audit input validation, normalization, token limits, tokenizer/MM routing,
    detokenizer registration, ring submission, aborts, failures, and sidecar
    cleanup.
- [ ] **21. `rust/sglang-server/src/tokenizer.rs` — 351 LOC**
  - Tokenizer interface, backend loading, model-file resolution, and worker pool.
  - Check local/cache resolution, revision handling, automatic special tokens,
    existing token IDs, backend errors, and skip-tokenizer mode.
- [ ] **22. `rust/sglang-server/src/tokenizer_manager/egress.rs` — 217 LOC**
  - Egress-ring dispatcher and detokenizer-shard router.
  - Review frame tags, corrupt-frame behavior, per-RID routing, blocking sends,
    heartbeat updates, and shutdown.
- [ ] **23. `rust/sglang-server/src/detokenizer.rs` — 688 LOC**
  - Stateful detokenizer shards and per-RID state.
  - Check registration/deregistration, hash collisions, incremental Unicode and
    byte-fallback decoding, stop trimming, logprob text, terminal delivery,
    failures, and abort races.

### Phase 3 completion

- [ ] Trace a text-prompt request end to end.
- [ ] Trace a token-ID request end to end.
- [ ] Trace batch, control, multimodal, and skip-tokenizer requests.
- [ ] Trace validation failure, tokenizer failure, scheduler failure, client
  disconnect, and process shutdown.
- [ ] Confirm that every registered RID has exactly one cleanup path.

## Phase 4: Core HTTP server and native API

**Scope:** 8 files, 2,165 LOC.

**Objective:** Review HTTP admission, the native generation protocol, streaming,
health checks, control endpoints, and disconnect handling.

- [ ] **24. `rust/sglang-server/src/api_server.rs` — 109 LOC**
  - Axum application composition and shared state.
  - Review route mounting, body limits, listener adoption, formatter loading,
    runtime shutdown, and the documented missing API-key boundary.
- [ ] **25. `rust/sglang-server/src/api_server/submit.rs` — 60 LOC**
  - Shared request-submission interface.
  - Check RID creation, sink capacity, request construction, TM send failure,
    and returned receiver ownership.
- [ ] **26. `rust/sglang-server/src/api_server/guard.rs` — 130 LOC**
  - Abort-on-disconnect RAII guard.
  - Audit arm/disarm behavior, partial batches, duplicate drops, channel failure,
    and exactly-once abort intent.
- [ ] **27. `rust/sglang-server/src/api_server/native_api.rs` — 623 LOC**
  - `/generate`, batch generation, unary/SSE responses, and health probes.
  - Compare request validation, status codes, error timing, cumulative output,
    index ordering, timeouts, and health semantics with the Python server.
- [ ] **28. `rust/sglang-server/src/api_server/frame.rs` — 875 LOC**
  - Native response shaping and cumulative output accumulator.
  - Check JSON escaping, logprob tuples, ragged arrays, hidden states, incremental
    versus cumulative data, finish reasons, error frames, and allocation growth.
- [ ] **29. `rust/sglang-server/src/api_server/common.rs` — 228 LOC**
  - `/server_info`, `/get_model_info`, aliases, and control-request handling.
  - Review internal-state allowlisting, response shaping, codec errors, timeouts,
    and consistency with static `server_args`.
- [ ] **30. `rust/sglang-server/src/utils/response.rs` — 99 LOC**
  - Shared JSON and SSE error-response mechanics.
  - Verify HTTP status versus in-stream error semantics, content types, SSE
    framing, and endpoint-owned error payloads.
- [ ] **31. `rust/sglang-server/src/api_server/log.rs` — 41 LOC**
  - HTTP access-log middleware.
  - Check configuration parity, peer-address handling, latency/status logging,
    and zero-cost behavior when disabled.

### Phase 4 completion

- [ ] Build a native endpoint matrix: request, success, error, streaming,
  cancellation, and timeout behavior.
- [ ] Compare native endpoint wire output with the Python HTTP server.
- [ ] Verify abort behavior for dropped unary handlers and SSE streams.
- [ ] Record the security impact and intended resolution of missing API-key auth.

## Phase 5: OpenAI-compatible API

**Scope:** 8 files, 5,573 LOC.

**Objective:** Verify OpenAI schema compatibility and consistency with the shared
native backend.

- [ ] **32. `rust/sglang-server/src/api_server/openai.rs` — 215 LOC**
  - OpenAI router, shared errors, submission, collection, and indexed streams.
  - Review route coverage, model checks, choice limits, media detection, stream
    failures, and OpenAI error shapes.
- [ ] **33. `rust/sglang-server/src/api_server/openai/chat.rs` — 1,188 LOC**
  - Chat Completions preparation and unary/stream response generation.
  - Audit template application, sampling defaults, merged stops, multi-choice
    behavior, usage, logprobs, finish reasons, tool calls, reasoning, and SSE.
- [ ] **34. `rust/sglang-server/src/api_server/openai/completions.rs` — 883 LOC**
  - Legacy text Completions endpoint.
  - Review all prompt forms, batching and choice indexing, token prompts, echo,
    sampling conversion, logprobs, usage, unary results, and stream ordering.
- [ ] **35. `rust/sglang-server/src/api_server/openai/template.rs` — 1,868 LOC**
  - Hugging Face/Jinja and legacy conversation-template resolution/rendering.
  - Compare every supported style with Python, including separators, roles,
    system messages, assistant prefixes, local files, built-ins, inference from
    model type, malformed configuration, and template selection precedence.
- [ ] **36. `rust/sglang-server/src/api_server/openai/tools.rs` — 836 LOC**
  - Tool-choice constraints and completed tool-call parsing.
  - Check parser-name mapping, tool/tool-choice agreement, JSON schemas,
    structural tags, strict tools, streaming jail behavior, IDs, and finish
    reasons.
- [ ] **37. `rust/sglang-server/src/api_server/openai/reasoning.rs` — 216 LOC**
  - Unary and incremental reasoning-content splitting.
  - Review lazy parser construction, marker splits across chunks, content and
    reasoning buffers, terminal flushing, and unknown parsers.
- [ ] **38. `rust/sglang-server/src/api_server/openai/models.rs` — 48 LOC**
  - Model listing and retrieval endpoints.
  - Verify model-card fields, served-name matching, and unknown-model errors.
- [ ] **39. `rust/sglang-server/src/api_server/openai/test_utils.rs` — 319 LOC**
  - Shared router fixtures and handler-level OpenAI tests.
  - Review fixture realism and coverage of validation-before-submit, closed
    channels, stream failures, error formats, and route availability.

### Phase 5 completion

- [ ] Build an OpenAI compatibility matrix for chat and completions.
- [ ] Cover unary/streaming, one/many choices, text/token prompts, usage, echo,
  logprobs, tools, reasoning, templates, and malformed requests.
- [ ] Compare supported and intentionally unsupported API fields with Python.
- [ ] Verify every streaming path emits one correct terminal condition.

## Phase 6: Specialized multimodal and disaggregation paths

**Scope:** 5 files, 1,447 LOC.

**Objective:** Review optional but critical behavior used by multimodal and
prefill/decode deployments.

- [ ] **40. `rust/sglang-server/src/mm.rs` — 413 LOC**
  - Multimodal workers, POSIX shared-memory segments, sidecar, and Python handoff.
  - Audit allocation, write bounds, names, ownership/unlink transfer, hashes,
    offsets, inline versus SHM features, stale entries, failure cleanup, and
    worker shutdown.
- [ ] **41. `rust/sglang-server/src/message/mm_payload.rs` — 222 LOC**
  - Conversion of request MM values into typed `sglang-mm` inputs.
  - Check modality rejection, image counts, source detection, prefetched-data
    alignment, malformed values, unsupported features, and error clarity.
- [ ] **42. `rust/sglang-server/src/api_server/prefetch.rs` — 199 LOC**
  - Concurrent URL/file prefetching before MM preprocessing.
  - Review byte budgets, global permits, source ordering, cancellation, timeouts,
    file/network behavior, partial failures, and memory retention.
- [ ] **43. `rust/sglang-server/src/api_server/disaggregation.rs` — 1 LOC**
  - Disaggregation module declaration.
  - Confirm visibility, naming, and expected module placement.
- [ ] **44. `rust/sglang-server/src/api_server/disaggregation/bootstrap.rs` — 612 LOC**
  - PD KV bootstrap registry, topology, route lookup, DP rooms, and cleanup.
  - Compare the complete Python wire protocol; check topology consistency,
    readiness counts, duplicate registration, concurrent copy-on-write updates,
    missing ranks, sentinel queries, integer coercion, room sharding/expiry, and
    cleanup scheduling.

### Phase 6 completion

- [ ] Trace MM input from HTTP body through prefetch, preprocessing, scheduler
  ingress, sidecar retrieval, Python materialization, and SHM unlink.
- [ ] Enumerate cleanup behavior for every MM failure or cancellation point.
- [ ] Trace prefill rank registration, topology lookup, and DP-room lookup.
- [ ] Compare bootstrap success and error responses with Python.

## Phase 7: Leaf utilities

**Scope:** 5 files, 1,483 LOC.

**Objective:** Finish with isolated helpers. Give regex admission a dedicated
security review despite its leaf position.

- [ ] **45. `rust/sglang-server/src/utils/regex.rs` — 1,223 LOC**
  - `stop_regex` Python compatibility and complexity admission.
  - Audit Python/Rust grammar differences, escapes, flags, assertions,
    repetitions, ambiguity, nested alternation, length calculations, integer
    overflow, AST nesting, ReDoS resistance, output bounds, and cache eviction.
- [ ] **46. `rust/sglang-server/src/utils/serialize.rs` — 118 LOC**
  - Python-`int(...)`-compatible deserialization.
  - Check signs, whitespace, empty strings, overflow, nulls, vectors, optional
    fields, nonnumeric values, and error reporting.
- [ ] **47. `rust/sglang-server/src/utils/sock.rs` — 36 LOC**
  - TCP-listener construction and socket options.
  - Review IPv4/IPv6 behavior, backlog, buffer sizes, address reuse, bind/listen
    ordering, nonblocking mode, and error propagation.
- [ ] **48. `rust/sglang-server/src/environ.rs` — 100 LOC**
  - Environment-variable parsing with Python `EnvField` semantics.
  - Confirm accepted boolean spellings, whitespace/case behavior, invalid-value
    warnings, numeric ranges, unset variables, and defaults.
- [ ] **49. `rust/sglang-server/src/utils.rs` — 6 LOC**
  - Utility-module facade.
  - Verify exports remain minimal and utilities do not acquire dependencies on
    higher-level pipeline stages.

### Phase 7 completion

- [ ] Create a regex compatibility and adversarial-input test matrix.
- [ ] Confirm utilities fail without panics or unbounded resource consumption.
- [ ] Confirm utility behavior is covered at boundary values.

## Final integration review

- [ ] Re-read all cross-file findings after completing every phase.
- [ ] Confirm every channel sender has a receiver and a closure/shutdown story.
- [ ] Confirm every RID registration and sidecar entry has a cleanup story.
- [ ] Confirm no malformed external input can panic a worker or API thread.
- [ ] Confirm Rust/Python positional protocols are checked field by field.
- [ ] Confirm native and OpenAI streaming paths handle disconnect and terminal
  errors consistently.
- [ ] Confirm all unsafe or OS-resource ownership assumptions are documented and
  tested, especially POSIX shared memory and socket setup.
- [ ] Run `cargo fmt --check`.
- [ ] Run `cargo clippy -p sglang-server --all-targets`.
- [ ] Run `cargo test -p sglang-server`.
- [ ] Run targeted Python/Rust compatibility or end-to-end tests for changed or
  questioned behavior.
- [ ] Triage all findings and assign owners before declaring the review complete.
