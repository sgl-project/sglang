# sglang-server

`sglang-server` is SGLang's Rust HTTP frontend and request-processing pipeline. It exchanges typed requests and responses with the Python scheduler while keeping latency-sensitive work outside Python.

## Code review principles

1. **Use strongly typed boundaries.** Model every supported protocol shape with structs, enums, and validated newtypes; avoid opaque values such as `serde_json::Value` and `rmpv::Value` in production paths.
2. **Keep one canonical schema.** Rust and Python must derive their wire contracts from one source of truth, aligned with `io_struct.py`, instead of independently duplicating field names, order, defaults, or validation.
3. **Make protocol declarations minimal and declarative.** A reviewer should be able to understand the wire format from its type declarations alone, without tracing fillers, conversion code, macros, or repeated field lists.
4. **Use one representation per semantic stage.** Separate external input, normalized domain data, and wire data, and convert between them once at explicit boundaries; do not keep multiple overlapping representations of the same state.
5. **Design compatibility and safety explicitly.** Version protocols, reject unsupported or malformed inputs clearly, validate lengths and resource bounds before allocation, preserve invariants in types, and test compatibility across the real Rust and Python codecs.

## api.v1 cancellation retirement

Opt-in `wait_for_retirement_ms` (0..30000) with a nonempty `cancellation_id` (≤128 UTF-8 bytes) waits for exact-request scheduler ownership retirement of one public RID. Absent the wait field, Abort stays the legacy accepted-only cancel (`success` means cancellation was requested; `retirement_confirmed` is always false).

**Supported scope.** Single-rank text serving when `SGLANG_ENABLE_CANCEL_RETIREMENT_RECEIPTS=1`. Multimodal, PD disaggregation, and distributed DP/peer fan-out return `UNIMPLEMENTED` without claiming retirement. Scheduler receipts still apply the parent retirement predicates (cache ownership, speculative, inflight, resources). Feature-disabled paths do not provide scheduler receipts.

**Semantics.** Capture exactly one original internal rid under the public ID before cancel; retries reuse that capture and must not recapture by public RID. Ambiguous duplicate public IDs and `abort_all` are refused for the wait mode. A wait timeout returns `success=true`, `retirement_confirmed=false`; retry the same `cancellation_id` within a bounded five-minute retention window (capacity 256). After retention expiry the server does not claim idempotency or old-target safety. Unknown or lost receipts return `FailedPrecondition` / `Unavailable`. A Generate `CANCELLED` outcome, channel closure, or load counter is not retirement. Positive confirmation covers scheduler ownership (or Intake-local proof the request was never admitted), not native/MM worker completion, PD drain, or CUDA-wide idle.

**Prior art.** [PR #48624](https://github.com/sgl-project/sglang/pull/48624) provides the Python HTTP scheduler receipt implementation reused here for Rust text control.

This change is stacked on that unmerged PR. Its opt-in feature flag must be set
before engine startup. The first supported scheduler scope is unified,
single-rank text serving without speculative decoding and with a supported
static full-attention cache. The scheduler rejects additional ownership modes
it cannot prove retired; the flag does not enable those modes.

**CPU checks to run (not executed during this implementation).** From `rust/`:
`cargo test -p sglang-api-types`,
`cargo test -p sglang-server api_server::handlers::retirement::tests`,
`cargo test -p sglang-server api_server::grpc::service::tests::abort_`, and
`cargo test -p sglang-server tokenizer_manager::to_scheduler::tests`.
Also run the Python scheduler retirement, Rust admission, and API parity suites.
Live-engine cancellation ordering and normal-generation controls remain pending.

**Validation plan.**

1. First-token cancel ordering: receipt only after scheduler retirement (or Intake-local nonadmission).
2. Timeout and disconnect: bounded wait expires or client disconnects without implying retirement; shared operation retains the result.
3. Public RID reuse: later request with the same public ID does not satisfy the prior instance’s receipt.
4. Unsupported topology: unknown or unsupported scope fails closed.
5. Normal generation positive control: completed generate still succeeds without opt-in retirement; legacy Abort remains 499.
