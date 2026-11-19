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

**Supported scope.** Enable with `SGLANG_ENABLE_CANCEL_RETIREMENT_RECEIPTS=1`. This covers unified single-rank text serving and the Decode worker's receive/drain queues in single-rank text P/D with Mooncake device KV and deferred release. Prefill retirement, other Decode stages, host or staged KV transfer, multimodal requests, and multi-rank scheduler receipts are outside this scope. Existing cache and execution restrictions still apply.

**Semantics.** Capture exactly one original internal rid under the public ID before cancel. Retries reuse that capture. Ambiguous duplicate public IDs and `abort_all` are refused for the wait mode. A wait timeout returns `success=true`, `retirement_confirmed=false`. Retry the same `cancellation_id` within the five-minute retention window (capacity 256). Unknown or lost receipts return `FailedPrecondition` / `Unavailable`. A positive Decode receipt requires a nonempty set of matching transfer-drain ACKs and completed local KV, metadata and receiver release. The Decode receipt covers the Decode worker's ownership, independent of Prefill request retirement. A Generate terminal outcome, channel closure, transfer release deadline, or load counter cannot establish retirement.

**Dependency.** [PR #50617](https://github.com/sgl-project/sglang/pull/50617) (replacing #48624) provides the Python HTTP and scheduler receipt implementation reused here for Rust text control. The integration includes its `44d5424d37` core, including failed-delivery retry, MLX rejection and retraction-abort retirement fences; the scheduler core should land first. Rust HTTP waiting remains unsupported; native `api.v1` uses the typed Rust integration below.

**Linux CPU checks.** From `rust/`:
`cargo test -p sglang-api-types`,
`cargo test -p sglang-server api_server::handlers::retirement::tests`,
`cargo test -p sglang-server api_server::grpc::service::tests::abort_`, and
`cargo test -p sglang-server tokenizer_manager::to_scheduler::tests`.
From the repository root, run the scheduler and transport regressions:

```bash
python -m pytest -q \
  test/registered/unit/managers/test_scheduler_cancellation_retirement.py \
  test/registered/unit/disaggregation/test_deferred_decode_kv_release.py
```

**Validation plan.**

1. First-token cancel ordering: receipt only after scheduler retirement (or Intake-local nonadmission).
2. Timeout and disconnect: bounded wait expires or client disconnects without implying retirement; shared operation retains the result.
3. Public RID reuse: later request with the same public ID does not satisfy the prior instance’s receipt.
4. Unsupported topology: unknown or unsupported scope fails closed.
5. Normal generation positive control: completed generate still succeeds without opt-in retirement. Legacy Abort still reports acceptance only.
6. Decode transport drain: run single-rank Prefill and Decode workers with the same text model, Mooncake, device KV and deferred release. Use native gRPC on Decode. Send the same P/D bootstrap room to both workers. Hold the Prefill transfer or its ACK briefly in the test environment and call `Abort` on Decode while the room is in the receive/drain queue. `wait_for_retirement_ms=0` must report an unconfirmed receipt while the transfer is held. Resume the transfer worker and poll with the same cancellation ID. The positive receipt must follow the ACKs and local release. Repeat with a fresh room and withhold the ACK through the deferred-release deadline. The receipt must remain unconfirmed or end with `FAILED_PRECONDITION`. Run a normal P/D request afterwards to check recovery.

## api.v1 engine instance identity

**Contract.** `GetServerInfoResponse.instance_id` (proto field 16) is an opaque string owning-runtime boot identity. The embedded Rust serving runtime mints a nonempty random UUIDv4 once when constructing `CoreHandle` / `CoreInner`, stores it on `CoreInner`, and exposes it through typed `ServerInfo` and the gRPC info mapper. HTTP `/server_info` carries the same typed field; typed fields override stale `launch_fields` snapshots, including a stale `launch_fields.instance_id`.

**Lifecycle.** The id is stable for one owning runtime across repeated reads, `CoreHandle` clones, and HTTP/gRPC adapters that share that handle. Reconnecting transports or restarting an out-of-process sidecar must not change it. A fresh owning runtime, even with the same host/port/model/version, mints a new value. It is never derived from version, topology, PID, or startup duration, and never minted per RPC. Do not reuse `PortArgs.instance_id` (shared IPC/load snapshot namespace across DP scheduler children; may be reused on worker recovery).

**Distributed scope.** Scope is the embedded Rust runtime that serves this listener and owns the scheduler process. Multiple protocol listeners sharing one `CoreHandle` share the id. Independent DP/node owning runtimes each have their own id. This is not a cluster-wide epoch and does not prove that every TP follower restarted. Server info still uses its normal scheduler round-trip; this field alone is not a health check or request-cleanup proof. An empty field from an older peer means identity is unsupported/unknown; clients must not replace it with static model metadata.

**Compatibility.** Older peers omit field 16; current decoders treat that as an empty id. Legacy readers ignore the new field and keep their `version`. `sglang-api-types` is bumped to 0.2.0 for the schema change.

**Validation.** Deterministic fake-transport coverage lives in the existing Rust unit tests (no GPU/model required):

```bash
cargo test -p sglang-server --lib server_info_instance_id
cargo test -p sglang-api-types get_server_info_instance_id_is_wire_compatible
cargo test -p sglang-api-types --test json_parity
```

These tests are written but have not been executed. Only code generation and
static checks have run; compilation, CPU execution, live engine restart and GPU
validation remain pending. Tracking: https://github.com/sgl-project/sglang/issues/38075.
