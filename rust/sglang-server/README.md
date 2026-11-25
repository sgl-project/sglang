# sglang-server

`sglang-server` is SGLang's Rust HTTP frontend and request-processing pipeline. It exchanges typed requests and responses with the Python scheduler while keeping latency-sensitive work outside Python.

## Code review principles

1. **Use strongly typed boundaries.** Model every supported protocol shape with structs, enums, and validated newtypes; avoid opaque values such as `serde_json::Value` and `rmpv::Value` in production paths.
2. **Keep one canonical schema.** Rust and Python must derive their wire contracts from one source of truth, aligned with `io_struct.py`, instead of independently duplicating field names, order, defaults, or validation.
3. **Make protocol declarations minimal and declarative.** A reviewer should be able to understand the wire format from its type declarations alone, without tracing fillers, conversion code, macros, or repeated field lists.
4. **Use one representation per semantic stage.** Separate external input, normalized domain data, and wire data, and convert between them once at explicit boundaries; do not keep multiple overlapping representations of the same state.
5. **Design compatibility and safety explicitly.** Version protocols, reject unsupported or malformed inputs clearly, validate lengths and resource bounds before allocation, preserve invariants in types, and test compatibility across the real Rust and Python codecs.

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
