# sglang-server

`sglang-server` is SGLang's Rust HTTP frontend and request-processing pipeline. It exchanges typed requests and responses with the Python scheduler while keeping latency-sensitive work outside Python.

## Code review principles

1. **Use strongly typed boundaries.** Model every supported protocol shape with structs, enums, and validated newtypes; avoid opaque values such as `serde_json::Value` and `rmpv::Value` in production paths.
2. **Keep one canonical schema.** Rust and Python must derive their wire contracts from one source of truth, aligned with `io_struct.py`, instead of independently duplicating field names, order, defaults, or validation.
3. **Make protocol declarations minimal and declarative.** A reviewer should be able to understand the wire format from its type declarations alone, without tracing fillers, conversion code, macros, or repeated field lists.
4. **Use one representation per semantic stage.** Separate external input, normalized domain data, and wire data, and convert between them once at explicit boundaries; do not keep multiple overlapping representations of the same state.
5. **Design compatibility and safety explicitly.** Version protocols, reject unsupported or malformed inputs clearly, validate lengths and resource bounds before allocation, preserve invariants in types, and test compatibility across the real Rust and Python codecs.

## Draft proposal: api.v1 engine instance identity

**Problem.** api.v1 metadata cannot tell a transport reconnect from an engine that restarted at the same address.

**Proposal.** Add an engine-owned opaque `instance_id` to api.v1 `GetServerInfo`. It stays stable across repeated reads, transport reconnects, and sidecar restarts, and changes only for a new owning-engine lifetime. Do not use version, topology, PID, startup duration, or a sidecar UUID as identity.

**Scope / non-goals.** Engine ownership across multiple listeners and DP ranks still needs review. No cluster-wide identity or independent-scheduler-restart guarantee is assumed. This is an initial design seed only; no proto or runtime changes yet. Tracking: https://github.com/sgl-project/sglang/issues/38075.

**Validation plan.** Same running engine → stable id; new engine at the same address → id changes; old clients remain compatible. This does not establish request retirement or device idleness.
