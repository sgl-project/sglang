# sglang-server

`sglang-server` is SGLang's Rust HTTP frontend and request-processing pipeline. It exchanges typed requests and responses with the Python scheduler while keeping latency-sensitive work outside Python.

## Code review principles

1. **Use strongly typed boundaries.** Model every supported protocol shape with structs, enums, and validated newtypes; avoid opaque values such as `serde_json::Value` and `rmpv::Value` in production paths.
2. **Keep one canonical schema.** Rust and Python must derive their wire contracts from one source of truth, aligned with `io_struct.py`, instead of independently duplicating field names, order, defaults, or validation.
3. **Make protocol declarations minimal and declarative.** A reviewer should be able to understand the wire format from its type declarations alone, without tracing fillers, conversion code, macros, or repeated field lists.
4. **Use one representation per semantic stage.** Separate external input, normalized domain data, and wire data, and convert between them once at explicit boundaries; do not keep multiple overlapping representations of the same state.
5. **Design compatibility and safety explicitly.** Version protocols, reject unsupported or malformed inputs clearly, validate lengths and resource bounds before allocation, preserve invariants in types, and test compatibility across the real Rust and Python codecs.

## Draft proposal: api.v1 cancellation retirement

This section documents the first DRAFT scaffold only. Protocol field shape is
pending reviewer feedback. A real request-instance capture/replay registry and
retirement backend are NOT implemented in this slice: opt-in
`wait_for_retirement_ms` is rejected with `UNIMPLEMENTED` after validation
(no cancel), and `retirement_confirmed` stays false on the legacy path. An
operation key (`cancellation_id`) is required with the wait field for future
retry safety so retries cannot recapture by public RID.

**Today / first scaffold.** Absent `wait_for_retirement_ms` preserves legacy
accepted-only Abort (`success` means cancellation was requested). Present
opts into exact-request bounded wait (including zero poll; cap 30000 ms) and
requires `cancellation_id`, but this slice does not wait, confirm retirement,
or replay captured instances. A Generate `CANCELLED` outcome or other final
output is still not scheduler retirement. Clients MUST require explicit
`retirement_confirmed=true` and MUST NOT infer completion from `success`; old
servers ignore the new request fields.

**Proposal (future).** Add a real opt-in, exact-request, bounded-wait
retirement receipt independent of stream lifetime. Bind retries to originally
captured internal request instances via `cancellation_id`; MUST NOT recapture
by public RID. Fail closed for unknown or unsupported scope. Preserve existing
Abort request/acknowledge behavior for callers that do not opt in.

A future positive receipt must cover every internal generation captured for
the exact public RID, including duplicate public IDs. An unknown target or a
normal generation terminal cannot establish retirement. The scope is scheduler
ownership, not device-wide idleness, native preprocessing, or PD transfer drain.

**Prior art.** [PR #48624](https://github.com/sgl-project/sglang/pull/48624)
is an unmerged Python HTTP scheduler receipt implementation. Reuse its
retirement predicates where possible; it currently excludes the Rust
frontend, which this proposal targets.

**Validation plan (when implemented).**

1. First-token cancel ordering: Abort after first token; receipt only after scheduler retirement.
2. Timeout and disconnect: bounded wait expires or client disconnects without implying retirement.
3. Public RID reuse: later request with the same public ID does not satisfy the prior instance’s receipt.
4. Unsupported topology: unknown or unsupported scope fails closed.
5. Normal generation positive control: completed generate still succeeds without opt-in retirement.
