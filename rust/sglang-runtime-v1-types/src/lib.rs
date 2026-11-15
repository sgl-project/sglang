//! Generated bindings for the deprecated `sglang.runtime.v1` package, the
//! legacy gRPC bridge's protocol (prost + tonic only). This crate is removed
//! together with `sglang.runtime.v1`; new code uses `sglang.api.v1` from
//! `sglang-api-types`.
//!
//! `src/generated/` is written by `cargo gen-api` from
//! `proto/sglang/runtime/v1/`; edit the .proto file and regenerate, never
//! these files.

include!("generated/sglang.runtime.v1.rs");
include!("generated/sglang.runtime.v1.tonic.rs");
