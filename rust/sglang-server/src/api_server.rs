//! API server (axum / tokio). I/O-bound; own pinned multi-thread runtime. The
//! transport-neutral [`core::CoreHandle`] is the shared entry into the runtime
//! pipeline; `http` and `grpc` are the wire adapters on top of it. Generation
//! handlers render semantic core events as unary JSON, SSE, or gRPC frames;
//! control handlers serialize typed core results such as server metadata.
pub(crate) mod core;
mod disaggregation;
pub(crate) mod grpc;
pub(crate) mod http;
mod log;
