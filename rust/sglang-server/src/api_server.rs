//! The API serving stack: transport-agnostic endpoint logic in `core`, the
//! HTTP transport in `http`, the gRPC transport in `grpc`, and the tower
//! layers shared by both stacks in `layers`.

pub(crate) mod core;
pub(crate) mod grpc;
pub(crate) mod http;
mod layers;
