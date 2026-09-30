# SGLang gRPC types

This crate provides the generated Prost messages and Tonic client/server
interfaces for SGLang's canonical `sglang.runtime.v1` API.

The language-independent contract remains in
`sglang/runtime/v1/sglang.proto`. Rust bindings are generated from that schema
at build time with a vendored `protoc`, so consumers do not need a system
protobuf compiler.

The generated API currently uses Tonic 0.12 to remain compatible with
SGLang's existing gRPC server.

This crate contains protocol types only. SGLang's request handling and server
implementation remain in their respective frontend crates.
