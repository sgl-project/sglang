# SGLang runtime protocol

This directory contains the canonical Protocol Buffers contract for the SGLang runtime gRPC API. The schema is published independently from SGLang's Python wheels and Rust server artifacts as [`buf.build/sgl-project/sglang`](https://buf.build/sgl-project/sglang).

The current API is `sglang.runtime.v1` in [`sglang/runtime/v1/sglang.proto`](sglang/runtime/v1/sglang.proto). Keep the schema here: do not copy it into a Rust crate, Python package, or wheel. The internal `rust/sglang-grpc-proto` crate compiles this file for the in-tree server, while external consumers should use the BSR module or a generated SDK.

## Labels

- `nightly` tracks the schema from Git `main` and is refreshed daily at 08:00 UTC.
- `main` advances only for a standard SGLang release.
- A release also creates or verifies the matching `vX.Y.Z` label from the same immutable BSR commit.

Buf commits are immutable, but labels are mutable pointers. Production consumers must pin an immutable BSR commit or an exact generated-SDK version instead of following `nightly`, `main`, or a label that repository policy does not protect. See [Buf commits and labels](https://buf.build/docs/bsr/commits-labels/) and [buf.md](buf.md) for publishing and SDK details.
