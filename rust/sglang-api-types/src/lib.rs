//! Generated SGLang API types. `src/generated/` is written by
//! `cargo gen-api` from `proto/sglang/` -- the schema is the ground truth;
//! edit the .proto files and regenerate, never these files, and commit the
//! regenerated output with the .proto change.
//!
//! Module paths mirror the proto packages: `api::v1` is `sglang.api.v1` (the
//! HTTP/gRPC contract, with schema-driven JSON serde), `runtime::v1` is
//! `sglang.runtime.v1` (the gRPC bridge's protocol types, prost + tonic only).

mod ext;

pub mod api {
    // Generated code optimizes for template uniformity, not lint cleanliness.
    #[allow(clippy::all)]
    pub mod v1 {
        include!("generated/sglang.api.v1.rs");
        include!("generated/sglang.api.v1.serde.rs");
        include!("generated/sglang.api.v1.tonic.rs");
    }
}

pub mod runtime {
    #[allow(clippy::all)]
    pub mod v1 {
        include!("generated/sglang.runtime.v1.rs");
        include!("generated/sglang.runtime.v1.tonic.rs");
    }
}
