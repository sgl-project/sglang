//! Generated SGLang API types. `src/generated/` is written by
//! `cargo gen-api` from `proto/sglang/` -- the schema is the ground truth;
//! edit the .proto files and regenerate, never these files, and commit the
//! regenerated output with the .proto change.
//!
//! Module paths mirror the proto packages: `api::v1` is `sglang.api.v1` (the
//! HTTP/gRPC contract, with schema-driven JSON serde).

mod ext;

/// Version of this crate, which versions the `sglang.api.v1` schema and its
/// generated Rust API independently of SGLang releases (see README.md).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod api {
    // Generated code optimizes for template uniformity, not lint cleanliness.
    #[allow(clippy::all)]
    pub mod v1 {
        include!("generated/sglang.api.v1.rs");
        include!("generated/sglang.api.v1.serde.rs");
        include!("generated/sglang.api.v1.tonic.rs");
    }
}
