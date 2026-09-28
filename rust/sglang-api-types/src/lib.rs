//! Generated SGLang API types. `src/generated/` is written by
//! `cargo run -p sglang-api-codegen` from `proto/sglang/` — the schema is the
//! ground truth; edit the .proto files and regenerate, never these files.
//! CI enforces regen == committed.

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
