//! Rust bindings for SGLang's canonical `sglang.runtime.v1` protobuf API.
//!
//! The generated messages and Tonic client/server interfaces intentionally
//! contain no frontend implementation or transport policy.

/// Protobuf packages exposed with the same namespace as the wire contract.
pub mod sglang {
    pub mod runtime {
        pub mod v1 {
            tonic::include_proto!("sglang.runtime.v1");
        }
    }
}

/// Short alias for consumers that work with only the current runtime API.
pub use sglang::runtime::v1 as proto;
