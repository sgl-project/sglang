//! Minimal validation failure for the shared DTO pipeline.
//!
//! The sampling/regex validators only ever reject malformed input, so the leaf
//! crate exposes just that. Service-flavored failures (shutdown, worker loss,
//! tokenize) stay in their owning crates, which map this into their own error
//! types at the call site.

use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum ValidationError {
    #[error("validation failed: {0}")]
    Validation(String),
}
