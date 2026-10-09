//! Failures reported by the Dynamo wrappers.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A rendering, tokenization, or output-parsing failure. Hosts map it onto
/// their own transport and engine error types.
#[derive(Debug, Clone, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum ProcessorError {
    /// The request cannot be rendered or constrained as given.
    #[error("{0}")]
    InvalidRequest(String),
    #[error("tokenize failed: {0}")]
    Tokenize(String),
    /// Output processing reached a state the request could not have caused.
    #[error("{0}")]
    Internal(String),
}

impl From<String> for ProcessorError {
    fn from(message: String) -> Self {
        Self::InvalidRequest(message)
    }
}

impl From<&str> for ProcessorError {
    fn from(message: &str) -> Self {
        Self::InvalidRequest(message.to_owned())
    }
}
