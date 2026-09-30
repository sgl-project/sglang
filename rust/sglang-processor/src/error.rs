//! Transport-neutral renderer failures.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RendererErrorKind {
    InvalidRequest,
    Tokenize,
    Unavailable,
    Internal,
}

#[derive(Debug, Clone, Error)]
pub enum RendererError {
    #[error("{0}")]
    Request(String),
    #[error("validation failed: {0}")]
    Validation(String),
    #[error("tokenize failed: {0}")]
    Tokenize(String),
    #[error("renderer is shutting down")]
    Unavailable,
    #[error("render preprocessing worker failed")]
    WorkerDropped,
    #[error("internal renderer error: {0}")]
    Internal(String),
}

impl From<String> for RendererError {
    fn from(message: String) -> Self {
        Self::Request(message)
    }
}

impl From<&str> for RendererError {
    fn from(message: &str) -> Self {
        Self::Request(message.to_owned())
    }
}

impl RendererError {
    pub fn kind(&self) -> RendererErrorKind {
        match self {
            Self::Request(_) | Self::Validation(_) => RendererErrorKind::InvalidRequest,
            Self::Tokenize(_) => RendererErrorKind::Tokenize,
            Self::Unavailable => RendererErrorKind::Unavailable,
            Self::WorkerDropped | Self::Internal(_) => RendererErrorKind::Internal,
        }
    }
}

/// A host error carried through semantic processing without interpreting it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    pub kind: ResponseErrorKind,
    pub message: String,
}

impl From<RendererError> for ResponseError {
    fn from(error: RendererError) -> Self {
        let kind = match error.kind() {
            RendererErrorKind::InvalidRequest => ResponseErrorKind::InvalidRequest,
            RendererErrorKind::Unavailable => ResponseErrorKind::Unavailable,
            RendererErrorKind::Tokenize | RendererErrorKind::Internal => {
                ResponseErrorKind::Internal
            }
        };
        ResponseError {
            kind,
            message: error.to_string(),
        }
    }
}

/// Failure category interpreted by the receiving transport adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResponseErrorKind {
    InvalidRequest,
    Unavailable,
    Internal,
    Upstream(UpstreamErrorCode),
}

/// Original upstream code, preserved without imposing response transport policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpstreamErrorCode {
    Http(u16),
}
