//! Transport-neutral renderer failures.

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

impl RendererError {
    pub fn kind(&self) -> RendererErrorKind {
        match self {
            Self::Validation(_) => RendererErrorKind::InvalidRequest,
            Self::Tokenize(_) => RendererErrorKind::Tokenize,
            Self::Unavailable => RendererErrorKind::Unavailable,
            Self::WorkerDropped | Self::Internal(_) => RendererErrorKind::Internal,
        }
    }
}

pub type Result<T> = std::result::Result<T, RendererError>;
