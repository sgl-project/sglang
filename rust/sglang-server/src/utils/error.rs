//! Error type shared by all stages. Kept `Clone` so a single failure can be
//! reported to the client stream and logged without moving ownership around.

use thiserror::Error;

// Some variants are emitted only once their stage matures (real validation,
// the deferred Encoder, HF detok). They are part of the stable error surface.
#[allow(dead_code)]
#[derive(Debug, Clone, Error)]
pub enum Error {
    #[error("validation failed: {0}")]
    Validation(String),

    #[error("tokenize failed: {0}")]
    Tokenize(String),

    #[error("encode failed: {0}")]
    Encode(String),

    #[error("detokenize failed: {0}")]
    Detokenize(String),

    /// To-scheduler channel full. Surfaced as backpressure.
    #[error("to_scheduler channel full")]
    QueueFull,

    /// Client went away mid-stream. Drives `Aborted`, not `Failed`.
    #[error("client disconnected")]
    Disconnected,

    #[error("serialization error: {0}")]
    Codec(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl Error {
    /// Whether this failure represents a server defect rather than an expected
    /// request, capacity, or cancellation outcome.
    pub fn is_server_fault(&self) -> bool {
        matches!(
            self,
            Error::Tokenize(_)
                | Error::Encode(_)
                | Error::Detokenize(_)
                | Error::Codec(_)
                | Error::Internal(_)
        )
    }
}

#[allow(dead_code)]
pub type Result<T> = std::result::Result<T, Error>;
