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

    /// Ingress ring full / scheduler not draining. Surfaced as backpressure.
    #[error("ingress queue full")]
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
    /// HTTP status to surface for the non-streaming error path. Mirrors the
    /// codes used in the Python `_create_error_response`.
    pub fn http_status(&self) -> u16 {
        match self {
            Error::Validation(_) => 400,
            Error::Disconnected => 499, // nginx-style client closed request
            Error::QueueFull => 503,
            _ => 500,
        }
    }
}

#[allow(dead_code)]
pub type Result<T> = std::result::Result<T, Error>;
