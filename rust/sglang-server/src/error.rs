//! Error type shared by all stages. Kept `Clone` so a single failure can be
//! reported to the client stream and logged without moving ownership around.

use std::path::PathBuf;

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

/// Chat-template resolution/rendering failures (`api_server::openai::template`).
/// Lives here with the other error types; not a variant of [`Error`] — it is
/// resolved at template-load/render time and reported through the OpenAI
/// error shape, never through the egress pipeline.
#[derive(Debug, Error)]
pub enum TemplateError {
    #[error("failed to read {kind} `{path}`: {source}")]
    Read {
        kind: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to parse {kind} `{path}`: {source}")]
    Parse {
        kind: &'static str,
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("chat template `{path}` is not a built-in name or a valid file path")]
    NotFound { path: PathBuf },

    #[error("chat template `{path}` is not a file")]
    NotFile { path: PathBuf },

    #[error("tokenizer config must be a JSON object")]
    ConfigNotObject,

    #[error("invalid chat template config: {source}")]
    Config {
        #[source]
        source: serde_json::Error,
    },

    #[error("tokenizer has no chat template")]
    Missing,

    #[error("tokenizer_config.json is required for this chat template source but was not found")]
    MissingConfig,

    #[error("invalid chat template: {message}")]
    Renderer { message: String },

    #[error("legacy chat template `{path}` must be a JSON object")]
    LegacyNotObject { path: PathBuf },

    #[error("legacy chat template `{path}` requires string field `{field}`")]
    LegacyMissingField { path: PathBuf, field: String },

    #[error("unknown separator style `{style}` in `{path}`")]
    UnknownStyle { path: PathBuf, style: String },

    #[error("unknown separator style `{style}`")]
    InvalidStyle { style: String },

    #[error("sep2 is required for separator style `{style}` but is not set")]
    MissingSep2 { style: String },

    #[error("stop_str must be a single string for separator style `{style}`")]
    InvalidStopString { style: String },

    #[error("the {role} message should be a single text")]
    NonTextContent { role: &'static str },

    #[error("multimodal {role} message content is not supported by legacy templates")]
    MediaContent { role: &'static str },

    #[error("unsupported message role `{role}` in legacy chat template")]
    UnsupportedRole { role: &'static str },
}
