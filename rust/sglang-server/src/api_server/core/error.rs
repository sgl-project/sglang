//! The API-boundary error: what an endpoint failure means, without committing
//! to a wire shape. Transports shape it themselves (native/OpenAI JSON body,
//! SSE error frame, and later a gRPC status) — the split that lets one core
//! serve every transport.

use crate::utils::error::Error;

/// An endpoint failure: the HTTP status it maps to plus the client-facing
/// message. `http_code` is authoritative and carried verbatim — including 499
/// (nginx-style client-closed, not an IANA code) and scheduler abort statuses —
/// so transport shaping never re-derives or launders a code.
///
/// Serde: the chat stream rides errors through dynamo's tool-calling jail,
/// whose `Annotated::error` slot is a `String` — the error crosses it
/// JSON-encoded and is decoded on the other side (`chat_event_stream`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ApiError {
    pub(crate) http_code: u16,
    pub(crate) message: String,
}

impl ApiError {
    pub(crate) fn new(http_code: u16, message: impl Into<String>) -> Self {
        Self {
            http_code,
            message: message.into(),
        }
    }

    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self::new(400, message)
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(500, message)
    }

    /// The submit-path refusal: to_scheduler inbox closed, client may retry.
    pub(crate) fn unavailable() -> Self {
        Self::new(503, "service unavailable")
    }

    /// A pipeline error arriving as `ResponseItem::Error`, message via `Display`.
    pub(crate) fn from_pipeline(error: &Error) -> Self {
        Self::new(error.http_status(), error.to_string())
    }

    /// A scheduler validation abort (`finish_reason.abort_status()`), which
    /// carries its own HTTP status + diagnostic.
    pub(crate) fn from_abort(code: u16, message: &str) -> Self {
        Self::new(code, message)
    }

    /// `http_code` as a typed status; an out-of-range code (never expected —
    /// every constructor takes a real one) falls back to 500 rather than
    /// panicking on a scheduler-supplied value.
    pub(crate) fn http_status(&self) -> http::StatusCode {
        http::StatusCode::from_u16(self.http_code)
            .unwrap_or(http::StatusCode::INTERNAL_SERVER_ERROR)
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.http_code)
    }
}
