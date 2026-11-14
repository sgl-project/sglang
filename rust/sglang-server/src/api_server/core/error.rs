//! Transport-neutral failure categories, and how each wire adapter maps them
//! into its own status space.

use http::StatusCode;
use sglang_api_types::api::v1 as api;

use super::CoreOutput;
use crate::multi_modality::worker::MediaError;

/// Python's stream error frame `type` for a `ValueError`.
pub(crate) const STREAM_INVALID_REQUEST_ERROR: &str = "invalid_request_error";
/// The stream error frame `type` for any other failure: the server's own,
/// so a retry may succeed.
pub(crate) const STREAM_INTERNAL_ERROR: &str = "internal_error";

/// Protocol-independent error category used by transport adapters.
///
/// The set intentionally follows operation semantics rather than HTTP or gRPC
/// status spaces. An adapter maps this category into its own wire status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CoreErrorKind {
    InvalidArgument,
    NotFound,
    FailedPrecondition,
    ResourceExhausted,
    Cancelled,
    DeadlineExceeded,
    Unavailable,
    Internal,
}

/// A transport-neutral failure category.
///
/// HTTP and gRPC adapters map these categories into their own status spaces;
/// neither adapter needs to understand errors emitted by runtime stages.
#[derive(Clone, Debug, thiserror::Error)]
pub(crate) enum CoreError {
    #[error("{0}")]
    InvalidArgument(String),

    /// The intake loop has shut down. A caller may retry elsewhere.
    #[error("service unavailable")]
    Unavailable,

    /// Runtime capacity was exhausted before the request could make progress.
    #[error("{0}")]
    Overloaded(String),

    #[error("{0}")]
    Cancelled(String),

    /// A DP rank did not answer a control request within the fan-out's
    /// deadline (see [`super::fanout`]).
    #[error("{0}")]
    DeadlineExceeded(String),

    /// The internal producer disappeared before returning a terminal result.
    ///
    /// This is a server-side failure, not evidence that the caller cancelled.
    #[error("response truncated before completion")]
    ResponseTruncated,

    /// A scheduler-side abort that carried an error status.
    ///
    /// The current Python scheduler communicates this category as an HTTP
    /// status code inside its finish reason. The semantic event is still
    /// `Failed`; retaining the legacy code only lets the HTTP adapter preserve
    /// existing behavior while another adapter maps it into its own status
    /// space. New core failures should use the semantic variants above.
    ///
    /// `output` is the request's terminal output, abort finish reason
    /// included, for the one surface that reports the abort as an output: a
    /// native `/generate` stream (see [`Self::into_rejected_output`]).
    #[error("{message}")]
    RuntimeRejected {
        kind: CoreErrorKind,
        message: String,
        legacy_http_status: u16,
        output: Box<CoreOutput>,
    },

    /// The runtime replied with an unexpected or malformed result.
    #[error("{0}")]
    InvalidResponse(String),

    #[error("{0}")]
    Internal(String),
}

/// A media failure keeps Python's class and message, whether admission or the
/// MM processor raised it: `ValueError` is a 400, anything else a 500.
impl From<MediaError> for CoreError {
    fn from(error: MediaError) -> Self {
        match error {
            MediaError::Client(message) => Self::InvalidArgument(message),
            MediaError::Server(message) => Self::Internal(message),
        }
    }
}

impl CoreError {
    /// Semantic status for adapters that do not use the scheduler's legacy
    /// HTTP code space (notably the future gRPC adapter).
    pub(crate) fn kind(&self) -> CoreErrorKind {
        match self {
            Self::InvalidArgument(_) => CoreErrorKind::InvalidArgument,
            Self::Unavailable => CoreErrorKind::Unavailable,
            Self::Overloaded(_) => CoreErrorKind::ResourceExhausted,
            Self::Cancelled(_) => CoreErrorKind::Cancelled,
            Self::DeadlineExceeded(_) => CoreErrorKind::DeadlineExceeded,
            Self::ResponseTruncated => CoreErrorKind::Internal,
            Self::RuntimeRejected { kind, .. } => *kind,
            Self::InvalidResponse(_) | Self::Internal(_) => CoreErrorKind::Internal,
        }
    }

    /// Translate the scheduler's inherited HTTP-coded failure into a semantic
    /// category once, at the runtime boundary. HTTP can retain the exact legacy
    /// status while other adapters consume [`Self::kind`].
    pub(crate) fn from_runtime_rejection(
        message: String,
        legacy_http_status: u16,
        output: CoreOutput,
    ) -> Self {
        let kind = match legacy_http_status {
            404 => CoreErrorKind::NotFound,
            408 | 504 => CoreErrorKind::DeadlineExceeded,
            412 => CoreErrorKind::FailedPrecondition,
            413 | 429 => CoreErrorKind::ResourceExhausted,
            499 => CoreErrorKind::Cancelled,
            502 | 503 => CoreErrorKind::Unavailable,
            400..=499 => CoreErrorKind::InvalidArgument,
            _ => CoreErrorKind::Internal,
        };
        Self::RuntimeRejected {
            kind,
            message,
            legacy_http_status,
            output: Box::new(output),
        }
    }

    /// The terminal output of a scheduler-side abort that carried an error
    /// status, finish reason and counts included; any other failure comes
    /// back as is. Python's server ends a native `/generate` stream on that
    /// output as a normal frame, so a client reads the abort from
    /// `meta_info.finish_reason`, not from an error item.
    pub(crate) fn into_rejected_output(self) -> Result<CoreOutput, Self> {
        match self {
            Self::RuntimeRejected { output, .. } => Ok(*output),
            error => Err(error),
        }
    }

    /// This failure as one unary `/generate` batch item: the native error
    /// body with the HTTP status as `code`, plus the batch position where
    /// there is one. Python's unary body has no `type` or `retryable`.
    pub(crate) fn item_error(&self, index: Option<u32>) -> api::GenerateStreamError {
        api::GenerateStreamError {
            error: Some(api::ErrorBody {
                message: self.to_string(),
                code: u32::from(self.http_status().as_u16()),
                ..Default::default()
            }),
            index,
        }
    }

    /// This failure as one streamed `/generate` item: [`Self::item_error`]
    /// plus the `type` and `retryable` Python's stream error frame carries.
    /// Python reports a `ValueError` (a client error, or one raised with its
    /// own status, such as 503) as `invalid_request_error`, not retryable,
    /// whatever its code. Python ends the stream without a frame on any other
    /// exception; this server answers that failure (an internal one here, see
    /// the `MediaError` conversion above) with a 500 frame typed
    /// `internal_error` and retryable, since the fault is the server's.
    pub(crate) fn stream_error(&self, index: Option<u32>) -> api::GenerateStreamError {
        let mut item = self.item_error(index);
        if let Some(body) = item.error.as_mut() {
            let (error_type, retryable) = match self.kind() {
                CoreErrorKind::Internal => (STREAM_INTERNAL_ERROR, true),
                _ => (STREAM_INVALID_REQUEST_ERROR, false),
            };
            body.r#type = Some(error_type.to_owned());
            body.retryable = Some(retryable);
        }
        item
    }

    /// The HTTP status for this failure. Response bodies and stream framing
    /// remain the responsibility of each HTTP API surface.
    pub(crate) fn http_status(&self) -> StatusCode {
        if let CoreError::RuntimeRejected {
            legacy_http_status, ..
        } = self
        {
            return StatusCode::from_u16(*legacy_http_status)
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        }

        match self.kind() {
            CoreErrorKind::InvalidArgument => StatusCode::BAD_REQUEST,
            CoreErrorKind::NotFound => StatusCode::NOT_FOUND,
            CoreErrorKind::FailedPrecondition => StatusCode::PRECONDITION_FAILED,
            // Existing intake backpressure has historically surfaced as 503.
            CoreErrorKind::ResourceExhausted | CoreErrorKind::Unavailable => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            CoreErrorKind::Cancelled => StatusCode::from_u16(499).expect("499 is valid"),
            CoreErrorKind::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
            CoreErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_error_categories_map_at_the_http_boundary() {
        assert_eq!(
            CoreError::InvalidArgument("bad".into()).http_status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            CoreError::Unavailable.http_status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            CoreError::Overloaded("full".into()).http_status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            CoreError::Cancelled("gone".into()).http_status().as_u16(),
            499
        );
        assert_eq!(
            CoreError::DeadlineExceeded("late".into()).http_status(),
            StatusCode::GATEWAY_TIMEOUT
        );
        assert_eq!(
            CoreError::RuntimeRejected {
                kind: CoreErrorKind::InvalidArgument,
                message: "runtime rejected request".into(),
                legacy_http_status: 432,
                output: Box::default(),
            }
            .http_status()
            .as_u16(),
            432
        );
        assert_eq!(
            CoreError::InvalidResponse("bad reply".into()).http_status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            CoreError::Internal("bug".into()).http_status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    /// A streamed item carries Python's `type` and `retryable`: a client or
    /// status-carrying failure (Python's `ValueError`) is a non-retryable
    /// `invalid_request_error` whatever its code; an internal one is a
    /// retryable `internal_error`. A unary item carries neither, as Python's
    /// unary body doesn't.
    #[test]
    fn stream_errors_carry_python_type_and_retryable() {
        let cases = [
            (
                CoreError::InvalidArgument("bad".into()),
                400,
                "invalid_request_error",
                false,
            ),
            (CoreError::Unavailable, 503, "invalid_request_error", false),
            (
                CoreError::Overloaded("full".into()),
                503,
                "invalid_request_error",
                false,
            ),
            (
                CoreError::Internal("bug".into()),
                500,
                "internal_error",
                true,
            ),
            (
                CoreError::InvalidResponse("bad reply".into()),
                500,
                "internal_error",
                true,
            ),
            (CoreError::ResponseTruncated, 500, "internal_error", true),
        ];
        for (error, code, error_type, retryable) in cases {
            let streamed = serde_json::to_value(error.stream_error(Some(1))).unwrap();
            assert_eq!(
                streamed,
                serde_json::json!({
                    "error": {
                        "message": error.to_string(),
                        "code": code,
                        "type": error_type,
                        "retryable": retryable,
                    },
                    "index": 1,
                }),
                "{error:?}"
            );
            let unary = serde_json::to_value(error.item_error(None)).unwrap();
            assert_eq!(
                unary,
                serde_json::json!({"error": {"message": error.to_string(), "code": code}}),
                "{error:?}"
            );
        }
    }
}
