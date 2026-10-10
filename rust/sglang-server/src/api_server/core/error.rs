//! Transport-neutral failure categories, and how each wire adapter maps them
//! into its own status space.

use http::StatusCode;
use sglang_api_types::api::v1 as api;

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
    #[error("{message}")]
    RuntimeRejected {
        kind: CoreErrorKind,
        message: String,
        legacy_http_status: u16,
    },

    /// The runtime replied with an unexpected or malformed result.
    #[error("{0}")]
    InvalidResponse(String),

    #[error("{0}")]
    Internal(String),
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
            Self::ResponseTruncated => CoreErrorKind::Internal,
            Self::RuntimeRejected { kind, .. } => *kind,
            Self::InvalidResponse(_) | Self::Internal(_) => CoreErrorKind::Internal,
        }
    }

    /// Translate the scheduler's inherited HTTP-coded failure into a semantic
    /// category once, at the runtime boundary. HTTP can retain the exact legacy
    /// status while other adapters consume [`Self::kind`].
    pub(super) fn from_runtime_rejection(message: String, legacy_http_status: u16) -> Self {
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
        }
    }

    /// The HTTP status for this failure. Response bodies and stream framing
    /// remain the responsibility of each HTTP API surface.
    /// This failure as one `/generate` item: the native error body with the
    /// HTTP status as `code`, plus the batch position where there is one.
    pub(crate) fn stream_error(&self, index: Option<u32>) -> api::GenerateStreamError {
        api::GenerateStreamError {
            error: Some(api::ErrorBody {
                message: self.to_string(),
                code: u32::from(self.http_status().as_u16()),
            }),
            index,
        }
    }

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
            CoreError::RuntimeRejected {
                kind: CoreErrorKind::InvalidArgument,
                message: "runtime rejected request".into(),
                legacy_http_status: 432,
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
}
