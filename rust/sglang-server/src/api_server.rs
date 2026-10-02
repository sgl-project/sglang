//! API server (axum / tokio). I/O-bound; own pinned multi-thread runtime. Only
//! this module knows HTTP. Its handlers use the transport-neutral
//! [`crate::frontend::FrontendHandle`] to enter the shared runtime pipeline.
//! Generation handlers render semantic frontend events as unary JSON or SSE;
//! control handlers serialize typed frontend results such as server metadata.
pub mod app;
mod common;
mod disaggregation;
mod frame;
mod log;
mod native_api;
mod openai;

use axum::http::StatusCode;

use crate::frontend::{FrontendError, FrontendErrorKind};

/// Map transport-neutral failures into HTTP status codes. Response bodies and
/// stream framing remain the responsibility of each HTTP API surface.
pub(crate) fn frontend_error_status(error: &FrontendError) -> StatusCode {
    if let FrontendError::RuntimeRejected {
        legacy_http_status, ..
    } = error
    {
        return StatusCode::from_u16(*legacy_http_status)
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    }

    match error.kind() {
        FrontendErrorKind::InvalidArgument => StatusCode::BAD_REQUEST,
        FrontendErrorKind::NotFound => StatusCode::NOT_FOUND,
        FrontendErrorKind::FailedPrecondition => StatusCode::PRECONDITION_FAILED,
        // Existing intake backpressure has historically surfaced as 503.
        FrontendErrorKind::ResourceExhausted | FrontendErrorKind::Unavailable => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        FrontendErrorKind::Cancelled => StatusCode::from_u16(499).expect("499 is valid"),
        FrontendErrorKind::DeadlineExceeded => StatusCode::GATEWAY_TIMEOUT,
        FrontendErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_error_categories_map_at_the_http_boundary() {
        assert_eq!(
            frontend_error_status(&FrontendError::InvalidArgument("bad".into())),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            frontend_error_status(&FrontendError::Unavailable),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            frontend_error_status(&FrontendError::Overloaded("full".into())),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            frontend_error_status(&FrontendError::Cancelled("gone".into())).as_u16(),
            499
        );
        assert_eq!(
            frontend_error_status(&FrontendError::RuntimeRejected {
                kind: FrontendErrorKind::InvalidArgument,
                message: "runtime rejected request".into(),
                legacy_http_status: 432,
            })
            .as_u16(),
            432
        );
        assert_eq!(
            frontend_error_status(&FrontendError::InvalidResponse("bad reply".into())),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            frontend_error_status(&FrontendError::Internal("bug".into())),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
