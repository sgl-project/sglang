//! API server (axum / tokio). I/O-bound; own pinned multi-thread runtime. The
//! transport-neutral [`core::CoreHandle`] is the shared entry into the runtime
//! pipeline; `http` and `grpc` are the wire adapters on top of it. Generation
//! handlers render semantic core events as unary JSON, SSE, or gRPC frames;
//! control handlers serialize typed core results such as server metadata.
pub(crate) mod core;
mod disaggregation;
pub mod grpc;
pub mod http;
mod log;

use axum::http::StatusCode;

use crate::api_server::core::{CoreError, CoreErrorKind};

/// Map transport-neutral failures into HTTP status codes. Response bodies and
/// stream framing remain the responsibility of each HTTP API surface.
pub(crate) fn core_error_status(error: &CoreError) -> StatusCode {
    if let CoreError::RuntimeRejected {
        legacy_http_status, ..
    } = error
    {
        return StatusCode::from_u16(*legacy_http_status)
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    }

    match error.kind() {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontend_error_categories_map_at_the_http_boundary() {
        assert_eq!(
            core_error_status(&CoreError::InvalidArgument("bad".into())),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            core_error_status(&CoreError::Unavailable),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            core_error_status(&CoreError::Overloaded("full".into())),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            core_error_status(&CoreError::Cancelled("gone".into())).as_u16(),
            499
        );
        assert_eq!(
            core_error_status(&CoreError::RuntimeRejected {
                kind: CoreErrorKind::InvalidArgument,
                message: "runtime rejected request".into(),
                legacy_http_status: 432,
            })
            .as_u16(),
            432
        );
        assert_eq!(
            core_error_status(&CoreError::InvalidResponse("bad reply".into())),
            StatusCode::INTERNAL_SERVER_ERROR
        );
        assert_eq!(
            core_error_status(&CoreError::Internal("bug".into())),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
