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
pub(crate) mod openai;

#[cfg(test)]
use crate::frontend::{FrontendError, FrontendErrorKind};
#[cfg(test)]
use axum::http::StatusCode;

/// Map transport-neutral failures into HTTP status codes. Response bodies and
/// stream framing remain the responsibility of each HTTP API surface.
pub(crate) use crate::openai::frontend_error_status;

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
