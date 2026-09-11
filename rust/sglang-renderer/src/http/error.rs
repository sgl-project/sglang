use axum::{
    Json,
    extract::rejection::JsonRejection,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::ResponseError;

fn openai_error(code: StatusCode, message: impl Into<String>) -> Response {
    (code, Json(error_payload(code, message))).into_response()
}

pub(super) fn json_rejection_response(rejection: JsonRejection) -> Response {
    let status = if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
        StatusCode::PAYLOAD_TOO_LARGE
    } else {
        StatusCode::BAD_REQUEST
    };
    openai_error(status, rejection.body_text())
}

pub(super) fn response_error(error: ResponseError) -> Response {
    let status = response_status(&error);
    openai_error(status, error.message)
}

pub(super) fn response_status(error: &ResponseError) -> StatusCode {
    use crate::{ResponseErrorKind, UpstreamErrorCode};
    match error.kind {
        ResponseErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
        ResponseErrorKind::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        ResponseErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        ResponseErrorKind::Upstream(UpstreamErrorCode::Http(code)) => {
            StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

pub(super) fn error_payload(status: StatusCode, message: impl Into<String>) -> serde_json::Value {
    let error_type = if status.is_server_error() {
        "InternalServerError"
    } else {
        "BadRequestError"
    };
    crate::openai::response::error_payload(status.as_u16(), message, error_type)
}
