use axum::{
    Json,
    extract::rejection::JsonRejection,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};

use crate::{RendererErrorKind, error::RendererError};

pub(super) fn error_payload(code: StatusCode, message: impl Into<String>) -> serde_json::Value {
    let error_type = if code.is_server_error() {
        "InternalServerError"
    } else {
        "BadRequestError"
    };
    serde_json::json!({
        "error": {
            "object": "error", "message": message.into(), "type": error_type,
            "param": null, "code": code.as_u16(),
        }
    })
}

pub(super) fn openai_error(code: StatusCode, message: impl Into<String>, stream: bool) -> Response {
    let payload = error_payload(code, message);
    if !stream {
        return (code, Json(payload)).into_response();
    }
    let frames = [payload.to_string(), "[DONE]".to_owned()];
    Sse::new(futures::stream::iter(frames.map(|data| {
        Ok::<_, std::convert::Infallible>(Event::default().data(data))
    })))
    .into_response()
}

pub(super) fn json_rejection_response(rejection: JsonRejection) -> Response {
    let status = if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
        StatusCode::PAYLOAD_TOO_LARGE
    } else {
        StatusCode::BAD_REQUEST
    };
    openai_error(status, rejection.body_text(), false)
}

pub(super) fn renderer_status(error: &RendererError) -> StatusCode {
    match error.kind() {
        RendererErrorKind::InvalidRequest => StatusCode::BAD_REQUEST,
        RendererErrorKind::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        RendererErrorKind::Tokenize | RendererErrorKind::Internal => {
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}
