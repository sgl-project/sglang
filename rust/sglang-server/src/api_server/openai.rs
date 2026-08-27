//! OpenAI-compatible model discovery endpoints.

use std::sync::Arc;

use axum::{Router, http::StatusCode, response::Response};

mod models;

use super::app::AppState;
use crate::utils::response::error_response;

pub(super) fn routes() -> Router<Arc<AppState>> {
    models::routes()
}

fn unix_seconds_u32() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| u32::try_from(duration.as_secs()).unwrap_or(u32::MAX))
        .unwrap_or(0)
}

fn error_payload(code: StatusCode, message: impl Into<String>) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "object": "error",
            "message": message.into(),
            "type": "BadRequestError",
            "param": null,
            "code": code.as_u16(),
        }
    })
}

fn openai_error(code: StatusCode, message: impl Into<String>, stream: bool) -> Response {
    error_response(code, error_payload(code, message), stream)
}
