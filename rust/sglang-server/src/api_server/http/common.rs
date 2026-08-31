//! Common control-plane HTTP handlers — `/server_info`, `/get_model_info`
//! (+ `/model_info` alias). The bodies come from `api_server::core::control`;
//! data-plane endpoints (incl. `/health*`, which round-trips a generate
//! probe) live in the sibling `native_api` and `openai` modules.

use std::sync::Arc;

use http::StatusCode;

use super::app::AppState;
use super::response::{HttpResponse, bytes_response, text_response};
use crate::api_server::core::control::{model_info_value, server_info_json};

/// `GET /get_model_info` (+ `/model_info` alias).
pub(super) async fn model_info(state: Arc<AppState>) -> HttpResponse {
    let body = model_info_value(&state.server_args);
    bytes_response(
        StatusCode::OK,
        "application/json",
        serde_json::to_vec(&body).unwrap_or_default(),
    )
}

/// `GET /server_info`.
pub(super) async fn server_info(state: Arc<AppState>) -> HttpResponse {
    match server_info_json(&state).await {
        Ok(json) => bytes_response(StatusCode::OK, "application/json", json),
        // Control-plane errors answer as plain text (Python parity), not the
        // native `{"error": ...}` body.
        Err(e) => text_response(e.http_status(), e.message),
    }
}
