//! OpenAI model discovery endpoints.

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};

use super::{AppState, error_payload, unix_seconds_u32};
use crate::utils::response::error_response;

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/v1/models", get(available_models))
        .route("/v1/models/{model}", get(retrieve_model))
}

/// `GET /v1/models` — OpenAI-compatible model list. Served from `server_args`;
/// no scheduler round-trip.
async fn available_models(State(state): State<AppState>) -> Response {
    let base = model_card(&state);
    Json(serde_json::json!({ "object": "list", "data": [base] })).into_response()
}

async fn retrieve_model(State(state): State<AppState>, Path(model): Path<String>) -> Response {
    if model != state.server_args.served_model_name {
        return error_response(
            StatusCode::NOT_FOUND,
            error_payload(
                StatusCode::NOT_FOUND,
                format!("The model `{model}` does not exist"),
            ),
            false,
        );
    }
    Json(model_card(&state)).into_response()
}

fn model_card(state: &AppState) -> serde_json::Value {
    let name = &state.server_args.served_model_name;
    serde_json::json!({
        "id": name,
        "object": "model",
        "created": unix_seconds_u32(),
        "owned_by": "sglang",
        "root": name,
        "parent": serde_json::Value::Null,
        "max_model_len": state.server_args.model_config.context_len,
    })
}
