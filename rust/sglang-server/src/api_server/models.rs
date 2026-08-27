//! OpenAI model discovery endpoints.

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use std::sync::Arc;

use super::app::AppState;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/models", get(available_models))
        .route("/v1/models/{model}", get(retrieve_model))
}

/// `GET /v1/models` — OpenAI-compatible model list. Served from `server_args`;
/// no scheduler round-trip.
async fn available_models(State(state): State<Arc<AppState>>) -> Response {
    let base = model_card(&state);
    Json(serde_json::json!({ "object": "list", "data": [base] })).into_response()
}

async fn retrieve_model(State(state): State<Arc<AppState>>, Path(model): Path<String>) -> Response {
    if model != state.server_args.served_model_name {
        return (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "object": "error",
                    "message": format!("The model `{model}` does not exist"),
                    "type": "BadRequestError",
                    "param": null,
                    "code": StatusCode::NOT_FOUND.as_u16(),
                }
            })),
        )
            .into_response();
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

fn unix_seconds_u32() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| u32::try_from(duration.as_secs()).unwrap_or(u32::MAX))
        .unwrap_or(0)
}
