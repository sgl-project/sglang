//! OpenAI model discovery endpoints.

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use std::sync::Arc;

use sglang_api_types::api::v1 as api;

use super::{AppState, error_payload, unix_seconds_u32};
use crate::utils::response::error_response;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/models", get(available_models))
        .route("/v1/models/{model}", get(retrieve_model))
}

/// `GET /v1/models` — OpenAI-compatible model list. Served from core metadata;
/// no scheduler round-trip.
async fn available_models(State(state): State<Arc<AppState>>) -> Response {
    let models: Vec<_> = state
        .core
        .list_models()
        .into_iter()
        .map(model_card)
        .collect();
    Json(serde_json::json!({ "object": "list", "data": models })).into_response()
}

async fn retrieve_model(State(state): State<Arc<AppState>>, Path(model): Path<String>) -> Response {
    let Some(card) = state
        .core
        .list_models()
        .into_iter()
        .find(|card| card.id == model)
    else {
        let mut payload = error_payload(
            StatusCode::NOT_FOUND,
            format!("The model `{model}` does not exist"),
        );
        payload["error"]["param"] = "model".into();
        payload["error"]["code"] = "model_not_found".into();
        return error_response(StatusCode::NOT_FOUND, payload, false);
    };
    Json(model_card(card)).into_response()
}

fn model_card(card: api::ModelCard) -> serde_json::Value {
    serde_json::json!({
        "id": card.id,
        "object": "model",
        "created": unix_seconds_u32(),
        "owned_by": "sglang",
        "root": card.root,
        "parent": card.parent,
        "max_model_len": card.max_model_len,
    })
}
