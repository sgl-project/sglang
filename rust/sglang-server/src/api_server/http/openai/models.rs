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

use super::{AppState, openai_error, unix_seconds_u32};

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/models", get(available_models))
        // A served model name is a path, not a segment: it defaults to
        // `--model-path`, so the id this server publishes is `Qwen/Qwen3-8B` or an
        // absolute checkpoint path. A single-segment capture answers 404 for the
        // id the list endpoint just returned. Python serves the same route with a
        // `:path` converter for this reason.
        .route("/v1/models/{*model}", get(retrieve_model))
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
    // An absolute checkpoint path carries a leading `/` of its own, and the
    // wildcard capture keeps only what follows the matched separator, so compare
    // the ids with that separator normalised on both sides.
    let requested = model.trim_start_matches('/');
    let Some(card) = state
        .core
        .list_models()
        .into_iter()
        .find(|card| card.id.trim_start_matches('/') == requested)
    else {
        return openai_error(
            StatusCode::NOT_FOUND,
            format!("The model `{model}` does not exist"),
            false,
        );
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
