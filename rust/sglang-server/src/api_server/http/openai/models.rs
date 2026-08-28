//! OpenAI model discovery endpoints.

use std::sync::Arc;

use http::StatusCode;

use super::super::plumbing::{HttpResponse, json_response};
use super::{AppState, openai_error};
use crate::api_server::core::openai::models::model_card;

/// `GET /v1/models` — OpenAI-compatible model list. Served from `server_args`;
/// no scheduler round-trip.
pub(in crate::api_server) async fn available_models(state: Arc<AppState>) -> HttpResponse {
    let base = model_card(&state.server_args);
    json_response(
        StatusCode::OK,
        &serde_json::json!({ "object": "list", "data": [base] }),
    )
}

/// `GET /v1/models/{model}` — `model` arrives percent-decoded (the router owns
/// the decode, matching axum's `Path<String>`).
pub(in crate::api_server) async fn retrieve_model(
    state: Arc<AppState>,
    model: &str,
) -> HttpResponse {
    if model != state.server_args.served_model_name {
        return openai_error(
            StatusCode::NOT_FOUND,
            format!("The model `{model}` does not exist"),
            false,
        );
    }
    json_response(StatusCode::OK, &model_card(&state.server_args))
}
