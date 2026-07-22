//! OpenAI-compatible endpoints: `/v1/completions`, `/v1/chat/completions`, and
//! `/v1/models`. Each runs the same tokenize→generate→detok pipeline as
//! `/generate` and shapes the neutral [`ChunkEvent`] delta into OpenAI types
//! (`dynamo-protocols`), with chat-template rendering (`dynamo-renderer`) and
//! reasoning / tool-call parsing (`dynamo-parsers`).
//!
//! Mounted on the shared [`AppState`](super::AppState) by the parent
//! `api_server` module; the submit machinery and control plane live there.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use axum::{Router, routing::get};

use super::AppState;

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<AppState> {
    // `/v1/models` is OpenAI-compatible; completions/chat land here too.
    Router::new().route("/v1/models", get(available_models))
}

/// `GET /v1/models` — OpenAI-compatible model list. Served from `server_args`;
/// no scheduler round-trip. Mirrors `http_server.available_models`.
///
/// TODO(v1/models): when `--enable-lora`, append a `ModelCard` per loaded LoRA
/// adapter (`id=lora_name, root=lora_path, parent=served_model_name,
/// max_model_len=None`). Adapters load/unload at runtime, so that part needs a
/// control-request query to the scheduler's LoRA registry.
async fn available_models(State(state): State<AppState>) -> Response {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let name = &state.server_args.served_model_name;
    let base = serde_json::json!({
        "id": name,
        "object": "model",
        "created": created,
        "owned_by": "sglang",
        "root": name,
        "parent": serde_json::Value::Null,
        "max_model_len": state.server_args.model_config.context_len,
    });
    let list = serde_json::json!({ "object": "list", "data": [base] });
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        serde_json::to_vec(&list).unwrap_or_default(),
    )
        .into_response()
}
