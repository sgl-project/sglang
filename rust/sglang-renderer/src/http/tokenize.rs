//! SGLang-compatible prompt tokenization routes.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    response::{IntoResponse, Response},
    routing::post,
};
use dynamo_protocols::types::CreateChatCompletionRequest;
use futures::future::try_join_all;
use serde_json::{Value, json};

use crate::RendererService;

use super::{error_payload, renderer_status};

pub(super) fn routes(renderer: Arc<RendererService>) -> Router<()> {
    Router::new()
        .route("/tokenize", post(tokenize))
        .route("/v1/tokenize", post(tokenize))
        .with_state(renderer)
}

async fn tokenize(
    State(renderer): State<Arc<RendererService>>,
    Json(mut body): Json<Value>,
) -> Result<Json<Value>, Response> {
    let object = body
        .as_object_mut()
        .ok_or_else(|| bad_request("request body must be a JSON object"))?;
    let prompt = object.get("prompt").filter(|v| !v.is_null()).cloned();
    let messages = object.get("messages").filter(|v| !v.is_null()).cloned();
    if prompt.is_some() == messages.is_some() {
        return Err(bad_request(
            "Exactly one of 'prompt' or 'messages' must be provided.",
        ));
    }

    let tokens = if let Some(prompt) = prompt {
        let add_special_tokens = object
            .get("add_special_tokens")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        match prompt {
            Value::String(text) => json!(
                renderer
                    .tokenize_prompt(text, add_special_tokens)
                    .await
                    .map_err(renderer_error)?
            ),
            Value::Array(values) => {
                let texts = values
                    .into_iter()
                    .map(|value| {
                        value
                            .as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| bad_request("prompt must contain only strings"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                json!(
                    try_join_all(
                        texts
                            .into_iter()
                            .map(|text| renderer.tokenize_prompt(text, add_special_tokens)),
                    )
                    .await
                    .map_err(renderer_error)?
                )
            }
            _ => return Err(bad_request("prompt must be a string or list of strings")),
        }
    } else {
        object.remove("prompt");
        object.remove("add_special_tokens");
        object
            .entry("model")
            .or_insert_with(|| json!(renderer.config().served_model_name));
        let request: CreateChatCompletionRequest =
            serde_json::from_value(body).map_err(|error| bad_request(error.to_string()))?;
        json!(
            renderer
                .tokenize_chat(request)
                .await
                .map_err(renderer_error)?
        )
    };

    let count = match &tokens {
        Value::Array(values) if values.first().is_some_and(Value::is_array) => json!(
            values
                .iter()
                .map(|tokens| tokens.as_array().map_or(0, Vec::len))
                .collect::<Vec<_>>()
        ),
        Value::Array(values) => json!(values.len()),
        _ => unreachable!("token IDs serialize as arrays"),
    };
    Ok(Json(json!({
        "tokens": tokens,
        "count": count,
        "max_model_len": renderer.config().limits.context_len,
    })))
}

fn renderer_error(error: crate::RendererError) -> Response {
    let status = renderer_status(error.kind());
    (status, Json(error_payload(status, error.to_string()))).into_response()
}

fn bad_request(message: impl Into<String>) -> Response {
    let status = axum::http::StatusCode::BAD_REQUEST;
    (status, Json(error_payload(status, message))).into_response()
}
