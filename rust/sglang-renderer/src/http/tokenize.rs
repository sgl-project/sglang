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

use super::error::{error_payload, renderer_status};

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

    let (tokens, count) = if let Some(prompt) = prompt {
        let add_special_tokens = object
            .get("add_special_tokens")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        match prompt {
            Value::String(text) => {
                let tokens = renderer
                    .tokenize_prompt(text, add_special_tokens)
                    .await
                    .map_err(renderer_error)?;
                (json!(tokens), json!(tokens.len()))
            }
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
                let tokens = try_join_all(
                    texts
                        .into_iter()
                        .map(|text| renderer.tokenize_prompt(text, add_special_tokens)),
                )
                .await
                .map_err(renderer_error)?;
                let count = tokens.iter().map(Vec::len).collect::<Vec<_>>();
                (json!(tokens), json!(count))
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
        let tokens = renderer
            .tokenize_chat(request)
            .await
            .map_err(renderer_error)?;
        (json!(tokens), json!(tokens.len()))
    };
    Ok(Json(json!({
        "tokens": tokens,
        "count": count,
        "max_model_len": renderer.config().limits.context_len,
    })))
}

fn renderer_error(error: crate::RendererError) -> Response {
    let status = renderer_status(&error);
    (status, Json(error_payload(status, error.to_string()))).into_response()
}

fn bad_request(message: impl Into<String>) -> Response {
    let status = axum::http::StatusCode::BAD_REQUEST;
    (status, Json(error_payload(status, message))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    use crate::{
        OpenAIRequestLowerer, PooledTokenizer, RendererConfig, RendererError, RendererLimits,
        SamplingDefaults, TextTokenizer,
    };

    struct PrefixTokenizer;

    impl TextTokenizer for PrefixTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<i32>, RendererError> {
            Ok(std::iter::once(1)
                .chain(text.split_whitespace().map(|_| 7))
                .collect())
        }

        fn auto_specials(&self) -> Vec<i32> {
            vec![1]
        }
    }

    fn app() -> Router<()> {
        let config = RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: ".".into(),
            revision: None,
            model_path: String::new(),
            chat_template: Some("chatml".into()),
            tool_call_parser: None,
            reasoning_parser: None,
            stream_response_default_include_usage: false,
            skip_tokenizer_init: false,
            vocab_size: 100,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                skip_tokenizer_init: false,
                vocab_size: 100,
                context_len: 64,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        };
        routes(Arc::new(RendererService::new(
            OpenAIRequestLowerer::new(config),
            Arc::new(PooledTokenizer::new(Arc::new(PrefixTokenizer), 2, 2)),
        )))
    }

    async fn post(body: Value) -> (StatusCode, Value) {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/tokenize")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body =
            serde_json::from_slice(&to_bytes(response.into_body(), 64 * 1024).await.unwrap())
                .unwrap();
        (status, body)
    }

    #[tokio::test]
    async fn prompt_tokenization_preserves_batch_shape_and_special_token_choice() {
        let (status, body) = post(json!({
            "prompt": ["one two", ""],
            "add_special_tokens": false
        }))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["tokens"], json!([[7, 7], []]));
        assert_eq!(body["count"], json!([2, 0]));

        let (_, body) = post(json!({"prompt": "one"})).await;
        assert_eq!(body["tokens"], json!([1, 7]));
    }

    #[tokio::test]
    async fn chat_tokenization_applies_the_template_without_generation_limits() {
        let (status, body) = post(json!({
            "messages": [{"role": "user", "content": "hello"}],
            "max_completion_tokens": 10_000
        }))
        .await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            body["tokens"]
                .as_array()
                .is_some_and(|tokens| !tokens.is_empty())
        );
        assert_ne!(body["tokens"][0], json!(1));
        assert_eq!(
            body["count"],
            json!(body["tokens"].as_array().unwrap().len())
        );
    }
}
