//! `/v1/tokenize` and `/v1/detokenize` (+ the unversioned aliases), Python's
//! `OpenAIServingTokenize` / `OpenAIServingDetokenize`: the same request and
//! response shapes and the same flat error body. Encoding runs on the
//! tokenizer pool and decoding on a detok shard, never on these I/O threads.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use futures::future::try_join_all;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::chat::{ChatBody, ChatRequest};
use super::template::{RenderedChat, render_chat};
use super::{AppState, contains_media};
use crate::api_server::core::CoreError;
use crate::message::types::{OneOrMany, TokenIds};

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/tokenize", post(tokenize))
        .route("/tokenize", post(tokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/detokenize", post(detokenize))
}

fn default_true() -> bool {
    true
}

/// Python's `TokenizeRequest`. Unknown keys are kept for the chat-template
/// path, as its `extra="allow"` does.
#[derive(Deserialize)]
struct TokenizeBody {
    #[serde(default)]
    prompt: Option<OneOrMany<String>>,
    #[serde(default)]
    messages: Option<Value>,
    #[serde(default = "default_true")]
    add_special_tokens: bool,
    #[serde(flatten)]
    rest: Map<String, Value>,
}

/// Python's `DetokenizeRequest`.
#[derive(Deserialize)]
struct DetokenizeBody {
    tokens: OneOrMany<TokenIds>,
    #[serde(default = "default_true")]
    skip_special_tokens: bool,
}

/// Python's `ErrorResponse`, which these endpoints return flat rather than
/// under an `error` key.
fn error_body(status: StatusCode, error_type: &str, message: impl Into<String>) -> Response {
    let body = json!({
        "object": "error",
        "message": message.into(),
        "type": error_type,
        "param": null,
        "code": status.as_u16(),
    });
    (status, Json(body)).into_response()
}

/// A body that fails schema validation: what Python's
/// `validation_exception_handler` answers for a pydantic error.
fn invalid_body(message: impl Into<String>) -> Response {
    error_body(StatusCode::BAD_REQUEST, "Bad Request", message)
}

fn bad_request(message: impl Into<String>) -> Response {
    error_body(StatusCode::BAD_REQUEST, "BadRequestError", message)
}

/// A failed core operation: a rejected input is the client's 400; anything
/// else is the 500 Python's catch-all returns.
fn core_error(error: CoreError, operation: &str) -> Response {
    match error {
        CoreError::InvalidArgument(message) => bad_request(message),
        other => error_body(
            StatusCode::INTERNAL_SERVER_ERROR,
            "InternalServerError",
            format!("Internal server error during {operation}: {other}"),
        ),
    }
}

async fn tokenize(
    State(state): State<Arc<AppState>>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => return invalid_body(rejection.body_text()),
    };
    let body: TokenizeBody = match serde_json::from_value(body) {
        Ok(body) => body,
        Err(error) => return invalid_body(error.to_string()),
    };
    let (texts, add_special_tokens) = match (body.prompt, body.messages) {
        (Some(prompt), None) => (prompt, body.add_special_tokens),
        // The rendered template owns its special tokens, as on the chat path
        // (Python encodes it with `add_special_tokens=False`).
        (None, Some(messages)) => {
            let rendered = match render_chat_prompt(&state, messages, body.rest) {
                Ok(rendered) => rendered,
                Err((error_type, message)) => {
                    return error_body(StatusCode::BAD_REQUEST, error_type, message);
                }
            };
            match &state.prompt_encoder {
                // Encoded as the chat path does, keeping segments and the continuation apart.
                Some(encoder) => {
                    return match encoder.encode_blocking(rendered).await {
                        Ok(tokens) => Json(json!({
                            "tokens": tokens,
                            "count": tokens.len(),
                            "max_model_len": state.server_args.tokenizer_model_max_length,
                        }))
                        .into_response(),
                        Err(message) => {
                            error_body(StatusCode::BAD_REQUEST, "BadRequestError", message)
                        }
                    };
                }
                None => (
                    OneOrMany::One(rendered.into_text()),
                    false, /* add_special_tokens */
                ),
            }
        }
        _ => return invalid_body("Exactly one of 'prompt' or 'messages' must be provided."),
    };
    let max_model_len = state.server_args.tokenizer_model_max_length;
    let response = match texts {
        OneOrMany::One(text) => match state.core.tokenize(text, add_special_tokens).await {
            Ok(tokens) => json!({
                "tokens": tokens,
                "count": tokens.len(),
                "max_model_len": max_model_len,
            }),
            Err(error) => return core_error(error, "tokenization"),
        },
        OneOrMany::Many(texts) => {
            let calls = texts
                .into_iter()
                .map(|text| state.core.tokenize(text, add_special_tokens));
            match try_join_all(calls).await {
                Ok(tokens) => {
                    let count: Vec<usize> = tokens.iter().map(Vec::len).collect();
                    json!({ "tokens": tokens, "count": count, "max_model_len": max_model_len })
                }
                Err(error) => return core_error(error, "tokenization"),
            }
        }
    };
    Json(response).into_response()
}

/// Render `messages` through the chat template, like Python's
/// `TokenizeRequest.to_chat_completion_request`: every other tokenize field
/// except `prompt` and `add_special_tokens` carries over to the chat request.
/// A failure is a 400, returned as its error type and message.
fn render_chat_prompt(
    state: &AppState,
    messages: Value,
    mut fields: Map<String, Value>,
) -> Result<RenderedChat, (&'static str, String)> {
    const INVALID_BODY: &str = "Bad Request";
    const BAD_REQUEST: &str = "BadRequestError";
    fields.insert("messages".into(), messages);
    // Python's `TokenizeRequest.model` default; the chat request requires one.
    fields
        .entry("model")
        .or_insert_with(|| Value::from("default"));
    let ChatBody {
        request:
            ChatRequest {
                request,
                chat_template_kwargs,
                ..
            },
        raw: body,
    } = ChatBody::deserialize(Value::Object(fields))
        .map_err(|error| (INVALID_BODY, error.to_string()))?;
    if request.messages.is_empty() {
        return Err((BAD_REQUEST, "messages cannot be empty".into()));
    }
    if serde_json::to_value(&request.messages).is_ok_and(|messages| contains_media(&messages)) {
        return Err((
            BAD_REQUEST,
            "image, audio, video, and file message content is not supported".into(),
        ));
    }
    let Some(formatter) = &state.chat_formatter else {
        return Err((BAD_REQUEST, "this model has no usable chat template".into()));
    };
    render_chat(formatter, &body, &request, chat_template_kwargs.as_ref())
        .map_err(|error| (BAD_REQUEST, format!("chat template render failed: {error}")))
}

async fn detokenize(
    State(state): State<Arc<AppState>>,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    let body = match body {
        Ok(Json(body)) => body,
        Err(rejection) => return invalid_body(rejection.body_text()),
    };
    let body: DetokenizeBody = match serde_json::from_value(body) {
        Ok(body) => body,
        Err(error) => return invalid_body(error.to_string()),
    };
    let text = match body.tokens {
        // Python answers an empty list without decoding.
        OneOrMany::One(tokens) if tokens.is_empty() => Value::from(""),
        OneOrMany::One(tokens) => match state
            .core
            .detokenize(tokens, body.skip_special_tokens)
            .await
        {
            Ok(text) => Value::from(text),
            Err(error) => return core_error(error, "detokenization"),
        },
        OneOrMany::Many(lists) => {
            let calls = lists
                .into_iter()
                .map(|tokens| state.core.detokenize(tokens, body.skip_special_tokens));
            match try_join_all(calls).await {
                Ok(texts) => Value::from(texts),
                Err(error) => return core_error(error, "detokenization"),
            }
        }
    };
    Json(json!({ "text": text })).into_response()
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use super::super::test_utils::{body_json, post_json};
    use super::*;
    use crate::api_server::core::{CoreConfig, CoreHandle, CoreMetadata};
    use crate::message::config::ServerArgs;
    use crate::message::request::RequestKind;
    use crate::message::response::ResponseItem;
    use crate::tokenizer_manager::wiring::TmEvent;

    /// A core whose runtime answers tokenize with `[0?, len(word)...]` (the 0
    /// standing in for an auto-added BOS) and detokenize with a description
    /// of what it was asked, so the tests see exactly what reached it.
    fn answering_state() -> Arc<AppState> {
        let (intake_tx, intake_rx) = flume::unbounded::<TmEvent>();
        tokio::spawn(async move {
            while let Ok(TmEvent::Intake { request, admission }) = intake_rx.recv_async().await {
                assert!(admission.try_accept());
                let item = match &request.kind {
                    RequestKind::Tokenize {
                        text,
                        add_special_tokens,
                    } => {
                        let bos = add_special_tokens.then_some(0);
                        let words = text.split_whitespace().map(|w| w.len() as i64);
                        ResponseItem::TokenIds(bos.into_iter().chain(words).collect())
                    }
                    RequestKind::Detokenize {
                        token_ids,
                        skip_special_tokens,
                    } => ResponseItem::Data(format!("{token_ids:?} {skip_special_tokens}").into()),
                    _ => panic!("unexpected request kind"),
                };
                request.sink.try_send(item).unwrap();
            }
        });
        let server_args = Arc::new(ServerArgs {
            tokenizer_model_max_length: 131072,
            ..Default::default()
        });
        Arc::new(AppState {
            core: CoreHandle::new(
                intake_tx,
                flume::unbounded().0,
                CoreConfig {
                    response_capacity: 8,
                    response_activity: Default::default(),
                    startup_ready: true,
                    is_disaggregation: false,
                    mm_limits: Default::default(),
                    metadata: CoreMetadata::from(server_args.as_ref()),
                },
            ),
            generation_policy: crate::message::request::GenerationPolicy::from(
                server_args.as_ref(),
            ),
            server_args,
            chat_formatter: Some(
                sglang_processor::load_chat_formatter(
                    None,           /* config_file */
                    None,           /* model_path */
                    None,           /* model_type */
                    Some("chatml"), /* chat_template_arg */
                )
                .unwrap(),
            ),
            prompt_encoder: None,
        })
    }

    async fn call(path: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
        let app = routes().with_state(answering_state());
        let response = post_json(app, path, body).await;
        (response.status(), body_json(response).await)
    }

    #[tokio::test]
    async fn tokenize_returns_python_response_shape() {
        for path in ["/v1/tokenize", "/tokenize"] {
            let (status, body) = call(path, json!({"model": "m", "prompt": "a bcd"})).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(
                body.to_string(),
                r#"{"tokens":[0,1,3],"count":3,"max_model_len":131072}"#
            );
        }
        let (_, body) = call(
            "/v1/tokenize",
            json!({"prompt": ["a", "bb cc"], "add_special_tokens": false}),
        )
        .await;
        assert_eq!(body["tokens"], json!([[1], [2, 2]]));
        assert_eq!(body["count"], json!([1, 2]));
        let (_, body) = call("/v1/tokenize", json!({"prompt": []})).await;
        assert_eq!((&body["tokens"], &body["count"]), (&json!([]), &json!([])));

        // `messages` render through the chat template, whose output keeps
        // its own specials: encoded without auto-added ones.
        let (status, body) = call(
            "/v1/tokenize",
            json!({"messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_ne!(body["tokens"][0], 0, "no auto BOS on a rendered template");
        assert_eq!(body["count"], body["tokens"].as_array().unwrap().len());
    }

    #[tokio::test]
    async fn detokenize_returns_python_response_shape() {
        for path in ["/v1/detokenize", "/detokenize"] {
            let (status, body) = call(path, json!({"tokens": [1, 2]})).await;
            assert_eq!(status, StatusCode::OK);
            assert_eq!(body, json!({"text": "[1, 2] true"}));
        }
        let (_, body) = call(
            "/v1/detokenize",
            json!({"tokens": [[1], []], "skip_special_tokens": false}),
        )
        .await;
        assert_eq!(body, json!({"text": ["[1] false", "[] false"]}));
        let (_, body) = call("/v1/detokenize", json!({"tokens": []})).await;
        assert_eq!(body, json!({"text": ""}));
    }

    /// Bad bodies get Python's flat `ErrorResponse` with a 400.
    #[tokio::test]
    async fn invalid_requests_get_the_flat_python_error() {
        for (path, body) in [
            ("/v1/tokenize", json!({"model": "m"})),
            (
                "/v1/tokenize",
                json!({"prompt": "a", "messages": [{"role": "user", "content": "hi"}]}),
            ),
            ("/v1/tokenize", json!({"prompt": 3})),
            ("/v1/detokenize", json!({"tokens": [1, [2]]})),
            ("/v1/detokenize", json!({})),
        ] {
            let (status, error) = call(path, body.clone()).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(error["object"], "error", "{body}");
            assert_eq!(error["type"], "Bad Request", "{body}");
            assert_eq!(error["code"], 400, "{body}");
            assert!(error["param"].is_null(), "{body}");
        }
        let (status, error) = call("/v1/tokenize", json!({"messages": []})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(error["type"], "BadRequestError");
    }
}
