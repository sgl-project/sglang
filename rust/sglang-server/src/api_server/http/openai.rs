//! The OpenAI HTTP routes: chat completions, completions and score through
//! [`OpenAiService`], and `/v1/tokenize` and `/v1/detokenize` (+ the
//! unversioned aliases), Python's `OpenAIServingTokenize` /
//! `OpenAIServingDetokenize`. Encoding runs on the tokenizer pool and decoding
//! on a detok shard, never on these I/O threads.

use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode, header::CONTENT_TYPE},
    response::{IntoResponse, Response},
    routing::post,
};
use futures::StreamExt;
use futures::future::try_join_all;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use sglang_api_types::api::v1 as api;
use sglang_processor::openai::Reply;

use super::app::AppState;
use super::auth::DeclaredRoute;
use crate::api_server::core::CoreError;
use crate::api_server::log::ReceivedAs;
use crate::api_server::openai::{OpenAiService, Served, prompt_ids};
use crate::message::types::{OneOrMany, TokenIds};
use crate::utils::response::error_response;

/// The routes this module owns, mounted by `api_server::serve`.
pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .declared_route("/v1/chat/completions", post(chat))
        .declared_route("/v1/completions", post(completions))
        .declared_route("/v1/score", post(score))
        .declared_route("/v1/tokenize", post(tokenize))
        .declared_route("/tokenize", post(tokenize))
        .declared_route("/v1/detokenize", post(detokenize))
        .declared_route("/detokenize", post(detokenize))
}

impl AppState {
    pub(in crate::api_server::http) fn openai_service(&self) -> OpenAiService<'_> {
        OpenAiService {
            core: &self.core,
            server_args: &self.server_args,
            generation_policy: &self.generation_policy,
            generate_hook: self.generate_hook.as_ref(),
            openai: &self.openai,
        }
    }
}

fn served_response(served: Served) -> Response {
    match served {
        Served::Reply(reply) => reply_response(reply),
        Served::Events(events) => (
            [(CONTENT_TYPE, "text/event-stream; charset=utf-8")],
            Body::from_stream(events.map(Ok::<_, Infallible>)),
        )
            .into_response(),
    }
}

fn reply_response(reply: Reply) -> Response {
    let status = StatusCode::from_u16(reply.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, [(CONTENT_TYPE, "application/json")], reply.body).into_response()
}

async fn chat(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Bytes) -> Response {
    let service = state.openai_service();
    served_response(service.chat_completions(&headers, &body).await)
}

async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let service = state.openai_service();
    served_response(service.completions(&headers, &body).await)
}

async fn score(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Result<Json<api::OpenAiScoreRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => return reply_response(Reply::error(400, &rejection.body_text())),
    };
    let received = ReceivedAs::Typed(state.core.logged_headers(&headers));
    reply_response(state.openai_service().score(&request, received).await)
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

/// A failed core operation: a rejected input is the client's 400, and a full
/// queue the 503 every other overloaded route answers (typed by its status,
/// as Python types an `HTTPException`); anything else is the 500 Python's
/// catch-all returns.
fn core_error(error: CoreError, operation: &str) -> Response {
    match error {
        CoreError::InvalidArgument(message) => bad_request(message),
        error @ CoreError::Overloaded(_) => {
            let status = error.http_status();
            error_body(status, status.as_str(), error.to_string())
        }
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
    let max_model_len = state.server_args.tokenizer_model_max_length;
    let one = |tokens: TokenIds| {
        json!({
            "tokens": tokens,
            "count": tokens.len(),
            "max_model_len": max_model_len,
        })
    };
    let (texts, add_special_tokens) = match (body.prompt, body.messages) {
        (Some(prompt), None) => (prompt, body.add_special_tokens),
        // Encoded as on the chat path (Python's `add_special_tokens=False`).
        (None, Some(messages)) => {
            let chat = chat_body(messages, body.rest);
            // The ids chat would send, from the lowered path where chat takes it.
            let bytes = serde_json::to_vec(&chat).expect("JSON values serialize");
            return match prompt_ids(&state.openai, bytes.into()).await {
                Ok(ids) => Json(one(ids)).into_response(),
                Err(reply) => reply_response(reply),
            };
        }
        _ => return invalid_body("Exactly one of 'prompt' or 'messages' must be provided."),
    };
    let response = match texts {
        OneOrMany::One(text) => match state.core.tokenize(text, add_special_tokens).await {
            Ok(tokens) => one(tokens),
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

/// The chat request for `messages`, like Python's
/// `TokenizeRequest.to_chat_completion_request`: every other tokenize field
/// except `prompt` and `add_special_tokens` carries over.
fn chat_body(messages: Value, mut fields: Map<String, Value>) -> Value {
    fields.insert("messages".into(), messages);
    // Python's `TokenizeRequest.model` default; the chat request requires one.
    fields
        .entry("model")
        .or_insert_with(|| Value::from("default"));
    Value::Object(fields)
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

/// The OpenAI error payload.
pub(crate) fn error_payload(code: StatusCode, message: impl Into<String>) -> serde_json::Value {
    let message = message.into();
    let error_type = if code == StatusCode::UNAUTHORIZED {
        "AuthenticationError"
    } else if code.is_server_error() {
        "InternalServerError"
    } else {
        "BadRequestError"
    };
    serde_json::json!({
        "error": {
            "object": "error",
            "message": message,
            "type": error_type,
            "param": null,
            "code": code.as_u16(),
        }
    })
}

/// Form an OpenAI error response: unary → `code` plus the JSON `body`,
/// streaming → 200 with one SSE error frame + `[DONE]`.
pub(crate) fn openai_error(code: StatusCode, message: impl Into<String>, stream: bool) -> Response {
    error_response(code, error_payload(code, message), stream)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::test_utils::*;
    use super::*;
    use crate::api_server::handlers::CoreHandle;
    use crate::api_server::openai::ChatFormatter;
    use crate::message::config::ServerArgs;
    use axum::http::Request;
    use sglang_processor::dynamo_tokenizers::Tokenizer;

    /// A template's detected toggle decides reasoning before the parser's
    /// default: one off by default answers in `content`.
    #[tokio::test]
    async fn detected_template_toggle_decides_reasoning() {
        let template = "{%- set enable_thinking = enable_thinking | default(false) -%}\
            {% for m in messages %}{{ m.content }}{% endfor %}";
        let formatter = checkpoint_formatter(&json!({}), template);
        let args = ServerArgs {
            served_model_name: "model".into(),
            reasoning_parser: Some("qwen3".into()),
            ..Default::default()
        };
        let state =
            answering_app_state_with_item(Arc::new(args), formatter, Some(tokenizer()), |rid| {
                chunk(rid, "Hello!", true)
            });
        let app = super::routes().with_state(state);
        let body = json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]});
        let value = body_json(post_json(app, "/v1/chat/completions", body).await).await;
        let message = &value["choices"][0]["message"];
        assert_eq!(message["content"], "Hello!");
        assert!(message["reasoning_content"].is_null(), "{value}");
    }

    /// A template without a detected reasoning toggle reasons by the parser's
    /// default: Qwen3's thinks unless `enable_thinking` is false.
    #[tokio::test]
    async fn undetected_template_reasons_by_the_parser_default() {
        let template = "{% for m in messages %}{{ m.content }}{% endfor %}<think>\n";
        let formatter = checkpoint_formatter(&json!({}), template);
        let args = ServerArgs {
            served_model_name: "model".into(),
            reasoning_parser: Some("qwen3".into()),
            ..Default::default()
        };
        let state =
            answering_app_state_with_item(Arc::new(args), formatter, Some(tokenizer()), |rid| {
                chunk(rid, "work</think>answer", true)
            });
        let app = super::routes().with_state(state);
        let body = json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]});
        let value = body_json(post_json(app, "/v1/chat/completions", body).await).await;
        let message = &value["choices"][0]["message"];
        assert_eq!(
            (
                message["reasoning_content"].as_str(),
                message["content"].as_str()
            ),
            (Some("work"), Some("answer"))
        );
    }

    /// The DeepSeek-V4.1 checkpoint the processor's `openai_parity` fixture
    /// pins, from the HF cache CI fills; `None` skips the test elsewhere.
    fn deepseek_v41() -> Option<(ChatFormatter, Tokenizer)> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../sglang-processor/tests/fixtures/openai_parity/deepseek-v4.1-flash.json"
        )))
        .unwrap();
        let path = sglang_processor::resolve_model_file(
            fixture["model"].as_str().unwrap(),
            fixture["revision"].as_str(),
            "tokenizer.json",
        );
        let Some(path) = path else {
            assert!(
                std::env::var_os("CI").is_none(),
                "CI caches the tokenizer: tests/scripts/cache_deepseek_files.sh"
            );
            return None;
        };
        let json = std::fs::read_to_string(&path).unwrap();
        let formatter = ChatFormatter::deepseek_v4(true, Some(&json)).unwrap();
        Some((formatter, Tokenizer::from_file(&path).unwrap()))
    }

    /// The engine's last chunk for `rid`: `text`'s ids, ended by `finish`.
    fn ids_chunk(
        rid: &str,
        tokenizer: &Tokenizer,
        text: &str,
        finish: serde_json::Value,
    ) -> crate::message::response::ResponseItem {
        let ids = tokenizer.encode(text).unwrap().token_ids().to_vec();
        crate::message::response::ResponseItem::Done(crate::message::response::ChunkEvent {
            rid: rid.into(),
            text: text.into(),
            completion_tokens: ids.len() as u64,
            token_ids: ids.into_iter().map(i64::from).collect(),
            prompt_tokens: 5,
            finish_reason: Some(serde_json::from_value(finish).unwrap()),
            ..Default::default()
        })
    }

    /// A continuation that already closed its think block generates content.
    #[tokio::test]
    async fn continuation_after_think_end_is_content() {
        let Some((formatter, tokenizer)) = deepseek_v41() else {
            return;
        };
        let body = json!({
            "model": "model",
            "messages": [
                {"role": "user", "content": "What is 17*23?"},
                {"role": "assistant", "content": "17*23</think>The answer is"}
            ],
            "continue_final_message": true,
            "chat_template_kwargs": {"thinking": true}
        });
        let args = ServerArgs {
            served_model_name: "model".into(),
            reasoning_parser: Some("deepseek-v41".into()),
            ..Default::default()
        };
        let stop = json!({"type": "stop"});
        let answer = tokenizer.clone();
        let state =
            answering_app_state_with_item(Arc::new(args), formatter, Some(tokenizer), move |rid| {
                ids_chunk(rid, &answer, " 391.", stop.clone())
            });
        let app = super::routes().with_state(state);
        let response = body_json(post_json(app, "/v1/chat/completions", body).await).await;
        let message = &response["choices"][0]["message"];
        assert_eq!(message["content"], " 391.");
        assert!(message["reasoning_content"].is_null());
    }

    /// `/v1/score` concatenates query and item (item first when asked),
    /// submits one zero-token generate per item carrying the labels, reads
    /// the label logprobs back, and answers Python's shape; refusals and an
    /// empty `items` never reach the scheduler.
    #[tokio::test]
    async fn score_route_scores_each_item_by_label_logprobs() {
        use crate::api_server::handlers::{CoreConfig, CoreMetadata};
        use crate::message::request::RequestKind;
        use crate::message::response::{ChunkEvent, ChunkExtras, ResponseItem};
        use crate::tokenizer_manager::wiring::TmEvent;
        use axum::body::to_bytes;
        use axum::http::{Request as HttpRequest, header::CONTENT_TYPE};
        use tower::ServiceExt;
        let server_args = super::super::test_utils::server_args();
        let (intake_tx, intake_rx) = flume::unbounded();
        let (abort_tx, _abort_rx) = flume::unbounded();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: Default::default(),
                startup_ready: true,
                is_disaggregation: false,
                mm_limits: Default::default(),
                multimodal: false,
                metrics: None,
                request_log: None,
                generate_validator: None,
                abort_hook: None,
                peer_endpoints: Vec::new(),
                metadata: CoreMetadata::from(server_args.as_ref()),
            },
        );
        let state = super::super::test_utils::app_state(core);
        let app = routes().with_state(state.clone());
        let response = tokio::spawn(
            app.clone().oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/v1/score")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"query": "Q", "items": ["a", "b"], "label_token_ids": [5, 7],
                            "apply_softmax": true, "return_token_logprobs": true,
                            "item_first": true, "rid": "r"}"#,
                    ))
                    .unwrap(),
            ),
        );
        for expected_text in ["aQ", "bQ"] {
            let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap()
            else {
                panic!("expected a generation");
            };
            assert!(admission.try_accept());
            let RequestKind::Generate(generate) = &request.kind else {
                panic!("expected a generate request");
            };
            assert_eq!(generate.text.as_deref(), Some(expected_text));
            assert_eq!(generate.sampling_params.max_new_tokens, Some(0));
            assert!(generate.return_logprob);
            assert_eq!(
                generate.token_ids_logprob.as_deref(),
                Some([5, 7].as_slice())
            );
            assert!(generate.rid.client_facing().starts_with("r_"));
            let rid = request.rid.clone();
            request
                .sink
                .try_send(ResponseItem::Done(ChunkEvent {
                    rid,
                    prompt_tokens: 3,
                    extras: Some(Box::new(ChunkExtras {
                        out_tid_val: vec![-1.0, -2.0],
                        out_tid_idx: vec![7, 5],
                        out_tid_lens: vec![2],
                        ..Default::default()
                    })),
                    ..Default::default()
                }))
                .unwrap();
        }
        let response = response.await.unwrap().unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["object"], "scoring");
        assert_eq!(body["model"], "default");
        assert_eq!(body["usage"]["prompt_tokens"], 6);
        let scores = body["scores"].as_array().unwrap();
        assert_eq!(scores.len(), 2);
        let row = scores[0].as_array().unwrap();
        // label 5 -> -2.0, label 7 -> -1.0: softmax favours 7.
        assert!(row[1].as_f64().unwrap() > row[0].as_f64().unwrap());
        assert!((row[0].as_f64().unwrap() + row[1].as_f64().unwrap() - 1.0).abs() < 1e-9);
        assert_eq!(body["token_logprobs"][0], json!([-2.0, -1.0]));

        for (body, message) in [
            (
                r#"{"query": "Q", "items": ["a"]}"#,
                "label_token_ids is required",
            ),
            (
                r#"{"query": "Q", "items": ["a"], "label_token_ids": [[1], [2]]}"#,
                "one list per item",
            ),
            (
                r#"{"query": "Q", "items": ["a"], "label_token_ids": [1, 1]}"#,
                "duplicate",
            ),
            (
                r#"{"query": [1], "items": ["a"], "label_token_ids": [1]}"#,
                "Invalid combination",
            ),
            (
                r#"{"query": "Q", "items": ["a"], "label_token_ids": [1], "return_pooled_hidden_states": true}"#,
                "CausalLM",
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    HttpRequest::builder()
                        .method("POST")
                        .uri("/v1/score")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
            let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
            let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert!(
                json["message"].as_str().unwrap().contains(message),
                "{body}: {json}"
            );
            assert!(
                intake_rx.try_recv().is_err(),
                "{body}: refused before the scheduler"
            );
        }
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/v1/score")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        r#"{"query": "Q", "items": [], "label_token_ids": [1]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1 << 16).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["scores"], json!([]));
    }

    /// The common StatusCode→error helper follows `error_response`'s shape:
    /// unary requests get the JSON error with its status; a committed stream gets
    /// 200 + one SSE error frame + `[DONE]`, and the frame carries the OpenAI
    /// error fields (`type`, `param`, `code`) that the SDKs dispatch on.
    #[tokio::test]
    async fn openai_error_response_covers_unary_and_sse() {
        let unary = openai_error(StatusCode::BAD_REQUEST, "bad input", false);
        assert_eq!(unary.status(), StatusCode::BAD_REQUEST);
        let value = body_json(unary).await;
        assert_eq!(value["error"]["message"], "bad input");
        assert_eq!(value["error"]["type"], "BadRequestError");
        assert_eq!(value["error"]["code"], 400);
        assert!(value["error"]["param"].is_null());

        let streamed = openai_error(StatusCode::BAD_REQUEST, "bad input", true);
        assert_eq!(streamed.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(streamed.into_body(), 64 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let frame = text
            .split("\n\n")
            .next()
            .unwrap()
            .strip_prefix("data: ")
            .unwrap();
        let frame: serde_json::Value = serde_json::from_str(frame).unwrap();
        assert_eq!(frame["error"]["message"], "bad input");
        assert_eq!(frame["error"]["type"], "BadRequestError");
        assert!(text.contains("[DONE]"));
    }

    /// A closed tm inbox (shutdown) is a 503, for a stream too: Python awaits the
    /// first chunk before it starts the stream.
    #[tokio::test]
    async fn completions_on_a_closed_intake_is_a_503() {
        for stream in [false, true] {
            let app = routes().with_state(app_state(frontend_closed()));
            let body = json!({"model": "model", "prompt": "hi", "stream": stream});
            let response = post_json(app, "/v1/completions", body).await;
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(body_json(response).await["code"], 503);
        }
    }

    #[tokio::test]
    async fn basic_openai_router_excludes_responses_api() {
        let app = routes().with_state(app_state(core_handle()));
        let response = post_json(app, "/v1/responses", json!({"input": "hi"})).await;
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    /// The OpenAI routes log their requests with the whitelisted headers of the
    /// HTTP request, as Python's tokenizer manager does, and from level 2 the
    /// OpenAI request first, as Python's `serving_base` does.
    #[tokio::test]
    async fn openai_request_lines_carry_whitelisted_headers() {
        let log = crate::api_server::log::RequestLog::for_test(
            Some(crate::message::config::RequestLogConfig {
                max_length: 1 << 30,
                skip_names: vec!["text".into()],
                out_skip_names: vec!["text".into(), "output_ids".into()],
                json: false,
                class_name: "GenerateReqInput".into(),
                fields: vec![("rid".into(), "None".into())],
                batch_item_fields: Vec::new(),
                batch_shared_fields: Vec::new(),
                headers: vec!["x-sgl-routing-key".into()],
                exceeded_ms: -1,
                openai_requests: true,
                chat_completion_fields: vec![("model".into(), None)],
                completion_fields: vec![("model".into(), None)],
                stdout: true,
                files: Vec::new(),
            }),
            false,
        );
        let app = routes().with_state(answering_app_state_logged(log.clone()));
        let messages = json!([{"role": "user", "content": "hi"}]);
        for (path, body) in [
            ("/v1/completions", json!({"model": "model", "prompt": "hi"})),
            (
                "/v1/chat/completions",
                json!({"model": "model", "messages": messages}),
            ),
        ] {
            let req = Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .header("x-sgl-routing-key", "conv-1")
                .body(Body::from(body.to_string()))
                .unwrap();
            let response = oneshot(app.clone(), req).await;
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            let lines = log.logged_lines();
            assert_eq!(lines.len(), 3, "{path}: {lines:?}");
            assert!(
                lines[0].starts_with("Receive OpenAI: obj={'model': 'model'}"),
                "{path}"
            );
            assert!(
                lines[1].starts_with("Receive: obj=GenerateReqInput(rid="),
                "{path}"
            );
            assert!(
                lines[2].starts_with("Finish: obj=GenerateReqInput(rid="),
                "{path}"
            );
            for line in &lines {
                assert!(
                    line.contains(", headers={'x-sgl-routing-key': 'conv-1'}"),
                    "{path}: {line}"
                );
            }
        }

        // A request the processor refuses is not logged.
        let body = json!({"model": "model", "prompt": ["", ""]});
        let response = post_json(app.clone(), "/v1/completions", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(log.logged_lines().len(), 0);
    }
}

#[cfg(test)]
mod tokenize_tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use super::super::test_utils::{body_json, post_json};
    use super::*;
    use crate::api_server::handlers::{CoreConfig, CoreHandle, CoreMetadata};
    use crate::message::config::ServerArgs;
    use crate::message::request::{RequestKind, TokenizeInput};
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
                    // What intake answers when the tokenizer pool is full.
                    RequestKind::Tokenize(TokenizeInput::Text { text, .. }) if text == "full" => {
                        ResponseItem::Error(crate::utils::error::Error::QueueFull)
                    }
                    RequestKind::Tokenize(TokenizeInput::Text {
                        text,
                        add_special_tokens,
                    }) => {
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
                    multimodal: false,
                    peer_endpoints: Vec::new(),
                    metadata: CoreMetadata::from(server_args.as_ref()),
                    metrics: None,
                    request_log: None,
                    generate_validator: None,
                    abort_hook: None,
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
            generate_hook: None,
            openai: Default::default(),
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

    /// A full tokenizer pool is overload, not an internal error: the 503 the
    /// generate routes answer, in this endpoint's flat error body.
    #[tokio::test]
    async fn full_tokenizer_pool_is_a_503() {
        for prompt in [json!("full"), json!(["a", "full"])] {
            let (status, error) = call("/v1/tokenize", json!({"prompt": prompt})).await;
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{prompt}");
            assert_eq!(
                error,
                json!({
                    "object": "error",
                    "message": "to_scheduler channel full",
                    "type": "503",
                    "param": null,
                    "code": 503,
                }),
                "{prompt}"
            );
        }
    }
}
