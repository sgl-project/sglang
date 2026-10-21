//! Chat completions through sglang-processor's port of Python's OpenAI layer.
//!
//! The processor lowers the chat body into a `/generate` body, which the
//! native route's decoder and the core serve, and builds the OpenAI response
//! from the `/generate` frames. Requests it does not reproduce go to `chat.rs`.

use std::convert::Infallible;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, StatusCode, header::CONTENT_TYPE};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value};
use sglang_processor::DynamoTokenizer;
use sglang_processor::dynamo_tokenizers::{DecodeResult, Tokenizer};
use sglang_processor::openai::{
    ChatModel, OpenAiHeaders, OpenAiSettings, OpenAiTokenizer, Reply, TokenPieces, lower_chat,
};

use super::super::request_headers::apply_header_overrides;
use super::chat::ChatBody;
use super::template::{render_body, render_chat};
use super::{AppState, ChatFormatter};
use crate::api_server::http::native_api::{
    RequestTiming, drain_unary, generate_requests, generation_event_stream,
};
use crate::message::config::ServerArgs;
use crate::message::request::check_prompt_copies;

/// The server facts and tokenizer the processor's chat layer reads.
pub struct OpenAiChat {
    settings: OpenAiSettings,
    model: ChatModel,
    tokenizer: Arc<DynamoTokenizer>,
    template_defaults: super::ChatTemplateKwargs,
    /// Token text and bytes for logprobs.
    pieces: Arc<dyn OpenAiTokenizer>,
}

struct LogprobPieces {
    tokenizer: Tokenizer,
    /// `None` for a tokenizer without `tokenizer.json` (tiktoken).
    pieces: Option<TokenPieces>,
}

impl OpenAiTokenizer for LogprobPieces {
    fn decode(&self, ids: &[u32]) -> Option<String> {
        match self.tokenizer.decode(ids, true).ok()? {
            DecodeResult::Complete(text) | DecodeResult::Partial(text) => Some(text),
        }
    }

    fn byte_level_bytes(&self, id: u32) -> Option<Vec<u8>> {
        self.pieces.as_ref()?.byte_level_bytes(id)
    }
}

impl OpenAiChat {
    pub fn new(
        server_args: &ServerArgs,
        formatter: &ChatFormatter,
        tokenizer: Arc<DynamoTokenizer>,
    ) -> Self {
        let defaults = &server_args.model_config.default_sampling_params;
        let generation_config: Map<String, Value> = [
            ("temperature", defaults.temperature.map(Value::from)),
            ("top_p", defaults.top_p.map(Value::from)),
            ("top_k", defaults.top_k.map(Value::from)),
            ("min_p", defaults.min_p.map(Value::from)),
            (
                "repetition_penalty",
                defaults.repetition_penalty.map(Value::from),
            ),
        ]
        .into_iter()
        .filter_map(|(key, value)| Some((key.to_owned(), value?)))
        .collect();
        let revision = server_args.revision.as_deref();
        let pieces = sglang_processor::resolve_model_file(
            &server_args.tokenizer_path,
            revision,
            "tokenizer.json",
        )
        .and_then(|file| std::fs::read(file).ok())
        .and_then(|json| serde_json::from_slice(&json).ok())
        .map(|json| TokenPieces::from_tokenizer_json(&json));
        let pieces = Arc::new(LogprobPieces {
            tokenizer: tokenizer.tokenizer().clone(),
            pieces,
        });
        Self {
            settings: OpenAiSettings {
                allow_auto_truncate: server_args.allow_auto_truncate,
                default_chat_template_kwargs: Some(
                    server_args
                        .default_chat_template_kwargs
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                ),
                incremental_streaming_output: server_args.incremental_streaming_output,
                reasoning_parser: server_args.reasoning_parser.clone(),
                stream_response_default_include_usage: server_args
                    .stream_response_default_include_usage,
                tool_call_parser: server_args.tool_call_parser.clone(),
                ..Default::default()
            },
            model: ChatModel {
                context_length: Some(server_args.model_config.context_len),
                generation_config,
                ..ChatModel::for_formatter(formatter)
            },
            tokenizer,
            template_defaults: server_args.default_chat_template_kwargs.clone(),
            pieces,
        }
    }

    /// The prompt ids Python's chat layer encodes for `request`.
    fn render(&self, formatter: &ChatFormatter, request: &Value) -> Option<Vec<u32>> {
        let rendered = match render_body(formatter, request.clone(), &self.template_defaults) {
            Some(rendered) => rendered.ok()?,
            None => {
                let ChatBody { request: chat, raw } = ChatBody::deserialize(request).ok()?;
                render_chat(formatter, raw, &chat.request, &self.template_defaults).ok()?
            }
        };
        let ids = rendered.encode(&*self.tokenizer).ok()?;
        Some(ids.into_iter().map(|id| id as u32).collect())
    }
}

/// Serve a chat body through the processor, or `None` to leave it to the
/// Dynamo-typed handler.
pub(super) async fn chat(
    state: &Arc<AppState>,
    headers: &HeaderMap,
    body: &Bytes,
) -> Option<Response> {
    state.openai_chat.as_ref()?;
    let header = |name: &str| headers.get(name)?.to_str().ok().map(str::to_owned);
    let routing_key = header("x-smg-routing-key");
    let sglext_ids = ["x-sglext-return-input-ids", "x-sglext-return-output-ids"]
        .into_iter()
        .any(|name| header(name).as_deref() == Some("1"));
    let (lowering, body) = (state.clone(), body.clone());
    // Rendering and tokenizing are CPU work, kept off the async workers.
    let (mut generate, mut responder) = tokio::task::spawn_blocking(move || {
        let openai = lowering.openai_chat.as_ref()?;
        let formatter = lowering.chat_formatter.as_ref()?;
        let headers = OpenAiHeaders {
            routing_key: routing_key.as_deref(),
            custom_labels: None,
            sglext_ids,
        };
        let render = |request: &Value| openai.render(formatter, request);
        lower_chat(
            &body,
            &headers,
            &openai.settings,
            &openai.model,
            render,
            Some(openai.pieces.clone()),
        )
        .ok()
    })
    .await
    .ok()??;

    // The core samples one sequence per request, so `n` choices are a batch of
    // the prompt, whose indexed frames are what Python's `n` streams.
    let n = generate["sampling_params"]["n"].as_u64().unwrap_or(1);
    let n = usize::try_from(n).unwrap_or(usize::MAX);
    let stream = generate["stream"] == true;
    if n > 1 {
        // Bounded before the prompt is copied: a huge `n` would overflow or
        // exhaust the allocation.
        let per_clone = generate["input_ids"]
            .as_array()
            .map_or(0, |ids| ids.len().saturating_mul(size_of::<Value>()));
        if let Err(error) = check_prompt_copies(per_clone, n) {
            return Some(error_response(StatusCode::BAD_REQUEST, &error.to_string()));
        }
        generate["sampling_params"]["n"] = 1.into();
        generate["input_ids"] = Value::Array(vec![generate["input_ids"].take(); n]);
        // Python's parallel sampling repeats the prompt's list-form fields
        // with it, and a rid list names the one prompt.
        for key in [
            "bootstrap_host",
            "bootstrap_port",
            "bootstrap_room",
            "extra_key",
            "cache_salt",
            "lora_path",
        ] {
            if let Some(Value::Array(items)) = generate.get_mut(key) {
                *items = items
                    .iter()
                    .cycle()
                    .take(items.len() * n)
                    .cloned()
                    .collect();
            }
        }
        if let Some(Value::Array(rids)) = generate.get_mut("rid")
            && rids.len() == 1
        {
            generate["rid"] = rids.remove(0);
        }
    }
    // Python's `adapted_request` takes the router's override headers.
    if state.server_args.enable_request_header_overrides
        && let Err(error) = apply_header_overrides(&mut generate, headers)
    {
        return Some(error_response(StatusCode::BAD_REQUEST, &error));
    }
    // Failures before the first frame answer unary even for a stream: Python
    // awaits the first chunk before it starts the stream.
    let (requests, is_batch) = match generate_requests(state, generate) {
        Ok(requests) => requests,
        Err((code, message, _)) => return Some(error_response(code, &message)),
    };
    let timing = RequestTiming::new();
    let calls = match state.core.generate_batch(requests).await {
        Ok(calls) => calls,
        Err(error) => return Some(error_response(error.http_status(), &error.to_string())),
    };

    if !stream {
        let drained = futures::future::join_all(calls.into_iter().map(|mut call| {
            let timing = timing.clone();
            async move {
                let rid = call.public_id().to_owned();
                drain_unary(&mut call, &rid, timing).await
            }
        }))
        .await;
        // One failed choice fails the request, as Python's batch does.
        if let Some((status, item)) = drained.iter().find(|(status, _)| *status != StatusCode::OK) {
            let body = serde_json::to_vec(item).expect("api.v1 types serialize to JSON");
            return Some(failed_response(*status, &body));
        }
        let items: Vec<_> = drained.into_iter().map(|(_, item)| item).collect();
        let body = match is_batch {
            true => serde_json::to_vec(&items),
            false => serde_json::to_vec(&items[0]),
        }
        .expect("api.v1 types serialize to JSON");
        return Some(match responder.unary(&body) {
            Some(reply) => reply_response(reply),
            None => (StatusCode::OK, body).into_response(),
        });
    }

    let calls = calls
        .into_iter()
        .map(|call| (call, timing.clone()))
        .collect();
    let incremental = state.server_args.incremental_streaming_output;
    let mut frames = Box::pin(generation_event_stream(calls, incremental, is_batch));
    // An error before the first frame answers the whole request, as on Python.
    let mut first = Vec::new();
    while first.is_empty() && !responder.done() {
        let Some(data) = frames.next().await else {
            break;
        };
        match responder.stream_data(data.as_bytes()) {
            Ok(events) => first = events,
            Err(reply) => return Some(reply_response(reply)),
        }
    }
    let events = async_stream::stream! {
        for event in first {
            yield Ok::<_, Infallible>(event);
        }
        while !responder.done() {
            let Some(data) = frames.next().await else {
                break;
            };
            for event in responder.stream_data(data.as_bytes()).unwrap_or_default() {
                yield Ok(event);
            }
        }
    };
    Some(
        (
            [(CONTENT_TYPE, "text/event-stream")],
            Body::from_stream(events),
        )
            .into_response(),
    )
}

/// A failed unary choice, as Python's chat layer answers its abort.
fn failed_response(status: StatusCode, body: &[u8]) -> Response {
    let message = serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|body| Some(body.pointer("/error/message")?.as_str()?.to_owned()))
        .unwrap_or_default();
    error_response(status, &message)
}

/// `OpenAIServingBase.create_error_response`: a 400 is the `ValueError` it
/// answers as a `BadRequestError`; any other status is an `HTTPException`,
/// whose status code is the type.
fn error_response(status: StatusCode, message: &str) -> Response {
    let code = status.as_u16();
    let err_type = match status {
        StatusCode::BAD_REQUEST => "BadRequestError".to_owned(),
        _ => code.to_string(),
    };
    let error = serde_json::json!({
        "object": "error",
        "message": message,
        "type": err_type,
        "param": null,
        "code": code,
    });
    (status, axum::Json(error)).into_response()
}

fn reply_response(reply: Reply) -> Response {
    let status = StatusCode::from_u16(reply.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, [(CONTENT_TYPE, "application/json")], reply.body).into_response()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use axum::Router;

    use super::super::test_utils::{
        answering_app_state_with, answering_app_state_with_item, body_json, chunk, post_json,
    };
    use super::*;
    use crate::message::response::ResponseItem;
    use crate::utils::error::Error;

    /// A word-level tokenizer; the mocked core ignores the prompt ids.
    fn tokenizer() -> Tokenizer {
        static TOKENIZER: std::sync::OnceLock<Tokenizer> = std::sync::OnceLock::new();
        let load = || {
            let path = std::env::temp_dir().join(format!("lowered-{}.json", std::process::id()));
            let model = json!({"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"});
            let config = json!({"model": model, "pre_tokenizer": {"type": "Whitespace"}});
            std::fs::write(&path, config.to_string()).unwrap();
            Tokenizer::from_file(path.to_str().unwrap()).unwrap()
        };
        TOKENIZER.get_or_init(load).clone()
    }

    async fn chat(body: Value) -> Value {
        let args = ServerArgs {
            served_model_name: "model".into(),
            reasoning_parser: Some("deepseek-v41".into()),
            ..Default::default()
        };
        let formatter = ChatFormatter::DeepSeekV41;
        let text = "17*23</think>391";
        let state = answering_app_state_with(Arc::new(args), formatter, Some(tokenizer()), text);
        let app = super::super::routes().with_state(state);
        body_json(post_json(app, "/v1/chat/completions", body).await).await
    }

    fn lowered_state(answer: impl Fn(&str) -> ResponseItem + Send + 'static) -> Router {
        lowered_state_with(ServerArgs::default(), answer)
    }

    fn lowered_state_with(
        args: ServerArgs,
        answer: impl Fn(&str) -> ResponseItem + Send + 'static,
    ) -> Router {
        let args = ServerArgs {
            served_model_name: "model".into(),
            reasoning_parser: Some("deepseek-v41".into()),
            ..args
        };
        let formatter = ChatFormatter::DeepSeekV41;
        let state =
            answering_app_state_with_item(Arc::new(args), formatter, Some(tokenizer()), answer);
        super::super::routes().with_state(state)
    }

    /// An `n` past the batch cap is a 400 before the prompt is copied, not a
    /// capacity-overflow panic.
    #[tokio::test]
    async fn oversized_n_is_rejected_before_copying_the_prompt() {
        let app = lowered_state(|rid| chunk(rid, "ok", true));
        let body = json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hi"}],
            "n": i64::MAX,
            "stream": true,
        });
        let response = post_json(app, "/v1/chat/completions", body).await;
        // Unary, as Python answers a failure before the first chunk.
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let value = body_json(response).await;
        assert_eq!(value["type"], "BadRequestError");
        let message = value["message"].as_str().unwrap();
        assert!(message.contains("exceeds the maximum"), "{message}");
    }

    /// With `n` choices, a prompt's one-element lists cover every choice, as
    /// Python's parallel sampling repeats them.
    #[tokio::test]
    async fn n_choices_repeat_the_prompts_list_fields() {
        let app = lowered_state(|rid| chunk(rid, "ok", true));
        let messages = json!([{"role": "user", "content": "hi"}]);
        let body = json!({
            "model": "model",
            "messages": messages,
            "n": 2,
            "rid": ["x"],
            "bootstrap_host": ["h"],
            "bootstrap_room": [5],
        });
        let response = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            body_json(response).await["choices"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    /// The router's override headers reach the request, as they reach
    /// Python's `adapted_request`.
    #[tokio::test]
    async fn override_headers_reach_the_request() {
        let args = ServerArgs {
            enable_request_header_overrides: true,
            ..Default::default()
        };
        let app = lowered_state_with(args, |rid| chunk(rid, "ok", true));
        let body = json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]});
        let request = axum::http::Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("x-override-rid", "routed")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap();
        let response = tower::ServiceExt::oneshot(app.clone(), request)
            .await
            .unwrap();
        assert_eq!(body_json(response).await["id"], "routed");

        let request = axum::http::Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("x-override-priority", "high")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap();
        let response = tower::ServiceExt::oneshot(app, request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// A scheduler failure keeps its 500, as Python's `HTTPException` does,
    /// and a validation failure is the 400 `BadRequestError`.
    #[tokio::test]
    async fn failed_choice_keeps_its_status() {
        let body = json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]});
        let app = lowered_state(|_| ResponseItem::Error(Error::Internal("boom".into())));
        let response = post_json(app, "/v1/chat/completions", body.clone()).await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let value = body_json(response).await;
        assert_eq!(
            (value["type"].as_str(), value["code"].as_u64()),
            (Some("500"), Some(500))
        );
        assert!(
            value["message"].as_str().unwrap().contains("boom"),
            "{value}"
        );

        let app = lowered_state(|_| ResponseItem::Error(Error::Validation("bad".into())));
        let response = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body_json(response).await["type"], "BadRequestError");
    }

    /// `separate_reasoning`, which the typed handler ignores, is Python's.
    #[tokio::test]
    async fn processor_serves_separate_reasoning() {
        let mut body = json!({
            "model": "model",
            "messages": [{"role": "user", "content": "17*23?"}],
            "chat_template_kwargs": {"thinking": true}
        });
        let split = chat(body.clone()).await;
        assert_eq!(split["choices"][0]["message"]["reasoning_content"], "17*23");
        assert_eq!(split["choices"][0]["message"]["content"], "391");
        body["separate_reasoning"] = false.into();
        let joined = chat(body).await;
        assert_eq!(
            joined["choices"][0]["message"]["content"],
            "17*23</think>391"
        );
    }

    /// Logprobs are served by the processor, which reports `reasoning_tokens`.
    #[tokio::test]
    async fn processor_serves_logprobs() {
        let messages = json!([{"role": "user", "content": "17*23?"}]);
        let body = json!({"model": "model", "messages": messages, "logprobs": true});
        assert!(chat(body).await["usage"].get("reasoning_tokens").is_some());
    }

    /// A request the processor does not reproduce reaches the typed handler.
    #[tokio::test]
    async fn required_tool_choice_falls_back() {
        let tool = json!({"type": "function", "function": {"name": "f", "parameters": {}}});
        let body = json!({
            "model": "model",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [tool],
            "tool_choice": "required"
        });
        let value = chat(body).await;
        assert_eq!(
            value["error"]["message"],
            "tool calls require --tool-call-parser"
        );
    }
    /// Launch defaults and request overrides must reach both rendering and response parsing.
    #[tokio::test]
    async fn processor_honors_server_thinking_defaults() {
        let args = ServerArgs {
            served_model_name: "model".into(),
            reasoning_parser: Some("deepseek-v41".into()),
            default_chat_template_kwargs: [("thinking".into(), Value::Bool(false))].into(),
            ..Default::default()
        };
        let state = answering_app_state_with(
            Arc::new(args),
            ChatFormatter::DeepSeekV41,
            Some(tokenizer()),
            "Hello!",
        );
        let body = json!({"model":"model", "messages":[{"role":"user","content":"Hi"}]});
        let value = body_json(
            post_json(
                super::super::routes().with_state(state.clone()),
                "/v1/chat/completions",
                body.clone(),
            )
            .await,
        )
        .await;
        assert_eq!(value["choices"][0]["message"]["content"], "Hello!");
        assert!(value["choices"][0]["message"]["reasoning_content"].is_null());
        let mut override_body = body;
        override_body["chat_template_kwargs"] = json!({"thinking":true});
        let value = body_json(
            post_json(
                super::super::routes().with_state(state),
                "/v1/chat/completions",
                override_body,
            )
            .await,
        )
        .await;
        assert_eq!(
            value["choices"][0]["message"]["reasoning_content"],
            "Hello!"
        );
    }
}
