//! Chat completions through sglang-processor's port of Python's OpenAI layer:
//! the chat body is lowered to a `/generate` body and the reply built from its
//! frames. Requests the processor does not reproduce go to `chat.rs`.

use std::convert::Infallible;
use std::sync::Arc;

use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, StatusCode, header::CONTENT_TYPE};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value};
use sglang_processor::openai::{ChatModel, OpenAiHeaders, OpenAiSettings, Reply, lower_chat};
use sglang_processor::{DynamoTokenizer, TemplateError};

use super::chat::ChatBody;
use super::template::{RenderedChat, render_chat};
use super::{AppState, ChatFormatter};
use crate::api_server::http::native_api::{
    GenerateItem, RequestTiming, drain_unary, generate_requests, generation_event_stream,
};
use crate::message::config::ServerArgs;
use crate::message::request::check_prompt_copies;
use crate::utils::error::Error;

/// The server facts and tokenizer the processor's chat layer reads.
pub struct OpenAiChat {
    settings: OpenAiSettings,
    model: ChatModel,
    tokenizer: Arc<DynamoTokenizer>,
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
        }
    }

    /// The prompt ids Python's chat layer encodes for `request`.
    fn render(
        &self,
        formatter: &ChatFormatter,
        request: &Value,
        defaults: &super::ChatTemplateKwargs,
    ) -> Option<Vec<u32>> {
        // Only the template needs the typed request, so a body the SGLang path
        // renders need not fit Dynamo's stricter type.
        let rendered = match formatter.render_request(request.clone(), defaults) {
            Err(TemplateError::NeedsRenderFallback { .. }) => {
                let ChatBody { request: chat, raw } = ChatBody::deserialize(request).ok()?;
                render_chat(formatter, raw, &chat.request, defaults).ok()?
            }
            rendered => {
                let (prompt, continuation) = rendered.ok()?;
                RenderedChat {
                    prompt,
                    continuation,
                }
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
        let defaults = &lowering.server_args.default_chat_template_kwargs;
        let render = |request: &Value| openai.render(formatter, request, defaults);
        lower_chat(
            &body,
            &headers,
            &openai.settings,
            &openai.model,
            render,
            None,
        )
        .ok()
    })
    .await
    .ok()??;

    let stream = generate["stream"] == true;
    if let Err(error) = copy_prompt_per_choice(&mut generate) {
        return Some(error_response(StatusCode::BAD_REQUEST, &error.to_string()));
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
        // One failed choice fails the request, as Python's batch does; the
        // unfinished choices are dropped, which cancels them.
        let drained = futures::future::try_join_all(calls.into_iter().map(|mut call| {
            let timing = timing.clone();
            async move {
                let rid = call.public_id().to_owned();
                match drain_unary(&mut call, &rid, timing).await {
                    (StatusCode::OK, item) => Ok(item),
                    failed => Err(failed),
                }
            }
        }))
        .await;
        let items = match drained {
            Ok(items) => items,
            Err((status, item)) => {
                let message = match &item {
                    GenerateItem::Error(failure) => {
                        failure.error.as_ref().map(|e| e.message.clone())
                    }
                    GenerateItem::Frame(_) => None,
                };
                return Some(error_response(status, &message.unwrap_or_default()));
            }
        };
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
    let frames = generation_event_stream(calls, incremental, is_batch);
    let mut frames = Box::pin(frames);
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
            [(CONTENT_TYPE, "text/event-stream; charset=utf-8")],
            Body::from_stream(events),
        )
            .into_response(),
    )
}

/// The core samples one sequence per request, so `n` choices are a batch of
/// the prompt, whose indexed frames are what Python's `n` streams.
fn copy_prompt_per_choice(generate: &mut Value) -> Result<(), Error> {
    let n = generate["sampling_params"]["n"].as_u64().unwrap_or(1);
    let n = usize::try_from(n).unwrap_or(usize::MAX);
    if n <= 1 {
        return Ok(());
    }
    // Bounded before the copy: a huge `n` would overflow or exhaust the allocation.
    let per_clone = generate["input_ids"]
        .as_array()
        .map_or(0, |ids| ids.len().saturating_mul(size_of::<Value>()));
    check_prompt_copies(per_clone, n)?;
    generate["sampling_params"]["n"] = 1.into();
    generate["input_ids"] = Value::Array(vec![generate["input_ids"].take(); n]);
    Ok(())
}

/// `OpenAIServingBase.create_error_response`: a 400 is a `BadRequestError`;
/// any other status is an `HTTPException`, typed by its status code.
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

    use sglang_processor::dynamo_tokenizers::Tokenizer;

    use super::super::test_utils::{
        answering_app_state_with_item, body_json, checkpoint_formatter, chunk, post_json,
    };
    use super::*;
    use crate::message::response::ResponseItem;

    /// A byte-level word tokenizer's `tokenizer.json`; id 3 is `é`'s bytes.
    fn tokenizer_path() -> &'static str {
        static PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        PATH.get_or_init(|| {
            let dir = std::env::temp_dir().join(format!("lowered-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let vocab = json!({"[UNK]": 0, "!": 1, "#": 2, "Ã©": 3});
            let model = json!({"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"});
            let config = json!({"model": model, "pre_tokenizer": {"type": "Whitespace"}});
            let path = dir.join("tokenizer.json");
            std::fs::write(&path, config.to_string()).unwrap();
            path.to_string_lossy().into_owned()
        })
    }

    /// The mocked core ignores the prompt ids.
    fn tokenizer() -> Tokenizer {
        Tokenizer::from_file(tokenizer_path()).unwrap()
    }

    async fn chat(body: Value) -> Value {
        let app = lowered_state(|rid| chunk(rid, "17*23</think>391", true));
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

    /// A failed choice answers at once, without waiting on an unfinished sibling.
    #[tokio::test]
    async fn failed_choice_does_not_wait_for_siblings() {
        let answered = std::sync::atomic::AtomicUsize::new(0);
        let app = lowered_state(move |rid| {
            match answered.fetch_add(1, std::sync::atomic::Ordering::Relaxed) {
                0 => ResponseItem::Error(Error::Internal("boom".into())),
                _ => chunk(rid, "still going", false),
            }
        });
        let body =
            json!({"model": "model", "messages": [{"role": "user", "content": "hi"}], "n": 2});
        let response = post_json(app, "/v1/chat/completions", body);
        let response = tokio::time::timeout(std::time::Duration::from_secs(5), response)
            .await
            .expect("answered");
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    /// A scheduler failure keeps its 500, as Python's `HTTPException` does.
    #[tokio::test]
    async fn failed_choice_keeps_its_status() {
        let body = json!({"model": "model", "messages": [{"role": "user", "content": "hi"}]});
        let app = lowered_state(|_| ResponseItem::Error(Error::Internal("boom".into())));
        let response = post_json(app, "/v1/chat/completions", body).await;
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
        let app = super::super::routes().with_state(state);
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
            default_chat_template_kwargs: [("thinking".into(), Value::Bool(false))].into(),
            ..Default::default()
        };
        let app = lowered_state_with(args, |rid| chunk(rid, "Hello!", true));
        let body = json!({"model":"model", "messages":[{"role":"user","content":"Hi"}]});
        let value =
            body_json(post_json(app.clone(), "/v1/chat/completions", body.clone()).await).await;
        assert_eq!(value["choices"][0]["message"]["content"], "Hello!");
        assert!(value["choices"][0]["message"]["reasoning_content"].is_null());
        let mut override_body = body;
        override_body["chat_template_kwargs"] = json!({"thinking":true});
        let value = body_json(post_json(app, "/v1/chat/completions", override_body).await).await;
        assert_eq!(
            value["choices"][0]["message"]["reasoning_content"],
            "Hello!"
        );
    }
}
