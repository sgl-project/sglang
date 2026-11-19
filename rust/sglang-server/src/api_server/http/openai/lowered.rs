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
use sglang_api_types::api::v1::{GenerateResponse, GenerateStreamError};
use sglang_processor::dynamo_tokenizers::{DecodeResult, Tokenizer};
use sglang_processor::openai::{
    ChatModel, ChatResponder, OpenAiHeaders, OpenAiSettings, OpenAiTokenizer, ROUTING_KEY_HEADER,
    Reply, TokenPieces, lower_chat,
};
use sglang_processor::{DynamoTokenizer, TemplateArgs, TemplateError};

use super::super::dp_relay::HintRoute;
use super::super::request_headers::apply_header_overrides;
use super::chat::ChatBody;
use super::template::RenderedChat;
use super::{AppState, ChatFormatter};
use crate::api_server::core::{CoreError, OpenAiRequest, ReceivedAs};
use crate::api_server::http::native_api::{
    GenerateItem, RequestState, RequestTiming, drain_unary_frame, generate_requests,
    generation_event_stream_with,
};
use crate::message::config::ServerArgs;
use crate::message::request::check_prompt_copies;
use crate::utils::error::Error;

/// The server facts and tokenizer the processor's chat layer reads.
pub struct OpenAiChat {
    settings: OpenAiSettings,
    model: ChatModel,
    tokenizer: Arc<DynamoTokenizer>,
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
                enable_cache_report: server_args.enable_cache_report,
                incremental_streaming_output: server_args.incremental_streaming_output,
                reasoning_parser: server_args.reasoning_parser.clone(),
                served_model_name: Some(server_args.served_model_name.clone()),
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
            pieces,
        }
    }

    /// The prompt ids Python's chat layer encodes for `request`.
    fn render(
        &self,
        formatter: &ChatFormatter,
        request: &Value,
        defaults: &super::ChatTemplateKwargs,
    ) -> Option<Vec<u32>> {
        if let Some(ids) = formatter.encode_request(request.clone(), defaults) {
            return ids.ok();
        }
        // Only the template needs the typed request, so a body the SGLang path
        // renders need not fit Dynamo's stricter type.
        let (prompt, continuation) = match formatter.render_request(request.clone(), defaults) {
            Err(TemplateError::NeedsRenderFallback { .. }) => {
                let ChatBody { request: chat, raw } = ChatBody::deserialize(request).ok()?;
                let args = TemplateArgs::new(&raw, defaults);
                formatter.render_template(&chat.request, &args).ok()?
            }
            rendered => rendered.ok()?,
        };
        let rendered = RenderedChat {
            prompt,
            continuation,
            ids: None,
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
    let routing_key = header(ROUTING_KEY_HEADER);
    let sglext_ids = ["x-sglext-return-input-ids", "x-sglext-return-output-ids"]
        .into_iter()
        .any(|name| header(name).as_deref() == Some("1"));
    let (mut generate, mut responder) = lower(state, body.clone(), routing_key, sglext_ids).await?;
    // Python logs the request once it validates, before adapting it.
    state
        .core
        .log_openai_request(OpenAiRequest::ChatCompletion, body, headers);

    let stream = generate["stream"] == true;
    if let Err(error) = copy_prompt_per_choice(&mut generate) {
        return Some(error_response(StatusCode::BAD_REQUEST, &error.to_string()));
    }
    // Python's `adapted_request` takes the router's override headers.
    if state.server_args.enable_request_header_overrides
        && let Err(error) = apply_header_overrides(&mut generate, headers)
    {
        // `apply_header_overrides` raises `HTTPException(400)`, typed by its status.
        return Some(typed_error_response(StatusCode::BAD_REQUEST, "400", &error));
    }
    // Below the override header, `X-Data-Parallel-Rank` names the DP rank.
    match HintRoute::ChatCompletions
        .header_rank(headers, state.server_args.enable_request_header_overrides)
    {
        Ok(Some(rank)) => generate["routed_dp_rank"] = rank.into(),
        Ok(None) => {}
        Err(error) => {
            return Some(typed_error_response(StatusCode::BAD_REQUEST, "400", &error));
        }
    }
    // Failures before the first frame answer unary even for a stream: Python
    // awaits the first chunk before it starts the stream.
    let (requests, is_batch) = match generate_requests(state, generate) {
        Ok(requests) => requests,
        Err((code, message, _)) => return Some(error_response(code, &message)),
    };
    let timing = RequestTiming::new();
    let request_states: Vec<RequestState> = requests
        .iter()
        .map(|request| RequestState::new(state, request, headers, None, timing.clone()))
        .collect();
    let received = ReceivedAs::Typed(state.core.logged_headers(headers));
    let calls: Vec<_> = match state.core.generate_batch(requests, received).await {
        Ok(calls) => calls.into_iter().zip(request_states).collect(),
        Err(error) => return Some(error_response(error.http_status(), &error.to_string())),
    };

    if !stream {
        // One failed choice fails the request, as Python's batch does; the
        // unfinished choices are dropped, which cancels them.
        let drained = futures::future::try_join_all(calls.into_iter().map(
            |(mut call, request)| async move {
                let rid = call.public_id().to_owned();
                drain_unary_frame(&mut call, &rid, request)
                    .await
                    .map(|frame| GenerateItem::Frame(Box::new(frame)))
            },
        ))
        .await;
        let items = match drained {
            Ok(items) => items,
            Err(error) => return Some(call_error_response(&error)),
        };
        // Every item is a frame: a failed one answered above.
        let output = items
            .iter()
            .map(|item| {
                let value = serde_json::to_value(item).expect("api.v1 types serialize to JSON");
                serde_json::from_value::<GenerateResponse>(value)
            })
            .collect::<Result<Vec<_>, _>>();
        return Some(
            match output.ok().and_then(|output| responder.unary(output)) {
                Some(reply) => reply_response(reply),
                None => error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "unreadable /generate response",
                ),
            },
        );
    }

    let incremental = state.server_args.incremental_streaming_output;
    let frames = generation_event_stream_with(calls, incremental, is_batch, python_error_frame);
    let mut frames = Box::pin(frames);
    // An error before the first frame answers the whole request, as on Python.
    let mut first = Vec::new();
    while first.is_empty() && !responder.done() {
        let Some(data) = frames.next().await else {
            break;
        };
        match stream_data(&mut responder, &data) {
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
            for event in stream_data(&mut responder, &data).unwrap_or_default() {
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

/// A `/generate` item as the processor reads it: a frame, or its error body.
fn generate_result(
    value: Value,
) -> serde_json::Result<Result<GenerateResponse, GenerateStreamError>> {
    if value.get("error").is_some() {
        serde_json::from_value(value).map(Err)
    } else {
        serde_json::from_value(value).map(Ok)
    }
}

/// One `/generate` stream event, through to the `[DONE]` that ends it.
fn stream_data(responder: &mut ChatResponder, data: &str) -> Result<Vec<String>, Reply> {
    if data == "[DONE]" {
        return Ok(responder.stream_end());
    }
    match serde_json::from_str(data).and_then(generate_result) {
        Ok(output) => responder.stream_data(output),
        Err(_) => Ok(Vec::new()),
    }
}

/// The prompt ids lowered chat would send for a chat `body`, or `None` where
/// chat leaves the request to the typed handler.
pub(super) async fn prompt_ids(state: &Arc<AppState>, body: Bytes) -> Option<Vec<i64>> {
    let (generate, _) = lower(state, body, None, false).await?;
    generate["input_ids"]
        .as_array()?
        .iter()
        .map(Value::as_i64)
        .collect()
}

/// [`lower_chat`] with this server's processor, as JSON, under the request's
/// routing key and sglext-ids header.
async fn lower(
    state: &Arc<AppState>,
    body: Bytes,
    routing_key: Option<String>,
    sglext_ids: bool,
) -> Option<(Value, ChatResponder)> {
    let state = state.clone();
    // Rendering and tokenizing are CPU work, kept off the async workers.
    let (generate, responder) = tokio::task::spawn_blocking(move || {
        let openai = state.openai_chat.as_ref()?;
        let formatter = state.chat_formatter.as_ref()?;
        let headers = OpenAiHeaders {
            routing_key: routing_key.as_deref(),
            custom_labels: None,
            sglext_ids,
        };
        let defaults = &state.server_args.default_chat_template_kwargs;
        let render = |request: &Value| openai.render(formatter, request, defaults);
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
    let generate = serde_json::to_value(generate).expect("api.v1 types serialize to JSON");
    Some((generate, responder))
}

/// A prompt's fields whose list form is repeated with it across `n` choices.
const PROMPT_LIST_FIELDS: [&str; 6] = [
    "bootstrap_host",
    "bootstrap_port",
    "bootstrap_room",
    "extra_key",
    "cache_salt",
    "lora_path",
];

/// The core samples one sequence per request, so `n` choices are a batch of
/// the prompt, whose indexed frames are what Python's `n` streams.
fn copy_prompt_per_choice(generate: &mut Value) -> Result<(), Error> {
    let n = generate["sampling_params"]["n"].as_u64().unwrap_or(1);
    let n = usize::try_from(n).unwrap_or(usize::MAX);
    if n <= 1 {
        return Ok(());
    }
    // Bounded before the copy: a huge `n`, or a large list field, would overflow
    // or exhaust the allocation.
    let ids = generate["input_ids"].as_array().map_or(0, Vec::len);
    let fields: Vec<&Value> = PROMPT_LIST_FIELDS
        .iter()
        .map(|key| &generate[*key])
        .filter(|field| field.is_array())
        .collect();
    check_prompt_copies(ids, &fields, n)?;
    generate["sampling_params"]["n"] = 1.into();
    generate["input_ids"] = Value::Array(vec![generate["input_ids"].take(); n]);
    // Python's parallel sampling repeats the prompt's list-form fields with it,
    // and a rid list names the one prompt.
    for key in PROMPT_LIST_FIELDS {
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
    Ok(())
}

/// Python's `/generate` streams a scheduler abort as a frame whose
/// `finish_reason` carries its status, which its chat layer answers as an error event.
fn python_error_frame(error: &CoreError, index: Option<u32>) -> String {
    let CoreError::RuntimeRejected {
        message,
        legacy_http_status,
        ..
    } = error
    else {
        return serde_json::to_string(&error.stream_error(index))
            .expect("api.v1 types serialize to JSON");
    };
    let finish_reason = serde_json::json!({
        "type": "abort",
        "message": message,
        "status_code": legacy_http_status,
    });
    serde_json::json!({
        "text": "",
        "index": index,
        "meta_info": {"finish_reason": finish_reason},
    })
    .to_string()
}

/// `OpenAIServingBase.create_error_response`: a 400 is a `BadRequestError`;
/// any other status is an `HTTPException`, typed by its status code.
fn error_response(status: StatusCode, message: &str) -> Response {
    let err_type = match status {
        StatusCode::BAD_REQUEST => "BadRequestError".to_owned(),
        _ => status.as_u16().to_string(),
    };
    typed_error_response(status, &err_type, message)
}

/// `handle_request`'s answer to a call that failed before its first frame: a
/// scheduler abort is the `HTTPException` the tokenizer manager raises, while a
/// failure of this server's own is Python's generic exception.
fn call_error_response(error: &CoreError) -> Response {
    let status = error.http_status();
    match error {
        CoreError::RuntimeRejected { .. } => error_response(status, &error.to_string()),
        _ if status == StatusCode::INTERNAL_SERVER_ERROR => typed_error_response(
            status,
            "InternalServerError",
            &format!("Internal server error: {error}"),
        ),
        _ => error_response(status, &error.to_string()),
    }
}

fn typed_error_response(status: StatusCode, err_type: &str, message: &str) -> Response {
    let code = status.as_u16();
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

    use super::super::test_utils::{
        answering_app_state_with_item, body_json, checkpoint_formatter, chunk, post_json,
    };
    use super::*;

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
        let app = super::super::routes().with_state(state);
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
}
