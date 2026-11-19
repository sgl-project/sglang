//! The OpenAI API through sglang-processor's port of Python's OpenAI layer,
//! shared by the HTTP routes, the gRPC service and the frontends that re-shape
//! a chat reply (Anthropic, Responses): a request is lowered to a `/generate`
//! body and the reply built from its frames. A request the processor declines
//! is an error.

use std::sync::Arc;

use axum::body::Bytes;
use axum::http::{HeaderMap, StatusCode};
use futures::StreamExt;
use futures::stream::BoxStream;
use serde_json::{Map, Value};
use sglang_api_types::api::v1::{self as api, GenerateResponse, GenerateStreamError};
use sglang_processor::dynamo_tokenizers::{DecodeResult, Tokenizer};
use sglang_processor::openai::{
    ChatModel, OpenAiHeaders, OpenAiSettings, OpenAiTokenizer, PROMPT_LIST_FIELDS,
    ROUTING_KEY_HEADER, Reply, Responder, TokenPieces, Unsupported, choices_per_prompt,
    copy_prompt_per_choice, lower_chat, lower_completion, lower_score,
};
use sglang_processor::{ChatFormatterOptions, DynamoTokenizer};

pub(crate) use sglang_processor::ChatFormatter;

use super::core::CoreError;
use super::handlers::CoreHandle;
use super::http::dp_relay::HintRoute;
use super::http::generate_hook::GenerateHook;
use super::http::native_api::{
    GenerateContext, GenerateItem, RequestState, RequestTiming, drain_unary_frame,
    generate_requests, generation_event_stream_with,
};
use super::http::request_headers::apply_header_overrides;
use super::log::{OpenAiRequest, ReceivedAs};
use crate::message::config::ServerArgs;
use crate::message::request::{GenerationPolicy, RenderedChat, check_prompt_copies};
use crate::utils::error::Error;

/// Extra variables for the chat template (`chat_template_kwargs`).
pub type ChatTemplateKwargs = std::collections::HashMap<String, serde_json::Value>;

/// Resolve the chat formatter, or `None` to disable the OpenAI chat-completions
/// endpoint. Tokenization is the tokenizer pool's job (the api server never
/// encodes); the formatter needs at most `tokenizer_config.json` — a built-in
/// `--chat-template` name or a model-path-inferred legacy template resolve
/// without it, so its absence must not disable chat.
pub(crate) fn load_chat_support(server_args: &ServerArgs) -> Option<ChatFormatter> {
    // Chat needs the tokenizer pool behind it: under `skip_tokenizer_init`
    // there is none (text cannot be submitted), so chat is disabled.
    if server_args.skip_tokenizer_init || server_args.tokenizer_path.is_empty() {
        return None;
    }
    let (formatter, error) = sglang_processor::select_chat_formatter(&ChatFormatterOptions {
        tokenizer_path: server_args.tokenizer_path.clone(),
        model_path: server_args.model_path.clone(),
        revision: server_args.revision.clone(),
        chat_template: server_args.chat_template.clone(),
    });
    if let Some(error) = error {
        tracing::warn!(%error, "OpenAI chat completions disabled");
    }
    // recipe parses a DeepSeek model's output in place of the server's parsers;
    // Python refuses to start where it cannot.
    let deepseek = formatter.as_ref().map(|formatter| {
        sglang_processor::openai::ChatModel::for_formatter(formatter)
            .deepseek_output(
                server_args.reasoning_parser.as_deref(),
                server_args.tool_call_parser.as_deref(),
            )
            .map(|_| ())
    });
    if let Some(Err(error)) = deepseek {
        tracing::warn!(%error, "OpenAI chat completions disabled");
        return None;
    }
    formatter
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub(crate) fn unix_seconds_u32() -> u32 {
    u32::try_from(unix_seconds()).unwrap_or(u32::MAX)
}

/// The server facts and tokenizer the processor's OpenAI layer reads.
#[derive(Default)]
pub struct OpenAi {
    settings: OpenAiSettings,
    /// Echoed token prompts, and token text and bytes for logprobs.
    pieces: Option<Arc<dyn OpenAiTokenizer>>,
    /// `None` without a tokenizer and a chat template the processor renders.
    chat: Option<ChatSupport>,
    /// `--default-chat-template-kwargs`, as the renderer takes them.
    template_defaults: ChatTemplateKwargs,
}

struct ChatSupport {
    model: ChatModel,
    tokenizer: Arc<DynamoTokenizer>,
    formatter: ChatFormatter,
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

impl OpenAi {
    pub fn new(
        server_args: &ServerArgs,
        formatter: Option<&ChatFormatter>,
        tokenizer: Option<Arc<DynamoTokenizer>>,
    ) -> Self {
        let pieces = tokenizer.as_ref().map(|tokenizer| {
            let file = sglang_processor::resolve_model_file(
                &server_args.tokenizer_path,
                server_args.revision.as_deref(),
                "tokenizer.json",
            );
            let pieces = file
                .and_then(|file| std::fs::read(file).ok())
                .and_then(|json| serde_json::from_slice(&json).ok())
                .map(|json| TokenPieces::from_tokenizer_json(&json));
            Arc::new(LogprobPieces {
                tokenizer: tokenizer.tokenizer().clone(),
                pieces,
            }) as Arc<dyn OpenAiTokenizer>
        });
        let chat = formatter
            .zip(tokenizer)
            .map(|(formatter, tokenizer)| ChatSupport::new(server_args, formatter, tokenizer));
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
            pieces,
            chat,
            template_defaults: server_args.default_chat_template_kwargs.clone(),
        }
    }
}

impl ChatSupport {
    fn new(
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
            model: ChatModel {
                context_length: Some(server_args.model_config.context_len),
                generation_config,
                ..ChatModel::for_formatter(formatter)
            },
            tokenizer,
            formatter: formatter.clone(),
        }
    }

    /// The prompt ids Python's chat layer encodes for `request`.
    fn render(&self, request: &Value, defaults: &ChatTemplateKwargs) -> Option<Vec<u32>> {
        let formatter = &self.formatter;
        if let Some(ids) = formatter.encode_request(request.clone(), defaults) {
            return ids.ok();
        }
        let (prompt, continuation) = formatter.render_chat(request.clone(), defaults).ok()?;
        let rendered = RenderedChat {
            prompt,
            continuation,
            ids: None,
        };
        let ids = rendered.encode(&*self.tokenizer).ok()?;
        Some(ids.into_iter().map(|id| id as u32).collect())
    }
}

/// The request headers the processor reads: the routing key, and whether an
/// sglext-ids header is set.
fn processor_headers(headers: &HeaderMap) -> (Option<String>, bool) {
    let header = |name: &str| headers.get(name)?.to_str().ok().map(str::to_owned);
    let sglext_ids = ["x-sglext-return-input-ids", "x-sglext-return-output-ids"]
        .into_iter()
        .any(|name| header(name).as_deref() == Some("1"));
    (header(ROUTING_KEY_HEADER), sglext_ids)
}

/// What serving an OpenAI request through the processor reads of a listener;
/// each transport builds one from its own state.
pub(crate) struct OpenAiService<'a> {
    pub(crate) core: &'a CoreHandle,
    pub(crate) server_args: &'a ServerArgs,
    pub(crate) generation_policy: &'a GenerationPolicy,
    pub(crate) generate_hook: Option<&'a Arc<dyn GenerateHook>>,
    pub(crate) openai: &'a Arc<OpenAi>,
}

/// How a request was answered: one reply, or the events of a started stream,
/// each a whole `data:` line.
pub(crate) enum Served {
    Reply(Reply),
    Events(BoxStream<'static, String>),
}

fn declined(reason: Unsupported) -> Reply {
    tracing::warn!(reason = reason.0, "declined OpenAI request");
    reason.reply()
}

impl OpenAiService<'_> {
    /// `/v1/chat/completions` as HTTP receives it.
    pub(crate) async fn chat_completions(&self, headers: &HeaderMap, body: &Bytes) -> Served {
        let (routing_key, sglext_ids) = processor_headers(headers);
        let lowered = lower(self.openai, body.clone(), routing_key, sglext_ids).await;
        let (generate, responder) = match lowered {
            Ok(lowered) => lowered,
            Err(reason) => return Served::Reply(declined(reason)),
        };
        // Python logs the request once it validates, before adapting it.
        self.core
            .log_openai_request(OpenAiRequest::ChatCompletion, body, headers);
        let received = ReceivedAs::Typed(self.core.logged_headers(headers));
        let route = Some(HintRoute::ChatCompletions);
        self.serve(headers, generate, responder, route, received)
            .await
    }

    /// `/v1/completions` as HTTP receives it.
    pub(crate) async fn completions(&self, headers: &HeaderMap, body: &Bytes) -> Served {
        let (routing_key, sglext_ids) = processor_headers(headers);
        let processor_headers = OpenAiHeaders {
            routing_key: routing_key.as_deref(),
            custom_labels: None,
            sglext_ids,
        };
        let openai = self.openai;
        let lowered = lower_completion(
            body,
            &processor_headers,
            &openai.settings,
            openai.pieces.clone(),
        );
        let (generate, responder) = match lowered {
            Ok(lowered) => lowered,
            Err(reason) => return Served::Reply(declined(reason)),
        };
        self.core
            .log_openai_request(OpenAiRequest::Completion, body, headers);
        let generate = serde_json::to_value(generate).expect("api.v1 types serialize to JSON");
        let responder = Responder::Completion(Box::new(responder));
        let received = ReceivedAs::Typed(self.core.logged_headers(headers));
        let route = Some(HintRoute::Completions);
        self.serve(headers, generate, responder, route, received)
            .await
    }

    /// Serve a chat `body` that came over no HTTP request of its own, as the
    /// frontends that re-shape the chat reply send it.
    pub(crate) async fn chat(&self, body: &Value) -> Served {
        let body = Bytes::from(body.to_string());
        let (generate, responder) = match lower(self.openai, body, None, false).await {
            Ok(lowered) => lowered,
            Err(reason) => return Served::Reply(declined(reason)),
        };
        let received = ReceivedAs::without_headers();
        self.serve(&HeaderMap::new(), generate, responder, None, received)
            .await
    }

    /// `/v1/score`: one zero-token generate per item, scored by its labels.
    pub(crate) async fn score(
        &self,
        request: &api::OpenAiScoreRequest,
        received: ReceivedAs,
    ) -> Reply {
        let vocab_size = self.server_args.model_config.vocab_size;
        let (generate, responder) = match lower_score(request, vocab_size) {
            Ok(lowered) => lowered,
            Err(message) => return Reply::error(400, &message),
        };
        // No items: nothing to generate.
        let Some(generate) = generate else {
            return responder.unary(Vec::new());
        };
        let generate = serde_json::to_value(generate).expect("api.v1 types serialize to JSON");
        let responder = Responder::Score(Box::new(responder));
        let served = self
            .serve(&HeaderMap::new(), generate, responder, None, received)
            .await;
        match served {
            Served::Reply(reply) => reply,
            Served::Events(_) => unreachable!("a score never streams"),
        }
    }

    /// Send a lowered `/generate` body and answer with `responder`'s reply.
    async fn serve(
        &self,
        headers: &HeaderMap,
        mut generate: Value,
        mut responder: Responder,
        route: Option<HintRoute>,
        received: ReceivedAs,
    ) -> Served {
        let reply = |status, message: &str| Served::Reply(error_reply(status, message));
        let overrides = self.server_args.enable_request_header_overrides;
        let stream = generate["stream"] == true;
        // The core samples one sequence per request, so `n` choices are a batch.
        if let Err(error) = check_choices(&generate) {
            return reply(StatusCode::BAD_REQUEST, &error.to_string());
        }
        copy_prompt_per_choice(&mut generate);
        // Python's `adapted_request` takes the router's override headers.
        if overrides && let Err(error) = apply_header_overrides(&mut generate, headers) {
            // `apply_header_overrides` raises `HTTPException(400)`, typed by its status.
            return Served::Reply(typed_error_reply(StatusCode::BAD_REQUEST, "400", &error));
        }
        // Below the override header, `X-Data-Parallel-Rank` names the DP rank.
        match route.map(|route| route.header_rank(headers, overrides)) {
            Some(Ok(Some(rank))) => generate["routed_dp_rank"] = rank.into(),
            Some(Ok(None)) | None => {}
            Some(Err(error)) => {
                return Served::Reply(typed_error_reply(StatusCode::BAD_REQUEST, "400", &error));
            }
        }
        // Failures before the first frame answer unary even for a stream: Python
        // awaits the first chunk before it starts the stream.
        let context = GenerateContext {
            server_args: self.server_args,
            generation_policy: self.generation_policy,
            generate_hook: self.generate_hook,
        };
        let (requests, is_batch) = match generate_requests(&context, generate) {
            Ok(requests) => requests,
            Err((code, message, _)) => return reply(code, &message),
        };
        let timing = RequestTiming::new();
        let request_states: Vec<RequestState> = requests
            .iter()
            .map(|request| RequestState::new(&context, request, headers, None, timing.clone()))
            .collect();
        let calls: Vec<_> = match self.core.generate_batch(requests, received).await {
            Ok(calls) => calls.into_iter().zip(request_states).collect(),
            Err(error) => return reply(error.http_status(), &error.to_string()),
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
                Err(error) => return Served::Reply(call_error_reply(&error)),
            };
            // Every item is a frame: a failed one answered above.
            let output = items
                .iter()
                .map(|item| {
                    let value = serde_json::to_value(item).expect("api.v1 types serialize to JSON");
                    serde_json::from_value::<GenerateResponse>(value)
                })
                .collect::<Result<Vec<_>, _>>();
            let output = output.map(|output| output.into_iter().map(Ok).collect());
            return match output.ok().and_then(|output| responder.unary(output)) {
                Some(reply) => Served::Reply(reply),
                None => reply(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "unreadable /generate response",
                ),
            };
        }

        let incremental = self.server_args.incremental_streaming_output;
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
                Err(reply) => return Served::Reply(reply),
            }
        }
        let events = async_stream::stream! {
            for event in first {
                yield event;
            }
            while !responder.done() {
                let Some(data) = frames.next().await else {
                    break;
                };
                for event in stream_data(&mut responder, &data).unwrap_or_default() {
                    yield event;
                }
            }
        };
        Served::Events(events.boxed())
    }
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
fn stream_data(responder: &mut Responder, data: &str) -> Result<Vec<String>, Reply> {
    if data == "[DONE]" {
        return Ok(responder.stream_end());
    }
    match serde_json::from_str(data).and_then(generate_result) {
        Ok(output) => responder.stream_data(output),
        Err(_) => Ok(Vec::new()),
    }
}

/// The prompt ids chat would send for a chat `body`, or its refusal.
pub(crate) async fn prompt_ids(openai: &Arc<OpenAi>, body: Bytes) -> Result<Vec<i64>, Reply> {
    let (generate, _) = lower(openai, body, None, false).await.map_err(declined)?;
    let ids = generate["input_ids"].as_array().into_iter().flatten();
    Ok(ids.filter_map(Value::as_i64).collect())
}

/// [`lower_chat`] with this server's processor, as JSON, under the request's
/// routing key and sglext-ids header.
async fn lower(
    openai: &Arc<OpenAi>,
    body: Bytes,
    routing_key: Option<String>,
    sglext_ids: bool,
) -> Result<(Value, Responder), Unsupported> {
    let shared = openai.clone();
    // Rendering and tokenizing are CPU work, kept off the async workers.
    let lowered = tokio::task::spawn_blocking(move || {
        let openai = shared;
        let Some(chat) = &openai.chat else {
            return Err(Unsupported("chat_template"));
        };
        let headers = OpenAiHeaders {
            routing_key: routing_key.as_deref(),
            custom_labels: None,
            sglext_ids,
        };
        let render = |request: &Value| chat.render(request, &openai.template_defaults);
        lower_chat(
            &body,
            &headers,
            &openai.settings,
            &chat.model,
            render,
            openai.pieces.clone(),
        )
    })
    .await;
    let (generate, responder) = lowered.map_err(|_| Unsupported("render"))??;
    let generate = serde_json::to_value(generate).expect("api.v1 types serialize to JSON");
    Ok((generate, Responder::Chat(Box::new(responder))))
}

/// Bound `n` choices before the prompt is copied for each: a huge `n`, or a
/// large list field, would overflow or exhaust the allocation.
fn check_choices(generate: &Value) -> Result<(), Error> {
    let n = choices_per_prompt(generate);
    let ids = match &generate["input_ids"] {
        Value::Array(ids) => ids.iter().map(|id| id.as_array().map_or(1, Vec::len)).sum(),
        _ => 0,
    };
    let fields: Vec<&Value> = PROMPT_LIST_FIELDS
        .iter()
        .chain(&["text"])
        .map(|key| &generate[*key])
        .filter(|field| field.is_array() || field.is_string())
        .collect();
    check_prompt_copies(ids, &fields, n)
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

fn error_reply(status: StatusCode, message: &str) -> Reply {
    Reply::error(status.as_u16(), message)
}

/// `handle_request`'s answer to a call that failed before its first frame: a
/// scheduler abort is the `HTTPException` the tokenizer manager raises, while a
/// failure of this server's own is Python's generic exception.
fn call_error_reply(error: &CoreError) -> Reply {
    let status = error.http_status();
    match error {
        CoreError::RuntimeRejected { .. } => error_reply(status, &error.to_string()),
        _ if status == StatusCode::INTERNAL_SERVER_ERROR => typed_error_reply(
            status,
            "InternalServerError",
            &format!("Internal server error: {error}"),
        ),
        _ => error_reply(status, &error.to_string()),
    }
}

fn typed_error_reply(status: StatusCode, err_type: &str, message: &str) -> Reply {
    Reply::typed_error(status.as_u16(), err_type, message)
}
