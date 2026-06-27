//! OpenAI-compatible endpoints: `/v1/completions`, `/v1/chat/completions`, and
//! `/v1/models`. Each runs the same tokenize→generate→detok pipeline as
//! `/generate` and shapes the neutral [`GenerationOutput`] into OpenAI types
//! (`dynamo-protocols`), with chat-template rendering (`dynamo-renderer`) and
//! reasoning / tool-call parsing (`dynamo-parsers`).
//!
//! Mounted on the shared [`AppState`](super::AppState) by the parent
//! `api_server` module; the submit machinery and control plane live there.

use std::convert::Infallible;

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
};

use dynamo_parsers::{
    ReasoningParser, ReasoningParserType, ReasoningParserWrapper, ToolCallResponse,
    detect_and_parse_tool_call,
};
use dynamo_protocols::types::{
    Choice, CompletionFinishReason, CompletionUsage, CreateChatCompletionRequest,
    CreateCompletionRequest, CreateCompletionResponse, Prompt,
};
use dynamo_renderer::{ChatTemplate, ContextMixins, PromptContextMixin, PromptFormatter};

use super::{AppState, submit};
use crate::message::{EgressItem, GeneratePayload, GenerateRequest, RequestKind};
use crate::runtime::ServerArgs;

/// `GET /v1/models` — OpenAI-compatible model list. Served from `server_args`;
/// no scheduler round-trip. Mirrors `http_server.available_models`.
///
/// TODO(v1/models): when `--enable-lora`, append a `ModelCard` per loaded LoRA
/// adapter (`id=lora_name, root=lora_path, parent=served_model_name,
/// max_model_len=None`). Adapters load/unload at runtime, so that part needs a
/// control-request query to the scheduler's LoRA registry.
pub(super) async fn available_models(State(state): State<AppState>) -> Response {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let name = state.server_args.served_model_name();
    let base = serde_json::json!({
        "id": name,
        "object": "model",
        "created": created,
        "owned_by": "sglang",
        "root": name,
        "parent": serde_json::Value::Null,
        "max_model_len": state.server_args.context_len(),
    });
    let list = serde_json::json!({ "object": "list", "data": [base] });
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        serde_json::to_vec(&list).unwrap_or_default(),
    )
        .into_response()
}

/// `POST /v1/completions` — OpenAI-compatible text completion. Maps the request
/// onto the same tokenize→generate→detok pipeline as `/generate`, then formats
/// the neutral output as OpenAI `text_completion` (unary JSON or SSE stream).
///
/// Accepts a single prompt (string / token-ids) or a batch (string array /
/// array-of-token-id-arrays → one choice per prompt). `n>1` and `best_of` are
/// rejected; `echo`, `logprobs`, and `suffix` are ignored. Streaming is limited
/// to a single prompt (batched callers use the unary path).
pub(super) async fn openai_completions(
    State(state): State<AppState>,
    Json(req): Json<CreateCompletionRequest>,
) -> Response {
    if req.n.unwrap_or(1) > 1 || req.best_of.is_some() {
        return (StatusCode::BAD_REQUEST, "n>1 / best_of not supported").into_response();
    }
    // Normalize the prompt into one-or-more `(text | token-ids)` prompts. OpenAI
    // batch multiple prompts per request as a string array or an array-of-integer-arrays;
    // each maps to one output choice.
    let prompts: Vec<(Option<String>, Option<Vec<i32>>)> = match &req.prompt {
        Prompt::String(s) => vec![(Some(s.clone()), None)],
        Prompt::IntegerArray(v) => vec![(None, Some(v.iter().map(|&x| x as i32).collect()))],
        Prompt::StringArray(v) => v.iter().map(|s| (Some(s.clone()), None)).collect(),
        Prompt::ArrayOfIntegerArray(v) => v
            .iter()
            .map(|ids| (None, Some(ids.iter().map(|&x| x as i32).collect())))
            .collect(),
    };
    if prompts.is_empty() {
        return (StatusCode::BAD_REQUEST, "empty prompt").into_response();
    }

    let stream = req.stream.unwrap_or(false);
    if stream && prompts.len() > 1 {
        // Interleaved multi-prompt SSE (per-choice `index`) isn't implemented.
        return (
            StatusCode::BAD_REQUEST,
            "streaming with multiple prompts is not supported",
        )
            .into_response();
    }

    let sampling_params = completion_sampling_params(&req);
    let id = format!("cmpl-{}", state.id_gen.next().0);
    let created = unix_secs();
    let model = req.model;

    if stream {
        // Single prompt (guaranteed by the guard above).
        let (text, input_ids) = prompts.into_iter().next().unwrap();
        let payload = GeneratePayload {
            text,
            input_ids,
            stream,
            sampling_params: Some(sampling_params),
            extra: Default::default(),
        };
        let kind = RequestKind::Generate(GenerateRequest {
            payload,
            input_ids: None,
            stream,
        });
        let mut rx = match submit(&state, kind).await {
            Ok(rx) => rx,
            Err(()) => {
                return (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response();
            }
        };
        // SSE: each chunk carries the text *delta*; final chunk carries the
        // finish_reason; then `data: [DONE]`.
        let s = async_stream::stream! {
            let mut prev = 0usize; // byte offset already emitted
            while let Some(item) = rx.recv().await {
                match item {
                    EgressItem::Frame(out) => {
                        let delta = out.text[prev..].to_string();
                        prev = out.text.len();
                        let chunk = completion_response(&id, created, &model, delta, None, None);
                        yield Ok::<_, Infallible>(Event::default().data(json_string(&chunk)));
                    }
                    EgressItem::Done(out) => {
                        let delta = out.text[prev..].to_string();
                        let chunk = completion_response(
                            &id, created, &model, delta,
                            finish_reason(out.finish_reason.as_deref()), None,
                        );
                        yield Ok(Event::default().data(json_string(&chunk)));
                        break;
                    }
                    EgressItem::Error(e) => {
                        let body = serde_json::json!({
                            "error": { "message": e.to_string(), "code": e.http_status() }
                        });
                        yield Ok(Event::default().data(body.to_string()));
                        break;
                    }
                    EgressItem::Control(_) => break, // never on a generate request
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        Sse::new(s).into_response()
    } else {
        // Unary, possibly batched: submit every prompt up front (so they run
        // concurrently in the scheduler), then collect one choice per prompt in
        // request order. Non-streaming requests emit a single `Done` each, so
        // draining the receivers sequentially can't deadlock on egress buffers.
        let mut rxs = Vec::with_capacity(prompts.len());
        for (text, input_ids) in prompts {
            let payload = GeneratePayload {
                text,
                input_ids,
                stream: false,
                sampling_params: Some(sampling_params.clone()),
                extra: Default::default(),
            };
            let kind = RequestKind::Generate(GenerateRequest {
                payload,
                input_ids: None,
                stream: false,
            });
            match submit(&state, kind).await {
                Ok(rx) => rxs.push(rx),
                Err(()) => {
                    return (StatusCode::SERVICE_UNAVAILABLE, "service unavailable")
                        .into_response();
                }
            }
        }

        let mut choices = Vec::with_capacity(rxs.len());
        let mut prompt_tokens = 0u32;
        let mut completion_tokens = 0u32;
        for (i, mut rx) in rxs.into_iter().enumerate() {
            loop {
                match rx.recv().await {
                    Some(EgressItem::Frame(_)) => continue,
                    Some(EgressItem::Done(out)) => {
                        prompt_tokens += out.prompt_tokens;
                        completion_tokens += out.completion_tokens as u32;
                        choices.push(Choice {
                            text: out.text,
                            index: i as u32,
                            logprobs: None,
                            finish_reason: finish_reason(out.finish_reason.as_deref()),
                        });
                        break;
                    }
                    Some(EgressItem::Error(e)) => {
                        let code = StatusCode::from_u16(e.http_status())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                        let body = serde_json::json!({
                            "error": { "message": e.to_string(), "code": e.http_status() }
                        });
                        return (code, Json(body)).into_response();
                    }
                    Some(EgressItem::Control(_)) => continue,
                    None => {
                        return (StatusCode::from_u16(499).unwrap(), "request aborted")
                            .into_response();
                    }
                }
            }
        }

        let usage = CompletionUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        };
        let resp = CreateCompletionResponse {
            id,
            choices,
            created,
            model,
            system_fingerprint: None,
            object: "text_completion".to_string(),
            usage: Some(usage),
        };
        (
            StatusCode::OK,
            [("content-type", "application/json")],
            json_vec(&resp),
        )
            .into_response()
    }
}

/// Map OpenAI completion sampling fields to the scheduler's `SamplingParams`
/// (msgpack map, carried through opaquely — `stop` defaults are added at the
/// ingress wire boundary, so we only emit what the client set).
fn completion_sampling_params(req: &CreateCompletionRequest) -> rmpv::Value {
    use rmpv::Value;
    let mut m: Vec<(Value, Value)> = Vec::new();
    if let Some(v) = req.max_tokens {
        m.push((Value::from("max_new_tokens"), Value::from(v)));
    }
    if let Some(v) = req.temperature {
        m.push((Value::from("temperature"), Value::F64(v as f64)));
    }
    if let Some(v) = req.top_p {
        m.push((Value::from("top_p"), Value::F64(v as f64)));
    }
    if let Some(v) = req.presence_penalty {
        m.push((Value::from("presence_penalty"), Value::F64(v as f64)));
    }
    if let Some(v) = req.frequency_penalty {
        m.push((Value::from("frequency_penalty"), Value::F64(v as f64)));
    }
    if let Some(strs) = req.stop.as_ref().and_then(|s| s.strings()) {
        m.push((
            Value::from("stop"),
            Value::Array(strs.into_iter().map(Value::from).collect()),
        ));
    }
    Value::Map(m)
}

/// SGLang finish reasons collapse onto OpenAI's three: `length` → `Length`,
/// anything else (stop string / EOS / abort) → `Stop`.
fn finish_reason(reason: Option<&str>) -> Option<CompletionFinishReason> {
    reason.map(|r| {
        if r.contains("length") {
            CompletionFinishReason::Length
        } else {
            CompletionFinishReason::Stop
        }
    })
}

/// Build one OpenAI `text_completion` object (used for both the unary response
/// and each streamed chunk — chunks just carry a text delta + no usage).
fn completion_response(
    id: &str,
    created: u32,
    model: &str,
    text: String,
    finish: Option<CompletionFinishReason>,
    usage: Option<CompletionUsage>,
) -> CreateCompletionResponse {
    CreateCompletionResponse {
        id: id.to_string(),
        choices: vec![Choice {
            text,
            index: 0,
            logprobs: None,
            finish_reason: finish,
        }],
        created,
        model: model.to_string(),
        system_fingerprint: None,
        object: "text_completion".to_string(),
        usage,
    }
}

fn unix_secs() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(0)
}

fn json_string<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string(v).unwrap_or_default()
}

fn json_vec<T: serde::Serialize>(v: &T) -> Vec<u8> {
    serde_json::to_vec(v).unwrap_or_default()
}

/// Build the chat-template renderer from the model's `tokenizer_config.json`
/// (resolved next to `tokenizer.json` / from the HF cache). `None` when there's
/// no tokenizer (`skip_tokenizer_init`), the config can't be located/parsed, or
/// the model ships no `chat_template` — in which case `/v1/chat/completions`
/// returns a 400 and `/v1/completions` should be used instead.
pub(super) fn load_chat_formatter(server_args: &ServerArgs) -> Option<PromptFormatter> {
    if server_args.skip_tokenizer_init() {
        return None;
    }
    let path = server_args.tokenizer_path()?;
    let cfg = crate::tokenizer::resolve_model_file(
        &path,
        server_args.revision().as_deref(),
        "tokenizer_config.json",
    )?;
    let json = std::fs::read_to_string(&cfg).ok()?;
    let template: ChatTemplate = serde_json::from_str(&json).ok()?;
    template.chat_template.as_ref()?; // no chat_template → base model, no chat support
    match PromptFormatter::from_parts(
        template,
        ContextMixins::new(&[PromptContextMixin::OaiChat]),
        false,
    ) {
        Ok(f) => {
            tracing::info!(%cfg, "loaded chat template");
            Some(f)
        }
        Err(e) => {
            tracing::warn!(error = %e, "chat template build failed; /v1/chat/completions disabled");
            None
        }
    }
}

/// `POST /v1/chat/completions` — OpenAI-compatible chat. Renders `messages` to a
/// prompt via the model's chat template (`dynamo-renderer`), runs the same
/// pipeline as `/generate`, and formats the result as OpenAI `chat.completion`.
///
/// Content, reasoning (`reasoning_content`), and tool calls (`tool_calls`) are
/// all supported; `n>1` is rejected. Streaming tool calls land in the final
/// chunk rather than token-incrementally (see the buffering note below).
pub(super) async fn openai_chat_completions(
    State(state): State<AppState>,
    Json(req): Json<CreateChatCompletionRequest>,
) -> Response {
    let Some(chat) = &state.chat else {
        return (
            StatusCode::BAD_REQUEST,
            "this model has no chat template; use /v1/completions",
        )
            .into_response();
    };
    if req.n.unwrap_or(1) > 1 {
        return (StatusCode::BAD_REQUEST, "n>1 not supported").into_response();
    }

    // Render messages → prompt string via the HF chat template.
    let PromptFormatter::OAI(formatter) = &chat.0;
    let prompt = match formatter.render(&req) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                format!("chat template render failed: {e}"),
            )
                .into_response();
        }
    };

    // When the chat template prefills the reasoning-open marker (`<think>` for
    // Qwen3 / DeepSeek-R1 / GLM), the model's output *starts* inside a reasoning
    // block with no opening tag in the stream. These parsers run with
    // `force_reasoning=false`, so the streaming parser must be told it begins in
    // reasoning via `set_in_reasoning(true)` (the one-shot non-streaming parse
    // recovers this on its own by scanning for the closing `</think>`).
    let starts_in_reasoning = prompt.trim_end().ends_with("<think>");

    let stream = req.stream.unwrap_or(false);
    let payload = GeneratePayload {
        text: Some(prompt),
        input_ids: None,
        stream,
        sampling_params: Some(chat_sampling_params(&req)),
        extra: Default::default(),
    };
    let kind = RequestKind::Generate(GenerateRequest {
        payload,
        input_ids: None,
        stream,
    });
    let mut rx = match submit(&state, kind).await {
        Ok(rx) => rx,
        Err(()) => return (StatusCode::SERVICE_UNAVAILABLE, "service unavailable").into_response(),
    };

    let id = format!("chatcmpl-{}", state.id_gen.next().0);
    let created = unix_secs();
    let model = req.model;
    // Reasoning family (e.g. deepseek-r1, qwen3); fresh per-request parser since
    // it's stateful. `None` → content passes through unsplit.
    let reasoning = state.server_args.reasoning_parser();
    // Tool-call parser family — only engaged when the request actually carries
    // `tools` *and* a parser is configured. `None` → output stays plain content.
    let tool_parser = req
        .tools
        .as_ref()
        .is_some_and(|t| !t.is_empty())
        .then(|| state.server_args.tool_call_parser())
        .flatten();

    if stream {
        let mut reasoner =
            reasoning.map(|n| ReasoningParserType::get_reasoning_parser_from_name(&n));
        // Seed the streaming parser's state when the prompt prefilled `<think>`.
        if let Some(r) = reasoner.as_mut().filter(|_| starts_in_reasoning) {
            r.set_in_reasoning(true);
        }
        // Streaming tool calls aren't emitted token-incrementally: when a parser
        // is engaged we buffer the post-reasoning content (it may be a tool call,
        // indistinguishable from plain text until the closing marker) and parse it
        // once at the end, emitting `tool_calls` (or the buffered content) in the
        // final chunk. Reasoning still streams live.
        let tools_on = tool_parser.is_some();
        let s = async_stream::stream! {
            let mut prev = 0usize;
            let mut first = true;
            let mut content_buf = String::new();
            while let Some(item) = rx.recv().await {
                match item {
                    EgressItem::Frame(out) => {
                        let text = out.text[prev..].to_string();
                        prev = out.text.len();
                        let (content, reason) = split_reasoning(reasoner.as_mut(), &text, false);
                        if tools_on {
                            if let Some(c) = content {
                                content_buf.push_str(&c);
                            }
                            if reason.is_some() || first {
                                let d = chat_delta(first, None, reason);
                                first = false;
                                yield Ok::<_, Infallible>(Event::default().data(chat_chunk(&id, created, &model, d, None)));
                            }
                        } else if content.is_some() || reason.is_some() || first {
                            let d = chat_delta(first, content, reason);
                            first = false;
                            yield Ok::<_, Infallible>(Event::default().data(chat_chunk(&id, created, &model, d, None)));
                        }
                    }
                    EgressItem::Done(out) => {
                        let text = out.text[prev..].to_string();
                        let (content, reason) = split_reasoning(reasoner.as_mut(), &text, true);
                        if tools_on {
                            if let Some(c) = content {
                                content_buf.push_str(&c);
                            }
                            if reason.is_some() {
                                let d = chat_delta(first, None, reason);
                                first = false;
                                yield Ok::<_, Infallible>(Event::default().data(chat_chunk(&id, created, &model, d, None)));
                            }
                            let (tool_calls, leftover) = parse_tool_calls(
                                tool_parser.as_deref(),
                                std::mem::take(&mut content_buf),
                            )
                            .await;
                            let (delta, finish) = match tool_calls {
                                Some(tc) => (tool_delta(first, tc, &leftover), "tool_calls"),
                                None => (
                                    chat_delta(first, (!leftover.is_empty()).then_some(leftover), None),
                                    chat_finish(out.finish_reason.as_deref()).unwrap_or("stop"),
                                ),
                            };
                            yield Ok(Event::default().data(chat_chunk(&id, created, &model, delta, Some(finish))));
                        } else {
                            let d = chat_delta(first, content, reason);
                            yield Ok(Event::default().data(chat_chunk(&id, created, &model, d, chat_finish(out.finish_reason.as_deref()))));
                        }
                        break;
                    }
                    EgressItem::Error(e) => {
                        let body = serde_json::json!({
                            "error": { "message": e.to_string(), "code": e.http_status() }
                        });
                        yield Ok(Event::default().data(body.to_string()));
                        break;
                    }
                    EgressItem::Control(_) => break,
                }
            }
            yield Ok(Event::default().data("[DONE]"));
        };
        Sse::new(s).into_response()
    } else {
        while let Some(item) = rx.recv().await {
            match item {
                EgressItem::Frame(_) => continue,
                EgressItem::Done(out) => {
                    // Non-streaming: split the full text in one pass.
                    let (content, reason) = match &reasoning {
                        Some(n) => {
                            let mut p = ReasoningParserType::get_reasoning_parser_from_name(n);
                            let r = p.detect_and_parse_reasoning(&out.text, &[]);
                            (
                                r.normal_text,
                                (!r.reasoning_text.is_empty()).then_some(r.reasoning_text),
                            )
                        }
                        None => (out.text, None),
                    };
                    // Extract tool calls from the (post-reasoning) content; on no
                    // calls / parser error the text stays plain content.
                    let (tool_calls, content) =
                        parse_tool_calls(tool_parser.as_deref(), content).await;
                    let finish = if tool_calls.is_some() {
                        Some("tool_calls")
                    } else {
                        chat_finish(out.finish_reason.as_deref())
                    };
                    let body = chat_response(
                        &id,
                        created,
                        &model,
                        &content,
                        reason.as_deref(),
                        tool_calls,
                        finish,
                        out.prompt_tokens,
                        out.completion_tokens as u32,
                    );
                    return (StatusCode::OK, [("content-type", "application/json")], body)
                        .into_response();
                }
                EgressItem::Error(e) => {
                    let code = StatusCode::from_u16(e.http_status())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    let body = serde_json::json!({
                        "error": { "message": e.to_string(), "code": e.http_status() }
                    });
                    return (code, Json(body)).into_response();
                }
                EgressItem::Control(_) => continue,
            }
        }
        (StatusCode::from_u16(499).unwrap(), "request aborted").into_response()
    }
}

/// Split a text delta into `(content, reasoning_content)` using the per-request
/// reasoning parser, or pass it straight through as content when none is
/// configured. `finalize` flushes any buffered partial marker at end-of-stream.
fn split_reasoning(
    parser: Option<&mut ReasoningParserWrapper>,
    delta: &str,
    finalize: bool,
) -> (Option<String>, Option<String>) {
    match parser {
        None => ((!delta.is_empty()).then(|| delta.to_string()), None),
        Some(p) => {
            let mut r = p.parse_reasoning_streaming_incremental(delta, &[]);
            if finalize {
                let f = p.finish_reasoning_stream();
                r.normal_text.push_str(&f.normal_text);
                r.reasoning_text.push_str(&f.reasoning_text);
            }
            (r.get_some_normal_text(), r.get_some_reasoning())
        }
    }
}

/// Parse tool calls out of `content` using the configured parser family. Returns
/// `(tool_calls, leftover_content)`; on no calls or a parser error the original
/// text is returned as plain content (`None`). `parser` is `None` when tool
/// parsing isn't engaged (no `tools` in the request, or none configured).
async fn parse_tool_calls(
    parser: Option<&str>,
    content: String,
) -> (Option<serde_json::Value>, String) {
    let Some(name) = parser else {
        return (None, content);
    };
    match detect_and_parse_tool_call(&content, Some(name), None).await {
        Ok((calls, normal)) if !calls.is_empty() => {
            (Some(tool_calls_json(&calls)), normal.unwrap_or_default())
        }
        _ => (None, content),
    }
}

/// Shape parsed tool calls as OpenAI `tool_calls` JSON. `index` is included for
/// the streaming-delta form and harmlessly ignored by non-streaming clients.
fn tool_calls_json(calls: &[ToolCallResponse]) -> serde_json::Value {
    serde_json::Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                serde_json::json!({
                    "index": i,
                    "id": c.id,
                    "type": "function",
                    "function": { "name": c.function.name, "arguments": c.function.arguments },
                })
            })
            .collect(),
    )
}

/// Final streaming delta carrying `tool_calls` (and any leftover content). Mirrors
/// [`chat_delta`] but for the tool-call case.
fn tool_delta(first: bool, tool_calls: serde_json::Value, content: &str) -> serde_json::Value {
    let mut d = serde_json::Map::new();
    if first {
        d.insert("role".into(), serde_json::json!("assistant"));
    }
    if !content.is_empty() {
        d.insert("content".into(), serde_json::json!(content));
    }
    d.insert("tool_calls".into(), tool_calls);
    serde_json::Value::Object(d)
}

/// Same OpenAI sampling fields as completions; chat prefers `max_completion_tokens`
/// but still honors the deprecated `max_tokens` that many clients send.
#[allow(deprecated)]
fn chat_sampling_params(req: &CreateChatCompletionRequest) -> rmpv::Value {
    use rmpv::Value;
    let mut m: Vec<(Value, Value)> = Vec::new();
    if let Some(v) = req.max_completion_tokens.or(req.max_tokens) {
        m.push((Value::from("max_new_tokens"), Value::from(v)));
    }
    if let Some(v) = req.temperature {
        m.push((Value::from("temperature"), Value::F64(v as f64)));
    }
    if let Some(v) = req.top_p {
        m.push((Value::from("top_p"), Value::F64(v as f64)));
    }
    if let Some(v) = req.presence_penalty {
        m.push((Value::from("presence_penalty"), Value::F64(v as f64)));
    }
    if let Some(v) = req.frequency_penalty {
        m.push((Value::from("frequency_penalty"), Value::F64(v as f64)));
    }
    if let Some(strs) = req.stop.as_ref().and_then(|s| s.strings()) {
        m.push((
            Value::from("stop"),
            Value::Array(strs.into_iter().map(Value::from).collect()),
        ));
    }
    Value::Map(m)
}

/// OpenAI chat finish reason: `length` → `length`, anything else → `stop`.
/// (`tool_calls` is set by the tool-parsing step, not here.)
fn chat_finish(reason: Option<&str>) -> Option<&'static str> {
    reason.map(|r| {
        if r.contains("length") {
            "length"
        } else {
            "stop"
        }
    })
}

/// Streaming delta object: the first chunk carries `role: "assistant"`; reasoning
/// tokens go to `reasoning_content`, normal tokens to `content`.
fn chat_delta(
    first: bool,
    content: Option<String>,
    reasoning: Option<String>,
) -> serde_json::Value {
    let mut d = serde_json::Map::new();
    if first {
        d.insert("role".into(), serde_json::json!("assistant"));
    }
    if let Some(r) = reasoning {
        d.insert("reasoning_content".into(), serde_json::json!(r));
    }
    if let Some(c) = content {
        d.insert("content".into(), serde_json::json!(c));
    }
    serde_json::Value::Object(d)
}

fn chat_chunk(
    id: &str,
    created: u32,
    model: &str,
    delta: serde_json::Value,
    finish: Option<&str>,
) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
    })
    .to_string()
}

// A flat formatting helper: the fields are the OpenAI response columns, not a
// call-ergonomics problem worth a wrapper struct.
#[allow(clippy::too_many_arguments)]
fn chat_response(
    id: &str,
    created: u32,
    model: &str,
    content: &str,
    reasoning: Option<&str>,
    tool_calls: Option<serde_json::Value>,
    finish: Option<&str>,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> Vec<u8> {
    // OpenAI sets `content: null` when the model emitted only tool calls.
    let mut message = if tool_calls.is_some() && content.is_empty() {
        serde_json::json!({ "role": "assistant", "content": serde_json::Value::Null })
    } else {
        serde_json::json!({ "role": "assistant", "content": content })
    };
    if let Some(r) = reasoning {
        message["reasoning_content"] = serde_json::json!(r);
    }
    if let Some(tc) = tool_calls {
        message["tool_calls"] = tc;
    }
    serde_json::to_vec(&serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }))
    .unwrap_or_default()
}
