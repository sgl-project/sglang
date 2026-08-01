//! OpenAI Chat Completions endpoint and chat-template preparation.

use std::collections::BTreeMap;
use std::convert::Infallible;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::post,
};
use dynamo_parsers::reasoning::{ReasoningParser as _, ReasoningParserWrapper};
use dynamo_parsers::tool_calling::jail::Annotated;
use dynamo_parsers::{ToolChoice as DynamoToolChoice, ToolDefinition};
use dynamo_protocols::types::{
    ChatChoice, ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionMessageContent,
    ChatCompletionResponseMessage, ChatCompletionTokenLogprob, ChatCompletionToolChoiceOption,
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    FinishReason as OpenAIFinishReason, ResponseFormat, Role, ServiceTier as ChatServiceTier, Stop,
    TopLogprobs,
};
use futures::StreamExt;
use tokio::sync::mpsc;

use super::super::guard::AbortGuard;
use super::completions::completion_usage;
use super::reasoning::build_reasoning_parser;
use super::tools::{
    apply_tool_constraint, chat_delta, chat_finish_reason, dynamo_parser_name,
    parse_chat_tool_calls, parse_streaming_tool_calls,
};
use super::{
    AppState, ChatFormatter, collect_output, contains_media, indexed_egress_stream, openai_error,
    streaming_error, submit_generation, unix_seconds_u32,
};
use crate::ids::Rid;
use crate::message::{ChunkExtras, EgressItem, GenerateRequest, OneOrMany, SamplingParams};

pub(super) fn routes() -> Router<AppState> {
    Router::new().route("/v1/chat/completions", post(chat_completions))
}

async fn chat_completions(
    State(state): State<AppState>,
    body: Result<Json<CreateChatCompletionRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(request)) => request,
        Err(rejection) => return openai_error(StatusCode::BAD_REQUEST, rejection.body_text()),
    };
    if request.model != state.server_args.served_model_name {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("The model `{}` does not exist", request.model),
        );
    }
    if request.messages.is_empty() {
        return openai_error(StatusCode::BAD_REQUEST, "messages cannot be empty");
    }
    if serde_json::to_value(&request.messages).is_ok_and(|messages| contains_media(&messages)) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "image, audio, video, and file message content is not supported",
        );
    }
    if request.n == Some(0) {
        return openai_error(StatusCode::BAD_REQUEST, "n must be at least 1");
    }
    #[allow(deprecated)]
    let max_tokens = request.max_completion_tokens.or(request.max_tokens);
    if max_tokens == Some(0) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "max_completion_tokens must be positive",
        );
    }
    if request.modalities.as_ref().is_some_and(|modalities| {
        serde_json::to_value(modalities).is_ok_and(|value| value.to_string().contains("\"audio\""))
    }) || request.audio.is_some()
        || request.prediction.is_some()
        || request.web_search_options.is_some()
        || request.mm_processor_kwargs.is_some()
    {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "audio, prediction, web search, and multimodal inputs are not supported",
        );
    }
    #[allow(deprecated)]
    if request.function_call.is_some() || request.functions.is_some() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "deprecated function_call/functions are not supported; use tools and tool_choice",
        );
    }

    let tool_choice = match &request.tool_choice {
        Some(ChatCompletionToolChoiceOption::None) => DynamoToolChoice::None,
        Some(ChatCompletionToolChoiceOption::Required) => DynamoToolChoice::Required,
        Some(ChatCompletionToolChoiceOption::Named(choice)) => {
            DynamoToolChoice::Named(choice.function.name.clone())
        }
        Some(ChatCompletionToolChoiceOption::Auto) | None => DynamoToolChoice::Auto,
    };
    let tools_enabled = request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        && tool_choice != DynamoToolChoice::None;
    let parser = tools_enabled
        .then(|| state.server_args.tool_call_parser.clone())
        .flatten();
    if tools_enabled && parser.is_none() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "tool calls require --tool-call-parser",
        );
    }
    // Python gates the split on `request.separate_reasoning` (default true);
    // the Dynamo request type has no such field, so it is always on when the
    // server was launched with `--reasoning-parser`.
    let reasoning_parser = state.server_args.reasoning_parser.clone();
    let tools = request.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|tool| ToolDefinition {
                name: tool.function.name.clone(),
                parameters: tool.function.parameters.clone(),
                strict: tool.function.strict,
            })
            .collect::<Vec<_>>()
    });
    let tools_slice = tools.as_deref().unwrap_or_default();
    if tool_choice == DynamoToolChoice::Required && tools_slice.is_empty() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "tool_choice is \"required\" but tools is empty",
        );
    }
    if let DynamoToolChoice::Named(name) = &tool_choice
        && !tools_slice.iter().any(|tool| &tool.name == name)
    {
        return openai_error(
            StatusCode::BAD_REQUEST,
            format!("tool named \"{name}\" in tool_choice is not present in tools"),
        );
    }

    let (Some(formatter), Some(tokenizer)) =
        (state.chat_formatter.clone(), state.chat_tokenizer.clone())
    else {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "this model has no usable chat template",
        );
    };
    let prepared = prepare_chat_request(request, formatter, tokenizer).await;
    let (request, input_ids) = match prepared {
        Ok(prepared) => prepared,
        Err(message) => return openai_error(StatusCode::BAD_REQUEST, message),
    };

    let mut sampling = match chat_sampling_params(
        &request,
        &SamplingDefaults::CHAT
            .with_model_defaults(&state.server_args.model_config.default_sampling_params),
    ) {
        Ok(sampling) => sampling,
        Err(message) => return openai_error(StatusCode::BAD_REQUEST, message),
    };
    if let Some(parser) = parser.as_deref()
        && let Err(message) = apply_tool_constraint(
            &mut sampling,
            parser,
            &tool_choice,
            tools_slice,
            request.parallel_tool_calls,
        )
    {
        return openai_error(StatusCode::BAD_REQUEST, message);
    }
    if let Err(error) = sampling.normalize(
        state.server_args.skip_tokenizer_init,
        state
            .server_args
            .model_config
            .vocab_size
            .unwrap_or(u64::MAX),
    ) {
        return openai_error(StatusCode::BAD_REQUEST, error.to_string());
    }

    let stream = request.stream.unwrap_or(false);
    let n = request.n.unwrap_or(1) as usize;
    let want_logprobs = request.logprobs.unwrap_or(false);
    let parallel_tool_calls = request.parallel_tool_calls.unwrap_or(true);
    let stream_tool_choice = request.tool_choice.clone();
    let uses_tool_call_structural_tag = sampling.structural_tag.is_some();
    let service_tier = request.service_tier;
    let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = unix_seconds_u32();
    let model = request.model;
    let include_usage = request
        .stream_options
        .is_some_and(|options| options.include_usage)
        || state.server_args.stream_response_default_include_usage;
    let mut guard = AbortGuard::new_empty(state.senders.clone());
    let mut submitted = Vec::with_capacity(n);

    let mut input_ids = Some(input_ids);
    for index in 0..n {
        let rid = Rid::from_client(&format!("{response_id}-{index}"));
        let choice_input_ids = if index + 1 == n {
            input_ids.take().expect("last chat choice owns the prompt")
        } else {
            input_ids
                .as_ref()
                .expect("chat prompt exists until the last choice")
                .clone()
        };
        let native = GenerateRequest {
            rid: rid.clone(),
            input_ids: Some(choice_input_ids),
            sampling_params: sampling.clone(),
            stream,
            return_logprob: want_logprobs,
            logprob_start_len: -1,
            top_logprobs_num: request.top_logprobs.unwrap_or(0) as i64,
            return_text_in_logprobs: want_logprobs.then_some(true),
            ..Default::default()
        };
        let rx = match submit_generation(&state, native, stream, &mut guard).await {
            Ok(rx) => rx,
            Err(response) => return response,
        };
        submitted.push((index, rid, rx));
    }

    if stream {
        let event_stream = chat_event_stream(
            submitted,
            guard,
            response_id,
            model,
            created,
            want_logprobs,
            include_usage,
            parser,
            reasoning_parser,
            tools,
            stream_tool_choice,
            uses_tool_call_structural_tag,
            parallel_tool_calls,
            service_tier,
        )
        .map(|data| Ok::<_, Infallible>(Event::default().data(data)));
        Sse::new(event_stream).into_response()
    } else {
        unary_chat(
            submitted,
            guard,
            response_id,
            model,
            created,
            want_logprobs,
            parser,
            reasoning_parser,
            tools,
            parallel_tool_calls,
            service_tier,
        )
        .await
    }
}

pub(super) async fn prepare_chat_request(
    mut request: CreateChatCompletionRequest,
    formatter: ChatFormatter,
    tokenizer: dynamo_tokenizers::Tokenizer,
) -> Result<(CreateChatCompletionRequest, Vec<i32>), String> {
    // Template stops first, then the request's own — Python
    // `_apply_conversation_template` (`conv.stop_str` + `request.stop`). A
    // token-id stop cannot be merged into the string list (Python has no such
    // field), so it is kept alone.
    merge_template_stops(&mut request, &formatter);
    tokio::task::spawn_blocking(move || {
        let prompt = formatter
            .render(&request)
            .map_err(|error| format!("chat template render failed: {error}"))?;
        let encoding = tokenizer
            .encode(&prompt)
            .map_err(|error| format!("chat prompt tokenization failed: {error}"))?;
        let input_ids = encoding
            .token_ids()
            .iter()
            .map(|&id| i32::try_from(id).map_err(|_| format!("token ID {id} is out of range")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok::<_, String>((request, input_ids))
    })
    .await
    .map_err(|error| format!("chat preparation failed: {error}"))?
}

/// Merge the formatter's template stops into the request's `stop`.
///
/// Python `_apply_conversation_template`: `stop = copy.copy(conv.stop_str or [])
/// + request.stop` (a string request stop appends as one entry). Without this,
/// generation with a legacy/builtin template would run past the template's own
/// delimiters (e.g. chatml's `<|im_end|>`) whenever they are not model EOS ids.
fn merge_template_stops(request: &mut CreateChatCompletionRequest, formatter: &ChatFormatter) {
    let Some(template_stops) = formatter.stop_strs() else {
        return;
    };
    let mut stops = match template_stops {
        OneOrMany::One(one) => vec![one],
        OneOrMany::Many(many) => many,
    };
    match &request.stop {
        Some(Stop::String(one)) => stops.push(one.clone()),
        Some(Stop::StringArray(many)) => stops.extend(many.iter().cloned()),
        Some(Stop::TokenIdArray(_)) => return,
        None => {}
    }
    request.stop = Some(Stop::StringArray(stops));
}

/// Where an omitted `temperature` / `top_p` gets its value. Mirrors Python's
/// `to_sampling_params` priority: user value > model generation_config (when
/// `--sampling-defaults model`) > OpenAI terminal default
/// (`_DEFAULT_SAMPLING_PARAMS`, which differs per endpoint: chat 1.0/1.0,
/// responses 0.7/1.0).
pub(super) struct SamplingDefaults {
    /// Model defaults; `None` when the model config doesn't set them or when
    /// `--sampling-defaults openai` (the Python dump is then empty).
    temperature: Option<f64>,
    top_p: Option<f64>,
    /// OpenAI terminal defaults for this endpoint.
    fallback_temperature: f64,
    fallback_top_p: f64,
}

impl SamplingDefaults {
    /// `protocol.py` chat `_DEFAULT_SAMPLING_PARAMS`: temperature 1.0, top_p 1.0.
    pub(super) const CHAT: SamplingDefaults = SamplingDefaults {
        temperature: None,
        top_p: None,
        fallback_temperature: 1.0,
        fallback_top_p: 1.0,
    };
    /// `protocol.py` responses `_DEFAULT_SAMPLING_PARAMS`: temperature 0.7, top_p 1.0.
    pub(super) const RESPONSES: SamplingDefaults = SamplingDefaults {
        temperature: None,
        top_p: None,
        fallback_temperature: 0.7,
        fallback_top_p: 1.0,
    };

    /// The resolved model defaults (empty in `--sampling-defaults openai`
    /// mode), which slot between the user's values and the OpenAI terminals.
    pub(super) fn with_model_defaults(
        mut self,
        model: &crate::runtime::DefaultSamplingParams,
    ) -> SamplingDefaults {
        self.temperature = model.temperature;
        self.top_p = model.top_p;
        self
    }
}

#[allow(deprecated)]
pub(super) fn chat_sampling_params(
    request: &CreateChatCompletionRequest,
    defaults: &SamplingDefaults,
) -> Result<SamplingParams, String> {
    let mut stop = None;
    let mut stop_token_ids = None;
    match request.stop.as_ref() {
        Some(Stop::String(value)) => stop = Some(OneOrMany::One(value.clone())),
        Some(Stop::StringArray(values)) => stop = Some(OneOrMany::Many(values.clone())),
        Some(Stop::TokenIdArray(values)) => {
            stop_token_ids = Some(values.iter().map(|&id| id as i64).collect())
        }
        None => {}
    }
    let mut logit_bias = BTreeMap::new();
    if let Some(values) = request.logit_bias.as_ref() {
        for (token, bias) in values {
            let bias = bias
                .as_f64()
                .ok_or_else(|| format!("logit_bias[{token:?}] must be a number"))?;
            logit_bias.insert(token.clone(), bias);
        }
    }
    let json_schema = match request.response_format.as_ref() {
        Some(ResponseFormat::JsonSchema { json_schema }) => Some(json_schema.schema.to_string()),
        Some(ResponseFormat::JsonObject) => Some(r#"{"type":"object"}"#.into()),
        _ => None,
    };

    Ok(SamplingParams {
        max_new_tokens: request
            .max_completion_tokens
            .or(request.max_tokens)
            .map(i64::from),
        stop,
        stop_token_ids,
        temperature: request
            .temperature
            .map(f64::from)
            .or(defaults.temperature)
            .unwrap_or(defaults.fallback_temperature),
        top_p: request
            .top_p
            .map(f64::from)
            .or(defaults.top_p)
            .unwrap_or(defaults.fallback_top_p),
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0) as f64,
        presence_penalty: request.presence_penalty.unwrap_or(0.0) as f64,
        n: 1,
        logit_bias: (!logit_bias.is_empty()).then_some(logit_bias),
        sampling_seed: request.seed,
        json_schema,
        ..Default::default()
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn unary_chat(
    submitted: Vec<(usize, Rid, mpsc::Receiver<EgressItem>)>,
    mut guard: AbortGuard,
    response_id: String,
    model: String,
    created: u32,
    want_logprobs: bool,
    parser: Option<String>,
    reasoning_parser: Option<String>,
    tools: Option<Vec<ToolDefinition>>,
    parallel_tool_calls: bool,
    service_tier: Option<ChatServiceTier>,
) -> Response {
    let mut choices = Vec::with_capacity(submitted.len());
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0u64;

    for (index, rid, rx) in submitted {
        let output = match collect_output(rx, &mut guard, &rid).await {
            Ok(output) => output,
            Err((status, message)) => return openai_error(status, message),
        };

        if prompt_tokens == 0 {
            prompt_tokens = output.prompt_tokens;
        }
        completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
        let logprobs = want_logprobs.then(|| chat_logprobs(output.extras.as_deref()));
        let finish_reason = chat_finish_reason(&output);
        // Split reasoning markers out of the content first (Python splits
        // before tool-call parsing too), then parse tool calls on the clean
        // normal text.
        let (reasoning_text, text) = if let Some(name) = reasoning_parser.as_deref() {
            let mut reasoning_parser = build_reasoning_parser(name);
            let token_ids = output
                .token_ids
                .iter()
                .filter_map(|&id| u32::try_from(id).ok())
                .collect::<Vec<_>>();
            let split = reasoning_parser.detect_and_parse_reasoning(&output.text, &token_ids);
            (split.reasoning_text, split.normal_text)
        } else {
            (String::new(), output.text)
        };
        let (content, tool_calls) = parse_chat_tool_calls(
            text,
            parser.as_deref(),
            tools.as_deref(),
            parallel_tool_calls,
        )
        .await;
        let finish_reason = if tool_calls.is_some() {
            Some(OpenAIFinishReason::ToolCalls)
        } else {
            finish_reason
        };
        #[allow(deprecated)]
        let message = ChatCompletionResponseMessage {
            content: (!content.is_empty()).then_some(ChatCompletionMessageContent::Text(content)),
            refusal: None,
            tool_calls,
            role: Role::Assistant,
            function_call: None,
            audio: None,
            // Python: `reasoning_text if reasoning_text else None`.
            reasoning_content: (!reasoning_text.is_empty()).then_some(reasoning_text),
        };
        choices.push(ChatChoice {
            index: u32::try_from(index).unwrap_or(u32::MAX),
            message,
            finish_reason,
            logprobs,
        });
    }

    Json(CreateChatCompletionResponse {
        id: response_id,
        choices,
        created,
        model,
        service_tier,
        system_fingerprint: None,
        object: "chat.completion".into(),
        usage: Some(completion_usage(
            prompt_tokens,
            u32::try_from(completion_tokens).unwrap_or(u32::MAX),
        )),
    })
    .into_response()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn chat_event_stream(
    submitted: Vec<(usize, Rid, mpsc::Receiver<EgressItem>)>,
    mut guard: AbortGuard,
    response_id: String,
    model: String,
    created: u32,
    want_logprobs: bool,
    include_usage: bool,
    parser: Option<String>,
    reasoning_parser: Option<String>,
    tools: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
    parallel_tool_calls: bool,
    service_tier: Option<ChatServiceTier>,
) -> impl futures::Stream<Item = String> {
    let count = submitted.len();
    let raw = async_stream::stream! {
        let count = submitted.len();
        let mut rids = Vec::with_capacity(count);
        let mut streams = Vec::with_capacity(count);
        let mut prompt_tokens = 0u32;
        let mut completion_tokens = 0u64;
        // One stateful reasoning parser per choice, built lazily on the first
        // content delta (Python keeps a `reasoning_parser_dict` per index).
        let mut reasoning_parsers: Vec<Option<ReasoningParserWrapper>> =
            if reasoning_parser.is_some() { (0..count).map(|_| None).collect() } else { vec![] };
        let reasoning_name = reasoning_parser.as_deref();

        for (index, rid, rx) in submitted {
            rids.push(rid);
            streams.push(indexed_egress_stream(index, rx));
            yield Annotated {
                data: Some(CreateChatCompletionStreamResponse {
                    id: response_id.clone(),
                    choices: vec![ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(None, Some(Role::Assistant), None, None),
                        finish_reason: None,
                        logprobs: None,
                    }],
                    created,
                    model: model.clone(),
                    service_tier: service_tier.clone(),
                    system_fingerprint: None,
                    object: "chat.completion.chunk".into(),
                    usage: None,
                }),
                id: None,
                event: None,
                comment: None,
                error: None,
            };
        }

        let mut events = futures::stream::select_all(streams);
        while let Some((index, item)) = events.next().await {
            let Some(item) = item else {
                yield Annotated {
                    data: None,
                    id: None,
                    event: None,
                    comment: None,
                    error: Some(streaming_error(500, "response truncated before completion")),
                };
                continue;
            };
            let output = match item {
                EgressItem::Frame(output) => output,
                EgressItem::Done(output) => {
                    guard.disarm(&rids[index]);
                    output
                }
                EgressItem::Error(error) => {
                    guard.disarm(&rids[index]);
                    yield Annotated {
                        data: None,
                        id: None,
                        event: None,
                        comment: None,
                        error: Some(streaming_error(error.http_status(), error.to_string())),
                    };
                    continue;
                }
                EgressItem::Control(_) => continue,
            };
            if let Some((code, message)) = output
                .finish_reason
                .as_ref()
                .and_then(|reason| reason.abort_status())
            {
                yield Annotated {
                    data: None,
                    id: None,
                    event: None,
                    comment: None,
                    error: Some(streaming_error(code, message)),
                };
                continue;
            }

            if prompt_tokens == 0 {
                prompt_tokens = output.prompt_tokens;
            }
            completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
            let finish_reason = chat_finish_reason(&output);
            // Split the step's text into (reasoning, normal) deltas when
            // `--reasoning-parser` is set. Mirrors Python's per-step emission:
            // reasoning chunk first (logprobs ride it), then the content chunk.
            let mut emitted = Vec::with_capacity(2);
            if let Some(name) = reasoning_name {
                let parser = reasoning_parsers[index]
                    .get_or_insert_with(|| build_reasoning_parser(name));
                let token_ids = output
                    .token_ids
                    .iter()
                    .filter_map(|&id| u32::try_from(id).ok())
                    .collect::<Vec<_>>();
                let split = parser.parse_reasoning_streaming_incremental(&output.text, &token_ids);
                let mut remaining_logprobs =
                    want_logprobs.then(|| chat_logprobs(output.extras.as_deref()));
                if !split.reasoning_text.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(None, None, None, Some(split.reasoning_text)),
                        finish_reason: None,
                        logprobs: remaining_logprobs.clone(),
                    });
                    remaining_logprobs = None;
                }
                if !split.normal_text.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(Some(split.normal_text), None, None, None),
                        finish_reason: None,
                        logprobs: remaining_logprobs,
                    });
                }
            } else {
                emitted.push(ChatChoiceStream {
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    delta: chat_delta(
                        (!output.text.is_empty()).then_some(output.text),
                        None,
                        None,
                        None,
                    ),
                    finish_reason: None,
                    logprobs: want_logprobs.then(|| chat_logprobs(output.extras.as_deref())),
                });
            };
            // Flush the choice's buffered reasoning tail before its terminal
            // frame (Python `parse_stream_end`, which skips aborts — abort
            // frames already became error chunks above).
            if reasoning_name.is_some()
                && finish_reason.is_some()
                && let Some(parser) = reasoning_parsers[index].as_mut()
            {
                let tail = parser.finish_reasoning_stream();
                if !tail.reasoning_text.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(None, None, None, Some(tail.reasoning_text)),
                        finish_reason: None,
                        logprobs: None,
                    });
                }
            }
            // The finish reason rides the last emitted chunk (the wire format
            // the equivalence tests pin); a step whose text was entirely
            // buffered inside the parser still gets a finish-only frame.
            match emitted.last_mut() {
                Some(last) => last.finish_reason = finish_reason,
                None => emitted.push(ChatChoiceStream {
                    index: u32::try_from(index).unwrap_or(u32::MAX),
                    delta: chat_delta(None, None, None, None),
                    finish_reason,
                    logprobs: None,
                }),
            }
            for choice in emitted {
                yield Annotated {
                    data: Some(CreateChatCompletionStreamResponse {
                        id: response_id.clone(),
                        choices: vec![choice],
                        created,
                        model: model.clone(),
                        service_tier: service_tier.clone(),
                        system_fingerprint: None,
                        object: "chat.completion.chunk".into(),
                        usage: None,
                    }),
                    id: None,
                    event: None,
                    comment: None,
                    error: None,
                };
            }
        }

        if include_usage {
            yield Annotated {
                data: Some(CreateChatCompletionStreamResponse {
                    id: response_id,
                    choices: vec![],
                    created,
                    model,
                    service_tier,
                    system_fingerprint: None,
                    object: "chat.completion.chunk".into(),
                    usage: Some(completion_usage(
                        prompt_tokens,
                        u32::try_from(completion_tokens).unwrap_or(u32::MAX),
                    )),
                }),
                id: None,
                event: None,
                comment: None,
                error: None,
            };
        }
    };

    let parsed: std::pin::Pin<
        Box<dyn futures::Stream<Item = Annotated<CreateChatCompletionStreamResponse>> + Send>,
    > = if let Some(parser) = parser {
        let starts_immediately = !uses_tool_call_structural_tag
            && matches!(
                tool_choice,
                Some(ChatCompletionToolChoiceOption::Required)
                    | Some(ChatCompletionToolChoiceOption::Named(_))
            );
        Box::pin(parse_streaming_tool_calls(
            raw,
            dynamo_parser_name(&parser).to_owned(),
            tools,
            starts_immediately,
        ))
    } else {
        Box::pin(raw)
    };

    async_stream::stream! {
        let mut tool_calls_seen = vec![false; count];
        futures::pin_mut!(parsed);
        while let Some(mut item) = parsed.next().await {
            if let Some(response) = item.data.as_mut() {
                if !parallel_tool_calls {
                    for choice in &mut response.choices {
                        let index = choice.index as usize;
                        if let Some(calls) = choice.delta.tool_calls.as_mut() {
                            if tool_calls_seen.get(index).copied().unwrap_or(false) {
                                calls.clear();
                            } else {
                                calls.truncate(1);
                                if !calls.is_empty()
                                    && let Some(seen) = tool_calls_seen.get_mut(index)
                                {
                                    *seen = true;
                                }
                            }
                            if calls.is_empty() {
                                choice.delta.tool_calls = None;
                            }
                        }
                    }
                }
                yield serialize_chat_stream_response(response.clone());
            } else if let Some(error) = item.error {
                yield error;
            }
        }
        yield "[DONE]".to_string();
    }
}

fn serialize_chat_stream_response(response: CreateChatCompletionStreamResponse) -> String {
    let mut response = serde_json::to_value(response).expect("OpenAI response must serialize");
    if let Some(delta) = response
        .pointer_mut("/choices/0/delta")
        .and_then(serde_json::Value::as_object_mut)
    {
        delta
            .entry("reasoning_content")
            .or_insert(serde_json::Value::Null);
    }
    response.to_string()
}

#[allow(deprecated)]
pub(super) fn chat_logprobs(extras: Option<&ChunkExtras>) -> ChatChoiceLogprobs {
    let mut content = Vec::new();
    let Some(extras) = extras else {
        return ChatChoiceLogprobs {
            content: Some(content),
            refusal: None,
        };
    };
    let mut top_offset = 0usize;
    for (position, (&logprob, &token_id)) in
        extras.out_lp_val.iter().zip(&extras.out_lp_idx).enumerate()
    {
        let token = extras
            .out_lp_txt
            .get(position)
            .cloned()
            .unwrap_or_else(|| format!("token_id:{token_id}"));
        let top_len = extras.out_top_lens.get(position).copied().unwrap_or(0) as usize;
        let top_logprobs = extras.out_top_val[top_offset..]
            .iter()
            .zip(&extras.out_top_idx[top_offset..])
            .take(top_len)
            .enumerate()
            .map(|(offset, (&logprob, &id))| {
                let text = extras
                    .out_top_txt
                    .get(top_offset + offset)
                    .cloned()
                    .unwrap_or_else(|| format!("token_id:{id}"));
                TopLogprobs {
                    bytes: Some(text.as_bytes().to_vec()),
                    token: text,
                    logprob,
                }
            })
            .collect();
        top_offset = top_offset.saturating_add(top_len);
        content.push(ChatCompletionTokenLogprob {
            bytes: Some(token.as_bytes().to_vec()),
            token,
            logprob,
            top_logprobs,
        });
    }
    ChatChoiceLogprobs {
        content: Some(content),
        refusal: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::DefaultSamplingParams;

    fn request() -> CreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap()
    }

    /// Python `to_sampling_params` priority: user value > model generation
    /// config (`--sampling-defaults model`) > OpenAI terminal default.
    #[test]
    fn sampling_defaults_follow_python_priority_chain() {
        let model = DefaultSamplingParams {
            temperature: Some(0.6),
            top_p: Some(0.9),
            ..Default::default()
        };
        // Omitted → model defaults, not the 1.0 OpenAI terminals.
        let sampling = chat_sampling_params(
            &request(),
            &SamplingDefaults::CHAT.with_model_defaults(&model),
        )
        .unwrap();
        assert_eq!(sampling.temperature, 0.6);
        assert_eq!(sampling.top_p, 0.9);
        // Explicit request values win. `Option<f32>` loses precision in f64 —
        // compare with tolerance.
        let mut request = request();
        request.temperature = Some(0.2);
        request.top_p = Some(0.5);
        let sampling = chat_sampling_params(
            &request,
            &SamplingDefaults::CHAT.with_model_defaults(&model),
        )
        .unwrap();
        assert!((sampling.temperature - 0.2).abs() < 1e-6);
        assert!((sampling.top_p - 0.5).abs() < 1e-6);
    }

    /// `--sampling-defaults openai` resolves an empty model-config slice, so the
    /// conversion falls back to the OpenAI terminal defaults.
    #[test]
    fn sampling_defaults_fall_back_to_openai_terminals_in_openai_mode() {
        let openai_mode = DefaultSamplingParams::default();
        let sampling = chat_sampling_params(
            &request(),
            &SamplingDefaults::CHAT.with_model_defaults(&openai_mode),
        )
        .unwrap();
        assert_eq!(sampling.temperature, 1.0);
        assert_eq!(sampling.top_p, 1.0);
    }

    /// The responses endpoint has its own OpenAI terminal default (0.7), from
    /// `protocol.py` `_DEFAULT_SAMPLING_PARAMS` — 1.0 must not leak in.
    #[test]
    fn responses_terminal_default_is_0_7_not_1_0() {
        let openai_mode = DefaultSamplingParams::default();
        let sampling = chat_sampling_params(
            &request(),
            &SamplingDefaults::RESPONSES.with_model_defaults(&openai_mode),
        )
        .unwrap();
        assert_eq!(sampling.temperature, 0.7);
        assert_eq!(sampling.top_p, 1.0);
    }

    /// Python `_apply_conversation_template`: template `stop_str` first, then
    /// the request's own stops.
    #[test]
    fn template_stops_merge_before_request_stops() {
        let chatml = super::super::template::builtin_template("chatml").unwrap();
        let formatter = super::super::ChatFormatter::Legacy(Box::new(
            super::super::template::LegacyFormatter { spec: chatml },
        ));
        assert_eq!(
            formatter.stop_strs(),
            Some(crate::message::OneOrMany::Many(vec![
                "<|endoftext|>".into(),
                "<|im_end|>".into()
            ]))
        );
        // No request stop → the template's delimiters alone.
        let mut req = request();
        merge_template_stops(&mut req, &formatter);
        assert_eq!(
            req.stop,
            Some(Stop::StringArray(vec![
                "<|endoftext|>".into(),
                "<|im_end|>".into()
            ]))
        );
        // A string request stop appends as one entry.
        let mut req = request();
        req.stop = Some(Stop::String("<stop>".into()));
        merge_template_stops(&mut req, &formatter);
        assert_eq!(
            req.stop,
            Some(Stop::StringArray(vec![
                "<|endoftext|>".into(),
                "<|im_end|>".into(),
                "<stop>".into()
            ]))
        );
        // A list request stop extends the list.
        let mut req = request();
        req.stop = Some(Stop::StringArray(vec!["a".into(), "b".into()]));
        merge_template_stops(&mut req, &formatter);
        assert_eq!(
            req.stop,
            Some(Stop::StringArray(vec![
                "<|endoftext|>".into(),
                "<|im_end|>".into(),
                "a".into(),
                "b".into()
            ]))
        );
        // Token-id stops cannot be merged (Python has no such field) — kept alone.
        let mut req = request();
        req.stop = Some(Stop::TokenIdArray(vec![2, 3]));
        merge_template_stops(&mut req, &formatter);
        assert_eq!(req.stop, Some(Stop::TokenIdArray(vec![2, 3])));
    }

    /// The HuggingFace renderer carries no template stops (Python's jinja path
    /// keeps only the request's stops), so the request is left unchanged.
    #[test]
    fn huggingface_formatter_leaves_request_stops_alone() {
        let mut req = request();
        req.stop = Some(Stop::String("x".into()));
        // A prompt formatter is not constructible here without a tokenizer; the
        // empty-legacy-spec twin proves the merge is formatter-gated, and the
        // `HuggingFace` arm returns `None` by construction (see `stop_strs`).
        let legacy = super::super::ChatFormatter::Legacy(Box::new(
            super::super::template::LegacyFormatter {
                spec: super::super::template::LegacySpec::default(),
            },
        ));
        assert!(legacy.stop_strs().is_none());
        merge_template_stops(&mut req, &legacy);
        assert_eq!(req.stop, Some(Stop::String("x".into())));
    }
}
