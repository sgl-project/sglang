//! OpenAI Chat Completions endpoint and chat-template preparation.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::sync::Arc;

use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::post,
};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::Value;
use sglang_processor::dynamo_parsers::tool_calling::jail::Annotated;
use sglang_processor::dynamo_parsers::{ToolChoice as DynamoToolChoice, ToolDefinition};
use sglang_processor::dynamo_protocols::types::{
    ChatChoice, ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionMessageContent,
    ChatCompletionMessageToolCall, ChatCompletionMessageToolCallChunk,
    ChatCompletionResponseMessage, ChatCompletionTokenLogprob, ChatCompletionToolChoiceOption,
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    FinishReason as OpenAIFinishReason, FunctionCall, FunctionCallStream, FunctionType,
    ResponseFormat, Role, ServiceTier as ChatServiceTier, Stop, TopLogprobs,
};
use sglang_processor::openai::ChatModel;
use sglang_processor::{
    DeepSeekDelta, DeepSeekFinish, DeepSeekOutputOptions, DeepSeekOutputParser, DeepSeekStopMatch,
    ReasoningOptions, ReasoningStreamSplitter, dynamo_tool_choice, split_reasoning, tool_call_id,
    tool_call_stream,
};

use super::completions::completion_usage;
use super::lowered;
use super::pd_routing::PDRoutingFields;
use super::template::{RenderedChat, render_chat};
use super::tools::{apply_tool_constraint, chat_delta, chat_finish_reason, parse_chat_tool_calls};
use super::{
    AppState, ChatFormatter, ChatTemplateKwargs, collect_output, contains_media, error_payload,
    indexed_decode_stream, model_error, openai_error, submit_generation, unix_seconds_u32,
};
use crate::api_server::core::{CoreEvent, CoreOutput};
use crate::api_server::handlers::CoreCall;
use crate::message::config::{DefaultSamplingParams, ServerArgs};
use crate::message::finish_reason::{FinishKind, FinishReason, Matched};
use crate::message::ids::Rid;
use crate::message::request::GenerateRequest;
use crate::message::response::ChunkExtras;
use crate::message::sampling::SamplingParams;
use crate::message::types::OneOrMany;

pub(super) fn routes() -> Router<Arc<AppState>> {
    Router::new().route("/v1/chat/completions", post(chat_completions))
}

#[derive(Deserialize)]
pub(super) struct ChatRequest {
    #[serde(flatten)]
    pub(super) request: CreateChatCompletionRequest,
    /// The renderer reads it off the raw body; typed here so a malformed one
    /// is a 400, as Python's request model makes it.
    #[allow(dead_code)]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
    /// Python's `skip_special_tokens` request extension; absent keeps the
    /// sampling default (drop them).
    skip_special_tokens: Option<bool>,
    #[serde(flatten)]
    routing: PDRoutingFields,
}

/// A chat request with its raw JSON, which SGLang's renderers read for fields
/// the typed request drops (`continue_final_message`, `task`, ...).
pub(super) struct ChatBody {
    pub(super) request: ChatRequest,
    pub(super) raw: Value,
}

impl<'de> Deserialize<'de> for ChatBody {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = Value::deserialize(deserializer)?;
        let request = ChatRequest::deserialize(&raw).map_err(serde::de::Error::custom)?;
        Ok(Self { request, raw })
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Some(response) = lowered::chat(&state, &headers, &body).await {
        return response;
    }
    let ChatBody {
        request:
            ChatRequest {
                request,
                skip_special_tokens,
                routing,
                ..
            },
        raw,
    } = match Json::from_bytes(&body) {
        Ok(Json(body)) => body,
        Err(rejection) => {
            return openai_error(StatusCode::BAD_REQUEST, rejection.body_text(), false);
        }
    };
    if let Some(message) = model_error(&request.model, &state.server_args.served_model_name) {
        return openai_error(StatusCode::BAD_REQUEST, message, false);
    }
    if request.messages.is_empty() {
        return openai_error(StatusCode::BAD_REQUEST, "messages cannot be empty", false);
    }
    if serde_json::to_value(&request.messages).is_ok_and(|messages| contains_media(&messages)) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "image, audio, video, and file message content is not supported",
            false,
        );
    }
    if request.n == Some(0) {
        return openai_error(StatusCode::BAD_REQUEST, "n must be at least 1", false);
    }
    #[allow(deprecated)]
    let max_tokens = request.max_completion_tokens.or(request.max_tokens);
    if max_tokens == Some(0) {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "max_completion_tokens must be positive",
            false,
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
            false,
        );
    }
    #[allow(deprecated)]
    if request.function_call.is_some() || request.functions.is_some() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "deprecated function_call/functions are not supported; use tools and tool_choice",
            false,
        );
    }

    let tool_choice = dynamo_tool_choice(&request.tool_choice);
    let tools = match sglang_processor::chat_tool_definitions(
        request.tools.as_deref(),
        &request.messages,
    ) {
        Ok(tools) => tools,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, error.to_string(), false),
    };
    let tools_enabled = !tools.is_empty() && tool_choice != DynamoToolChoice::None;
    let parser = tools_enabled
        .then(|| state.server_args.tool_call_parser.clone())
        .flatten();
    if tools_enabled && parser.is_none() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "tool calls require --tool-call-parser",
            false,
        );
    }
    // Python gates the split on `request.separate_reasoning` (default true);
    // the Dynamo request type has no such field, so the text parsers always
    // split when the server was launched with `--reasoning-parser`. recipe
    // reads the flag off the raw request.
    let reasoning_parser = state.server_args.reasoning_parser.clone();
    let tools = (!tools.is_empty()).then_some(tools);
    let tools_slice = tools.as_deref().unwrap_or_default();

    // Python's chat-layer facts for the served template, and whether the
    // reply starts in reasoning, as Python decides it from the request.
    let chat_model = state
        .chat_formatter
        .as_ref()
        .map(ChatModel::for_formatter)
        .unwrap_or_default();
    let force_reasoning = chat_model.request_thinking(
        reasoning_parser.as_deref(),
        &raw,
        &state.server_args.default_chat_template_kwargs,
    );
    // `ReasoningParser`: a continued assistant turn's text is its previous content.
    let previous_content = sglang_processor::continues_final_message(&raw)
        .unwrap_or(false)
        .then(|| raw["messages"].as_array()?.last())
        .flatten()
        // Roles are compared lowercased, as Pydantic normalizes them.
        .filter(|message| {
            message["role"]
                .as_str()
                .is_some_and(|role| role.eq_ignore_ascii_case("assistant"))
        })
        .and_then(|message| message["content"].as_str())
        .map(str::to_owned);
    // recipe parses DeepSeek output from its token ids, in place of both parsers.
    let deepseek = match chat_model.deepseek_output(
        reasoning_parser.as_deref(),
        state.server_args.tool_call_parser.as_deref(),
    ) {
        Ok(None) => None,
        Ok(Some(output)) => match output.for_request(
            &raw,
            &state.server_args.default_chat_template_kwargs,
            reasoning_parser.is_some(),
            parser.is_some(),
        ) {
            Ok(options) => Some(options),
            Err(message) => return openai_error(StatusCode::BAD_REQUEST, message, false),
        },
        // `load_chat_support` serves no chat under such a configuration.
        Err(error) => {
            return openai_error(StatusCode::INTERNAL_SERVER_ERROR, error.to_string(), false);
        }
    };
    let (request, rendered) = match prepare_chat_request(&state, raw, request).await {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };

    let mut sampling = match chat_sampling(
        &request,
        SamplingDefaults::CHAT,
        parser.as_deref(),
        &tool_choice,
        tools_slice,
        request.parallel_tool_calls,
        &state.server_args,
    ) {
        Ok(sampling) => sampling,
        Err(message) => {
            return openai_error(StatusCode::BAD_REQUEST, message, false);
        }
    };
    if let Some(skip_special_tokens) = skip_special_tokens {
        sampling.skip_special_tokens = skip_special_tokens;
    }

    if chat_model.keeps_special_tokens(reasoning_parser.as_deref()) {
        sampling.skip_special_tokens = false;
    }

    let stream = request.stream.unwrap_or(false);
    let prompt_count = 1;
    let choices_per_prompt = request.n.unwrap_or(1) as usize;
    let routing = match routing.normalize(prompt_count, choices_per_prompt) {
        Ok(routing) => routing,
        Err(error) => return openai_error(StatusCode::BAD_REQUEST, error.to_string(), false),
    };
    // Chat normalizes one prompt, so each bootstrap column has one entry.
    let bootstrap_host = &routing.bootstrap.bootstrap_hosts[0];
    let bootstrap_port = routing.bootstrap.bootstrap_ports[0];
    let bootstrap_room = routing.bootstrap.bootstrap_rooms[0];
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
    let mut submitted = Vec::with_capacity(choices_per_prompt);

    let prompt_stub_tokens = chat_model.unreported_prompt_tokens() as u32;
    // As Python, the continued text decides whether generation starts in reasoning.
    let reasoning_options = ReasoningOptions {
        force_reasoning,
        previous_content,
        ..Default::default()
    };
    let mut prompt = Some(Box::new(rendered));
    for index in 0..choices_per_prompt {
        let rid = Rid::from_client(&format!("{response_id}-{index}"));
        let choice_prompt = if index + 1 == choices_per_prompt {
            prompt.take().expect("last chat choice owns the prompt")
        } else {
            prompt
                .as_ref()
                .expect("chat prompt exists until the last choice")
                .clone()
        };
        let native = GenerateRequest {
            rid: rid.clone(),
            chat_prompt: Some(choice_prompt),
            sampling_params: sampling.clone(),
            stream,
            return_logprob: want_logprobs,
            logprob_start_len: -1,
            top_logprobs_num: request.top_logprobs.unwrap_or(0) as i64,
            return_text_in_logprobs: want_logprobs.then_some(true),
            bootstrap_host: bootstrap_host.clone(),
            bootstrap_port,
            bootstrap_room,
            routed_dp_rank: routing.routed_dp_rank,
            disagg_prefill_dp_rank: routing.disagg_prefill_dp_rank,
            // Python's thinking mode: the grammar starts after the reasoning,
            // which the scheduler also counts.
            require_reasoning: reasoning_options.force_reasoning == Some(true),
            ..Default::default()
        };
        let call = match submit_generation(&state, native, stream).await {
            Ok(call) => call,
            Err(response) => return response,
        };
        submitted.push((index, call));
    }

    let parsing = match deepseek {
        Some(options) => OutputParsing::DeepSeek(options),
        None => OutputParsing::Text {
            reasoning_parser,
            reasoning_options,
            tool_parser: parser,
            tools,
        },
    };
    if stream {
        let event_stream = chat_event_stream(
            submitted,
            response_id,
            model,
            created,
            want_logprobs,
            include_usage,
            parsing,
            prompt_stub_tokens,
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
            response_id,
            model,
            created,
            want_logprobs,
            parsing,
            prompt_stub_tokens,
            parallel_tool_calls,
            service_tier,
        )
        .await
    }
}

/// Render the chat template for an OpenAI request, mapping a missing
/// formatter or a render failure to the standard 400.
pub(super) async fn prepare_chat_request(
    state: &AppState,
    body: Value,
    mut request: CreateChatCompletionRequest,
) -> Result<(CreateChatCompletionRequest, RenderedChat), Response> {
    let Some(formatter) = state.chat_formatter.clone() else {
        return Err(openai_error(
            StatusCode::BAD_REQUEST,
            "this model has no usable chat template",
            false,
        ));
    };
    // Template stops first, then the request's own — Python
    // `_apply_conversation_template` (`conv.stop_str` + `request.stop`). A
    // token-id stop cannot be merged into the string list (Python has no such
    // field), so it is kept alone.
    merge_template_stops(&mut request, &formatter);
    let rendered = render_chat(
        &formatter,
        body,
        &request,
        &state.server_args.default_chat_template_kwargs,
    )
    .map_err(|error| {
        openai_error(
            StatusCode::BAD_REQUEST,
            format!("chat template render failed: {error}"),
            false,
        )
    })?;
    Ok((request, rendered))
}

/// Full sampling resolution for an OpenAI request, mirroring the Python
/// handler: endpoint defaults → tool-choice validation + constraint → clamp.
/// The tool-choice checks run regardless of whether a parser is configured
/// (see `apply_tool_constraint`).
pub(super) fn chat_sampling(
    request: &CreateChatCompletionRequest,
    defaults: SamplingDefaults,
    parser: Option<&str>,
    tool_choice: &DynamoToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
    server_args: &ServerArgs,
) -> Result<SamplingParams, String> {
    let mut sampling = chat_sampling_params(
        request,
        &defaults.with_model_defaults(&server_args.model_config.default_sampling_params),
    )?;
    apply_tool_constraint(
        &mut sampling,
        parser,
        tool_choice,
        tools,
        parallel_tool_calls,
    )?;
    sampling
        .normalize(
            server_args.skip_tokenizer_init,
            server_args.model_config.vocab_size,
        )
        .map_err(|error| error.to_string())?;
    Ok(sampling)
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
        sglang_processor::OneOrMany::One(one) => vec![one],
        sglang_processor::OneOrMany::Many(many) => many,
    };
    if let Some(request_stop) = &request.stop {
        let Some(request_stops) = request_stop.strings() else {
            return;
        };
        stops.extend(request_stops);
    }
    request.stop = Some(Stop::StringArray(stops));
}

/// Where an omitted `temperature` / `top_p` gets its value. Mirrors Python's
/// `to_sampling_params` priority: user value > model generation_config (when
/// `--sampling-defaults model`) > OpenAI terminal default
/// (`_DEFAULT_SAMPLING_PARAMS`: chat uses 1.0/1.0).
pub(super) struct SamplingDefaults {
    /// Model defaults; `None` when the model config doesn't set them or when
    /// `--sampling-defaults openai` (the Python dump is then empty).
    temperature: Option<f64>,
    top_p: Option<f64>,
    /// OpenAI terminal defaults for chat completions.
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
    /// The resolved model defaults (empty in `--sampling-defaults openai`
    /// mode), which slot between the user's values and the OpenAI terminals.
    pub(super) fn with_model_defaults(mut self, model: &DefaultSamplingParams) -> SamplingDefaults {
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

/// How a choice's output is parsed.
pub(super) enum OutputParsing {
    /// `--reasoning-parser`, and `--tool-call-parser` over the request's tools,
    /// over the output text.
    Text {
        reasoning_parser: Option<String>,
        reasoning_options: ReasoningOptions,
        tool_parser: Option<String>,
        tools: Option<Vec<ToolDefinition>>,
    },
    /// recipe over the output token ids, in place of both parsers.
    DeepSeek(DeepSeekOutputOptions),
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn unary_chat(
    submitted: Vec<(usize, CoreCall)>,
    response_id: String,
    model: String,
    created: u32,
    want_logprobs: bool,
    parsing: OutputParsing,
    prompt_stub_tokens: u32,
    parallel_tool_calls: bool,
    service_tier: Option<ChatServiceTier>,
) -> Response {
    let mut choices = Vec::with_capacity(submitted.len());
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0u64;

    for (index, call) in submitted {
        let output = match collect_output(call).await {
            Ok(output) => output,
            Err((status, message)) => {
                return openai_error(status, message, false);
            }
        };

        if prompt_tokens == 0 {
            prompt_tokens = output.prompt_tokens.saturating_sub(prompt_stub_tokens);
        }
        completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
        let logprobs = want_logprobs.then(|| chat_logprobs(output.extras.as_deref()));
        let finish_reason = chat_finish_reason(&output);
        let (reasoning_text, content, tool_calls, finish_reason) = match &parsing {
            OutputParsing::DeepSeek(options) => {
                let delta = options.parser().finish(
                    &output.token_ids,
                    deepseek_finish(output.finish_reason.as_ref()),
                );
                let calls: Vec<_> = delta
                    .whole_tool_calls()
                    .into_iter()
                    .map(|call| ChatCompletionMessageToolCall {
                        id: tool_call_id(),
                        r#type: FunctionType::Function,
                        function: FunctionCall {
                            name: call.name,
                            arguments: call.arguments,
                        },
                    })
                    .collect();
                // `parse_chat_response`: calls turn only a stop into `tool_calls`.
                let finish_reason = match !calls.is_empty() && stopped(&output) {
                    true => Some(OpenAIFinishReason::ToolCalls),
                    false => finish_reason,
                };
                let calls = (!calls.is_empty()).then_some(calls);
                (delta.reasoning, delta.content, calls, finish_reason)
            }
            OutputParsing::Text {
                reasoning_parser,
                reasoning_options,
                tool_parser,
                tools,
            } => {
                // Split reasoning markers out of the content first (Python splits
                // before tool-call parsing too), then parse tool calls on the clean
                // normal text.
                let (reasoning_text, text) = split_reasoning(
                    reasoning_parser.as_deref(),
                    reasoning_options,
                    &output.text,
                    output.stop_trimmed_token_ids(),
                );
                let (content, tool_calls) = parse_chat_tool_calls(
                    text,
                    tool_parser.as_deref(),
                    tools.as_deref(),
                    parallel_tool_calls,
                )
                .await;
                let finish_reason = match tool_calls {
                    Some(_) => Some(OpenAIFinishReason::ToolCalls),
                    None => finish_reason,
                };
                (reasoning_text, content, tool_calls, finish_reason)
            }
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
    submitted: Vec<(usize, CoreCall)>,
    response_id: String,
    model: String,
    created: u32,
    want_logprobs: bool,
    include_usage: bool,
    parsing: OutputParsing,
    prompt_stub_tokens: u32,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
    parallel_tool_calls: bool,
    service_tier: Option<ChatServiceTier>,
) -> impl futures::Stream<Item = String> {
    let count = submitted.len();
    let (reasoning_parser, reasoning_options, parser, tools, deepseek) = match parsing {
        OutputParsing::Text {
            reasoning_parser,
            reasoning_options,
            tool_parser,
            tools,
        } => (
            reasoning_parser,
            reasoning_options,
            tool_parser,
            tools,
            None,
        ),
        OutputParsing::DeepSeek(options) => {
            (None, ReasoningOptions::default(), None, None, Some(options))
        }
    };
    let recipe_parses = deepseek.is_some();
    let raw = async_stream::stream! {
        let count = submitted.len();
        let mut streams = Vec::with_capacity(count);
        let mut prompt_tokens = 0u32;
        let mut completion_tokens = 0u64;
        // One stateful reasoning splitter per choice (Python keeps a
        // `reasoning_parser_dict` per index).
        let mut reasoning_splitters: Vec<ReasoningStreamSplitter> =
            if reasoning_parser.is_some() {
                (0..count)
                    .map(|_| ReasoningStreamSplitter::new(reasoning_parser.as_deref(), reasoning_options.clone()))
                    .collect()
            } else {
                vec![]
            };
        let reasoning_enabled = !reasoning_splitters.is_empty();
        let mut deepseek_parsers: Vec<DeepSeekOutputParser> = deepseek
            .map(|options| (0..count).map(|_| options.parser()).collect())
            .unwrap_or_default();

        for (index, call) in submitted {
            streams.push(indexed_decode_stream(index, call));
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
        while let Some((index, event)) = events.next().await {
            let output = match event {
                CoreEvent::Delta(output) | CoreEvent::Finished(output) => output,
                CoreEvent::Failed(error) => {
                    yield Annotated {
                        data: None,
                        id: None,
                        event: None,
                        comment: None,
                        error: Some(error_payload(error.http_status(), error.to_string()).to_string()),
                    };
                    continue;
                }
            };
            if prompt_tokens == 0 {
                prompt_tokens = output.prompt_tokens.saturating_sub(prompt_stub_tokens);
            }
            completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
            let mut finish_reason = chat_finish_reason(&output);
            let mut emitted = Vec::with_capacity(2);
            if let Some(parser) = deepseek_parsers.get_mut(index) {
                let finish = finish_reason.map(|_| deepseek_finish(output.finish_reason.as_ref()));
                let delta = parser.step(&output.token_ids, finish);
                let logprobs = want_logprobs.then(|| chat_logprobs(output.extras.as_deref()));
                emitted = deepseek_choices(index, delta, logprobs);
                if parser.has_tool_calls() && stopped(&output) {
                    finish_reason = Some(OpenAIFinishReason::ToolCalls);
                }
            } else if reasoning_enabled {
                // Split the step's text into (reasoning, normal) deltas when
                // `--reasoning-parser` is set. Mirrors Python's per-step emission:
                // reasoning chunk first (logprobs ride it), then the content chunk.
                let (reasoning_text, normal_text) =
                    reasoning_splitters[index].split(&output.text, output.stop_trimmed_token_ids());
                let mut remaining_logprobs =
                    want_logprobs.then(|| chat_logprobs(output.extras.as_deref()));
                if !reasoning_text.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(None, None, None, Some(reasoning_text)),
                        finish_reason: None,
                        logprobs: remaining_logprobs.clone(),
                    });
                    remaining_logprobs = None;
                }
                if !normal_text.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(Some(normal_text), None, None, None),
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
            // frames already became error chunks above). Both columns flush:
            // some parsers buffer the answer text until EOF.
            if reasoning_enabled && finish_reason.is_some() {
                let (reasoning_tail, normal_tail) = reasoning_splitters[index].finish();
                if !reasoning_tail.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(None, None, None, Some(reasoning_tail)),
                        finish_reason: None,
                        logprobs: None,
                    });
                }
                if !normal_tail.is_empty() {
                    emitted.push(ChatChoiceStream {
                        index: u32::try_from(index).unwrap_or(u32::MAX),
                        delta: chat_delta(Some(normal_tail), None, None, None),
                        finish_reason: None,
                        logprobs: None,
                    });
                }
            }
            // The finish reason rides the last emitted chunk (the wire format
            // the equivalence tests pin); a step whose text was entirely
            // buffered inside a text parser still gets a frame. recipe emits
            // nothing for a step it holds, as Python does.
            match emitted.last_mut() {
                Some(last) => last.finish_reason = finish_reason,
                None if recipe_parses && finish_reason.is_none() => {}
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
    > = if let Some(parser) = parser.filter(|_| !recipe_parses) {
        tool_call_stream(
            &parser,
            tool_choice,
            tools,
            uses_tool_call_structural_tag,
            raw,
        )
    } else {
        Box::pin(raw)
    };

    async_stream::stream! {
        // Each choice's first tool call, whose deltas (a name and argument
        // fragments, for some parsers) alone pass without parallel tool calls.
        let mut first_tool_call: Vec<Option<u32>> = vec![None; count];
        futures::pin_mut!(parsed);
        while let Some(mut item) = parsed.next().await {
            if let Some(response) = item.data.as_mut() {
                // recipe's calls pass whole, as Python's recipe path passes them.
                if !parallel_tool_calls && !recipe_parses {
                    for choice in &mut response.choices {
                        let index = choice.index as usize;
                        if let Some(calls) = choice.delta.tool_calls.as_mut() {
                            let seen = first_tool_call.get(index).copied().flatten();
                            let first = seen.or(calls.first().map(|call| call.index));
                            calls.retain(|call| Some(call.index) == first);
                            if let (Some(first), Some(slot)) = (first, first_tool_call.get_mut(index))
                                && !calls.is_empty()
                            {
                                *slot = Some(first);
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

/// `token_output_sse_chunks`: a reasoning chunk, a content chunk, then one
/// chunk per tool-call piece. The step's logprobs ride on the reasoning or
/// content chunk, else on a chunk of their own.
fn deepseek_choices(
    index: usize,
    delta: DeepSeekDelta,
    mut logprobs: Option<ChatChoiceLogprobs>,
) -> Vec<ChatChoiceStream> {
    let index = u32::try_from(index).unwrap_or(u32::MAX);
    let mut choices = Vec::new();
    let mut choice = |delta, logprobs| {
        choices.push(ChatChoiceStream {
            index,
            delta,
            finish_reason: None,
            logprobs,
        })
    };
    if !delta.reasoning.is_empty() {
        choice(
            chat_delta(None, None, None, Some(delta.reasoning)),
            logprobs.take(),
        );
    }
    if !delta.content.is_empty() {
        choice(
            chat_delta(Some(delta.content), None, None, None),
            logprobs.take(),
        );
    }
    for call in delta.tool_calls {
        // The first piece of a call names it; later pieces extend its arguments.
        let call = ChatCompletionMessageToolCallChunk {
            index: u32::try_from(call.index).unwrap_or(u32::MAX),
            id: call.name.is_some().then(tool_call_id),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: call.name,
                arguments: Some(call.arguments),
            }),
        };
        choice(chat_delta(None, None, Some(vec![call]), None), None);
    }
    if logprobs.is_some() {
        choice(chat_delta(None, None, None, None), logprobs);
    }
    choices
}

/// The scheduler's finish reason as recipe's parser takes it; a type other than
/// `stop` and `length` ends the parse without a backend finish.
fn deepseek_finish(reason: Option<&FinishReason>) -> DeepSeekFinish<'_> {
    match reason {
        Some(FinishReason::Known(FinishKind::Stop { matched })) => {
            DeepSeekFinish::Stop(match matched {
                Some(Matched::Token(id)) => Some(DeepSeekStopMatch::Token(*id)),
                Some(Matched::Str(stop)) => Some(DeepSeekStopMatch::Text(stop)),
                _ => None,
            })
        }
        Some(FinishReason::Known(FinishKind::Length { .. })) => DeepSeekFinish::Length,
        _ => DeepSeekFinish::Abort,
    }
}

/// Whether the scheduler stopped the choice, as `finish_reason["type"] == "stop"`;
/// an abort, which [`chat_finish_reason`] also reports as a stop, did not.
fn stopped(output: &CoreOutput) -> bool {
    matches!(
        output.finish_reason,
        Some(FinishReason::Known(FinishKind::Stop { .. }))
    )
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
            token_id: u32::try_from(token_id).ok(),
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
    use std::sync::Arc;

    use super::super::ChatFormatter;
    use super::super::test_utils::{
        answering_app_state_with, answering_app_state_with_item, body_json, chat_submitted, chunk,
        post_json,
    };
    use super::{
        OutputParsing, SamplingDefaults, chat_event_stream, chat_logprobs, chat_sampling_params,
        merge_template_stops, unary_chat,
    };
    use crate::message::config::{DefaultSamplingParams, ServerArgs};
    use crate::message::response::{ChunkEvent, ChunkExtras, ResponseItem};
    use axum::http::StatusCode;
    use futures::StreamExt;
    use sglang_processor::dynamo_protocols::types::{CreateChatCompletionRequest, Stop};
    use sglang_processor::dynamo_tokenizers::Tokenizer;
    use sglang_processor::openai::ChatModel;
    use sglang_processor::{DeepSeekOutputOptions, ReasoningOptions};

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

    /// Python `_apply_conversation_template`: template `stop_str` first, then
    /// the request's own stops.
    #[test]
    fn template_stops_merge_before_request_stops() {
        let formatter =
            sglang_processor::load_chat_formatter(None, None, None, Some("chatml")).unwrap();
        assert_eq!(
            formatter.stop_strs(),
            Some(sglang_processor::OneOrMany::Many(vec![
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

    /// A formatter without template stops (Python's jinja path keeps only the
    /// request's stops) leaves the request unchanged.
    #[test]
    fn formatter_without_stops_leaves_request_stops_alone() {
        let mut req = request();
        req.stop = Some(Stop::String("x".into()));
        merge_template_stops(&mut req, &ChatFormatter::KimiK3);
        assert_eq!(req.stop, Some(Stop::String("x".into())));
    }

    /// A request with no `max_tokens`/`max_completion_tokens` stays unbounded —
    /// no terminal default is imposed.
    #[test]
    fn chat_without_a_token_limit_stays_unbounded() {
        let request: CreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        assert_eq!(
            chat_sampling_params(&request, &SamplingDefaults::CHAT)
                .unwrap()
                .max_new_tokens,
            None
        );
    }

    #[test]
    fn chat_logprobs_use_dynamo_wire_types() {
        let extras = ChunkExtras {
            out_lp_val: vec![-0.25],
            out_lp_idx: vec![7],
            out_lp_txt: vec!["x".into()],
            out_top_val: vec![-0.25, -1.0],
            out_top_idx: vec![7, 8],
            out_top_lens: vec![2],
            out_top_txt: vec!["x".into(), "y".into()],
            ..Default::default()
        };
        let logprobs = chat_logprobs(Some(&extras));
        let token = &logprobs.content.unwrap()[0];
        assert_eq!(token.token, "x");
        assert_eq!(token.token_id, Some(7));
        assert_eq!(token.top_logprobs.len(), 2);
        assert_eq!(token.top_logprobs[1].token, "y");
    }

    #[tokio::test]
    async fn unary_chat_fans_in_choices_and_usage() {
        let (choice0, tx0) = chat_submitted(0, "r0");
        let (choice1, tx1) = chat_submitted(1, "r1");
        tx0.send(chunk("r0", "Paris", true)).await.unwrap();
        tx1.send(chunk("r1", "Paris", true)).await.unwrap();

        let response = unary_chat(
            vec![choice0, choice1],
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            text_parsing(None),
            3, // Kimi K3's generation stub
            true,
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["choices"][0]["message"]["role"], "assistant");
        assert_eq!(value["choices"][0]["message"]["content"], "Paris");
        assert_eq!(value["choices"][1]["index"], 1);
        assert_eq!(
            value["usage"]["prompt_tokens"], 2,
            "the stub is not reported"
        );
        assert_eq!(value["usage"]["completion_tokens"], 2);
    }

    /// The first choice's message for `body`, against a core answering `text`.
    /// Parsing by `reasoning_parser` alone, without tools.
    fn text_parsing(reasoning_parser: Option<&str>) -> OutputParsing {
        OutputParsing::Text {
            reasoning_parser: reasoning_parser.map(str::to_owned),
            reasoning_options: ReasoningOptions::default(),
            tool_parser: None,
            tools: None,
        }
    }

    async fn chat_message(
        reasoning_parser: &str,
        formatter: ChatFormatter,
        text: &'static str,
        body: serde_json::Value,
    ) -> serde_json::Value {
        let args = ServerArgs {
            served_model_name: "model".into(),
            reasoning_parser: Some(reasoning_parser.into()),
            ..Default::default()
        };
        let state = answering_app_state_with(Arc::new(args), formatter, None, text);
        let app = super::routes().with_state(state);
        let response = post_json(app, "/v1/chat/completions", body).await;
        body_json(response).await["choices"][0]["message"].clone()
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
    ) -> ResponseItem {
        let ids = tokenizer.encode(text).unwrap().token_ids().to_vec();
        ResponseItem::Done(ChunkEvent {
            rid: rid.into(),
            text: text.into(),
            completion_tokens: ids.len() as u64,
            token_ids: ids.into_iter().map(i64::from).collect(),
            prompt_tokens: 5,
            finish_reason: Some(serde_json::from_value(finish).unwrap()),
            ..Default::default()
        })
    }

    /// recipe's parse of `body`'s output, under the V4.1 parsers.
    fn deepseek_options(
        formatter: &ChatFormatter,
        body: serde_json::Value,
    ) -> DeepSeekOutputOptions {
        let model = ChatModel::for_formatter(formatter);
        let output = model.deepseek.expect("the tokenizer is attached");
        output
            .for_request(&body, &Default::default(), true, true)
            .unwrap()
    }

    const DSML_CALL: &str = "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"get_weather\">\n\
        <｜DSML｜ parameter name=\"city\" string=\"true\">Paris</｜DSML｜ parameter>\n\
        </｜DSML｜ invoke>\n</｜DSML｜ calls>";

    /// A continuation that already closed its think block generates content.
    #[tokio::test]
    async fn continuation_after_think_end_is_content() {
        let Some((formatter, tokenizer)) = deepseek_v41() else {
            return;
        };
        let body = serde_json::json!({
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
        let stop = serde_json::json!({"type": "stop"});
        let state = answering_app_state_with_item(Arc::new(args), formatter, None, move |rid| {
            ids_chunk(rid, &tokenizer, " 391.", stop.clone())
        });
        let app = super::super::routes().with_state(state);
        let response = body_json(post_json(app, "/v1/chat/completions", body).await).await;
        let message = &response["choices"][0]["message"];
        assert_eq!(message["content"], " 391.");
        assert!(message["reasoning_content"].is_null());
    }

    /// A call `max_tokens` cuts stays a `length` finish, as Python reports it.
    #[tokio::test]
    async fn deepseek_call_cut_by_length_is_not_a_tool_calls_finish() {
        let Some((formatter, tokenizer)) = deepseek_v41() else {
            return;
        };
        let body = serde_json::json!({"messages": [{"role": "user", "content": "Weather?"}]});
        let (choice, tx) = chat_submitted(0, "r0");
        let cut = &DSML_CALL[..DSML_CALL.find("Paris").unwrap()];
        let length = serde_json::json!({"type": "length", "length": 16});
        tx.send(ids_chunk("r0", &tokenizer, cut, length))
            .await
            .unwrap();
        let parsing = OutputParsing::DeepSeek(deepseek_options(&formatter, body));
        let response = unary_chat(
            vec![choice],
            "id".into(),
            "model".into(),
            1,
            false,
            parsing,
            0,
            true,
            None,
        )
        .await;
        let value = body_json(response).await;
        assert_eq!(value["choices"][0]["finish_reason"], "length");
        let call = &value["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["function"]["name"], "get_weather");
    }

    /// A streamed call is named once, its id on that first piece, and every
    /// frame carries a delta or the finish.
    #[tokio::test]
    async fn deepseek_stream_names_a_call_once() {
        let Some((formatter, tokenizer)) = deepseek_v41() else {
            return;
        };
        let body = serde_json::json!({"messages": [{"role": "user", "content": "Weather?"}]});
        let (choice, tx) = chat_submitted(0, "r0");
        let ids = tokenizer.encode(DSML_CALL).unwrap().token_ids().to_vec();
        // One frame per token, then the finish.
        tokio::spawn(async move {
            for id in ids {
                let frame = ChunkEvent {
                    rid: "r0".into(),
                    token_ids: vec![i64::from(id)],
                    completion_tokens: 1,
                    ..Default::default()
                };
                tx.send(ResponseItem::Frame(frame)).await.unwrap();
            }
            let stop = serde_json::json!({"type": "stop"});
            tx.send(ids_chunk("r0", &tokenizer, "", stop))
                .await
                .unwrap();
        });
        let parsing = OutputParsing::DeepSeek(deepseek_options(&formatter, body));
        let stream = chat_event_stream(
            vec![choice],
            "id".into(),
            "model".into(),
            1,
            false,
            false,
            parsing,
            0,
            None,
            false,
            false,
            None,
        );
        let frames: Vec<serde_json::Value> = stream
            .filter_map(|frame| async move { serde_json::from_str(&frame).ok() })
            .collect()
            .await;
        let choices: Vec<&serde_json::Value> = frames.iter().map(|f| &f["choices"][0]).collect();
        let pieces: Vec<&serde_json::Value> = choices
            .iter()
            .filter_map(|c| c["delta"]["tool_calls"].get(0))
            .collect();
        assert_eq!(pieces[0]["function"]["name"], "get_weather");
        assert!(pieces[0]["id"].is_string());
        for piece in &pieces[1..] {
            assert_eq!(piece["index"], 0);
            assert!(piece.get("id").is_none_or(|id| id.is_null()));
        }
        assert_eq!(choices.last().unwrap()["finish_reason"], "tool_calls");
        let carries_something = |choice: &&serde_json::Value| {
            let delta = &choice["delta"];
            ["role", "content", "reasoning_content", "tool_calls"]
                .iter()
                .any(|field| !delta[field].is_null())
                || !choice["finish_reason"].is_null()
        };
        assert!(choices.iter().all(carries_something));
    }

    /// Kimi K3 with thinking off generates content, though its detector defaults to reasoning.
    #[tokio::test]
    async fn kimi_k3_thinking_off_is_content() {
        let messages = serde_json::json!([{"role": "user", "content": "Hi"}]);
        for body in [
            serde_json::json!({"model": "model", "messages": messages, "chat_template_kwargs": {"thinking": false}}),
            serde_json::json!({"model": "model", "messages": messages, "reasoning_effort": "none"}),
        ] {
            let message = chat_message("kimi_k3", ChatFormatter::KimiK3, "Hello!", body).await;
            assert_eq!(message["content"], "Hello!");
            assert!(message["reasoning_content"].is_null());
        }
    }

    #[tokio::test]
    async fn unary_chat_separates_reasoning_content_with_parser_configured() {
        let (choice, tx) = chat_submitted(0, "r0");
        tx.send(chunk(
            "r0",
            "<think>because Paris is famous</think>Paris",
            true,
        ))
        .await
        .unwrap();

        let response = unary_chat(
            vec![choice],
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            text_parsing(Some("deepseek-r1")),
            0,
            true,
            None,
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            value["choices"][0]["message"]["reasoning_content"],
            "because Paris is famous"
        );
        assert_eq!(value["choices"][0]["message"]["content"], "Paris");
        assert!(value["choices"][0]["message"]["reasoning_content"].is_string());
    }

    #[tokio::test]
    async fn streaming_chat_separates_reasoning_into_own_deltas() {
        let (choice, tx) = chat_submitted(0, "r0");
        // Force mode starts in reasoning, so the opener is stripped and the first
        // reasoning fragment streams immediately.
        tx.send(chunk("r0", "<think>be", false)).await.unwrap();
        tx.send(chunk("r0", "cause</think>Par", false))
            .await
            .unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();

        let stream = chat_event_stream(
            vec![choice],
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            true,
            text_parsing(Some("deepseek-r1")),
            0,
            None,
            false,
            true,
            None,
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        let role: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
        let first_reasoning: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
        let second_reasoning: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
        let content: serde_json::Value = serde_json::from_str(&frames[3]).unwrap();
        let terminal: serde_json::Value = serde_json::from_str(&frames[4]).unwrap();
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(
            first_reasoning["choices"][0]["delta"]["reasoning_content"],
            "be"
        );
        assert!(first_reasoning["choices"][0]["delta"]["content"].is_null());
        assert_eq!(
            second_reasoning["choices"][0]["delta"]["reasoning_content"],
            "cause"
        );
        assert_eq!(content["choices"][0]["delta"]["content"], "Par");
        assert!(content["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["delta"]["content"], "is");
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert_eq!(frames.len(), 7);
    }

    #[tokio::test]
    async fn streaming_chat_emits_role_deltas_usage_and_done() {
        let (choice, tx) = chat_submitted(0, "r0");
        tx.send(chunk("r0", "Par", false)).await.unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();

        let stream = chat_event_stream(
            vec![choice],
            "chatcmpl-test".into(),
            "model".into(),
            1,
            false,
            true,
            text_parsing(None),
            0,
            None,
            false,
            true,
            None,
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.collect().await;
        assert_eq!(frames.len(), 5);
        let role: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
        let delta: serde_json::Value = serde_json::from_str(&frames[1]).unwrap();
        let terminal: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
        let usage: serde_json::Value = serde_json::from_str(&frames[3]).unwrap();
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert!(role["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(delta["choices"][0]["delta"]["content"], "Par");
        assert!(delta["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["delta"]["content"], "is");
        assert!(terminal["choices"][0]["delta"]["reasoning_content"].is_null());
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert_eq!(usage["usage"]["completion_tokens"], 2);
        assert_eq!(frames[4], "[DONE]");
    }

    // A named dynamic tool must pass request validation and reach both wire formats.
    #[tokio::test]
    async fn system_message_tools_reach_http_responses() {
        use super::super::test_utils::{body_json, post_json};
        use crate::api_server::handlers::{CoreConfig, CoreHandle, CoreMetadata};
        use crate::message::config::ServerArgs;
        use crate::message::request::RequestKind;
        use crate::tokenizer_manager::wiring::TmEvent;
        use std::sync::Arc;

        for stream in [false, true] {
            let args = Arc::new(ServerArgs {
                served_model_name: "model".into(),
                tool_call_parser: Some("kimi_k3".into()),
                ..Default::default()
            });
            let (intake_tx, intake_rx) = flume::unbounded();
            let core = CoreHandle::new(
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
                    metadata: CoreMetadata::from(args.as_ref()),
                    metrics: None,
                },
            );
            let state = Arc::new(super::super::AppState {
                core,
                generation_policy: crate::message::request::GenerationPolicy::from(args.as_ref()),
                server_args: args,
                chat_formatter: Some(super::super::ChatFormatter::KimiK3),
                generate_hook: None,
                openai_chat: None,
            });
            let responder = tokio::spawn(async move {
                let TmEvent::Intake { request, admission } = intake_rx.recv_async().await.unwrap()
                else {
                    panic!("expected generation intake");
                };
                assert!(admission.try_accept());
                let RequestKind::Generate(generate) = &request.kind else {
                    panic!("expected generation");
                };
                let prompt = generate.chat_prompt.as_ref().unwrap();
                assert!(prompt.prompt.as_str().contains("lookup"));
                assert!(!generate.sampling_params.skip_special_tokens);
                assert!(
                    generate
                        .sampling_params
                        .structural_tag
                        .as_ref()
                        .unwrap()
                        .contains("lookup")
                );
                request.sink.try_send(chunk(request.rid.as_str(),
                    "<|open|>tools<|sep|><|open|>call tool=\"lookup\" index=\"1\"<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>", true)).unwrap();
            });
            let response = post_json(super::super::routes().with_state(state), "/v1/chat/completions", serde_json::json!({
                "model":"model", "stream":stream,
                "messages":[{"role":"system","tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}]},
                            {"role":"user","content":"Call lookup"}],
                "tool_choice":{"type":"function","function":{"name":"lookup"}},
                "chat_template_kwargs":{"thinking":false}
            })).await;
            assert_eq!(response.status(), StatusCode::OK);
            if stream {
                let bytes = axum::body::to_bytes(response.into_body(), 65536)
                    .await
                    .unwrap();
                let text = std::str::from_utf8(&bytes).unwrap();
                let frames: Vec<serde_json::Value> = text
                    .lines()
                    .filter_map(|line| {
                        line.strip_prefix("data: ")
                            .filter(|data| *data != "[DONE]")
                            .map(|data| serde_json::from_str(data).unwrap())
                    })
                    .collect();
                assert!(
                    frames.iter().any(
                        |frame| frame["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
                            == "lookup"
                    ),
                    "{text}"
                );
                assert!(
                    frames
                        .iter()
                        .any(|frame| frame["choices"][0]["finish_reason"] == "tool_calls"),
                    "{text}"
                );
            } else {
                let body = body_json(response).await;
                assert_eq!(
                    body["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "lookup",
                    "{body}"
                );
                assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
            }
            responder.await.unwrap();
        }
    }
}
