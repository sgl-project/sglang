//! The transport-neutral half of OpenAI Chat Completions: template
//! preparation, sampling resolution, the unary fan-in, and the typed chunk
//! stream (reasoning split + tool-call jail). The HTTP handler stays in
//! `api_server::openai::chat`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use dynamo_parsers::tool_calling::jail::{Annotated, apply_tool_calling_jail};
use dynamo_parsers::{ToolChoice as DynamoToolChoice, ToolDefinition};
use dynamo_protocols::types::{
    ChatChoice, ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionMessageContent,
    ChatCompletionResponseMessage, ChatCompletionStreamOptions, ChatCompletionTokenLogprob,
    ChatCompletionToolChoiceOption, CreateChatCompletionRequest, CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse, FinishReason as OpenAIFinishReason, ResponseFormat, Role,
    ServiceTier as ChatServiceTier, Stop, TopLogprobs,
};
use futures::StreamExt;

use crate::api_server::core::error::ApiError;
use crate::api_server::core::event::CoreEvent;
use crate::api_server::core::frame::OutputAccumulator;
use crate::api_server::core::generate::{
    FrameShaper, GeneratePlan, RequestTiming, UnaryDrainPolicy, drain_plan_unary,
    generation_event_stream_with, unary_output,
};
use crate::api_server::core::openai::completions::completion_usage;
use crate::api_server::core::openai::matched_stop_value;
use crate::api_server::core::openai::reasoning::ReasoningStreamSplitter;
use crate::api_server::core::openai::template::ChatFormatter;
use crate::api_server::core::openai::tools::{
    apply_tool_constraint, chat_delta, chat_finish_reason, dynamo_parser_name,
    parse_chat_tool_calls,
};
use crate::api_server::core::openai::{UsageDetails, usage_value, weight_metadata_value};
use crate::api_server::core::state::CoreState;
use crate::message::config::{DefaultSamplingParams, ServerArgs};
use crate::message::response::{ChunkEvent, ChunkExtras};
use crate::message::sampling::SamplingParams;
use crate::message::types::OneOrMany;

/// Python's `should_include_usage`: with `stream_options` present the client
/// picks `continuous_usage_stats` and may only raise `include_usage` through
/// the server default; without options both come from defaults.
pub(crate) fn chat_stream_usage_options(
    options: Option<&ChatCompletionStreamOptions>,
    include_usage_default: bool,
) -> (bool, bool) {
    match options {
        Some(options) => (
            options.include_usage || include_usage_default,
            options.continuous_usage_stats,
        ),
        None => (include_usage_default, false),
    }
}

/// Render the chat template for an OpenAI request, mapping a missing
/// formatter or a render failure to the standard 400. The rendered prompt is
/// submitted as text — the tokenizer pool encodes it (with
/// `skip_special_tokens`, since the template owns its special tokens).
pub(crate) async fn prepare_chat_request(
    state: &CoreState,
    mut request: CreateChatCompletionRequest,
) -> Result<(CreateChatCompletionRequest, String), ApiError> {
    let Some(formatter) = state.chat_formatter.clone() else {
        return Err(ApiError::bad_request(
            "this model has no usable chat template",
        ));
    };
    // Template stops first, then the request's own — Python
    // `_apply_conversation_template` (`conv.stop_str` + `request.stop`). A
    // token-id stop cannot be merged into the string list (Python has no such
    // field), so it is kept alone.
    merge_template_stops(&mut request, &formatter);
    let prompt = formatter
        .render(&request)
        .map_err(|error| ApiError::bad_request(format!("chat template render failed: {error}")))?;
    Ok((request, prompt))
}

/// Full sampling resolution for an OpenAI request, mirroring the Python
/// handler: endpoint defaults → tool-choice validation + constraint → clamp.
/// The tool-choice checks run regardless of whether a parser is configured
/// (see `apply_tool_constraint`).
pub(crate) fn chat_sampling(
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
        OneOrMany::One(one) => vec![one],
        OneOrMany::Many(many) => many,
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
pub(crate) struct SamplingDefaults {
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
    pub(crate) const CHAT: SamplingDefaults = SamplingDefaults {
        temperature: None,
        top_p: None,
        fallback_temperature: 1.0,
        fallback_top_p: 1.0,
    };
    /// The resolved model defaults (empty in `--sampling-defaults openai`
    /// mode), which slot between the user's values and the OpenAI terminals.
    pub(crate) fn with_model_defaults(mut self, model: &DefaultSamplingParams) -> SamplingDefaults {
        self.temperature = model.temperature;
        self.top_p = model.top_p;
        self
    }
}

#[allow(deprecated)]
pub(crate) fn chat_sampling_params(
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

pub(crate) struct ChatRenderingOptions {
    pub(crate) response_id: String,
    pub(crate) model: String,
    pub(crate) created: u32,
    pub(crate) want_logprobs: bool,
    pub(crate) parser: Option<String>,
    pub(crate) reasoning_parser: Option<String>,
    pub(crate) tools: Option<Vec<ToolDefinition>>,
    pub(crate) parallel_tool_calls: bool,
    pub(crate) service_tier: Option<ChatServiceTier>,
    /// `--enable-cache-report`: OpenAI usage exposes
    /// `prompt_tokens_details.cached_tokens` only when enabled.
    pub(crate) enable_cache_report: bool,
    /// Launch-time `--weight-version`, the scalar fallback when the request has
    /// no weight-version spans.
    pub(crate) weight_version: Option<String>,
}

/// The unary fan-in returns the serialized response so the native matched stop
/// can be added per choice: Dynamo's `ChatChoice` has no slot for it.
pub(crate) async fn unary_chat(
    plan: GeneratePlan,
    options: ChatRenderingOptions,
) -> Result<serde_json::Value, ApiError> {
    let drained = drain_plan_unary(plan, UnaryDrainPolicy::AggregateFailFast).await?;
    let mut choices = Vec::with_capacity(drained.len());
    let mut matched_stops = Vec::with_capacity(drained.len());
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0u64;
    let mut reasoning_tokens = 0u32;
    // Python takes the cached count from the prompt representative (choice 0);
    // every choice shares the one rendered prompt.
    let mut cached_tokens = None;
    let mut weight_spans = None;

    for (index, (_, outcome)) in drained.into_iter().enumerate() {
        let output = unary_output(outcome)?;

        if prompt_tokens == 0 {
            prompt_tokens = output.prompt_tokens;
        }
        completion_tokens = completion_tokens.saturating_add(output.completion_tokens);
        if let Some(extras) = output.extras.as_deref() {
            reasoning_tokens = reasoning_tokens.saturating_add(extras.reasoning_tokens);
            if index == 0 && extras.cached_tokens != 0 {
                cached_tokens = Some(extras.cached_tokens);
            }
            // Python reads metadata from the first response only.
            if index == 0 {
                weight_spans = extras.weight_versions.clone();
            }
        }
        matched_stops.push(output.finish_reason.as_ref().and_then(matched_stop_value));
        let logprobs = options
            .want_logprobs
            .then(|| chat_logprobs(output.extras.as_deref()));
        let finish_reason = chat_finish_reason(&output);
        // Split reasoning markers out of the content first (Python splits
        // before tool-call parsing too), then parse tool calls on the clean
        // normal text.
        let (reasoning_text, text) =
            ReasoningStreamSplitter::new(options.reasoning_parser.as_deref())
                .split_complete(&output.text, &output.token_ids);
        let (content, tool_calls) = parse_chat_tool_calls(
            text,
            options.parser.as_deref(),
            options.tools.as_deref(),
            options.parallel_tool_calls,
        )
        .await;
        let finish_reason = if tool_calls.is_some() {
            Some(OpenAIFinishReason::ToolCalls)
        } else {
            finish_reason
        };
        #[allow(deprecated)]
        let message = ChatCompletionResponseMessage {
            // Python always emits `content` as a string (`text if text else ""`).
            content: Some(ChatCompletionMessageContent::Text(content)),
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

    let usage = completion_usage(
        prompt_tokens,
        u32::try_from(completion_tokens).unwrap_or(u32::MAX),
    );
    let response = CreateChatCompletionResponse {
        id: options.response_id,
        choices,
        created: options.created,
        model: options.model,
        service_tier: options.service_tier,
        system_fingerprint: None,
        object: "chat.completion".into(),
        usage: None,
    };
    // Add the native matched stop per choice; Dynamo's `ChatChoice` has no slot
    // for it. Python always emits the key, null when nothing matched. The
    // message shape is aligned with Python's `ChatMessage`: `tool_calls` is
    // always present (null when none) with a per-call `index`, and its
    // logprobs have no `refusal` key.
    let mut value = serde_json::to_value(response).expect("OpenAI response must serialize");
    if let Some(choices) = value["choices"].as_array_mut() {
        for (choice, matched_stop) in choices.iter_mut().zip(&matched_stops) {
            choice["matched_stop"] = matched_stop.clone().unwrap_or(serde_json::Value::Null);
            if let Some(message) = choice.get_mut("message") {
                match message["tool_calls"].as_array_mut() {
                    Some(calls) => {
                        for (index, call) in calls.iter_mut().enumerate() {
                            call["index"] = serde_json::json!(index);
                        }
                    }
                    None => message["tool_calls"] = serde_json::Value::Null,
                }
            }
            if let Some(logprobs) = choice
                .get_mut("logprobs")
                .and_then(serde_json::Value::as_object_mut)
            {
                logprobs.remove("refusal");
            }
        }
    }
    value["usage"] = usage_value(
        usage,
        UsageDetails {
            reasoning_tokens,
            cached_tokens: options
                .enable_cache_report
                .then_some(cached_tokens)
                .flatten(),
        },
    );
    value["metadata"] =
        weight_metadata_value(weight_spans.as_deref(), options.weight_version.as_deref());
    Ok(value)
}

/// Accumulated usage and native matched stops for one chat stream. The tool
/// jail can drop, rewrite, or synthesize chunks after the shaper, so the
/// post-jail loop attaches usage and matched stops from this snapshot: a
/// generation event is counted once per choice even when it expands into
/// several chunks.
#[derive(Default)]
struct ChatStreamState {
    prompt_tokens: Option<u32>,
    completion_tokens: Vec<u64>,
    matched_stop: Vec<Option<serde_json::Value>>,
    /// Latest reasoning-count snapshot per choice (Python sums them at the end).
    reasoning_tokens: Vec<u32>,
    /// Cached prompt count per choice; continuous chunks report the current
    /// choice's count, the trailer the prompt representative (choice 0).
    cached_tokens: Vec<u32>,
}

impl ChatStreamState {
    fn new(count: usize) -> Self {
        Self {
            completion_tokens: vec![0; count],
            matched_stop: vec![None; count],
            reasoning_tokens: vec![0; count],
            cached_tokens: vec![0; count],
            ..Default::default()
        }
    }

    /// Record one generation event for `choice_index`; `completion_tokens` is
    /// the choice's running count from the shared accumulator.
    fn record(&mut self, choice_index: usize, output: &ChunkEvent, completion_tokens: u64) {
        // First non-zero count wins: an early frame may carry 0 before the
        // scheduler reports the real prompt token count.
        if self.prompt_tokens.unwrap_or(0) == 0 {
            self.prompt_tokens = Some(output.prompt_tokens);
        }
        self.completion_tokens[choice_index] = completion_tokens;
        if let Some(extras) = output.extras.as_deref() {
            if extras.reasoning_tokens != 0 {
                self.reasoning_tokens[choice_index] = extras.reasoning_tokens;
            }
            if extras.cached_tokens != 0 {
                self.cached_tokens[choice_index] = extras.cached_tokens;
            }
        }
        if output.finish_reason.is_some() {
            self.matched_stop[choice_index] =
                output.finish_reason.as_ref().and_then(matched_stop_value);
        }
    }

    fn choice_usage(&self, choice_index: usize) -> dynamo_protocols::types::CompletionUsage {
        completion_usage(
            self.prompt_tokens.unwrap_or_default(),
            u32::try_from(
                self.completion_tokens
                    .get(choice_index)
                    .copied()
                    .unwrap_or(0),
            )
            .unwrap_or(u32::MAX),
        )
    }

    fn total_usage(&self) -> dynamo_protocols::types::CompletionUsage {
        completion_usage(
            self.prompt_tokens.unwrap_or_default(),
            u32::try_from(
                self.completion_tokens
                    .iter()
                    .copied()
                    .fold(0u64, u64::saturating_add),
            )
            .unwrap_or(u32::MAX),
        )
    }

    fn choice_reasoning(&self, choice_index: usize) -> u32 {
        self.reasoning_tokens
            .get(choice_index)
            .copied()
            .unwrap_or(0)
    }

    fn choice_cached(&self, choice_index: usize) -> u32 {
        self.cached_tokens.get(choice_index).copied().unwrap_or(0)
    }

    /// The prompt representative's cached count (Python `idx % n == 0`).
    fn representative_cached(&self) -> u32 {
        self.choice_cached(0)
    }

    fn total_reasoning(&self) -> u32 {
        self.reasoning_tokens
            .iter()
            .copied()
            .fold(0u32, u32::saturating_add)
    }
}

struct ChatStreamFrameShaper {
    response_id: String,
    model: String,
    created: u32,
    want_logprobs: bool,
    reasoning_splitters: Vec<ReasoningStreamSplitter>,
    prelude_emitted: Vec<bool>,
    state: Arc<Mutex<ChatStreamState>>,
    service_tier: Option<ChatServiceTier>,
}

impl ChatStreamFrameShaper {
    fn annotated(
        data: Option<CreateChatCompletionStreamResponse>,
        error: Option<String>,
    ) -> Annotated<CreateChatCompletionStreamResponse> {
        Annotated {
            data,
            error,
            id: None,
            event: None,
            comment: None,
        }
    }

    fn response(
        &self,
        choices: Vec<ChatChoiceStream>,
        usage: Option<dynamo_protocols::types::CompletionUsage>,
    ) -> Annotated<CreateChatCompletionStreamResponse> {
        Self::annotated(
            Some(CreateChatCompletionStreamResponse {
                id: self.response_id.clone(),
                choices,
                created: self.created,
                model: self.model.clone(),
                service_tier: self.service_tier.clone(),
                system_fingerprint: None,
                object: "chat.completion.chunk".into(),
                usage,
            }),
            None,
        )
    }

    fn error_frame(&self, error: &ApiError) -> Annotated<CreateChatCompletionStreamResponse> {
        Self::annotated(None, Some(encode_stream_error(error)))
    }

    fn role_prelude(
        &mut self,
        index: usize,
    ) -> Option<Annotated<CreateChatCompletionStreamResponse>> {
        if std::mem::replace(&mut self.prelude_emitted[index], true) {
            return None;
        }
        Some(self.response(
            vec![ChatChoiceStream {
                index: u32::try_from(index).unwrap_or(u32::MAX),
                delta: chat_delta(None, Some(Role::Assistant), None, None),
                finish_reason: None,
                logprobs: None,
            }],
            None,
        ))
    }

    fn render_output(
        &mut self,
        choice_index: usize,
        mut output: ChunkEvent,
        completion_tokens: u64,
        terminal: bool,
    ) -> Vec<Annotated<CreateChatCompletionStreamResponse>> {
        let index = u32::try_from(choice_index).unwrap_or(u32::MAX);
        self.state
            .lock()
            .expect("chat stream state poisoned")
            .record(choice_index, &output, completion_tokens);
        let finish_reason = if terminal {
            chat_finish_reason(&output)
        } else {
            None
        };
        let reasoning_enabled = !self.reasoning_splitters.is_empty();
        let mut emitted = Vec::with_capacity(2);
        if reasoning_enabled {
            let (reasoning_text, normal_text) =
                self.reasoning_splitters[choice_index].split(&output.text, &output.token_ids);
            let mut remaining_logprobs = self
                .want_logprobs
                .then(|| chat_logprobs(output.extras.as_deref()));
            if !reasoning_text.is_empty() {
                emitted.push(ChatChoiceStream {
                    index,
                    delta: chat_delta(None, None, None, Some(reasoning_text)),
                    finish_reason: None,
                    logprobs: remaining_logprobs.take(),
                });
            }
            if !normal_text.is_empty() {
                emitted.push(ChatChoiceStream {
                    index,
                    delta: chat_delta(Some(normal_text), None, None, None),
                    finish_reason: None,
                    logprobs: remaining_logprobs,
                });
            }
        } else {
            emitted.push(ChatChoiceStream {
                index,
                delta: chat_delta(
                    (!output.text.is_empty()).then_some(std::mem::take(&mut output.text)),
                    None,
                    None,
                    None,
                ),
                finish_reason: None,
                logprobs: self
                    .want_logprobs
                    .then(|| chat_logprobs(output.extras.as_deref())),
            });
        }
        if terminal && reasoning_enabled && finish_reason.is_some() {
            let (reasoning_tail, normal_tail) = self.reasoning_splitters[choice_index].finish();
            if !reasoning_tail.is_empty() {
                emitted.push(ChatChoiceStream {
                    index,
                    delta: chat_delta(None, None, None, Some(reasoning_tail)),
                    finish_reason: None,
                    logprobs: None,
                });
            }
            if !normal_tail.is_empty() {
                emitted.push(ChatChoiceStream {
                    index,
                    delta: chat_delta(Some(normal_tail), None, None, None),
                    finish_reason: None,
                    logprobs: None,
                });
            }
        }
        match emitted.last_mut() {
            Some(last) => last.finish_reason = finish_reason,
            None => emitted.push(ChatChoiceStream {
                index,
                delta: chat_delta(None, None, None, None),
                finish_reason,
                logprobs: None,
            }),
        }
        self.role_prelude(choice_index)
            .into_iter()
            .chain(
                emitted
                    .into_iter()
                    .map(|choice| self.response(vec![choice], None)),
            )
            .collect()
    }
}

impl FrameShaper for ChatStreamFrameShaper {
    type Frame = Vec<Annotated<CreateChatCompletionStreamResponse>>;

    fn delta(
        &mut self,
        out: ChunkEvent,
        acc: &OutputAccumulator,
        _rid: &str,
        index: Option<usize>,
    ) -> Self::Frame {
        self.render_output(
            index.unwrap_or(0),
            out,
            acc.snapshot().completion_tokens,
            false,
        )
    }

    fn coalesced(
        &mut self,
        acc: &OutputAccumulator,
        _rid: &str,
        index: Option<usize>,
    ) -> Self::Frame {
        self.delta(acc.snapshot().clone(), acc, "", index)
    }

    fn terminal(
        &mut self,
        out: ChunkEvent,
        acc: &OutputAccumulator,
        _incremental: bool,
        _rid: &str,
        index: Option<usize>,
        _timing: &RequestTiming,
    ) -> Self::Frame {
        self.render_output(
            index.unwrap_or(0),
            out,
            acc.snapshot().completion_tokens,
            true,
        )
    }

    fn item_error(&mut self, code: u16, message: &str, index: Option<usize>) -> Self::Frame {
        let mut frames: Vec<_> = self.role_prelude(index.unwrap_or(0)).into_iter().collect();
        frames.push(self.error_frame(&ApiError::new(code, message)));
        frames
    }
}

pub(crate) fn chat_event_stream(
    plan: GeneratePlan,
    options: ChatRenderingOptions,
    include_usage: bool,
    continuous_usage: bool,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
    uses_tool_call_structural_tag: bool,
) -> impl futures::Stream<Item = CoreEvent<serde_json::Value>> {
    let count = plan.receivers.len();
    let ChatRenderingOptions {
        response_id,
        model,
        created,
        want_logprobs,
        parser,
        reasoning_parser,
        tools,
        parallel_tool_calls,
        service_tier,
        enable_cache_report,
        weight_version: _,
    } = options;
    let reasoning_splitters = reasoning_parser
        .as_deref()
        .map(|parser| {
            (0..count)
                .map(|_| ReasoningStreamSplitter::new(Some(parser)))
                .collect()
        })
        .unwrap_or_default();
    let state = Arc::new(Mutex::new(ChatStreamState::new(count)));
    let plan = GeneratePlan {
        incremental: true,
        ..plan
    };
    let raw = generation_event_stream_with(
        plan,
        ChatStreamFrameShaper {
            response_id: response_id.clone(),
            model: model.clone(),
            created,
            want_logprobs,
            reasoning_splitters,
            prelude_emitted: vec![false; count],
            state: Arc::clone(&state),
            service_tier: service_tier.clone(),
        },
    )
    .flat_map(futures::stream::iter);
    let parsed: std::pin::Pin<
        Box<dyn futures::Stream<Item = Annotated<CreateChatCompletionStreamResponse>> + Send>,
    > = if let Some(parser) = parser {
        Box::pin(apply_tool_calling_jail(
            Some(dynamo_parser_name(&parser).to_owned()),
            tool_choice,
            tools,
            uses_tool_call_structural_tag,
            raw,
        ))
    } else {
        Box::pin(raw)
    };
    async_stream::stream! {
        let mut tool_calls_seen = vec![false; count];
        futures::pin_mut!(parsed);
        while let Some(mut item) = parsed.next().await {
            if let Some(mut response) = item.data.take() {
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
                // The parser can pack several choices into one event and only
                // flush them at EOF, but usage and the matched stop are per
                // choice and Python emits one choice per event: split only
                // packed events, leaving ordinary single-choice chunks as-is.
                if response.choices.len() <= 1 {
                    yield CoreEvent::Item(chat_chunk_value(
                        response,
                        &state,
                        continuous_usage,
                        enable_cache_report,
                    ));
                } else {
                    for choice in std::mem::take(&mut response.choices) {
                        let mut chunk = response.clone();
                        chunk.choices = vec![choice];
                        yield CoreEvent::Item(chat_chunk_value(
                            chunk,
                            &state,
                            continuous_usage,
                            enable_cache_report,
                        ));
                    }
                }
            } else if let Some(error) = item.error {
                yield CoreEvent::ItemError(decode_stream_error(error));
            }
        }

        // The final usage trailer follows the parser's EOF flush, so buffered
        // content and terminal choices always precede it.
        if include_usage {
            let (usage, details) = {
                let state = state.lock().expect("chat stream state poisoned");
                (
                    state.total_usage(),
                    UsageDetails {
                        reasoning_tokens: state.total_reasoning(),
                        // Python aggregates cached over the prompt
                        // representative, not every choice.
                        cached_tokens: enable_cache_report
                            .then(|| state.representative_cached()),
                    },
                )
            };
            let trailer = CreateChatCompletionStreamResponse {
                id: response_id,
                choices: Vec::new(),
                created,
                model,
                service_tier,
                system_fingerprint: None,
                object: "chat.completion.chunk".into(),
                usage: None,
            };
            let mut value =
                serde_json::to_value(trailer).expect("OpenAI response must serialize");
            value["usage"] = usage_value(usage, details);
            yield CoreEvent::Item(value);
        }
    }
}

/// Serialize one single-choice chunk the way Python's SSE choice does: force
/// `reasoning_content` present-as-null in the delta, attach continuous usage
/// when enabled, and always emit `matched_stop` (null when nothing matched).
fn chat_chunk_value(
    chunk: CreateChatCompletionStreamResponse,
    state: &Mutex<ChatStreamState>,
    continuous_usage: bool,
    enable_cache_report: bool,
) -> serde_json::Value {
    let index = chunk.choices.first().map(|choice| choice.index as usize);
    let carries_delta = chunk.choices.first().is_some_and(choice_carries_delta);
    let terminal = chunk
        .choices
        .first()
        .is_some_and(|choice| choice.finish_reason.is_some());
    // One short lock scope: gather usage, per-choice details, and the matched
    // stop before serializing the chunk.
    let (usage, details, matched_stop) = {
        let state = state.lock().expect("chat stream state poisoned");
        let choice_index = index.unwrap_or(0);
        let usage = (continuous_usage && carries_delta).then(|| state.choice_usage(choice_index));
        let details = UsageDetails {
            reasoning_tokens: state.choice_reasoning(choice_index),
            // Python's continuous chat chunks report the current choice's
            // cached count.
            cached_tokens: enable_cache_report.then(|| state.choice_cached(choice_index)),
        };
        let matched_stop = terminal
            .then(|| index.and_then(|index| state.matched_stop.get(index).cloned().flatten()))
            .flatten();
        (usage, details, matched_stop)
    };
    let mut value = serde_json::to_value(chunk).expect("OpenAI response must serialize");
    if let Some(usage) = usage {
        value["usage"] = usage_value(usage, details);
    }
    if let Some(delta) = value
        .pointer_mut("/choices/0/delta")
        .and_then(serde_json::Value::as_object_mut)
    {
        delta
            .entry("reasoning_content")
            .or_insert(serde_json::Value::Null);
    }
    if let Some(choice) = value["choices"].get_mut(0) {
        choice["matched_stop"] = matched_stop.unwrap_or(serde_json::Value::Null);
        // Python's chat logprobs carry only `content`.
        if let Some(logprobs) = choice
            .get_mut("logprobs")
            .and_then(serde_json::Value::as_object_mut)
        {
            logprobs.remove("refusal");
        }
    }
    value
}

/// Content-bearing chunk: Python attaches continuous usage to reasoning,
/// content, and tool-call chunks, but not to the role prelude or an
/// empty-delta terminal chunk.
fn choice_carries_delta(choice: &ChatChoiceStream) -> bool {
    choice.delta.content.is_some()
        || choice.delta.reasoning_content.is_some()
        || choice.delta.tool_calls.is_some()
}

/// [`ApiError`] rides the jail's string-typed `Annotated::error` slot
/// JSON-encoded (the jail itself never writes that slot — parser failures are
/// logged and their content dropped).
fn encode_stream_error(error: &ApiError) -> String {
    serde_json::to_string(error).expect("ApiError serializes")
}

fn decode_stream_error(encoded: String) -> ApiError {
    serde_json::from_str(&encoded).unwrap_or_else(|_| ApiError::internal(encoded))
}

/// One chat stream event as its SSE `data` payload: the chunk's OpenAI JSON
/// (already extension-shaped by the adapter) or the OpenAI error body.
pub(crate) fn chat_sse_payload(event: CoreEvent<serde_json::Value>) -> String {
    match event {
        CoreEvent::Item(value) => value.to_string(),
        CoreEvent::ItemError(e) => {
            crate::api_server::core::openai::error_payload_value(e.http_code, &e.message)
                .to_string()
        }
    }
}

#[allow(deprecated)]
pub(crate) fn chat_logprobs(extras: Option<&ChunkExtras>) -> ChatChoiceLogprobs {
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
            // Python's `ChatCompletionTokenLogprob` has no `token_id` field.
            token_id: None,
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
    use super::{
        ChatRenderingOptions, SamplingDefaults, chat_event_stream, chat_logprobs,
        chat_sampling_params, chat_stream_usage_options, merge_template_stops, unary_chat,
    };
    use crate::api_server::core::openai::template::ChatFormatter;
    use crate::api_server::core::test_utils::{
        abort_senders, aborted_guard_rids, chunk, chunk_with_metadata, plan, planned, senders,
    };
    use crate::message::config::DefaultSamplingParams;
    use crate::message::response::{ChunkEvent, ChunkExtras, ResponseItem, WeightVersionSpan};
    use dynamo_protocols::types::{ChatCompletionStreamOptions, CreateChatCompletionRequest, Stop};
    use futures::StreamExt;

    /// The common rendering options; tests override one field with struct-update
    /// syntax.
    fn chat_options() -> ChatRenderingOptions {
        ChatRenderingOptions {
            response_id: "chatcmpl-test".into(),
            model: "model".into(),
            created: 1,
            want_logprobs: false,
            parser: None,
            reasoning_parser: None,
            tools: None,
            parallel_tool_calls: true,
            service_tier: None,
            enable_cache_report: false,
            weight_version: Some("wv-test".into()),
        }
    }

    fn chunk_with_finish(rid: &str, text: &str, finish: serde_json::Value) -> ResponseItem {
        let mut item = chunk(rid, text, true);
        if let ResponseItem::Done(event) = &mut item {
            event.finish_reason = Some(serde_json::from_value(finish).unwrap());
        }
        item
    }

    #[test]
    fn chat_stream_usage_options_mirrors_python() {
        let options = |include_usage, continuous_usage_stats| ChatCompletionStreamOptions {
            include_usage,
            continuous_usage_stats,
        };
        assert_eq!(chat_stream_usage_options(None, false), (false, false));
        assert_eq!(chat_stream_usage_options(None, true), (true, false));
        assert_eq!(
            chat_stream_usage_options(Some(&options(false, false)), true),
            (true, false)
        );
        assert_eq!(
            chat_stream_usage_options(Some(&options(false, true)), true),
            (true, true)
        );
        assert_eq!(
            chat_stream_usage_options(Some(&options(true, false)), false),
            (true, false)
        );
        assert_eq!(
            chat_stream_usage_options(Some(&options(true, true)), false),
            (true, true)
        );
    }

    #[tokio::test]
    async fn dropping_during_chat_frame_expansion_aborts_only_unfinished_choices() {
        for terminal in [false, true] {
            let (senders, abort_rx) = abort_senders();
            let (choice0, tx0) = planned("r0");
            let (choice1, _tx1) = planned("r1");
            tx0.send(chunk("r0", "content", terminal)).await.unwrap();
            let plan = plan(vec![choice0, choice1], senders);

            {
                let stream = chat_event_stream(plan, chat_options(), false, false, None, false);
                futures::pin_mut!(stream);
                let role: serde_json::Value =
                    serde_json::from_str(&super::chat_sse_payload(stream.next().await.unwrap()))
                        .unwrap();
                assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
                // Drop with content still buffered inside the adapter's expansion.
                // A terminal RID was already disarmed by the shared core.
            }

            let expected = if terminal {
                vec!["r1"]
            } else {
                vec!["r0", "r1"]
            };
            assert_eq!(
                aborted_guard_rids(&abort_rx),
                expected,
                "terminal={terminal}"
            );
        }
    }

    fn request() -> CreateChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap()
    }

    /// Python `to_sampling_params` priority: user value > model generation
    /// config (`--sampling-defaults model`) > OpenAI terminal default. A
    /// `None` model value is the `--sampling-defaults openai` shape.
    #[test]
    fn sampling_defaults_follow_python_priority_chain() {
        let model = DefaultSamplingParams {
            temperature: Some(0.6),
            top_p: Some(0.9),
            ..Default::default()
        };
        // Omitted → model defaults, not the 1.0 OpenAI terminals.
        for (model, temperature, top_p) in [
            (model.clone(), 0.6_f64, 0.9_f64),
            (DefaultSamplingParams::default(), 1.0, 1.0),
        ] {
            let sampling = chat_sampling_params(
                &request(),
                &SamplingDefaults::CHAT.with_model_defaults(&model),
            )
            .unwrap();
            assert_eq!(sampling.temperature, temperature);
            assert_eq!(sampling.top_p, top_p);
        }
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

    /// Python `_apply_conversation_template`: template `stop_str` first, then
    /// the request's own stops.
    #[test]
    fn template_stops_merge_before_request_stops() {
        let chatml = crate::api_server::core::openai::template::builtin_template("chatml").unwrap();
        let formatter = ChatFormatter::Legacy(Box::new(
            crate::api_server::core::openai::template::LegacyFormatter { spec: chatml },
        ));
        assert_eq!(
            formatter.stop_strs(),
            Some(crate::message::types::OneOrMany::Many(vec![
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

        // A formatter with no template stops — the HuggingFace renderer's shape
        // (Python's jinja path keeps only the request's stops); the empty legacy
        // spec is that branch's constructible twin — leaves the request alone.
        let legacy = ChatFormatter::Legacy(Box::new(
            crate::api_server::core::openai::template::LegacyFormatter {
                spec: crate::api_server::core::openai::template::LegacySpec::default(),
            },
        ));
        assert!(legacy.stop_strs().is_none());
        let mut req = request();
        req.stop = Some(Stop::String("x".into()));
        merge_template_stops(&mut req, &legacy);
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
        // Python's chat logprob has no `token_id` field; the dynamo field stays
        // None so it is omitted on the wire.
        assert_eq!(token.token_id, None);
        assert_eq!(token.top_logprobs.len(), 2);
        assert_eq!(token.top_logprobs[1].token, "y");
    }

    #[tokio::test]
    async fn unary_chat_fans_in_choices_and_usage() {
        let (choice0, tx0) = planned("r0");
        let (choice1, tx1) = planned("r1");
        tx0.send(chunk("r0", "Paris", true)).await.unwrap();
        tx1.send(chunk("r1", "London", true)).await.unwrap();

        let value = unary_chat(plan(vec![choice0, choice1], senders()), chat_options())
            .await
            .expect("unary chat succeeds");
        assert_eq!(value["choices"][0]["message"]["role"], "assistant");
        assert_eq!(value["choices"][0]["message"]["content"], "Paris");
        assert_eq!(value["choices"][1]["index"], 1);
        assert_eq!(value["choices"][1]["message"]["content"], "London");
        assert_eq!(value["usage"]["prompt_tokens"], 5);
        assert_eq!(value["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn unary_chat_separates_reasoning_content_with_parser_configured() {
        let (choice, tx) = planned("r0");
        tx.send(chunk(
            "r0",
            "<think>because Paris is famous</think>Paris",
            true,
        ))
        .await
        .unwrap();

        let value = unary_chat(
            plan(vec![choice], senders()),
            ChatRenderingOptions {
                reasoning_parser: Some("deepseek-r1".into()),
                ..chat_options()
            },
        )
        .await
        .expect("unary chat succeeds");
        assert_eq!(
            value["choices"][0]["message"]["reasoning_content"],
            "because Paris is famous"
        );
        assert_eq!(value["choices"][0]["message"]["content"], "Paris");
        assert!(value["choices"][0]["message"]["reasoning_content"].is_string());
    }

    /// Whitespace-preserving reasoning split must not disturb tool parsing: the
    /// normal text after `</think>` still yields the call and its arguments.
    #[tokio::test]
    async fn unary_chat_reasoning_whitespace_survives_tool_parsing() {
        let (choice, tx) = planned("r0");
        tx.send(chunk(
            "r0",
            "<think>\nOkay \n</think> <tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>",
            true,
        ))
        .await
        .unwrap();

        let value = unary_chat(
            plan(vec![choice], senders()),
            ChatRenderingOptions {
                reasoning_parser: Some("qwen3".into()),
                parser: Some("qwen".into()),
                ..chat_options()
            },
        )
        .await
        .expect("unary chat succeeds");
        assert_eq!(
            value["choices"][0]["message"]["reasoning_content"],
            "\nOkay \n"
        );
        assert_eq!(value["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            value["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert!(
            value["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap()
                .contains("Paris")
        );
    }

    #[tokio::test]
    async fn streaming_chat_separates_reasoning_into_own_deltas() {
        let (choice, tx) = planned("r0");
        // Force mode starts in reasoning, so the opener is stripped and the first
        // reasoning fragment streams immediately.
        tx.send(chunk("r0", "<think>be", false)).await.unwrap();
        tx.send(chunk("r0", "cause</think>Par", false))
            .await
            .unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();

        let stream = chat_event_stream(
            plan(vec![choice], senders()),
            ChatRenderingOptions {
                reasoning_parser: Some("deepseek-r1".into()),
                ..chat_options()
            },
            true,
            false,
            None,
            false,
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.map(super::chat_sse_payload).collect().await;
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
        // Role + 2 reasoning + content + terminal + tool-jail flush; `[DONE]`
        // is SSE framing appended by `sse_encode`, not a stream item.
        assert_eq!(frames.len(), 6);
    }

    #[tokio::test]
    async fn streaming_chat_emits_role_deltas_usage_and_done() {
        let (choice, tx) = planned("r0");
        tx.send(chunk("r0", "Par", false)).await.unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();

        let stream = chat_event_stream(
            plan(vec![choice], senders()),
            chat_options(),
            true,
            false,
            None,
            false,
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.map(super::chat_sse_payload).collect().await;
        assert_eq!(frames.len(), 4);
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
    }

    /// With `include_usage=false` the stream ends at the terminal chunk; no
    /// usage-only trailer is emitted.
    #[tokio::test]
    async fn streaming_chat_omits_usage_trailer_when_disabled() {
        let (choice, tx) = planned("r0");
        tx.send(chunk("r0", "Par", false)).await.unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();

        let stream = chat_event_stream(
            plan(vec![choice], senders()),
            chat_options(),
            false,
            false,
            None,
            false,
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.map(super::chat_sse_payload).collect().await;
        assert_eq!(frames.len(), 3, "role + delta + terminal, no usage chunk");
        let terminal: serde_json::Value = serde_json::from_str(&frames[2]).unwrap();
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert!(terminal["usage"].is_null());
    }

    /// Usage must recover the prompt count when an early frame carries 0: the
    /// streaming shaper keeps the first NON-zero count (matching the unary
    /// fan-in) instead of latching the first frame's value.
    #[tokio::test]
    async fn streaming_usage_recovers_prompt_tokens_from_a_later_frame() {
        let (choice, tx) = planned("r0");
        let mut first = chunk("r0", "Par", false);
        if let ResponseItem::Frame(event) = &mut first {
            event.prompt_tokens = 0;
        }
        let mut terminal = chunk("r0", "is", true);
        if let ResponseItem::Done(event) = &mut terminal {
            event.prompt_tokens = 19;
        }
        tx.send(first).await.unwrap();
        tx.send(terminal).await.unwrap();

        let stream = chat_event_stream(
            plan(vec![choice], senders()),
            chat_options(),
            true,
            false,
            None,
            false,
        );
        futures::pin_mut!(stream);
        let frames: Vec<String> = stream.map(super::chat_sse_payload).collect().await;
        let usage: serde_json::Value = serde_json::from_str(frames.last().unwrap()).unwrap();
        assert_eq!(usage["usage"]["prompt_tokens"], 19);
    }

    /// One generation event can expand into a reasoning chunk and a content
    /// chunk: both share the same usage snapshot, the trailer counts the tokens
    /// once, and the role prelude carries no usage (Python behavior).
    #[tokio::test]
    async fn stream_continuous_usage_counts_each_generation_event_once() {
        let (choice, tx) = planned("r0");
        tx.send(chunk("r0", "<think>be", false)).await.unwrap();
        tx.send(chunk("r0", "cause</think>Par", false))
            .await
            .unwrap();
        tx.send(chunk("r0", "is", true)).await.unwrap();

        let stream = chat_event_stream(
            plan(vec![choice], senders()),
            ChatRenderingOptions {
                reasoning_parser: Some("deepseek-r1".into()),
                ..chat_options()
            },
            true,
            true,
            None,
            false,
        );
        futures::pin_mut!(stream);
        let frames: Vec<serde_json::Value> = stream
            .map(super::chat_sse_payload)
            .map(|payload| serde_json::from_str::<serde_json::Value>(&payload).unwrap())
            .collect()
            .await;
        assert_eq!(
            frames.len(),
            6,
            "role + 2 reasoning + content + terminal + usage"
        );
        assert_eq!(frames[0]["choices"][0]["delta"]["role"], "assistant");
        assert!(
            frames[0]["usage"].is_null(),
            "role prelude carries no usage"
        );
        assert_eq!(frames[1]["usage"]["completion_tokens"], 1);
        assert_eq!(frames[2]["usage"]["completion_tokens"], 2);
        assert_eq!(
            frames[3]["usage"]["completion_tokens"], 2,
            "content shares its event's snapshot, not a second increment"
        );
        assert_eq!(frames[4]["usage"]["completion_tokens"], 3);
        assert_eq!(frames[4]["choices"][0]["finish_reason"], "stop");
        assert_eq!(frames[4]["choices"][0]["matched_stop"], "</s>");

        let trailer = frames.last().unwrap();
        assert_eq!(trailer["usage"]["prompt_tokens"], 5);
        assert_eq!(trailer["usage"]["completion_tokens"], 3);
        assert_eq!(trailer["usage"]["total_tokens"], 8);
        assert!(trailer["choices"].as_array().unwrap().is_empty());
    }

    /// Continuous usage survives the tool jail: the synthesized tool-call
    /// chunk and its terminal carry the accumulated snapshot.
    #[tokio::test]
    async fn stream_continuous_usage_reaches_tool_call_chunks() {
        let (choice, tx) = planned("r0");
        tx.send(chunk(
            "r0",
            r#"<|python_tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
            false,
        ))
        .await
        .unwrap();
        tx.send(chunk("r0", "", true)).await.unwrap();

        let stream = chat_event_stream(
            plan(vec![choice], senders()),
            ChatRenderingOptions {
                parser: Some("llama3_json".into()),
                ..chat_options()
            },
            true,
            true,
            None,
            false,
        );
        futures::pin_mut!(stream);
        let frames: Vec<serde_json::Value> = stream
            .map(super::chat_sse_payload)
            .map(|payload| serde_json::from_str::<serde_json::Value>(&payload).unwrap())
            .collect()
            .await;
        let tool_chunk = frames
            .iter()
            .find(|frame| !frame["choices"][0]["delta"]["tool_calls"].is_null())
            .expect("tool-call chunk");
        assert_eq!(
            tool_chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(tool_chunk["usage"]["prompt_tokens"], 5);

        let terminal = frames
            .iter()
            .find(|frame| frame["choices"][0]["finish_reason"] == "tool_calls")
            .expect("tool-calls terminal");
        assert_eq!(terminal["choices"][0]["matched_stop"], "</s>");
        let trailer = frames.last().unwrap();
        assert!(trailer["choices"].as_array().unwrap().is_empty());
        assert_eq!(trailer["usage"]["prompt_tokens"], 5);
        assert_eq!(trailer["usage"]["completion_tokens"], 2);
    }

    /// The parser can flush several buffered choices in one packed event; the
    /// adapter splits them so each terminal keeps its own usage and matched
    /// stop, and the final trailer follows the flush.
    #[tokio::test]
    async fn stream_splits_packed_parser_output_with_per_choice_metadata() {
        let (choice0, tx0) = planned("r0");
        let (choice1, tx1) = planned("r1");
        // A truncated tool marker: the parser buffers both choices until EOF
        // and releases them packed into one event.
        let partial = r#"<|python_tag|>{"name":"get_weather","par"#;
        tx0.send(chunk("r0", partial, false)).await.unwrap();
        tx0.send(chunk("r0", "tial", false)).await.unwrap();
        tx0.send(chunk_with_finish(
            "r0",
            "",
            serde_json::json!({"type": "stop", "matched": "Paris"}),
        ))
        .await
        .unwrap();
        tx1.send(chunk("r1", partial, false)).await.unwrap();
        tx1.send(chunk_with_finish(
            "r1",
            "",
            serde_json::json!({"type": "stop", "matched": 151645}),
        ))
        .await
        .unwrap();

        let stream = chat_event_stream(
            plan(vec![choice0, choice1], senders()),
            ChatRenderingOptions {
                parser: Some("llama3_json".into()),
                ..chat_options()
            },
            true,
            true,
            None,
            false,
        );
        futures::pin_mut!(stream);
        let frames: Vec<serde_json::Value> = stream
            .map(super::chat_sse_payload)
            .map(|payload| serde_json::from_str::<serde_json::Value>(&payload).unwrap())
            .collect()
            .await;
        let terminal_of = |index: u64| {
            frames
                .iter()
                .find(|frame| {
                    frame["choices"][0]["index"] == index
                        && !frame["choices"][0]["finish_reason"].is_null()
                })
                .unwrap_or_else(|| panic!("terminal for choice {index}"))
        };
        let first = terminal_of(0);
        assert_eq!(first["choices"][0]["matched_stop"], "Paris");
        assert_eq!(first["usage"]["completion_tokens"], 3);
        let second = terminal_of(1);
        assert_eq!(second["choices"][0]["matched_stop"], 151645);
        assert_eq!(second["usage"]["completion_tokens"], 2);

        let trailer = frames.last().unwrap();
        assert!(trailer["choices"].as_array().unwrap().is_empty());
        assert_eq!(trailer["usage"]["prompt_tokens"], 5);
        assert_eq!(trailer["usage"]["completion_tokens"], 5);
        assert_eq!(trailer["usage"]["total_tokens"], 10);
    }

    /// Continuous usage can be requested without the final trailer: every
    /// content-bearing chunk carries a snapshot and no empty-choices event is
    /// emitted.
    #[tokio::test]
    async fn stream_continuous_usage_without_final_trailer() {
        let (choice, tx) = planned("r0");
        tx.send(chunk("r0", "a", false)).await.unwrap();
        tx.send(chunk("r0", "b", true)).await.unwrap();

        let stream = chat_event_stream(
            plan(vec![choice], senders()),
            chat_options(),
            false,
            true,
            None,
            false,
        );
        futures::pin_mut!(stream);
        let frames: Vec<serde_json::Value> = stream
            .map(super::chat_sse_payload)
            .map(|payload| serde_json::from_str::<serde_json::Value>(&payload).unwrap())
            .collect()
            .await;
        assert_eq!(frames.len(), 3, "role + 2 chunks, no usage-only trailer");
        assert!(
            frames
                .iter()
                .all(|frame| !frame["choices"].as_array().unwrap().is_empty())
        );
        assert_eq!(frames[1]["usage"]["completion_tokens"], 1);
        assert_eq!(frames[2]["usage"]["completion_tokens"], 2);
    }

    /// Python always emits `matched_stop` in a choice: content chunks carry
    /// null and a length finish carries null on the terminal too.
    #[tokio::test]
    async fn stream_choices_emit_null_matched_stop_without_a_match() {
        let (choice, tx) = planned("r0");
        tx.send(chunk("r0", "a", false)).await.unwrap();
        tx.send(chunk_with_finish(
            "r0",
            "b",
            serde_json::json!({"type": "length", "length": 8}),
        ))
        .await
        .unwrap();

        let stream = chat_event_stream(
            plan(vec![choice], senders()),
            chat_options(),
            false,
            false,
            None,
            false,
        );
        futures::pin_mut!(stream);
        let frames: Vec<serde_json::Value> = stream
            .map(super::chat_sse_payload)
            .map(|payload| serde_json::from_str::<serde_json::Value>(&payload).unwrap())
            .collect()
            .await;
        assert_eq!(
            frames[1]["choices"][0]["matched_stop"],
            serde_json::Value::Null
        );
        assert!(frames[1]["choices"][0].get("matched_stop").is_some());
        assert_eq!(frames[2]["choices"][0]["finish_reason"], "length");
        assert_eq!(
            frames[2]["choices"][0]["matched_stop"],
            serde_json::Value::Null
        );
    }

    /// The final trailer counts the shared prompt once but sums every choice's
    /// completion tokens; continuous chunks report each choice's own count.
    #[tokio::test]
    async fn stream_usage_trailer_deduplicates_shared_prompt_across_choices() {
        let (choice0, tx0) = planned("r0");
        let (choice1, tx1) = planned("r1");
        tx0.send(chunk("r0", "a", false)).await.unwrap();
        tx0.send(chunk("r0", "b", true)).await.unwrap();
        tx1.send(chunk("r1", "x", false)).await.unwrap();
        tx1.send(chunk("r1", "y", true)).await.unwrap();

        let stream = chat_event_stream(
            plan(vec![choice0, choice1], senders()),
            chat_options(),
            true,
            true,
            None,
            false,
        );
        futures::pin_mut!(stream);
        let frames: Vec<serde_json::Value> = stream
            .map(super::chat_sse_payload)
            .map(|payload| serde_json::from_str::<serde_json::Value>(&payload).unwrap())
            .collect()
            .await;

        let mut completion_by_choice = std::collections::BTreeMap::new();
        for frame in &frames[..frames.len() - 1] {
            if frame["choices"][0]["delta"]["content"].is_null() {
                continue; // the role prelude carries no usage (Python parity)
            }
            assert!(
                !frame["usage"].is_null(),
                "every content chunk carries usage"
            );
            let index = frame["choices"][0]["index"].as_u64().unwrap();
            let completion = frame["usage"]["completion_tokens"].as_u64().unwrap();
            completion_by_choice
                .entry(index)
                .and_modify(|value: &mut u64| *value = (*value).max(completion))
                .or_insert(completion);
        }
        assert_eq!(
            completion_by_choice,
            std::collections::BTreeMap::from([(0, 2), (1, 2)])
        );

        let trailer = frames.last().unwrap();
        assert!(trailer["choices"].as_array().unwrap().is_empty());
        assert_eq!(trailer["usage"]["prompt_tokens"], 5);
        assert_eq!(trailer["usage"]["completion_tokens"], 4);
        assert_eq!(trailer["usage"]["total_tokens"], 9);
    }

    /// Unary choices carry the native matched stop when there is one: a
    /// matched string, an EOS token id, and null when nothing matched.
    #[tokio::test]
    async fn unary_chat_preserves_matched_stop() {
        let (choice0, tx0) = planned("r0");
        let (choice1, tx1) = planned("r1");
        let (choice2, tx2) = planned("r2");
        tx0.send(chunk_with_finish(
            "r0",
            "a",
            serde_json::json!({"type": "stop", "matched": "Paris"}),
        ))
        .await
        .unwrap();
        tx1.send(chunk_with_finish(
            "r1",
            "b",
            serde_json::json!({"type": "stop", "matched": 151645}),
        ))
        .await
        .unwrap();
        tx2.send(chunk_with_finish(
            "r2",
            "c",
            serde_json::json!({"type": "length", "length": 8}),
        ))
        .await
        .unwrap();

        let value = unary_chat(
            plan(vec![choice0, choice1, choice2], senders()),
            chat_options(),
        )
        .await
        .expect("unary chat succeeds");
        assert_eq!(value["choices"][0]["matched_stop"], "Paris");
        assert_eq!(value["choices"][1]["matched_stop"], 151645);
        assert_eq!(
            value["choices"][2]["matched_stop"],
            serde_json::Value::Null,
            "no match still emits the key, as null"
        );
    }
    /// Python's `UsageInfo` on a unary chat response: top-level
    /// `reasoning_tokens`, cached prompt details with cache reporting, and the
    /// request's weight metadata (spans' last version, launch scalar fallback).
    #[tokio::test]
    async fn unary_chat_usage_exposes_reasoning_cached_and_metadata() {
        let (choice, tx) = planned("r0");
        let mut item = chunk_with_metadata("r0", "answer", true, 5, 4);
        if let ResponseItem::Done(event) = &mut item {
            event.extras.as_mut().unwrap().weight_versions = Some(vec![
                WeightVersionSpan {
                    version: "v1".into(),
                    start: 0,
                    end: 1,
                },
                WeightVersionSpan {
                    version: "v2".into(),
                    start: 1,
                    end: 2,
                },
            ]);
        }
        tx.send(item).await.unwrap();

        let value = unary_chat(
            plan(vec![choice], senders()),
            ChatRenderingOptions {
                enable_cache_report: true,
                ..chat_options()
            },
        )
        .await
        .expect("unary chat succeeds");
        assert_eq!(value["usage"]["reasoning_tokens"], 5);
        assert_eq!(value["usage"]["prompt_tokens_details"]["cached_tokens"], 4);
        assert_eq!(value["metadata"]["weight_version"], "v2");
        assert_eq!(value["metadata"]["weight_versions"][1]["end"], 2);
    }

    /// Continuous chat chunks carry the choice's reasoning snapshot and (with
    /// cache reporting) the shared cached count; the trailer sums the choices.
    #[tokio::test]
    async fn stream_continuous_usage_exposes_reasoning_and_cached() {
        let (choice, tx) = planned("r0");
        tx.send(chunk_with_metadata("r0", "a", false, 2, 3))
            .await
            .unwrap();
        tx.send(chunk_with_metadata("r0", "b", true, 7, 3))
            .await
            .unwrap();

        let stream = chat_event_stream(
            plan(vec![choice], senders()),
            ChatRenderingOptions {
                enable_cache_report: true,
                ..chat_options()
            },
            true,
            true,
            None,
            false,
        );
        futures::pin_mut!(stream);
        let frames: Vec<serde_json::Value> = stream
            .map(super::chat_sse_payload)
            .map(|payload| serde_json::from_str::<serde_json::Value>(&payload).unwrap())
            .collect()
            .await;
        assert_eq!(frames[1]["usage"]["reasoning_tokens"], 2);
        assert_eq!(
            frames[1]["usage"]["prompt_tokens_details"]["cached_tokens"],
            3
        );
        let trailer = frames.last().unwrap();
        assert!(trailer["choices"].as_array().unwrap().is_empty());
        assert_eq!(trailer["usage"]["reasoning_tokens"], 7);
        assert_eq!(
            trailer["usage"]["prompt_tokens_details"]["cached_tokens"],
            3
        );
    }
    /// Cached counts are retained per choice: a continuous chunk reports its
    /// own choice's count even when it renders first. The trailer check here
    /// only shapes the usage object the way the production trailer does;
    /// actual trailer emission and ordering are covered by the stream fixture
    /// tests.
    #[test]
    fn stream_cached_counts_are_per_choice_with_a_choice_zero_trailer() {
        let event = |cached: u32| ChunkEvent {
            extras: Some(Box::new(ChunkExtras {
                cached_tokens: cached,
                ..Default::default()
            })),
            ..Default::default()
        };
        let mut state = super::ChatStreamState::new(2);
        // Choice 1 reports before choice 0 has recorded anything.
        state.record(1, &event(4), 1);
        let state = std::sync::Mutex::new(state);

        let chunk = |index: u32| super::CreateChatCompletionStreamResponse {
            id: "chatcmpl-test".into(),
            choices: vec![super::ChatChoiceStream {
                index,
                delta: super::chat_delta(Some("x".into()), None, None, None),
                finish_reason: None,
                logprobs: None,
            }],
            created: 1,
            model: "model".into(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion.chunk".into(),
            usage: None,
        };

        // Early arrival: choice 1's chunk carries its own count.
        let second = super::chat_chunk_value(chunk(1), &state, true, true);
        assert_eq!(second["usage"]["prompt_tokens_details"]["cached_tokens"], 4);

        state.lock().unwrap().record(0, &event(3), 2);
        let first = super::chat_chunk_value(chunk(0), &state, true, true);
        assert_eq!(first["usage"]["prompt_tokens_details"]["cached_tokens"], 3);

        // The trailer's usage is shaped exactly like the production trailer:
        // aggregated counts plus the representative's cached count.
        let trailer_usage = {
            let state = state.lock().unwrap();
            super::usage_value(
                state.total_usage(),
                super::UsageDetails {
                    reasoning_tokens: state.total_reasoning(),
                    cached_tokens: Some(state.representative_cached()),
                },
            )
        };
        assert_eq!(trailer_usage["prompt_tokens_details"]["cached_tokens"], 3);
    }
    /// Python's `ChatMessage`/`ChoiceLogprobs` presence rules: content is
    /// always a string (empty stopped output included), `tool_calls` is always
    /// present (null when none) with a per-call `index`, there is no `refusal`
    /// on the message or logprobs, and logprob tokens have no `token_id`.
    #[tokio::test]
    async fn unary_chat_message_and_logprobs_presence_match_python() {
        // Empty stopped content still serializes as ""; tool_calls is null.
        let (choice, tx) = planned("r0");
        tx.send(chunk("r0", "", true)).await.unwrap();
        let value = unary_chat(plan(vec![choice], senders()), chat_options())
            .await
            .expect("unary chat succeeds");
        let message = &value["choices"][0]["message"];
        assert_eq!(message["content"], "");
        assert!(message.get("tool_calls").is_some());
        assert!(message["tool_calls"].is_null());
        assert!(message.get("refusal").is_none());
        assert!(message["reasoning_content"].is_null());

        // Populated logprobs: no `refusal`, no `token_id`.
        let (choice, tx) = planned("r0");
        let mut frame = chunk("r0", "answer", true);
        if let ResponseItem::Done(event) = &mut frame {
            event.extras = Some(Box::new(ChunkExtras {
                out_lp_val: vec![-0.25],
                out_lp_idx: vec![7],
                out_lp_txt: vec!["x".into()],
                ..Default::default()
            }));
        }
        tx.send(frame).await.unwrap();
        let value = unary_chat(
            plan(vec![choice], senders()),
            ChatRenderingOptions {
                want_logprobs: true,
                ..chat_options()
            },
        )
        .await
        .expect("unary chat succeeds");
        let logprobs = &value["choices"][0]["logprobs"];
        assert!(logprobs.get("refusal").is_none());
        assert_eq!(logprobs["content"][0]["token"], "x");
        assert!(logprobs["content"][0].get("token_id").is_none());

        // Populated tool calls carry the per-call index.
        let (choice, tx) = planned("r0");
        tx.send(chunk(
            "r0",
            r#"<|python_tag|>{"name":"get_weather","parameters":{"city":"Paris"}}"#,
            true,
        ))
        .await
        .unwrap();
        let value = unary_chat(
            plan(vec![choice], senders()),
            ChatRenderingOptions {
                parser: Some("llama3_json".into()),
                ..chat_options()
            },
        )
        .await
        .expect("unary chat succeeds");
        let message = &value["choices"][0]["message"];
        assert_eq!(message["content"], "");
        assert_eq!(message["tool_calls"][0]["index"], 0);
        assert_eq!(message["tool_calls"][0]["type"], "function");
        assert_eq!(message["tool_calls"][0]["function"]["name"], "get_weather");
    }
    /// Each Chat choice emits its role before its own first output/error; all
    /// roles do not have to precede all content.
    #[test]
    fn chat_role_preludes_stay_per_choice() {
        use crate::api_server::core::generate::FrameShaper;

        let mut shaper = super::ChatStreamFrameShaper {
            response_id: "chatcmpl-test".into(),
            model: "model".into(),
            created: 1,
            want_logprobs: false,
            reasoning_splitters: Vec::new(),
            prelude_emitted: vec![false; 2],
            state: std::sync::Arc::new(std::sync::Mutex::new(super::ChatStreamState::new(2))),
            service_tier: None,
        };
        let indexed = |frame: &super::Annotated<super::CreateChatCompletionStreamResponse>| {
            frame.data.as_ref().expect("data frame").choices[0].index
        };
        let role = |frame: &super::Annotated<super::CreateChatCompletionStreamResponse>| {
            frame.data.as_ref().expect("data frame").choices[0]
                .delta
                .role
        };

        // Choice 1 errors first: its role precedes its own error.
        let frames = shaper.item_error(500, "boom", Some(1));
        assert_eq!(frames.len(), 2);
        assert_eq!(indexed(&frames[0]), 1);
        assert_eq!(role(&frames[0]), Some(super::Role::Assistant));
        assert!(frames[1].data.is_none() && frames[1].error.is_some());

        // Choice 0's first output emits choice 0's role, not choice 1's.
        let output = ChunkEvent {
            text: "hi".into(),
            ..Default::default()
        };
        let frames = shaper.delta(output, &super::OutputAccumulator::default(), "", Some(0));
        assert_eq!(frames.len(), 2);
        assert_eq!(indexed(&frames[0]), 0);
        assert_eq!(role(&frames[0]), Some(super::Role::Assistant));
    }
}
