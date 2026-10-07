//! `/v1/chat/completions`, as `serving_chat.py` serves it, for requests without
//! tools. The host renders the prompt; this builds everything around it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::request::{
    OneOrList, ResponseFormat, StreamOptions, custom_labels, default_model, float_logit_bias,
    format_json_schema, lora_path, one, usage_flags, yes,
};
use super::wire::{
    Payload, Reply, StreamState, http_status_name, malformed_output_reply, meta_u64, now,
    stream_error_event, unary_reply,
};
use super::{OpenAiHeaders, OpenAiSettings, OpenAiTokenizer, Unsupported};
use crate::parser::models::think_config;
use crate::parser::{ReasoningOptions, ReasoningStreamSplitter, split_reasoning};

/// Facts about the served model that Python's chat layer reads.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChatModel {
    /// `get_model().context_length`.
    pub context_length: Option<u64>,
    /// `model_config.get_default_sampling_params()` under `--sampling-defaults model`.
    pub generation_config: Map<String, Value>,
}

/// `ChatCompletionRequest` with Pydantic's defaults, for the fields that reach
/// `GenerateReqInput` or the response. A body this rejects goes to the engine.
#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Value>,
    #[serde(default = "default_model")]
    model: String,
    #[serde(default)]
    frequency_penalty: f64,
    logit_bias: Option<Map<String, Value>>,
    #[serde(default)]
    logprobs: bool,
    top_logprobs: Option<i64>,
    max_tokens: Option<i64>,
    max_completion_tokens: Option<i64>,
    #[serde(default = "one")]
    n: i64,
    #[serde(default)]
    presence_penalty: f64,
    response_format: Option<ResponseFormat>,
    seed: Option<i64>,
    stop: Option<OneOrList<String>>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    #[allow(dead_code)]
    user: Option<String>,
    top_k: Option<i64>,
    min_p: Option<f64>,
    #[serde(default)]
    min_tokens: i64,
    regex: Option<String>,
    ebnf: Option<String>,
    repetition_penalty: Option<f64>,
    stop_token_ids: Option<Vec<i64>>,
    stop_regex: Option<OneOrList<String>>,
    #[serde(default)]
    no_stop_trim: bool,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default)]
    continue_final_message: bool,
    #[serde(default = "yes")]
    skip_special_tokens: bool,
    lora_path: Option<OneOrList<Option<String>>>,
    session_id: Option<String>,
    #[serde(default = "yes")]
    separate_reasoning: bool,
    #[serde(default = "yes")]
    stream_reasoning: bool,
    chat_template_kwargs: Option<Map<String, Value>>,
    custom_logit_processor: Option<OneOrList<Option<String>>>,
    custom_params: Option<Map<String, Value>>,
    images_config: Option<Map<String, Value>>,
    rid: Option<OneOrList<String>>,
    extra_key: Option<OneOrList<String>>,
    cache_salt: Option<OneOrList<String>>,
    priority: Option<i64>,
    data_parallel_rank: Option<i64>,
    bootstrap_host: Option<OneOrList<String>>,
    bootstrap_port: Option<OneOrList<Option<i64>>>,
    bootstrap_room: Option<OneOrList<i64>>,
    routed_dp_rank: Option<i64>,
    disagg_prefill_dp_rank: Option<i64>,
}

/// Fields that change behavior this module does not reproduce yet.
const UNSUPPORTED_FIELDS: &[&str] = &[
    "tools",
    "input_ids",
    "return_hidden_states",
    "return_routed_experts",
    "return_cached_tokens_details",
    "return_spec_tokens_details",
    "return_prompt_token_ids",
    "return_token_ids",
    "return_meta_info",
    "return_input_ids_in_sglext",
    "return_output_ids_in_sglext",
    "return_sampling_mask",
    "sampling_logprobs_mode",
    "video_config",
    "max_dynamic_patch",
    "min_dynamic_patch",
    "use_audio_in_video",
];

const TASKS: &[&str] = &[
    "action",
    "query",
    "authority",
    "domain",
    "title",
    "read_url",
];

const EFFORT_TIERS: &[&str] = &["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// `ChatCompletionMessageGenericParam` roles, matched case-insensitively.
const GENERIC_ROLES: &[&str] = &[
    "system",
    "assistant",
    "tool",
    "function",
    "developer",
    "latest_reminder",
];

/// Lower a chat body into a `/generate` body, as
/// `OpenAIServingChat._convert_to_internal_request` does. `render` returns the
/// prompt's token ids for the body as sent, or `None` when it cannot.
pub fn lower_chat(
    body: &[u8],
    headers: &OpenAiHeaders<'_>,
    settings: &OpenAiSettings,
    model: &ChatModel,
    render: impl FnOnce(&Value) -> Option<Vec<u32>>,
    tokenizer: Option<Arc<dyn OpenAiTokenizer>>,
) -> Result<(Value, ChatResponder), Unsupported> {
    let original: Value =
        serde_json::from_slice(body).map_err(|_| Unsupported("unparsed_request"))?;
    let Value::Object(mut raw) = original.clone() else {
        return Err(Unsupported("unparsed_request"));
    };
    if UNSUPPORTED_FIELDS
        .iter()
        .any(|field| raw.get(*field).is_some_and(|v| v != false))
        || raw
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(|m| m.iter().any(has_tools_or_media))
    {
        return Err(Unsupported("unsupported_chat_feature"));
    }
    custom_labels(headers, settings)?;
    if settings.return_input_ids || settings.return_output_ids || headers.sglext_ids {
        return Err(Unsupported("sglext_output"));
    }
    let parser = settings.reasoning_parser.as_deref();
    if parser.is_some_and(|parser| think_config(parser).is_none()) {
        return Err(Unsupported("reasoning_parser"));
    }
    // Fields the typed request below does not read, typed as Pydantic types them.
    let task = raw.get("task").filter(|t| !t.is_null());
    let tool_choice = raw.get("tool_choice").filter(|c| !c.is_null());
    if task.is_some_and(|t| !TASKS.iter().any(|task| t == task))
        || tool_choice.is_some_and(|c| c != "auto" && c != "none")
        || raw
            .get("parallel_tool_calls")
            .is_some_and(|p| !p.is_boolean())
        || raw
            .get("session_params")
            .is_some_and(|p| !p.is_object() && !p.is_null())
    {
        return Err(Unsupported("invalid_request"));
    }
    let thinking_kwarg = normalize_reasoning_inputs(&mut raw)?;
    if raw
        .get("reasoning_effort")
        .is_some_and(|e| !valid_effort(e))
    {
        return Err(Unsupported("invalid_request"));
    }
    let mut request: ChatRequest =
        serde_json::from_value(Value::Object(raw)).map_err(|_| Unsupported("unparsed_request"))?;
    validate(&request, settings, model)?;
    if request.logprobs && tokenizer.is_none() {
        return Err(Unsupported("no_tokenizer"));
    }
    let prompt_ids = render(&original).ok_or(Unsupported("render"))?;

    // `_process_messages`: the server's default kwargs fill in the request's.
    let mut kwargs = request.chat_template_kwargs.take().unwrap_or_default();
    if let Some(thinking) = thinking_kwarg {
        kwargs.entry("thinking").or_insert(thinking.clone());
        kwargs.entry("enable_thinking").or_insert(thinking);
    }
    kwargs.remove("reasoning_effort");
    for (key, value) in settings.default_chat_template_kwargs.iter().flatten() {
        kwargs.entry(key.clone()).or_insert_with(|| value.clone());
    }
    let thinking = parser.is_some() && kwargs.get("thinking") == Some(&Value::Bool(true));
    let previous_content = match request.messages.last() {
        Some(last)
            // Roles are compared lowercased, as Pydantic normalizes them.
            if request.continue_final_message
                && last["role"]
                    .as_str()
                    .is_some_and(|r| r.eq_ignore_ascii_case("assistant")) =>
        {
            // Python's reasoning parser raises on null content.
            if last["content"].is_null() && parser.is_some() && request.separate_reasoning {
                return Err(Unsupported("invalid_request"));
            }
            Some(last["content"].as_str().unwrap_or_default().to_owned())
        }
        _ => None,
    };

    let lowered = generate_body(
        &request, &kwargs, model, prompt_ids, thinking, headers, settings,
    );
    let (include_usage, continuous_usage_stats) =
        usage_flags(request.stream_options.as_ref(), settings);
    let reasoning = parser
        .filter(|_| request.separate_reasoning)
        .map(|parser| ReasoningConfig {
            parser: parser.to_owned(),
            options: ReasoningOptions {
                force_reasoning: Some(thinking),
                stream_reasoning: request.stream_reasoning,
                previous_content,
                force_nonempty_content: kwargs.get("force_nonempty_content")
                    == Some(&Value::Bool(true)),
            },
        });
    let responder = ChatResponder {
        model: request.model,
        n: request.n as usize,
        logprobs: request.logprobs,
        reasoning,
        include_usage,
        continuous_usage_stats,
        settings: settings.clone(),
        tokenizer,
        stream: StreamState::default(),
        roles_sent: HashSet::new(),
        detectors: HashMap::new(),
        finish_reasons: Vec::new(),
    };
    Ok((lowered, responder))
}

fn truthy(value: &Value) -> bool {
    !matches!(value, Value::Null | Value::Bool(false))
        && value.as_array().is_none_or(|items| !items.is_empty())
}

fn has_tools_or_media(message: &Value) -> bool {
    let media = |part: &Value| {
        matches!(
            part["type"].as_str(),
            Some("image_url" | "video_url" | "audio_url" | "input_audio")
        )
    };
    message.get("tools").is_some_and(truthy)
        || message["content"]
            .as_array()
            .is_some_and(|parts| parts.iter().any(media))
}

fn valid_effort(effort: &Value) -> bool {
    match effort {
        Value::String(tier) => EFFORT_TIERS.contains(&tier.as_str()),
        Value::Number(n) => n.as_f64().is_some_and(|f| (0.0..=0.99).contains(&f)),
        _ => false,
    }
}

/// `ChatCompletionRequest.normalize_reasoning_inputs`: lift `reasoning` into
/// `reasoning_effort` and return the `thinking` it implies.
fn normalize_reasoning_inputs(raw: &mut Map<String, Value>) -> Result<Option<Value>, Unsupported> {
    let mut thinking = None;
    if let Some(Value::Object(reasoning)) = raw.get("reasoning").cloned() {
        let effort = match reasoning.get("effort").filter(|e| !e.is_null()) {
            Some(effort) => effort.clone(),
            None => reasoning
                .get("reasoning_effort")
                .cloned()
                .unwrap_or(Value::Null),
        };
        match &effort {
            Value::String(tier) if EFFORT_TIERS.contains(&tier.as_str()) => {
                raw.insert("reasoning_effort".into(), effort.clone());
            }
            Value::Number(n) => {
                raw.insert("reasoning_effort".into(), json!(n.as_f64()));
            }
            Value::Null => {}
            _ => return Err(Unsupported("invalid_request")),
        }
        let enabled = match reasoning.get("enabled").filter(|e| !e.is_null()) {
            Some(enabled) => Some(enabled),
            None => reasoning.get("enable"),
        };
        let enabled = match enabled {
            Some(Value::String(s)) => {
                ["1", "true", "yes", "y", "on"].contains(&s.trim().to_lowercase().as_str())
            }
            Some(Value::Bool(b)) => *b,
            Some(Value::Null) | None => false,
            Some(_) => return Err(Unsupported("invalid_request")),
        };
        if enabled {
            thinking = Some(Value::Bool(true));
        }
    }
    if let Some(effort) = raw.get("reasoning_effort").filter(|e| !e.is_null()) {
        if effort.is_boolean() {
            return Err(Unsupported("invalid_request"));
        }
        thinking = Some(Value::Bool(effort != "none"));
    }
    Ok(thinking)
}

/// `OpenAIServingChat._validate_request`, for the checks requests here can fail.
fn validate(
    request: &ChatRequest,
    settings: &OpenAiSettings,
    model: &ChatModel,
) -> Result<(), Unsupported> {
    let invalid = Err(Unsupported("invalid_request"));
    if request.messages.is_empty() || request.n < 1 {
        return invalid;
    }
    if !request.messages.iter().all(valid_message) {
        return invalid;
    }
    if request
        .chat_template_kwargs
        .as_ref()
        .is_some_and(|k| k.contains_key("chat_template"))
    {
        return invalid;
    }
    if request
        .logit_bias
        .as_ref()
        .is_some_and(|b| !b.values().all(Value::is_number))
    {
        return invalid;
    }
    let max_output = request
        .max_completion_tokens
        .filter(|&n| n != 0)
        .or(request.max_tokens);
    if let Some(max_output) = max_output.filter(|&n| n != 0) {
        let Some(context) = settings.context_length.or(model.context_length) else {
            return Err(Unsupported("context_length_unknown"));
        };
        if max_output > context as i64 && !settings.allow_auto_truncate {
            return invalid;
        }
    }
    format_json_schema(request.response_format.as_ref())?;
    Ok(())
}

/// Whether Pydantic accepts `message` as a `ChatCompletionMessageParam` with
/// text-only content. The renderer drops content parts it does not know, so
/// any other message goes to the engine, which validates it as Python does.
fn valid_message(message: &Value) -> bool {
    let Some(message) = message.as_object() else {
        return false;
    };
    let text_content = |content: &Value| match content {
        Value::String(_) => true,
        Value::Array(parts) => parts
            .iter()
            .all(|part| part["type"] == "text" && part["text"].is_string()),
        _ => false,
    };
    match message.get("role").and_then(Value::as_str) {
        // `ChatCompletionMessageUserParam`: the role is case-sensitive and content is required.
        Some("user") => message.get("content").is_some_and(text_content),
        Some(role) if GENERIC_ROLES.contains(&role.to_lowercase().as_str()) => {
            optional(message, "content", text_content)
                && optional(message, "name", Value::is_string)
                && optional(message, "tool_call_id", Value::is_string)
                && optional(message, "reasoning_content", Value::is_string)
                && optional(message, "phase", |p| {
                    p == "commentary" || p == "final_answer"
                })
                && optional(message, "tool_calls", |calls| {
                    calls
                        .as_array()
                        .is_some_and(|calls| calls.iter().all(valid_tool_call))
                })
                // Non-empty tools never get here.
                && optional(message, "tools", |tools| {
                    tools.as_array().is_some_and(Vec::is_empty)
                })
        }
        _ => false,
    }
}

/// `ToolCall`: `type` defaults to `"function"` but is not nullable.
fn valid_tool_call(call: &Value) -> bool {
    let Some(call) = call.as_object() else {
        return false;
    };
    optional(call, "id", Value::is_string)
        && optional(call, "index", |i| i.is_i64() || i.is_u64())
        && call.get("type").is_none_or(|t| t == "function")
        && call
            .get("function")
            .and_then(Value::as_object)
            .is_some_and(|function| {
                optional(function, "name", Value::is_string)
                    && optional(function, "arguments", |a| a.is_string() || a.is_object())
            })
}

/// An `Optional[...]` field: absent, null, or `valid`.
fn optional(object: &Map<String, Value>, key: &str, valid: impl Fn(&Value) -> bool) -> bool {
    object
        .get(key)
        .is_none_or(|value| value.is_null() || valid(value))
}

fn generate_body(
    request: &ChatRequest,
    kwargs: &Map<String, Value>,
    model: &ChatModel,
    prompt_ids: Vec<u32>,
    thinking: bool,
    headers: &OpenAiHeaders<'_>,
    settings: &OpenAiSettings,
) -> Value {
    // `ChatCompletionRequest.to_sampling_params`: request, then generation config, then OpenAI.
    let model_defaults = settings.sampling_defaults.as_deref().unwrap_or("model") == "model";
    let param = |value: Option<Value>, name: &str, default: Value| {
        let configured = model
            .generation_config
            .get(name)
            .filter(|v| model_defaults && !v.is_null());
        value.unwrap_or_else(|| configured.cloned().unwrap_or(default))
    };
    let logit_bias = float_logit_bias(&request.logit_bias);
    let max_new_tokens = request
        .max_completion_tokens
        .filter(|&n| n != 0)
        .or(request.max_tokens);
    let mut sampling = json!({
        "temperature": param(request.temperature.map(Value::from), "temperature", json!(1.0)),
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": request.min_tokens,
        "stop": request.stop,
        "stop_token_ids": request.stop_token_ids,
        "stop_regex": request.stop_regex,
        "top_p": param(request.top_p.map(Value::from), "top_p", json!(1.0)),
        "top_k": param(request.top_k.map(Value::from), "top_k", json!(-1)),
        "min_p": param(request.min_p.map(Value::from), "min_p", json!(0.0)),
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "repetition_penalty": param(request.repetition_penalty.map(Value::from), "repetition_penalty", json!(1.0)),
        "regex": request.regex,
        "ebnf": request.ebnf,
        "n": request.n,
        "no_stop_trim": request.no_stop_trim,
        "ignore_eos": request.ignore_eos,
        "skip_special_tokens": request.skip_special_tokens,
        "logit_bias": logit_bias,
        "custom_params": request.custom_params,
        "sampling_seed": request.seed,
        "spaces_between_special_tokens": kwargs.get("spaces_between_special_tokens").cloned().unwrap_or(json!(true)),
    });
    if let Ok(Some(schema)) = format_json_schema(request.response_format.as_ref()) {
        sampling["json_schema"] = schema.into();
    }
    json!({
        "input_ids": prompt_ids,
        "image_data": null,
        "video_data": null,
        "audio_data": null,
        "sampling_params": sampling,
        "return_logprob": request.logprobs,
        "logprob_start_len": -1,
        "top_logprobs_num": request.top_logprobs.unwrap_or(0),
        "return_sampling_mask": false,
        "sampling_logprobs_mode": null,
        "stream": request.stream,
        "return_text_in_logprobs": true,
        "modalities": [],
        "lora_path": lora_path(&request.model, &request.lora_path),
        "bootstrap_host": request.bootstrap_host,
        "bootstrap_port": request.bootstrap_port,
        "bootstrap_room": request.bootstrap_room,
        "routed_dp_rank": request.routed_dp_rank.or(request.data_parallel_rank),
        "disagg_prefill_dp_rank": request.disagg_prefill_dp_rank,
        "return_hidden_states": false,
        "return_routed_experts": false,
        "routed_experts_start_len": 0,
        "rid": request.rid,
        "session_id": request.session_id,
        "extra_key": request.extra_key,
        "cache_salt": request.cache_salt,
        "require_reasoning": thinking,
        "priority": request.priority,
        "routing_key": headers.routing_key,
        "custom_labels": custom_labels(headers, settings).unwrap_or_default(),
        "custom_logit_processor": request.custom_logit_processor,
        "images_config": request.images_config,
        "video_config": null,
        "image_max_dynamic_patch": null,
        "video_max_dynamic_patch": null,
        "max_dynamic_patch": null,
        "use_audio_in_video": false,
        "return_prompt_token_ids": false,
    })
}

struct ReasoningConfig {
    parser: String,
    options: ReasoningOptions,
}

/// Builds the OpenAI chat response from the engine's `/generate` output.
pub struct ChatResponder {
    model: String,
    n: usize,
    logprobs: bool,
    reasoning: Option<ReasoningConfig>,
    include_usage: bool,
    continuous_usage_stats: bool,
    settings: OpenAiSettings,
    tokenizer: Option<Arc<dyn OpenAiTokenizer>>,
    stream: StreamState,
    roles_sent: HashSet<u64>,
    detectors: HashMap<u64, ReasoningStreamSplitter>,
    finish_reasons: Vec<(u64, Value)>,
}

impl ChatResponder {
    /// `_handle_non_streaming_request` over a `/generate` response.
    pub fn unary(self, body: &[u8]) -> Option<Reply> {
        let ret = match serde_json::from_slice(body).ok()? {
            Value::Array(items) => items,
            item => vec![item],
        };
        let mut choices = Vec::with_capacity(ret.len());
        for (index, item) in ret.iter().enumerate() {
            let meta = &item["meta_info"];
            let logprobs = self.logprobs.then(|| {
                json!({"content": self.token_logprobs(&meta["output_token_logprobs"], &meta["output_top_logprobs"])})
            });
            let Some(text) = item["text"].as_str() else {
                return Some(malformed_output_reply("text"));
            };
            let mut text = text.to_owned();
            let mut reasoning = String::new();
            if let Some(config) = &self.reasoning {
                (reasoning, text) =
                    split_reasoning::<u32>(Some(&config.parser), &config.options, &text, &[]);
            }
            let finish_reason = &meta["finish_reason"];
            choices.push(json!({
                "index": index,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "reasoning_content": (!reasoning.is_empty()).then_some(reasoning),
                    "tool_calls": null,
                },
                "logprobs": logprobs,
                "finish_reason": finish_reason.get("type"),
                "matched_stop": finish_reason.get("matched"),
            }));
        }
        unary_reply(
            &ret,
            self.n,
            self.settings.enable_cache_report,
            "chat.completion",
            &self.model,
            choices,
        )
    }

    /// One `/generate` SSE `data:` payload, as `_generate_chat_stream` handles
    /// it. `Err` replaces the whole stream, before anything was sent.
    pub fn stream_data(&mut self, data: &[u8]) -> Result<Vec<String>, Reply> {
        let mut events = match self.stream.payload(data)? {
            Payload::Done => return Ok(self.stream_end()),
            Payload::Frame(content) => self.stream_frame(&content),
            Payload::Events(events) => events,
        };
        if self.stream.ending() {
            events.extend(self.stream_end());
        }
        Ok(events)
    }

    /// Whether the stream has ended; later `/generate` frames are ignored.
    pub fn done(&self) -> bool {
        self.stream.done
    }

    fn stream_frame(&mut self, content: &Value) -> Vec<String> {
        let index = content.get("index").and_then(Value::as_u64).unwrap_or(0);
        let meta = &content["meta_info"];
        self.stream.record(index, meta);
        let finish_reason = meta["finish_reason"].clone();
        let finish_type = finish_reason["type"].as_str().map(str::to_owned);

        let mut logprobs = None;
        if self.logprobs {
            let prev = self.stream.logprob_counts.get(&index).copied().unwrap_or(0);
            let total = meta_u64(meta, "output_token_logprobs_length") as usize;
            if prev < total {
                let incremental = self.settings.incremental_streaming_output;
                let slice = |key: &str| -> Value {
                    let items = meta[key].as_array().cloned().unwrap_or_default();
                    match incremental {
                        true => Value::Array(items),
                        false => {
                            Value::Array(items.into_iter().skip(prev).take(total - prev).collect())
                        }
                    }
                };
                let content = self.token_logprobs(
                    &slice("output_token_logprobs"),
                    &slice("output_top_logprobs"),
                );
                logprobs = Some(json!({"content": content}));
            }
            self.stream.logprob_counts.insert(index, total);
        }

        let mut events = Vec::new();
        if let Some(finish_type) = &finish_type {
            // An abort carrying a status is an error; Python checks for an
            // `HTTPStatus`, which `/generate` serializes as its integer.
            if finish_type == "abort"
                && let Some(code) = finish_reason["status_code"].as_u64()
                && let Some(name) = http_status_name(code)
            {
                let message = finish_reason["message"]
                    .as_str()
                    .unwrap_or("Generation aborted.");
                self.stream.stopped = true;
                return vec![stream_error_event(message, name, code as u16)];
            }
            self.finish_reasons.push((index, finish_reason.clone()));
        }
        if self.roles_sent.insert(index) {
            events.push(self.chunk(
                index,
                json!({"reasoning_content": null, "role": "assistant", "content": ""}),
                None,
                None,
            ));
        }

        // `_generate_stream_content`.
        let text = content["text"].as_str().unwrap_or_default();
        let mut delta = match self.settings.incremental_streaming_output {
            true => text.to_owned(),
            false => {
                let offset = self
                    .stream
                    .text_offsets
                    .insert(index, text.chars().count())
                    .unwrap_or(0);
                text.chars().skip(offset).collect()
            }
        };
        let mut remaining_logprobs = logprobs;
        if let Some(config) = &self.reasoning {
            let detector = self.detectors.entry(index).or_insert_with(|| {
                ReasoningStreamSplitter::new(Some(&config.parser), config.options.clone())
            });
            let (mut reasoning, mut normal) = detector.split::<u32>(&delta, &[]);
            if finish_type.as_deref().is_some_and(|t| t != "abort") {
                let (end_reasoning, end_normal) = detector.finish();
                reasoning.push_str(&end_reasoning);
                normal.push_str(&end_normal);
            }
            delta = normal;
            if !reasoning.is_empty() {
                let usage = self.chunk_usage(index);
                events.push(self.chunk(
                    index,
                    json!({"reasoning_content": reasoning}),
                    remaining_logprobs.take(),
                    usage,
                ));
            }
        }
        if !delta.is_empty() {
            let usage = self.chunk_usage(index);
            events.push(self.chunk(
                index,
                json!({"reasoning_content": null, "content": delta}),
                remaining_logprobs.take(),
                usage,
            ));
        }
        let has_parser =
            self.settings.reasoning_parser.is_some() || self.settings.tool_call_parser.is_some();
        if remaining_logprobs.is_some() && has_parser {
            let usage = self.chunk_usage(index);
            events.push(self.chunk(
                index,
                json!({"reasoning_content": null}),
                remaining_logprobs,
                usage,
            ));
        }
        events
    }

    fn stream_end(&mut self) -> Vec<String> {
        self.stream.done = true;
        let mut events = Vec::new();
        if !self.stream.failed && self.stream.started {
            for (index, reason) in std::mem::take(&mut self.finish_reasons) {
                let mut chunk =
                    self.chunk_json(index, json!({"reasoning_content": null}), None, None);
                chunk["choices"][0]["finish_reason"] = reason["type"].clone();
                chunk["choices"][0]["matched_stop"] =
                    reason.get("matched").cloned().unwrap_or(Value::Null);
                events.push(format!("data: {chunk}\n\n"));
            }
            if self.include_usage {
                let chunk = json!({
                    "id": self.stream.last_id,
                    "object": "chat.completion.chunk",
                    "created": now(),
                    "model": self.model,
                    "choices": [],
                    "usage": self.stream.total_usage(self.n, self.settings.enable_cache_report),
                });
                events.push(format!("data: {chunk}\n\n"));
            }
        }
        events.push("data: [DONE]\n\n".into());
        events
    }

    /// `build_sse_content`: msgspec drops a null `usage`, never a null choice field.
    fn chunk(
        &self,
        index: u64,
        delta: Value,
        logprobs: Option<Value>,
        usage: Option<Value>,
    ) -> String {
        format!(
            "data: {}\n\n",
            self.chunk_json(index, delta, logprobs, usage)
        )
    }

    fn chunk_json(
        &self,
        index: u64,
        delta: Value,
        logprobs: Option<Value>,
        usage: Option<Value>,
    ) -> Value {
        let mut chunk = json!({
            "id": self.stream.last_id,
            "object": "chat.completion.chunk",
            "created": now(),
            "model": self.model,
            "choices": [{
                "index": index,
                "delta": delta,
                "logprobs": logprobs,
                "finish_reason": null,
                "matched_stop": null,
            }],
        });
        if let Some(usage) = usage {
            chunk["usage"] = usage;
        }
        chunk
    }

    fn chunk_usage(&self, index: u64) -> Option<Value> {
        self.continuous_usage_stats.then(|| {
            self.stream
                .choice_usage(index, self.settings.enable_cache_report)
        })
    }

    /// `_build_token_logprobs_from_raw`.
    fn token_logprobs(&self, output: &Value, top: &Value) -> Vec<Value> {
        let top_rows = top.as_array();
        let mut out = Vec::new();
        for (i, triple) in output.as_array().into_iter().flatten().enumerate() {
            let text = triple[2].as_str().unwrap_or_default();
            let top_logprobs: Vec<Value> = top_rows
                .and_then(|rows| rows.get(i))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|t| {
                    let text = t[2].as_str().unwrap_or_default();
                    json!({"token": text, "bytes": self.token_bytes(&t[1], text), "logprob": t[0]})
                })
                .collect();
            out.push(json!({
                "token": text,
                "bytes": self.token_bytes(&triple[1], text),
                "logprob": triple[0],
                "top_logprobs": top_logprobs,
            }));
        }
        out
    }

    fn token_bytes(&self, id: &Value, text: &str) -> Vec<u8> {
        id.as_u64()
            .and_then(|id| self.tokenizer.as_ref()?.byte_level_bytes(id as u32))
            .unwrap_or_else(|| text.as_bytes().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reason `messages` is not lowered, with a renderer that accepts anything.
    fn unsupported(messages: Value) -> Option<&'static str> {
        let body = serde_json::to_vec(&json!({"model": "m", "messages": messages})).unwrap();
        let render = |_: &Value| Some(vec![1]);
        let settings = OpenAiSettings::default();
        lower_chat(
            &body,
            &OpenAiHeaders::default(),
            &settings,
            &ChatModel::default(),
            render,
            None,
        )
        .err()
        .map(|reason| reason.0)
    }

    #[test]
    fn messages_python_rejects_go_to_the_engine() {
        for messages in [
            json!([{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "file", "file": {"file_id": "f"}}]}]),
            json!([{"role": "user"}]),
            json!([{"role": "user", "content": null}]),
            json!([{"role": "USER", "content": "hi"}]),
            json!([{"role": "bot", "content": "hi"}]),
            json!([{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]),
            json!([{"role": "user", "content": [{"type": "text"}]}]),
            json!([{"role": "user", "content": 1}]),
            json!(["hi"]),
            json!([{"role": "assistant", "content": "a", "name": 1}]),
            json!([{"role": "assistant", "content": "a", "phase": "draft"}]),
            json!([{"role": "system", "content": "a", "tools": false}]),
            json!([{"role": "assistant", "content": null, "tool_calls": [{"function": null}]}]),
            json!([{"role": "assistant", "content": null, "tool_calls": [{"type": null, "function": {}}]}]),
            json!([{"role": "assistant", "content": null, "tool_calls": [{"function": {"arguments": 1}}]}]),
        ] {
            assert_eq!(
                unsupported(messages.clone()),
                Some("invalid_request"),
                "{messages}"
            );
        }
    }

    #[test]
    fn text_messages_are_lowered() {
        let messages = json!([
            {"role": "System", "content": "be brief", "tools": []},
            {"role": "user", "content": [{"type": "text", "text": "hi"}], "name": 1},
            {"role": "assistant", "content": null, "phase": null, "tool_calls": [
                {"id": "c", "index": 0, "type": "function", "function": {"name": "f", "arguments": {}}}
            ]},
            {"role": "tool", "content": "42", "tool_call_id": "c"},
        ]);
        assert_eq!(unsupported(messages), None);
    }
}
