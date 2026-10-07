//! `/v1/completions`, as `serving_completions.py` serves it.

use std::collections::HashMap;
use std::sync::Arc;

use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::pieces::byte_level_byte;
use super::wire::{
    Reply, error_reply, http_status_name, malformed_output_reply, meta_u64, py_strip, python_json,
    stream_error_event, usage, without_nulls,
};
use super::{OpenAiHeaders, OpenAiSettings, OpenAiTokenizer, Unsupported};

/// `CompletionRequest` with Pydantic's defaults. A body this rejects goes to
/// the engine, which answers with its own validation error.
#[derive(Deserialize)]
struct CompletionRequest {
    #[serde(default = "default_model")]
    model: String,
    prompt: Prompt,
    #[allow(dead_code)]
    best_of: Option<i64>,
    #[serde(default)]
    echo: bool,
    #[serde(default)]
    frequency_penalty: f64,
    logit_bias: Option<Map<String, Value>>,
    logprobs: Option<i64>,
    #[serde(default = "default_max_tokens")]
    max_tokens: i64,
    #[serde(default = "one")]
    n: i64,
    #[serde(default)]
    presence_penalty: f64,
    seed: Option<i64>,
    stop: Option<OneOrList<String>>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    #[allow(dead_code)]
    suffix: Option<String>,
    #[serde(default = "one_f64")]
    temperature: f64,
    #[serde(default = "one_f64")]
    top_p: f64,
    #[allow(dead_code)]
    user: Option<String>,
    return_hidden_states: Option<Value>,
    #[serde(default)]
    return_routed_experts: bool,
    #[serde(default)]
    routed_experts_start_len: i64,
    #[serde(default)]
    return_cached_tokens_details: bool,
    #[serde(default)]
    return_spec_tokens_details: bool,
    #[serde(default)]
    return_token_ids: bool,
    #[serde(default = "minus_one")]
    top_k: i64,
    #[serde(default)]
    min_p: f64,
    #[serde(default)]
    min_tokens: i64,
    json_schema: Option<String>,
    regex: Option<String>,
    ebnf: Option<String>,
    #[serde(default = "one_f64")]
    repetition_penalty: f64,
    stop_token_ids: Option<Vec<i64>>,
    stop_regex: Option<OneOrList<String>>,
    #[serde(default)]
    no_stop_trim: bool,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default = "yes")]
    skip_special_tokens: bool,
    lora_path: Option<OneOrList<Option<String>>>,
    session_id: Option<String>,
    #[allow(dead_code)]
    session_params: Option<Map<String, Value>>,
    response_format: Option<ResponseFormat>,
    custom_params: Option<Map<String, Value>>,
    custom_logit_processor: Option<String>,
    images_config: Option<Map<String, Value>>,
    data_parallel_rank: Option<i64>,
    rid: Option<OneOrList<String>>,
    extra_key: Option<OneOrList<String>>,
    cache_salt: Option<OneOrList<String>>,
    priority: Option<i64>,
    #[allow(dead_code)]
    custom_labels: Option<HashMap<String, String>>,
    bootstrap_host: Option<OneOrList<String>>,
    bootstrap_port: Option<OneOrList<Option<i64>>>,
    bootstrap_room: Option<OneOrList<i64>>,
    routed_dp_rank: Option<i64>,
    disagg_prefill_dp_rank: Option<i64>,
}

fn default_model() -> String {
    "default".into()
}
fn default_max_tokens() -> i64 {
    16
}
fn one() -> i64 {
    1
}
fn one_f64() -> f64 {
    1.0
}
fn minus_one() -> i64 {
    -1
}
fn yes() -> bool {
    true
}

#[derive(Deserialize, serde::Serialize)]
#[serde(untagged)]
enum OneOrList<T> {
    One(T),
    List(Vec<T>),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Prompt {
    Ids(Vec<u32>),
    IdBatch(Vec<Vec<u32>>),
    Text(String),
    TextBatch(Vec<String>),
}

#[derive(Deserialize)]
struct StreamOptions {
    include_usage: Option<bool>,
    continuous_usage_stats: Option<bool>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchemaFormat },
}

#[derive(Deserialize)]
struct JsonSchemaFormat {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    description: Option<String>,
    schema: Option<Map<String, Value>>,
    #[allow(dead_code)]
    strict: Option<bool>,
}

/// Lower a completions body into a `/generate` body, as
/// `OpenAIServingCompletion._convert_to_internal_request` does.
pub fn lower_completion(
    body: &[u8],
    headers: &OpenAiHeaders<'_>,
    settings: &OpenAiSettings,
    tokenizer: Option<Arc<dyn OpenAiTokenizer>>,
) -> Result<(Value, CompletionResponder), Unsupported> {
    let mut request: CompletionRequest =
        serde_json::from_slice(body).map_err(|_| Unsupported("unparsed_request"))?;
    if settings.completion_template.is_some() {
        return Err(Unsupported("completion_template"));
    }
    let numeric_bias = request
        .logit_bias
        .as_ref()
        .is_none_or(|bias| bias.values().all(Value::is_number));
    if request.max_tokens <= 0 || request.n < 1 || !numeric_bias || prompt_is_empty(&request.prompt)
    {
        return Err(Unsupported("invalid_request"));
    }
    if request
        .return_hidden_states
        .as_ref()
        .is_some_and(|v| v != false)
        || request.return_routed_experts
        || request.return_cached_tokens_details
        || request.return_spec_tokens_details
    {
        return Err(Unsupported("sglext_output"));
    }
    if !custom_labels_are_strings(&custom_labels(headers, settings)) {
        return Err(Unsupported("invalid_request"));
    }
    if (request.echo || request.logprobs.is_some()) && tokenizer.is_none() {
        return Err(Unsupported("no_tokenizer"));
    }
    // A `structural_tag` format fails to parse above, so it goes to the engine.
    match &request.response_format {
        Some(ResponseFormat::JsonSchema { json_schema }) => {
            let schema = json_schema
                .schema
                .clone()
                .ok_or(Unsupported("invalid_request"))?;
            request.json_schema = Some(python_json(&Value::Object(schema)));
        }
        Some(ResponseFormat::JsonObject) => {
            request.json_schema = Some(r#"{"type": "object"}"#.into());
        }
        Some(ResponseFormat::Text) | None => {}
    }
    if request.routed_dp_rank.is_none() {
        request.routed_dp_rank = request.data_parallel_rank;
    }

    let echo_prompts = if request.echo {
        echo_prompts(&request.prompt, tokenizer.as_deref()).ok_or(Unsupported("echo_decode"))?
    } else {
        Vec::new()
    };
    let lowered = generate_body(&request, headers, settings);
    // `should_include_usage`.
    let default_usage = settings.stream_response_default_include_usage;
    let (include_usage, continuous_usage_stats) = match &request.stream_options {
        Some(options) => (
            options.include_usage.unwrap_or(false) || default_usage,
            options.continuous_usage_stats.unwrap_or(false),
        ),
        None => (default_usage, false),
    };
    let responder = CompletionResponder {
        model: request.model,
        n: request.n as usize,
        echo_prompts,
        logprobs: request.logprobs.is_some(),
        echo_logprobs: request.echo && request.logprobs.is_some_and(|n| n != 0),
        return_token_ids: request.return_token_ids,
        include_usage,
        continuous_usage_stats,
        settings: settings.clone(),
        tokenizer,
        created: now(),
        stream: StreamState::default(),
    };
    Ok((lowered, responder))
}

fn generate_body(
    request: &CompletionRequest,
    headers: &OpenAiHeaders<'_>,
    settings: &OpenAiSettings,
) -> Value {
    let logit_bias = request.logit_bias.as_ref().map(|bias| {
        bias.iter()
            .map(|(k, v)| (k.clone(), json!(v.as_f64())))
            .collect::<Map<_, _>>()
    });
    let sampling = json!({
        "temperature": request.temperature,
        "max_new_tokens": request.max_tokens,
        "min_new_tokens": request.min_tokens,
        "stop": request.stop,
        "stop_token_ids": request.stop_token_ids,
        "stop_regex": request.stop_regex,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "min_p": request.min_p,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "repetition_penalty": request.repetition_penalty,
        "regex": request.regex,
        "json_schema": request.json_schema,
        "ebnf": request.ebnf,
        "n": request.n,
        "no_stop_trim": request.no_stop_trim,
        "ignore_eos": request.ignore_eos,
        "skip_special_tokens": request.skip_special_tokens,
        "logit_bias": logit_bias,
        "custom_params": request.custom_params,
        "sampling_seed": request.seed,
    });
    let (prompt_key, prompt) = match &request.prompt {
        Prompt::Text(text) => ("text", json!(text)),
        Prompt::TextBatch(texts) => ("text", json!(texts)),
        Prompt::Ids(ids) => ("input_ids", json!(ids)),
        Prompt::IdBatch(ids) => ("input_ids", json!(ids)),
    };
    let lora_path = match request.model.split_once(':') {
        Some((_, adapter)) if !py_strip(adapter).is_empty() => json!(py_strip(adapter)),
        _ => json!(request.lora_path),
    };
    let mut body = Map::new();
    body.insert(prompt_key.into(), prompt);
    let fields = json!({
        "sampling_params": sampling,
        "return_logprob": request.logprobs.is_some(),
        "top_logprobs_num": request.logprobs.unwrap_or(0),
        "logprob_start_len": if request.echo && request.logprobs.is_some_and(|n| n != 0) { 0 } else { -1 },
        "return_text_in_logprobs": true,
        "stream": request.stream,
        "lora_path": lora_path,
        "bootstrap_host": request.bootstrap_host,
        "bootstrap_port": request.bootstrap_port,
        "bootstrap_room": request.bootstrap_room,
        "routed_dp_rank": request.routed_dp_rank,
        "disagg_prefill_dp_rank": request.disagg_prefill_dp_rank,
        "return_hidden_states": false,
        "return_routed_experts": false,
        "routed_experts_start_len": request.routed_experts_start_len,
        "return_prompt_token_ids": request.return_token_ids,
        "rid": request.rid,
        "session_id": request.session_id,
        "extra_key": request.extra_key,
        "cache_salt": request.cache_salt,
        "priority": request.priority,
        "routing_key": headers.routing_key,
        "custom_labels": custom_labels(headers, settings),
        "custom_logit_processor": request.custom_logit_processor,
        "images_config": request.images_config,
    });
    body.extend(fields.as_object().expect("object literal").clone());
    Value::Object(body)
}

/// `OpenAIServingBase.extract_custom_labels`.
fn custom_labels(headers: &OpenAiHeaders<'_>, settings: &OpenAiSettings) -> Value {
    let allowed = settings
        .tokenizer_metrics_allowed_custom_labels
        .as_deref()
        .unwrap_or_default();
    if allowed.is_empty() || settings.tokenizer_metrics_custom_labels_header.is_none() {
        return Value::Null;
    }
    match headers
        .custom_labels
        .and_then(|raw| serde_json::from_str(raw).ok())
    {
        Some(Value::Object(labels)) => Value::Object(
            labels
                .into_iter()
                .filter(|(label, _)| allowed.contains(label))
                .collect(),
        ),
        _ => Value::Null,
    }
}

/// Custom labels `/generate` accepts as its `Dict[str, str]`.
pub(super) fn custom_labels_are_strings(labels: &Value) -> bool {
    labels
        .as_object()
        .is_none_or(|labels| labels.values().all(Value::is_string))
}

fn prompt_is_empty(prompt: &Prompt) -> bool {
    match prompt {
        Prompt::Text(text) => text.is_empty(),
        Prompt::TextBatch(texts) => texts.iter().all(String::is_empty),
        // Python's `not 0` is true, so all-zero ids count as empty.
        Prompt::Ids(ids) => ids.iter().all(|&id| id == 0),
        Prompt::IdBatch(ids) => ids.iter().all(Vec::is_empty),
    }
}

/// `_prepare_echo_prompts`: the text echoed before each prompt's choices.
fn echo_prompts(prompt: &Prompt, tokenizer: Option<&dyn OpenAiTokenizer>) -> Option<Vec<String>> {
    Some(match prompt {
        Prompt::Text(text) => vec![text.clone()],
        Prompt::TextBatch(texts) => texts.clone(),
        Prompt::Ids(ids) => vec![tokenizer?.decode(ids)?],
        Prompt::IdBatch(ids) => ids
            .iter()
            .map(|ids| tokenizer?.decode(ids))
            .collect::<Option<_>>()?,
    })
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Builds the OpenAI response from the engine's `/generate` output.
pub struct CompletionResponder {
    model: String,
    n: usize,
    echo_prompts: Vec<String>,
    logprobs: bool,
    echo_logprobs: bool,
    return_token_ids: bool,
    include_usage: bool,
    continuous_usage_stats: bool,
    settings: OpenAiSettings,
    tokenizer: Option<Arc<dyn OpenAiTokenizer>>,
    created: u64,
    stream: StreamState,
}

#[derive(Default)]
struct StreamState {
    started: bool,
    stopped: bool,
    failed: bool,
    last_id: Value,
    text_offsets: HashMap<u64, usize>,
    logprob_counts: HashMap<u64, usize>,
    token_id_counts: HashMap<u64, usize>,
    prompt_tokens: HashMap<u64, u64>,
    completion_tokens: HashMap<u64, u64>,
    reasoning_tokens: HashMap<u64, u64>,
    cached_tokens: HashMap<u64, u64>,
}

impl CompletionResponder {
    /// `_handle_non_streaming_request` over a `/generate` response.
    pub fn unary(self, body: &[u8]) -> Option<Reply> {
        let ret = match serde_json::from_slice(body).ok()? {
            Value::Array(items) => items,
            item => vec![item],
        };
        let mut choices = Vec::with_capacity(ret.len());
        for (index, item) in ret.iter().enumerate() {
            let meta = &item["meta_info"];
            let Some(text) = item["text"].as_str() else {
                return Some(malformed_output_reply("text"));
            };
            let mut text = text.to_owned();
            if let Some(echo) = self.echo_prompts.get(index / self.n) {
                text.insert_str(0, echo);
            }
            let logprobs = self.logprobs.then(|| {
                self.openai_logprobs(
                    self.echo_logprobs.then_some(meta),
                    &meta["output_token_logprobs"],
                    &meta["output_top_logprobs"],
                )
            });
            choices.push(self.choice(index as u64, text, logprobs, meta, item));
        }
        let first = &ret.first()?["meta_info"];
        let stride = |key: &str| -> u64 {
            ret.iter()
                .step_by(self.n)
                .map(|r| meta_u64(&r["meta_info"], key))
                .sum()
        };
        let all = |key: &str| -> u64 { ret.iter().map(|r| meta_u64(&r["meta_info"], key)).sum() };
        let cached = self
            .settings
            .enable_cache_report
            .then(|| stride("cached_tokens"));
        let mut metadata = json!({"weight_version": first["weight_version"]});
        if let Some(versions) = first.get("weight_versions") {
            metadata["weight_versions"] = versions.clone();
        }
        let response = json!({
            "id": first["id"],
            "object": "text_completion",
            "created": now(),
            "model": self.model,
            "choices": choices,
            "usage": usage(stride("prompt_tokens"), all("completion_tokens"), all("reasoning_tokens"), cached),
            "metadata": metadata,
        });
        Some(Reply {
            status: 200,
            body: response.to_string(),
        })
    }

    /// One `/generate` SSE `data:` payload, as `_generate_completion_stream`
    /// handles it. `Err` replaces the whole stream, before anything was sent.
    pub fn stream_data(&mut self, data: &[u8]) -> Result<Vec<String>, Reply> {
        if data == b"[DONE]" {
            return Ok(self.stream_end());
        }
        if self.stream.stopped || self.stream.failed {
            return Ok(Vec::new());
        }
        let Ok(content) = serde_json::from_slice::<Value>(data) else {
            return Ok(Vec::new());
        };
        if let Some(message) = content.pointer("/error/message").and_then(Value::as_str) {
            if !self.stream.started {
                return Err(error_reply(message, "BadRequestError", 400));
            }
            self.stream.failed = true;
            return Ok(vec![stream_error_event(message, "BadRequestError", 400)]);
        }
        self.stream.started = true;
        Ok(self.stream_chunk(&content).into_iter().collect())
    }

    fn stream_chunk(&mut self, content: &Value) -> Option<String> {
        let state = &mut self.stream;
        let index = content.get("index").and_then(Value::as_u64).unwrap_or(0);
        let meta = &content["meta_info"];
        state.last_id = meta["id"].clone();
        state
            .prompt_tokens
            .insert(index, meta_u64(meta, "prompt_tokens"));
        state
            .completion_tokens
            .insert(index, meta_u64(meta, "completion_tokens"));
        state
            .reasoning_tokens
            .insert(index, meta_u64(meta, "reasoning_tokens"));
        state
            .cached_tokens
            .insert(index, meta_u64(meta, "cached_tokens"));
        let finish_reason = &meta["finish_reason"];
        let raw_text = content["text"].as_str().unwrap_or_default();

        let is_first_chunk = !state.text_offsets.contains_key(&index);
        let offset = state.text_offsets.get(&index).copied().unwrap_or(0);
        let incremental = self.settings.incremental_streaming_output;
        let mut delta = if incremental {
            raw_text.to_owned()
        } else {
            raw_text.chars().skip(offset).collect()
        };
        state.text_offsets.insert(index, raw_text.chars().count());
        if is_first_chunk && let Some(echo) = self.echo_prompts.get(index as usize / self.n) {
            delta.insert_str(0, echo);
        }

        let mut logprobs = Value::Null;
        if self.logprobs {
            let echo_input = is_first_chunk && self.echo_logprobs;
            let prev = state.logprob_counts.get(&index).copied().unwrap_or(0);
            let total = meta_u64(meta, "output_token_logprobs_length") as usize;
            if prev < total || echo_input {
                let slice = |key: &str| -> Value {
                    let items = meta
                        .get(key)
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    match incremental {
                        true => Value::Array(items),
                        false => Value::Array(
                            items
                                .into_iter()
                                .skip(prev)
                                .take(total.saturating_sub(prev))
                                .collect(),
                        ),
                    }
                };
                let (output, output_top) =
                    (slice("output_token_logprobs"), slice("output_top_logprobs"));
                logprobs = self.openai_logprobs(echo_input.then_some(meta), &output, &output_top);
            }
            self.stream.logprob_counts.insert(index, total);
        }

        let mut token_ids = None;
        let mut prompt_token_ids = None;
        if self.return_token_ids {
            let ids = content["output_ids"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            token_ids = Some(match incremental {
                true => ids,
                false => {
                    let prev = self
                        .stream
                        .token_id_counts
                        .insert(index, ids.len())
                        .unwrap_or(0);
                    ids.into_iter().skip(prev).collect()
                }
            });
            if is_first_chunk {
                prompt_token_ids = Some(
                    content
                        .get("prompt_token_ids")
                        .cloned()
                        .unwrap_or(Value::Null),
                );
            }
        }

        // An abort carrying a status is an error; Python checks for an
        // `HTTPStatus`, which `/generate` serializes as its integer.
        if finish_reason["type"] == "abort"
            && let Some(code) = finish_reason.get("status_code").and_then(Value::as_u64)
            && let Some(name) = http_status_name(code)
        {
            self.stream.stopped = true;
            let message = finish_reason["message"]
                .as_str()
                .unwrap_or("Generation aborted.");
            return Some(stream_error_event(message, name, code as u16));
        }

        let mut choice = json!({
            "index": index,
            "text": delta,
            "logprobs": logprobs,
            "finish_reason": finish_reason.get("type"),
            "matched_stop": finish_reason.get("matched"),
        });
        if let Some(ids) = token_ids {
            choice["token_ids"] = Value::Array(ids);
        }
        if let Some(Value::Array(ids)) = prompt_token_ids {
            choice["prompt_token_ids"] = Value::Array(ids);
        }
        let usage_value = match self.continuous_usage_stats {
            true => usage(
                self.stream.prompt_tokens[&index],
                self.stream.completion_tokens[&index],
                self.stream.reasoning_tokens[&index],
                None,
            ),
            false => Value::Null,
        };
        let chunk = json!({
            "id": meta["id"],
            "object": "text_completion",
            "created": self.created,
            "model": self.model,
            "choices": [choice],
            "usage": usage_value,
        });
        Some(format!("data: {chunk}\n\n"))
    }

    fn stream_end(&mut self) -> Vec<String> {
        let mut events = Vec::new();
        let state = &self.stream;
        if self.include_usage && !state.failed && state.started {
            let first_choice = |map: &HashMap<u64, u64>| -> u64 {
                map.iter()
                    .filter(|(i, _)| *i % self.n as u64 == 0)
                    .map(|(_, v)| v)
                    .sum()
            };
            let cached = self
                .settings
                .enable_cache_report
                .then(|| first_choice(&state.cached_tokens));
            let chunk = json!({
                "id": state.last_id,
                "object": "text_completion",
                "created": self.created,
                "model": self.model,
                "choices": [],
                "usage": usage(
                    first_choice(&state.prompt_tokens),
                    state.completion_tokens.values().sum(),
                    state.reasoning_tokens.values().sum(),
                    cached,
                ),
            });
            events.push(format!("data: {}\n\n", without_nulls(chunk)));
        }
        events.push("data: [DONE]\n\n".into());
        events
    }

    fn choice(
        &self,
        index: u64,
        text: String,
        logprobs: Option<Value>,
        meta: &Value,
        item: &Value,
    ) -> Value {
        let finish_reason = &meta["finish_reason"];
        let mut choice = json!({
            "index": index,
            "text": text,
            "logprobs": logprobs,
            "finish_reason": finish_reason.get("type"),
            "matched_stop": finish_reason.get("matched"),
        });
        if self.return_token_ids {
            for (key, ids) in [
                ("token_ids", item.get("output_ids")),
                ("prompt_token_ids", item.get("prompt_token_ids")),
            ] {
                if let Some(ids) = ids.filter(|ids| !ids.is_null()) {
                    choice[key] = ids.clone();
                }
            }
        }
        choice
    }

    /// `to_openai_style_logprobs`.
    /// `echo_meta` adds the prompt's logprobs from its `meta_info`.
    fn openai_logprobs(
        &self,
        echo_meta: Option<&Value>,
        output: &Value,
        output_top: &Value,
    ) -> Value {
        let input = echo_meta.map(|meta| &meta["input_token_logprobs"]);
        let input_top = echo_meta.map(|meta| &meta["input_top_logprobs"]);
        let mut text_offset = Vec::new();
        let mut token_logprobs = Vec::new();
        let mut tokens = Vec::new();
        let mut top_logprobs = Vec::new();
        for list in [input, Some(output)].into_iter().flatten() {
            for triple in list.as_array().into_iter().flatten() {
                tokens.push(json!(self.token_text(triple)));
                token_logprobs.push(triple[0].clone());
                text_offset.push(json!(-1));
            }
        }
        for list in [input_top, Some(output_top)].into_iter().flatten() {
            for position in list.as_array().into_iter().flatten() {
                top_logprobs.push(match position.as_array() {
                    Some(triples) => Value::Object(
                        triples
                            .iter()
                            .map(|t| (self.token_text(t), t[0].clone()))
                            .collect(),
                    ),
                    None => Value::Null,
                });
            }
        }
        json!({
            "text_offset": text_offset,
            "token_logprobs": token_logprobs,
            "tokens": tokens,
            "top_logprobs": top_logprobs,
        })
    }

    /// `_lossless_token_text`: a fragment of a multi-byte character, shown as
    /// U+FFFD by the engine, is rendered as latin-1 so its bytes round-trip.
    fn token_text(&self, triple: &Value) -> String {
        let text = triple[2].as_str();
        if let Some(text) = text
            && !text.contains('\u{FFFD}')
        {
            return text.to_owned();
        }
        let fallback = text.unwrap_or_default().to_owned();
        let bytes = triple[1]
            .as_u64()
            .and_then(|id| self.tokenizer.as_ref()?.byte_level_piece(id as u32))
            .and_then(|piece| {
                piece
                    .chars()
                    .map(byte_level_byte)
                    .collect::<Option<Vec<u8>>>()
            })
            .filter(|bytes| !bytes.is_empty());
        match bytes {
            Some(bytes) if std::str::from_utf8(&bytes).is_err() => {
                bytes.into_iter().map(char::from).collect()
            }
            _ => fallback,
        }
    }
}
