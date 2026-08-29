//! OpenAI wire types lowered into renderer-owned requests.

use std::collections::{BTreeMap, HashMap};

use dynamo_protocols::types::{
    ChatCompletionAudio, ChatCompletionFunctionCall, ChatCompletionFunctions,
    ChatCompletionRequestMessage, ChatCompletionStreamOptions, ChatCompletionTool,
    ChatCompletionToolChoiceOption, PredictionContent, Prompt, ResponseFormat, ServiceTier, Stop,
    WebSearchOptions,
};
use serde::Deserialize;
use serde_json::Value;

use crate::preprocessing::{GenerateRequestIdentity, TextRequestGroup};
use crate::{
    ChatRequest, GenerateRequestMetadata, GenerationOptions, OneOrMany, ReasoningEffort,
    RendererConfig, RendererError, SamplingDefaults, SamplingParams, SamplingParamsOverrides,
    TokenIds, TokenIdsRequest,
};

const MAX_OPENAI_CHOICES: usize = 4096;

#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum ResponseModality {
    Text,
    Audio,
}

fn reject_unsupported_fields(fields: &HashMap<String, Value>) -> Result<(), String> {
    if fields.is_empty() {
        return Ok(());
    }
    let mut names = fields.keys().cloned().collect::<Vec<_>>();
    names.sort_unstable();
    Err(format!(
        "unsupported request field{}: {}",
        if names.len() == 1 { "" } else { "s" },
        names.join(", ")
    ))
}

/// SGLang's OpenAI-compatible chat-completions request.
#[derive(Deserialize)]
pub(crate) struct ChatCompletionRequest {
    pub messages: Vec<ChatCompletionRequestMessage>,
    pub model: String,
    #[serde(default)]
    pub mm_processor_kwargs: Option<Value>,
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(default)]
    pub reasoning: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub n: Option<u8>,
    #[serde(default)]
    modalities: Option<Vec<ResponseModality>>,
    #[serde(default)]
    pub prediction: Option<PredictionContent>,
    #[serde(default)]
    pub audio: Option<ChatCompletionAudio>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub service_tier: Option<ServiceTier>,
    #[serde(default)]
    pub stop: Option<Stop>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub tools: Option<Vec<ChatCompletionTool>>,
    #[serde(default)]
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub function_call: Option<ChatCompletionFunctionCall>,
    #[serde(default)]
    pub functions: Option<Vec<ChatCompletionFunctions>>,
    #[serde(default)]
    pub web_search_options: Option<WebSearchOptions>,
    #[serde(default)]
    pub chat_template_kwargs: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub continue_final_message: bool,
    #[serde(flatten)]
    pub sampling_overrides: SamplingParamsOverrides,
    #[serde(flatten)]
    pub extensions: RequestExtensions,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, Value>,
}

/// SGLang's OpenAI-compatible legacy-completions request.
#[derive(Deserialize)]
pub(crate) struct CompletionRequest {
    pub model: String,
    pub prompt: Prompt,
    #[serde(default)]
    pub prompt_embeds: Option<String>,
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<u8>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(default)]
    pub logprobs: Option<u8>,
    #[serde(default)]
    pub echo: Option<bool>,
    #[serde(default)]
    pub stop: Option<Stop>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub best_of: Option<u8>,
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(flatten)]
    pub sampling_overrides: SamplingParamsOverrides,
    #[serde(flatten)]
    pub extensions: RequestExtensions,
    #[serde(flatten)]
    pub unsupported_fields: HashMap<String, Value>,
}

/// SGLang extensions carried by the OpenAI-compatible request contract.
#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct RequestExtensions {
    #[serde(default)]
    pub return_meta_info: Option<bool>,
    #[serde(default)]
    pub rid: Option<OneOrMany<String>>,
    #[serde(default)]
    pub cache_salt: Option<OneOrMany<String>>,
    #[serde(default)]
    pub extra_key: Option<OneOrMany<String>>,
    #[serde(default)]
    pub priority: Option<i64>,
    #[serde(default)]
    pub bootstrap_host: Option<OneOrMany<String>>,
    #[serde(default)]
    pub bootstrap_port: Option<OneOrMany<Option<i64>>>,
    #[serde(default)]
    pub bootstrap_room: Option<OneOrMany<i64>>,
    #[serde(default)]
    pub routed_dp_rank: Option<i64>,
    #[serde(default)]
    pub disagg_prefill_dp_rank: Option<i64>,
    #[serde(default)]
    pub data_parallel_rank: Option<i64>,
    #[serde(default)]
    pub session_id: Option<serde_json::Value>,
    #[serde(default)]
    pub session_params: Option<serde_json::Value>,
    #[serde(default)]
    pub lora_path: Option<serde_json::Value>,
    #[serde(default)]
    pub custom_logit_processor: Option<serde_json::Value>,
    #[serde(default)]
    pub image_data: Option<serde_json::Value>,
    #[serde(default)]
    pub video_data: Option<serde_json::Value>,
    #[serde(default)]
    pub audio_data: Option<serde_json::Value>,
    #[serde(default)]
    pub mm_hashes: Option<serde_json::Value>,
}

#[derive(Debug)]
struct ExpandedRequestContext {
    request_id: String,
    metadata: GenerateRequestMetadata,
}

impl RequestExtensions {
    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("session_id", &self.session_id),
            ("session_params", &self.session_params),
            ("lora_path", &self.lora_path),
            ("custom_logit_processor", &self.custom_logit_processor),
            ("image_data", &self.image_data),
            ("video_data", &self.video_data),
            ("audio_data", &self.audio_data),
            ("mm_hashes", &self.mm_hashes),
        ] {
            if value.is_some() {
                return Err(format!(
                    "{name} is not supported by the text-only Rust frontend"
                ));
            }
        }
        Ok(())
    }

    fn response_id(&self, prefix: &str) -> String {
        match self.rid.as_ref() {
            Some(OneOrMany::One(rid)) => rid.clone(),
            Some(OneOrMany::Many(rids)) => rids
                .first()
                .cloned()
                .unwrap_or_else(|| generated_response_id(prefix)),
            None => generated_response_id(prefix),
        }
    }

    fn expand(
        self,
        model: String,
        prompt_count: usize,
        choice_count: usize,
        response_id: &str,
    ) -> Result<Vec<ExpandedRequestContext>, String> {
        let list_rids = matches!(&self.rid, Some(OneOrMany::Many(_)));
        let rids = expand_per_prompt("rid", self.rid, prompt_count)?;
        if list_rids {
            let mut seen = std::collections::HashSet::new();
            for rid in rids.iter().flatten() {
                if !seen.insert(rid) {
                    return Err(format!("duplicate request ID in rid: {rid}"));
                }
            }
        }
        let cache_salts = expand_per_prompt("cache_salt", self.cache_salt, prompt_count)?;
        let extra_keys = expand_per_prompt("extra_key", self.extra_key, prompt_count)?;
        let bootstrap_hosts =
            expand_per_prompt("bootstrap_host", self.bootstrap_host, prompt_count)?;
        let bootstrap_ports =
            expand_per_prompt("bootstrap_port", self.bootstrap_port, prompt_count)?;
        let bootstrap_rooms = match self.bootstrap_room {
            Some(OneOrMany::One(base)) => (0..prompt_count)
                .map(|prompt_index| {
                    let offset = i64::try_from(prompt_index)
                        .map_err(|_| "bootstrap_room prompt index exceeds i64".to_owned())?;
                    base.checked_add(offset)
                        .map(Some)
                        .ok_or_else(|| "bootstrap_room overflows i64".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?,
            value => expand_per_prompt("bootstrap_room", value, prompt_count)?,
        };
        let routed_dp_rank = self.routed_dp_rank.or(self.data_parallel_rank);
        let total = prompt_count
            .checked_mul(choice_count)
            .ok_or_else(|| "prompt count times n overflows usize".to_owned())?;
        let mut contexts = Vec::with_capacity(total);
        for prompt_index in 0..prompt_count {
            for sample_index in 0..choice_count {
                let index = prompt_index * choice_count + sample_index;
                let request_id = match (&rids[prompt_index], list_rids) {
                    (Some(rid), true) if choice_count == 1 => rid.clone(),
                    (Some(rid), true) => format!("{rid}-{sample_index}"),
                    _ => format!("{response_id}-{index}"),
                };
                contexts.push(ExpandedRequestContext {
                    request_id,
                    metadata: GenerateRequestMetadata {
                        model: Some(model.clone()),
                        cache_salt: cache_salts[prompt_index]
                            .clone()
                            .filter(|value| !value.is_empty()),
                        extra_key: extra_keys[prompt_index]
                            .clone()
                            .filter(|value| !value.is_empty()),
                        priority: self.priority,
                        bootstrap_host: bootstrap_hosts[prompt_index].clone(),
                        bootstrap_port: bootstrap_ports[prompt_index].flatten(),
                        bootstrap_room: bootstrap_rooms[prompt_index],
                        routed_dp_rank,
                        disagg_prefill_dp_rank: self.disagg_prefill_dp_rank,
                    },
                });
            }
        }
        Ok(contexts)
    }
}

fn expand_per_prompt<T: Clone>(
    name: &str,
    value: Option<OneOrMany<T>>,
    prompt_count: usize,
) -> Result<Vec<Option<T>>, String> {
    match value {
        None => Ok(vec![None; prompt_count]),
        Some(OneOrMany::One(value)) => Ok(vec![Some(value); prompt_count]),
        Some(OneOrMany::Many(values)) if values.len() == prompt_count => {
            Ok(values.into_iter().map(Some).collect())
        }
        Some(OneOrMany::Many(values)) => Err(format!(
            "the length of {name} must equal the prompt batch size ({prompt_count}), got {}",
            values.len()
        )),
    }
}

fn generated_response_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
}

/// Lower the OpenAI Chat wire type into the structured internal chat request.
/// Chat template rendering and tool constraints deliberately happen later in
/// `ChatPreprocessor`, where every transport shares them.
pub(crate) fn lower_chat_request(
    config: &RendererConfig,
    mut request: ChatCompletionRequest,
) -> Result<(String, ChatRequest), RendererError> {
    normalize_reasoning_inputs(
        &mut request.reasoning_effort,
        request.reasoning.take(),
        &mut request.chat_template_kwargs,
    )?;
    // Accepted OpenAI metadata fields do not affect SGLang generation.
    let _ = (&request.store, &request.metadata, &request.user);
    reject_unsupported_fields(&request.unsupported_fields)?;
    request.extensions.validate()?;
    validate_chat_request(config, &request)?;
    let response_id = request.extensions.response_id("chatcmpl");
    let metadata = request
        .extensions
        .clone()
        .expand(request.model.clone(), 1, 1, &response_id)?
        .pop()
        .expect("one chat prompt produces one metadata context")
        .metadata;
    let mut sampling_params = chat_sampling_params(&request, &config.default_sampling_params)?;
    request.sampling_overrides.apply(&mut sampling_params);
    Ok((
        response_id.clone(),
        ChatRequest {
            rid: response_id,
            model: request.model,
            messages: request.messages,
            tools: request.tools,
            tool_choice: request.tool_choice,
            response_format: request.response_format,
            reasoning_effort: request.reasoning_effort,
            continue_final_message: request.continue_final_message,
            chat_template_args: request.chat_template_kwargs,
            sampling_params,
            choice_count: request.n.unwrap_or(1) as usize,
            stream: request.stream.unwrap_or(false),
            return_logprob: request.logprobs.unwrap_or(false),
            top_logprobs_num: request.top_logprobs.unwrap_or(0) as i64,
            parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
            metadata,
        },
    ))
}

pub(crate) fn normalize_reasoning_inputs(
    reasoning_effort: &mut Option<ReasoningEffort>,
    reasoning: Option<Value>,
    chat_template_kwargs: &mut Option<HashMap<String, Value>>,
) -> Result<(), RendererError> {
    let mut thinking = None;
    if let Some(Value::Object(reasoning)) = reasoning {
        let nested_effort = reasoning
            .get("effort")
            .filter(|value| !value.is_null())
            .or_else(|| {
                reasoning
                    .get("reasoning_effort")
                    .filter(|value| !value.is_null())
            });
        if let Some(nested_effort) = nested_effort {
            *reasoning_effort = Some(
                serde_json::from_value(nested_effort.clone())
                    .map_err(|error| format!("invalid reasoning effort: {error}"))?,
            );
        }

        let enabled = reasoning
            .get("enabled")
            .filter(|value| !value.is_null())
            .or_else(|| reasoning.get("enable"));
        if enabled.is_some_and(json_truthy) {
            thinking = Some(true);
        }
    }

    if let Some(effort) = reasoning_effort.as_ref() {
        thinking = Some(!effort.disables_thinking());
    }
    if let Some(thinking) = thinking {
        let args = chat_template_kwargs.get_or_insert_with(HashMap::new);
        args.entry("thinking".into()).or_insert(thinking.into());
        args.entry("enable_thinking".into())
            .or_insert(thinking.into());
    }
    Ok(())
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "y" | "on"
        ),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn validate_chat_request(
    config: &RendererConfig,
    request: &ChatCompletionRequest,
) -> Result<(), RendererError> {
    if request.model != config.served_model_name {
        return Err(format!("The model `{}` does not exist", request.model).into());
    }
    if request.n == Some(0) {
        return Err("n must be at least 1".into());
    }
    if request.extensions.return_meta_info == Some(true) {
        return Err("return_meta_info=true is not supported by the renderer".into());
    }
    #[allow(deprecated)]
    let max_tokens = request.max_completion_tokens.or(request.max_tokens);
    if max_tokens == Some(0) {
        return Err("max_completion_tokens must be positive".into());
    }
    if request
        .modalities
        .as_ref()
        .is_some_and(|modalities| modalities.contains(&ResponseModality::Audio))
        || request.audio.is_some()
        || request.prediction.is_some()
        || request.web_search_options.is_some()
        || request.mm_processor_kwargs.is_some()
    {
        return Err(
            "audio, prediction, web search, and multimodal inputs are not supported".into(),
        );
    }
    #[allow(deprecated)]
    if request.function_call.is_some() || request.functions.is_some() {
        return Err(
            "deprecated function_call/functions are not supported; use tools and tool_choice"
                .into(),
        );
    }
    Ok(())
}

#[allow(deprecated)]
pub fn chat_sampling_params(
    request: &ChatCompletionRequest,
    model_defaults: &SamplingDefaults,
) -> Result<SamplingParams, String> {
    let defaults = sampling_params_with_model_defaults(model_defaults);
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
            .unwrap_or(defaults.temperature),
        top_p: request.top_p.map(f64::from).unwrap_or(defaults.top_p),
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0) as f64,
        presence_penalty: request.presence_penalty.unwrap_or(0.0) as f64,
        n: 1,
        logit_bias: (!logit_bias.is_empty()).then_some(logit_bias),
        sampling_seed: request.seed,
        json_schema,
        ..defaults
    })
}

fn sampling_params_with_model_defaults(model_defaults: &SamplingDefaults) -> SamplingParams {
    let terminals = SamplingParams::default();
    SamplingParams {
        temperature: model_defaults.temperature.unwrap_or(terminals.temperature),
        top_p: model_defaults.top_p.unwrap_or(terminals.top_p),
        top_k: model_defaults.top_k.unwrap_or(terminals.top_k),
        min_p: model_defaults.min_p.unwrap_or(terminals.min_p),
        repetition_penalty: model_defaults
            .repetition_penalty
            .unwrap_or(terminals.repetition_penalty),
        ..terminals
    }
}

/// Lower a textual OpenAI completion into text-only internal requests.
pub(crate) fn lower_text_completion_request(
    config: &RendererConfig,
    request: &CompletionRequest,
) -> Result<(String, Vec<TextRequestGroup>), RendererError> {
    // Accepted OpenAI request attribution does not affect generation.
    let _ = &request.user;
    reject_unsupported_fields(&request.unsupported_fields)?;
    request.extensions.validate()?;
    let prompts = text_completion_prompts(&request.prompt)?;
    let prompt_count = prompts.len();
    let (mut sampling, n, _) = completion_lowering_context(config, request, prompt_count)?;
    request.sampling_overrides.clone().apply(&mut sampling);
    let response_id = request.extensions.response_id("cmpl");
    let mut contexts = request
        .extensions
        .clone()
        .expand(request.model.clone(), prompt_count, n, &response_id)?
        .into_iter();
    let mut requests = Vec::with_capacity(prompt_count);
    for prompt in prompts {
        let mut choices = Vec::with_capacity(n);
        for _ in 0..n {
            let context = contexts
                .next()
                .expect("metadata expansion matches completion choice count");
            choices.push(GenerateRequestIdentity {
                rid: context.request_id,
                metadata: context.metadata,
            });
        }
        requests.push(TextRequestGroup {
            prompt: dynamo_renderer::RenderedPrompt::text(prompt),
            add_special_tokens: true,
            options: completion_generation_options(request, sampling.clone()),
            requests: choices,
        });
    }
    Ok((response_id, requests))
}

/// Lower a pre-tokenized OpenAI completion directly into token-ID requests.
pub(crate) fn lower_token_ids_completion_request(
    config: &RendererConfig,
    request: &CompletionRequest,
) -> Result<(String, Vec<TokenIdsRequest>), RendererError> {
    // Accepted OpenAI request attribution does not affect generation.
    let _ = &request.user;
    reject_unsupported_fields(&request.unsupported_fields)?;
    request.extensions.validate()?;
    let prompts = token_ids_completion_prompts(&request.prompt)?;
    let prompt_count = prompts.len();
    let (mut sampling, n, choice_count) =
        completion_lowering_context(config, request, prompt_count)?;
    request.sampling_overrides.clone().apply(&mut sampling);
    let response_id = request.extensions.response_id("cmpl");
    let mut contexts = request
        .extensions
        .clone()
        .expand(request.model.clone(), prompt_count, n, &response_id)?
        .into_iter();
    let mut requests = Vec::with_capacity(choice_count);
    for input_ids in prompts {
        for _ in 0..n {
            let context = contexts
                .next()
                .expect("metadata expansion matches completion choice count");
            requests.push(
                TokenIdsRequest::new(
                    context.request_id,
                    input_ids.clone(),
                    completion_generation_options(request, sampling.clone()),
                )
                .with_metadata(context.metadata),
            );
        }
    }
    Ok((response_id, requests))
}

fn completion_lowering_context(
    config: &RendererConfig,
    request: &CompletionRequest,
    prompt_count: usize,
) -> Result<(SamplingParams, usize, usize), RendererError> {
    if request.model != config.served_model_name {
        return Err(format!("The model `{}` does not exist", request.model).into());
    }
    if request.prompt_embeds.is_some() {
        return Err("prompt_embeds is not supported by the Rust frontend".into());
    }
    if request.suffix.is_some() {
        return Err("suffix is not supported by this model".into());
    }
    if request.best_of.is_some_and(|best_of| best_of != 1) {
        return Err("best_of values greater than 1 are not supported".into());
    }
    if request.max_tokens == Some(0) {
        return Err("max_tokens must be positive".into());
    }
    if request.n == Some(0) {
        return Err("n must be at least 1".into());
    }
    let sampling = completion_sampling_params(request, &config.default_sampling_params)?;
    let n = request.n.unwrap_or(1) as usize;
    let choice_count = prompt_count
        .checked_mul(n)
        .filter(|&count| count <= MAX_OPENAI_CHOICES)
        .ok_or_else(|| {
            format!("prompt count times n exceeds the maximum of {MAX_OPENAI_CHOICES}")
        })?;
    Ok((sampling, n, choice_count))
}

fn completion_generation_options(
    request: &CompletionRequest,
    sampling_params: SamplingParams,
) -> GenerationOptions {
    GenerationOptions {
        sampling_params,
        stream: request.stream.unwrap_or(false),
        return_logprob: request.logprobs.is_some(),
        logprob_start_len: if request.echo.unwrap_or(false) && request.logprobs.is_some() {
            0
        } else {
            -1
        },
        top_logprobs_num: request.logprobs.unwrap_or(0) as i64,
        return_text_in_logprobs: request.logprobs.map(|_| true),
        ..Default::default()
    }
}

pub fn text_completion_prompts(prompt: &Prompt) -> Result<Vec<String>, String> {
    match prompt {
        Prompt::String(text) => {
            if text.is_empty() {
                return Err("Prompt cannot be empty".into());
            }
            Ok(vec![text.clone()])
        }
        Prompt::StringArray(texts) => {
            if texts.is_empty() || texts.iter().any(String::is_empty) {
                return Err("Prompt cannot be empty".into());
            }
            Ok(texts.clone())
        }
        Prompt::IntegerArray(_) | Prompt::ArrayOfIntegerArray(_) => {
            Err("text completion lowerer requires a text prompt".into())
        }
    }
}

pub fn token_ids_completion_prompts(prompt: &Prompt) -> Result<Vec<TokenIds>, String> {
    match prompt {
        Prompt::IntegerArray(ids) => Ok(vec![token_prompt_ids(ids)?]),
        Prompt::ArrayOfIntegerArray(prompts) => {
            if prompts.is_empty() {
                return Err("Prompt cannot be empty".into());
            }
            prompts.iter().map(|ids| token_prompt_ids(ids)).collect()
        }
        Prompt::String(_) | Prompt::StringArray(_) => {
            Err("token-ID completion lowerer requires a token-ID prompt".into())
        }
    }
}

fn token_prompt_ids(ids: &[u32]) -> Result<TokenIds, String> {
    if ids.is_empty() {
        return Err("Prompt cannot be empty".into());
    }
    let input_ids = ids
        .iter()
        .map(|&id| i32::try_from(id).map_err(|_| format!("Token ID {id} is out of range")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(input_ids)
}

pub fn completion_sampling_params(
    request: &CompletionRequest,
    model_defaults: &SamplingDefaults,
) -> Result<SamplingParams, String> {
    let defaults = sampling_params_with_model_defaults(model_defaults);
    let mut stop = None;
    let mut stop_token_ids = None;
    match request.stop.as_ref() {
        Some(Stop::String(value)) => stop = Some(OneOrMany::One(value.clone())),
        Some(Stop::StringArray(values)) => stop = Some(OneOrMany::Many(values.clone())),
        Some(Stop::TokenIdArray(values)) => {
            stop_token_ids
                .get_or_insert_with(Vec::new)
                .extend(values.iter().map(|&id| id as i64));
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

    Ok(SamplingParams {
        max_new_tokens: Some(request.max_tokens.unwrap_or(16) as i64),
        stop,
        stop_token_ids,
        temperature: request
            .temperature
            .map(f64::from)
            .unwrap_or(defaults.temperature),
        top_p: request.top_p.map(f64::from).unwrap_or(defaults.top_p),
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0) as f64,
        presence_penalty: request.presence_penalty.unwrap_or(0.0) as f64,
        // OpenAI `n` is implemented by fan-out: every native request has one
        // output, avoiding the native path's intentional `n > 1` rejection.
        n: 1,
        logit_bias: (!logit_bias.is_empty()).then_some(logit_bias),
        sampling_seed: request.seed,
        ..defaults
    })
}
