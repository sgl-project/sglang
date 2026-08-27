//! OpenAI wire types lowered into renderer-owned requests.

use std::collections::BTreeMap;

use dynamo_protocols::types::{
    CreateChatCompletionRequest, CreateCompletionRequest, Prompt, ResponseFormat, Stop,
};

use crate::{
    ChatRequest, GenerateRequestMetadata, GenerationOptions, OneOrMany, RendererConfig,
    RendererError, SamplingDefaults, SamplingParams, TextRequest, TokenIds,
};

const MAX_OPENAI_CHOICES: usize = 4096;

/// Lower the OpenAI Chat wire type into the structured internal chat request.
/// Chat template rendering and tool constraints deliberately happen later in
/// `ChatPreprocessor`, where non-HTTP protocol adapters can reuse them.
pub(crate) fn lower_chat_request(
    config: &RendererConfig,
    request: CreateChatCompletionRequest,
    response_id: &str,
) -> Result<ChatRequest, RendererError> {
    lower_chat_request_with_template_args(config, request, response_id, None, false)
}

pub(crate) fn lower_chat_request_with_template_args(
    config: &RendererConfig,
    request: CreateChatCompletionRequest,
    response_id: &str,
    chat_template_args: Option<std::collections::HashMap<String, serde_json::Value>>,
    continue_final_message: bool,
) -> Result<ChatRequest, RendererError> {
    validate_chat_request(config, &request)?;
    let sampling_params = chat_sampling_params(
        &request,
        &ChatSamplingDefaults::CHAT.with_model_defaults(&config.default_sampling_params),
    )?;
    Ok(ChatRequest {
        rid: response_id.to_owned(),
        model: request.model.clone(),
        messages: request.messages,
        tools: request.tools,
        tool_choice: request.tool_choice,
        response_format: request.response_format,
        reasoning_effort: request.reasoning_effort,
        continue_final_message,
        chat_template_args,
        sampling_params,
        choice_count: request.n.unwrap_or(1) as usize,
        stream: request.stream.unwrap_or(false),
        return_logprob: request.logprobs.unwrap_or(false),
        top_logprobs_num: request.top_logprobs.unwrap_or(0) as i64,
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        metadata: GenerateRequestMetadata {
            model: Some(request.model),
            ..Default::default()
        },
    })
}

fn validate_chat_request(
    config: &RendererConfig,
    request: &CreateChatCompletionRequest,
) -> Result<(), RendererError> {
    if request.model != config.served_model_name {
        return Err(format!("The model `{}` does not exist", request.model).into());
    }
    if request.n == Some(0) {
        return Err("n must be at least 1".into());
    }
    #[allow(deprecated)]
    let max_tokens = request.max_completion_tokens.or(request.max_tokens);
    if max_tokens == Some(0) {
        return Err("max_completion_tokens must be positive".into());
    }
    if request.modalities.as_ref().is_some_and(|modalities| {
        serde_json::to_value(modalities).is_ok_and(|value| value.to_string().contains("\"audio\""))
    }) || request.audio.is_some()
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

/// Where an omitted `temperature` / `top_p` gets its value. Mirrors Python's
/// `to_sampling_params` priority: user value > model generation_config (when
/// `--sampling-defaults model`) > OpenAI terminal default
/// (`_DEFAULT_SAMPLING_PARAMS`: chat uses 1.0/1.0).
pub struct ChatSamplingDefaults {
    /// Model defaults; `None` when the model config doesn't set them or when
    /// `--sampling-defaults openai` (the Python dump is then empty).
    temperature: Option<f64>,
    top_p: Option<f64>,
    /// OpenAI terminal defaults for chat completions.
    fallback_temperature: f64,
    fallback_top_p: f64,
}

impl ChatSamplingDefaults {
    /// `protocol.py` chat `_DEFAULT_SAMPLING_PARAMS`: temperature 1.0, top_p 1.0.
    pub const CHAT: ChatSamplingDefaults = ChatSamplingDefaults {
        temperature: None,
        top_p: None,
        fallback_temperature: 1.0,
        fallback_top_p: 1.0,
    };
    /// The resolved model defaults (empty in `--sampling-defaults openai`
    /// mode), which slot between the user's values and the OpenAI terminals.
    pub fn with_model_defaults(mut self, model: &SamplingDefaults) -> ChatSamplingDefaults {
        self.temperature = model.temperature;
        self.top_p = model.top_p;
        self
    }
}

#[allow(deprecated)]
pub fn chat_sampling_params(
    request: &CreateChatCompletionRequest,
    defaults: &ChatSamplingDefaults,
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

#[derive(Debug, PartialEq, Eq)]
pub enum PromptSpec {
    Text(String),
    TokenIds(TokenIds),
}
/// Validate and process one OpenAI completion request into the ordered
/// model-facing requests consumed by inference or standalone rendering.
pub(crate) fn lower_completion_request(
    config: &RendererConfig,
    request: &CreateCompletionRequest,
    response_id: &str,
) -> Result<Vec<TextRequest>, RendererError> {
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
    let prompt_specs = completion_prompt_specs(&request.prompt)?;
    let sampling = completion_sampling_params(request)?;
    let n = request.n.unwrap_or(1) as usize;
    let choice_count = prompt_specs
        .len()
        .checked_mul(n)
        .filter(|&count| count <= MAX_OPENAI_CHOICES)
        .ok_or_else(|| {
            format!("prompt count times n exceeds the maximum of {MAX_OPENAI_CHOICES}")
        })?;
    let metadata = GenerateRequestMetadata {
        model: Some(request.model.clone()),
        ..Default::default()
    };

    let mut requests = Vec::with_capacity(choice_count);
    for (prompt_index, prompt) in prompt_specs.into_iter().enumerate() {
        for sample_index in 0..n {
            let index = prompt_index * n + sample_index;
            let options = GenerationOptions {
                sampling_params: sampling.clone(),
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
            };
            let rid = format!("{response_id}-{index}");
            requests.push(
                match &prompt {
                    PromptSpec::Text(text) => TextRequest::text(rid, text.clone(), true, options),
                    PromptSpec::TokenIds(input_ids) => {
                        TextRequest::token_ids(rid, input_ids.clone(), false, options)
                    }
                }
                .with_metadata(metadata.clone()),
            );
        }
    }
    Ok(requests)
}
pub fn completion_prompt_specs(prompt: &Prompt) -> Result<Vec<PromptSpec>, String> {
    match prompt {
        Prompt::String(text) => {
            if text.is_empty() {
                return Err("Prompt cannot be empty".into());
            }
            Ok(vec![PromptSpec::Text(text.clone())])
        }
        Prompt::StringArray(texts) => {
            if texts.is_empty() || texts.iter().any(String::is_empty) {
                return Err("Prompt cannot be empty".into());
            }
            Ok(texts.iter().cloned().map(PromptSpec::Text).collect())
        }
        Prompt::IntegerArray(ids) => Ok(vec![token_prompt_spec(ids)?]),
        Prompt::ArrayOfIntegerArray(prompts) => {
            if prompts.is_empty() {
                return Err("Prompt cannot be empty".into());
            }
            prompts.iter().map(|ids| token_prompt_spec(ids)).collect()
        }
    }
}

pub fn token_prompt_spec(ids: &[u32]) -> Result<PromptSpec, String> {
    if ids.is_empty() {
        return Err("Prompt cannot be empty".into());
    }
    let input_ids = ids
        .iter()
        .map(|&id| i32::try_from(id).map_err(|_| format!("Token ID {id} is out of range")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PromptSpec::TokenIds(input_ids))
}

pub fn completion_sampling_params(
    request: &CreateCompletionRequest,
) -> Result<SamplingParams, String> {
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
        temperature: request.temperature.unwrap_or(1.0) as f64,
        top_p: request.top_p.unwrap_or(1.0) as f64,
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0) as f64,
        presence_penalty: request.presence_penalty.unwrap_or(0.0) as f64,
        // OpenAI `n` is implemented by fan-out: every native request has one
        // output, avoiding the native path's intentional `n > 1` rejection.
        n: 1,
        logit_bias: (!logit_bias.is_empty()).then_some(logit_bias),
        sampling_seed: request.seed,
        ..Default::default()
    })
}
