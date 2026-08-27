//! OpenAI request lowering owned by the engine-free renderer.

use std::collections::BTreeMap;

use dynamo_parsers::parsers::get_tool_parser_map;
use dynamo_parsers::{
    StructuralTagBuilder, StructuralTagSchemaMode, ToolCallFormatBuildContext,
    ToolChoice as DynamoToolChoice, ToolDefinition, TriggeredTagsConfig,
};
use dynamo_protocols::types::{
    ChatCompletionToolChoiceOption, CreateChatCompletionRequest, CreateCompletionRequest, Prompt,
    ResponseFormat, Stop,
};

use crate::{
    ChatFormatter, OneOrMany, RendererConfig, RendererError, RendererRequest, SamplingDefaults,
    SamplingParams, TokenIds,
};

const MAX_OPENAI_CHOICES: usize = 4096;

fn contains_media(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Array(values) => values.iter().any(contains_media),
        serde_json::Value::Object(object) => {
            object.keys().any(|key| {
                matches!(
                    key.as_str(),
                    "image_url" | "video_url" | "input_audio" | "audio_url" | "file"
                )
            }) || object.values().any(contains_media)
        }
        _ => false,
    }
}

/// Canonicalize a tool-call parser name onto the dynamo-parsers registry keys.
///
/// SGLang canonicalizes these legacy CLI names in the opposite direction from
/// the current Dynamo parser registry.
pub fn dynamo_parser_name(parser: &str) -> &str {
    match parser {
        "llama3" => "llama3_json",
        "qwen" => "qwen25",
        "glm" | "glm45" => "glm47",
        other => other,
    }
}

/// Map the OpenAI wire `tool_choice` onto the Dynamo choice. A missing/auto
/// choice reads as `Auto`.
pub fn dynamo_tool_choice(choice: &Option<ChatCompletionToolChoiceOption>) -> DynamoToolChoice {
    match choice {
        Some(ChatCompletionToolChoiceOption::None) => DynamoToolChoice::None,
        Some(ChatCompletionToolChoiceOption::Required) => DynamoToolChoice::Required,
        Some(ChatCompletionToolChoiceOption::Named(choice)) => {
            DynamoToolChoice::Named(choice.function.name.clone())
        }
        Some(ChatCompletionToolChoiceOption::Auto) | None => DynamoToolChoice::Auto,
    }
}

/// Validate `tool_choice` against `tools`, then — when a tool-call `parser`
/// is configured — turn it into a sampling constraint, mirroring Python's
/// `serving_chat` logic. Validation runs even without a parser, so an invalid
/// choice (required/named with nothing to select) is rejected before
/// submission in every mode.
///
/// Prefers a structural-tag constraint: the parser's own registered builder,
/// or — for llama3 with strict tools under `auto` — a triggered-tag builder
/// so the model emits calls in the exact `<|python_tag|>` format. Otherwise
/// `required`/`named` choices fall back to a JSON-schema array constraining
/// the output to `{"name", "parameters"}` objects (`maxItems: 1` when
/// `parallel_tool_calls` is false).
pub fn apply_tool_constraint(
    sampling: &mut SamplingParams,
    parser: Option<&str>,
    tool_choice: &DynamoToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
) -> Result<(), String> {
    if *tool_choice == DynamoToolChoice::None {
        return Ok(());
    }
    if *tool_choice == DynamoToolChoice::Required && tools.is_empty() {
        return Err("tool_choice is \"required\" but tools is empty".into());
    }
    if let DynamoToolChoice::Named(name) = tool_choice
        && !tools.iter().any(|tool| &tool.name == name)
    {
        return Err(format!(
            "tool named \"{name}\" in tool_choice is not present in tools"
        ));
    }

    let Some(parser) = parser else {
        return Ok(()); // validation only
    };
    let parser = dynamo_parser_name(parser);
    let config = get_tool_parser_map()
        .get(parser)
        .ok_or_else(|| format!("tool-call parser `{parser}` is not supported by Dynamo"))?;
    let builder = config.structural_tag_builder.clone().or_else(|| {
        (parser == "llama3_json"
            && *tool_choice == DynamoToolChoice::Auto
            && tools.iter().any(|tool| tool.strict.unwrap_or(false)))
        .then(|| {
            StructuralTagBuilder::TriggeredTags(TriggeredTagsConfig {
                begin_template: r#"<|python_tag|>{"name":"{name}", "arguments":"#.to_string(),
                end_template: "}".to_string(),
                triggers: vec!["<|python_tag|>".to_string()],
                content_style: Default::default(),
                tool_call_ban_tokens: Vec::new(),
                reasoning_end: None,
            })
        })
    });
    if let Some(builder) = builder
        && let Some(tag) = builder
            .build_tool_call_format(&ToolCallFormatBuildContext {
                tool_choice,
                tools,
                parallel_tool_calls,
                schema_mode: StructuralTagSchemaMode::Auto,
                starts_in_reasoning: false,
            })
            .map_err(|error| error.to_string())?
    {
        sampling.structural_tag = Some(tag.to_string());
        return Ok(());
    }

    if matches!(
        tool_choice,
        DynamoToolChoice::Required | DynamoToolChoice::Named(_)
    ) {
        let selected = match tool_choice {
            DynamoToolChoice::Named(name) => tools
                .iter()
                .filter(|tool| tool.name == *name)
                .collect::<Vec<_>>(),
            _ => tools.iter().collect(),
        };
        let schemas = selected
            .into_iter()
            .map(|tool| {
                serde_json::json!({
                    "properties": {
                        "name": {"type": "string", "enum": [tool.name]},
                        "parameters": tool.parameters.clone().unwrap_or_else(|| {
                            serde_json::json!({"type": "object", "properties": {}})
                        }),
                    },
                    "required": ["name", "parameters"],
                })
            })
            .collect::<Vec<_>>();
        let items = if schemas.len() == 1 {
            schemas.into_iter().next().expect("one schema")
        } else {
            serde_json::json!({"type": "object", "anyOf": schemas})
        };
        let mut schema = serde_json::json!({
            "type": "array",
            "minItems": 1,
            "items": items,
        });
        if parallel_tool_calls == Some(false) {
            schema["maxItems"] = serde_json::json!(1);
        }
        sampling.json_schema = Some(schema.to_string());
    }
    Ok(())
}

pub struct LoweredChatRequests {
    pub requests: Vec<RendererRequest>,
    pub parser: Option<String>,
    pub tools: Option<Vec<ToolDefinition>>,
}

pub async fn lower_chat_requests(
    config: &RendererConfig,
    chat_formatter: Option<ChatFormatter>,
    request: &mut CreateChatCompletionRequest,
    response_id: &str,
) -> Result<LoweredChatRequests, RendererError> {
    if request.model != config.served_model_name {
        return Err(format!("The model `{}` does not exist", request.model).into());
    }
    if request.messages.is_empty() {
        return Err("messages cannot be empty".into());
    }
    if serde_json::to_value(&request.messages).is_ok_and(|messages| contains_media(&messages)) {
        return Err("image, audio, video, and file message content is not supported".into());
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

    let tool_choice = dynamo_tool_choice(&request.tool_choice);
    let parser = resolve_chat_parser(config, request, &tool_choice)?;
    let tools = chat_tool_definitions(request);
    let prompt = prepare_chat_request(chat_formatter, request).await?;
    let sampling = chat_sampling(
        request,
        ChatSamplingDefaults::CHAT,
        parser.as_deref(),
        &tool_choice,
        tools.as_deref().unwrap_or_default(),
        request.parallel_tool_calls,
        config,
    )?;

    let n = request.n.unwrap_or(1) as usize;
    let stream = request.stream.unwrap_or(false);
    let want_logprobs = request.logprobs.unwrap_or(false);
    let mut requests = Vec::with_capacity(n);
    let mut prompt = Some(prompt);
    for index in 0..n {
        let choice_prompt = if index + 1 == n {
            prompt.take().expect("last chat choice owns the prompt")
        } else {
            prompt
                .as_ref()
                .expect("chat prompt exists until the last choice")
                .clone()
        };
        requests.push(RendererRequest {
            rid: format!("{response_id}-{index}"),
            text: Some(choice_prompt),
            // Rendered templates own their special tokens — the pool must not
            // add another BOS/EOS (Python's `add_special_tokens=False`).
            skip_special_tokens: true,
            sampling_params: sampling.clone(),
            stream,
            return_logprob: want_logprobs,
            logprob_start_len: -1,
            top_logprobs_num: request.top_logprobs.unwrap_or(0) as i64,
            return_text_in_logprobs: want_logprobs.then_some(true),
            ..Default::default()
        });
    }
    Ok(LoweredChatRequests {
        requests,
        parser,
        tools,
    })
}

fn resolve_chat_parser(
    config: &RendererConfig,
    request: &CreateChatCompletionRequest,
    tool_choice: &DynamoToolChoice,
) -> Result<Option<String>, &'static str> {
    let tools_enabled = request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
        && *tool_choice != DynamoToolChoice::None;
    let parser = tools_enabled
        .then(|| config.tool_call_parser.clone())
        .flatten();
    if tools_enabled && parser.is_none() {
        return Err("tool calls require --tool-call-parser");
    }
    Ok(parser)
}

fn chat_tool_definitions(request: &CreateChatCompletionRequest) -> Option<Vec<ToolDefinition>> {
    request.tools.as_ref().map(|tools| {
        tools
            .iter()
            .map(|tool| ToolDefinition {
                name: tool.function.name.clone(),
                parameters: tool.function.parameters.clone(),
                strict: tool.function.strict,
            })
            .collect::<Vec<_>>()
    })
}
/// Render the chat template for an OpenAI request, mapping a missing
/// formatter or a render failure to the standard 400. The rendered prompt is
/// submitted as text — the tokenizer pool encodes it (with
/// `skip_special_tokens`, since the template owns its special tokens).
pub async fn prepare_chat_request(
    chat_formatter: Option<ChatFormatter>,
    request: &mut CreateChatCompletionRequest,
) -> Result<String, RendererError> {
    let Some(formatter) = chat_formatter else {
        return Err("this model has no usable chat template".into());
    };
    // Template stops first, then the request's own — Python
    // `_apply_conversation_template` (`conv.stop_str` + `request.stop`). A
    // token-id stop cannot be merged into the string list (Python has no such
    // field), so it is kept alone.
    merge_template_stops(request, &formatter);
    let prompt = formatter
        .render(request)
        .map_err(|error| format!("chat template render failed: {error}"))?;
    Ok(prompt)
}

/// Full sampling resolution for an OpenAI request, mirroring the Python
/// handler: endpoint defaults → tool-choice validation + constraint → clamp.
/// The tool-choice checks run regardless of whether a parser is configured
/// (see `apply_tool_constraint`).
pub fn chat_sampling(
    request: &CreateChatCompletionRequest,
    defaults: ChatSamplingDefaults,
    parser: Option<&str>,
    tool_choice: &DynamoToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
    config: &RendererConfig,
) -> Result<SamplingParams, String> {
    let mut sampling = chat_sampling_params(
        request,
        &defaults.with_model_defaults(&config.default_sampling_params),
    )?;
    apply_tool_constraint(
        &mut sampling,
        parser,
        tool_choice,
        tools,
        parallel_tool_calls,
    )?;
    sampling
        .normalize(config.skip_tokenizer_init, config.vocab_size)
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
/// Validate and lower one OpenAI completion request into the ordered native
/// requests consumed by inference or standalone rendering.
pub fn lower_completion_requests(
    config: &RendererConfig,
    request: &CreateCompletionRequest,
    response_id: &str,
) -> Result<Vec<RendererRequest>, RendererError> {
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
    let mut sampling = completion_sampling_params(request)?;
    sampling
        .normalize(config.skip_tokenizer_init, config.vocab_size)
        .map_err(|error| error.to_string())?;
    let n = request.n.unwrap_or(1) as usize;
    let choice_count = prompt_specs
        .len()
        .checked_mul(n)
        .filter(|&count| count <= MAX_OPENAI_CHOICES)
        .ok_or_else(|| {
            format!("prompt count times n exceeds the maximum of {MAX_OPENAI_CHOICES}")
        })?;

    let mut requests = Vec::with_capacity(choice_count);
    for (prompt_index, prompt) in prompt_specs.into_iter().enumerate() {
        let (text, input_ids) = match prompt {
            PromptSpec::Text(text) => (Some(text), None),
            PromptSpec::TokenIds(ids) => (None, Some(ids)),
        };
        for sample_index in 0..n {
            let index = prompt_index * n + sample_index;
            requests.push(RendererRequest {
                rid: format!("{response_id}-{index}"),
                text: text.clone(),
                input_ids: input_ids.clone(),
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
            });
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
