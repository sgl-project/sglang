//! Transport-neutral chat preprocessing over a canonical OpenAI-compatible
//! message vocabulary.

use std::collections::HashMap;

use dynamo_parsers::parsers::get_tool_parser_map;
use dynamo_parsers::{
    StructuralTagBuilder, StructuralTagSchemaMode, ToolCallFormatBuildContext,
    ToolChoice as DynamoToolChoice, ToolDefinition, TriggeredTagsConfig,
};
use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage, ChatCompletionTool,
    ChatCompletionToolChoiceOption, ResponseFormat,
};
use dynamo_renderer::{
    OAIChatLikeRequest, RenderedPrompt, RenderedSegment, may_be_fix_tool_schema,
};
use minijinja::Value;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::ChatResponseProcessor;
use crate::{
    ChatFormatter, GenerateRequestMetadata, GenerationOptions, OneOrMany, RendererConfig,
    RendererError, SamplingParams, TextRequest,
};

use super::{GenerateRequestIdentity, TextRequestGroup};

/// SGLang reasoning effort, including Inkling's fine-grained numeric form.
#[derive(Debug, Clone, PartialEq)]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
    Numeric(f64),
}

impl ReasoningEffort {
    pub(crate) const fn disables_thinking(&self) -> bool {
        matches!(self, Self::None)
    }

    const fn name(&self) -> Option<&'static str> {
        match self {
            Self::None => Some("none"),
            Self::Minimal => Some("minimal"),
            Self::Low => Some("low"),
            Self::Medium => Some("medium"),
            Self::High => Some("high"),
            Self::XHigh => Some("xhigh"),
            Self::Max => Some("max"),
            Self::Numeric(_) => None,
        }
    }
}

impl Serialize for ReasoningEffort {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Numeric(value) => serializer.serialize_f64(*value),
            _ => serializer.serialize_str(self.name().expect("named reasoning effort")),
        }
    }
}

impl<'de> Deserialize<'de> for ReasoningEffort {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(value) => {
                let effort = match value.as_str() {
                    "none" => Some(Self::None),
                    "minimal" => Some(Self::Minimal),
                    "low" => Some(Self::Low),
                    "medium" => Some(Self::Medium),
                    "high" => Some(Self::High),
                    "xhigh" => Some(Self::XHigh),
                    "max" => Some(Self::Max),
                    _ => None,
                };
                if let Some(effort) = effort {
                    return Ok(effort);
                }
                let numeric = value.parse::<f64>().map_err(|_| {
                    serde::de::Error::custom(format!("invalid reasoning effort: {value:?}"))
                })?;
                numeric_reasoning_effort(numeric).map_err(serde::de::Error::custom)
            }
            serde_json::Value::Number(value) => {
                let numeric = value.as_f64().ok_or_else(|| {
                    serde::de::Error::custom("reasoning_effort must be a finite number")
                })?;
                numeric_reasoning_effort(numeric).map_err(serde::de::Error::custom)
            }
            serde_json::Value::Bool(_) => Err(serde::de::Error::custom(
                "reasoning_effort must not be a boolean",
            )),
            _ => Err(serde::de::Error::custom(
                "reasoning_effort must be a string or number",
            )),
        }
    }
}

fn numeric_reasoning_effort(value: f64) -> Result<ReasoningEffort, String> {
    if !value.is_finite() || !(0.0..=0.99).contains(&value) {
        return Err(format!(
            "reasoning_effort must be a finite number in [0.0, 0.99], got {value}"
        ));
    }
    Ok(ReasoningEffort::Numeric(value))
}

/// Renderer-owned normalized chat state.
///
/// Message and tool values remain Dynamo OpenAI protocol types until
/// [`ChatPreprocessor`] applies the model chat template and lowers the request
/// to the same [`TextRequest`] consumed by text completions.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub rid: String,
    pub model: String,
    pub messages: Vec<ChatCompletionRequestMessage>,
    pub tools: Option<Vec<ChatCompletionTool>>,
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    pub response_format: Option<ResponseFormat>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub continue_final_message: bool,
    pub chat_template_args: Option<HashMap<String, serde_json::Value>>,
    pub sampling_params: SamplingParams,
    pub choice_count: usize,
    pub stream: bool,
    pub return_logprob: bool,
    pub top_logprobs_num: i64,
    pub parallel_tool_calls: bool,
    pub metadata: GenerateRequestMetadata,
}

impl OAIChatLikeRequest for ChatRequest {
    fn model(&self) -> String {
        self.model.clone()
    }

    fn messages(&self) -> Value {
        Value::from_serialize(
            serde_json::to_value(&self.messages).expect("chat messages serialize"),
        )
    }

    fn typed_messages(&self) -> Option<&[ChatCompletionRequestMessage]> {
        Some(&self.messages)
    }

    fn tools(&self) -> Option<Value> {
        self.tools.as_ref().and_then(|tools| {
            may_be_fix_tool_schema(serde_json::to_value(tools).expect("chat tools serialize"))
        })
    }

    fn tool_choice(&self) -> Option<Value> {
        self.tool_choice.as_ref().map(Value::from_serialize)
    }

    fn response_format(&self) -> Option<Value> {
        self.response_format.as_ref().map(Value::from_serialize)
    }

    fn reasoning_effort(&self) -> Option<Value> {
        self.reasoning_effort.as_ref().map(Value::from_serialize)
    }

    fn should_add_generation_prompt(&self) -> bool {
        !self.continue_final_message
    }

    fn chat_template_args(&self) -> Option<&HashMap<String, serde_json::Value>> {
        self.chat_template_args.as_ref()
    }
}

/// Chat-to-text result plus the state needed to interpret generated output.
pub(crate) struct LoweredChat {
    pub text_requests: Vec<TextRequestGroup>,
    pub response_processor: ChatResponseProcessor,
}

struct RenderPreparation {
    require_reasoning: bool,
    reasoning_state: Option<bool>,
    tools_enabled: bool,
}

/// Applies structured chat semantics before the shared text generation path.
pub struct ChatPreprocessor {
    formatter: Option<ChatFormatter>,
    formatter_error: Option<String>,
    tool_call_parser: Option<String>,
    reasoning_parser: Option<String>,
    default_chat_template_kwargs: HashMap<String, serde_json::Value>,
}

impl ChatPreprocessor {
    pub(crate) fn new(config: &RendererConfig, formatter: Option<ChatFormatter>) -> Self {
        Self {
            formatter,
            formatter_error: None,
            tool_call_parser: config.tool_call_parser.clone(),
            reasoning_parser: config.reasoning_parser.clone(),
            default_chat_template_kwargs: config.default_chat_template_kwargs.clone(),
        }
    }

    pub(crate) fn with_formatter_error(mut self, error: Option<String>) -> Self {
        self.formatter_error = error;
        self
    }

    pub fn preprocess(&self, mut request: ChatRequest) -> Result<LoweredChat, RendererError> {
        let preparation = self.prepare_for_render(&mut request)?;
        merge_template_stops(&mut request.sampling_params, self.formatter.as_ref());

        let tool_choice = dynamo_tool_choice(&request.tool_choice);
        let tools = chat_tool_definitions(&request);
        let parser =
            resolve_chat_parser(self.tool_call_parser.as_deref(), preparation.tools_enabled)?;
        if parser.is_some() {
            request.sampling_params.skip_special_tokens = false;
        }
        apply_tool_constraint(
            &mut request.sampling_params,
            parser.as_deref(),
            &tool_choice,
            &tools,
            Some(request.parallel_tool_calls),
        )?;
        let prompt = self.render(&request)?;
        let uses_tool_call_structural_tag = request.sampling_params.structural_tag.is_some();

        let options = GenerationOptions {
            sampling_params: request.sampling_params.clone(),
            require_reasoning: preparation.require_reasoning,
            stream: request.stream,
            return_logprob: request.return_logprob,
            logprob_start_len: -1,
            top_logprobs_num: request.top_logprobs_num,
            return_text_in_logprobs: request.return_logprob.then_some(true),
            ..Default::default()
        };
        let mut choices = Vec::with_capacity(request.choice_count);
        for index in 0..request.choice_count {
            choices.push(GenerateRequestIdentity {
                rid: format!("{}-{index}", request.rid),
                metadata: request.metadata.clone(),
            });
        }
        let text_requests = vec![TextRequestGroup {
            prompt,
            add_special_tokens: false,
            options,
            requests: choices,
        }];

        let response_processor = ChatResponseProcessor::new(
            parser,
            self.reasoning_parser.clone(),
            (!tools.is_empty()).then_some(tools),
            request.tool_choice,
            uses_tool_call_structural_tag,
            request.parallel_tool_calls,
            request.choice_count,
        )
        .with_reasoning_state(preparation.reasoning_state);
        Ok(LoweredChat {
            text_requests,
            response_processor,
        })
    }

    /// Render chat for tokenization without creating generation/output state.
    pub fn lower_to_text(&self, mut request: ChatRequest) -> Result<TextRequest, RendererError> {
        let preparation = self.prepare_for_render(&mut request)?;
        let prompt = self.render(&request)?;
        Ok(TextRequest::rendered(
            request.rid,
            prompt,
            false,
            GenerationOptions {
                sampling_params: request.sampling_params,
                require_reasoning: preparation.require_reasoning,
                ..Default::default()
            },
        )
        .with_metadata(request.metadata))
    }

    fn prepare_for_render(
        &self,
        request: &mut ChatRequest,
    ) -> Result<RenderPreparation, RendererError> {
        validate_chat(request)?;
        self.normalize_template_args(request);
        let tool_choice = dynamo_tool_choice(&request.tool_choice);
        let tools_enabled = request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
            && tool_choice != DynamoToolChoice::None;
        let named_tool_choice = matches!(tool_choice, DynamoToolChoice::Named(_));
        let thinking = self.formatter.as_ref().and_then(|formatter| {
            formatter.resolve_thinking(
                &mut request.chat_template_args,
                tools_enabled,
                named_tool_choice,
            )
        });
        Ok(RenderPreparation {
            require_reasoning: self.reasoning_parser.is_some() && thinking == Some(true),
            reasoning_state: thinking,
            tools_enabled,
        })
    }

    fn normalize_template_args(&self, request: &mut ChatRequest) {
        let request_args = request.chat_template_args.take().unwrap_or_default();
        let mut args = self.default_chat_template_kwargs.clone();
        if let Some(reasoning_effort) = request.reasoning_effort.as_ref() {
            args.insert(
                "reasoning_effort".into(),
                serde_json::to_value(reasoning_effort).expect("reasoning effort must serialize"),
            );
            let thinking = !reasoning_effort.disables_thinking();
            let has_explicit_toggle = request_args.contains_key("thinking")
                || request_args.contains_key("enable_thinking");
            if !has_explicit_toggle {
                args.insert("thinking".into(), thinking.into());
                args.insert("enable_thinking".into(), thinking.into());
            }
        }
        args.extend(request_args);
        request.chat_template_args = (!args.is_empty()).then_some(args);
    }

    fn render(&self, request: &ChatRequest) -> Result<RenderedPrompt, RendererError> {
        let formatter = self.formatter.as_ref().ok_or_else(|| {
            RendererError::from(
                self.formatter_error
                    .clone()
                    .unwrap_or_else(|| "this model has no usable chat template".to_owned()),
            )
        })?;
        let mut request = request.clone();
        let final_message = prepare_continuation(&mut request);
        let template_args = request.chat_template_args.get_or_insert_with(HashMap::new);
        template_args.insert(
            "add_generation_prompt".into(),
            (!request.continue_final_message).into(),
        );
        template_args.insert(
            "continue_final_message".into(),
            request.continue_final_message.into(),
        );
        let prompt = formatter
            .render_prompt(&request)
            .map_err(|error| format!("chat template render failed: {error}"))?;
        match final_message {
            Some(final_message) => truncate_continuation(prompt, &final_message),
            None => Ok(prompt),
        }
    }
}

const CONTINUE_FINAL_MESSAGE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";

fn prepare_continuation(request: &mut ChatRequest) -> Option<String> {
    if !request.continue_final_message {
        return None;
    }
    let Some(ChatCompletionRequestMessage::Assistant(message)) = request.messages.last_mut() else {
        request.continue_final_message = false;
        return None;
    };
    let Some(ChatCompletionRequestAssistantMessageContent::Text(text)) = message.content.as_mut()
    else {
        request.continue_final_message = false;
        return None;
    };
    let original = text.clone();
    text.push_str(CONTINUE_FINAL_MESSAGE_TAG);
    Some(original)
}

fn truncate_continuation(
    prompt: RenderedPrompt,
    final_message: &str,
) -> Result<RenderedPrompt, RendererError> {
    let text = prompt.as_str();
    let tag_location = text
        .rfind(CONTINUE_FINAL_MESSAGE_TAG.trim_end())
        .filter(|_| text.contains(final_message.trim()))
        .ok_or_else(|| {
            RendererError::from(
                "continue_final_message is set but the final message does not appear in the rendered prompt",
            )
        })?;
    let truncate_at = if text[tag_location..].starts_with(CONTINUE_FINAL_MESSAGE_TAG) {
        tag_location
    } else {
        text[..tag_location].trim_end().len()
    };
    Ok(truncate_rendered_prompt(&prompt, truncate_at))
}

fn truncate_rendered_prompt(prompt: &RenderedPrompt, truncate_at: usize) -> RenderedPrompt {
    let Some(segments) = prompt.segments() else {
        return RenderedPrompt::text(prompt.as_str()[..truncate_at].to_owned());
    };
    let mut remaining = truncate_at;
    let mut truncated = Vec::new();
    for segment in segments {
        if remaining == 0 {
            break;
        }
        let take = remaining.min(segment.text.len());
        if take != 0 {
            truncated.push(RenderedSegment::new(
                segment.text[..take].to_owned(),
                segment.allow_special,
            ));
        }
        remaining -= take;
    }
    RenderedPrompt::segmented(truncated)
}

fn validate_chat(request: &ChatRequest) -> Result<(), RendererError> {
    if request.messages.is_empty() {
        return Err("messages cannot be empty".into());
    }
    if request.choice_count == 0 {
        return Err("choice_count must be at least 1".into());
    }
    if serde_json::to_value(&request.messages).is_ok_and(|messages| contains_media(&messages)) {
        return Err("image, audio, video, and file message content is not supported".into());
    }
    Ok(())
}

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

fn merge_template_stops(sampling: &mut SamplingParams, formatter: Option<&ChatFormatter>) {
    let Some(template_stops) = formatter.and_then(ChatFormatter::stop_strs) else {
        return;
    };
    let mut stops = match template_stops {
        OneOrMany::One(stop) => vec![stop],
        OneOrMany::Many(stops) => stops,
    };
    if let Some(request_stops) = sampling.stop.take() {
        match request_stops {
            OneOrMany::One(stop) => stops.push(stop),
            OneOrMany::Many(request_stops) => stops.extend(request_stops),
        }
    }
    sampling.stop = Some(OneOrMany::Many(stops));
}

fn resolve_chat_parser(
    configured_parser: Option<&str>,
    tools_enabled: bool,
) -> Result<Option<String>, RendererError> {
    if tools_enabled && configured_parser.is_none() {
        return Err("tool calls require --tool-call-parser".into());
    }
    Ok(tools_enabled.then(|| configured_parser.expect("checked").to_owned()))
}

fn chat_tool_definitions(request: &ChatRequest) -> Vec<ToolDefinition> {
    request
        .tools
        .iter()
        .flatten()
        .map(|tool| ToolDefinition {
            name: tool.function.name.clone(),
            parameters: tool.function.parameters.clone(),
            strict: tool.function.strict,
        })
        .collect()
}

pub(crate) fn dynamo_parser_name(parser: &str) -> &str {
    match parser {
        "llama3" => "llama3_json",
        "qwen" => "qwen25",
        "glm" | "glm45" => "glm47",
        other => other,
    }
}

fn dynamo_tool_choice(choice: &Option<ChatCompletionToolChoiceOption>) -> DynamoToolChoice {
    match choice {
        Some(ChatCompletionToolChoiceOption::None) => DynamoToolChoice::None,
        Some(ChatCompletionToolChoiceOption::Required) => DynamoToolChoice::Required,
        Some(ChatCompletionToolChoiceOption::Named(choice)) => {
            DynamoToolChoice::Named(choice.function.name.clone())
        }
        Some(ChatCompletionToolChoiceOption::Auto) | None => DynamoToolChoice::Auto,
    }
}

fn apply_tool_constraint(
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
        return Ok(());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RendererLimits, SamplingDefaults};
    use dynamo_protocols::types::{
        ChatCompletionNamedToolChoice, ChatCompletionToolType, FunctionName,
    };

    fn tool(name: &str, strict: bool) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            })),
            strict: Some(strict),
        }
    }

    fn chat_request(tool_choice: Option<ChatCompletionToolChoiceOption>) -> ChatRequest {
        ChatRequest {
            rid: "chatcmpl-test".into(),
            model: "model".into(),
            messages: serde_json::from_value(serde_json::json!([
                {"role": "user", "content": "hello"}
            ]))
            .unwrap(),
            tools: Some(
                serde_json::from_value(serde_json::json!([{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"}
                    }
                }]))
                .unwrap(),
            ),
            tool_choice,
            response_format: None,
            reasoning_effort: None,
            continue_final_message: false,
            chat_template_args: None,
            sampling_params: SamplingParams::default(),
            choice_count: 1,
            stream: false,
            return_logprob: false,
            top_logprobs_num: 0,
            parallel_tool_calls: true,
            metadata: GenerateRequestMetadata::default(),
        }
    }

    fn chat_preprocessor() -> ChatPreprocessor {
        chat_preprocessor_with(
            Some("llama3"),
            None,
            crate::preprocessing::template::load_chat_formatter(None, None, Some("chatml"))
                .unwrap(),
        )
    }

    fn chat_preprocessor_with(
        tool_call_parser: Option<&str>,
        reasoning_parser: Option<&str>,
        formatter: ChatFormatter,
    ) -> ChatPreprocessor {
        let config = RendererConfig {
            served_model_name: "model".into(),
            tokenizer_path: ".".into(),
            revision: None,
            model_path: String::new(),
            chat_template: Some("chatml".into()),
            tool_call_parser: tool_call_parser.map(str::to_owned),
            reasoning_parser: reasoning_parser.map(str::to_owned),
            default_chat_template_kwargs: Default::default(),
            stream_response_default_include_usage: false,
            default_sampling_params: SamplingDefaults::default(),
            limits: RendererLimits {
                vocab_size: 128,
                context_len: 128,
                num_reserved_tokens: 0,
                allow_auto_truncate: false,
                enable_return_hidden_states: false,
            },
        };
        ChatPreprocessor::new(&config, Some(formatter))
    }

    #[test]
    fn wire_tool_choices_lower_to_internal_choices() {
        let named = Some(ChatCompletionToolChoiceOption::Named(
            ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: FunctionName {
                    name: "get_weather".into(),
                },
            },
        ));

        assert!(matches!(dynamo_tool_choice(&None), DynamoToolChoice::Auto));
        assert!(matches!(
            dynamo_tool_choice(&Some(ChatCompletionToolChoiceOption::Required)),
            DynamoToolChoice::Required
        ));
        assert!(matches!(
            dynamo_tool_choice(&named),
            DynamoToolChoice::Named(name) if name == "get_weather"
        ));
    }

    #[test]
    fn required_choice_builds_a_single_call_constraint() {
        let mut sampling = SamplingParams::default();
        apply_tool_constraint(
            &mut sampling,
            Some("llama3"),
            &DynamoToolChoice::Required,
            &[tool("get_weather", false), tool("get_time", false)],
            Some(false),
        )
        .unwrap();

        let schema: serde_json::Value =
            serde_json::from_str(sampling.json_schema.as_deref().unwrap()).unwrap();
        assert_eq!(schema["minItems"], 1);
        assert_eq!(schema["maxItems"], 1);
    }

    #[test]
    fn invalid_tool_choices_are_rejected_before_generation() {
        let mut sampling = SamplingParams::default();
        assert!(
            apply_tool_constraint(&mut sampling, None, &DynamoToolChoice::Required, &[], None,)
                .unwrap_err()
                .contains("required")
        );
        assert!(
            apply_tool_constraint(
                &mut sampling,
                None,
                &DynamoToolChoice::Named("missing".into()),
                &[tool("get_weather", false)],
                None,
            )
            .unwrap_err()
            .contains("missing")
        );
    }

    #[test]
    fn tool_parsing_preserves_special_tokens_for_output_processing() {
        let mut request = chat_request(None);
        request.sampling_params.skip_special_tokens = true;

        let chat = chat_preprocessor().preprocess(request).unwrap();

        assert!(
            !chat.text_requests[0]
                .options
                .sampling_params
                .skip_special_tokens
        );
    }

    #[test]
    fn tool_choice_none_keeps_the_requested_special_token_behavior() {
        let mut request = chat_request(Some(ChatCompletionToolChoiceOption::None));
        request.sampling_params.skip_special_tokens = true;

        let chat = chat_preprocessor().preprocess(request).unwrap();

        assert!(
            chat.text_requests[0]
                .options
                .sampling_params
                .skip_special_tokens
        );
    }

    #[test]
    fn qwen_required_tools_forward_effective_template_thinking() {
        let formatter = crate::preprocessing::template::test_hugging_face_formatter(
            "{% if enable_thinking is not defined %}{% set enable_thinking = true %}{% endif %}{{ enable_thinking }}",
        );
        let preprocessor = chat_preprocessor_with(Some("qwen"), Some("qwen3"), formatter);

        let enabled = preprocessor
            .preprocess(chat_request(Some(ChatCompletionToolChoiceOption::Required)))
            .unwrap();
        assert!(enabled.text_requests[0].options.require_reasoning);

        let mut disabled_request = chat_request(Some(ChatCompletionToolChoiceOption::Required));
        disabled_request.reasoning_effort = Some(ReasoningEffort::Max);
        disabled_request.chat_template_args = Some(HashMap::from([(
            "enable_thinking".into(),
            serde_json::Value::Bool(false),
        )]));
        let disabled = preprocessor.preprocess(disabled_request).unwrap();
        assert!(!disabled.text_requests[0].options.require_reasoning);
    }

    #[test]
    fn thinking_policy_uses_the_effective_tool_template() {
        let formatter = crate::preprocessing::template::test_hugging_face_formatter_from_config(
            serde_json::json!({
                "chat_template": [
                    {"default": "{{ enable_thinking | default(false) }}"},
                    {"tool_use": "{{ enable_thinking | default(true) }}"}
                ]
            }),
        );
        let preprocessor = chat_preprocessor_with(Some("qwen"), Some("qwen3"), formatter);

        let mut no_tools = chat_request(None);
        no_tools.tools = None;
        assert!(
            !preprocessor.preprocess(no_tools).unwrap().text_requests[0]
                .options
                .require_reasoning
        );

        let mut empty_tools = chat_request(None);
        empty_tools.tools = Some(Vec::new());
        assert!(
            !preprocessor.preprocess(empty_tools).unwrap().text_requests[0]
                .options
                .require_reasoning
        );

        assert!(
            !preprocessor
                .preprocess(chat_request(Some(ChatCompletionToolChoiceOption::None)))
                .unwrap()
                .text_requests[0]
                .options
                .require_reasoning
        );
        assert!(
            preprocessor
                .preprocess(chat_request(Some(ChatCompletionToolChoiceOption::Required)))
                .unwrap()
                .text_requests[0]
                .options
                .require_reasoning
        );
    }

    #[test]
    fn always_on_channel_template_requires_reasoning() {
        let formatter = crate::preprocessing::template::test_hugging_face_formatter(
            "<|start|>assistant<|channel|>analysis<|message|>",
        );
        let preprocessor = chat_preprocessor_with(None, Some("gpt-oss"), formatter);
        let mut request = chat_request(None);
        request.tools = None;
        request.response_format = Some(
            serde_json::from_value(serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"}
                }
            }))
            .unwrap(),
        );

        let lowered = preprocessor.preprocess(request).unwrap();

        assert!(lowered.text_requests[0].options.require_reasoning);
    }
}
