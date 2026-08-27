//! Protocol-neutral structured chat preprocessing.

use std::collections::HashMap;

use dynamo_parsers::parsers::get_tool_parser_map;
use dynamo_parsers::{
    StructuralTagBuilder, StructuralTagSchemaMode, ToolCallFormatBuildContext,
    ToolChoice as DynamoToolChoice, ToolDefinition, TriggeredTagsConfig,
};
use dynamo_protocols::types::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestMessage, ChatCompletionTool,
    ChatCompletionToolChoiceOption, ReasoningEffort, ResponseFormat,
};
use dynamo_renderer::{
    OAIChatLikeRequest, RenderedPrompt, RenderedSegment, TextInput, may_be_fix_tool_schema,
};
use minijinja::Value;

use crate::{
    ChatFormatter, ChatResponseProcessor, GenerationOptions, OneOrMany, RendererConfig,
    RendererError, SamplingParams, TextRequest,
};

/// Structured chat request shared by protocol adapters.
///
/// HTTP, gRPC, Messages, and Responses adapters lower into this type. It stays
/// structured until [`ChatPreprocessor`] applies the model chat template and
/// lowers it to the same [`TextRequest`] consumed by text completions.
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

    fn extract_text(&self) -> Option<TextInput> {
        Some(TextInput::Single(String::new()))
    }
}

/// Chat-to-text result plus the state needed to interpret generated output.
pub struct LoweredChat {
    pub text_requests: Vec<TextRequest>,
    pub response_processor: ChatResponseProcessor,
}

/// Applies structured chat semantics before the shared text generation path.
pub struct ChatPreprocessor {
    formatter: Option<ChatFormatter>,
    formatter_error: Option<String>,
    tool_call_parser: Option<String>,
    reasoning_parser: Option<String>,
}

impl ChatPreprocessor {
    pub(crate) fn new(config: &RendererConfig, formatter: Option<ChatFormatter>) -> Self {
        Self {
            formatter,
            formatter_error: None,
            tool_call_parser: config.tool_call_parser.clone(),
            reasoning_parser: config.reasoning_parser.clone(),
        }
    }

    pub(crate) fn with_formatter_error(mut self, error: Option<String>) -> Self {
        self.formatter_error = error;
        self
    }

    pub fn preprocess(&self, mut request: ChatRequest) -> Result<LoweredChat, RendererError> {
        validate_chat(&request)?;
        merge_template_stops(&mut request.sampling_params, self.formatter.as_ref());

        let tool_choice = dynamo_tool_choice(&request.tool_choice);
        let tools = chat_tool_definitions(&request);
        let parser = resolve_chat_parser(
            self.tool_call_parser.as_deref(),
            !tools.is_empty(),
            &tool_choice,
        )?;
        apply_tool_constraint(
            &mut request.sampling_params,
            parser.as_deref(),
            &tool_choice,
            &tools,
            Some(request.parallel_tool_calls),
        )?;
        let prompt = self.render(&request)?;
        let uses_tool_call_structural_tag = request.sampling_params.structural_tag.is_some();

        let mut text_requests = Vec::with_capacity(request.choice_count);
        for index in 0..request.choice_count {
            text_requests.push(TextRequest::rendered(
                format!("{}-{index}", request.rid),
                prompt.clone(),
                false,
                GenerationOptions {
                    sampling_params: request.sampling_params.clone(),
                    stream: request.stream,
                    return_logprob: request.return_logprob,
                    logprob_start_len: -1,
                    top_logprobs_num: request.top_logprobs_num,
                    return_text_in_logprobs: request.return_logprob.then_some(true),
                    ..Default::default()
                },
            ));
        }

        let response_processor = ChatResponseProcessor::new(
            parser,
            self.reasoning_parser.clone(),
            (!tools.is_empty()).then_some(tools),
            request.tool_choice,
            uses_tool_call_structural_tag,
            request.parallel_tool_calls,
            request.choice_count,
        );
        Ok(LoweredChat {
            text_requests,
            response_processor,
        })
    }

    /// Render chat for tokenization without creating generation/output state.
    pub fn lower_to_text(&self, mut request: ChatRequest) -> Result<TextRequest, RendererError> {
        validate_chat(&request)?;
        merge_template_stops(&mut request.sampling_params, self.formatter.as_ref());
        let prompt = self.render(&request)?;
        Ok(TextRequest::rendered(
            request.rid,
            prompt,
            false,
            GenerationOptions {
                sampling_params: request.sampling_params,
                ..Default::default()
            },
        ))
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
    has_tools: bool,
    tool_choice: &DynamoToolChoice,
) -> Result<Option<String>, RendererError> {
    let tools_enabled = has_tools && *tool_choice != DynamoToolChoice::None;
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
}
