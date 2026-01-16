// OpenAI Responses API types
// https://platform.openai.com/docs/api-reference/responses

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

use super::{
    common::{
        default_model, default_true, validate_stop, ChatLogProbs, Function, GenerationRequest,
        PromptTokenUsageInfo, StringOrArray, ToolChoice, ToolChoiceValue, ToolReference, UsageInfo,
    },
    sampling_params::{validate_top_k_value, validate_top_p_value},
};
use crate::protocols::{builders::ResponsesResponseBuilder, validated::Normalizable};

// ============================================================================
// Response Tools (MCP and others)
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseTool {
    #[serde(rename = "type")]
    pub r#type: ResponseToolType,
    // Function tool fields (used when type == "function")
    // In Responses API, function fields are flattened at the top level
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<Function>,
    // MCP-specific fields (used when type == "mcp")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_approval: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_tools: Option<Vec<String>>,
}

impl Default for ResponseTool {
    fn default() -> Self {
        Self {
            r#type: ResponseToolType::WebSearchPreview,
            function: None,
            server_url: None,
            authorization: None,
            headers: None,
            server_label: None,
            server_description: None,
            require_approval: None,
            allowed_tools: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseToolType {
    Function,
    WebSearchPreview,
    CodeInterpreter,
    Mcp,
}

// ============================================================================
// Reasoning Parameters
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseReasoningParam {
    #[serde(default = "default_reasoning_effort")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<ReasoningSummary>,
}

fn default_reasoning_effort() -> Option<ReasoningEffort> {
    Some(ReasoningEffort::Medium)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Minimal,
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

// ============================================================================
// Input/Output Items
// ============================================================================

/// Content can be either a simple string or array of content parts (for SimpleInputMessage)
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StringOrContentParts {
    String(String),
    Array(Vec<ResponseContentPart>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseInputOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<ResponseContentPart>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        summary: Vec<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        #[serde(default)]
        content: Vec<ResponseReasoningContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "function_call")]
    FunctionToolCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        id: Option<String>,
        call_id: String,
        output: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(untagged)]
    SimpleInputMessage {
        content: StringOrContentParts,
        role: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(rename = "type")]
        r#type: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseContentPart {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
        #[serde(default)]
        #[serde(skip_serializing_if = "Vec::is_empty")]
        annotations: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<ChatLogProbs>,
    },
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseReasoningContent {
    #[serde(rename = "reasoning_text")]
    ReasoningText { text: String },
}

/// MCP Tool information for the mcp_list_tools output item
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpToolInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<ResponseContentPart>,
        status: String,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        summary: Vec<String>,
        content: Vec<ResponseReasoningContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "function_call")]
    FunctionToolCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        status: String,
    },
    #[serde(rename = "mcp_list_tools")]
    McpListTools {
        id: String,
        server_label: String,
        tools: Vec<McpToolInfo>,
    },
    #[serde(rename = "mcp_call")]
    McpCall {
        id: String,
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        approval_request_id: Option<String>,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        name: String,
        output: String,
        server_label: String,
    },
}

// ============================================================================
// Configuration Enums
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    #[default]
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Truncation {
    Auto,
    #[default]
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReasoningInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

// ============================================================================
// Text Format (structured outputs)
// ============================================================================

/// Text configuration for structured output requests
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TextConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<TextFormat>,
}

/// Text format: text (default), json_object (legacy), or json_schema (recommended)
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum TextFormat {
    #[serde(rename = "text")]
    Text,

    #[serde(rename = "json_object")]
    JsonObject,

    #[serde(rename = "json_schema")]
    JsonSchema {
        name: String,
        schema: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum IncludeField {
    #[serde(rename = "code_interpreter_call.outputs")]
    CodeInterpreterCallOutputs,
    #[serde(rename = "computer_call_output.output.image_url")]
    ComputerCallOutputImageUrl,
    #[serde(rename = "file_search_call.results")]
    FileSearchCallResults,
    #[serde(rename = "message.input_image.image_url")]
    MessageInputImageUrl,
    #[serde(rename = "message.output_text.logprobs")]
    MessageOutputTextLogprobs,
    #[serde(rename = "reasoning.encrypted_content")]
    ReasoningEncryptedContent,
}

// ============================================================================
// Usage Types (Responses API format)
// ============================================================================

/// OpenAI Responses API usage format (different from standard UsageInfo)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResponsesUsage {
    Classic(UsageInfo),
    Modern(ResponseUsage),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u32,
}

impl UsageInfo {
    /// Convert to OpenAI Responses API format
    pub fn to_response_usage(&self) -> ResponseUsage {
        ResponseUsage {
            input_tokens: self.prompt_tokens,
            output_tokens: self.completion_tokens,
            total_tokens: self.total_tokens,
            input_tokens_details: self.prompt_tokens_details.as_ref().map(|details| {
                InputTokensDetails {
                    cached_tokens: details.cached_tokens,
                }
            }),
            output_tokens_details: self.reasoning_tokens.map(|tokens| OutputTokensDetails {
                reasoning_tokens: tokens,
            }),
        }
    }
}

impl From<UsageInfo> for ResponseUsage {
    fn from(usage: UsageInfo) -> Self {
        usage.to_response_usage()
    }
}

impl ResponseUsage {
    /// Convert back to standard UsageInfo format
    pub fn to_usage_info(&self) -> UsageInfo {
        UsageInfo {
            prompt_tokens: self.input_tokens,
            completion_tokens: self.output_tokens,
            total_tokens: self.total_tokens,
            reasoning_tokens: self
                .output_tokens_details
                .as_ref()
                .map(|details| details.reasoning_tokens),
            prompt_tokens_details: self.input_tokens_details.as_ref().map(|details| {
                PromptTokenUsageInfo {
                    cached_tokens: details.cached_tokens,
                }
            }),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ResponsesGetParams {
    #[serde(default)]
    pub include: Vec<String>,
    #[serde(default)]
    pub include_obfuscation: Option<bool>,
    #[serde(default)]
    pub starting_after: Option<i64>,
    #[serde(default)]
    pub stream: Option<bool>,
}

impl ResponsesUsage {
    pub fn to_response_usage(&self) -> ResponseUsage {
        match self {
            ResponsesUsage::Classic(usage) => usage.to_response_usage(),
            ResponsesUsage::Modern(usage) => usage.clone(),
        }
    }

    pub fn to_usage_info(&self) -> UsageInfo {
        match self {
            ResponsesUsage::Classic(usage) => usage.clone(),
            ResponsesUsage::Modern(usage) => usage.to_usage_info(),
        }
    }
}

// ============================================================================
// Helper Functions for Defaults
// ============================================================================

fn default_top_k() -> i32 {
    -1
}

fn default_repetition_penalty() -> f32 {
    1.0
}

fn default_temperature() -> Option<f32> {
    Some(1.0)
}

fn default_top_p() -> Option<f32> {
    Some(1.0)
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Validate)]
#[validate(schema(function = "validate_responses_cross_parameters"))]
pub struct ResponsesRequest {
    /// Run the request in the background
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,

    /// Fields to include in the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeField>>,

    /// Input content - can be string or structured items
    #[validate(custom(function = "validate_response_input"))]
    pub input: ResponseInput,

    /// System instructions for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Maximum number of output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub max_output_tokens: Option<u32>,

    /// Maximum number of tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub max_tool_calls: Option<u32>,

    /// Additional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,

    /// Model to use
    #[serde(default = "default_model")]
    pub model: String,

    /// Optional conversation id to persist input/output as items
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_conversation_id"))]
    pub conversation: Option<String>,

    /// Whether to enable parallel tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// ID of previous response to continue from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Reasoning configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponseReasoningParam>,

    /// Service tier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,

    /// Whether to store the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: Option<bool>,

    /// Temperature for sampling
    #[serde(
        default = "default_temperature",
        skip_serializing_if = "Option::is_none"
    )]
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: Option<f32>,

    /// Tool choice behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Available tools
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_response_tools"))]
    pub tools: Option<Vec<ResponseTool>>,

    /// Number of top logprobs to return
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0, max = 20))]
    pub top_logprobs: Option<u32>,

    /// Top-p sampling parameter
    #[serde(default = "default_top_p", skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,

    /// Truncation behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,

    /// Text format for structured outputs (text, json_object, json_schema)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_text_format"))]
    pub text: Option<TextConfig>,

    /// User identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Request ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,

    /// Request priority
    #[serde(default)]
    pub priority: i32,

    /// Frequency penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_stop"))]
    pub stop: Option<StringOrArray>,

    /// Top-k sampling parameter (SGLang extension)
    #[serde(default = "default_top_k")]
    #[validate(custom(function = "validate_top_k_value"))]
    pub top_k: i32,

    /// Min-p sampling parameter (SGLang extension)
    #[serde(default)]
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: f32,

    /// Repetition penalty (SGLang extension)
    #[serde(default = "default_repetition_penalty")]
    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Items(Vec<ResponseInputOutputItem>),
    Text(String),
}

impl Default for ResponsesRequest {
    fn default() -> Self {
        Self {
            background: None,
            include: None,
            input: ResponseInput::Text(String::new()),
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            metadata: None,
            model: default_model(),
            conversation: None,
            parallel_tool_calls: None,
            previous_response_id: None,
            reasoning: None,
            service_tier: None,
            store: None,
            stream: None,
            temperature: None,
            tool_choice: None,
            tools: None,
            top_logprobs: None,
            top_p: None,
            truncation: None,
            text: None,
            user: None,
            request_id: None,
            priority: 0,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            top_k: default_top_k(),
            min_p: 0.0,
            repetition_penalty: default_repetition_penalty(),
        }
    }
}

impl Normalizable for ResponsesRequest {
    /// Normalize the request by applying defaults:
    /// 1. Apply tool_choice defaults based on tools presence
    /// 2. Apply parallel_tool_calls defaults
    /// 3. Apply store field defaults
    fn normalize(&mut self) {
        // 1. Apply tool_choice defaults
        if self.tool_choice.is_none() {
            if let Some(tools) = &self.tools {
                let choice_value = if !tools.is_empty() {
                    ToolChoiceValue::Auto
                } else {
                    ToolChoiceValue::None
                };
                self.tool_choice = Some(ToolChoice::Value(choice_value));
            }
            // If tools is None, leave tool_choice as None (don't set it)
        }

        // 2. Apply default for parallel_tool_calls if tools are present
        if self.parallel_tool_calls.is_none() && self.tools.is_some() {
            self.parallel_tool_calls = Some(true);
        }

        // 3. Ensure store defaults to true if not specified
        if self.store.is_none() {
            self.store = Some(true);
        }
    }
}

impl GenerationRequest for ResponsesRequest {
    fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    fn get_model(&self) -> Option<&str> {
        Some(self.model.as_str())
    }

    fn extract_text_for_routing(&self) -> String {
        match &self.input {
            ResponseInput::Text(text) => text.clone(),
            ResponseInput::Items(items) => items
                .iter()
                .filter_map(|item| match item {
                    ResponseInputOutputItem::Message { content, .. } => {
                        let texts: Vec<String> = content
                            .iter()
                            .filter_map(|part| match part {
                                ResponseContentPart::OutputText { text, .. } => Some(text.clone()),
                                ResponseContentPart::InputText { text } => Some(text.clone()),
                                ResponseContentPart::Unknown => None,
                            })
                            .collect();
                        if texts.is_empty() {
                            None
                        } else {
                            Some(texts.join(" "))
                        }
                    }
                    ResponseInputOutputItem::SimpleInputMessage { content, .. } => {
                        match content {
                            StringOrContentParts::String(s) => Some(s.clone()),
                            StringOrContentParts::Array(parts) => {
                                // SimpleInputMessage only supports InputText
                                let texts: Vec<String> = parts
                                    .iter()
                                    .filter_map(|part| match part {
                                        ResponseContentPart::InputText { text } => {
                                            Some(text.clone())
                                        }
                                        _ => None,
                                    })
                                    .collect();
                                if texts.is_empty() {
                                    None
                                } else {
                                    Some(texts.join(" "))
                                }
                            }
                        }
                    }
                    ResponseInputOutputItem::Reasoning { content, .. } => {
                        let texts: Vec<String> = content
                            .iter()
                            .map(|part| match part {
                                ResponseReasoningContent::ReasoningText { text } => text.clone(),
                            })
                            .collect();
                        if texts.is_empty() {
                            None
                        } else {
                            Some(texts.join(" "))
                        }
                    }
                    ResponseInputOutputItem::FunctionToolCall { arguments, .. } => {
                        Some(arguments.clone())
                    }
                    ResponseInputOutputItem::FunctionCallOutput { output, .. } => {
                        Some(output.clone())
                    }
                })
                .collect::<Vec<String>>()
                .join(" "),
        }
    }
}

/// Validate conversation ID format
pub fn validate_conversation_id(conv_id: &str) -> Result<(), validator::ValidationError> {
    if !conv_id.starts_with("conv_") {
        let mut error = validator::ValidationError::new("invalid_conversation_id");
        error.message = Some(std::borrow::Cow::Owned(format!(
            "Invalid 'conversation': '{}'. Expected an ID that begins with 'conv_'.",
            conv_id
        )));
        return Err(error);
    }

    // Check if the conversation ID contains only valid characters
    let is_valid = conv_id
        .chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == '-');

    if !is_valid {
        let mut error = validator::ValidationError::new("invalid_conversation_id");
        error.message = Some(std::borrow::Cow::Owned(format!(
            "Invalid 'conversation': '{}'. Expected an ID that contains letters, numbers, underscores, or dashes, but this value contained additional characters.",
            conv_id
        )));
        return Err(error);
    }
    Ok(())
}

/// Validates tool_choice requires tools and references exist
fn validate_tool_choice_with_tools(
    request: &ResponsesRequest,
) -> Result<(), validator::ValidationError> {
    let Some(tool_choice) = &request.tool_choice else {
        return Ok(());
    };

    let has_tools = request.tools.as_ref().is_some_and(|t| !t.is_empty());
    let is_some_choice = !matches!(tool_choice, ToolChoice::Value(ToolChoiceValue::None));

    // Check if tool_choice requires tools but none are provided
    if is_some_choice && !has_tools {
        let mut e = validator::ValidationError::new("tool_choice_requires_tools");
        e.message = Some("Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.".into());
        return Err(e);
    }

    // Validate tool references exist when tools are present
    if !has_tools {
        return Ok(());
    }

    // Extract function tool names from ResponseTools
    let tools = request.tools.as_ref().unwrap();
    let function_tool_names: Vec<&str> = tools
        .iter()
        .filter_map(|t| match t.r#type {
            ResponseToolType::Function => t.function.as_ref().map(|f| f.name.as_str()),
            _ => None,
        })
        .collect();

    // Validate tool references exist
    match tool_choice {
        ToolChoice::Function { function, .. } => {
            if !function_tool_names.contains(&function.name.as_str()) {
                let mut e = validator::ValidationError::new("tool_choice_function_not_found");
                e.message = Some(
                    format!(
                        "Invalid value for 'tool_choice': function '{}' not found in 'tools'.",
                        function.name
                    )
                    .into(),
                );
                return Err(e);
            }
        }
        ToolChoice::AllowedTools {
            mode,
            tools: allowed_tools,
            ..
        } => {
            // Validate mode is "auto" or "required"
            if mode != "auto" && mode != "required" {
                let mut e = validator::ValidationError::new("tool_choice_invalid_mode");
                e.message = Some(
                    format!(
                        "Invalid value for 'tool_choice.mode': must be 'auto' or 'required', got '{}'.",
                        mode
                    )
                    .into(),
                );
                return Err(e);
            }

            // Validate that all function tool references exist
            for tool_ref in allowed_tools {
                if let ToolReference::Function { name } = tool_ref {
                    if !function_tool_names.contains(&name.as_str()) {
                        let mut e = validator::ValidationError::new("tool_choice_tool_not_found");
                        e.message = Some(
                            format!(
                                "Invalid value for 'tool_choice.tools': tool '{}' not found in 'tools'.",
                                name
                            )
                            .into(),
                        );
                        return Err(e);
                    }
                }
                // Note: MCP and hosted tools don't need existence validation here
                // as they are resolved dynamically at runtime
            }
        }
        _ => {}
    }

    Ok(())
}

/// Schema-level validation for cross-field dependencies
fn validate_responses_cross_parameters(
    request: &ResponsesRequest,
) -> Result<(), validator::ValidationError> {
    // 1. Validate tool_choice requires tools (enhanced)
    validate_tool_choice_with_tools(request)?;

    // 2. Validate MCP server labels and URLs (Responses API)
    if let Some(tools) = &request.tools {
        let mut seen_labels = std::collections::HashSet::new();
        for tool in tools {
            if tool.r#type != ResponseToolType::Mcp {
                continue;
            }

            if tool
                .server_url
                .as_ref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .is_none()
            {
                let mut e = validator::ValidationError::new("mcp_server_url_required");
                e.message = Some("MCP tool must have a non-empty 'server_url'.".into());
                return Err(e);
            }

            let Some(label) = tool
                .server_label
                .as_ref()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            else {
                let mut e = validator::ValidationError::new("mcp_server_label_required");
                e.message = Some(
                    "MCP tool must have a non-empty 'server_label' when 'server_url' is provided."
                        .into(),
                );
                return Err(e);
            };

            if label.contains("__") {
                let mut e = validator::ValidationError::new("mcp_server_label_invalid");
                e.message = Some(
                    "MCP 'server_label' must not contain '__' to avoid tool name ambiguity.".into(),
                );
                return Err(e);
            }

            if !seen_labels.insert(label.to_string()) {
                let mut e = validator::ValidationError::new("mcp_server_label_duplicate");
                e.message = Some(
                    format!(
                        "Duplicate MCP 'server_label' found: '{}'. Each MCP server must have a unique label.",
                        label
                    )
                    .into(),
                );
                return Err(e);
            }
        }
    }

    // 3. Validate top_logprobs requires include field
    if request.top_logprobs.is_some() {
        let has_logprobs_include = request
            .include
            .as_ref()
            .is_some_and(|inc| inc.contains(&IncludeField::MessageOutputTextLogprobs));

        if !has_logprobs_include {
            let mut e = validator::ValidationError::new("top_logprobs_requires_include");
            e.message = Some(
                "top_logprobs requires include field with 'message.output_text.logprobs'".into(),
            );
            return Err(e);
        }
    }

    // 4. Validate background/stream conflict
    if request.background == Some(true) && request.stream == Some(true) {
        let mut e = validator::ValidationError::new("background_conflicts_with_stream");
        e.message = Some("Cannot use background mode with streaming".into());
        return Err(e);
    }

    // 5. Validate conversation and previous_response_id are mutually exclusive
    if request.conversation.is_some() && request.previous_response_id.is_some() {
        let mut e = validator::ValidationError::new("mutually_exclusive_parameters");
        e.message = Some("Mutually exclusive parameters. Ensure you are only providing one of: 'previous_response_id' or 'conversation'.".into());
        return Err(e);
    }

    // 6. Validate input items structure
    if let ResponseInput::Items(items) = &request.input {
        // Check for at least one valid input message
        let has_valid_input = items.iter().any(|item| {
            matches!(
                item,
                ResponseInputOutputItem::Message { .. }
                    | ResponseInputOutputItem::SimpleInputMessage { .. }
            )
        });

        if !has_valid_input {
            let mut e = validator::ValidationError::new("input_missing_user_message");
            e.message = Some("Input items must contain at least one message".into());
            return Err(e);
        }
    }

    // 7. Validate text format conflicts (for future structured output constraints)
    // Currently, Responses API doesn't have regex/ebnf like Chat API,
    // but this is here for completeness and future-proofing

    Ok(())
}

// ============================================================================
// Field-Level Validation Functions
// ============================================================================

/// Validates response input is not empty and has valid content
fn validate_response_input(input: &ResponseInput) -> Result<(), validator::ValidationError> {
    match input {
        ResponseInput::Text(text) => {
            if text.is_empty() {
                let mut e = validator::ValidationError::new("input_text_empty");
                e.message = Some("Input text cannot be empty".into());
                return Err(e);
            }
        }
        ResponseInput::Items(items) => {
            if items.is_empty() {
                let mut e = validator::ValidationError::new("input_items_empty");
                e.message = Some("Input items cannot be empty".into());
                return Err(e);
            }
            // Validate each item has valid content
            for item in items {
                validate_input_item(item)?;
            }
        }
    }
    Ok(())
}

/// Validates individual input items have valid content
fn validate_input_item(item: &ResponseInputOutputItem) -> Result<(), validator::ValidationError> {
    match item {
        ResponseInputOutputItem::Message { content, .. } => {
            if content.is_empty() {
                let mut e = validator::ValidationError::new("message_content_empty");
                e.message = Some("Message content cannot be empty".into());
                return Err(e);
            }
        }
        ResponseInputOutputItem::SimpleInputMessage { content, .. } => match content {
            StringOrContentParts::String(s) if s.is_empty() => {
                let mut e = validator::ValidationError::new("message_content_empty");
                e.message = Some("Message content cannot be empty".into());
                return Err(e);
            }
            StringOrContentParts::Array(parts) if parts.is_empty() => {
                let mut e = validator::ValidationError::new("message_content_empty");
                e.message = Some("Message content parts cannot be empty".into());
                return Err(e);
            }
            _ => {}
        },
        ResponseInputOutputItem::Reasoning { .. } => {
            // Reasoning content can be empty - no validation needed
        }
        ResponseInputOutputItem::FunctionCallOutput { output, .. } => {
            if output.is_empty() {
                let mut e = validator::ValidationError::new("function_output_empty");
                e.message = Some("Function call output cannot be empty".into());
                return Err(e);
            }
        }
        _ => {}
    }
    Ok(())
}

/// Validates ResponseTool structure based on tool type
fn validate_response_tools(tools: &[ResponseTool]) -> Result<(), validator::ValidationError> {
    for tool in tools {
        match tool.r#type {
            ResponseToolType::Function => {
                if tool.function.is_none() {
                    let mut e = validator::ValidationError::new("function_tool_missing_function");
                    e.message = Some("Function tool must have a function definition".into());
                    return Err(e);
                }
            }
            ResponseToolType::Mcp => {
                if tool.server_url.is_none() {
                    let mut e = validator::ValidationError::new("mcp_tool_missing_server_url");
                    e.message = Some("MCP tool must have a server_url".into());
                    return Err(e);
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Validates text format configuration (JSON schema name cannot be empty)
fn validate_text_format(text: &TextConfig) -> Result<(), validator::ValidationError> {
    if let Some(TextFormat::JsonSchema { name, .. }) = &text.format {
        if name.is_empty() {
            let mut e = validator::ValidationError::new("json_schema_name_empty");
            e.message = Some("JSON schema name cannot be empty".into());
            return Err(e);
        }
    }
    Ok(())
}

/// Normalize a SimpleInputMessage to a proper Message item
///
/// This helper converts SimpleInputMessage (which can have flexible content)
/// into a fully-structured Message item with a generated ID, role, and content array.
///
/// SimpleInputMessage items are converted to Message items with IDs generated using
/// the centralized ID generation pattern with "msg_" prefix for consistency.
///
/// # Arguments
/// * `item` - The input item to normalize
///
/// # Returns
/// A normalized ResponseInputOutputItem (either Message if converted, or original if not SimpleInputMessage)
pub fn normalize_input_item(item: &ResponseInputOutputItem) -> ResponseInputOutputItem {
    match item {
        ResponseInputOutputItem::SimpleInputMessage { content, role, .. } => {
            let content_vec = match content {
                StringOrContentParts::String(s) => {
                    vec![ResponseContentPart::InputText { text: s.clone() }]
                }
                StringOrContentParts::Array(parts) => parts.clone(),
            };

            ResponseInputOutputItem::Message {
                id: generate_id("msg"),
                role: role.clone(),
                content: content_vec,
                status: Some("completed".to_string()),
            }
        }
        _ => item.clone(),
    }
}

pub fn generate_id(prefix: &str) -> String {
    use rand::RngCore;
    let mut rng = rand::rng();
    // Generate exactly 50 hex characters (25 bytes) for the part after the underscore
    let mut bytes = [0u8; 25];
    rng.fill_bytes(&mut bytes);
    let hex_string: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
    format!("{}_{}", prefix, hex_string)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesResponse {
    /// Response ID
    pub id: String,

    /// Object type
    #[serde(default = "default_object_type")]
    pub object: String,

    /// Creation timestamp
    pub created_at: i64,

    /// Response status
    pub status: ResponseStatus,

    /// Error information if status is failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,

    /// Incomplete details if response was truncated
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<Value>,

    /// System instructions used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Max output tokens setting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// Model name
    pub model: String,

    /// Output items
    #[serde(default)]
    pub output: Vec<ResponseOutputItem>,

    /// Whether parallel tool calls are enabled
    #[serde(default = "default_true")]
    pub parallel_tool_calls: bool,

    /// Previous response ID if this is a continuation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Reasoning information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningInfo>,

    /// Whether the response is stored
    #[serde(default = "default_true")]
    pub store: bool,

    /// Temperature setting used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Text format settings
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<TextConfig>,

    /// Tool choice setting
    #[serde(default = "default_tool_choice")]
    pub tool_choice: String,

    /// Available tools
    #[serde(default)]
    pub tools: Vec<ResponseTool>,

    /// Top-p setting used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Truncation strategy used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<String>,

    /// Usage information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponsesUsage>,

    /// User identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Safety identifier for content moderation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,

    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

fn default_object_type() -> String {
    "response".to_string()
}

fn default_tool_choice() -> String {
    "auto".to_string()
}

impl ResponsesResponse {
    /// Create a builder for constructing a ResponsesResponse
    pub fn builder(id: impl Into<String>, model: impl Into<String>) -> ResponsesResponseBuilder {
        ResponsesResponseBuilder::new(id, model)
    }

    /// Check if the response is complete
    pub fn is_complete(&self) -> bool {
        matches!(self.status, ResponseStatus::Completed)
    }

    /// Check if the response is in progress
    pub fn is_in_progress(&self) -> bool {
        matches!(self.status, ResponseStatus::InProgress)
    }

    /// Check if the response failed
    pub fn is_failed(&self) -> bool {
        matches!(self.status, ResponseStatus::Failed)
    }
}

impl ResponseOutputItem {
    /// Create a new message output item
    pub fn new_message(
        id: String,
        role: String,
        content: Vec<ResponseContentPart>,
        status: String,
    ) -> Self {
        Self::Message {
            id,
            role,
            content,
            status,
        }
    }

    /// Create a new reasoning output item
    pub fn new_reasoning(
        id: String,
        summary: Vec<String>,
        content: Vec<ResponseReasoningContent>,
        status: Option<String>,
    ) -> Self {
        Self::Reasoning {
            id,
            summary,
            content,
            status,
        }
    }

    /// Create a new function tool call output item
    pub fn new_function_tool_call(
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        output: Option<String>,
        status: String,
    ) -> Self {
        Self::FunctionToolCall {
            id,
            call_id,
            name,
            arguments,
            output,
            status,
        }
    }
}

impl ResponseContentPart {
    /// Create a new text content part
    pub fn new_text(
        text: String,
        annotations: Vec<String>,
        logprobs: Option<ChatLogProbs>,
    ) -> Self {
        Self::OutputText {
            text,
            annotations,
            logprobs,
        }
    }
}

impl ResponseReasoningContent {
    /// Create a new reasoning text content
    pub fn new_reasoning_text(text: String) -> Self {
        Self::ReasoningText { text }
    }
}
