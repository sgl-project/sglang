use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator::Validate;

use super::{
    common::*,
    sampling_params::{validate_top_k_value, validate_top_p_value},
};
use crate::protocols::{
    builders::{ChatCompletionResponseBuilder, ChatCompletionStreamResponseBuilder},
    validated::Normalizable,
};

// ============================================================================
// Chat Messages
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "role")]
pub enum ChatMessage {
    #[serde(rename = "system")]
    System {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "user")]
    User {
        content: UserMessageContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "assistant")]
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
        /// Reasoning content for O1-style models (SGLang extension)
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
    },
    #[serde(rename = "tool")]
    Tool {
        content: String,
        tool_call_id: String,
    },
    #[serde(rename = "function")]
    Function { content: String, name: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum UserMessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

// ============================================================================
// Chat Completion Request
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Default, Validate)]
#[validate(schema(function = "validate_chat_cross_parameters"))]
pub struct ChatCompletionRequest {
    /// A list of messages comprising the conversation so far
    #[validate(custom(function = "validate_messages"))]
    pub messages: Vec<ChatMessage>,

    /// ID of the model to use
    #[serde(default = "default_model")]
    pub model: String,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,

    /// Deprecated: Replaced by tool_choice
    #[serde(skip_serializing_if = "Option::is_none")]
    #[deprecated(note = "Use tool_choice instead")]
    pub function_call: Option<FunctionCall>,

    /// Deprecated: Replaced by tools
    #[serde(skip_serializing_if = "Option::is_none")]
    #[deprecated(note = "Use tools instead")]
    pub functions: Option<Vec<Function>>,

    /// Modify the likelihood of specified tokens appearing in the completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,

    /// Whether to return log probabilities of the output tokens
    #[serde(default)]
    pub logprobs: bool,

    /// Deprecated: Replaced by max_completion_tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    #[deprecated(note = "Use max_completion_tokens instead")]
    #[validate(range(min = 1))]
    pub max_tokens: Option<u32>,

    /// An upper bound for the number of tokens that can be generated for a completion
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub max_completion_tokens: Option<u32>,

    /// Developer-defined tags and values used for filtering completions in the dashboard
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,

    /// Output types that you would like the model to generate for this request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,

    /// How many chat completion choices to generate for each input message
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1, max = 10))]
    pub n: Option<u32>,

    /// Whether to enable parallel function calling during tool use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,

    /// Cache key for prompts (beta feature)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,

    /// Effort level for reasoning models (low, medium, high)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,

    /// An object specifying the format that the model must output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Safety identifier for content moderation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,

    /// Deprecated: This feature is in Legacy mode
    #[serde(skip_serializing_if = "Option::is_none")]
    #[deprecated(note = "This feature is in Legacy mode")]
    pub seed: Option<i64>,

    /// The service tier to use for this request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    /// Up to 4 sequences where the API will stop generating further tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_stop"))]
    pub stop: Option<StringOrArray>,

    /// If set, partial message deltas will be sent
    #[serde(default)]
    pub stream: bool,

    /// Options for streaming response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// What sampling temperature to use, between 0 and 2
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: Option<f32>,

    /// Controls which (if any) tool is called by the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// A list of tools the model may call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// An integer between 0 and 20 specifying the number of most likely tokens to return
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0, max = 20))]
    pub top_logprobs: Option<u32>,

    /// An alternative to sampling with temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,

    /// Verbosity level for debugging
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<i32>,

    // =============================================================================
    // Engine-Specific Sampling Parameters
    // =============================================================================
    // These parameters are extensions beyond the OpenAI API specification and
    // control model generation behavior in engine-specific ways.
    // =============================================================================
    /// Top-k sampling parameter (-1 to disable)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_top_k_value"))]
    pub top_k: Option<i32>,

    /// Min-p nucleus sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: Option<f32>,

    /// Minimum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1))]
    pub min_tokens: Option<u32>,

    /// Repetition penalty for reducing repetitive text
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: Option<f32>,

    /// Regex constraint for output generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,

    /// EBNF grammar constraint for structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ebnf: Option<String>,

    /// Specific token IDs to use as stop conditions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<u32>>,

    /// Skip trimming stop tokens from output
    #[serde(default)]
    pub no_stop_trim: bool,

    /// Ignore end-of-sequence tokens during generation
    #[serde(default)]
    pub ignore_eos: bool,

    /// Continue generating from final assistant message
    #[serde(default)]
    pub continue_final_message: bool,

    /// Skip special tokens during detokenization
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    /// Path to LoRA adapter(s) for model customization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<String>,

    /// Session parameters for continual prompting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_params: Option<HashMap<String, Value>>,

    /// Separate reasoning content from final answer (O1-style models)
    #[serde(default = "default_true")]
    pub separate_reasoning: bool,

    /// Stream reasoning tokens during generation
    #[serde(default = "default_true")]
    pub stream_reasoning: bool,

    /// Chat template kwargs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<HashMap<String, Value>>,

    /// Return model hidden states
    #[serde(default)]
    pub return_hidden_states: bool,

    /// Random seed for sampling for deterministic outputs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_seed: Option<u64>,
}

// ============================================================================
// Validation Functions
// ============================================================================

/// Validates stop sequences (max 4, non-empty strings)
fn validate_stop(stop: &StringOrArray) -> Result<(), validator::ValidationError> {
    match stop {
        StringOrArray::String(s) => {
            if s.is_empty() {
                return Err(validator::ValidationError::new(
                    "stop sequences cannot be empty",
                ));
            }
        }
        StringOrArray::Array(arr) => {
            if arr.len() > 4 {
                return Err(validator::ValidationError::new(
                    "maximum 4 stop sequences allowed",
                ));
            }
            for s in arr {
                if s.is_empty() {
                    return Err(validator::ValidationError::new(
                        "stop sequences cannot be empty",
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Validates messages array is not empty and has valid content
fn validate_messages(messages: &[ChatMessage]) -> Result<(), validator::ValidationError> {
    if messages.is_empty() {
        return Err(validator::ValidationError::new("messages cannot be empty"));
    }

    for msg in messages.iter() {
        if let ChatMessage::User { content, .. } = msg {
            match content {
                UserMessageContent::Text(text) if text.is_empty() => {
                    return Err(validator::ValidationError::new(
                        "message content cannot be empty",
                    ));
                }
                UserMessageContent::Parts(parts) if parts.is_empty() => {
                    return Err(validator::ValidationError::new(
                        "message content parts cannot be empty",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Schema-level validation for cross-field dependencies
fn validate_chat_cross_parameters(
    req: &ChatCompletionRequest,
) -> Result<(), validator::ValidationError> {
    // 1. Validate logprobs dependency
    if req.top_logprobs.is_some() && !req.logprobs {
        let mut e = validator::ValidationError::new("top_logprobs_requires_logprobs");
        e.message = Some("top_logprobs is only allowed when logprobs is enabled".into());
        return Err(e);
    }

    // 2. Validate stream_options dependency
    if req.stream_options.is_some() && !req.stream {
        let mut e = validator::ValidationError::new("stream_options_requires_stream");
        e.message =
            Some("The 'stream_options' parameter is only allowed when 'stream' is enabled".into());
        return Err(e);
    }

    // 3. Validate token limits - min <= max
    if let (Some(min), Some(max)) = (req.min_tokens, req.max_completion_tokens) {
        if min > max {
            let mut e = validator::ValidationError::new("min_tokens_exceeds_max");
            e.message = Some("min_tokens cannot exceed max_tokens/max_completion_tokens".into());
            return Err(e);
        }
    }

    // 4. Validate structured output conflicts
    let has_json_format = matches!(
        req.response_format,
        Some(ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. })
    );

    if has_json_format && req.regex.is_some() {
        let mut e = validator::ValidationError::new("regex_conflicts_with_json");
        e.message = Some("cannot use regex constraint with JSON response format".into());
        return Err(e);
    }

    if has_json_format && req.ebnf.is_some() {
        let mut e = validator::ValidationError::new("ebnf_conflicts_with_json");
        e.message = Some("cannot use EBNF constraint with JSON response format".into());
        return Err(e);
    }

    // 5. Validate mutually exclusive structured output constraints
    let constraint_count = [
        req.regex.is_some(),
        req.ebnf.is_some(),
        matches!(req.response_format, Some(ResponseFormat::JsonSchema { .. })),
    ]
    .iter()
    .filter(|&&x| x)
    .count();

    if constraint_count > 1 {
        let mut e = validator::ValidationError::new("multiple_constraints");
        e.message = Some("only one structured output constraint (regex, ebnf, or json_schema) can be active at a time".into());
        return Err(e);
    }

    // 6. Validate response format JSON schema name
    if let Some(ResponseFormat::JsonSchema { json_schema }) = &req.response_format {
        if json_schema.name.is_empty() {
            let mut e = validator::ValidationError::new("json_schema_name_empty");
            e.message = Some("JSON schema name cannot be empty".into());
            return Err(e);
        }
    }

    // 7. Validate tool_choice requires tools (except for "none")
    if let Some(ref tool_choice) = req.tool_choice {
        let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());

        // Check if tool_choice is anything other than "none"
        let is_some_choice = !matches!(tool_choice, ToolChoice::Value(ToolChoiceValue::None));

        if is_some_choice && !has_tools {
            let mut e = validator::ValidationError::new("tool_choice_requires_tools");
            e.message = Some("Invalid value for 'tool_choice': 'tool_choice' is only allowed when 'tools' are specified.".into());
            return Err(e);
        }

        // Additional validation when tools are present
        if has_tools {
            let tools = req.tools.as_ref().unwrap();

            match tool_choice {
                ToolChoice::Function { function, .. } => {
                    // Validate that the specified function name exists in tools
                    let function_exists = tools.iter().any(|tool| {
                        tool.tool_type == "function" && tool.function.name == function.name
                    });

                    if !function_exists {
                        let mut e =
                            validator::ValidationError::new("tool_choice_function_not_found");
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
                        e.message = Some(format!(
                            "Invalid value for 'tool_choice.mode': must be 'auto' or 'required', got '{}'.",
                            mode
                        ).into());
                        return Err(e);
                    }

                    // Validate that all ToolReferences are Function type (Chat API only supports function tools)
                    for tool_ref in allowed_tools {
                        match tool_ref {
                            ToolReference::Function { name } => {
                                // Validate that the function exists in tools array
                                let tool_exists = tools.iter().any(|tool| {
                                    tool.tool_type == "function" && tool.function.name == *name
                                });

                                if !tool_exists {
                                    let mut e = validator::ValidationError::new(
                                        "tool_choice_tool_not_found",
                                    );
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
                            _ => {
                                // Chat Completion API only supports function tools in tool_choice
                                let mut e = validator::ValidationError::new(
                                    "tool_choice_invalid_tool_type",
                                );
                                e.message = Some(
                                    format!(
                                        "Invalid value for 'tool_choice.tools': Chat Completion API only supports function tools, got '{}'.",
                                        tool_ref.identifier()
                                    )
                                    .into(),
                                );
                                return Err(e);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    Ok(())
}

// ============================================================================
// Normalizable Implementation
// ============================================================================

impl Normalizable for ChatCompletionRequest {
    /// Normalize the request by applying migrations and defaults:
    /// 1. Migrate deprecated fields to their replacements
    /// 2. Clear deprecated fields and log warnings
    /// 3. Apply OpenAI defaults for tool_choice
    fn normalize(&mut self) {
        // Migrate deprecated max_tokens → max_completion_tokens
        #[allow(deprecated)]
        if self.max_completion_tokens.is_none() && self.max_tokens.is_some() {
            self.max_completion_tokens = self.max_tokens;
            self.max_tokens = None; // Clear deprecated field
        }

        // Migrate deprecated functions → tools
        #[allow(deprecated)]
        if self.tools.is_none() && self.functions.is_some() {
            tracing::warn!("functions is deprecated, use tools instead");
            self.tools = self.functions.as_ref().map(|functions| {
                functions
                    .iter()
                    .map(|func| Tool {
                        tool_type: "function".to_string(),
                        function: func.clone(),
                    })
                    .collect()
            });
            self.functions = None; // Clear deprecated field
        }

        // Migrate deprecated function_call → tool_choice
        #[allow(deprecated)]
        if self.tool_choice.is_none() && self.function_call.is_some() {
            tracing::warn!("function_call is deprecated, use tool_choice instead");
            self.tool_choice = self.function_call.as_ref().map(|fc| match fc {
                FunctionCall::None => ToolChoice::Value(ToolChoiceValue::None),
                FunctionCall::Auto => ToolChoice::Value(ToolChoiceValue::Auto),
                FunctionCall::Function { name } => ToolChoice::Function {
                    tool_type: "function".to_string(),
                    function: FunctionChoice { name: name.clone() },
                },
            });
            self.function_call = None; // Clear deprecated field
        }

        // Apply tool_choice defaults
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
    }
}

// ============================================================================
// GenerationRequest Trait Implementation
// ============================================================================

impl GenerationRequest for ChatCompletionRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        // Extract text from messages for routing decisions
        self.messages
            .iter()
            .filter_map(|msg| match msg {
                ChatMessage::System { content, .. } => Some(content.clone()),
                ChatMessage::User { content, .. } => match content {
                    UserMessageContent::Text(text) => Some(text.clone()),
                    UserMessageContent::Parts(parts) => {
                        let texts: Vec<String> = parts
                            .iter()
                            .filter_map(|part| match part {
                                ContentPart::Text { text } => Some(text.clone()),
                                _ => None,
                            })
                            .collect();
                        Some(texts.join(" "))
                    }
                },
                ChatMessage::Assistant {
                    content,
                    reasoning_content,
                    ..
                } => {
                    // Combine content and reasoning content for routing decisions
                    let main_content = content.clone().unwrap_or_default();
                    let reasoning = reasoning_content.clone().unwrap_or_default();
                    if main_content.is_empty() && reasoning.is_empty() {
                        None
                    } else {
                        Some(format!("{} {}", main_content, reasoning).trim().to_string())
                    }
                }
                ChatMessage::Tool { content, .. } => Some(content.clone()),
                ChatMessage::Function { content, .. } => Some(content.clone()),
            })
            .collect::<Vec<String>>()
            .join(" ")
    }
}

// ============================================================================
// Response Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String, // "chat.completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

impl ChatCompletionResponse {
    /// Create a new builder for ChatCompletionResponse
    pub fn builder(
        id: impl Into<String>,
        model: impl Into<String>,
    ) -> ChatCompletionResponseBuilder {
        ChatCompletionResponseBuilder::new(id, model)
    }
}

/// Response message structure for ChatCompletionResponse (different from request ChatMessage)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionMessage {
    pub role: String, // Always "assistant" for responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    pub reasoning_content: Option<String>,
    // Note: function_call is deprecated and not included
    // Note: refusal, annotations, audio are not added yet
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>, // "stop", "length", "tool_calls", "content_filter", "function_call"
    /// Information about which stop condition was matched
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop: Option<Value>, // Can be string or integer
    /// Hidden states from the model (SGLang extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_states: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionStreamResponse {
    pub id: String,
    pub object: String, // "chat.completion.chunk"
    pub created: u64,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl ChatCompletionStreamResponse {
    /// Create a new builder for ChatCompletionStreamResponse
    pub fn builder(
        id: impl Into<String>,
        model: impl Into<String>,
    ) -> ChatCompletionStreamResponseBuilder {
        ChatCompletionStreamResponseBuilder::new(id, model)
    }
}

/// Delta structure for streaming chat completion responses
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatStreamChoice {
    pub index: u32,
    pub delta: ChatMessageDelta,
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop: Option<Value>,
}
