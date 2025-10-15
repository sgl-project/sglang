use serde::{Deserialize, Serialize};
use serde_json::{to_value, Map, Number, Value};
use std::collections::HashMap;
use validator::Validate;

use crate::protocols::validated::Normalizable;

// Default model value when not specified
fn default_model() -> String {
    "unknown".to_string()
}

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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>, // "auto", "low", or "high"
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema { json_schema: JsonSchemaFormat },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JsonSchemaFormat {
    pub name: String,
    pub schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    /// Reasoning content delta for O1-style models (SGLang extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "type")]
    pub tool_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

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

// Validation functions for ChatCompletionRequest
// These are automatically called by the validator derive macro

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

/// Validates top_p: 0.0 < top_p <= 1.0 (exclusive lower bound - can't use range validator)
fn validate_top_p_value(top_p: f32) -> Result<(), validator::ValidationError> {
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(validator::ValidationError::new(
            "top_p must be in (0, 1] - greater than 0.0 and at most 1.0",
        ));
    }
    Ok(())
}

/// Validates top_k: -1 (disabled) or >= 1 (special -1 case - can't use range validator)
fn validate_top_k_value(top_k: i32) -> Result<(), validator::ValidationError> {
    if top_k != -1 && top_k < 1 {
        return Err(validator::ValidationError::new(
            "top_k must be -1 (disabled) or at least 1",
        ));
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

                    // Validate that all referenced tool names exist in tools
                    for tool_ref in allowed_tools {
                        let tool_exists = tools.iter().any(|tool| {
                            tool.tool_type == tool_ref.tool_type
                                && tool.function.name == tool_ref.name
                        });

                        if !tool_exists {
                            let mut e =
                                validator::ValidationError::new("tool_choice_tool_not_found");
                            e.message = Some(format!(
                                "Invalid value for 'tool_choice.tools': tool '{}' not found in 'tools'.",
                                tool_ref.name
                            ).into());
                            return Err(e);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    Ok(())
}

impl Normalizable for ChatCompletionRequest {
    /// Normalize the request by applying migrations and defaults:
    /// 1. Migrate deprecated fields to their replacements
    /// 2. Clear deprecated fields and log warnings
    /// 3. Apply OpenAI defaults for tool_choice
    fn normalize(&mut self) {
        // Migrate deprecated max_tokens → max_completion_tokens
        #[allow(deprecated)]
        if self.max_completion_tokens.is_none() && self.max_tokens.is_some() {
            tracing::warn!("max_tokens is deprecated, use max_completion_tokens instead");
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
            let has_tools = self.tools.as_ref().is_some_and(|t| !t.is_empty());

            self.tool_choice = if has_tools {
                Some(ToolChoice::Value(ToolChoiceValue::Auto))
            } else {
                Some(ToolChoice::Value(ToolChoiceValue::None))
            };
        }
    }
}

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

/// Response message structure for ChatCompletionResponse (different from request ChatMessage)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionMessage {
    pub role: String, // Always "assistant" for responses
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Reasoning content for O1-style models (SGLang extension)
    #[serde(skip_serializing_if = "Option::is_none")]
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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatStreamChoice {
    pub index: u32,
    pub delta: ChatMessageDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop: Option<Value>,
}

// Completions API request types (v1/completions) - DEPRECATED but still supported

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionRequest {
    /// ID of the model to use (required for OpenAI, optional for some implementations, such as SGLang)
    pub model: String,

    /// The prompt(s) to generate completions for
    pub prompt: StringOrArray,

    /// The suffix that comes after a completion of inserted text
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,

    /// The maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// What sampling temperature to use, between 0 and 2
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// An alternative to sampling with temperature (nucleus sampling)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// How many completions to generate for each prompt
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// Whether to stream back partial progress
    #[serde(default)]
    pub stream: bool,

    /// Options for streaming response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Include the log probabilities on the logprobs most likely tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,

    /// Echo back the prompt in addition to the completion
    #[serde(default)]
    pub echo: bool,

    /// Up to 4 sequences where the API will stop generating further tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrArray>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Generates best_of completions server-side and returns the "best"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,

    /// Modify the likelihood of specified tokens appearing in the completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,

    /// A unique identifier representing your end-user
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// If specified, our system will make a best effort to sample deterministically
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    // -------- Engine Specific Sampling Parameters --------
    /// Top-k sampling parameter (-1 to disable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,

    /// Min-p nucleus sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,

    /// Minimum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u32>,

    /// Repetition penalty for reducing repetitive text
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,

    /// Regex constraint for output generation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,

    /// EBNF grammar constraint for structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ebnf: Option<String>,

    /// JSON schema constraint for structured output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<String>,

    /// Specific token IDs to use as stop conditions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<u32>>,

    /// Skip trimming stop tokens from output
    #[serde(default)]
    pub no_stop_trim: bool,

    /// Ignore end-of-sequence tokens during generation
    #[serde(default)]
    pub ignore_eos: bool,

    /// Skip special tokens during detokenization
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    /// Path to LoRA adapter(s) for model customization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<String>,

    /// Session parameters for continual prompting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_params: Option<HashMap<String, Value>>,

    /// Return model hidden states
    #[serde(default)]
    pub return_hidden_states: bool,

    /// Sampling seed for deterministic outputs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_seed: Option<u64>,

    /// Additional fields including bootstrap info for PD routing
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

impl GenerationRequest for CompletionRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        match &self.prompt {
            StringOrArray::String(s) => s.clone(),
            StringOrArray::Array(v) => v.join(" "),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String, // "text_completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>, // "stop", "length", "content_filter", etc.
    /// Information about which stop condition was matched
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop: Option<Value>, // Can be string or integer
    /// Hidden states from the model (SGLang extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_states: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionStreamResponse {
    pub id: String,
    pub object: String, // "text_completion"
    pub created: u64,
    pub choices: Vec<CompletionStreamChoice>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionStreamChoice {
    pub text: String,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseTool {
    #[serde(rename = "type")]
    pub r#type: ResponseToolType,
    // MCP-specific fields (used when type == "mcp")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authorization: Option<String>,
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
            server_url: None,
            authorization: None,
            server_label: None,
            server_description: None,
            require_approval: None,
            allowed_tools: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseToolType {
    WebSearchPreview,
    CodeInterpreter,
    Mcp,
}

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
        #[serde(skip_serializing_if = "Vec::is_empty")]
        summary: Vec<String>,
        content: Vec<ResponseReasoningContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "function_tool_call")]
    FunctionToolCall {
        id: String,
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        output: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseContentPart {
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
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
        #[serde(skip_serializing_if = "Vec::is_empty")]
        summary: Vec<String>,
        content: Vec<ResponseReasoningContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "function_tool_call")]
    FunctionToolCall {
        id: String,
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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
}

impl Default for ServiceTier {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Truncation {
    Auto,
    Disabled,
}

impl Default for Truncation {
    fn default() -> Self {
        Self::Disabled
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseTextFormat {
    pub format: TextFormatType,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TextFormatType {
    #[serde(rename = "type")]
    pub format_type: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokenUsageInfo>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromptTokenUsageInfo {
    pub cached_tokens: u32,
}

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

fn generate_request_id() -> String {
    format!("resp_{}", uuid::Uuid::new_v4().simple())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesRequest {
    /// Run the request in the background
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,

    /// Fields to include in the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeField>>,

    /// Input content - can be string or structured items
    pub input: ResponseInput,

    /// System instructions for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Maximum number of output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    /// Maximum number of tool calls
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,

    /// Additional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,

    /// Model to use (optional to match vLLM)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Optional conversation id to persist input/output as items
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Tool choice behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Available tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponseTool>>,

    /// Number of top logprobs to return
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    /// Top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Truncation behavior
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,

    /// User identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Request ID
    #[serde(default = "generate_request_id")]
    pub request_id: String,

    /// Request priority
    #[serde(default)]
    pub priority: i32,

    /// Frequency penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrArray>,

    /// Top-k sampling parameter
    #[serde(default = "default_top_k")]
    pub top_k: i32,

    /// Min-p sampling parameter
    #[serde(default)]
    pub min_p: f32,

    /// Repetition penalty
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResponseInput {
    Text(String),
    Items(Vec<ResponseInputOutputItem>),
}

fn default_top_k() -> i32 {
    -1
}

fn default_repetition_penalty() -> f32 {
    1.0
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
            model: None,
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
            user: None,
            request_id: generate_request_id(),
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

impl ResponsesRequest {
    /// Default sampling parameters
    const DEFAULT_TEMPERATURE: f32 = 0.7;
    const DEFAULT_TOP_P: f32 = 1.0;

    /// Convert to sampling parameters for generation
    pub fn to_sampling_params(
        &self,
        default_max_tokens: u32,
        default_params: Option<HashMap<String, Value>>,
    ) -> HashMap<String, Value> {
        let mut params = HashMap::new();

        // Use max_output_tokens if available
        let max_tokens = if let Some(max_output) = self.max_output_tokens {
            std::cmp::min(max_output, default_max_tokens)
        } else {
            default_max_tokens
        };

        // Avoid exceeding context length by minus 1 token
        let max_tokens = max_tokens.saturating_sub(1);

        // Temperature
        let temperature = self.temperature.unwrap_or_else(|| {
            default_params
                .as_ref()
                .and_then(|p| p.get("temperature"))
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(Self::DEFAULT_TEMPERATURE)
        });

        // Top-p
        let top_p = self.top_p.unwrap_or_else(|| {
            default_params
                .as_ref()
                .and_then(|p| p.get("top_p"))
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(Self::DEFAULT_TOP_P)
        });

        params.insert(
            "max_new_tokens".to_string(),
            Value::Number(Number::from(max_tokens)),
        );
        params.insert(
            "temperature".to_string(),
            Value::Number(Number::from_f64(temperature as f64).unwrap()),
        );
        params.insert(
            "top_p".to_string(),
            Value::Number(Number::from_f64(top_p as f64).unwrap()),
        );
        if let Some(fp) = self.frequency_penalty {
            params.insert(
                "frequency_penalty".to_string(),
                Value::Number(Number::from_f64(fp as f64).unwrap()),
            );
        }
        if let Some(pp) = self.presence_penalty {
            params.insert(
                "presence_penalty".to_string(),
                Value::Number(Number::from_f64(pp as f64).unwrap()),
            );
        }
        params.insert("top_k".to_string(), Value::Number(Number::from(self.top_k)));
        params.insert(
            "min_p".to_string(),
            Value::Number(Number::from_f64(self.min_p as f64).unwrap()),
        );
        params.insert(
            "repetition_penalty".to_string(),
            Value::Number(Number::from_f64(self.repetition_penalty as f64).unwrap()),
        );

        if let Some(ref stop) = self.stop {
            match to_value(stop) {
                Ok(value) => params.insert("stop".to_string(), value),
                Err(_) => params.insert("stop".to_string(), Value::Null),
            };
        }

        // Apply any additional default parameters
        if let Some(default_params) = default_params {
            for (key, value) in default_params {
                params.entry(key).or_insert(value);
            }
        }

        params
    }
}

impl GenerationRequest for ResponsesRequest {
    fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    fn get_model(&self) -> Option<&str> {
        self.model.as_deref()
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
                })
                .collect::<Vec<String>>()
                .join(" "),
        }
    }
}

fn generate_response_id() -> String {
    format!("resp_{}", uuid::Uuid::new_v4().simple())
}

fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_secs() as i64
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesResponse {
    /// Response ID
    #[serde(default = "generate_response_id")]
    pub id: String,

    /// Object type
    #[serde(default = "default_object_type")]
    pub object: String,

    /// Creation timestamp
    #[serde(default = "current_timestamp")]
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
    pub text: Option<ResponseTextFormat>,

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
    /// Create a response from a request
    #[allow(clippy::too_many_arguments)]
    pub fn from_request(
        request: &ResponsesRequest,
        _sampling_params: &HashMap<String, Value>,
        model_name: String,
        created_time: i64,
        output: Vec<ResponseOutputItem>,
        status: ResponseStatus,
        usage: Option<UsageInfo>,
    ) -> Self {
        Self {
            id: request.request_id.clone(),
            object: "response".to_string(),
            created_at: created_time,
            status,
            error: None,
            incomplete_details: None,
            instructions: request.instructions.clone(),
            max_output_tokens: request.max_output_tokens,
            model: model_name,
            output,
            parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
            previous_response_id: request.previous_response_id.clone(),
            reasoning: request.reasoning.as_ref().map(|r| ReasoningInfo {
                effort: r.effort.as_ref().map(|e| format!("{:?}", e)),
                summary: None,
            }),
            store: request.store.unwrap_or(false),
            temperature: request.temperature,
            text: Some(ResponseTextFormat {
                format: TextFormatType {
                    format_type: "text".to_string(),
                },
            }),
            tool_choice: match &request.tool_choice {
                Some(ToolChoice::Value(ToolChoiceValue::Auto)) => "auto".to_string(),
                Some(ToolChoice::Value(ToolChoiceValue::Required)) => "required".to_string(),
                Some(ToolChoice::Value(ToolChoiceValue::None)) => "none".to_string(),
                Some(ToolChoice::Function { .. }) => "function".to_string(),
                Some(ToolChoice::AllowedTools { mode, .. }) => mode.clone(),
                None => "auto".to_string(),
            },
            tools: request.tools.clone().unwrap_or_default(),
            top_p: request.top_p,
            truncation: match &request.truncation {
                Some(Truncation::Auto) => Some("auto".to_string()),
                Some(Truncation::Disabled) => Some("disabled".to_string()),
                None => None,
            },
            usage: usage.map(ResponsesUsage::Classic),
            user: request.user.clone(),
            metadata: request.metadata.clone().unwrap_or_default(),
        }
    }

    /// Create a new response with default values
    pub fn new(request_id: String, model: String, status: ResponseStatus) -> Self {
        Self {
            id: request_id,
            object: "response".to_string(),
            created_at: current_timestamp(),
            status,
            error: None,
            incomplete_details: None,
            instructions: None,
            max_output_tokens: None,
            model,
            output: Vec::new(),
            parallel_tool_calls: true,
            previous_response_id: None,
            reasoning: None,
            store: true,
            temperature: None,
            text: None,
            tool_choice: "auto".to_string(),
            tools: Vec::new(),
            top_p: None,
            truncation: None,
            usage: None,
            user: None,
            metadata: HashMap::new(),
        }
    }

    /// Add an output item to the response
    pub fn add_output(&mut self, item: ResponseOutputItem) {
        self.output.push(item);
    }

    /// Set the usage information
    pub fn set_usage(&mut self, usage: UsageInfo) {
        self.usage = Some(ResponsesUsage::Classic(usage));
    }

    /// Update the status
    pub fn set_status(&mut self, status: ResponseStatus) {
        self.status = status;
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

    /// Check if the response was cancelled
    pub fn is_cancelled(&self) -> bool {
        matches!(self.status, ResponseStatus::Cancelled)
    }

    /// Check if the response is queued
    pub fn is_queued(&self) -> bool {
        matches!(self.status, ResponseStatus::Queued)
    }

    /// Convert usage to OpenAI Responses API format
    pub fn usage_in_response_format(&self) -> Option<ResponseUsage> {
        self.usage.as_ref().map(|usage| usage.to_response_usage())
    }

    /// Get the response as a JSON value with usage in response format
    pub fn to_response_format(&self) -> Value {
        let mut response = to_value(self).unwrap_or(Value::Null);

        // Convert usage to response format if present
        if let Some(usage) = &self.usage {
            if let Ok(usage_value) = to_value(usage.to_response_usage()) {
                response["usage"] = usage_value;
            }
        }

        response
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
        name: String,
        arguments: String,
        output: Option<String>,
        status: String,
    ) -> Self {
        Self::FunctionToolCall {
            id,
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

impl UsageInfo {
    /// Create a new usage info with token counts
    pub fn new(prompt_tokens: u32, completion_tokens: u32, reasoning_tokens: Option<u32>) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            reasoning_tokens,
            prompt_tokens_details: None,
        }
    }

    /// Create usage info with cached token details
    pub fn new_with_cached(
        prompt_tokens: u32,
        completion_tokens: u32,
        reasoning_tokens: Option<u32>,
        cached_tokens: u32,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            reasoning_tokens,
            prompt_tokens_details: Some(PromptTokenUsageInfo { cached_tokens }),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// Tool choice value for simple string options
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceValue {
    Auto,
    Required,
    None,
}

/// Tool choice for both Chat Completion and Responses APIs
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Value(ToolChoiceValue),
    Function {
        #[serde(rename = "type")]
        tool_type: String, // "function"
        function: FunctionChoice,
    },
    AllowedTools {
        #[serde(rename = "type")]
        tool_type: String, // "allowed_tools"
        mode: String, // "auto" | "required" TODO: need validation
        tools: Vec<ToolReference>,
    },
}

impl Default for ToolChoice {
    fn default() -> Self {
        Self::Value(ToolChoiceValue::Auto)
    }
}

/// Function choice specification for ToolChoice::Function
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionChoice {
    pub name: String,
}

/// Tool reference for ToolChoice::AllowedTools
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolReference {
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: Function,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value, // JSON Schema
    /// Whether to enable strict schema adherence (OpenAI structured outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: FunctionCallResponse,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum FunctionCall {
    None,
    Auto,
    Function { name: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCallResponse {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<String>, // JSON string
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LogProbs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<Option<HashMap<String, f32>>>,
    pub text_offset: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ChatLogProbs {
    Detailed {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<Vec<ChatLogProbsContent>>,
    },
    Raw(Value),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatLogProbsContent {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogProb>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TopLogProb {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InputIds {
    Single(Vec<i32>),
    Batch(Vec<Vec<i32>>),
}

#[derive(Debug, Clone, Deserialize, Serialize, Default, Validate)]
#[validate(schema(function = "validate_sampling_params"))]
pub struct SamplingParams {
    /// Temperature for sampling (must be >= 0.0, no upper limit)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0.0))]
    pub temperature: Option<f32>,
    /// Maximum number of new tokens to generate (must be >= 0)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0))]
    pub max_new_tokens: Option<u32>,
    /// Top-p nucleus sampling (0.0 < top_p <= 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_top_p_value"))]
    pub top_p: Option<f32>,
    /// Top-k sampling (-1 to disable, or >= 1)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(custom(function = "validate_top_k_value"))]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = -2.0, max = 2.0))]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = -2.0, max = 2.0))]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0.0, max = 2.0))]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrArray>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_special_tokens: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ebnf: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_p: Option<f32>,
    /// Minimum number of new tokens (validated in schema function for cross-field check with max_new_tokens)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_stop_trim: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_seed: Option<u64>,
}

/// Validation function for SamplingParams - cross-field validation only
fn validate_sampling_params(params: &SamplingParams) -> Result<(), validator::ValidationError> {
    // 1. Cross-field validation: min_new_tokens <= max_new_tokens
    if let (Some(min), Some(max)) = (params.min_new_tokens, params.max_new_tokens) {
        if min > max {
            return Err(validator::ValidationError::new(
                "min_new_tokens cannot exceed max_new_tokens",
            ));
        }
    }

    // 2. Validate mutually exclusive structured output constraints
    let constraint_count = [
        params.regex.is_some(),
        params.ebnf.is_some(),
        params.json_schema.is_some(),
    ]
    .iter()
    .filter(|&&x| x)
    .count();

    if constraint_count > 1 {
        return Err(validator::ValidationError::new(
            "only one of regex, ebnf, or json_schema can be set",
        ));
    }

    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize, Validate)]
#[validate(schema(function = "validate_generate_request"))]
pub struct GenerateRequest {
    /// Text input - SGLang native format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Input IDs for tokenized input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_ids: Option<InputIds>,

    /// Input embeddings for direct embedding input
    /// Can be a 2D array (single request) or 3D array (batch of requests)
    /// Placeholder for future use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_embeds: Option<Value>,

    /// Image input data
    /// Can be an image instance, file name, URL, or base64 encoded string
    /// Supports single images, lists of images, or nested lists for batch processing
    /// Placeholder for future use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_data: Option<Value>,

    /// Video input data
    /// Can be a file name, URL, or base64 encoded string
    /// Supports single videos, lists of videos, or nested lists for batch processing
    /// Placeholder for future use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub video_data: Option<Value>,

    /// Audio input data
    /// Can be a file name, URL, or base64 encoded string
    /// Supports single audio files, lists of audio, or nested lists for batch processing
    /// Placeholder for future use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_data: Option<Value>,

    /// Sampling parameters (sglang style)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<SamplingParams>,

    /// Whether to return logprobs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_logprob: Option<bool>,

    /// If return logprobs, the start location in the prompt for returning logprobs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprob_start_len: Option<i32>,

    /// If return logprobs, the number of top logprobs to return at each position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs_num: Option<i32>,

    /// If return logprobs, the token ids to return logprob for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids_logprob: Option<Vec<u32>>,

    /// Whether to detokenize tokens in text in the returned logprobs.
    #[serde(default)]
    pub return_text_in_logprobs: bool,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Whether to log metrics for this request (e.g. health_generate calls do not log metrics)
    #[serde(default = "default_true")]
    pub log_metrics: bool,

    /// Return model hidden states
    #[serde(default)]
    pub return_hidden_states: bool,

    /// The modalities of the image data [image, multi-images, video]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,

    /// Session parameters for continual prompting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_params: Option<HashMap<String, Value>>,

    /// Path to LoRA adapter(s) for model customization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<String>,

    /// LoRA adapter ID (if pre-loaded)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_id: Option<String>,

    /// Custom logit processor for advanced sampling control. Must be a serialized instance
    /// of `CustomLogitProcessor` in python/sglang/srt/sampling/custom_logit_processor.py
    /// Use the processor's `to_str()` method to generate the serialized string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_logit_processor: Option<String>,

    /// For disaggregated inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_host: Option<String>,

    /// For disaggregated inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_port: Option<i32>,

    /// For disaggregated inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_room: Option<i32>,

    /// For disaggregated inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap_pair_key: Option<String>,

    /// Data parallel rank routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_parallel_rank: Option<i32>,

    /// Background response
    #[serde(default)]
    pub background: bool,

    /// Conversation ID for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,

    /// Priority for the request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,

    /// Extra key for classifying the request (e.g. cache_salt)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_key: Option<String>,

    /// Whether to disallow logging for this request (e.g. due to ZDR)
    #[serde(default)]
    pub no_logs: bool,

    /// Custom metric labels
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_labels: Option<HashMap<String, String>>,

    /// Whether to return bytes for image generation
    #[serde(default)]
    pub return_bytes: bool,

    /// Whether to return entropy
    #[serde(default)]
    pub return_entropy: bool,

    /// Request ID for tracking (inherited from BaseReq in Python)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rid: Option<String>,
}

impl Normalizable for GenerateRequest {
    // Use default no-op implementation - no normalization needed for GenerateRequest
}

/// Validation function for GenerateRequest - ensure exactly one input type is provided
fn validate_generate_request(req: &GenerateRequest) -> Result<(), validator::ValidationError> {
    // Exactly one of text or input_ids must be provided
    // Note: input_embeds not yet supported in Rust implementation
    let has_text = req.text.is_some();
    let has_input_ids = req.input_ids.is_some();

    let count = [has_text, has_input_ids].iter().filter(|&&x| x).count();

    if count == 0 {
        return Err(validator::ValidationError::new(
            "Either text or input_ids should be provided.",
        ));
    }

    if count > 1 {
        return Err(validator::ValidationError::new(
            "Either text or input_ids should be provided.",
        ));
    }

    Ok(())
}

impl GenerationRequest for GenerateRequest {
    fn is_stream(&self) -> bool {
        self.stream
    }

    fn get_model(&self) -> Option<&str> {
        // Generate requests typically don't have a model field
        None
    }

    fn extract_text_for_routing(&self) -> String {
        // Check fields in priority order: text, input_ids
        if let Some(ref text) = self.text {
            return text.clone();
        }

        if let Some(ref input_ids) = self.input_ids {
            return match input_ids {
                InputIds::Single(ids) => ids
                    .iter()
                    .map(|&id| id.to_string())
                    .collect::<Vec<String>>()
                    .join(" "),
                InputIds::Batch(batches) => batches
                    .iter()
                    .flat_map(|batch| batch.iter().map(|&id| id.to_string()))
                    .collect::<Vec<String>>()
                    .join(" "),
            };
        }

        // No text input found
        String::new()
    }
}

// ============================================================================
// SGLang Generate Response Types
// ============================================================================

/// SGLang generate response (single completion or array for n>1)
///
/// Format for n=1:
/// ```json
/// {
///   "text": "...",
///   "output_ids": [...],
///   "meta_info": { ... }
/// }
/// ```
///
/// Format for n>1:
/// ```json
/// [
///   {"text": "...", "output_ids": [...], "meta_info": {...}},
///   {"text": "...", "output_ids": [...], "meta_info": {...}}
/// ]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub text: String,
    pub output_ids: Vec<u32>,
    pub meta_info: GenerateMetaInfo,
}

/// Metadata for a single generate completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateMetaInfo {
    pub id: String,
    pub finish_reason: GenerateFinishReason,
    pub prompt_tokens: u32,
    pub weight_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_token_logprobs: Option<Vec<Vec<Option<f64>>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_token_logprobs: Option<Vec<Vec<Option<f64>>>>,
    pub completion_tokens: u32,
    pub cached_tokens: u32,
    pub e2e_latency: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop: Option<Value>,
}

/// Finish reason for generate endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum GenerateFinishReason {
    Length {
        length: u32,
    },
    Stop,
    #[serde(untagged)]
    Other(Value),
}

// Constants for rerank API
pub const DEFAULT_MODEL_NAME: &str = "default";

/// Rerank request for scoring documents against a query
/// Used for RAG systems and document relevance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankRequest {
    /// The query text to rank documents against
    pub query: String,

    /// List of documents to be ranked
    pub documents: Vec<String>,

    /// Model to use for reranking
    #[serde(default = "default_model_name")]
    pub model: String,

    /// Maximum number of documents to return (optional)
    pub top_k: Option<usize>,

    /// Whether to return documents in addition to scores
    #[serde(default = "default_return_documents")]
    pub return_documents: bool,

    // SGLang specific extensions
    /// Request ID for tracking
    pub rid: Option<StringOrArray>,

    /// User identifier
    pub user: Option<String>,
}

pub fn default_model_name() -> String {
    DEFAULT_MODEL_NAME.to_string()
}

fn default_return_documents() -> bool {
    true
}

/// Individual rerank result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    /// Relevance score for the document
    pub score: f32,

    /// The document text (if return_documents was true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,

    /// Original index of the document in the request
    pub index: usize,

    /// Additional metadata about the ranking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta_info: Option<HashMap<String, Value>>,
}

/// Rerank response containing sorted results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
    /// Ranked results sorted by score (highest first)
    pub results: Vec<RerankResult>,

    /// Model used for reranking
    pub model: String,

    /// Usage information
    pub usage: Option<UsageInfo>,

    /// Response object type
    #[serde(default = "default_rerank_object")]
    pub object: String,

    /// Response ID
    pub id: Option<StringOrArray>,

    /// Creation timestamp
    pub created: i64,
}

fn default_rerank_object() -> String {
    "rerank".to_string()
}

/// V1 API compatibility format for rerank requests
/// Matches Python's V1RerankReqInput
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V1RerankReqInput {
    pub query: String,
    pub documents: Vec<String>,
}

/// Convert V1RerankReqInput to RerankRequest
impl From<V1RerankReqInput> for RerankRequest {
    fn from(v1: V1RerankReqInput) -> Self {
        RerankRequest {
            query: v1.query,
            documents: v1.documents,
            model: default_model_name(),
            top_k: None,
            return_documents: true,
            rid: None,
            user: None,
        }
    }
}

/// Implementation of GenerationRequest trait for RerankRequest
impl GenerationRequest for RerankRequest {
    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn is_stream(&self) -> bool {
        false // Reranking doesn't support streaming
    }

    fn extract_text_for_routing(&self) -> String {
        self.query.clone()
    }
}

impl RerankRequest {
    pub fn validate(&self) -> Result<(), String> {
        // Validate query is not empty
        if self.query.trim().is_empty() {
            return Err("Query cannot be empty".to_string());
        }

        // Validate documents list
        if self.documents.is_empty() {
            return Err("Documents list cannot be empty".to_string());
        }

        // Validate top_k if specified
        if let Some(k) = self.top_k {
            if k == 0 {
                return Err("top_k must be greater than 0".to_string());
            }
            if k > self.documents.len() {
                // This is allowed but we log a warning
                tracing::warn!(
                    "top_k ({}) is greater than number of documents ({})",
                    k,
                    self.documents.len()
                );
            }
        }

        Ok(())
    }

    /// Get the effective top_k value
    pub fn effective_top_k(&self) -> usize {
        self.top_k.unwrap_or(self.documents.len())
    }
}

impl RerankResponse {
    pub fn new(
        results: Vec<RerankResult>,
        model: String,
        request_id: Option<StringOrArray>,
    ) -> Self {
        RerankResponse {
            results,
            model,
            usage: None,
            object: default_rerank_object(),
            id: request_id,
            created: current_timestamp(),
        }
    }

    /// Sort results by score in descending order
    pub fn sort_by_score(&mut self) {
        self.results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Apply top_k limit to results
    pub fn apply_top_k(&mut self, k: usize) {
        self.results.truncate(k);
    }

    /// Drop documents from results
    pub fn drop_documents(&mut self) {
        self.results.iter_mut().for_each(|result| {
            result.document = None;
        });
    }
}

/// Embeddings request compatible with OpenAI API
/// We intentionally keep fields flexible to pass through to workers.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingRequest {
    /// ID of the model to use
    pub model: String,

    /// Input can be a string, array of strings, tokens, or batch inputs
    pub input: Value,

    /// Optional encoding format (e.g., "float", "base64")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,

    /// Optional user identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Optional number of dimensions for the embedding
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,

    /// SGLang extension: request id for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rid: Option<String>,
}

impl GenerationRequest for EmbeddingRequest {
    fn is_stream(&self) -> bool {
        // Embeddings are non-streaming
        false
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
        // Best effort: extract text content for routing decisions
        match &self.input {
            Value::String(s) => s.clone(),
            Value::Array(arr) => arr
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            _ => String::new(),
        }
    }
}

/// Helper function for serde default value
pub fn default_true() -> bool {
    true
}

/// Common trait for all generation requests across different APIs
pub trait GenerationRequest: Send + Sync {
    /// Check if the request is for streaming
    fn is_stream(&self) -> bool;

    /// Get the model name if specified
    fn get_model(&self) -> Option<&str>;

    /// Extract text content for routing decisions
    fn extract_text_for_routing(&self) -> String;
}

/// Helper type for string or array of strings
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StringOrArray {
    String(String),
    Array(Vec<String>),
}
impl StringOrArray {
    /// Get the number of items in the StringOrArray
    pub fn len(&self) -> usize {
        match self {
            StringOrArray::String(_) => 1,
            StringOrArray::Array(arr) => arr.len(),
        }
    }

    /// Check if the StringOrArray is empty
    pub fn is_empty(&self) -> bool {
        match self {
            StringOrArray::String(s) => s.is_empty(),
            StringOrArray::Array(arr) => arr.is_empty(),
        }
    }

    /// Convert to a vector of strings
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            StringOrArray::String(s) => vec![s.clone()],
            StringOrArray::Array(arr) => arr.clone(),
        }
    }
}

/// LoRA adapter path - can be single path or batch of paths (SGLang extension)
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum LoRAPath {
    Single(Option<String>),
    Batch(Vec<Option<String>>),
}
