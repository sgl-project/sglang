use std::collections::HashMap;

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use validator;

// ============================================================================
// Default value helpers
// ============================================================================

/// Default model value when not specified
pub(crate) fn default_model() -> String {
    "unknown".to_string()
}

/// Helper function for serde default value (returns true)
pub fn default_true() -> bool {
    true
}

/// Environment variable to force ignore_eos to true for all requests.
/// WARNING: This is for research purposes only. Do not use in production.
static FORCE_IGNORE_EOS: Lazy<bool> = Lazy::new(|| {
    std::env::var("RESEARCH_ONLY_FORCE_IGNORE_EOS")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false)
});

/// Default value for ignore_eos field.
/// Returns true if RESEARCH_ONLY_FORCE_IGNORE_EOS environment variable is set,
/// otherwise returns false.
pub fn default_ignore_eos() -> bool {
    *FORCE_IGNORE_EOS
}

/// Default value for ignore_eos field when it's an Option<bool>.
/// Returns Some(true) if RESEARCH_ONLY_FORCE_IGNORE_EOS environment variable is set,
/// otherwise returns None.
pub fn default_ignore_eos_option() -> Option<bool> {
    if *FORCE_IGNORE_EOS {
        Some(true)
    } else {
        None
    }
}

// ============================================================================
// GenerationRequest Trait
// ============================================================================

/// Trait for unified access to generation request properties
/// Implemented by ChatCompletionRequest, CompletionRequest, GenerateRequest,
/// EmbeddingRequest, RerankRequest, and ResponsesRequest
pub trait GenerationRequest: Send + Sync {
    /// Check if the request is for streaming
    fn is_stream(&self) -> bool;

    /// Get the model name if specified
    fn get_model(&self) -> Option<&str>;

    /// Extract text content for routing decisions
    fn extract_text_for_routing(&self) -> String;
}

// ============================================================================
// String/Array Utilities
// ============================================================================

/// A type that can be either a single string or an array of strings
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
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

/// Validates stop sequences (max 4, non-empty strings)
/// Used by both ChatCompletionRequest and ResponsesRequest
pub fn validate_stop(stop: &StringOrArray) -> Result<(), validator::ValidationError> {
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

// ============================================================================
// Content Parts (for multimodal messages)
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>, // "auto", "low", or "high"
}

// ============================================================================
// Response Format (for structured outputs)
// ============================================================================

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

// ============================================================================
// Streaming
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
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

// ============================================================================
// Tools and Function Calling
// ============================================================================

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

impl ToolChoice {
    /// Serialize tool_choice to string for ResponsesResponse
    ///
    /// Returns the JSON-serialized tool_choice or "auto" as default
    pub fn serialize_to_string(tool_choice: &Option<ToolChoice>) -> String {
        tool_choice
            .as_ref()
            .map(|tc| serde_json::to_string(tc).unwrap_or_else(|_| "auto".to_string()))
            .unwrap_or_else(|| "auto".to_string())
    }
}

/// Function choice specification for ToolChoice::Function
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionChoice {
    pub name: String,
}

/// Tool reference for ToolChoice::AllowedTools
///
/// Represents a reference to a specific tool in the allowed_tools array.
/// Different tool types have different required fields.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ToolReference {
    /// Reference to a function tool
    #[serde(rename = "function")]
    Function { name: String },

    /// Reference to an MCP tool
    #[serde(rename = "mcp")]
    Mcp {
        server_label: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },

    /// File search hosted tool
    #[serde(rename = "file_search")]
    FileSearch,

    /// Web search preview hosted tool
    #[serde(rename = "web_search_preview")]
    WebSearchPreview,

    /// Computer use preview hosted tool
    #[serde(rename = "computer_use_preview")]
    ComputerUsePreview,

    /// Code interpreter hosted tool
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,

    /// Image generation hosted tool
    #[serde(rename = "image_generation")]
    ImageGeneration,
}

impl ToolReference {
    /// Get a unique identifier for this tool reference
    pub fn identifier(&self) -> String {
        match self {
            ToolReference::Function { name } => format!("function:{}", name),
            ToolReference::Mcp { server_label, name } => {
                if let Some(n) = name {
                    format!("mcp:{}:{}", server_label, n)
                } else {
                    format!("mcp:{}", server_label)
                }
            }
            ToolReference::FileSearch => "file_search".to_string(),
            ToolReference::WebSearchPreview => "web_search_preview".to_string(),
            ToolReference::ComputerUsePreview => "computer_use_preview".to_string(),
            ToolReference::CodeInterpreter => "code_interpreter".to_string(),
            ToolReference::ImageGeneration => "image_generation".to_string(),
        }
    }

    /// Get the tool name if this is a function tool
    pub fn function_name(&self) -> Option<&str> {
        match self {
            ToolReference::Function { name } => Some(name.as_str()),
            _ => None,
        }
    }
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

// ============================================================================
// Usage and Logging
// ============================================================================

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

/// Usage information (used by rerank and other endpoints)
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

// ============================================================================
// Error Types
// ============================================================================

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

// ============================================================================
// Input Types
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InputIds {
    Single(Vec<i32>),
    Batch(Vec<Vec<i32>>),
}

/// LoRA adapter path - can be single path or batch of paths (SGLang extension)
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum LoRAPath {
    Single(Option<String>),
    Batch(Vec<Option<String>>),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test struct to verify ignore_eos default behavior
    #[derive(Debug, Deserialize)]
    struct TestIgnoreEos {
        #[serde(default = "default_ignore_eos")]
        ignore_eos: bool,
    }

    /// Test struct for Option<bool> variant
    #[derive(Debug, Deserialize)]
    struct TestIgnoreEosOption {
        #[serde(default = "default_ignore_eos_option")]
        ignore_eos: Option<bool>,
    }

    #[test]
    fn test_ignore_eos_default_without_env() {
        // When RESEARCH_ONLY_FORCE_IGNORE_EOS is not set (or false), default should be false
        // Note: Since Lazy is initialized once, this test checks the current state
        let json = r#"{}"#;
        let parsed: TestIgnoreEos = serde_json::from_str(json).unwrap();
        // The default value depends on whether RESEARCH_ONLY_FORCE_IGNORE_EOS env var is set
        // In normal test runs, it should be false
        assert!(
            !parsed.ignore_eos || std::env::var("RESEARCH_ONLY_FORCE_IGNORE_EOS").is_ok(),
            "ignore_eos should be false unless RESEARCH_ONLY_FORCE_IGNORE_EOS is set"
        );
    }

    #[test]
    fn test_ignore_eos_explicit_value_overrides_default() {
        // When ignore_eos is explicitly set in JSON, it should override the default
        let json = r#"{"ignore_eos": true}"#;
        let parsed: TestIgnoreEos = serde_json::from_str(json).unwrap();
        assert!(parsed.ignore_eos);

        let json = r#"{"ignore_eos": false}"#;
        let parsed: TestIgnoreEos = serde_json::from_str(json).unwrap();
        assert!(!parsed.ignore_eos);
    }

    #[test]
    fn test_ignore_eos_option_explicit_value() {
        // When ignore_eos is explicitly set in JSON, it should be Some(value)
        let json = r#"{"ignore_eos": true}"#;
        let parsed: TestIgnoreEosOption = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.ignore_eos, Some(true));

        let json = r#"{"ignore_eos": false}"#;
        let parsed: TestIgnoreEosOption = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.ignore_eos, Some(false));
    }
}
