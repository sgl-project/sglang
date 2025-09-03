use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// # Protocol Specifications
//
// This module contains all protocol definitions for OpenAI and SGLang APIs.
//
// ## Table of Contents
//
// 1. **OPENAI SPEC - Chat Completions API**
//    - Message Types
//    - Response Format Types
//    - Tool/Function Types
//    - Streaming Delta Types
//    - Request/Response structures
//
// 2. **OPENAI SPEC - Completions API**
//    - Request/Response structures
//    - Streaming support
//
// 3. **OPENAI SPEC - Responses API**
//    - Tool Definitions
//    - Reasoning Configuration
//    - Input/Output Items
//    - Service Tier & Tool Choice
//    - Request/Response structures
//
// 4. **OPENAI SPEC - Common**
//    - Shared Request Components
//    - Tool Choice Types
//    - Usage Tracking
//    - Logprobs Types
//    - Error Response Types
//
// 5. **SGLANG SPEC - GENERATE API**
//    - Generate Parameters
//    - Sampling Parameters
//    - Request/Response structures
//
// 6. **SGLANG SPEC - RERANK API**
//    - Request/Response structures
//
// 7. **COMMON**
//    - GenerationRequest trait
//    - StringOrArray & LoRAPath types
//    - Helper functions

// ==================================================================
// =            OPENAI SPEC - Chat Completions API                  =
// ==================================================================

// ============= Message Types =============

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ChatMessage {
    System {
        role: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        role: String, // "user"
        content: UserMessageContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        role: String, // "assistant"
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        function_call: Option<FunctionCallResponse>,
        /// Reasoning content for O1-style models (SGLang extension)
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
    },
    Tool {
        role: String, // "tool"
        content: String,
        tool_call_id: String,
    },
    Function {
        role: String, // "function"
        content: String,
        name: String,
    },
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

// ============= Response Format Types =============

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

// ============= Streaming Delta Types =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCallDelta>,
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

// ============= Request =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    /// ID of the model to use
    pub model: String,

    /// A list of messages comprising the conversation so far
    pub messages: Vec<ChatMessage>,

    /// What sampling temperature to use, between 0 and 2
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// An alternative to sampling with temperature
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// How many chat completion choices to generate for each input message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,

    /// If set, partial message deltas will be sent
    #[serde(default)]
    pub stream: bool,

    /// Options for streaming response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    /// Up to 4 sequences where the API will stop generating further tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrArray>,

    /// The maximum number of tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// An upper bound for the number of tokens that can be generated for a completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Modify the likelihood of specified tokens appearing in the completion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,

    /// A unique identifier representing your end-user
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// If specified, our system will make a best effort to sample deterministically
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// Whether to return log probabilities of the output tokens
    #[serde(default)]
    pub logprobs: bool,

    /// An integer between 0 and 20 specifying the number of most likely tokens to return
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,

    /// An object specifying the format that the model must output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// A list of tools the model may call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// Controls which (if any) tool is called by the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Whether to enable parallel function calling during tool use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Deprecated: use tools instead
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<Function>>,

    /// Deprecated: use tool_choice instead
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,

    // ============= SGLang Extensions =============
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

    /// Specific token IDs to use as stop conditions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<i32>>,

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

    // ============= SGLang Extensions =============
    /// Path to LoRA adapter(s) for model customization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<LoRAPath>,

    /// Session parameters for continual prompting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_params: Option<HashMap<String, serde_json::Value>>,

    /// Separate reasoning content from final answer (O1-style models)
    #[serde(default = "default_true")]
    pub separate_reasoning: bool,

    /// Stream reasoning tokens during generation
    #[serde(default = "default_true")]
    pub stream_reasoning: bool,

    /// Chat template kwargs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<HashMap<String, serde_json::Value>>,

    /// Return model hidden states
    #[serde(default)]
    pub return_hidden_states: bool,
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

// ============= Regular Response =============

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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogProbs>,
    pub finish_reason: Option<String>, // "stop", "length", "tool_calls", "content_filter", "function_call"
    /// Information about which stop condition was matched
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_stop: Option<serde_json::Value>, // Can be string or integer
    /// Hidden states from the model (SGLang extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_states: Option<Vec<f32>>,
}

// ============= Streaming Response =============

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
}

// ==================================================================
// =            OPENAI SPEC - Completions API                       =
// ==================================================================
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

    // ============= SGLang Extensions =============
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
    pub stop_token_ids: Option<Vec<i32>>,

    /// Skip trimming stop tokens from output
    #[serde(default)]
    pub no_stop_trim: bool,

    /// Ignore end-of-sequence tokens during generation
    #[serde(default)]
    pub ignore_eos: bool,

    /// Skip special tokens during detokenization
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,

    // ============= SGLang Extensions =============
    /// Path to LoRA adapter(s) for model customization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<LoRAPath>,

    /// Session parameters for continual prompting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_params: Option<HashMap<String, serde_json::Value>>,

    /// Return model hidden states
    #[serde(default)]
    pub return_hidden_states: bool,

    /// Additional fields including bootstrap info for PD routing
    #[serde(flatten)]
    pub other: serde_json::Map<String, serde_json::Value>,
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

// ============= Regular Response =============

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
    pub matched_stop: Option<serde_json::Value>, // Can be string or integer
    /// Hidden states from the model (SGLang extension)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_states: Option<Vec<f32>>,
}

// ============= Streaming Response =============

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

// ==================================================================
// =            OPENAI SPEC - Responses API                         =
// ==================================================================

// ============= Tool Definitions =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseTool {
    #[serde(rename = "type")]
    pub r#type: ResponseToolType,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseToolType {
    WebSearchPreview,
    CodeInterpreter,
}

// ============= Reasoning Configuration =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseReasoningParam {
    #[serde(default = "default_reasoning_effort")]
    pub effort: Option<ReasoningEffort>,
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

// ============= Input/Output Items =============

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
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponseReasoningContent {
    #[serde(rename = "reasoning_text")]
    ReasoningText { text: String },
}

// ============= Output Items for Response =============

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
}

// ============= Service Tier =============

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

// ============= Truncation =============

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

// ============= Response Status =============

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

// ============= Include Fields =============

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

// ============= Usage Info =============

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

// ============= Response Usage Format =============

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

fn generate_request_id() -> String {
    format!("resp_{}", uuid::Uuid::new_v4().simple())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesRequest {
    // ============= Core OpenAI API fields =============
    /// Run the request in the background
    #[serde(default)]
    pub background: bool,

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
    pub metadata: Option<HashMap<String, serde_json::Value>>,

    /// Model to use (optional to match vLLM)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Whether to enable parallel tool calls
    #[serde(default = "default_true")]
    pub parallel_tool_calls: bool,

    /// ID of previous response to continue from
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Reasoning configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponseReasoningParam>,

    /// Service tier
    #[serde(default)]
    pub service_tier: ServiceTier,

    /// Whether to store the response
    #[serde(default = "default_true")]
    pub store: bool,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Tool choice behavior
    #[serde(default)]
    pub tool_choice: ToolChoice,

    /// Available tools
    #[serde(default)]
    pub tools: Vec<ResponseTool>,

    /// Number of top logprobs to return
    #[serde(default)]
    pub top_logprobs: u32,

    /// Top-p sampling parameter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Truncation behavior
    #[serde(default)]
    pub truncation: Truncation,

    /// User identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    // ============= SGLang Extensions =============
    /// Request ID
    #[serde(default = "generate_request_id")]
    pub request_id: String,

    /// Request priority
    #[serde(default)]
    pub priority: i32,

    /// Frequency penalty
    #[serde(default)]
    pub frequency_penalty: f32,

    /// Presence penalty
    #[serde(default)]
    pub presence_penalty: f32,

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

impl ResponsesRequest {
    /// Default sampling parameters
    const DEFAULT_TEMPERATURE: f32 = 0.7;
    const DEFAULT_TOP_P: f32 = 1.0;

    /// Convert to sampling parameters for generation
    pub fn to_sampling_params(
        &self,
        default_max_tokens: u32,
        default_params: Option<HashMap<String, serde_json::Value>>,
    ) -> HashMap<String, serde_json::Value> {
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
            serde_json::Value::Number(serde_json::Number::from(max_tokens)),
        );
        params.insert(
            "temperature".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(temperature as f64).unwrap()),
        );
        params.insert(
            "top_p".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(top_p as f64).unwrap()),
        );
        params.insert(
            "frequency_penalty".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(self.frequency_penalty as f64).unwrap(),
            ),
        );
        params.insert(
            "presence_penalty".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(self.presence_penalty as f64).unwrap(),
            ),
        );
        params.insert(
            "top_k".to_string(),
            serde_json::Value::Number(serde_json::Number::from(self.top_k)),
        );
        params.insert(
            "min_p".to_string(),
            serde_json::Value::Number(serde_json::Number::from_f64(self.min_p as f64).unwrap()),
        );
        params.insert(
            "repetition_penalty".to_string(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(self.repetition_penalty as f64).unwrap(),
            ),
        );

        if let Some(ref stop) = self.stop {
            match serde_json::to_value(stop) {
                Ok(value) => params.insert("stop".to_string(), value),
                Err(_) => params.insert("stop".to_string(), serde_json::Value::Null),
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
        self.stream
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
                            .map(|part| match part {
                                ResponseContentPart::OutputText { text, .. } => text.clone(),
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

    /// Model name
    pub model: String,

    /// Output items
    #[serde(default)]
    pub output: Vec<ResponseOutputItem>,

    /// Response status
    pub status: ResponseStatus,

    /// Usage information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageInfo>,

    /// Whether parallel tool calls are enabled
    #[serde(default = "default_true")]
    pub parallel_tool_calls: bool,

    /// Tool choice setting
    #[serde(default = "default_tool_choice")]
    pub tool_choice: String,

    /// Available tools
    #[serde(default)]
    pub tools: Vec<ResponseTool>,
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
        _sampling_params: &HashMap<String, serde_json::Value>,
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
            model: model_name,
            output,
            status,
            usage,
            parallel_tool_calls: request.parallel_tool_calls,
            tool_choice: match &request.tool_choice {
                ToolChoice::Value(ToolChoiceValue::Auto) => "auto".to_string(),
                ToolChoice::Value(ToolChoiceValue::Required) => "required".to_string(),
                ToolChoice::Value(ToolChoiceValue::None) => "none".to_string(),
                ToolChoice::Function { .. } => "function".to_string(),
            },
            tools: request.tools.clone(),
        }
    }

    /// Create a new response with default values
    pub fn new(request_id: String, model: String, status: ResponseStatus) -> Self {
        Self {
            id: request_id,
            object: "response".to_string(),
            created_at: current_timestamp(),
            model,
            output: Vec::new(),
            status,
            usage: None,
            parallel_tool_calls: true,
            tool_choice: "auto".to_string(),
            tools: Vec::new(),
        }
    }

    /// Add an output item to the response
    pub fn add_output(&mut self, item: ResponseOutputItem) {
        self.output.push(item);
    }

    /// Set the usage information
    pub fn set_usage(&mut self, usage: UsageInfo) {
        self.usage = Some(usage);
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
    pub fn to_response_format(&self) -> serde_json::Value {
        let mut response = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);

        // Convert usage to response format if present
        if let Some(usage) = &self.usage {
            if let Ok(usage_value) = serde_json::to_value(usage.to_response_usage()) {
                response["usage"] = usage_value;
            }
        }

        response
    }
}

// ============= Helper Functions =============

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

// ==================================================================
// =            OPENAI SPEC - Common                                =
// ==================================================================

// ============= Shared Request Components =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

// ============= Tool Choice Types =============

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

// ============= Usage Tracking =============

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

// ============= Logprobs Types =============

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LogProbs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<Option<HashMap<String, f32>>>,
    pub text_offset: Vec<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatLogProbs {
    pub content: Option<Vec<ChatLogProbsContent>>,
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

// ==================================================================
// =            SGLANG SPEC - GENERATE API                          =
// ==================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum InputIds {
    Single(Vec<i32>),
    Batch(Vec<Vec<i32>>),
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct GenerateParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_of: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoder_input_details: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub do_sample: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_full_text: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub watermark: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct SamplingParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
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
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_stop_trim: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// The prompt to generate from (OpenAI style)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<StringOrArray>,

    /// Text input - SGLang native format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Input IDs for tokenized input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_ids: Option<InputIds>,

    /// Generation parameters
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<GenerateParameters>,

    /// Sampling parameters (sglang style)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_params: Option<SamplingParams>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    /// Whether to return logprobs
    #[serde(default)]
    pub return_logprob: bool,

    // ============= SGLang Extensions =============
    /// Path to LoRA adapter(s) for model customization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<LoRAPath>,

    /// Session parameters for continual prompting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_params: Option<HashMap<String, serde_json::Value>>,

    /// Return model hidden states
    #[serde(default)]
    pub return_hidden_states: bool,

    /// Request ID for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rid: Option<String>,
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
        // Check fields in priority order: text, prompt, inputs
        if let Some(ref text) = self.text {
            return text.clone();
        }

        if let Some(ref prompt) = self.prompt {
            return match prompt {
                StringOrArray::String(s) => s.clone(),
                StringOrArray::Array(v) => v.join(" "),
            };
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

// ==================================================================
// =            SGLANG SPEC - RERANK API                            =
// ==================================================================

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

fn default_model_name() -> String {
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
    pub id: String,

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
    pub fn new(results: Vec<RerankResult>, model: String, request_id: String) -> Self {
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
}

// ==================================================================
// =            COMMON                                              =
// ==================================================================

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    // ==================================================================
    // =            RERANK REQUEST TESTS                                =
    // ==================================================================

    #[test]
    fn test_rerank_request_serialization() {
        let request = RerankRequest {
            query: "test query".to_string(),
            documents: vec!["doc1".to_string(), "doc2".to_string()],
            model: "test-model".to_string(),
            top_k: Some(5),
            return_documents: true,
            rid: Some(StringOrArray::String("req-123".to_string())),
            user: Some("user-456".to_string()),
        };

        let serialized = serde_json::to_string(&request).unwrap();
        let deserialized: RerankRequest = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.query, request.query);
        assert_eq!(deserialized.documents, request.documents);
        assert_eq!(deserialized.model, request.model);
        assert_eq!(deserialized.top_k, request.top_k);
        assert_eq!(deserialized.return_documents, request.return_documents);
        assert_eq!(deserialized.rid, request.rid);
        assert_eq!(deserialized.user, request.user);
    }

    #[test]
    fn test_rerank_request_deserialization_with_defaults() {
        let json = r#"{
            "query": "test query",
            "documents": ["doc1", "doc2"]
        }"#;

        let request: RerankRequest = serde_json::from_str(json).unwrap();

        assert_eq!(request.query, "test query");
        assert_eq!(request.documents, vec!["doc1", "doc2"]);
        assert_eq!(request.model, default_model_name());
        assert_eq!(request.top_k, None);
        assert!(request.return_documents);
        assert_eq!(request.rid, None);
        assert_eq!(request.user, None);
    }

    #[test]
    fn test_rerank_request_validation_success() {
        let request = RerankRequest {
            query: "valid query".to_string(),
            documents: vec!["doc1".to_string(), "doc2".to_string()],
            model: "test-model".to_string(),
            top_k: Some(2),
            return_documents: true,
            rid: None,
            user: None,
        };

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_rerank_request_validation_empty_query() {
        let request = RerankRequest {
            query: "".to_string(),
            documents: vec!["doc1".to_string()],
            model: "test-model".to_string(),
            top_k: None,
            return_documents: true,
            rid: None,
            user: None,
        };

        let result = request.validate();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Query cannot be empty");
    }

    #[test]
    fn test_rerank_request_validation_whitespace_query() {
        let request = RerankRequest {
            query: "   ".to_string(),
            documents: vec!["doc1".to_string()],
            model: "test-model".to_string(),
            top_k: None,
            return_documents: true,
            rid: None,
            user: None,
        };

        let result = request.validate();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Query cannot be empty");
    }

    #[test]
    fn test_rerank_request_validation_empty_documents() {
        let request = RerankRequest {
            query: "test query".to_string(),
            documents: vec![],
            model: "test-model".to_string(),
            top_k: None,
            return_documents: true,
            rid: None,
            user: None,
        };

        let result = request.validate();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Documents list cannot be empty");
    }

    #[test]
    fn test_rerank_request_validation_top_k_zero() {
        let request = RerankRequest {
            query: "test query".to_string(),
            documents: vec!["doc1".to_string(), "doc2".to_string()],
            model: "test-model".to_string(),
            top_k: Some(0),
            return_documents: true,
            rid: None,
            user: None,
        };

        let result = request.validate();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "top_k must be greater than 0");
    }

    #[test]
    fn test_rerank_request_validation_top_k_greater_than_docs() {
        let request = RerankRequest {
            query: "test query".to_string(),
            documents: vec!["doc1".to_string(), "doc2".to_string()],
            model: "test-model".to_string(),
            top_k: Some(5),
            return_documents: true,
            rid: None,
            user: None,
        };

        // This should pass but log a warning
        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_rerank_request_effective_top_k() {
        let request = RerankRequest {
            query: "test query".to_string(),
            documents: vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()],
            model: "test-model".to_string(),
            top_k: Some(2),
            return_documents: true,
            rid: None,
            user: None,
        };

        assert_eq!(request.effective_top_k(), 2);
    }

    #[test]
    fn test_rerank_request_effective_top_k_none() {
        let request = RerankRequest {
            query: "test query".to_string(),
            documents: vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()],
            model: "test-model".to_string(),
            top_k: None,
            return_documents: true,
            rid: None,
            user: None,
        };

        assert_eq!(request.effective_top_k(), 3);
    }

    // ==================================================================
    // =            RERANK RESPONSE TESTS                               =
    // ==================================================================

    #[test]
    fn test_rerank_response_creation() {
        let results = vec![
            RerankResult {
                score: 0.8,
                document: Some("doc1".to_string()),
                index: 0,
                meta_info: None,
            },
            RerankResult {
                score: 0.6,
                document: Some("doc2".to_string()),
                index: 1,
                meta_info: None,
            },
        ];

        let response = RerankResponse::new(
            results.clone(),
            "test-model".to_string(),
            "req-123".to_string(),
        );

        assert_eq!(response.results.len(), 2);
        assert_eq!(response.model, "test-model");
        assert_eq!(response.id, "req-123");
        assert_eq!(response.object, "rerank");
        assert!(response.created > 0);
    }

    #[test]
    fn test_rerank_response_serialization() {
        let results = vec![RerankResult {
            score: 0.8,
            document: Some("doc1".to_string()),
            index: 0,
            meta_info: None,
        }];

        let response =
            RerankResponse::new(results, "test-model".to_string(), "req-123".to_string());

        let serialized = serde_json::to_string(&response).unwrap();
        let deserialized: RerankResponse = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.results.len(), response.results.len());
        assert_eq!(deserialized.model, response.model);
        assert_eq!(deserialized.id, response.id);
        assert_eq!(deserialized.object, response.object);
    }

    #[test]
    fn test_rerank_response_sort_by_score() {
        let results = vec![
            RerankResult {
                score: 0.6,
                document: Some("doc2".to_string()),
                index: 1,
                meta_info: None,
            },
            RerankResult {
                score: 0.8,
                document: Some("doc1".to_string()),
                index: 0,
                meta_info: None,
            },
            RerankResult {
                score: 0.4,
                document: Some("doc3".to_string()),
                index: 2,
                meta_info: None,
            },
        ];

        let mut response =
            RerankResponse::new(results, "test-model".to_string(), "req-123".to_string());

        response.sort_by_score();

        assert_eq!(response.results[0].score, 0.8);
        assert_eq!(response.results[0].index, 0);
        assert_eq!(response.results[1].score, 0.6);
        assert_eq!(response.results[1].index, 1);
        assert_eq!(response.results[2].score, 0.4);
        assert_eq!(response.results[2].index, 2);
    }

    #[test]
    fn test_rerank_response_apply_top_k() {
        let results = vec![
            RerankResult {
                score: 0.8,
                document: Some("doc1".to_string()),
                index: 0,
                meta_info: None,
            },
            RerankResult {
                score: 0.6,
                document: Some("doc2".to_string()),
                index: 1,
                meta_info: None,
            },
            RerankResult {
                score: 0.4,
                document: Some("doc3".to_string()),
                index: 2,
                meta_info: None,
            },
        ];

        let mut response =
            RerankResponse::new(results, "test-model".to_string(), "req-123".to_string());

        response.apply_top_k(2);

        assert_eq!(response.results.len(), 2);
        assert_eq!(response.results[0].score, 0.8);
        assert_eq!(response.results[1].score, 0.6);
    }

    #[test]
    fn test_rerank_response_apply_top_k_larger_than_results() {
        let results = vec![RerankResult {
            score: 0.8,
            document: Some("doc1".to_string()),
            index: 0,
            meta_info: None,
        }];

        let mut response =
            RerankResponse::new(results, "test-model".to_string(), "req-123".to_string());

        response.apply_top_k(5);

        assert_eq!(response.results.len(), 1);
    }

    // ==================================================================
    // =            RERANK RESULT TESTS                                 =
    // ==================================================================

    #[test]
    fn test_rerank_result_serialization() {
        let result = RerankResult {
            score: 0.85,
            document: Some("test document".to_string()),
            index: 42,
            meta_info: Some(HashMap::from([
                ("confidence".to_string(), Value::String("high".to_string())),
                (
                    "processing_time".to_string(),
                    Value::Number(serde_json::Number::from(150)),
                ),
            ])),
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: RerankResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.score, result.score);
        assert_eq!(deserialized.document, result.document);
        assert_eq!(deserialized.index, result.index);
        assert_eq!(deserialized.meta_info, result.meta_info);
    }

    #[test]
    fn test_rerank_result_serialization_without_document() {
        let result = RerankResult {
            score: 0.85,
            document: None,
            index: 42,
            meta_info: None,
        };

        let serialized = serde_json::to_string(&result).unwrap();
        let deserialized: RerankResult = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.score, result.score);
        assert_eq!(deserialized.document, result.document);
        assert_eq!(deserialized.index, result.index);
        assert_eq!(deserialized.meta_info, result.meta_info);
    }

    // ==================================================================
    // =            V1 COMPATIBILITY TESTS                              =
    // ==================================================================

    #[test]
    fn test_v1_rerank_req_input_serialization() {
        let v1_input = V1RerankReqInput {
            query: "test query".to_string(),
            documents: vec!["doc1".to_string(), "doc2".to_string()],
        };

        let serialized = serde_json::to_string(&v1_input).unwrap();
        let deserialized: V1RerankReqInput = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.query, v1_input.query);
        assert_eq!(deserialized.documents, v1_input.documents);
    }

    #[test]
    fn test_v1_to_rerank_request_conversion() {
        let v1_input = V1RerankReqInput {
            query: "test query".to_string(),
            documents: vec!["doc1".to_string(), "doc2".to_string()],
        };

        let request: RerankRequest = v1_input.into();

        assert_eq!(request.query, "test query");
        assert_eq!(request.documents, vec!["doc1", "doc2"]);
        assert_eq!(request.model, default_model_name());
        assert_eq!(request.top_k, None);
        assert!(request.return_documents);
        assert_eq!(request.rid, None);
        assert_eq!(request.user, None);
    }

    // ==================================================================
    // =            GENERATION REQUEST TRAIT TESTS                      =
    // ==================================================================

    #[test]
    fn test_rerank_request_generation_request_trait() {
        let request = RerankRequest {
            query: "test query".to_string(),
            documents: vec!["doc1".to_string()],
            model: "test-model".to_string(),
            top_k: None,
            return_documents: true,
            rid: None,
            user: None,
        };

        assert_eq!(request.get_model(), Some("test-model"));
        assert!(!request.is_stream());
        assert_eq!(request.extract_text_for_routing(), "test query");
    }

    // ==================================================================
    // =            EDGE CASES AND STRESS TESTS                         =
    // ==================================================================

    #[test]
    fn test_rerank_request_very_long_query() {
        let long_query = "a".repeat(100000);
        let request = RerankRequest {
            query: long_query,
            documents: vec!["doc1".to_string()],
            model: "test-model".to_string(),
            top_k: None,
            return_documents: true,
            rid: None,
            user: None,
        };

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_rerank_request_many_documents() {
        let documents: Vec<String> = (0..1000).map(|i| format!("doc{}", i)).collect();
        let request = RerankRequest {
            query: "test query".to_string(),
            documents,
            model: "test-model".to_string(),
            top_k: Some(100),
            return_documents: true,
            rid: None,
            user: None,
        };

        assert!(request.validate().is_ok());
        assert_eq!(request.effective_top_k(), 100);
    }

    #[test]
    fn test_rerank_request_special_characters() {
        let request = RerankRequest {
            query: "query with émojis 🚀 and unicode: 测试".to_string(),
            documents: vec![
                "doc with émojis 🎉".to_string(),
                "doc with unicode: 测试".to_string(),
            ],
            model: "test-model".to_string(),
            top_k: None,
            return_documents: true,
            rid: Some(StringOrArray::String("req-🚀-123".to_string())),
            user: Some("user-🎉-456".to_string()),
        };

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_rerank_request_rid_array() {
        let request = RerankRequest {
            query: "test query".to_string(),
            documents: vec!["doc1".to_string()],
            model: "test-model".to_string(),
            top_k: None,
            return_documents: true,
            rid: Some(StringOrArray::Array(vec![
                "req1".to_string(),
                "req2".to_string(),
            ])),
            user: None,
        };

        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_rerank_response_with_usage_info() {
        let results = vec![RerankResult {
            score: 0.8,
            document: Some("doc1".to_string()),
            index: 0,
            meta_info: None,
        }];

        let mut response =
            RerankResponse::new(results, "test-model".to_string(), "req-123".to_string());

        response.usage = Some(UsageInfo {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            reasoning_tokens: None,
            prompt_tokens_details: None,
        });

        let serialized = serde_json::to_string(&response).unwrap();
        let deserialized: RerankResponse = serde_json::from_str(&serialized).unwrap();

        assert!(deserialized.usage.is_some());
        let usage = deserialized.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    // ==================================================================
    // =            INTEGRATION TESTS                                   =
    // ==================================================================

    #[test]
    fn test_full_rerank_workflow() {
        // Create request
        let request = RerankRequest {
            query: "machine learning".to_string(),
            documents: vec![
                "Introduction to machine learning algorithms".to_string(),
                "Deep learning for computer vision".to_string(),
                "Natural language processing basics".to_string(),
                "Statistics and probability theory".to_string(),
            ],
            model: "rerank-model".to_string(),
            top_k: Some(2),
            return_documents: true,
            rid: Some(StringOrArray::String("req-123".to_string())),
            user: Some("user-456".to_string()),
        };

        // Validate request
        assert!(request.validate().is_ok());

        // Simulate reranking results (in real scenario, this would come from the model)
        let results = vec![
            RerankResult {
                score: 0.95,
                document: Some("Introduction to machine learning algorithms".to_string()),
                index: 0,
                meta_info: None,
            },
            RerankResult {
                score: 0.87,
                document: Some("Deep learning for computer vision".to_string()),
                index: 1,
                meta_info: None,
            },
            RerankResult {
                score: 0.72,
                document: Some("Natural language processing basics".to_string()),
                index: 2,
                meta_info: None,
            },
            RerankResult {
                score: 0.45,
                document: Some("Statistics and probability theory".to_string()),
                index: 3,
                meta_info: None,
            },
        ];

        // Create response
        let mut response = RerankResponse::new(
            results,
            request.model.clone(),
            request
                .rid
                .as_ref()
                .and_then(|r| match r {
                    StringOrArray::String(s) => Some(s.clone()),
                    StringOrArray::Array(arr) => arr.first().cloned(),
                })
                .unwrap_or_else(|| "unknown".to_string()),
        );

        // Sort by score
        response.sort_by_score();

        // Apply top_k
        response.apply_top_k(request.effective_top_k());

        // Verify results
        assert_eq!(response.results.len(), 2);
        assert_eq!(response.results[0].score, 0.95);
        assert_eq!(response.results[0].index, 0);
        assert_eq!(response.results[1].score, 0.87);
        assert_eq!(response.results[1].index, 1);
        assert_eq!(response.model, "rerank-model");

        // Serialize and deserialize
        let serialized = serde_json::to_string(&response).unwrap();
        let deserialized: RerankResponse = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.results.len(), 2);
        assert_eq!(deserialized.model, response.model);
    }
}
