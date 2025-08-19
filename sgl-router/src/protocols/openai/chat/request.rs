// Chat Completions API request types

use crate::protocols::common::{default_true, GenerationRequest, LoRAPath, StringOrArray};
use crate::protocols::openai::chat::types::*;
use crate::protocols::openai::common::StreamOptions;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
