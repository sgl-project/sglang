// Completions API response types

use crate::protocols::openai::common::{LogProbs, Usage};
use serde::{Deserialize, Serialize};

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
