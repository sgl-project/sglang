//! Immutable configuration required during request rendering.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SamplingDefaults {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RendererLimits {
    pub vocab_size: u64,
    pub context_len: u64,
    pub num_reserved_tokens: u64,
    pub allow_auto_truncate: bool,
    pub enable_return_hidden_states: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RendererConfig {
    pub served_model_name: String,
    pub tokenizer_path: String,
    pub revision: Option<String>,
    pub model_path: String,
    pub chat_template: Option<String>,
    pub tool_call_parser: Option<String>,
    pub reasoning_parser: Option<String>,
    #[serde(default)]
    pub default_chat_template_kwargs: std::collections::HashMap<String, serde_json::Value>,
    pub stream_response_default_include_usage: bool,
    pub default_sampling_params: SamplingDefaults,
    pub limits: RendererLimits,
}
