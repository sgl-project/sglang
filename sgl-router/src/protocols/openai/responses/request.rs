// Responses API request types

use crate::protocols::common::{GenerationRequest, StringOrArray};
use crate::protocols::openai::responses::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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

fn default_true() -> bool {
    true
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
