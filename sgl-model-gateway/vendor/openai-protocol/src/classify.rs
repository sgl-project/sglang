//! Classify API protocol definitions.
//!
//! This module defines the request and response types for the `/v1/classify` API,
//! which is compatible with vLLM's classification endpoint.
//!
//! Classification reuses the embedding backend - the scheduler returns logits as
//! "embeddings", and the classify layer applies softmax + label mapping.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::{GenerationRequest, UsageInfo};

// ============================================================================
// Classify API
// ============================================================================

/// Classification request - compatible with vLLM's /v1/classify API
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClassifyRequest {
    /// ID of the model to use
    pub model: String,

    /// Input can be a string, array of strings, or token IDs
    /// - Single string: "text to classify"
    /// - Array of strings: ["text1", "text2"]
    /// - Token IDs: [1, 2, 3] (advanced usage)
    pub input: Value,

    /// Optional user identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// SGLang extension: request id for tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rid: Option<String>,

    /// SGLang extension: request priority
    #[serde(skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,

    /// SGLang extension: enable/disable logging of metrics
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_metrics: Option<bool>,
}

impl GenerationRequest for ClassifyRequest {
    fn is_stream(&self) -> bool {
        false // Classification is always non-streaming
    }

    fn get_model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn extract_text_for_routing(&self) -> String {
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

// ============================================================================
// Classify Response
// ============================================================================

/// Single classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyData {
    /// Index of this result (for batch requests)
    pub index: u32,
    /// Predicted class label (from id2label mapping)
    pub label: String,
    /// Probability distribution over all classes (softmax of logits)
    pub probs: Vec<f32>,
    /// Number of classes
    pub num_classes: u32,
}

/// Classification response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifyResponse {
    /// Unique request ID (format: "classify-{uuid}")
    pub id: String,
    /// Always "list"
    pub object: String,
    /// Unix timestamp (seconds since epoch)
    pub created: u64,
    /// Model name
    pub model: String,
    /// Classification results (one per input in batch)
    pub data: Vec<ClassifyData>,
    /// Token usage info
    pub usage: UsageInfo,
}

impl ClassifyResponse {
    /// Create a new ClassifyResponse with the given data
    pub fn new(
        id: String,
        model: String,
        created: u64,
        data: Vec<ClassifyData>,
        usage: UsageInfo,
    ) -> Self {
        Self {
            id,
            object: "list".to_string(),
            created,
            model,
            data,
            usage,
        }
    }
}
