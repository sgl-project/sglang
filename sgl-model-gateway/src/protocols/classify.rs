use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::common::GenerationRequest;

// ============================================================================
// Embedding API
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ClassifyRequest {
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

impl GenerationRequest for ClassifyRequest {
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
