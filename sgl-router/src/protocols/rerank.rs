use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::chat::GenerationRequest;
use super::common::{StringOrArray, UsageInfo};

// Constants
pub const DEFAULT_MODEL_NAME: &str = "default";

pub fn default_model_name() -> String {
    DEFAULT_MODEL_NAME.to_string()
}

fn default_return_documents() -> bool {
    true
}

fn default_rerank_object() -> String {
    "rerank".to_string()
}

fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_secs() as i64
}

// ============================================================================
// Rerank API
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
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

impl RerankResponse {
    /// TODO: remove or move if the code is only used for uts
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

    /// Drop documents from results (when return_documents is false)
    pub fn drop_documents(&mut self) {
        for result in &mut self.results {
            result.document = None;
        }
    }
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
