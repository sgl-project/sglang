use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Response identifier
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResponseId(pub String);

impl ResponseId {
    pub fn new() -> Self {
        Self(ulid::Ulid::new().to_string())
    }
}

impl Default for ResponseId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<String> for ResponseId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ResponseId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

/// Stored response data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredResponse {
    /// Unique response ID
    pub id: ResponseId,

    /// ID of the previous response in the chain (if any)
    pub previous_response_id: Option<ResponseId>,

    /// The user input for this response
    pub input: String,

    /// System instructions used
    pub instructions: Option<String>,

    /// The model's output
    pub output: String,

    /// Tool calls made by the model (if any)
    pub tool_calls: Vec<Value>,

    /// Custom metadata
    pub metadata: HashMap<String, Value>,

    /// When this response was created
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// User identifier (optional)
    pub user: Option<String>,

    /// Model used for generation
    pub model: Option<String>,

    /// Conversation id if associated with a conversation
    #[serde(default)]
    pub conversation_id: Option<String>,

    /// Raw OpenAI response payload
    #[serde(default)]
    pub raw_response: Value,
}

impl StoredResponse {
    pub fn new(input: String, output: String, previous_response_id: Option<ResponseId>) -> Self {
        Self {
            id: ResponseId::new(),
            previous_response_id,
            input,
            instructions: None,
            output,
            tool_calls: Vec::new(),
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            user: None,
            model: None,
            conversation_id: None,
            raw_response: Value::Null,
        }
    }
}

/// Response chain - a sequence of related responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseChain {
    /// The responses in chronological order
    pub responses: Vec<StoredResponse>,

    /// Metadata about the chain
    pub metadata: HashMap<String, Value>,
}

impl Default for ResponseChain {
    fn default() -> Self {
        Self::new()
    }
}

impl ResponseChain {
    pub fn new() -> Self {
        Self {
            responses: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Get the ID of the most recent response in the chain
    pub fn latest_response_id(&self) -> Option<&ResponseId> {
        self.responses.last().map(|r| &r.id)
    }

    /// Add a response to the chain
    pub fn add_response(&mut self, response: StoredResponse) {
        self.responses.push(response);
    }

    /// Build context from the chain for the next request
    pub fn build_context(&self, max_responses: Option<usize>) -> Vec<(String, String)> {
        let responses = if let Some(max) = max_responses {
            let start = self.responses.len().saturating_sub(max);
            &self.responses[start..]
        } else {
            &self.responses[..]
        };

        responses
            .iter()
            .map(|r| (r.input.clone(), r.output.clone()))
            .collect()
    }
}

/// Error type for response storage operations
#[derive(Debug, thiserror::Error)]
pub enum ResponseStorageError {
    #[error("Response not found: {0}")]
    ResponseNotFound(String),

    #[error("Invalid chain: {0}")]
    InvalidChain(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ResponseStorageError>;

/// Trait for response storage
#[async_trait]
pub trait ResponseStorage: Send + Sync {
    /// Store a new response
    async fn store_response(&self, response: StoredResponse) -> Result<ResponseId>;

    /// Get a response by ID
    async fn get_response(&self, response_id: &ResponseId) -> Result<Option<StoredResponse>>;

    /// Delete a response
    async fn delete_response(&self, response_id: &ResponseId) -> Result<()>;

    /// Get the chain of responses leading to a given response
    /// Returns responses in chronological order (oldest first)
    async fn get_response_chain(
        &self,
        response_id: &ResponseId,
        max_depth: Option<usize>,
    ) -> Result<ResponseChain>;

    /// List recent responses for a user
    async fn list_user_responses(
        &self,
        user: &str,
        limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>>;

    /// Delete all responses for a user
    async fn delete_user_responses(&self, user: &str) -> Result<usize>;
}

/// Type alias for shared storage
pub type SharedResponseStorage = Arc<dyn ResponseStorage>;

impl Default for StoredResponse {
    fn default() -> Self {
        Self::new(String::new(), String::new(), None)
    }
}
