// core.rs
//
// Core types for the data connector module.
// Contains all traits, data types, error types, and IDs for all storage backends.
//
// Structure:
// 1. Conversation types + trait
// 2. ConversationItem types + trait
// 3. Response types + trait

use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value};

// ============================================================================
// PART 1: Conversation Storage
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ConversationId(pub String);

impl ConversationId {
    pub fn new() -> Self {
        let mut rng = rand::rng();
        let mut bytes = [0u8; 25];
        rng.fill_bytes(&mut bytes);
        let hex_string: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
        Self(format!("conv_{}", hex_string))
    }
}

impl Default for ConversationId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<String> for ConversationId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ConversationId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl Display for ConversationId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Metadata payload persisted with a conversation
pub type ConversationMetadata = JsonMap<String, Value>;

/// Input payload for creating a conversation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NewConversation {
    /// Optional conversation ID (if None, a random ID will be generated)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<ConversationId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ConversationMetadata>,
}

/// Stored conversation data structure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Conversation {
    pub id: ConversationId,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ConversationMetadata>,
}

impl Conversation {
    pub fn new(new_conversation: NewConversation) -> Self {
        Self {
            id: new_conversation.id.unwrap_or_default(),
            created_at: Utc::now(),
            metadata: new_conversation.metadata,
        }
    }

    pub fn with_parts(
        id: ConversationId,
        created_at: DateTime<Utc>,
        metadata: Option<ConversationMetadata>,
    ) -> Self {
        Self {
            id,
            created_at,
            metadata,
        }
    }
}

/// Result alias for conversation storage operations
pub type ConversationResult<T> = Result<T, ConversationStorageError>;

/// Error type for conversation storage operations
#[derive(Debug, thiserror::Error)]
pub enum ConversationStorageError {
    #[error("Conversation not found: {0}")]
    ConversationNotFound(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

/// Trait describing the CRUD interface for conversation storage backends
#[async_trait]
pub trait ConversationStorage: Send + Sync + 'static {
    async fn create_conversation(&self, input: NewConversation)
        -> ConversationResult<Conversation>;

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> ConversationResult<Option<Conversation>>;

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> ConversationResult<Option<Conversation>>;

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool>;
}

// ============================================================================
// PART 2: ConversationItem Storage
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ConversationItemId(pub String);

impl Display for ConversationItemId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ConversationItemId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ConversationItemId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItem {
    pub id: ConversationItemId,
    pub response_id: Option<String>,
    pub item_type: String,
    pub role: Option<String>,
    pub content: Value,
    pub status: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewConversationItem {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<ConversationItemId>,
    pub response_id: Option<String>,
    pub item_type: String,
    pub role: Option<String>,
    pub content: Value,
    pub status: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SortOrder {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListParams {
    pub limit: usize,
    pub order: SortOrder,
    pub after: Option<String>, // item_id cursor
}

pub type ConversationItemResult<T> = Result<T, ConversationItemStorageError>;

#[derive(Debug, thiserror::Error)]
pub enum ConversationItemStorageError {
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
}

#[async_trait]
pub trait ConversationItemStorage: Send + Sync + 'static {
    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> ConversationItemResult<ConversationItem>;

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()>;

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>>;

    /// Get a single item by ID
    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<Option<ConversationItem>>;

    /// Check if an item is linked to a conversation
    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool>;

    /// Delete an item link from a conversation (does not delete the item itself)
    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<()>;
}

/// Helper to build id prefix based on item_type
pub fn make_item_id(item_type: &str) -> ConversationItemId {
    // Generate exactly 50 hex characters (25 bytes) for the part after the underscore
    let mut rng = rand::rng();
    let mut bytes = [0u8; 25];
    rng.fill_bytes(&mut bytes);
    let hex_string: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();

    let prefix: String = match item_type {
        "message" => "msg".to_string(),
        "reasoning" => "rs".to_string(),
        "mcp_call" => "mcp".to_string(),
        "mcp_list_tools" => "mcpl".to_string(),
        "function_call" => "fc".to_string(),
        other => {
            // Fallback: first 3 letters of type or "itm"
            let mut p = other.chars().take(3).collect::<String>();
            if p.is_empty() {
                p = "itm".to_string();
            }
            p
        }
    };
    ConversationItemId(format!("{}_{}", prefix, hex_string))
}

// ============================================================================
// PART 3: Response Storage
// ============================================================================

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

    /// Input items as JSON array
    pub input: Value,

    /// System instructions used
    pub instructions: Option<String>,

    /// Output items as JSON array
    pub output: Value,

    /// Tool calls made by the model (if any)
    pub tool_calls: Vec<Value>,

    /// Custom metadata
    pub metadata: HashMap<String, Value>,

    /// When this response was created
    pub created_at: DateTime<Utc>,

    /// Safety identifier for content moderation
    pub safety_identifier: Option<String>,

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
    pub fn new(previous_response_id: Option<ResponseId>) -> Self {
        Self {
            id: ResponseId::new(),
            previous_response_id,
            input: Value::Array(vec![]),
            instructions: None,
            output: Value::Array(vec![]),
            tool_calls: Vec::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            safety_identifier: None,
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
    pub fn build_context(&self, max_responses: Option<usize>) -> Vec<(Value, Value)> {
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

pub type ResponseResult<T> = Result<T, ResponseStorageError>;

/// Trait for response storage
#[async_trait]
pub trait ResponseStorage: Send + Sync {
    /// Store a new response
    async fn store_response(&self, response: StoredResponse) -> ResponseResult<ResponseId>;

    /// Get a response by ID
    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> ResponseResult<Option<StoredResponse>>;

    /// Delete a response
    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()>;

    /// Get the chain of responses leading to a given response
    /// Returns responses in chronological order (oldest first)
    async fn get_response_chain(
        &self,
        response_id: &ResponseId,
        max_depth: Option<usize>,
    ) -> ResponseResult<ResponseChain>;

    /// List recent responses for a safety identifier
    async fn list_identifier_responses(
        &self,
        identifier: &str,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>>;

    /// Delete all responses for a safety identifier
    async fn delete_identifier_responses(&self, identifier: &str) -> ResponseResult<usize>;
}

impl Default for StoredResponse {
    fn default() -> Self {
        Self::new(None)
    }
}
