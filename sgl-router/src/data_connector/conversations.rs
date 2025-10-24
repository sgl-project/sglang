use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct ConversationId(pub String);

impl ConversationId {
    pub fn new() -> Self {
        let mut rng = rand::rng();
        let mut bytes = [0u8; 24];
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
pub type Result<T> = std::result::Result<T, ConversationStorageError>;

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
    async fn create_conversation(&self, input: NewConversation) -> Result<Conversation>;

    async fn get_conversation(&self, id: &ConversationId) -> Result<Option<Conversation>>;

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>>;

    async fn delete_conversation(&self, id: &ConversationId) -> Result<bool>;
}

/// Shared pointer alias for conversation storage
pub type SharedConversationStorage = Arc<dyn ConversationStorage>;
