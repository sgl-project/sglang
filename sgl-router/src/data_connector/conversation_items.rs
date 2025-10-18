use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::conversations::ConversationId;

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

pub type Result<T> = std::result::Result<T, ConversationItemStorageError>;

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
    async fn create_item(&self, item: NewConversationItem) -> Result<ConversationItem>;

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> Result<()>;

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> Result<Vec<ConversationItem>>;

    /// Get a single item by ID
    async fn get_item(&self, item_id: &ConversationItemId) -> Result<Option<ConversationItem>>;

    /// Check if an item is linked to a conversation
    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> Result<bool>;

    /// Delete an item link from a conversation (does not delete the item itself)
    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> Result<()>;
}

pub type SharedConversationItemStorage = Arc<dyn ConversationItemStorage>;

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
        "function_tool_call" => "ftc".to_string(),
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
