//! NoOp storage implementations
//!
//! These implementations do nothing - useful for when persistence is disabled.
//!
//! Structure:
//! 1. NoOpConversationStorage
//! 2. NoOpConversationItemStorage
//! 3. NoOpResponseStorage

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use super::core::*;

// ============================================================================
// PART 1: NoOpConversationStorage
// ============================================================================

/// No-op implementation that synthesizes conversation responses without persistence
#[derive(Default, Debug, Clone)]
pub struct NoOpConversationStorage;

impl NoOpConversationStorage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ConversationStorage for NoOpConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> ConversationResult<Conversation> {
        Ok(Conversation::new(input))
    }

    async fn get_conversation(
        &self,
        _id: &ConversationId,
    ) -> ConversationResult<Option<Conversation>> {
        Ok(None)
    }

    async fn update_conversation(
        &self,
        _id: &ConversationId,
        _metadata: Option<ConversationMetadata>,
    ) -> ConversationResult<Option<Conversation>> {
        Ok(None)
    }

    async fn delete_conversation(&self, _id: &ConversationId) -> ConversationResult<bool> {
        Ok(false)
    }
}

// ============================================================================
// PART 2: NoOpConversationItemStorage
// ============================================================================

/// No-op conversation item storage (does nothing)
#[derive(Clone, Copy, Default)]
pub struct NoOpConversationItemStorage;

impl NoOpConversationItemStorage {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ConversationItemStorage for NoOpConversationItemStorage {
    async fn create_item(
        &self,
        item: NewConversationItem,
    ) -> ConversationItemResult<ConversationItem> {
        let id = item
            .id
            .clone()
            .unwrap_or_else(|| make_item_id(&item.item_type));
        Ok(ConversationItem {
            id,
            response_id: item.response_id,
            item_type: item.item_type,
            role: item.role,
            content: item.content,
            status: item.status,
            created_at: Utc::now(),
        })
    }

    async fn link_item(
        &self,
        _conversation_id: &ConversationId,
        _item_id: &ConversationItemId,
        _added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()> {
        Ok(())
    }

    async fn list_items(
        &self,
        _conversation_id: &ConversationId,
        _params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>> {
        Ok(Vec::new())
    }

    async fn get_item(
        &self,
        _item_id: &ConversationItemId,
    ) -> ConversationItemResult<Option<ConversationItem>> {
        Ok(None)
    }

    async fn is_item_linked(
        &self,
        _conversation_id: &ConversationId,
        _item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool> {
        Ok(false)
    }

    async fn delete_item(
        &self,
        _conversation_id: &ConversationId,
        _item_id: &ConversationItemId,
    ) -> ConversationItemResult<()> {
        Ok(())
    }
}

// ============================================================================
// PART 3: NoOpResponseStorage
// ============================================================================

/// No-op implementation of response storage (does nothing)
pub struct NoOpResponseStorage;

impl NoOpResponseStorage {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoOpResponseStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ResponseStorage for NoOpResponseStorage {
    async fn store_response(&self, response: StoredResponse) -> ResponseResult<ResponseId> {
        Ok(response.id)
    }

    async fn get_response(
        &self,
        _response_id: &ResponseId,
    ) -> ResponseResult<Option<StoredResponse>> {
        Ok(None)
    }

    async fn delete_response(&self, _response_id: &ResponseId) -> ResponseResult<()> {
        Ok(())
    }

    async fn get_response_chain(
        &self,
        _response_id: &ResponseId,
        _max_depth: Option<usize>,
    ) -> ResponseResult<ResponseChain> {
        Ok(ResponseChain::new())
    }

    async fn list_identifier_responses(
        &self,
        _identifier: &str,
        _limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        Ok(Vec::new())
    }

    async fn delete_identifier_responses(&self, _identifier: &str) -> ResponseResult<usize> {
        Ok(0)
    }
}
