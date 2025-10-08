use async_trait::async_trait;

use super::conversations::{
    Conversation, ConversationId, ConversationMetadata, ConversationStorage, Result,
};

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
        input: super::conversations::NewConversation,
    ) -> Result<Conversation> {
        Ok(Conversation::new(input))
    }

    async fn get_conversation(&self, _id: &ConversationId) -> Result<Option<Conversation>> {
        Ok(None)
    }

    async fn update_conversation(
        &self,
        _id: &ConversationId,
        _metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>> {
        Ok(None)
    }

    async fn delete_conversation(&self, _id: &ConversationId) -> Result<bool> {
        Ok(false)
    }
}
