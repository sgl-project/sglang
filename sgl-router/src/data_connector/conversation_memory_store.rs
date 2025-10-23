use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use parking_lot::RwLock;

use super::conversations::{
    Conversation, ConversationId, ConversationMetadata, ConversationStorage, NewConversation,
    Result,
};

/// In-memory conversation storage used for development and tests
#[derive(Default, Clone)]
pub struct MemoryConversationStorage {
    inner: Arc<RwLock<HashMap<ConversationId, Conversation>>>,
}

impl MemoryConversationStorage {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl ConversationStorage for MemoryConversationStorage {
    async fn create_conversation(&self, input: NewConversation) -> Result<Conversation> {
        let conversation = Conversation::new(input);
        self.inner
            .write()
            .insert(conversation.id.clone(), conversation.clone());
        Ok(conversation)
    }

    async fn get_conversation(&self, id: &ConversationId) -> Result<Option<Conversation>> {
        Ok(self.inner.read().get(id).cloned())
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> Result<Option<Conversation>> {
        let mut store = self.inner.write();
        if let Some(entry) = store.get_mut(id) {
            entry.metadata = metadata;
            return Ok(Some(entry.clone()));
        }

        Ok(None)
    }

    async fn delete_conversation(&self, id: &ConversationId) -> Result<bool> {
        let removed = self.inner.write().remove(id).is_some();
        Ok(removed)
    }
}
