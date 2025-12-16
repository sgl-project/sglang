//! In-memory storage implementations
//!
//! Used for development and testing - no persistence.
//!
//! Structure:
//! 1. MemoryConversationStorage
//! 2. MemoryConversationItemStorage
//! 3. MemoryResponseStorage

use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, RwLock},
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::RwLock as ParkingLotRwLock;

use super::core::*;

// ============================================================================
// PART 1: MemoryConversationStorage
// ============================================================================

/// In-memory conversation storage used for development and tests
#[derive(Default, Clone)]
pub struct MemoryConversationStorage {
    inner: Arc<ParkingLotRwLock<HashMap<ConversationId, Conversation>>>,
}

impl MemoryConversationStorage {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ParkingLotRwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl ConversationStorage for MemoryConversationStorage {
    async fn create_conversation(
        &self,
        input: NewConversation,
    ) -> ConversationResult<Conversation> {
        let conversation = Conversation::new(input);
        self.inner
            .write()
            .insert(conversation.id.clone(), conversation.clone());
        Ok(conversation)
    }

    async fn get_conversation(
        &self,
        id: &ConversationId,
    ) -> ConversationResult<Option<Conversation>> {
        Ok(self.inner.read().get(id).cloned())
    }

    async fn update_conversation(
        &self,
        id: &ConversationId,
        metadata: Option<ConversationMetadata>,
    ) -> ConversationResult<Option<Conversation>> {
        let mut store = self.inner.write();
        if let Some(entry) = store.get_mut(id) {
            entry.metadata = metadata;
            return Ok(Some(entry.clone()));
        }

        Ok(None)
    }

    async fn delete_conversation(&self, id: &ConversationId) -> ConversationResult<bool> {
        let removed = self.inner.write().remove(id).is_some();
        Ok(removed)
    }
}

// ============================================================================
// PART 2: MemoryConversationItemStorage
// ============================================================================

#[derive(Default)]
pub struct MemoryConversationItemStorage {
    items: RwLock<HashMap<ConversationItemId, ConversationItem>>, // item_id -> item
    #[allow(clippy::type_complexity)]
    links: RwLock<HashMap<ConversationId, BTreeMap<(i64, String), ConversationItemId>>>,
    // Per-conversation reverse index for fast after cursor lookup: item_id_str -> (ts, item_id_str)
    #[allow(clippy::type_complexity)]
    rev_index: RwLock<HashMap<ConversationId, HashMap<String, (i64, String)>>>,
}

impl MemoryConversationItemStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ConversationItemStorage for MemoryConversationItemStorage {
    async fn create_item(
        &self,
        new_item: NewConversationItem,
    ) -> ConversationItemResult<ConversationItem> {
        let id = new_item
            .id
            .clone()
            .unwrap_or_else(|| make_item_id(&new_item.item_type));
        let created_at = Utc::now();
        let item = ConversationItem {
            id: id.clone(),
            response_id: new_item.response_id,
            item_type: new_item.item_type,
            role: new_item.role,
            content: new_item.content,
            status: new_item.status,
            created_at,
        };
        let mut items = self.items.write().unwrap();
        items.insert(id.clone(), item.clone());
        Ok(item)
    }

    async fn link_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
        added_at: DateTime<Utc>,
    ) -> ConversationItemResult<()> {
        {
            let mut links = self.links.write().unwrap();
            let entry = links.entry(conversation_id.clone()).or_default();
            entry.insert((added_at.timestamp(), item_id.0.clone()), item_id.clone());
        }
        {
            let mut rev = self.rev_index.write().unwrap();
            let entry = rev.entry(conversation_id.clone()).or_default();
            entry.insert(item_id.0.clone(), (added_at.timestamp(), item_id.0.clone()));
        }
        Ok(())
    }

    async fn list_items(
        &self,
        conversation_id: &ConversationId,
        params: ListParams,
    ) -> ConversationItemResult<Vec<ConversationItem>> {
        let links_guard = self.links.read().unwrap();
        let map = match links_guard.get(conversation_id) {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };

        let mut results: Vec<ConversationItem> = Vec::new();
        let after_key: Option<(i64, String)> = if let Some(after_id) = &params.after {
            // O(1) lookup via reverse index for this conversation
            if let Some(conv_idx) = self.rev_index.read().unwrap().get(conversation_id) {
                conv_idx.get(after_id).cloned()
            } else {
                None
            }
        } else {
            None
        };

        let take = params.limit;
        let items_guard = self.items.read().unwrap();

        use std::ops::Bound::{Excluded, Unbounded};

        // Helper to push item if it exists and stop when reaching the limit
        let mut push_item = |key: &ConversationItemId| -> bool {
            if let Some(it) = items_guard.get(key) {
                results.push(it.clone());
                if results.len() == take {
                    return true;
                }
            }
            false
        };

        match (params.order, after_key) {
            (SortOrder::Desc, Some(k)) => {
                for ((_ts, _id), item_key) in map.range(..k).rev() {
                    if push_item(item_key) {
                        break;
                    }
                }
            }
            (SortOrder::Desc, None) => {
                for ((_ts, _id), item_key) in map.iter().rev() {
                    if push_item(item_key) {
                        break;
                    }
                }
            }
            (SortOrder::Asc, Some(k)) => {
                for ((_ts, _id), item_key) in map.range((Excluded(k), Unbounded)) {
                    if push_item(item_key) {
                        break;
                    }
                }
            }
            (SortOrder::Asc, None) => {
                for ((_ts, _id), item_key) in map.iter() {
                    if push_item(item_key) {
                        break;
                    }
                }
            }
        }

        Ok(results)
    }

    async fn get_item(
        &self,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<Option<ConversationItem>> {
        let items = self.items.read().unwrap();
        Ok(items.get(item_id).cloned())
    }

    async fn is_item_linked(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<bool> {
        let rev = self.rev_index.read().unwrap();
        if let Some(conv_idx) = rev.get(conversation_id) {
            Ok(conv_idx.contains_key(&item_id.0))
        } else {
            Ok(false)
        }
    }

    async fn delete_item(
        &self,
        conversation_id: &ConversationId,
        item_id: &ConversationItemId,
    ) -> ConversationItemResult<()> {
        // Get the key from rev_index and remove the entry at the same time
        let key_to_remove = {
            let mut rev = self.rev_index.write().unwrap();
            if let Some(conv_idx) = rev.get_mut(conversation_id) {
                conv_idx.remove(&item_id.0)
            } else {
                None
            }
        };

        // If the item was in rev_index, remove it from links as well
        if let Some(key) = key_to_remove {
            let mut links = self.links.write().unwrap();
            if let Some(conv_links) = links.get_mut(conversation_id) {
                conv_links.remove(&key);
            }
        }

        Ok(())
    }
}

// ============================================================================
// PART 3: MemoryResponseStorage
// ============================================================================

/// Internal store structure holding both maps together
#[derive(Default)]
struct InnerStore {
    /// All stored responses indexed by ID
    responses: HashMap<ResponseId, StoredResponse>,
    /// Index of response IDs by safety identifier
    identifier_index: HashMap<String, Vec<ResponseId>>,
}

/// In-memory implementation of response storage
pub struct MemoryResponseStorage {
    /// Single lock wrapping both maps to prevent deadlocks and ensure atomic updates
    store: Arc<ParkingLotRwLock<InnerStore>>,
}

impl MemoryResponseStorage {
    pub fn new() -> Self {
        Self {
            store: Arc::new(ParkingLotRwLock::new(InnerStore::default())),
        }
    }

    /// Get statistics about the store
    pub fn stats(&self) -> MemoryStoreStats {
        let store = self.store.read();
        MemoryStoreStats {
            response_count: store.responses.len(),
            identifier_count: store.identifier_index.len(),
        }
    }

    /// Clear all data (useful for testing)
    pub fn clear(&self) {
        let mut store = self.store.write();
        store.responses.clear();
        store.identifier_index.clear();
    }
}

impl Default for MemoryResponseStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ResponseStorage for MemoryResponseStorage {
    async fn store_response(&self, mut response: StoredResponse) -> ResponseResult<ResponseId> {
        // Generate ID if not set
        if response.id.0.is_empty() {
            response.id = ResponseId::new();
        }

        let response_id = response.id.clone();

        // Single lock acquisition for atomic update
        let mut store = self.store.write();

        // Update safety identifier index if specified
        if let Some(ref safety_identifier) = response.safety_identifier {
            store
                .identifier_index
                .entry(safety_identifier.clone())
                .or_default()
                .push(response_id.clone());
        }

        // Store the response
        store.responses.insert(response_id.clone(), response);
        tracing::info!("memory_store_size" = store.responses.len());

        Ok(response_id)
    }

    async fn get_response(
        &self,
        response_id: &ResponseId,
    ) -> ResponseResult<Option<StoredResponse>> {
        let store = self.store.read();
        let result = store.responses.get(response_id).cloned();
        tracing::info!("memory_get_response" = %response_id.0, found = result.is_some());
        Ok(result)
    }

    async fn delete_response(&self, response_id: &ResponseId) -> ResponseResult<()> {
        let mut store = self.store.write();

        // Remove the response and update user index if needed
        if let Some(response) = store.responses.remove(response_id) {
            if let Some(ref safety_identifier) = response.safety_identifier {
                if let Some(user_responses) = store.identifier_index.get_mut(safety_identifier) {
                    user_responses.retain(|id| id != response_id);
                }
            }
        }

        Ok(())
    }

    async fn get_response_chain(
        &self,
        response_id: &ResponseId,
        max_depth: Option<usize>,
    ) -> ResponseResult<ResponseChain> {
        let mut chain = ResponseChain::new();
        let max_depth = max_depth.unwrap_or(100); // Default max depth to prevent infinite loops

        // Collect all response IDs first
        let mut response_ids = Vec::new();
        let mut current_id = Some(response_id.clone());
        let mut depth = 0;

        // Single lock acquisition to collect the chain
        {
            let store = self.store.read();
            while let Some(id) = current_id {
                if depth >= max_depth {
                    break;
                }

                if let Some(response) = store.responses.get(&id) {
                    response_ids.push(id);
                    current_id = response.previous_response_id.clone();
                    depth += 1;
                } else {
                    break;
                }
            }
        }

        // Reverse to get chronological order (oldest first)
        response_ids.reverse();

        // Now collect the actual responses
        let store = self.store.read();
        for id in response_ids {
            if let Some(response) = store.responses.get(&id) {
                chain.add_response(response.clone());
            }
        }

        Ok(chain)
    }

    async fn list_identifier_responses(
        &self,
        identifier: &str,
        limit: Option<usize>,
    ) -> ResponseResult<Vec<StoredResponse>> {
        let store = self.store.read();

        if let Some(user_response_ids) = store.identifier_index.get(identifier) {
            // Collect responses with their timestamps for sorting
            let mut responses_with_time: Vec<_> = user_response_ids
                .iter()
                .filter_map(|id| store.responses.get(id).map(|r| (r.created_at, id)))
                .collect();

            // Sort by creation time (newest first)
            responses_with_time.sort_by(|a, b| b.0.cmp(&a.0));

            // Apply limit and collect the actual responses
            let limit = limit.unwrap_or(responses_with_time.len());
            let user_responses: Vec<StoredResponse> = responses_with_time
                .into_iter()
                .take(limit)
                .filter_map(|(_, id)| store.responses.get(id).cloned())
                .collect();

            Ok(user_responses)
        } else {
            Ok(Vec::new())
        }
    }

    async fn delete_identifier_responses(&self, identifier: &str) -> ResponseResult<usize> {
        let mut store = self.store.write();

        if let Some(user_response_ids) = store.identifier_index.remove(identifier) {
            let count = user_response_ids.len();
            for id in user_response_ids {
                store.responses.remove(&id);
            }
            Ok(count)
        } else {
            Ok(0)
        }
    }
}

/// Statistics for the memory store
#[derive(Debug, Clone)]
pub struct MemoryStoreStats {
    pub response_count: usize,
    pub identifier_count: usize,
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    use super::*;

    // ========================================================================
    // ConversationItem Tests
    // ========================================================================

    fn make_item(
        item_type: &str,
        role: Option<&str>,
        content: serde_json::Value,
    ) -> NewConversationItem {
        NewConversationItem {
            id: None,
            response_id: None,
            item_type: item_type.to_string(),
            role: role.map(|r| r.to_string()),
            content,
            status: Some("completed".to_string()),
        }
    }

    #[tokio::test]
    async fn test_list_ordering_and_cursors() {
        let store = MemoryConversationItemStorage::new();
        let conv: ConversationId = "conv_test".into();

        // Create 3 items and link them at controlled timestamps
        let i1 = store
            .create_item(make_item("message", Some("user"), json!([])))
            .await
            .unwrap();
        let i2 = store
            .create_item(make_item("message", Some("assistant"), json!([])))
            .await
            .unwrap();
        let i3 = store
            .create_item(make_item("reasoning", None, json!([])))
            .await
            .unwrap();

        let t1 = Utc.timestamp_opt(1_700_000_001, 0).single().unwrap();
        let t2 = Utc.timestamp_opt(1_700_000_002, 0).single().unwrap();
        let t3 = Utc.timestamp_opt(1_700_000_003, 0).single().unwrap();

        store.link_item(&conv, &i1.id, t1).await.unwrap();
        store.link_item(&conv, &i2.id, t2).await.unwrap();
        store.link_item(&conv, &i3.id, t3).await.unwrap();

        // Desc order, no cursor
        let desc = store
            .list_items(
                &conv,
                ListParams {
                    limit: 2,
                    order: SortOrder::Desc,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert!(desc.len() >= 2);
        assert_eq!(desc[0].id, i3.id);
        assert_eq!(desc[1].id, i2.id);

        // Desc with cursor = i2 -> expect i1 next
        let desc_after = store
            .list_items(
                &conv,
                ListParams {
                    limit: 2,
                    order: SortOrder::Desc,
                    after: Some(i2.id.0.clone()),
                },
            )
            .await
            .unwrap();
        assert!(!desc_after.is_empty());
        assert_eq!(desc_after[0].id, i1.id);

        // Asc order, no cursor
        let asc = store
            .list_items(
                &conv,
                ListParams {
                    limit: 2,
                    order: SortOrder::Asc,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert!(asc.len() >= 2);
        assert_eq!(asc[0].id, i1.id);
        assert_eq!(asc[1].id, i2.id);

        // Asc with cursor = i2 -> expect i3 next
        let asc_after = store
            .list_items(
                &conv,
                ListParams {
                    limit: 2,
                    order: SortOrder::Asc,
                    after: Some(i2.id.0.clone()),
                },
            )
            .await
            .unwrap();
        assert!(!asc_after.is_empty());
        assert_eq!(asc_after[0].id, i3.id);
    }

    // ========================================================================
    // Response Tests
    // ========================================================================

    #[tokio::test]
    async fn test_store_with_custom_id() {
        let store = MemoryResponseStorage::new();
        let mut response = StoredResponse::new(None);
        response.id = ResponseId::from("resp_custom");
        response.input = json!("Input");
        response.output = json!("Output");
        store.store_response(response.clone()).await.unwrap();
        let retrieved = store
            .get_response(&ResponseId::from("resp_custom"))
            .await
            .unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().output, json!("Output"));
    }

    #[tokio::test]
    async fn test_memory_store_basic() {
        let store = MemoryResponseStorage::new();

        // Store a response
        let mut response = StoredResponse::new(None);
        response.input = json!("Hello");
        response.output = json!("Hi there!");
        let response_id = store.store_response(response).await.unwrap();

        // Retrieve it
        let retrieved = store.get_response(&response_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().input, json!("Hello"));

        // Delete it
        store.delete_response(&response_id).await.unwrap();
        let deleted = store.get_response(&response_id).await.unwrap();
        assert!(deleted.is_none());
    }

    #[tokio::test]
    async fn test_response_chain() {
        let store = MemoryResponseStorage::new();

        // Create a chain of responses
        let mut response1 = StoredResponse::new(None);
        response1.input = json!("First");
        response1.output = json!("First response");
        let id1 = store.store_response(response1).await.unwrap();

        let mut response2 = StoredResponse::new(Some(id1.clone()));
        response2.input = json!("Second");
        response2.output = json!("Second response");
        let id2 = store.store_response(response2).await.unwrap();

        let mut response3 = StoredResponse::new(Some(id2.clone()));
        response3.input = json!("Third");
        response3.output = json!("Third response");
        let id3 = store.store_response(response3).await.unwrap();

        // Get the chain
        let chain = store.get_response_chain(&id3, None).await.unwrap();
        assert_eq!(chain.responses.len(), 3);
        assert_eq!(chain.responses[0].input, json!("First"));
        assert_eq!(chain.responses[1].input, json!("Second"));
        assert_eq!(chain.responses[2].input, json!("Third"));

        let limited_chain = store.get_response_chain(&id3, Some(2)).await.unwrap();
        assert_eq!(limited_chain.responses.len(), 2);
        assert_eq!(limited_chain.responses[0].input, json!("Second"));
        assert_eq!(limited_chain.responses[1].input, json!("Third"));
    }

    #[tokio::test]
    async fn test_user_responses() {
        let store = MemoryResponseStorage::new();

        // Store responses for different users
        let mut response1 = StoredResponse::new(None);
        response1.input = json!("User1 message");
        response1.output = json!("Response to user1");
        response1.safety_identifier = Some("user1".to_string());
        store.store_response(response1).await.unwrap();

        let mut response2 = StoredResponse::new(None);
        response2.input = json!("Another user1 message");
        response2.output = json!("Another response to user1");
        response2.safety_identifier = Some("user1".to_string());
        store.store_response(response2).await.unwrap();

        let mut response3 = StoredResponse::new(None);
        response3.input = json!("User2 message");
        response3.output = json!("Response to user2");
        response3.safety_identifier = Some("user2".to_string());
        store.store_response(response3).await.unwrap();

        // List user1's responses
        let user1_responses = store
            .list_identifier_responses("user1", None)
            .await
            .unwrap();
        assert_eq!(user1_responses.len(), 2);

        // List user2's responses
        let user2_responses = store
            .list_identifier_responses("user2", None)
            .await
            .unwrap();
        assert_eq!(user2_responses.len(), 1);

        // Delete user1's responses
        let deleted_count = store.delete_identifier_responses("user1").await.unwrap();
        assert_eq!(deleted_count, 2);

        let user1_responses_after = store
            .list_identifier_responses("user1", None)
            .await
            .unwrap();
        assert_eq!(user1_responses_after.len(), 0);

        // User2's responses should still be there
        let user2_responses_after = store
            .list_identifier_responses("user2", None)
            .await
            .unwrap();
        assert_eq!(user2_responses_after.len(), 1);
    }

    #[tokio::test]
    async fn test_memory_store_stats() {
        let store = MemoryResponseStorage::new();

        let mut response1 = StoredResponse::new(None);
        response1.input = json!("Test1");
        response1.output = json!("Reply1");
        response1.safety_identifier = Some("user1".to_string());
        store.store_response(response1).await.unwrap();

        let mut response2 = StoredResponse::new(None);
        response2.input = json!("Test2");
        response2.output = json!("Reply2");
        response2.safety_identifier = Some("user2".to_string());
        store.store_response(response2).await.unwrap();

        let stats = store.stats();
        assert_eq!(stats.response_count, 2);
        assert_eq!(stats.identifier_count, 2);
    }
}
