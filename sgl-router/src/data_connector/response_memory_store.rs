use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use parking_lot::RwLock;

use super::responses::{ResponseChain, ResponseId, ResponseStorage, Result, StoredResponse};

/// Internal store structure holding both maps together
#[derive(Default)]
struct InnerStore {
    /// All stored responses indexed by ID
    responses: HashMap<ResponseId, StoredResponse>,
    /// Index of response IDs by user
    user_index: HashMap<String, Vec<ResponseId>>,
}

/// In-memory implementation of response storage
pub struct MemoryResponseStorage {
    /// Single lock wrapping both maps to prevent deadlocks and ensure atomic updates
    store: Arc<RwLock<InnerStore>>,
}

impl MemoryResponseStorage {
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(InnerStore::default())),
        }
    }

    /// Get statistics about the store
    pub fn stats(&self) -> MemoryStoreStats {
        let store = self.store.read();
        MemoryStoreStats {
            response_count: store.responses.len(),
            user_count: store.user_index.len(),
        }
    }

    /// Clear all data (useful for testing)
    pub fn clear(&self) {
        let mut store = self.store.write();
        store.responses.clear();
        store.user_index.clear();
    }
}

impl Default for MemoryResponseStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ResponseStorage for MemoryResponseStorage {
    async fn store_response(&self, mut response: StoredResponse) -> Result<ResponseId> {
        // Generate ID if not set
        if response.id.0.is_empty() {
            response.id = ResponseId::new();
        }

        let response_id = response.id.clone();

        // Single lock acquisition for atomic update
        let mut store = self.store.write();

        // Update user index if user is specified
        if let Some(ref user) = response.user {
            store
                .user_index
                .entry(user.clone())
                .or_default()
                .push(response_id.clone());
        }

        // Store the response
        store.responses.insert(response_id.clone(), response);
        tracing::info!("memory_store_size" = store.responses.len());

        Ok(response_id)
    }

    async fn get_response(&self, response_id: &ResponseId) -> Result<Option<StoredResponse>> {
        let store = self.store.read();
        let result = store.responses.get(response_id).cloned();
        tracing::info!("memory_get_response" = %response_id.0, found = result.is_some());
        Ok(result)
    }

    async fn delete_response(&self, response_id: &ResponseId) -> Result<()> {
        let mut store = self.store.write();

        // Remove the response and update user index if needed
        if let Some(response) = store.responses.remove(response_id) {
            if let Some(ref user) = response.user {
                if let Some(user_responses) = store.user_index.get_mut(user) {
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
    ) -> Result<ResponseChain> {
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

    async fn list_user_responses(
        &self,
        user: &str,
        limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>> {
        let store = self.store.read();

        if let Some(user_response_ids) = store.user_index.get(user) {
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

    async fn delete_user_responses(&self, user: &str) -> Result<usize> {
        let mut store = self.store.write();

        if let Some(user_response_ids) = store.user_index.remove(user) {
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
    pub user_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_store_with_custom_id() {
        let store = MemoryResponseStorage::new();
        let mut response = StoredResponse::new("Input".to_string(), "Output".to_string(), None);
        response.id = ResponseId::from("resp_custom");
        store.store_response(response.clone()).await.unwrap();
        let retrieved = store
            .get_response(&ResponseId::from("resp_custom"))
            .await
            .unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().output, "Output");
    }

    #[tokio::test]
    async fn test_memory_store_basic() {
        let store = MemoryResponseStorage::new();

        // Store a response
        let response = StoredResponse::new("Hello".to_string(), "Hi there!".to_string(), None);
        let response_id = store.store_response(response).await.unwrap();

        // Retrieve it
        let retrieved = store.get_response(&response_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().input, "Hello");

        // Delete it
        store.delete_response(&response_id).await.unwrap();
        let deleted = store.get_response(&response_id).await.unwrap();
        assert!(deleted.is_none());
    }

    #[tokio::test]
    async fn test_response_chain() {
        let store = MemoryResponseStorage::new();

        // Create a chain of responses
        let response1 =
            StoredResponse::new("First".to_string(), "First response".to_string(), None);
        let id1 = store.store_response(response1).await.unwrap();

        let response2 = StoredResponse::new(
            "Second".to_string(),
            "Second response".to_string(),
            Some(id1.clone()),
        );
        let id2 = store.store_response(response2).await.unwrap();

        let response3 = StoredResponse::new(
            "Third".to_string(),
            "Third response".to_string(),
            Some(id2.clone()),
        );
        let id3 = store.store_response(response3).await.unwrap();

        // Get the chain
        let chain = store.get_response_chain(&id3, None).await.unwrap();
        assert_eq!(chain.responses.len(), 3);
        assert_eq!(chain.responses[0].input, "First");
        assert_eq!(chain.responses[1].input, "Second");
        assert_eq!(chain.responses[2].input, "Third");

        let limited_chain = store.get_response_chain(&id3, Some(2)).await.unwrap();
        assert_eq!(limited_chain.responses.len(), 2);
        assert_eq!(limited_chain.responses[0].input, "Second");
        assert_eq!(limited_chain.responses[1].input, "Third");
    }

    #[tokio::test]
    async fn test_user_responses() {
        let store = MemoryResponseStorage::new();

        // Store responses for different users
        let mut response1 = StoredResponse::new(
            "User1 message".to_string(),
            "Response to user1".to_string(),
            None,
        );
        response1.user = Some("user1".to_string());
        store.store_response(response1).await.unwrap();

        let mut response2 = StoredResponse::new(
            "Another user1 message".to_string(),
            "Another response to user1".to_string(),
            None,
        );
        response2.user = Some("user1".to_string());
        store.store_response(response2).await.unwrap();

        let mut response3 = StoredResponse::new(
            "User2 message".to_string(),
            "Response to user2".to_string(),
            None,
        );
        response3.user = Some("user2".to_string());
        store.store_response(response3).await.unwrap();

        // List user1's responses
        let user1_responses = store.list_user_responses("user1", None).await.unwrap();
        assert_eq!(user1_responses.len(), 2);

        // List user2's responses
        let user2_responses = store.list_user_responses("user2", None).await.unwrap();
        assert_eq!(user2_responses.len(), 1);

        // Delete user1's responses
        let deleted_count = store.delete_user_responses("user1").await.unwrap();
        assert_eq!(deleted_count, 2);

        let user1_responses_after = store.list_user_responses("user1", None).await.unwrap();
        assert_eq!(user1_responses_after.len(), 0);

        // User2's responses should still be there
        let user2_responses_after = store.list_user_responses("user2", None).await.unwrap();
        assert_eq!(user2_responses_after.len(), 1);
    }

    #[tokio::test]
    async fn test_memory_store_stats() {
        let store = MemoryResponseStorage::new();

        let mut response1 = StoredResponse::new("Test1".to_string(), "Reply1".to_string(), None);
        response1.user = Some("user1".to_string());
        store.store_response(response1).await.unwrap();

        let mut response2 = StoredResponse::new("Test2".to_string(), "Reply2".to_string(), None);
        response2.user = Some("user2".to_string());
        store.store_response(response2).await.unwrap();

        let stats = store.stats();
        assert_eq!(stats.response_count, 2);
        assert_eq!(stats.user_count, 2);
    }
}
