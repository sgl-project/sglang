use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use super::responses::{ResponseChain, ResponseId, ResponseStorage, Result, StoredResponse};

/// In-memory implementation of response storage
pub struct MemoryResponseStorage {
    /// All stored responses indexed by ID
    responses: Arc<RwLock<HashMap<ResponseId, StoredResponse>>>,
    /// Index of responses by user
    user_index: Arc<RwLock<HashMap<String, Vec<ResponseId>>>>,
}

impl MemoryResponseStorage {
    pub fn new() -> Self {
        Self {
            responses: Arc::new(RwLock::new(HashMap::new())),
            user_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get statistics about the store
    pub fn stats(&self) -> MemoryStoreStats {
        let responses = self.responses.read();
        let user_index = self.user_index.read();

        MemoryStoreStats {
            response_count: responses.len(),
            user_count: user_index.len(),
        }
    }

    /// Clear all data (useful for testing)
    pub fn clear(&self) {
        self.responses.write().clear();
        self.user_index.write().clear();
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

        // Store the response
        self.responses
            .write()
            .insert(response_id.clone(), response.clone());

        // Update user index if user is specified
        if let Some(ref user) = response.user {
            self.user_index
                .write()
                .entry(user.clone())
                .or_default()
                .push(response_id.clone());
        }

        Ok(response_id)
    }

    async fn get_response(&self, response_id: &ResponseId) -> Result<Option<StoredResponse>> {
        Ok(self.responses.read().get(response_id).cloned())
    }

    async fn delete_response(&self, response_id: &ResponseId) -> Result<()> {
        let response = self.responses.write().remove(response_id);

        // Remove from user index if it existed
        if let Some(response) = response {
            if let Some(ref user) = response.user {
                if let Some(user_responses) = self.user_index.write().get_mut(user) {
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
        let mut current_id = Some(response_id.clone());
        let mut depth = 0;
        let max_depth = max_depth.unwrap_or(100); // Default max depth to prevent infinite loops

        // Build the chain by following previous_response_id links
        let mut responses_to_add = Vec::new();

        while let Some(id) = current_id {
            if depth >= max_depth {
                break;
            }

            let responses = self.responses.read();
            if let Some(response) = responses.get(&id) {
                responses_to_add.push(response.clone());
                current_id = response.previous_response_id.clone();
                depth += 1;
            } else {
                break;
            }
        }

        // Reverse to get chronological order (oldest first)
        responses_to_add.reverse();
        for response in responses_to_add {
            chain.add_response(response);
        }

        Ok(chain)
    }

    async fn list_user_responses(
        &self,
        user: &str,
        limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>> {
        let user_index = self.user_index.read();
        let responses = self.responses.read();

        if let Some(user_response_ids) = user_index.get(user) {
            let mut user_responses: Vec<StoredResponse> = user_response_ids
                .iter()
                .filter_map(|id| responses.get(id).cloned())
                .collect();

            // Sort by creation time (newest first)
            user_responses.sort_by(|a, b| b.created_at.cmp(&a.created_at));

            // Apply limit if specified
            if let Some(limit) = limit {
                user_responses.truncate(limit);
            }

            Ok(user_responses)
        } else {
            Ok(Vec::new())
        }
    }

    async fn delete_user_responses(&self, user: &str) -> Result<usize> {
        let mut user_index = self.user_index.write();
        let mut responses = self.responses.write();

        if let Some(user_response_ids) = user_index.remove(user) {
            let count = user_response_ids.len();
            for id in user_response_ids {
                responses.remove(&id);
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

        // Test with max_depth
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

        // Verify they're gone
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
