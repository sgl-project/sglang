use async_trait::async_trait;

use super::responses::{ResponseChain, ResponseId, ResponseStorage, Result, StoredResponse};

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
    async fn store_response(&self, response: StoredResponse) -> Result<ResponseId> {
        Ok(response.id)
    }

    async fn get_response(&self, _response_id: &ResponseId) -> Result<Option<StoredResponse>> {
        Ok(None)
    }

    async fn delete_response(&self, _response_id: &ResponseId) -> Result<()> {
        Ok(())
    }

    async fn get_response_chain(
        &self,
        _response_id: &ResponseId,
        _max_depth: Option<usize>,
    ) -> Result<ResponseChain> {
        Ok(ResponseChain::new())
    }

    async fn list_user_responses(
        &self,
        _user: &str,
        _limit: Option<usize>,
    ) -> Result<Vec<StoredResponse>> {
        Ok(Vec::new())
    }

    async fn delete_user_responses(&self, _user: &str) -> Result<usize> {
        Ok(0)
    }
}
