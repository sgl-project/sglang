//! Persistence stage
//!
//! This stage:
//! - Stores response to response storage
//! - Persists conversation items to conversation storage
//! - Updates conversation metadata
//! - Patches response with metadata

use async_trait::async_trait;
use axum::response::Response;

use super::PipelineStage;
use crate::routers::openai::context::{RequestContext, FinalResponse};

/// Persistence stage
pub struct PersistenceStage;

#[async_trait]
impl PipelineStage for PersistenceStage {
    async fn execute(&self, _ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // TODO: Phase 2 - Implement persistence logic
        // For now, this is a no-op that just passes through
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "Persistence"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_persistence_stage_passthrough() {
        // TODO: Add tests in Phase 2
    }
}
