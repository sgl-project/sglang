//! Context Loading stage
//!
//! This stage:
//! - Loads previous response chain (if previous_response_id provided)
//! - Loads conversation history (if conversation ID provided)
//! - Builds conversation context for the request

use async_trait::async_trait;
use axum::response::Response;

use super::PipelineStage;
use crate::routers::openai::context::{RequestContext, ContextOutput};

/// Context loading stage
pub struct ContextLoadingStage;

#[async_trait]
impl PipelineStage for ContextLoadingStage {
    async fn execute(&self, _ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // TODO: Phase 2 - Implement context loading logic
        // For now, this is a no-op that just passes through
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ContextLoading"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_context_loading_stage_passthrough() {
        // TODO: Add tests in Phase 2
    }
}
