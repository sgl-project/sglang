//! Request Building stage
//!
//! This stage:
//! - Builds the HTTP request payload
//! - Strips SGLang-specific fields
//! - Applies provider-specific transformations (xAI, Gemini, etc.)
//! - Injects conversation items if needed

use async_trait::async_trait;
use axum::response::Response;

use super::PipelineStage;
use crate::routers::openai::context::{RequestContext, PayloadOutput};

/// Request building stage
pub struct RequestBuildingStage;

#[async_trait]
impl PipelineStage for RequestBuildingStage {
    async fn execute(&self, _ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // TODO: Phase 2 - Implement request building logic
        // For now, this is a no-op that just passes through
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "RequestBuilding"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_request_building_stage_passthrough() {
        // TODO: Add tests in Phase 2
    }
}
