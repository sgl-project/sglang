//! Response Processing stage
//!
//! This stage:
//! - Handles streaming vs non-streaming responses
//! - Executes MCP tool loop if MCP is active
//! - Accumulates streaming responses
//! - Parses and validates responses

use async_trait::async_trait;
use axum::response::Response;

use super::PipelineStage;
use crate::routers::openai::context::{RequestContext, ProcessedResponse};

/// Response processing stage
pub struct ResponseProcessingStage;

#[async_trait]
impl PipelineStage for ResponseProcessingStage {
    async fn execute(&self, _ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // TODO: Phase 2 - Implement response processing logic
        // For now, this is a no-op that just passes through
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ResponseProcessing"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_response_processing_stage_passthrough() {
        // TODO: Add tests in Phase 2
    }
}
