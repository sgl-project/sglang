//! Request Execution stage
//!
//! This stage:
//! - Builds HTTP request to upstream OpenAI-compatible API
//! - Applies headers (auth, accept, content-type)
//! - Executes the request
//! - Handles errors and status codes

use async_trait::async_trait;
use axum::response::Response;

use super::PipelineStage;
use crate::routers::openai::context::{RequestContext, ExecutionResult};

/// Request execution stage
pub struct RequestExecutionStage;

#[async_trait]
impl PipelineStage for RequestExecutionStage {
    async fn execute(&self, _ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // TODO: Phase 2 - Implement request execution logic
        // For now, this is a no-op that just passes through
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "RequestExecution"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_execution_stage_passthrough() {
        // TODO: Add tests in Phase 2
    }
}
