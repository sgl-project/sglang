//! MCP Preparation stage
//!
//! This stage:
//! - Detects MCP tools in the request
//! - Ensures dynamic MCP client exists
//! - Transforms MCP tools to function format
//! - Initializes tool loop state

use async_trait::async_trait;
use axum::response::Response;

use super::PipelineStage;
use crate::routers::openai::context::{RequestContext, McpOutput};

/// MCP preparation stage
pub struct McpPreparationStage;

#[async_trait]
impl PipelineStage for McpPreparationStage {
    async fn execute(&self, _ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // TODO: Phase 2 - Implement MCP preparation logic
        // For now, this is a no-op that just passes through
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "McpPreparation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_preparation_stage_passthrough() {
        // TODO: Add tests in Phase 2
    }
}
