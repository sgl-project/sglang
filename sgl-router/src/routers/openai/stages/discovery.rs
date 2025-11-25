//! Model Discovery stage
//!
//! This stage:
//! - Determines which endpoint has the requested model
//! - Uses caching to avoid repeated endpoint probing
//! - Handles single endpoint (fast path) vs multiple endpoints

use async_trait::async_trait;
use axum::response::Response;

use super::PipelineStage;
use crate::routers::openai::context::{RequestContext, DiscoveryOutput};

/// Model discovery stage
pub struct ModelDiscoveryStage {
    worker_urls: Vec<String>,
}

impl ModelDiscoveryStage {
    pub fn new(worker_urls: Vec<String>) -> Self {
        Self { worker_urls }
    }
}

#[async_trait]
impl PipelineStage for ModelDiscoveryStage {
    async fn execute(&self, _ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        // TODO: Phase 2 - Implement model discovery logic
        // For now, this is a no-op that just passes through
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ModelDiscovery"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_discovery_stage_passthrough() {
        // TODO: Add tests in Phase 2
    }
}
