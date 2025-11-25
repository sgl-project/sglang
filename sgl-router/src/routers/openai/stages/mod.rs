//! Pipeline stages for OpenAI router
//!
//! This module defines the pipeline stages that process requests sequentially.
//! Each stage implements the PipelineStage trait and has a specific responsibility.

use async_trait::async_trait;
use axum::response::Response;

use crate::routers::openai::context::RequestContext;

// ============================================================================
// Pipeline Stage Trait
// ============================================================================

/// Trait for pipeline stages that process requests
#[async_trait]
pub trait PipelineStage: Send + Sync {
    /// Execute this stage, mutating the context
    ///
    /// Returns:
    /// - `Ok(None)` - Continue to next stage
    /// - `Ok(Some(response))` - Pipeline complete, return this response (e.g., streaming)
    /// - `Err(response)` - Error occurred, return this error response
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response>;

    /// Stage name for logging
    fn name(&self) -> &'static str;
}

// ============================================================================
// Stage Modules (imported here)
// ============================================================================

mod validation;
mod discovery;
mod context_loading;
mod request_building;
mod mcp_preparation;
mod execution;
mod response_processing;
mod persistence;

// Export stage implementations
pub use validation::ValidationStage;
pub use discovery::ModelDiscoveryStage;
pub use context_loading::ContextLoadingStage;
pub use request_building::RequestBuildingStage;
pub use mcp_preparation::McpPreparationStage;
pub use execution::RequestExecutionStage;
pub use response_processing::ResponseProcessingStage;
pub use persistence::PersistenceStage;
