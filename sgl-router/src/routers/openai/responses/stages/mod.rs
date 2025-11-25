//! Responses pipeline stages
//!
//! This module contains the 8 stages of the responses pipeline:
//! 1. Validation - Auth and circuit breaker checks
//! 2. ModelDiscovery - Find endpoint URL for model
//! 3. HistoryLoading - Load conversation history and previous response
//! 4. RequestBuilding - Build request payload with history
//! 5. McpPreparation - Setup MCP tool loop state
//! 6. RequestExecution - Execute HTTP request
//! 7. ResponseProcessing - Parse response, handle MCP tool calls
//! 8. Persistence - Save response and conversation items

use async_trait::async_trait;
use axum::response::Response;

use super::ResponsesRequestContext;

// Stage modules
pub mod history_loading;
pub mod mcp_preparation;
pub mod model_discovery;
pub mod persistence;
pub mod request_building;
pub mod request_execution;
pub mod response_processing;
pub mod validation;

// Re-export stage types
pub use history_loading::ResponsesHistoryLoadingStage;
pub use mcp_preparation::ResponsesMcpPreparationStage;
pub use model_discovery::ResponsesModelDiscoveryStage;
pub use persistence::ResponsesPersistenceStage;
pub use request_building::ResponsesRequestBuildingStage;
pub use request_execution::ResponsesRequestExecutionStage;
pub use response_processing::ResponsesResponseProcessingStage;
pub use validation::ResponsesValidationStage;

/// Trait for responses pipeline stages
///
/// Responses stages are more complex than chat stages and may need to
/// return early responses (for streaming with MCP tool loops).
#[async_trait]
pub trait ResponsesStage: Send + Sync {
    /// Execute this stage
    ///
    /// Returns:
    /// - Ok(None) if stage completed successfully, pipeline continues to next stage
    /// - Ok(Some(Response)) if stage needs to return early (e.g., streaming with MCP)
    /// - Err(Response) if stage failed, pipeline stops and returns error
    async fn execute(
        &self,
        ctx: &mut ResponsesRequestContext,
    ) -> Result<Option<Response>, Response>;

    /// Get stage name for logging/debugging
    fn name(&self) -> &'static str;
}
