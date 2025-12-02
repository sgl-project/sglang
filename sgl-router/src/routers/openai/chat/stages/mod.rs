//! Chat pipeline stages
//!
//! This module contains the 4 stages of the chat pipeline:
//! 1. Validation - Auth and circuit breaker checks
//! 2. ModelDiscovery - Find endpoint URL for model
//! 3. RequestBuilding - Strip SGLang fields, apply provider transformations
//! 4. RequestExecution - Execute HTTP request and track circuit breaker

use async_trait::async_trait;
use axum::response::Response;

use super::ChatRequestContext;

// Stage modules
pub mod model_discovery;
pub mod request_building;
pub mod request_execution;
pub mod validation;

// Re-export stage types
pub use model_discovery::ChatModelDiscoveryStage;
pub use request_building::ChatRequestBuildingStage;
pub use request_execution::ChatRequestExecutionStage;
pub use validation::ChatValidationStage;

/// Trait for chat pipeline stages
///
/// Unlike the legacy PipelineStage which returns Option<Response>,
/// chat stages are simpler: they either succeed (Ok) or fail with an error response (Err).
#[async_trait]
pub trait ChatStage: Send + Sync {
    /// Execute this stage
    ///
    /// Returns:
    /// - Ok(()) if stage completed successfully, pipeline continues
    /// - Err(Response) if stage failed, pipeline stops and returns error
    async fn execute(&self, ctx: &mut ChatRequestContext) -> Result<(), Response>;

    /// Get stage name for logging/debugging
    fn name(&self) -> &'static str;
}
