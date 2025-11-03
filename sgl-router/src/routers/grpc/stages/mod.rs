//! Pipeline stages for gRPC router request processing
//!
//! This module defines the core pipeline abstraction and individual processing stages
//! that transform a RequestContext through its lifecycle.

use async_trait::async_trait;
use axum::response::Response;

use crate::routers::grpc::context::RequestContext;

// ============================================================================
// Pipeline Trait
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
// Stage Modules
// ============================================================================

pub mod chat;
pub mod common;
pub mod generate;
mod preparation;
mod request_building;
mod response_processing;

// ============================================================================
// Public Exports
// ============================================================================

// Export common stages (endpoint-agnostic)
pub use common::{
    ClientAcquisitionStage, DispatchMetadataStage, ExecutionMode, RequestExecutionStage,
    WorkerSelectionMode, WorkerSelectionStage,
};

// Export endpoint-aware stages (used by pipeline.rs)
pub use preparation::PreparationStage;
pub use request_building::RequestBuildingStage;
pub use response_processing::ResponseProcessingStage;

// Export endpoint-specific stages (available for direct use if needed)
pub use chat::{ChatPreparationStage, ChatRequestBuildingStage, ChatResponseProcessingStage};
pub use generate::{
    GeneratePreparationStage, GenerateRequestBuildingStage, GenerateResponseProcessingStage,
};
