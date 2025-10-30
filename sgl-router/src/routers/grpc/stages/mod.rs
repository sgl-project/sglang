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

mod client_acquisition;
mod dispatch_metadata;
mod preparation;
mod request_building;
mod request_execution;
mod response_processing;
mod worker_selection;

// ============================================================================
// Public Exports
// ============================================================================

pub use client_acquisition::ClientAcquisitionStage;
pub use dispatch_metadata::DispatchMetadataStage;
pub use preparation::PreparationStage;
pub use request_building::RequestBuildingStage;
pub use request_execution::{ExecutionMode, RequestExecutionStage};
pub use response_processing::ResponseProcessingStage;
pub use worker_selection::{WorkerSelectionMode, WorkerSelectionStage};
