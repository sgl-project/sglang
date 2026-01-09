//! Common pipeline stages shared across all endpoints and model types
//!
//! These stages are endpoint-agnostic and model-agnostic:
//! - Worker selection
//! - Client acquisition
//! - Dispatch metadata generation
//! - Request execution

use async_trait::async_trait;
use axum::response::Response;

use crate::routers::grpc::context::RequestContext;

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

mod client_acquisition;
mod dispatch_metadata;
pub mod helpers;
mod request_execution;
mod worker_selection;

// Export stage implementations
pub use client_acquisition::ClientAcquisitionStage;
pub use dispatch_metadata::DispatchMetadataStage;
pub use request_execution::{ExecutionMode, RequestExecutionStage};
pub use worker_selection::{WorkerSelectionMode, WorkerSelectionStage};
