//! Common pipeline stages shared across all endpoints and model types
//!
//! These stages are endpoint-agnostic and model-agnostic:
//! - Worker selection
//! - Client acquisition
//! - Dispatch metadata generation
//! - Request execution

mod client_acquisition;
mod dispatch_metadata;
mod request_execution;
mod worker_selection;

pub use client_acquisition::ClientAcquisitionStage;
pub use dispatch_metadata::DispatchMetadataStage;
pub use request_execution::{ExecutionMode, RequestExecutionStage};
pub use worker_selection::{WorkerSelectionMode, WorkerSelectionStage};
