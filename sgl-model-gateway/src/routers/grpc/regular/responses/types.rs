//! Type definitions for /v1/responses endpoint

use std::sync::Arc;

use tokio::{sync::RwLock, task::JoinHandle};

/// Information stored for background tasks to enable end-to-end cancellation
///
/// This struct enables cancelling both the Rust task AND the Python scheduler processing.
/// The client field is lazily initialized during pipeline execution.
pub struct BackgroundTaskInfo {
    /// Tokio task handle for aborting the Rust task
    pub handle: JoinHandle<()>,
    /// gRPC request_id sent to Python scheduler (chatcmpl-* prefix)
    pub grpc_request_id: String,
    /// gRPC client for sending abort requests to Python (set after client acquisition)
    pub client: Arc<RwLock<Option<crate::grpc_client::SglangSchedulerClient>>>,
}
