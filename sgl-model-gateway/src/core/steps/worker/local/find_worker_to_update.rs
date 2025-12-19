//! Step to find a worker to update based on URL.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::debug;

use super::find_workers_by_url;
use crate::{
    app_context::AppContext,
    workflow::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

/// Step to find workers to update based on URL.
///
/// For DP-aware workers, finds all workers with matching URL prefix.
/// For regular workers, finds the single worker with exact URL match.
///
/// Expects the following context values:
/// - "worker_url": String - the URL of the worker to update
/// - "dp_aware": bool - whether to find all DP-aware workers with matching prefix
/// - "app_context": Arc<AppContext>
///
/// Sets the following context values:
/// - "workers_to_update": Vec<Arc<dyn Worker>>
pub struct FindWorkerToUpdateStep;

#[async_trait]
impl StepExecutor for FindWorkerToUpdateStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let worker_url: Arc<String> = context.get_or_err("worker_url")?;
        let dp_aware: Arc<bool> = context.get_or_err("dp_aware")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;

        let workers_to_update =
            find_workers_by_url(&app_context.worker_registry, &worker_url, *dp_aware);

        if workers_to_update.is_empty() {
            let error_msg = if *dp_aware {
                format!("No workers found with prefix {}@", *worker_url)
            } else {
                format!("Worker {} not found", *worker_url)
            };
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("find_worker_to_update"),
                message: error_msg,
            });
        }

        debug!(
            "Found {} worker(s) to update for {}",
            workers_to_update.len(),
            *worker_url
        );

        context.set("workers_to_update", workers_to_update);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
