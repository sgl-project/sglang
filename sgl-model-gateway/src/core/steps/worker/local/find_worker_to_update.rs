//! Step to find a worker to update based on URL.

use async_trait::async_trait;
use tracing::debug;

use super::find_workers_by_url;
use crate::{
    core::steps::workflow_data::WorkerUpdateWorkflowData,
    workflow::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

/// Step to find workers to update based on URL.
///
/// For DP-aware workers, finds all workers with matching URL prefix.
/// For regular workers, finds the single worker with exact URL match.
pub struct FindWorkerToUpdateStep;

#[async_trait]
impl StepExecutor<WorkerUpdateWorkflowData> for FindWorkerToUpdateStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerUpdateWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let worker_url = &context.data.worker_url;
        let dp_aware = context.data.dp_aware;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        let workers_to_update =
            find_workers_by_url(&app_context.worker_registry, worker_url, dp_aware);

        if workers_to_update.is_empty() {
            let error_msg = if dp_aware {
                format!("No workers found with prefix {}@", worker_url)
            } else {
                format!("Worker {} not found", worker_url)
            };
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("find_worker_to_update"),
                message: error_msg,
            });
        }

        debug!(
            "Found {} worker(s) to update for {}",
            workers_to_update.len(),
            worker_url
        );

        context.data.workers_to_update = Some(workers_to_update);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
