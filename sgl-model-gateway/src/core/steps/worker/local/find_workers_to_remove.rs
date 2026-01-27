//! Step to find workers to remove based on URL.

use std::collections::HashSet;

use async_trait::async_trait;
use tracing::debug;

use super::find_workers_by_url;
use crate::{
    core::steps::workflow_data::{WorkerList, WorkerRemovalWorkflowData},
    workflow::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

/// Request structure for worker removal.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkerRemovalRequest {
    pub url: String,
    pub dp_aware: bool,
}

/// Step to find workers to remove based on URL.
///
/// For DP-aware workers, finds all workers with matching URL prefix.
/// For regular workers, finds the single worker with exact URL match.
pub struct FindWorkersToRemoveStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for FindWorkersToRemoveStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let request = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

        let workers_to_remove =
            find_workers_by_url(&app_context.worker_registry, &request.url, request.dp_aware);

        if workers_to_remove.is_empty() {
            let error_msg = if request.dp_aware {
                format!("No workers found with prefix {}@", request.url)
            } else {
                format!("Worker {} not found", request.url)
            };
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("find_workers_to_remove"),
                message: error_msg,
            });
        }

        debug!(
            "Found {} worker(s) to remove for {}",
            workers_to_remove.len(),
            request.url
        );

        // Store workers and their model IDs for subsequent steps
        let worker_urls: Vec<String> = workers_to_remove
            .iter()
            .map(|w| w.url().to_string())
            .collect();

        let affected_models: HashSet<String> = workers_to_remove
            .iter()
            .map(|w| w.model_id().to_string())
            .collect();

        // Update workflow data
        context.data.workers_to_remove = Some(WorkerList::from_workers(&workers_to_remove));
        context.data.actual_workers_to_remove = Some(workers_to_remove);
        context.data.worker_urls = worker_urls;
        context.data.affected_models = affected_models;

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
