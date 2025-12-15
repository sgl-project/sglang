//! Step to find workers to remove based on URL.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use tracing::debug;

use super::find_workers_by_url;
use crate::{
    app_context::AppContext,
    workflow::{StepExecutor, StepId, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

/// Request structure for worker removal.
#[derive(Debug, Clone)]
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
impl StepExecutor for FindWorkersToRemoveStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let request: Arc<WorkerRemovalRequest> = context.get_or_err("removal_request")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;

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

        context.set("workers_to_remove", workers_to_remove);
        context.set("worker_urls", worker_urls);
        context.set("affected_models", affected_models);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
