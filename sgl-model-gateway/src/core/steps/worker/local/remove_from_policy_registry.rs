//! Step to remove workers from policy registry.

use async_trait::async_trait;
use tracing::debug;

use crate::{
    core::steps::workflow_data::WorkerRemovalWorkflowData,
    workflow::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

/// Step to remove workers from the policy registry.
///
/// Removes each worker from cache-aware policies and notifies
/// the policy registry of worker removal.
pub struct RemoveFromPolicyRegistryStep;

#[async_trait]
impl StepExecutor<WorkerRemovalWorkflowData> for RemoveFromPolicyRegistryStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let workers_to_remove = context
            .data
            .actual_workers_to_remove
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers_to_remove".to_string()))?;

        debug!(
            "Removing {} worker(s) from policy registry",
            workers_to_remove.len()
        );

        for worker in workers_to_remove.iter() {
            let model_id = worker.model_id().to_string();
            let worker_url = worker.url();

            // Remove from cache-aware policy
            app_context
                .policy_registry
                .remove_worker_from_cache_aware(&model_id, worker_url);

            // Notify policy registry
            app_context.policy_registry.on_worker_removed(&model_id);
        }

        debug!(
            "Removed {} worker(s) from policy registry",
            workers_to_remove.len()
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
