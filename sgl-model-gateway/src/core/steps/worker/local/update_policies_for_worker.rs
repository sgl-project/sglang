//! Step to update policies for updated workers.

use std::collections::HashSet;

use async_trait::async_trait;
use tracing::debug;

use crate::{
    core::steps::workflow_data::WorkerUpdateWorkflowData,
    workflow::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

/// Step to update policies for updated workers.
///
/// After workers are updated, this step re-initializes cache-aware policies
/// for the affected models to reflect any priority or label changes.
pub struct UpdatePoliciesForWorkerStep;

#[async_trait]
impl StepExecutor<WorkerUpdateWorkflowData> for UpdatePoliciesForWorkerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerUpdateWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let updated_workers =
            context.data.updated_workers.as_ref().ok_or_else(|| {
                WorkflowError::ContextValueNotFound("updated_workers".to_string())
            })?;

        // Collect affected models
        let affected_models: HashSet<String> = updated_workers
            .iter()
            .map(|w| w.model_id().to_string())
            .collect();

        debug!(
            "Updating policies for {} affected model(s) after worker update",
            affected_models.len()
        );

        for model_id in &affected_models {
            let workers = app_context.worker_registry.get_by_model(model_id);

            if let Some(policy) = app_context.policy_registry.get_policy(model_id) {
                if policy.name() == "cache_aware" && !workers.is_empty() {
                    // Re-initialize cache-aware policy with updated workers
                    app_context
                        .policy_registry
                        .init_cache_aware_policy(model_id, &workers);

                    debug!(
                        "Updated cache-aware policy for model {} ({} workers)",
                        model_id,
                        workers.len()
                    );
                }
            }

            // Notify policy registry of the update
            app_context.policy_registry.on_worker_added(model_id, None);
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
