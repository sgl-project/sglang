//! Unified policy update step.

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use tracing::debug;

use crate::{
    app_context::AppContext,
    core::Worker,
    workflow::{StepExecutor, StepResult, WorkflowContext, WorkflowResult},
};

/// Unified step to update policy registry for registered workers.
///
/// Handles both local workers (same model, possibly DP-aware) and
/// external workers (different models per worker).
pub struct UpdatePoliciesStep;

#[async_trait]
impl StepExecutor for UpdatePoliciesStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;
        let workers: Arc<Vec<Arc<dyn Worker>>> = context.get_or_err("workers")?;
        let labels: Arc<HashMap<String, String>> = context.get_or_err("labels")?;

        let policy_hint = labels.get("policy").map(|s| s.as_str());

        // Track unique model IDs we've updated policies for
        let mut updated_models = Vec::new();

        for worker in workers.iter() {
            let model_id = worker.model_id().to_string();

            // Notify policy registry
            app_context
                .policy_registry
                .on_worker_added(&model_id, policy_hint);

            // Initialize cache-aware policy if configured
            let all_workers = app_context.worker_registry.get_by_model_fast(&model_id);
            if let Some(policy) = app_context.policy_registry.get_policy(&model_id) {
                if policy.name() == "cache_aware" {
                    app_context
                        .policy_registry
                        .init_cache_aware_policy(&model_id, &all_workers);
                }
            }

            if !updated_models.contains(&model_id) {
                updated_models.push(model_id);
            }
        }

        // Initialize bucket policies for prefill workers (local workers only)
        let prefill_workers = app_context.worker_registry.get_prefill_workers();
        if !prefill_workers.is_empty() {
            let policy = app_context.policy_registry.get_prefill_policy();
            if policy.name() == "bucket" {
                app_context
                    .policy_registry
                    .init_pd_bucket_policies(&prefill_workers);
            }
        }

        debug!(
            "Updated policies for {} workers across {} models",
            workers.len(),
            updated_models.len()
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &crate::workflow::WorkflowError) -> bool {
        false
    }
}
