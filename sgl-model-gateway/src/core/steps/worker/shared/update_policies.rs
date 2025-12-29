//! Unified policy update step.

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use tracing::{debug, warn};

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

impl UpdatePoliciesStep {
    /// Check for conflicts between prefill and decode worker configurations for a model.
    fn check_worker_conflicts(&self, model_id: &str, workers: &[Arc<dyn Worker>]) {
        let mut prefill_workers = Vec::new();
        let mut decode_workers = Vec::new();

        for worker in workers {
            let labels = &worker.metadata().labels;
            let disagg_mode = labels.get("disaggregation_mode").map(|s| s.as_str());

            match disagg_mode {
                Some("prefill") => prefill_workers.push(worker),
                Some("decode") => decode_workers.push(worker),
                _ => {}
            }
        }

        if prefill_workers.is_empty() || decode_workers.is_empty() {
            return;
        }

        // Compare configurations of prefill vs decode workers
        // We take the first of each as representative for the check
        if let (Some(pw), Some(dw)) = (prefill_workers.first(), decode_workers.first()) {
            let pl = &pw.metadata().labels;
            let dl = &dw.metadata().labels;

            let ptp = pl.get("tp_size");
            let dtp = dl.get("tp_size");
            if ptp != dtp {
                warn!(
                    "Model {} has conflicting tp_size: prefill={:?}, decode={:?}",
                    model_id, ptp, dtp
                );
            }

            let pdp = pl.get("dp_size");
            let ddp = dl.get("dp_size");
            if pdp != ddp {
                warn!(
                    "Model {} has conflicting dp_size: prefill={:?}, decode={:?}",
                    model_id, pdp, ddp
                );
            }

            let plb = pl.get("load_balance_method");
            let dlb = dl.get("load_balance_method");
            if plb != dlb {
                warn!(
                    "Model {} has conflicting load_balance_method: prefill={:?}, decode={:?}",
                    model_id, plb, dlb
                );
            }

            // Specific check for the deprecated prefill_round_robin_balance flag
            if let Some(dp_size_str) = pdp {
                if let Ok(dp_size) = dp_size_str.parse::<usize>() {
                    if dp_size > 1 {
                        let prrb = pl.get("prefill_round_robin_balance");
                        if prrb.map(|s| s.as_str()) != Some("true") {
                            warn!(
                                "Model {} has dp_size > 1 but prefill_round_robin_balance is not enabled on prefill workers. This may cause rank mismatch.",
                                model_id
                            );
                        }
                    }
                }
            }
        }
    }
}

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

            // Check for configuration conflicts between prefill and decode
            self.check_worker_conflicts(&model_id, &all_workers);

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
