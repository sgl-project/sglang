//! Unified worker registration step.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use tracing::debug;

use crate::{
    app_context::AppContext,
    core::Worker,
    observability::metrics::Metrics,
    workflow::{StepExecutor, StepResult, WorkflowContext, WorkflowResult},
};

/// Unified step to register workers in the registry.
///
/// Works with both single workers and batches. Always expects `workers` key
/// in context containing `Vec<Arc<dyn Worker>>`.
pub struct RegisterWorkersStep;

#[async_trait]
impl StepExecutor for RegisterWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;
        let workers: Arc<Vec<Arc<dyn Worker>>> = context.get_or_err("workers")?;

        let mut worker_ids = Vec::with_capacity(workers.len());

        for worker in workers.iter() {
            let worker_id = app_context.worker_registry.register(Arc::clone(worker));
            debug!(
                "Registered worker {} (model: {}) with ID {:?}",
                worker.url(),
                worker.model_id(),
                worker_id
            );
            worker_ids.push(worker_id);
        }

        // Collect unique worker configurations to avoid redundant metric updates
        let unique_configs: HashSet<_> = workers
            .iter()
            .map(|w| {
                let meta = w.metadata();
                (
                    meta.worker_type.clone(),
                    meta.connection_mode.clone(),
                    w.model_id().to_string(),
                )
            })
            .collect();

        // Update Layer 3 worker pool size metrics per unique type/connection/model
        for (worker_type, connection_mode, model_id) in unique_configs {
            // Get labels before moving values into get_workers_filtered
            let worker_type_label = worker_type.as_metric_label();
            let connection_mode_label = connection_mode.as_metric_label();

            let pool_size = app_context
                .worker_registry
                .get_workers_filtered(
                    Some(&model_id),
                    Some(worker_type),
                    Some(connection_mode),
                    None,
                    false,
                )
                .len();

            Metrics::set_worker_pool_size(
                worker_type_label,
                connection_mode_label,
                &model_id,
                pool_size,
            );
        }

        context.set("worker_ids", worker_ids);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &crate::workflow::WorkflowError) -> bool {
        false
    }
}
