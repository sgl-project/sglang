//! Unified worker registration step.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::debug;

use crate::{
    app_context::AppContext,
    core::Worker,
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

        context.set("worker_ids", worker_ids);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &crate::workflow::WorkflowError) -> bool {
        false
    }
}
