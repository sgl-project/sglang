//! Unified worker activation step.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::info;

use crate::{
    core::Worker,
    workflow::{StepExecutor, StepResult, WorkflowContext, WorkflowResult},
};

/// Unified step to activate workers by marking them as healthy.
///
/// This is the final step in any worker registration workflow.
pub struct ActivateWorkersStep;

#[async_trait]
impl StepExecutor for ActivateWorkersStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let workers: Arc<Vec<Arc<dyn Worker>>> = context.get_or_err("workers")?;

        for worker in workers.iter() {
            worker.set_healthy(true);
        }

        info!("Activated {} worker(s) (marked as healthy)", workers.len());

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &crate::workflow::WorkflowError) -> bool {
        false
    }
}
