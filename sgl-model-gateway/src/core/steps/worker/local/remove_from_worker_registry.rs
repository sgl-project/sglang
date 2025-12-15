//! Step to remove workers from worker registry.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, warn};

use crate::{
    app_context::AppContext,
    observability::metrics::RouterMetrics,
    workflow::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

/// Step to remove workers from the worker registry.
///
/// Removes each worker by URL from the central worker registry.
pub struct RemoveFromWorkerRegistryStep;

#[async_trait]
impl StepExecutor for RemoveFromWorkerRegistryStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;
        let worker_urls: Arc<Vec<String>> = context.get_or_err("worker_urls")?;

        debug!(
            "Removing {} worker(s) from worker registry",
            worker_urls.len()
        );

        let mut removed_count = 0;
        for worker_url in worker_urls.iter() {
            if app_context
                .worker_registry
                .remove_by_url(worker_url)
                .is_some()
            {
                removed_count += 1;
            }
        }

        // Log if some workers were already removed (e.g., by another process)
        if removed_count != worker_urls.len() {
            warn!(
                "Removed {} of {} workers (some may have been removed by another process)",
                removed_count,
                worker_urls.len()
            );
        } else {
            debug!("Removed {} worker(s) from registry", removed_count);
        }

        // Update active workers metric
        RouterMetrics::set_active_workers(app_context.worker_registry.len());

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
