//! Step to remove workers from worker registry.

use std::{collections::HashSet, sync::Arc};

use async_trait::async_trait;
use tracing::{debug, warn};

use crate::{
    app_context::AppContext,
    observability::metrics::Metrics,
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

        // Collect unique worker configurations before removal for pool size updates
        let unique_configs: HashSet<_> = worker_urls
            .iter()
            .filter_map(|url| app_context.worker_registry.get_by_url(url))
            .map(|w| {
                let meta = w.metadata();
                (
                    meta.worker_type.clone(),
                    meta.connection_mode.clone(),
                    w.model_id().to_string(),
                )
            })
            .collect();

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

        // Update Layer 3 worker pool size metrics for unique configurations
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

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
