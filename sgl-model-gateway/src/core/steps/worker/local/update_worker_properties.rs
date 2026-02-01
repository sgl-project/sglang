//! Step to update worker properties.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, info};

use crate::{
    core::{
        steps::workflow_data::WorkerUpdateWorkflowData, BasicWorkerBuilder, HealthConfig, Worker,
    },
    workflow::{StepExecutor, StepResult, WorkflowContext, WorkflowError, WorkflowResult},
};

/// Step to update worker properties.
///
/// This step creates new worker instances with updated properties and
/// re-registers them to replace the old workers in the registry.
pub struct UpdateWorkerPropertiesStep;

#[async_trait]
impl StepExecutor<WorkerUpdateWorkflowData> for UpdateWorkerPropertiesStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WorkerUpdateWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let request = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();
        let workers_to_update = context
            .data
            .workers_to_update
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("workers_to_update".to_string()))?
            .clone();

        debug!(
            "Updating properties for {} worker(s)",
            workers_to_update.len()
        );

        let mut updated_workers: Vec<Arc<dyn Worker>> = Vec::with_capacity(workers_to_update.len());

        for worker in workers_to_update.iter() {
            // Build updated labels - merge new labels into existing ones
            let mut updated_labels = worker.metadata().labels.clone();
            if let Some(ref new_labels) = request.labels {
                for (key, value) in new_labels {
                    updated_labels.insert(key.clone(), value.clone());
                }
            }

            // Update priority if specified (stored in labels)
            if let Some(priority) = request.priority {
                updated_labels.insert("priority".to_string(), priority.to_string());
            }

            // Update cost if specified (stored in labels)
            if let Some(cost) = request.cost {
                updated_labels.insert("cost".to_string(), cost.to_string());
            }

            // Build updated health config
            let existing_health = &worker.metadata().health_config;
            let updated_health_config = HealthConfig {
                timeout_secs: request
                    .health_check_timeout_secs
                    .unwrap_or(existing_health.timeout_secs),
                check_interval_secs: request
                    .health_check_interval_secs
                    .unwrap_or(existing_health.check_interval_secs),
                endpoint: existing_health.endpoint.clone(),
                failure_threshold: request
                    .health_failure_threshold
                    .unwrap_or(existing_health.failure_threshold),
                success_threshold: request
                    .health_success_threshold
                    .unwrap_or(existing_health.success_threshold),
                disable_health_check: request
                    .disable_health_check
                    .unwrap_or(existing_health.disable_health_check),
            };

            // Determine API key: use new one if provided, otherwise keep existing
            let updated_api_key = request
                .api_key
                .clone()
                .or_else(|| worker.metadata().api_key.clone());

            // Create a new worker with updated properties
            let new_worker: Arc<dyn Worker> = if worker.is_dp_aware() {
                // For DP-aware workers, extract DP info and rebuild
                let dp_rank = worker.dp_rank().unwrap_or(0);
                let dp_size = worker.dp_size().unwrap_or(1);
                let base_url = worker.base_url().to_string();

                let mut builder =
                    crate::core::DPAwareWorkerBuilder::new(base_url, dp_rank, dp_size)
                        .worker_type(worker.worker_type().clone())
                        .connection_mode(worker.connection_mode().clone())
                        .runtime_type(worker.metadata().runtime_type.clone())
                        .labels(updated_labels)
                        .health_config(updated_health_config.clone())
                        .models(worker.metadata().models.clone());

                if let Some(ref api_key) = updated_api_key {
                    builder = builder.api_key(api_key.clone());
                }

                Arc::new(builder.build())
            } else {
                // For basic workers, rebuild with updated properties
                let mut builder = BasicWorkerBuilder::new(worker.url())
                    .worker_type(worker.worker_type().clone())
                    .connection_mode(worker.connection_mode().clone())
                    .runtime_type(worker.metadata().runtime_type.clone())
                    .labels(updated_labels)
                    .health_config(updated_health_config.clone())
                    .models(worker.metadata().models.clone());

                if let Some(ref api_key) = updated_api_key {
                    builder = builder.api_key(api_key.clone());
                }

                Arc::new(builder.build())
            };

            // Re-register the worker (this replaces the old one)
            app_context.worker_registry.register(new_worker.clone());

            updated_workers.push(new_worker);
        }

        // Log result
        if updated_workers.len() == 1 {
            info!("Updated worker {}", updated_workers[0].url());
        } else {
            info!("Updated {} workers", updated_workers.len());
        }

        // Store updated workers for subsequent steps
        context.data.updated_workers = Some(updated_workers);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false
    }
}
