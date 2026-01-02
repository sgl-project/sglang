//! Worker Removal Workflow Steps
//!
//! This module implements the workflow steps for removing workers from the router.
//! Handles both single worker removal and DP-aware worker removal with prefix matching.
//!
//! Steps:
//! 1. FindWorkersToRemove - Identify workers to remove based on URL (handles DP-aware prefix matching)
//! 2. RemoveFromPolicyRegistry - Remove workers from policy registry and cache-aware policies
//! 3. RemoveFromWorkerRegistry - Remove workers from worker registry
//! 4. UpdateRemainingPolicies - Update cache-aware policies for remaining workers

use std::{collections::HashSet, sync::Arc, time::Duration};

use async_trait::async_trait;
use tracing::{debug, info};

use crate::{
    app_context::AppContext,
    core::{workflow::*, Worker},
};

/// Request structure for worker removal
#[derive(Debug, Clone)]
pub struct WorkerRemovalRequest {
    pub url: String,
    pub dp_aware: bool,
}

/// Step 1: Find workers to remove based on URL
pub struct FindWorkersToRemoveStep;

#[async_trait]
impl StepExecutor for FindWorkersToRemoveStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let request: Arc<WorkerRemovalRequest> = context.get_or_err("removal_request")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;

        debug!(
            "Finding workers to remove for {} (dp_aware: {})",
            request.url, request.dp_aware
        );

        let workers_to_remove: Vec<Arc<dyn Worker>> = if request.dp_aware {
            // DP-aware: Find all workers with matching prefix
            let worker_url_prefix = format!("{}@", request.url);
            let all_workers = app_context.worker_registry.get_all();

            all_workers
                .iter()
                .filter(|worker| worker.url().starts_with(&worker_url_prefix))
                .cloned()
                .collect()
        } else {
            // Non-DP-aware: Find single worker by exact URL
            match app_context.worker_registry.get_by_url(&request.url) {
                Some(worker) => vec![worker],
                None => Vec::new(),
            }
        };

        if workers_to_remove.is_empty() {
            let error_msg = if request.dp_aware {
                format!("No workers found with prefix {}@", request.url)
            } else {
                format!("Worker {} not found", request.url)
            };
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("find_workers_to_remove"),
                message: error_msg,
            });
        }

        debug!(
            "Found {} worker(s) to remove for {}",
            workers_to_remove.len(),
            request.url
        );

        // Store workers and their model IDs for subsequent steps
        let worker_urls: Vec<String> = workers_to_remove
            .iter()
            .map(|w| w.url().to_string())
            .collect();

        let affected_models: HashSet<String> = workers_to_remove
            .iter()
            .map(|w| w.model_id().to_string())
            .collect();

        context.set("workers_to_remove", workers_to_remove);
        context.set("worker_urls", worker_urls);
        context.set("affected_models", affected_models);

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Worker not found is not retryable
    }
}

/// Step 2: Remove workers from policy registry
pub struct RemoveFromPolicyRegistryStep;

#[async_trait]
impl StepExecutor for RemoveFromPolicyRegistryStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;
        let workers_to_remove: Arc<Vec<Arc<dyn Worker>>> =
            context.get_or_err("workers_to_remove")?;

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

            debug!(
                "Removed worker {} from policy registry (model: {})",
                worker_url, model_id
            );
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Policy removal is not retryable
    }
}

/// Step 3: Remove workers from worker registry
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
                debug!("Removed worker {} from registry", worker_url);
            }
        }

        if removed_count != worker_urls.len() {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("remove_from_worker_registry"),
                message: format!(
                    "Expected to remove {} workers but only removed {}",
                    worker_urls.len(),
                    removed_count
                ),
            });
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Worker removal is not retryable
    }
}

/// Step 4: Update cache-aware policies for remaining workers
pub struct UpdateRemainingPoliciesStep;

#[async_trait]
impl StepExecutor for UpdateRemainingPoliciesStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;
        let affected_models: Arc<HashSet<String>> = context.get_or_err("affected_models")?;
        let worker_urls: Arc<Vec<String>> = context.get_or_err("worker_urls")?;

        debug!(
            "Updating cache-aware policies for {} affected model(s)",
            affected_models.len()
        );

        for model_id in affected_models.iter() {
            let remaining_workers = app_context.worker_registry.get_by_model_fast(model_id);

            if let Some(policy) = app_context.policy_registry.get_policy(model_id) {
                if policy.name() == "cache_aware" && !remaining_workers.is_empty() {
                    app_context
                        .policy_registry
                        .init_cache_aware_policy(model_id, &remaining_workers);

                    debug!(
                        "Updated cache-aware policy for model {} ({} remaining workers)",
                        model_id,
                        remaining_workers.len()
                    );
                }
            }
        }

        // Log final result at info level
        if worker_urls.len() == 1 {
            info!("Removed worker {}", worker_urls[0]);
        } else {
            info!(
                "Removed {} DP-aware workers: {:?}",
                worker_urls.len(),
                worker_urls
            );
        }

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Policy update is not retryable
    }
}

/// Create a worker removal workflow definition
pub fn create_worker_removal_workflow() -> WorkflowDefinition {
    WorkflowDefinition::new("worker_removal", "Remove worker from router")
        .add_step(
            StepDefinition::new(
                "find_workers_to_remove",
                "Find workers to remove",
                Arc::new(FindWorkersToRemoveStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            }),
        )
        .add_step(
            StepDefinition::new(
                "remove_from_policy_registry",
                "Remove workers from policy registry",
                Arc::new(RemoveFromPolicyRegistryStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            }),
        )
        .add_step(
            StepDefinition::new(
                "remove_from_worker_registry",
                "Remove workers from worker registry",
                Arc::new(RemoveFromWorkerRegistryStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            }),
        )
        .add_step(
            StepDefinition::new(
                "update_remaining_policies",
                "Update cache-aware policies for remaining workers",
                Arc::new(UpdateRemainingPoliciesStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            }),
        )
}
