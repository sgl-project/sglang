mod create_worker;
mod detect_connection;
mod discover_dp;
mod discover_metadata;
mod find_worker_to_update;
mod find_workers_to_remove;
mod remove_from_policy_registry;
mod remove_from_worker_registry;
mod submit_tokenizer_job;
mod update_policies_for_worker;
mod update_remaining_policies;
mod update_worker_properties;

use std::{sync::Arc, time::Duration};

/// Strip protocol prefix (http://, https://, grpc://) from URL.
pub(crate) fn strip_protocol(url: &str) -> String {
    url.trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_start_matches("grpc://")
        .to_string()
}

pub use create_worker::CreateLocalWorkerStep;
pub use detect_connection::DetectConnectionModeStep;
pub use discover_dp::{get_dp_info, DiscoverDPInfoStep, DpInfo};
pub use discover_metadata::DiscoverMetadataStep;
pub use find_worker_to_update::FindWorkerToUpdateStep;
pub use find_workers_to_remove::{FindWorkersToRemoveStep, WorkerRemovalRequest};
pub use remove_from_policy_registry::RemoveFromPolicyRegistryStep;
pub use remove_from_worker_registry::RemoveFromWorkerRegistryStep;
pub use submit_tokenizer_job::SubmitTokenizerJobStep;
pub use update_policies_for_worker::UpdatePoliciesForWorkerStep;
pub use update_remaining_policies::UpdateRemainingPoliciesStep;
pub use update_worker_properties::UpdateWorkerPropertiesStep;

use super::shared::{ActivateWorkersStep, RegisterWorkersStep, UpdatePoliciesStep};
use crate::{
    app_context::AppContext,
    config::RouterConfig,
    core::{
        steps::workflow_data::{
            LocalWorkerWorkflowData, WorkerRemovalWorkflowData, WorkerUpdateWorkflowData,
        },
        Worker, WorkerRegistry,
    },
    protocols::worker_spec::{WorkerConfigRequest, WorkerUpdateRequest},
    workflow::{BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, WorkflowDefinition},
};

/// Find workers by URL, supporting both DP-aware (prefix match) and regular (exact match) modes.
///
/// For DP-aware workers, finds all workers with URL prefix `{url}@`.
/// For regular workers, finds the single worker with exact URL match.
pub(crate) fn find_workers_by_url(
    registry: &WorkerRegistry,
    url: &str,
    dp_aware: bool,
) -> Vec<Arc<dyn Worker>> {
    if dp_aware {
        let worker_url_prefix = format!("{}@", url);
        registry
            .get_all()
            .iter()
            .filter(|worker| worker.url().starts_with(&worker_url_prefix))
            .cloned()
            .collect()
    } else {
        match registry.get_by_url(url) {
            Some(worker) => vec![worker],
            None => Vec::new(),
        }
    }
}

pub fn create_local_worker_workflow(
    router_config: &RouterConfig,
) -> WorkflowDefinition<LocalWorkerWorkflowData> {
    let detect_timeout = Duration::from_secs(router_config.worker_startup_timeout_secs);

    // Calculate max_attempts based on timeout
    let timeout_secs = detect_timeout.as_secs() as f64;
    let effective_timeout = timeout_secs * 0.9;
    let max_attempts = if effective_timeout > 10.0 {
        (5 + ((effective_timeout - 10.0) / 5.0).ceil() as u32).max(3)
    } else {
        3
    };

    WorkflowDefinition::new("local_worker_registration", "Local Worker Registration")
        // Step 1: Detect connection mode (HTTP vs gRPC)
        .add_step(
            StepDefinition::new(
                "detect_connection_mode",
                "Detect Connection Mode",
                Arc::new(DetectConnectionModeStep),
            )
            .with_retry(RetryPolicy {
                max_attempts,
                backoff: BackoffStrategy::Linear {
                    increment: Duration::from_secs(1),
                    max: Duration::from_secs(5),
                },
            })
            .with_timeout(detect_timeout)
            .with_failure_action(FailureAction::FailWorkflow),
        )
        // Step 2a: Discover metadata (parallel with DP discovery)
        .add_step(
            StepDefinition::new(
                "discover_metadata",
                "Discover Metadata",
                Arc::new(DiscoverMetadataStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["detect_connection_mode"]),
        )
        // Step 2b: Discover DP info (after metadata to avoid concurrent /server_info calls)
        .add_step(
            StepDefinition::new(
                "discover_dp_info",
                "Discover DP Info",
                Arc::new(DiscoverDPInfoStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(10))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["discover_metadata"]),
        )
        // Step 3: Create worker(s)
        .add_step(
            StepDefinition::new(
                "create_worker",
                "Create Worker",
                Arc::new(CreateLocalWorkerStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["discover_dp_info"]),
        )
        // Step 4: Register workers (shared step)
        .add_step(
            StepDefinition::new(
                "register_workers",
                "Register Workers",
                Arc::new(RegisterWorkersStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["create_worker"]),
        )
        .add_step(
            StepDefinition::new(
                "submit_tokenizer_job",
                "Submit Tokenizer Job",
                Arc::new(SubmitTokenizerJobStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["register_workers"]),
        )
        // Step 5a: Update policies (parallel with activation)
        .add_step(
            StepDefinition::new(
                "update_policies",
                "Update Policies",
                Arc::new(UpdatePoliciesStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::ContinueNextStep)
            .depends_on(&["register_workers"]),
        )
        // Step 5b: Activate workers (parallel with policy update)
        .add_step(
            StepDefinition::new(
                "activate_workers",
                "Activate Workers",
                Arc::new(ActivateWorkersStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["register_workers"]),
        )
}

/// Create a worker removal workflow definition.
///
/// DAG structure:
/// ```text
///     find_workers_to_remove
///              │
///     remove_from_policy_registry
///              │
///     remove_from_worker_registry
///              │
///     update_remaining_policies
/// ```
pub fn create_worker_removal_workflow() -> WorkflowDefinition<WorkerRemovalWorkflowData> {
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
            })
            .depends_on(&["find_workers_to_remove"]),
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
            })
            .depends_on(&["remove_from_policy_registry"]),
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
            })
            .depends_on(&["remove_from_worker_registry"]),
        )
}

/// Create a worker update workflow definition.
///
/// DAG structure:
/// ```text
///     find_worker_to_update
///              │
///     update_worker_properties
///              │
///     update_policies_for_worker
/// ```
pub fn create_worker_update_workflow() -> WorkflowDefinition<WorkerUpdateWorkflowData> {
    WorkflowDefinition::new("worker_update", "Update worker properties")
        .add_step(
            StepDefinition::new(
                "find_worker_to_update",
                "Find worker to update",
                Arc::new(FindWorkerToUpdateStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            }),
        )
        .add_step(
            StepDefinition::new(
                "update_worker_properties",
                "Update worker properties",
                Arc::new(UpdateWorkerPropertiesStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            })
            .depends_on(&["find_worker_to_update"]),
        )
        .add_step(
            StepDefinition::new(
                "update_policies_for_worker",
                "Update policies for updated worker",
                Arc::new(UpdatePoliciesForWorkerStep),
            )
            .with_timeout(Duration::from_secs(10))
            .with_retry(RetryPolicy {
                max_attempts: 1,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(0)),
            })
            .depends_on(&["update_worker_properties"]),
        )
}

/// Helper to create initial workflow data for local worker registration
pub fn create_local_worker_workflow_data(
    config: WorkerConfigRequest,
    app_context: Arc<AppContext>,
) -> LocalWorkerWorkflowData {
    LocalWorkerWorkflowData {
        config,
        connection_mode: None,
        discovered_labels: std::collections::HashMap::new(),
        dp_info: None,
        workers: None,
        final_labels: std::collections::HashMap::new(),
        detected_runtime_type: None,
        app_context: Some(app_context),
        actual_workers: None,
    }
}

/// Helper to create initial workflow data for worker removal
pub fn create_worker_removal_workflow_data(
    url: String,
    dp_aware: bool,
    app_context: Arc<AppContext>,
) -> WorkerRemovalWorkflowData {
    WorkerRemovalWorkflowData {
        config: WorkerRemovalRequest { url, dp_aware },
        workers_to_remove: None,
        worker_urls: Vec::new(),
        affected_models: std::collections::HashSet::new(),
        app_context: Some(app_context),
        actual_workers_to_remove: None,
    }
}

/// Helper to create initial workflow data for worker update
pub fn create_worker_update_workflow_data(
    worker_url: String,
    update_config: WorkerUpdateRequest,
    app_context: Arc<AppContext>,
) -> WorkerUpdateWorkflowData {
    // Determine if this is a DP-aware update based on URL pattern
    let dp_aware = worker_url.contains('@');
    WorkerUpdateWorkflowData {
        config: update_config,
        worker_url,
        dp_aware,
        app_context: Some(app_context),
        workers_to_update: None,
        updated_workers: None,
    }
}
