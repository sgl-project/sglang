//! External worker registration steps for OpenAI-compatible API endpoints.
//!
//! These steps handle the discovery and creation of workers that connect to
//! external API providers (OpenAI, Anthropic, etc.) via HTTP.

mod create_workers;
mod discover_models;

use std::{sync::Arc, time::Duration};

pub use create_workers::CreateExternalWorkersStep;
pub use discover_models::{
    group_models_into_cards, infer_model_type_from_id, DiscoverModelsStep, ModelInfo,
    ModelsResponse,
};

use super::shared::{ActivateWorkersStep, RegisterWorkersStep, UpdatePoliciesStep};
use crate::workflow::{
    BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, WorkflowDefinition,
};

/// Create external worker registration workflow definition.
///
/// DAG structure with parallel execution opportunities:
/// ```text
///              discover_models
///                    │
///              create_workers
///                    │
///             register_workers
///                    │
///       ┌────────────┴────────────┐
///       │                         │
///  update_policies         activate_workers
///       │                         │
///       └────────────┴────────────┘
/// ```
pub fn create_external_worker_workflow() -> WorkflowDefinition {
    WorkflowDefinition::new(
        "external_worker_registration",
        "External Worker Registration",
    )
    // Step 1: Discover models from /v1/models endpoint
    .add_step(
        StepDefinition::new(
            "discover_models",
            "Discover Models",
            Arc::new(DiscoverModelsStep),
        )
        .with_retry(RetryPolicy {
            max_attempts: 3,
            backoff: BackoffStrategy::Exponential {
                base: Duration::from_secs(1),
                max: Duration::from_secs(10),
            },
        })
        .with_timeout(Duration::from_secs(30))
        .with_failure_action(FailureAction::FailWorkflow),
    )
    // Step 2: Create workers for each model
    .add_step(
        StepDefinition::new(
            "create_workers",
            "Create Workers",
            Arc::new(CreateExternalWorkersStep),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::FailWorkflow)
        .depends_on(&["discover_models"]),
    )
    // Step 3: Register workers (shared step)
    .add_step(
        StepDefinition::new(
            "register_workers",
            "Register Workers",
            Arc::new(RegisterWorkersStep),
        )
        .with_timeout(Duration::from_secs(5))
        .with_failure_action(FailureAction::FailWorkflow)
        .depends_on(&["create_workers"]),
    )
    // Step 4a: Update policies (parallel with activation)
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
    // Step 4b: Activate workers (parallel with policy update)
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
