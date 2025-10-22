//! Workflow definition types

use std::{sync::Arc, time::Duration};

use super::{
    executor::StepExecutor,
    types::{FailureAction, RetryPolicy, StepId, WorkflowId},
};

/// Definition of a single step within a workflow
pub struct StepDefinition {
    pub id: StepId,
    pub name: String,
    pub executor: Arc<dyn StepExecutor>,
    pub retry_policy: Option<RetryPolicy>,
    pub timeout: Option<Duration>,
    pub on_failure: FailureAction,
}

impl StepDefinition {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        executor: Arc<dyn StepExecutor>,
    ) -> Self {
        Self {
            id: StepId::new(id.into()),
            name: name.into(),
            executor,
            retry_policy: None,
            timeout: None,
            on_failure: FailureAction::FailWorkflow,
        }
    }

    pub fn with_retry(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = Some(policy);
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn with_failure_action(mut self, action: FailureAction) -> Self {
        self.on_failure = action;
        self
    }
}

/// Complete workflow definition
pub struct WorkflowDefinition {
    pub id: WorkflowId,
    pub name: String,
    pub steps: Vec<StepDefinition>,
    pub default_retry_policy: RetryPolicy,
    pub default_timeout: Duration,
}

impl WorkflowDefinition {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: WorkflowId::new(id.into()),
            name: name.into(),
            steps: Vec::new(),
            default_retry_policy: RetryPolicy::default(),
            default_timeout: Duration::from_secs(300), // 5 minutes
        }
    }

    pub fn add_step(mut self, step: StepDefinition) -> Self {
        self.steps.push(step);
        self
    }

    pub fn with_default_retry(mut self, policy: RetryPolicy) -> Self {
        self.default_retry_policy = policy;
        self
    }

    pub fn with_default_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Get the retry policy for a step (step-specific or default)
    pub fn get_retry_policy<'a>(&'a self, step: &'a StepDefinition) -> &'a RetryPolicy {
        step.retry_policy
            .as_ref()
            .unwrap_or(&self.default_retry_policy)
    }

    /// Get the timeout for a step (step-specific or default)
    pub fn get_timeout(&self, step: &StepDefinition) -> Duration {
        step.timeout.unwrap_or(self.default_timeout)
    }
}
