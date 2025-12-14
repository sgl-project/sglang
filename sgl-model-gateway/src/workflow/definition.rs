//! Workflow definition types

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

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
    pub depends_on: Vec<StepId>,
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
            depends_on: Vec::new(),
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

    /// Set dependencies for this step.
    /// The step will only run after all specified dependencies have completed successfully.
    /// Empty slice means no dependencies - step can run immediately in parallel with others.
    pub fn depends_on(mut self, deps: &[&str]) -> Self {
        self.depends_on = deps.iter().map(|s| StepId::new(*s)).collect();
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

    /// Validate the workflow DAG structure.
    /// Returns an error if:
    /// - A step depends on a non-existent step
    /// - There's a cycle in the dependencies
    pub fn validate(&self) -> Result<(), String> {
        // Build HashMap for O(1) lookup instead of O(n) linear search
        let steps_map: HashMap<&StepId, &StepDefinition> =
            self.steps.iter().map(|s| (&s.id, s)).collect();

        // Check all dependencies exist
        for step in &self.steps {
            for dep in &step.depends_on {
                if !steps_map.contains_key(dep) {
                    return Err(format!(
                        "Step '{}' depends on non-existent step '{}'",
                        step.id, dep
                    ));
                }
            }
        }

        // Check for cycles using DFS
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for step in &self.steps {
            if !visited.contains(&step.id)
                && Self::has_cycle(&step.id, &steps_map, &mut visited, &mut rec_stack)
            {
                return Err(format!("Cycle detected involving step '{}'", step.id));
            }
        }

        Ok(())
    }

    /// DFS helper for cycle detection with O(1) HashMap lookup
    fn has_cycle<'a>(
        step_id: &'a StepId,
        steps_map: &HashMap<&'a StepId, &'a StepDefinition>,
        visited: &mut HashSet<&'a StepId>,
        rec_stack: &mut HashSet<&'a StepId>,
    ) -> bool {
        if rec_stack.contains(step_id) {
            return true; // Back edge found - cycle!
        }
        if visited.contains(step_id) {
            return false; // Already fully processed
        }

        visited.insert(step_id);
        rec_stack.insert(step_id);

        // O(1) lookup instead of linear search
        if let Some(step) = steps_map.get(step_id) {
            for dep in &step.depends_on {
                if Self::has_cycle(dep, steps_map, visited, rec_stack) {
                    return true;
                }
            }
        }

        rec_stack.remove(step_id);
        false
    }

    /// Get steps that have no dependencies (can run immediately)
    pub fn get_initial_steps(&self) -> Vec<&StepDefinition> {
        self.steps
            .iter()
            .filter(|s| s.depends_on.is_empty())
            .collect()
    }

    /// Get steps that depend on the given step
    pub fn get_dependents(&self, step_id: &StepId) -> Vec<&StepDefinition> {
        self.steps
            .iter()
            .filter(|s| s.depends_on.contains(step_id))
            .collect()
    }
}
