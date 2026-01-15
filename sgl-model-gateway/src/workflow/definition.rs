//! Workflow definition types

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::Arc,
    time::Duration,
};

use chrono::{DateTime, Utc};

use super::{
    executor::StepExecutor,
    types::{FailureAction, RetryPolicy, StepId, WorkflowContext, WorkflowData, WorkflowId},
};

/// A condition function that determines whether a step should run.
pub type StepCondition<D> = Arc<dyn Fn(&WorkflowContext<D>) -> bool + Send + Sync>;

/// Errors that can occur during workflow validation
#[derive(Debug, Clone, thiserror::Error)]
pub enum ValidationError {
    /// A step depends on another step that doesn't exist
    #[error("Step '{step}' depends on non-existent step '{dependency}'")]
    MissingDependency { step: StepId, dependency: StepId },

    /// A cycle was detected in the workflow DAG
    #[error("Cycle detected involving step '{0}'")]
    CycleDetected(StepId),
}

/// Definition of a single step within a workflow
pub struct StepDefinition<D: WorkflowData> {
    pub id: StepId,
    pub name: String,
    pub executor: Arc<dyn StepExecutor<D>>,
    pub retry_policy: Option<RetryPolicy>,
    pub timeout: Option<Duration>,
    pub on_failure: FailureAction,
    /// Dependencies that must ALL complete before this step runs
    pub depends_on: Vec<StepId>,
    /// Dependencies where ANY completing triggers this step (used with depends_on)
    pub depends_on_any: Vec<StepId>,
    /// Delay before starting the step (after dependencies satisfied)
    pub delay: Option<Duration>,
    /// Run step at or after this time (after dependencies satisfied)
    pub scheduled_at: Option<DateTime<Utc>>,
    /// Condition to evaluate; if false, step is skipped
    pub run_if: Option<StepCondition<D>>,
}

impl<D: WorkflowData> fmt::Debug for StepDefinition<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StepDefinition")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("retry_policy", &self.retry_policy)
            .field("timeout", &self.timeout)
            .field("on_failure", &self.on_failure)
            .field("depends_on", &self.depends_on)
            .field("depends_on_any", &self.depends_on_any)
            .field("delay", &self.delay)
            .field("scheduled_at", &self.scheduled_at)
            .field("run_if", &self.run_if.as_ref().map(|_| "<condition>"))
            .finish_non_exhaustive()
    }
}

impl<D: WorkflowData> StepDefinition<D> {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        executor: Arc<dyn StepExecutor<D>>,
    ) -> Self {
        Self {
            id: StepId::new(id.into()),
            name: name.into(),
            executor,
            retry_policy: None,
            timeout: None,
            on_failure: FailureAction::FailWorkflow,
            depends_on: Vec::new(),
            depends_on_any: Vec::new(),
            delay: None,
            scheduled_at: None,
            run_if: None,
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
    /// The step will only run after ALL specified dependencies have completed successfully.
    /// Empty slice means no dependencies - step can run immediately in parallel with others.
    pub fn depends_on(mut self, deps: &[&str]) -> Self {
        self.depends_on = deps.iter().map(|s| StepId::new(*s)).collect();
        self
    }

    /// Set "any of" dependencies for this step.
    /// The step will run when ANY of these dependencies complete (in addition to
    /// all `depends_on` dependencies).
    ///
    /// **Combined semantics** (when both `depends_on` and `depends_on_any` are set):
    /// - ALL `depends_on` must complete successfully, AND
    /// - AT LEAST ONE `depends_on_any` must complete successfully
    ///
    /// **Failure handling**:
    /// - For `depends_on`: if ANY fails, this step is blocked
    /// - For `depends_on_any`: only blocked if ALL fail (since we only need one)
    ///
    /// **Skipped dependencies**: A skipped dependency (e.g., via `run_if`) counts as
    /// "completed" for dependency satisfaction purposes.
    pub fn depends_on_any(mut self, deps: &[&str]) -> Self {
        self.depends_on_any = deps.iter().map(|s| StepId::new(*s)).collect();
        self
    }

    /// Set a delay before starting the step (after dependencies are satisfied).
    ///
    /// If both `delay` and `scheduled_at` are set, the step will wait for the
    /// scheduled time AND THEN wait for the delay duration (they stack).
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    /// Schedule the step to run at or after the specified time.
    /// The step will wait until this time even if dependencies are satisfied earlier.
    ///
    /// If the scheduled time is in the past when the step becomes ready, it will
    /// proceed immediately (a debug log is emitted).
    ///
    /// If both `delay` and `scheduled_at` are set, the step will wait for the
    /// scheduled time AND THEN wait for the delay duration (they stack).
    pub fn scheduled_at(mut self, time: DateTime<Utc>) -> Self {
        self.scheduled_at = Some(time);
        self
    }

    /// Set a condition for running this step.
    /// If the condition returns false, the step is skipped.
    ///
    /// **Skipped step semantics**: When a step is skipped:
    /// - It is marked as `StepStatus::Skipped`
    /// - Downstream steps that depend on it (via `depends_on`) will consider it satisfied
    /// - This allows conditional branches without blocking the workflow
    ///
    /// **Error handling**: If the context cannot be retrieved to evaluate the condition,
    /// the step will **fail** (not proceed blindly). This is a safety measure.
    ///
    /// **Note**: The condition closure is not serializable, so workflows with `run_if`
    /// cannot be persisted and resumed from external storage.
    pub fn run_if<F>(mut self, condition: F) -> Self
    where
        F: Fn(&WorkflowContext<D>) -> bool + Send + Sync + 'static,
    {
        self.run_if = Some(Arc::new(condition));
        self
    }
}

/// Complete workflow definition
pub struct WorkflowDefinition<D: WorkflowData> {
    pub id: WorkflowId,
    pub name: String,
    pub steps: Vec<StepDefinition<D>>,
    pub default_retry_policy: RetryPolicy,
    pub default_timeout: Duration,
    /// Pre-computed reverse dependencies: step_id -> indices of steps that depend on it
    reverse_deps: HashMap<StepId, Vec<usize>>,
    /// Pre-computed indices of steps with no dependencies (can start immediately)
    initial_step_indices: Vec<usize>,
}

impl<D: WorkflowData> fmt::Debug for WorkflowDefinition<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WorkflowDefinition")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("steps", &self.steps)
            .field("default_retry_policy", &self.default_retry_policy)
            .field("default_timeout", &self.default_timeout)
            .finish_non_exhaustive()
    }
}

impl<D: WorkflowData> WorkflowDefinition<D> {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: WorkflowId::new(id.into()),
            name: name.into(),
            steps: Vec::new(),
            default_retry_policy: RetryPolicy::default(),
            default_timeout: Duration::from_secs(300), // 5 minutes
            reverse_deps: HashMap::new(),
            initial_step_indices: Vec::new(),
        }
    }

    pub fn add_step(mut self, step: StepDefinition<D>) -> Self {
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
    pub fn get_retry_policy<'a>(&'a self, step: &'a StepDefinition<D>) -> &'a RetryPolicy {
        step.retry_policy
            .as_ref()
            .unwrap_or(&self.default_retry_policy)
    }

    /// Get the timeout for a step (step-specific or default)
    pub fn get_timeout(&self, step: &StepDefinition<D>) -> Duration {
        step.timeout.unwrap_or(self.default_timeout)
    }

    /// Validate the workflow DAG structure and build dependency graph.
    /// Returns an error if:
    /// - A step depends on a non-existent step
    /// - There's a cycle in the dependencies
    ///
    /// On success, pre-computes reverse dependencies for O(1) dependent lookup.
    #[must_use = "validation result should be checked"]
    pub fn validate(&mut self) -> Result<(), ValidationError> {
        // Build HashMap for O(1) lookup instead of O(n) linear search
        let steps_map: HashMap<&StepId, &StepDefinition<D>> =
            self.steps.iter().map(|s| (&s.id, s)).collect();

        // Check all dependencies exist (both depends_on and depends_on_any)
        for step in &self.steps {
            for dep in &step.depends_on {
                if !steps_map.contains_key(dep) {
                    return Err(ValidationError::MissingDependency {
                        step: step.id.clone(),
                        dependency: dep.clone(),
                    });
                }
            }
            for dep in &step.depends_on_any {
                if !steps_map.contains_key(dep) {
                    return Err(ValidationError::MissingDependency {
                        step: step.id.clone(),
                        dependency: dep.clone(),
                    });
                }
            }
        }

        // Check for cycles using DFS (considers both dependency types)
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for step in &self.steps {
            if !visited.contains(&step.id)
                && Self::has_cycle(&step.id, &steps_map, &mut visited, &mut rec_stack)
            {
                return Err(ValidationError::CycleDetected(step.id.clone()));
            }
        }

        // Build reverse dependency map: for each step, which steps depend on it?
        // Include both depends_on and depends_on_any
        self.reverse_deps.clear();
        for (idx, step) in self.steps.iter().enumerate() {
            for dep_id in &step.depends_on {
                self.reverse_deps
                    .entry(dep_id.clone())
                    .or_default()
                    .push(idx);
            }
            for dep_id in &step.depends_on_any {
                self.reverse_deps
                    .entry(dep_id.clone())
                    .or_default()
                    .push(idx);
            }
        }

        // Cache indices of steps with no dependencies (can start immediately)
        // A step with only depends_on_any still needs at least one to complete
        self.initial_step_indices = self
            .steps
            .iter()
            .enumerate()
            .filter(|(_, s)| s.depends_on.is_empty() && s.depends_on_any.is_empty())
            .map(|(i, _)| i)
            .collect();

        Ok(())
    }

    /// DFS helper for cycle detection with O(1) HashMap lookup
    fn has_cycle<'a>(
        step_id: &'a StepId,
        steps_map: &HashMap<&'a StepId, &'a StepDefinition<D>>,
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
        // Check both depends_on and depends_on_any for cycles
        if let Some(step) = steps_map.get(step_id) {
            for dep in &step.depends_on {
                if Self::has_cycle(dep, steps_map, visited, rec_stack) {
                    return true;
                }
            }
            for dep in &step.depends_on_any {
                if Self::has_cycle(dep, steps_map, visited, rec_stack) {
                    return true;
                }
            }
        }

        rec_stack.remove(step_id);
        false
    }

    /// Get indices of steps that depend on the given step
    pub fn get_dependent_indices(&self, step_id: &StepId) -> &[usize] {
        self.reverse_deps
            .get(step_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get indices of steps with no dependencies
    pub fn get_initial_step_indices(&self) -> &[usize] {
        &self.initial_step_indices
    }
}
