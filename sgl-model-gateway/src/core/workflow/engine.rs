//! Workflow execution engine

use std::{collections::HashMap, sync::Arc, time::Duration};

use backoff::{backoff::Backoff, ExponentialBackoffBuilder};
use chrono::Utc;
use parking_lot::RwLock;
use tokio::time::timeout;

use super::{
    definition::{StepDefinition, WorkflowDefinition},
    event::{EventBus, WorkflowEvent},
    state::WorkflowStateStore,
    types::*,
};

/// Linear backoff implementation that increases delay by a fixed amount each retry
struct LinearBackoff {
    current: Duration,
    increment: Duration,
    max: Duration,
}

impl LinearBackoff {
    fn new(increment: Duration, max: Duration) -> Self {
        Self {
            current: increment,
            increment,
            max,
        }
    }
}

impl Backoff for LinearBackoff {
    fn next_backoff(&mut self) -> Option<Duration> {
        let next = self.current;
        self.current = (self.current + self.increment).min(self.max);
        Some(next)
    }

    fn reset(&mut self) {
        self.current = self.increment;
    }
}

/// Main workflow execution engine
pub struct WorkflowEngine {
    definitions: Arc<RwLock<HashMap<WorkflowId, Arc<WorkflowDefinition>>>>,
    state_store: WorkflowStateStore,
    event_bus: Arc<EventBus>,
}

impl WorkflowEngine {
    pub fn new() -> Self {
        Self {
            definitions: Arc::new(RwLock::new(HashMap::new())),
            state_store: WorkflowStateStore::new(),
            event_bus: Arc::new(EventBus::new()),
        }
    }

    /// Start a background task to periodically clean up old workflow states
    ///
    /// This prevents unbounded memory growth by removing completed/failed workflows
    /// that are older than the specified TTL.
    ///
    /// # Arguments
    ///
    /// * `ttl` - Time-to-live for terminal workflows (default: 1 hour)
    /// * `interval` - How often to run cleanup (default: 5 minutes)
    ///
    /// # Returns
    ///
    /// A join handle for the cleanup task that can be used to stop it.
    pub fn start_cleanup_task(
        &self,
        ttl: Option<Duration>,
        interval: Option<Duration>,
    ) -> tokio::task::JoinHandle<()> {
        let state_store = self.state_store.clone();
        let ttl = ttl.unwrap_or(Duration::from_secs(3600)); // 1 hour default
        let interval = interval.unwrap_or(Duration::from_secs(300)); // 5 minutes default

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                ticker.tick().await;
                state_store.cleanup_old_workflows(ttl);
            }
        })
    }

    /// Register a workflow definition
    pub fn register_workflow(&self, definition: WorkflowDefinition) {
        let id = definition.id.clone();
        self.definitions.write().insert(id, Arc::new(definition));
    }

    /// Get the event bus for subscribing to workflow events
    pub fn event_bus(&self) -> Arc<EventBus> {
        Arc::clone(&self.event_bus)
    }

    /// Get the state store
    pub fn state_store(&self) -> &WorkflowStateStore {
        &self.state_store
    }

    /// Start a new workflow instance
    pub async fn start_workflow(
        &self,
        definition_id: WorkflowId,
        context: WorkflowContext,
    ) -> WorkflowResult<WorkflowInstanceId> {
        // Get workflow definition
        let definition = {
            let definitions = self.definitions.read();
            definitions
                .get(&definition_id)
                .cloned()
                .ok_or_else(|| WorkflowError::DefinitionNotFound(definition_id.clone()))?
        };

        // Create new workflow instance
        let instance_id = context.instance_id;
        let mut state = WorkflowState::new(instance_id, definition_id.clone());
        state.status = WorkflowStatus::Running;
        state.context = context;

        // Initialize step states
        for step in &definition.steps {
            state
                .step_states
                .insert(step.id.clone(), StepState::default());
        }

        // Save initial state
        self.state_store.save(state)?;

        // Emit workflow started event
        self.event_bus
            .publish(WorkflowEvent::WorkflowStarted {
                instance_id,
                definition_id,
            })
            .await;

        // Execute workflow in background
        let engine = self.clone_for_execution();
        let def = Arc::clone(&definition);
        tokio::spawn(async move {
            if let Err(e) = engine.execute_workflow(instance_id, def).await {
                tracing::error!(instance_id = %instance_id, error = ?e, "Workflow execution failed");
            }
        });

        Ok(instance_id)
    }

    /// Execute a workflow (internal)
    async fn execute_workflow(
        &self,
        instance_id: WorkflowInstanceId,
        definition: Arc<WorkflowDefinition>,
    ) -> WorkflowResult<()> {
        let start_time = std::time::Instant::now();

        for step in &definition.steps {
            // Check if workflow was cancelled
            let state = self.state_store.load(instance_id)?;
            if state.status == WorkflowStatus::Cancelled {
                self.event_bus
                    .publish(WorkflowEvent::WorkflowCancelled { instance_id })
                    .await;
                return Ok(());
            }

            // Execute step with retry
            match self
                .execute_step_with_retry(instance_id, step, &definition)
                .await
            {
                Ok(StepResult::Success) => {
                    // Continue to next step
                }
                Ok(StepResult::Skip) => {
                    // Step was skipped, continue to next
                    continue;
                }
                Ok(StepResult::Failure) | Err(_) => {
                    // Handle failure based on failure action
                    match step.on_failure {
                        FailureAction::FailWorkflow => {
                            let error_msg = format!("Step {} failed", step.id);
                            self.state_store.update(instance_id, |s| {
                                s.status = WorkflowStatus::Failed;
                            })?;

                            self.event_bus
                                .publish(WorkflowEvent::WorkflowFailed {
                                    instance_id,
                                    failed_step: step.id.clone(),
                                    error: error_msg,
                                })
                                .await;

                            return Ok(());
                        }
                        FailureAction::ContinueNextStep => {
                            // Mark step as skipped and continue
                            self.state_store.update(instance_id, |s| {
                                if let Some(step_state) = s.step_states.get_mut(&step.id) {
                                    step_state.status = StepStatus::Skipped;
                                }
                            })?;
                            continue;
                        }
                        FailureAction::RetryIndefinitely => {
                            // This should not happen as execute_step_with_retry handles it
                            unreachable!("RetryIndefinitely should be handled in retry logic");
                        }
                    }
                }
            }
        }

        // Workflow completed successfully
        self.state_store.update(instance_id, |s| {
            s.status = WorkflowStatus::Completed;
        })?;

        let duration = start_time.elapsed();
        self.event_bus
            .publish(WorkflowEvent::WorkflowCompleted {
                instance_id,
                duration,
            })
            .await;

        Ok(())
    }

    /// Execute a step with retry logic
    async fn execute_step_with_retry(
        &self,
        instance_id: WorkflowInstanceId,
        step: &StepDefinition,
        definition: &WorkflowDefinition,
    ) -> WorkflowResult<StepResult> {
        let retry_policy = definition.get_retry_policy(step);
        let step_timeout = definition.get_timeout(step);

        let mut attempt = 1;
        let max_attempts = if matches!(step.on_failure, FailureAction::RetryIndefinitely) {
            u32::MAX
        } else {
            retry_policy.max_attempts
        };

        let mut backoff = Self::create_backoff(&retry_policy.backoff);

        loop {
            // Check for cancellation before starting/retrying step
            {
                let state = self.state_store.load(instance_id)?;
                if state.status == WorkflowStatus::Cancelled {
                    return Err(WorkflowError::Cancelled(instance_id));
                }
            }

            // Update step state
            self.state_store.update(instance_id, |s| {
                s.current_step = Some(step.id.clone());
                if let Some(step_state) = s.step_states.get_mut(&step.id) {
                    step_state.status = if attempt == 1 {
                        StepStatus::Running
                    } else {
                        StepStatus::Retrying
                    };
                    step_state.attempt = attempt;
                    step_state.started_at = Some(Utc::now());
                }
            })?;

            // Emit step started event
            self.event_bus
                .publish(WorkflowEvent::StepStarted {
                    instance_id,
                    step_id: step.id.clone(),
                    attempt,
                })
                .await;

            // Get current context
            let mut context = self.state_store.load(instance_id)?.context;

            // Execute step with timeout
            let step_start = std::time::Instant::now();
            let result = timeout(step_timeout, step.executor.execute(&mut context)).await;

            let step_duration = step_start.elapsed();

            // Save updated context
            self.state_store.update(instance_id, |s| {
                s.context = context.clone();
            })?;

            match result {
                Ok(Ok(StepResult::Success)) => {
                    // Step succeeded
                    self.state_store.update(instance_id, |s| {
                        if let Some(step_state) = s.step_states.get_mut(&step.id) {
                            step_state.status = StepStatus::Succeeded;
                            step_state.completed_at = Some(Utc::now());
                        }
                    })?;

                    self.event_bus
                        .publish(WorkflowEvent::StepSucceeded {
                            instance_id,
                            step_id: step.id.clone(),
                            duration: step_duration,
                        })
                        .await;

                    // Call on_success hook
                    if let Err(e) = step.executor.on_success(&context).await {
                        tracing::warn!(step_id = %step.id, error = ?e, "on_success hook failed");
                    }

                    return Ok(StepResult::Success);
                }
                Ok(Ok(StepResult::Skip)) => {
                    return Ok(StepResult::Skip);
                }
                Ok(Ok(StepResult::Failure)) | Ok(Err(_)) | Err(_) => {
                    let (error_msg, should_retry) = match result {
                        Ok(Err(e)) => {
                            let msg = format!("{}", e);
                            let retryable = step.executor.is_retryable(&e);
                            (msg, retryable)
                        }
                        Err(_) => (
                            format!("Step timeout after {:?}", step_timeout),
                            true, // Timeouts are retryable
                        ),
                        _ => ("Step failed".to_string(), false),
                    };

                    let will_retry = should_retry && attempt < max_attempts;

                    // Update step state
                    self.state_store.update(instance_id, |s| {
                        if let Some(step_state) = s.step_states.get_mut(&step.id) {
                            step_state.status = if will_retry {
                                StepStatus::Retrying
                            } else {
                                StepStatus::Failed
                            };
                            step_state.last_error = Some(error_msg.clone());
                            if !will_retry {
                                step_state.completed_at = Some(Utc::now());
                            }
                        }
                    })?;

                    // Emit step failed event
                    self.event_bus
                        .publish(WorkflowEvent::StepFailed {
                            instance_id,
                            step_id: step.id.clone(),
                            error: error_msg.clone(),
                            will_retry,
                        })
                        .await;

                    if will_retry {
                        // Calculate backoff delay
                        let delay = backoff
                            .next_backoff()
                            .unwrap_or_else(|| Duration::from_secs(1));

                        self.event_bus
                            .publish(WorkflowEvent::StepRetrying {
                                instance_id,
                                step_id: step.id.clone(),
                                attempt: attempt + 1,
                                delay,
                            })
                            .await;

                        tokio::time::sleep(delay).await;
                        attempt += 1;
                    } else {
                        // No more retries, call on_failure hook
                        // Create a generic error for the hook
                        let hook_error = WorkflowError::StepFailed {
                            step_id: step.id.clone(),
                            message: error_msg,
                        };
                        if let Err(hook_err) = step.executor.on_failure(&context, &hook_error).await
                        {
                            tracing::warn!(step_id = %step.id, error = ?hook_err, "on_failure hook failed");
                        }

                        return Ok(StepResult::Failure);
                    }
                }
            }
        }
    }

    /// Create a backoff instance from strategy
    fn create_backoff(strategy: &BackoffStrategy) -> Box<dyn Backoff + Send> {
        match strategy {
            BackoffStrategy::Fixed(duration) => {
                // For fixed backoff, use exponential with multiplier 1.0
                let backoff = ExponentialBackoffBuilder::new()
                    .with_initial_interval(*duration)
                    .with_multiplier(1.0)
                    .with_max_interval(*duration)
                    .with_max_elapsed_time(None)
                    .build();
                Box::new(backoff)
            }
            BackoffStrategy::Exponential { base, max } => {
                let backoff = ExponentialBackoffBuilder::new()
                    .with_initial_interval(*base)
                    .with_max_interval(*max)
                    .with_max_elapsed_time(None)
                    .build();
                Box::new(backoff)
            }
            BackoffStrategy::Linear { increment, max } => {
                // Use proper linear backoff: increment, 2*increment, 3*increment, ...
                Box::new(LinearBackoff::new(*increment, *max))
            }
        }
    }

    /// Cancel a running workflow
    pub async fn cancel_workflow(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<()> {
        self.state_store.update(instance_id, |s| {
            s.status = WorkflowStatus::Cancelled;
        })?;

        self.event_bus
            .publish(WorkflowEvent::WorkflowCancelled { instance_id })
            .await;

        Ok(())
    }

    /// Get workflow status
    pub fn get_status(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<WorkflowState> {
        self.state_store.load(instance_id)
    }

    /// Clone engine for async execution
    fn clone_for_execution(&self) -> Self {
        Self {
            definitions: Arc::clone(&self.definitions),
            state_store: self.state_store.clone(),
            event_bus: Arc::clone(&self.event_bus),
        }
    }
}

impl Default for WorkflowEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for WorkflowEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkflowEngine")
            .field("definitions_count", &self.definitions.read().len())
            .field("state_count", &self.state_store.count())
            .finish()
    }
}
