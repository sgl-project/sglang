//! Workflow execution engine
//!
//! Supports DAG-based parallel execution of workflow steps.
//! Steps with no dependencies run in parallel, steps with dependencies
//! wait for all dependencies to complete successfully.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    marker::PhantomData,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use backoff::{backoff::Backoff, ExponentialBackoffBuilder};
use chrono::Utc;
use parking_lot::RwLock;
use tokio::{
    sync::{mpsc, watch},
    time::timeout,
};

use super::{
    definition::{StepDefinition, WorkflowDefinition},
    event::{EventBus, WorkflowEvent},
    state::{InMemoryStore, StateStore},
    types::*,
};

#[derive(Default)]
struct StepTracker {
    completed: HashSet<StepId>,
    failed: HashSet<StepId>,
    skipped: HashSet<StepId>,
    running: HashSet<StepId>,
}

impl StepTracker {
    fn total_processed(&self) -> usize {
        self.completed.len() + self.failed.len() + self.skipped.len()
    }

    fn is_step_processable(&self, step_id: &StepId) -> bool {
        !self.completed.contains(step_id)
            && !self.failed.contains(step_id)
            && !self.skipped.contains(step_id)
            && !self.running.contains(step_id)
    }

    fn are_dependencies_satisfied(&self, depends_on: &[StepId]) -> bool {
        depends_on
            .iter()
            .all(|dep| self.completed.contains(dep) || self.skipped.contains(dep))
    }

    fn has_failed_dependency(&self, depends_on: &[StepId]) -> bool {
        depends_on.iter().any(|dep| self.failed.contains(dep))
    }
}

/// Fixed backoff that returns the same delay every time
struct FixedBackoff(Duration);

impl Backoff for FixedBackoff {
    fn reset(&mut self) {}

    fn next_backoff(&mut self) -> Option<Duration> {
        Some(self.0)
    }
}

/// Linear backoff that increases delay by a fixed amount each retry
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
    fn reset(&mut self) {
        self.current = self.increment;
    }

    fn next_backoff(&mut self) -> Option<Duration> {
        let next = self.current;
        self.current = (self.current + self.increment).min(self.max);
        Some(next)
    }
}

/// Main workflow execution engine
///
/// # Type Parameters
///
/// * `D` - The workflow data type that implements `WorkflowData`
/// * `S` - The state store implementation (defaults to `InMemoryStore<D>`)
///
/// # Graceful Shutdown
///
/// The engine supports graceful shutdown via [`shutdown()`](Self::shutdown):
///
/// ```ignore
/// // Trigger shutdown - stops accepting new workflows
/// engine.shutdown();
///
/// // Wait for all running workflows to complete (with timeout)
/// if !engine.wait_for_shutdown(Duration::from_secs(30)).await {
///     // Force cancel remaining workflows
///     engine.force_cancel_all().await;
/// }
/// ```
pub struct WorkflowEngine<D: WorkflowData, S: StateStore<D> = InMemoryStore<D>> {
    definitions: Arc<RwLock<HashMap<WorkflowId, Arc<WorkflowDefinition<D>>>>>,
    state_store: S,
    event_bus: Arc<EventBus>,
    /// Shutdown signal sender - when true, engine is shutting down
    shutdown_tx: Arc<watch::Sender<bool>>,
    /// Shutdown signal receiver for cloning to tasks
    shutdown_rx: watch::Receiver<bool>,
    /// Count of active workflow executions
    active_workflows: Arc<AtomicUsize>,
    _phantom: PhantomData<D>,
}

impl<D: WorkflowData> WorkflowEngine<D, InMemoryStore<D>> {
    pub fn new() -> Self {
        Self::with_store(InMemoryStore::new())
    }
}

impl<D: WorkflowData, S: StateStore<D> + 'static> WorkflowEngine<D, S> {
    /// Create a new workflow engine with a custom state store
    pub fn with_store(state_store: S) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            definitions: Arc::new(RwLock::new(HashMap::new())),
            state_store,
            event_bus: Arc::new(EventBus::new()),
            shutdown_tx: Arc::new(shutdown_tx),
            shutdown_rx,
            active_workflows: Arc::new(AtomicUsize::new(0)),
            _phantom: PhantomData,
        }
    }

    /// Check if the engine is shutting down
    pub fn is_shutting_down(&self) -> bool {
        *self.shutdown_rx.borrow()
    }

    /// Initiate graceful shutdown
    ///
    /// This will:
    /// - Stop accepting new workflows (start_workflow will return an error)
    /// - Stop the cleanup task
    /// - Allow running workflows to complete
    ///
    /// Use [`wait_for_shutdown`](Self::wait_for_shutdown) to wait for completion.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
        tracing::info!("Workflow engine shutdown initiated");
    }

    /// Wait for all active workflows to complete
    ///
    /// Returns `true` if all workflows completed within the timeout,
    /// `false` if the timeout was reached with workflows still running.
    ///
    /// Uses simple polling - appropriate for shutdown which happens once per process.
    pub async fn wait_for_shutdown(&self, timeout_duration: Duration) -> bool {
        let start = tokio::time::Instant::now();

        loop {
            let active = self.active_workflows.load(Ordering::Acquire);
            if active == 0 {
                tracing::info!("All workflows completed, shutdown complete");
                return true;
            }

            if start.elapsed() >= timeout_duration {
                tracing::warn!(
                    remaining_workflows = active,
                    "Shutdown timeout reached with workflows still running"
                );
                return false;
            }

            tracing::debug!(
                active_workflows = active,
                "Waiting for workflows to complete"
            );

            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Force cancel all running workflows
    ///
    /// This should be called after `wait_for_shutdown` times out if you need
    /// to ensure all workflows are stopped. Note that this cancels workflows
    /// at the state level; running steps may still complete.
    pub async fn force_cancel_all(&self) -> usize {
        let active_states = match self.state_store.list_active() {
            Ok(states) => states,
            Err(e) => {
                tracing::error!(error = ?e, "Failed to list active workflows for force cancel");
                return 0;
            }
        };

        let mut cancelled = 0;
        for state in active_states {
            if let Err(e) = self.cancel_workflow(state.instance_id).await {
                tracing::warn!(
                    instance_id = %state.instance_id,
                    error = ?e,
                    "Failed to cancel workflow during force shutdown"
                );
            } else {
                cancelled += 1;
            }
        }

        tracing::info!(cancelled_count = cancelled, "Force cancelled workflows");
        cancelled
    }

    /// Get the number of currently active workflow executions
    pub fn active_workflow_count(&self) -> usize {
        self.active_workflows.load(Ordering::Acquire)
    }

    /// Decrement active workflow count
    fn workflow_finished(&self) {
        self.active_workflows.fetch_sub(1, Ordering::Release);
    }

    /// Create a guard that decrements active_workflows on drop.
    /// This ensures the count is decremented even if a task panics.
    fn active_workflow_guard(&self) -> ActiveWorkflowGuard {
        ActiveWorkflowGuard {
            active_workflows: Arc::clone(&self.active_workflows),
        }
    }

    /// Start a background task to periodically clean up old workflow states
    ///
    /// This prevents unbounded memory growth by removing completed/failed workflows
    /// that are older than the specified TTL.
    ///
    /// The task will automatically stop when [`shutdown()`](Self::shutdown) is called.
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
        let mut shutdown_rx = self.shutdown_rx.clone();

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        state_store.cleanup_old_workflows(ttl);
                    }
                    _ = shutdown_rx.changed() => {
                        tracing::info!("Cleanup task stopping due to shutdown");
                        break;
                    }
                }
            }
        })
    }

    /// Register a workflow definition
    pub fn register_workflow(&self, mut definition: WorkflowDefinition<D>) -> Result<(), String> {
        // Validate DAG and build dependency graph once at registration
        definition.validate()?;

        let id = definition.id.clone();
        self.definitions.write().insert(id, Arc::new(definition));
        Ok(())
    }

    /// Get the event bus for subscribing to workflow events
    pub fn event_bus(&self) -> Arc<EventBus> {
        Arc::clone(&self.event_bus)
    }

    /// Get the state store
    pub fn state_store(&self) -> &S {
        &self.state_store
    }

    /// Start a new workflow instance
    ///
    /// Returns `Err(WorkflowError::ShuttingDown)` if the engine is shutting down.
    pub async fn start_workflow(
        &self,
        definition_id: WorkflowId,
        data: D,
    ) -> WorkflowResult<WorkflowInstanceId> {
        // Guard increments counter and decrements on drop unless committed.
        // This handles all error paths automatically.
        let guard = StartGuard::new(self);

        if self.is_shutting_down() {
            return Err(WorkflowError::ShuttingDown);
        }

        let definition = self
            .definitions
            .read()
            .get(&definition_id)
            .cloned()
            .ok_or_else(|| WorkflowError::DefinitionNotFound(definition_id.clone()))?;

        let instance_id = WorkflowInstanceId::new();
        let mut state = WorkflowState::new(instance_id, definition_id.clone(), data);
        state.status = WorkflowStatus::Running;

        for step in &definition.steps {
            state
                .step_states
                .insert(step.id.clone(), StepState::default());
        }

        self.state_store.save(state)?;

        self.event_bus
            .publish(WorkflowEvent::WorkflowStarted {
                instance_id,
                definition_id,
            })
            .await;

        // Commit the guard - from here the spawned task takes ownership of the count
        guard.commit();

        let engine = self.clone_for_execution();
        let def = Arc::clone(&definition);
        tokio::spawn(async move {
            let _guard = engine.active_workflow_guard();
            let result = engine.execute_workflow(instance_id, def).await;
            if let Err(e) = result {
                tracing::error!(instance_id = %instance_id, error = ?e, "Workflow execution failed");
            }
        });

        Ok(instance_id)
    }

    /// Execute a workflow with DAG-based parallel execution
    ///
    /// Uses event-driven readiness: instead of scanning all steps each iteration,
    /// we only check steps whose dependencies just completed.
    async fn execute_workflow(
        &self,
        instance_id: WorkflowInstanceId,
        definition: Arc<WorkflowDefinition<D>>,
    ) -> WorkflowResult<()> {
        let start_time = std::time::Instant::now();
        let step_count = definition.steps.len();

        let tracker: Arc<RwLock<StepTracker>> = Arc::new(RwLock::new(StepTracker::default()));
        let (tx, mut rx) = mpsc::channel::<(StepId, StepResult)>(step_count.max(1));

        // Initialize with steps that have no dependencies (O(1) lookup)
        let mut pending_check: VecDeque<usize> = definition
            .get_initial_step_indices()
            .iter()
            .copied()
            .collect();

        loop {
            if self.state_store.is_cancelled(instance_id)? {
                self.event_bus
                    .publish(WorkflowEvent::WorkflowCancelled { instance_id })
                    .await;
                return Ok(());
            }

            // Find ready steps from pending_check (not all steps)
            let (ready_step_indices, total_processed, running_count) = {
                let t = tracker.read();

                // Only check steps in pending_check, not all steps
                let ready: Vec<usize> = pending_check
                    .drain(..)
                    .filter(|&idx| {
                        let step = &definition.steps[idx];
                        t.is_step_processable(&step.id)
                            && t.are_dependencies_satisfied(&step.depends_on)
                            && !t.has_failed_dependency(&step.depends_on)
                    })
                    .collect();

                (ready, t.total_processed(), t.running.len())
            };

            // Check if we're done
            if total_processed == step_count {
                break;
            }

            // Handle blocked workflow (no ready steps, none running, but work remains)
            if ready_step_indices.is_empty() && running_count == 0 && pending_check.is_empty() {
                let failed_step = tracker.read().failed.iter().next().cloned();
                let error_message = if failed_step.is_some() {
                    "Workflow failed due to step dependency failure".to_string()
                } else {
                    "Workflow deadlocked: no steps ready and none running. This may indicate a scheduler bug.".to_string()
                };

                self.state_store.update(instance_id, |s| {
                    s.status = WorkflowStatus::Failed;
                })?;
                self.event_bus
                    .publish(WorkflowEvent::WorkflowFailed {
                        instance_id,
                        failed_step: failed_step
                            .unwrap_or_else(|| StepId::new("internal_scheduler")),
                        error: error_message,
                    })
                    .await;
                return Ok(());
            }

            // Launch ready steps in parallel
            let mut tasks_launched = 0;
            for step_idx in ready_step_indices {
                let step = &definition.steps[step_idx];
                tracker.write().running.insert(step.id.clone());
                tasks_launched += 1;

                let engine = self.clone_for_execution();
                let def = Arc::clone(&definition);
                let step_id = step.id.clone();
                let tx = tx.clone();
                let tracker = Arc::clone(&tracker);

                tokio::spawn(async move {
                    let step = &def.steps[step_idx];
                    let result = engine
                        .execute_step_with_retry(instance_id, step, &def)
                        .await;

                    let signal = match &result {
                        Ok(r) => *r,
                        Err(_) => StepResult::Failure,
                    };

                    {
                        let mut t = tracker.write();
                        t.running.remove(&step_id);

                        match result {
                            Ok(StepResult::Success) => {
                                t.completed.insert(step_id.clone());
                            }
                            Ok(StepResult::Skip) => {
                                t.skipped.insert(step_id.clone());
                            }
                            Ok(StepResult::Failure) | Err(_) => match step.on_failure {
                                FailureAction::FailWorkflow | FailureAction::RetryIndefinitely => {
                                    t.failed.insert(step_id.clone());
                                }
                                FailureAction::ContinueNextStep => {
                                    if let Err(e) = engine.state_store.update(instance_id, |s| {
                                        if let Some(step_state) = s.step_states.get_mut(&step_id) {
                                            step_state.status = StepStatus::Skipped;
                                        }
                                    }) {
                                        tracing::warn!(
                                            step_id = %step_id,
                                            error = ?e,
                                            "Failed to update step state to Skipped"
                                        );
                                    }
                                    t.skipped.insert(step_id.clone());
                                }
                            },
                        }

                        if let Err(e) = tx.try_send((step_id.clone(), signal)) {
                            use mpsc::error::TrySendError;
                            match e {
                                TrySendError::Full(_) => {
                                    tracing::error!(
                                        step_id = %step_id,
                                        "Channel full when sending step completion - this is a bug"
                                    );
                                }
                                TrySendError::Closed(_) => {
                                    tracing::debug!(
                                        step_id = %step_id,
                                        "Channel closed, workflow likely cancelled"
                                    );
                                }
                            }
                        }
                    }
                });
            }

            let should_wait = tasks_launched > 0 || !tracker.read().running.is_empty();
            if should_wait {
                if let Some((completed_step_id, result)) = rx.recv().await {
                    tracing::debug!(
                        step_id = %completed_step_id,
                        result = ?result,
                        "Step completed"
                    );

                    // Add dependents of completed step to pending_check (O(1) lookup)
                    // Only if the step succeeded or was skipped (not failed)
                    if matches!(result, StepResult::Success | StepResult::Skip) {
                        for &dep_idx in definition.get_dependent_indices(&completed_step_id) {
                            pending_check.push_back(dep_idx);
                        }
                    }
                }
            }
        }

        let failed_step = {
            let t = tracker.read();
            t.failed.iter().next().cloned()
        };

        if let Some(ref step) = failed_step {
            self.state_store.update(instance_id, |s| {
                s.status = WorkflowStatus::Failed;
            })?;
            self.event_bus
                .publish(WorkflowEvent::WorkflowFailed {
                    instance_id,
                    failed_step: step.clone(),
                    error: "One or more steps failed".to_string(),
                })
                .await;
        } else {
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
        }

        Ok(())
    }

    /// Execute a step with retry logic
    async fn execute_step_with_retry(
        &self,
        instance_id: WorkflowInstanceId,
        step: &StepDefinition<D>,
        definition: &WorkflowDefinition<D>,
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
            if self.state_store.is_cancelled(instance_id)? {
                return Err(WorkflowError::Cancelled(instance_id));
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

            let mut context = self.state_store.get_context(instance_id)?;

            // Execute step with timeout
            let step_start = std::time::Instant::now();
            let result = timeout(step_timeout, step.executor.execute(&mut context)).await;

            let step_duration = step_start.elapsed();

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

    fn create_backoff(strategy: &BackoffStrategy) -> Box<dyn Backoff + Send> {
        match strategy {
            BackoffStrategy::Fixed(duration) => Box::new(FixedBackoff(*duration)),
            BackoffStrategy::Exponential { base, max } => {
                let backoff = ExponentialBackoffBuilder::new()
                    .with_initial_interval(*base)
                    .with_max_interval(*max)
                    .with_max_elapsed_time(None)
                    .build();
                Box::new(backoff)
            }
            BackoffStrategy::Linear { increment, max } => {
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
    pub fn get_status(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<WorkflowState<D>> {
        self.state_store.load(instance_id)
    }

    /// Wait for a workflow to complete with adaptive polling
    ///
    /// Returns Ok with success message on completion, Err on failure/timeout/cancellation.
    /// Automatically cleans up terminal workflow states.
    pub async fn wait_for_completion(
        &self,
        instance_id: WorkflowInstanceId,
        label: &str,
        timeout_duration: Duration,
    ) -> Result<String, String> {
        let start = std::time::Instant::now();
        let mut poll_interval = Duration::from_millis(100);
        let max_poll_interval = Duration::from_millis(2000);
        let poll_backoff = Duration::from_millis(200);

        loop {
            if start.elapsed() > timeout_duration {
                return Err(format!(
                    "Workflow timeout after {}s for {}",
                    timeout_duration.as_secs(),
                    label
                ));
            }

            let state = self
                .get_status(instance_id)
                .map_err(|e| format!("Failed to get workflow status: {:?}", e))?;

            let result = match state.status {
                WorkflowStatus::Completed => {
                    Ok(format!("{} completed successfully via workflow", label))
                }
                WorkflowStatus::Failed => {
                    let current_step = state.current_step.as_ref();
                    let step_name = current_step
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    let error_msg = current_step
                        .and_then(|step_id| state.step_states.get(step_id))
                        .and_then(|s| s.last_error.as_deref())
                        .unwrap_or("Unknown error");
                    Err(format!(
                        "Workflow failed at step {}: {}",
                        step_name, error_msg
                    ))
                }
                WorkflowStatus::Cancelled => Err(format!("Workflow cancelled for {}", label)),
                WorkflowStatus::Pending | WorkflowStatus::Paused | WorkflowStatus::Running => {
                    tokio::time::sleep(poll_interval).await;
                    poll_interval = (poll_interval + poll_backoff).min(max_poll_interval);
                    continue;
                }
            };

            self.state_store.cleanup_if_terminal(instance_id);
            return result;
        }
    }

    /// Clone engine for async execution
    fn clone_for_execution(&self) -> Self {
        Self {
            definitions: Arc::clone(&self.definitions),
            state_store: self.state_store.clone(),
            event_bus: Arc::clone(&self.event_bus),
            shutdown_tx: Arc::clone(&self.shutdown_tx),
            shutdown_rx: self.shutdown_rx.clone(),
            active_workflows: Arc::clone(&self.active_workflows),
            _phantom: PhantomData,
        }
    }
}

/// RAII guard that decrements active_workflows count on drop.
/// Ensures proper cleanup even if a workflow task panics.
struct ActiveWorkflowGuard {
    active_workflows: Arc<AtomicUsize>,
}

impl Drop for ActiveWorkflowGuard {
    fn drop(&mut self) {
        self.active_workflows.fetch_sub(1, Ordering::Release);
    }
}

/// RAII guard for start_workflow that increments on creation and decrements on drop
/// unless commit() is called. Handles all error paths automatically.
struct StartGuard<'a, D: WorkflowData, S: StateStore<D> + 'static> {
    engine: &'a WorkflowEngine<D, S>,
    committed: bool,
}

impl<'a, D: WorkflowData, S: StateStore<D> + 'static> StartGuard<'a, D, S> {
    fn new(engine: &'a WorkflowEngine<D, S>) -> Self {
        engine.active_workflows.fetch_add(1, Ordering::AcqRel);
        Self {
            engine,
            committed: false,
        }
    }

    fn commit(mut self) {
        self.committed = true;
    }
}

impl<D: WorkflowData, S: StateStore<D> + 'static> Drop for StartGuard<'_, D, S> {
    fn drop(&mut self) {
        if !self.committed {
            self.engine.workflow_finished();
        }
    }
}

/// Clone implementation for internal use.
///
/// **Note**: This creates a shallow clone that shares state with the original engine.
/// Both engines will share the same:
/// - Workflow definitions
/// - State store
/// - Event bus
/// - Shutdown signal
/// - Active workflow counter
///
/// This is intentional for spawning async tasks that need access to the engine.
/// For most use cases, prefer sharing the engine via `Arc<WorkflowEngine>` rather
/// than cloning.
impl<D: WorkflowData, S: StateStore<D> + 'static> Clone for WorkflowEngine<D, S> {
    fn clone(&self) -> Self {
        self.clone_for_execution()
    }
}

impl<D: WorkflowData> Default for WorkflowEngine<D, InMemoryStore<D>> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: WorkflowData, S: StateStore<D> + 'static> std::fmt::Debug for WorkflowEngine<D, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkflowEngine")
            .field("definitions_count", &self.definitions.read().len())
            .finish()
    }
}
