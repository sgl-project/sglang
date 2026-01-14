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
    /// Steps waiting for delay/scheduled_at: maps step INDEX to ready time
    /// Using index instead of StepId for O(w) iteration in the main loop
    waiting_until: HashMap<usize, std::time::Instant>,
}

impl StepTracker {
    fn total_processed(&self) -> usize {
        self.completed.len() + self.failed.len() + self.skipped.len()
    }

    fn is_step_processable(&self, step_id: &StepId, step_idx: usize) -> bool {
        !self.completed.contains(step_id)
            && !self.failed.contains(step_id)
            && !self.skipped.contains(step_id)
            && !self.running.contains(step_id)
            && !self.waiting_until.contains_key(&step_idx)
    }

    /// Get indices of waiting steps that are now ready to run - O(w) where w = waiting count
    fn get_ready_waiting_indices(&self) -> Vec<usize> {
        let now = std::time::Instant::now();
        self.waiting_until
            .iter()
            .filter(|(_, &ready_at)| now >= ready_at)
            .map(|(&idx, _)| idx)
            .collect()
    }

    /// Mark a step as waiting until a specific time (by index)
    fn set_waiting(&mut self, step_idx: usize, ready_at: std::time::Instant) {
        self.waiting_until.insert(step_idx, ready_at);
    }

    /// Clear waiting status for a step (it's now ready to run)
    fn clear_waiting(&mut self, step_idx: usize) {
        self.waiting_until.remove(&step_idx);
    }

    /// Check if ALL dependencies are satisfied (completed or skipped)
    fn are_dependencies_satisfied(&self, depends_on: &[StepId]) -> bool {
        depends_on
            .iter()
            .all(|dep| self.completed.contains(dep) || self.skipped.contains(dep))
    }

    /// Check if ANY dependency is satisfied (completed or skipped)
    /// Returns true if the list is empty (no "any" dependencies)
    fn is_any_dependency_satisfied(&self, depends_on_any: &[StepId]) -> bool {
        depends_on_any.is_empty()
            || depends_on_any
                .iter()
                .any(|dep| self.completed.contains(dep) || self.skipped.contains(dep))
    }

    /// Check if ANY dependency has failed (for depends_on - blocks if any fail)
    fn has_failed_dependency(&self, depends_on: &[StepId]) -> bool {
        depends_on.iter().any(|dep| self.failed.contains(dep))
    }

    /// Check if ALL dependencies have failed (for depends_on_any - blocks only if all fail)
    fn have_all_any_deps_failed(&self, depends_on_any: &[StepId]) -> bool {
        !depends_on_any.is_empty() && depends_on_any.iter().all(|dep| self.failed.contains(dep))
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

/// Enum-based backoff implementation to avoid heap allocation
enum BackoffImpl {
    Fixed(FixedBackoff),
    Exponential(backoff::ExponentialBackoff),
    Linear(LinearBackoff),
}

impl BackoffImpl {
    fn next_backoff(&mut self) -> Option<Duration> {
        match self {
            BackoffImpl::Fixed(b) => b.next_backoff(),
            BackoffImpl::Exponential(b) => b.next_backoff(),
            BackoffImpl::Linear(b) => b.next_backoff(),
        }
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
        let active_states = match self.state_store.list_active().await {
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

    /// Calculate how long a step needs to wait based on delay and/or scheduled_at.
    /// Returns None if no waiting is needed.
    fn calculate_wait_duration(step: &StepDefinition<D>) -> Option<Duration> {
        let now = Utc::now();

        // Calculate wait time for scheduled_at
        let schedule_wait = step.scheduled_at.and_then(|scheduled_time| {
            if now < scheduled_time {
                (scheduled_time - now).to_std().ok()
            } else {
                // Scheduled time is in the past
                tracing::debug!(
                    step_id = %step.id,
                    scheduled_time = %scheduled_time,
                    "Step scheduled_at is in the past, proceeding immediately"
                );
                None
            }
        });

        // Combine delay and schedule_wait (both apply if both are set)
        match (step.delay, schedule_wait) {
            (Some(delay), Some(schedule)) => Some(delay + schedule),
            (Some(delay), None) => Some(delay),
            (None, Some(schedule)) => Some(schedule),
            (None, None) => None,
        }
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
                        state_store.cleanup_old_workflows(ttl).await;
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
    #[must_use = "registration result should be checked"]
    pub fn register_workflow(
        &self,
        mut definition: WorkflowDefinition<D>,
    ) -> Result<(), super::definition::ValidationError> {
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
    #[must_use = "workflow instance ID should be stored or awaited"]
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

        self.state_store.save(state).await?;

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
            if self.state_store.is_cancelled(instance_id).await? {
                self.event_bus
                    .publish(WorkflowEvent::WorkflowCancelled { instance_id })
                    .await;
                return Ok(());
            }

            // Phase 0: Drain any pending completion signals to ensure dependents are added
            // This prevents race conditions where tasks finish but their signals aren't processed
            while let Ok((step_id, result)) = rx.try_recv() {
                tracing::debug!(
                    step_id = %step_id,
                    result = ?result,
                    "Step completed (startup drain)"
                );
                if matches!(result, StepResult::Success | StepResult::Skip) {
                    for &dep_idx in definition.get_dependent_indices(&step_id) {
                        pending_check.push_back(dep_idx);
                    }
                }
            }

            // Phase 1: Check waiting steps + deps-ready steps + blocked detection
            // Single lock acquisition for all read operations
            let (
                newly_ready_from_wait,
                deps_ready_indices,
                total_processed,
                current_running,
                current_waiting,
            ) = {
                let t = tracker.read();

                // O(w) iteration over waiting_until keys instead of O(n) over all steps
                let wait_ready: Vec<usize> = t.get_ready_waiting_indices();

                // Check pending_check for dependency-satisfied steps
                let deps_ready: Vec<usize> = pending_check
                    .drain(..)
                    .filter(|&idx| {
                        let step = &definition.steps[idx];
                        t.is_step_processable(&step.id, idx)
                            && t.are_dependencies_satisfied(&step.depends_on)
                            && t.is_any_dependency_satisfied(&step.depends_on_any)
                            && !t.has_failed_dependency(&step.depends_on)
                            && !t.have_all_any_deps_failed(&step.depends_on_any)
                    })
                    .collect();

                (
                    wait_ready,
                    deps_ready,
                    t.total_processed(),
                    t.running.len(),
                    t.waiting_until.len(),
                )
            };

            // Phase 2: Process waiting and deps-ready steps, update waiting_until
            // Single write lock for all mutations
            // Returns (ready_to_launch, steps_added_to_waiting)
            let (ready_to_launch, steps_added_to_waiting) = {
                let now = std::time::Instant::now();
                let mut t = tracker.write();
                let mut added_to_waiting = 0usize;

                // Clear waiting status for ready steps
                for &idx in &newly_ready_from_wait {
                    t.clear_waiting(idx);
                }

                // Collect steps ready to launch
                let mut ready = newly_ready_from_wait;

                for idx in deps_ready_indices {
                    let step = &definition.steps[idx];
                    let wait_duration = Self::calculate_wait_duration(step);

                    if let Some(duration) = wait_duration {
                        if duration > Duration::ZERO {
                            // Step needs to wait - add to waiting_until
                            let ready_at = now + duration;
                            tracing::debug!(
                                step_id = %step.id,
                                wait_ms = duration.as_millis(),
                                "Step waiting for delay/schedule"
                            );
                            t.set_waiting(idx, ready_at);
                            added_to_waiting += 1;
                            continue;
                        }
                    }
                    ready.push(idx);
                }

                (ready, added_to_waiting)
            };

            // Check if we're done
            if total_processed == step_count {
                break;
            }

            // Handle blocked workflow (no ready steps, none running/waiting, but work remains)
            // Use current_running/current_waiting from Phase 1, adjusted for Phase 2 changes.
            // current_running may be stale-high (tasks completed since), but that's safe -
            // we'd just do an extra loop iteration. current_waiting needs adjustment for
            // steps we just added to waiting in Phase 2.
            let effective_waiting = current_waiting + steps_added_to_waiting;
            if ready_to_launch.is_empty()
                && current_running == 0
                && effective_waiting == 0
                && pending_check.is_empty()
            {
                let failed_step = tracker.read().failed.iter().next().cloned();
                // Use &'static str to avoid allocation in common error paths
                let error_message: &'static str = if failed_step.is_some() {
                    "Workflow failed due to step dependency failure"
                } else {
                    "Workflow deadlocked: no steps ready and none running"
                };

                self.state_store
                    .update(instance_id, |s| {
                        s.status = WorkflowStatus::Failed;
                    })
                    .await?;
                self.event_bus
                    .publish(WorkflowEvent::WorkflowFailed {
                        instance_id,
                        failed_step: failed_step
                            .unwrap_or_else(|| StepId::new("internal_scheduler")),
                        error: error_message.to_string(),
                    })
                    .await;
                return Ok(());
            }

            // Launch ready steps in parallel
            let tasks_launched = ready_to_launch.len();
            if tasks_launched > 0 {
                let mut t = tracker.write();
                for &idx in &ready_to_launch {
                    t.running.insert(definition.steps[idx].id.clone());
                }
            }

            for step_idx in ready_to_launch {
                let step = &definition.steps[step_idx];
                let engine = self.clone_for_execution();
                let def = Arc::clone(&definition);
                let step_id = step.id.clone();
                let tx = tx.clone();
                let tracker = Arc::clone(&tracker);

                tokio::spawn(async move {
                    let step = &def.steps[step_idx];

                    // Evaluate run_if condition if present
                    if let Some(ref condition) = step.run_if {
                        match engine.state_store.get_context(instance_id).await {
                            Ok(ctx) => {
                                if !condition(&ctx) {
                                    // Condition is false - skip this step
                                    tracing::debug!(
                                        step_id = %step_id,
                                        "Step skipped due to run_if condition"
                                    );

                                    // Update tracker and send skip signal
                                    {
                                        let mut t = tracker.write();
                                        t.running.remove(&step_id);
                                        t.skipped.insert(step_id.clone());
                                    }

                                    // Update state store
                                    let _ = engine
                                        .state_store
                                        .update(instance_id, |s| {
                                            if let Some(step_state) =
                                                s.step_states.get_mut(&step_id)
                                            {
                                                step_state.status = StepStatus::Skipped;
                                            }
                                        })
                                        .await;

                                    // Send skip signal
                                    let _ = tx.try_send((step_id, StepResult::Skip));
                                    return;
                                }
                            }
                            Err(e) => {
                                // Failed to get context - fail the step rather than proceeding blindly
                                tracing::error!(
                                    step_id = %step_id,
                                    error = ?e,
                                    "Failed to get context for run_if evaluation, failing step"
                                );

                                // Update tracker
                                {
                                    let mut t = tracker.write();
                                    t.running.remove(&step_id);
                                    t.failed.insert(step_id.clone());
                                }

                                // Update state store
                                let _ = engine
                                    .state_store
                                    .update(instance_id, |s| {
                                        if let Some(step_state) = s.step_states.get_mut(&step_id) {
                                            step_state.status = StepStatus::Failed;
                                            step_state.last_error =
                                                Some(format!("run_if context error: {e}"));
                                        }
                                    })
                                    .await;

                                // Send failure signal
                                let _ = tx.try_send((step_id, StepResult::Failure));
                                return;
                            }
                        }
                    }

                    let result = engine
                        .execute_step_with_retry(instance_id, step, &def)
                        .await;

                    let signal = match &result {
                        Ok(r) => *r,
                        Err(_) => StepResult::Failure,
                    };

                    // Track whether we need to update state to Skipped after releasing lock
                    let needs_skip_update = {
                        let mut t = tracker.write();
                        t.running.remove(&step_id);

                        let needs_update = match result {
                            Ok(StepResult::Success) => {
                                t.completed.insert(step_id.clone());
                                false
                            }
                            Ok(StepResult::Skip) => {
                                t.skipped.insert(step_id.clone());
                                false
                            }
                            Ok(StepResult::Failure) | Err(_) => match step.on_failure {
                                FailureAction::FailWorkflow | FailureAction::RetryIndefinitely => {
                                    t.failed.insert(step_id.clone());
                                    false
                                }
                                FailureAction::ContinueNextStep => {
                                    t.skipped.insert(step_id.clone());
                                    true // Need to update state store after releasing lock
                                }
                            },
                        };

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

                        needs_update
                    };

                    // Perform async state update after releasing the tracker lock
                    if needs_skip_update {
                        if let Err(e) = engine
                            .state_store
                            .update(instance_id, |s| {
                                if let Some(step_state) = s.step_states.get_mut(&step_id) {
                                    step_state.status = StepStatus::Skipped;
                                }
                            })
                            .await
                        {
                            tracing::warn!(
                                step_id = %step_id,
                                error = ?e,
                                "Failed to update step state to Skipped"
                            );
                        }
                    }
                });
            }

            // Single lock read for both checks
            let (has_running, has_waiting) = {
                let t = tracker.read();
                (
                    tasks_launched > 0 || !t.running.is_empty(),
                    !t.waiting_until.is_empty(),
                )
            };

            if has_running {
                // Wait for a step to complete; Phase 0 will drain any others
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
            } else if has_waiting {
                // No running tasks, but some steps are waiting for their delay/schedule
                // Calculate how long to sleep based on the soonest waiting step
                let sleep_duration = {
                    let t = tracker.read();
                    let now = std::time::Instant::now();
                    t.waiting_until
                        .values()
                        .filter_map(|&ready_at| {
                            if ready_at > now {
                                Some(ready_at - now)
                            } else {
                                None
                            }
                        })
                        .min()
                        .unwrap_or(Duration::from_millis(10))
                };

                // Sleep until the next step is ready (with a cap to allow cancellation checks)
                let capped_sleep = sleep_duration.min(Duration::from_millis(100));
                tokio::time::sleep(capped_sleep).await;
            }
        }

        let failed_step = {
            let t = tracker.read();
            t.failed.iter().next().cloned()
        };

        if let Some(ref step) = failed_step {
            self.state_store
                .update(instance_id, |s| {
                    s.status = WorkflowStatus::Failed;
                })
                .await?;
            self.event_bus
                .publish(WorkflowEvent::WorkflowFailed {
                    instance_id,
                    failed_step: step.clone(),
                    error: "One or more steps failed".into(),
                })
                .await;
        } else {
            self.state_store
                .update(instance_id, |s| {
                    s.status = WorkflowStatus::Completed;
                })
                .await?;

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
            if self.state_store.is_cancelled(instance_id).await? {
                return Err(WorkflowError::Cancelled(instance_id));
            }

            // Update step state
            self.state_store
                .update(instance_id, |s| {
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
                })
                .await?;

            // Emit step started event
            self.event_bus
                .publish(WorkflowEvent::StepStarted {
                    instance_id,
                    step_id: step.id.clone(),
                    attempt,
                })
                .await;

            let mut context = self.state_store.get_context(instance_id).await?;

            // Execute step with timeout
            let step_start = std::time::Instant::now();
            let result = timeout(step_timeout, step.executor.execute(&mut context)).await;

            let step_duration = step_start.elapsed();

            self.state_store
                .update(instance_id, |s| {
                    s.context = context.clone();
                })
                .await?;

            match result {
                Ok(Ok(StepResult::Success)) => {
                    // Step succeeded
                    self.state_store
                        .update(instance_id, |s| {
                            if let Some(step_state) = s.step_states.get_mut(&step.id) {
                                step_state.status = StepStatus::Succeeded;
                                step_state.completed_at = Some(Utc::now());
                            }
                        })
                        .await?;

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
                    self.state_store
                        .update(instance_id, |s| {
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
                        })
                        .await?;

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

    fn create_backoff(strategy: &BackoffStrategy) -> BackoffImpl {
        match strategy {
            BackoffStrategy::Fixed(duration) => BackoffImpl::Fixed(FixedBackoff(*duration)),
            BackoffStrategy::Exponential { base, max } => {
                let backoff = ExponentialBackoffBuilder::new()
                    .with_initial_interval(*base)
                    .with_max_interval(*max)
                    .with_max_elapsed_time(None)
                    .build();
                BackoffImpl::Exponential(backoff)
            }
            BackoffStrategy::Linear { increment, max } => {
                BackoffImpl::Linear(LinearBackoff::new(*increment, *max))
            }
        }
    }

    /// Cancel a running workflow
    pub async fn cancel_workflow(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<()> {
        self.state_store
            .update(instance_id, |s| {
                s.status = WorkflowStatus::Cancelled;
            })
            .await?;

        self.event_bus
            .publish(WorkflowEvent::WorkflowCancelled { instance_id })
            .await;

        Ok(())
    }

    /// Get workflow status
    pub async fn get_status(
        &self,
        instance_id: WorkflowInstanceId,
    ) -> WorkflowResult<WorkflowState<D>> {
        self.state_store.load(instance_id).await
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
                .await
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

            self.state_store.cleanup_if_terminal(instance_id).await;
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
