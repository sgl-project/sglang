//! Workflow state management

use std::{collections::HashMap, sync::Arc};

use parking_lot::RwLock;

use super::types::{
    WorkflowError, WorkflowInstanceId, WorkflowResult, WorkflowState, WorkflowStatus,
};

/// In-memory state storage for workflow instances
#[derive(Clone)]
pub struct WorkflowStateStore {
    states: Arc<RwLock<HashMap<WorkflowInstanceId, WorkflowState>>>,
}

impl WorkflowStateStore {
    pub fn new() -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Save workflow state
    ///
    /// # Warning
    ///
    /// This emits a warning if the workflow context contains unserializable data,
    /// which would be lost if state persistence is later implemented.
    pub fn save(&self, state: WorkflowState) -> WorkflowResult<()> {
        if state.context.has_unserializable_data() {
            tracing::warn!(
                instance_id = %state.instance_id,
                data_count = state.context.data_len(),
                "Saving workflow state with {} unserializable context entries. \
                 This data cannot be persisted and will be lost on restart.",
                state.context.data_len()
            );
        }
        self.states.write().insert(state.instance_id, state);
        Ok(())
    }

    /// Load workflow state by instance ID
    pub fn load(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<WorkflowState> {
        self.states
            .read()
            .get(&instance_id)
            .cloned()
            .ok_or(WorkflowError::NotFound(instance_id))
    }

    /// List all active workflows (Running or Pending)
    pub fn list_active(&self) -> WorkflowResult<Vec<WorkflowState>> {
        let states = self.states.read();
        Ok(states
            .values()
            .filter(|s| matches!(s.status, WorkflowStatus::Running | WorkflowStatus::Pending))
            .cloned()
            .collect())
    }

    /// List all workflows
    pub fn list_all(&self) -> WorkflowResult<Vec<WorkflowState>> {
        let states = self.states.read();
        Ok(states.values().cloned().collect())
    }

    /// Delete workflow state
    pub fn delete(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<()> {
        self.states.write().remove(&instance_id);
        Ok(())
    }

    /// Update workflow state using a closure
    pub fn update<F>(&self, instance_id: WorkflowInstanceId, f: F) -> WorkflowResult<()>
    where
        F: FnOnce(&mut WorkflowState),
    {
        let mut states = self.states.write();
        let state = states
            .get_mut(&instance_id)
            .ok_or(WorkflowError::NotFound(instance_id))?;
        f(state);
        state.updated_at = chrono::Utc::now();
        Ok(())
    }

    /// Get count of workflows by status
    pub fn count_by_status(&self, status: WorkflowStatus) -> usize {
        self.states
            .read()
            .values()
            .filter(|s| s.status == status)
            .count()
    }

    /// Get total count of all workflows
    pub fn count(&self) -> usize {
        self.states.read().len()
    }

    /// Clean up old completed/failed/cancelled workflows beyond a time threshold
    ///
    /// This prevents unbounded memory growth by removing workflow states that
    /// have been in a terminal state (Completed, Failed, Cancelled) for longer
    /// than the specified TTL (time-to-live).
    ///
    /// Active workflows (Running, Pending, Paused) are never cleaned up.
    ///
    /// # Arguments
    ///
    /// * `ttl` - Time-to-live for terminal workflows. Workflows in terminal states
    ///   older than this will be removed.
    ///
    /// # Returns
    ///
    /// The number of workflow states removed.
    pub fn cleanup_old_workflows(&self, ttl: std::time::Duration) -> usize {
        let now = chrono::Utc::now();
        let mut states = self.states.write();
        let initial_count = states.len();

        states.retain(|_, state| {
            // Keep active workflows
            if matches!(
                state.status,
                WorkflowStatus::Running | WorkflowStatus::Pending | WorkflowStatus::Paused
            ) {
                return true;
            }

            // For terminal workflows, check age
            let age = now
                .signed_duration_since(state.updated_at)
                .to_std()
                .unwrap_or_default();
            age < ttl
        });

        let removed_count = initial_count - states.len();
        if removed_count > 0 {
            tracing::info!(
                removed = removed_count,
                remaining = states.len(),
                "Cleaned up old workflow states"
            );
        }
        removed_count
    }

    /// Clean up a specific completed workflow immediately
    ///
    /// This is useful for cleaning up workflows right after they complete
    /// when you know they won't be queried again.
    pub fn cleanup_if_terminal(&self, instance_id: WorkflowInstanceId) -> bool {
        let mut states = self.states.write();
        if let Some(state) = states.get(&instance_id) {
            if matches!(
                state.status,
                WorkflowStatus::Completed | WorkflowStatus::Failed | WorkflowStatus::Cancelled
            ) {
                states.remove(&instance_id);
                return true;
            }
        }
        false
    }
}

impl Default for WorkflowStateStore {
    fn default() -> Self {
        Self::new()
    }
}
