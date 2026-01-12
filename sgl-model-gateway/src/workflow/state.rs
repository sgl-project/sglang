//! Workflow state management

use std::{collections::HashMap, marker::PhantomData, sync::Arc, time::Duration};

use parking_lot::RwLock;

use super::types::{
    WorkflowContext, WorkflowData, WorkflowError, WorkflowInstanceId, WorkflowResult,
    WorkflowState, WorkflowStatus,
};

/// Trait for workflow state persistence.
///
/// Implement this trait to provide custom storage backends (e.g., PostgreSQL, Redis).
/// The default implementation is `InMemoryStore` which keeps state in memory.
pub trait StateStore<D: WorkflowData>: Send + Sync + Clone {
    /// Save workflow state
    fn save(&self, state: WorkflowState<D>) -> WorkflowResult<()>;

    /// Load workflow state by instance ID
    fn load(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<WorkflowState<D>>;

    /// Update workflow state using a closure
    fn update<F>(&self, instance_id: WorkflowInstanceId, f: F) -> WorkflowResult<()>
    where
        F: FnOnce(&mut WorkflowState<D>);

    /// Delete workflow state
    fn delete(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<()>;

    /// List all active workflows (Running or Pending)
    fn list_active(&self) -> WorkflowResult<Vec<WorkflowState<D>>>;

    /// List all workflows
    fn list_all(&self) -> WorkflowResult<Vec<WorkflowState<D>>>;

    /// Check if workflow is cancelled without loading full state
    fn is_cancelled(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<bool>;

    /// Clean up old completed/failed/cancelled workflows beyond a time threshold
    fn cleanup_old_workflows(&self, ttl: Duration) -> usize;

    /// Get just the workflow context without cloning the entire state
    fn get_context(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<WorkflowContext<D>>;
}

/// In-memory state storage for workflow instances
#[derive(Clone)]
pub struct InMemoryStore<D: WorkflowData> {
    states: Arc<RwLock<HashMap<WorkflowInstanceId, WorkflowState<D>>>>,
    _phantom: PhantomData<D>,
}

impl<D: WorkflowData> InMemoryStore<D> {
    pub fn new() -> Self {
        Self {
            states: Arc::new(RwLock::new(HashMap::new())),
            _phantom: PhantomData,
        }
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

    /// Clean up a specific completed workflow immediately
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

impl<D: WorkflowData> Default for InMemoryStore<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: WorkflowData> StateStore<D> for InMemoryStore<D> {
    fn save(&self, state: WorkflowState<D>) -> WorkflowResult<()> {
        self.states.write().insert(state.instance_id, state);
        Ok(())
    }

    fn load(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<WorkflowState<D>> {
        self.states
            .read()
            .get(&instance_id)
            .cloned()
            .ok_or(WorkflowError::NotFound(instance_id))
    }

    fn list_active(&self) -> WorkflowResult<Vec<WorkflowState<D>>> {
        let states = self.states.read();
        Ok(states
            .values()
            .filter(|s| matches!(s.status, WorkflowStatus::Running | WorkflowStatus::Pending))
            .cloned()
            .collect())
    }

    fn list_all(&self) -> WorkflowResult<Vec<WorkflowState<D>>> {
        let states = self.states.read();
        Ok(states.values().cloned().collect())
    }

    fn delete(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<()> {
        self.states.write().remove(&instance_id);
        Ok(())
    }

    fn update<F>(&self, instance_id: WorkflowInstanceId, f: F) -> WorkflowResult<()>
    where
        F: FnOnce(&mut WorkflowState<D>),
    {
        let mut states = self.states.write();
        let state = states
            .get_mut(&instance_id)
            .ok_or(WorkflowError::NotFound(instance_id))?;
        f(state);
        state.updated_at = chrono::Utc::now();
        Ok(())
    }

    fn get_context(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<WorkflowContext<D>> {
        self.states
            .read()
            .get(&instance_id)
            .map(|s| s.context.clone())
            .ok_or(WorkflowError::NotFound(instance_id))
    }

    fn is_cancelled(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<bool> {
        self.states
            .read()
            .get(&instance_id)
            .map(|s| s.status == WorkflowStatus::Cancelled)
            .ok_or(WorkflowError::NotFound(instance_id))
    }

    fn cleanup_old_workflows(&self, ttl: Duration) -> usize {
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
}
