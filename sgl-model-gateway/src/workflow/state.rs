//! Workflow state management

use std::{collections::HashMap, marker::PhantomData, sync::Arc, time::Duration};

use async_trait::async_trait;
use parking_lot::RwLock;

use super::types::{
    WorkflowContext, WorkflowData, WorkflowError, WorkflowInstanceId, WorkflowResult,
    WorkflowState, WorkflowStatus,
};

/// Trait for workflow state persistence.
///
/// Implement this trait to provide custom storage backends (e.g., PostgreSQL, Redis).
/// The default implementation is `InMemoryStore` which keeps state in memory.
#[async_trait]
pub trait StateStore<D: WorkflowData>: Send + Sync + Clone {
    /// Save workflow state
    async fn save(&self, state: WorkflowState<D>) -> WorkflowResult<()>;

    /// Load workflow state by instance ID
    async fn load(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<WorkflowState<D>>;

    /// Update workflow state using a closure
    async fn update<F>(&self, instance_id: WorkflowInstanceId, f: F) -> WorkflowResult<()>
    where
        F: FnOnce(&mut WorkflowState<D>) + Send;

    /// Delete workflow state
    async fn delete(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<()>;

    /// List all active workflows (Running or Pending)
    async fn list_active(&self) -> WorkflowResult<Vec<WorkflowState<D>>>;

    /// List all workflows
    async fn list_all(&self) -> WorkflowResult<Vec<WorkflowState<D>>>;

    /// Check if workflow is cancelled without loading full state
    async fn is_cancelled(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<bool>;

    /// Clean up old completed/failed/cancelled workflows beyond a time threshold
    async fn cleanup_old_workflows(&self, ttl: Duration) -> usize;

    /// Get just the workflow context without cloning the entire state
    async fn get_context(
        &self,
        instance_id: WorkflowInstanceId,
    ) -> WorkflowResult<WorkflowContext<D>>;

    /// Clean up a specific workflow immediately if it's in a terminal state
    /// Returns true if the workflow was removed, false otherwise
    async fn cleanup_if_terminal(&self, instance_id: WorkflowInstanceId) -> bool;
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
}

impl<D: WorkflowData> Default for InMemoryStore<D> {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl<D: WorkflowData> StateStore<D> for InMemoryStore<D> {
    async fn save(&self, state: WorkflowState<D>) -> WorkflowResult<()> {
        self.states.write().insert(state.instance_id, state);
        Ok(())
    }

    async fn load(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<WorkflowState<D>> {
        self.states
            .read()
            .get(&instance_id)
            .cloned()
            .ok_or(WorkflowError::NotFound(instance_id))
    }

    async fn update<F>(&self, instance_id: WorkflowInstanceId, f: F) -> WorkflowResult<()>
    where
        F: FnOnce(&mut WorkflowState<D>) + Send,
    {
        let mut states = self.states.write();
        let state = states
            .get_mut(&instance_id)
            .ok_or(WorkflowError::NotFound(instance_id))?;
        f(state);
        state.updated_at = chrono::Utc::now();
        Ok(())
    }

    async fn delete(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<()> {
        self.states.write().remove(&instance_id);
        Ok(())
    }

    async fn list_active(&self) -> WorkflowResult<Vec<WorkflowState<D>>> {
        let states = self.states.read();
        Ok(states
            .values()
            .filter(|s| matches!(s.status, WorkflowStatus::Running | WorkflowStatus::Pending))
            .cloned()
            .collect())
    }

    async fn list_all(&self) -> WorkflowResult<Vec<WorkflowState<D>>> {
        let states = self.states.read();
        Ok(states.values().cloned().collect())
    }

    async fn is_cancelled(&self, instance_id: WorkflowInstanceId) -> WorkflowResult<bool> {
        self.states
            .read()
            .get(&instance_id)
            .map(|s| s.status == WorkflowStatus::Cancelled)
            .ok_or(WorkflowError::NotFound(instance_id))
    }

    async fn cleanup_old_workflows(&self, ttl: Duration) -> usize {
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

    async fn get_context(
        &self,
        instance_id: WorkflowInstanceId,
    ) -> WorkflowResult<WorkflowContext<D>> {
        self.states
            .read()
            .get(&instance_id)
            .map(|s| s.context.clone())
            .ok_or(WorkflowError::NotFound(instance_id))
    }

    async fn cleanup_if_terminal(&self, instance_id: WorkflowInstanceId) -> bool {
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
