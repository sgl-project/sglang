//! Core workflow types and definitions

use std::{collections::HashMap, fmt, sync::Arc, time::Duration};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique identifier for a workflow definition
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkflowId(String);

impl WorkflowId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for WorkflowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a workflow instance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkflowInstanceId(Uuid);

impl WorkflowInstanceId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for WorkflowInstanceId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for WorkflowInstanceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a workflow step
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StepId(String);

impl StepId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl fmt::Display for StepId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub backoff: BackoffStrategy,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff: BackoffStrategy::Exponential {
                base: Duration::from_secs(1),
                max: Duration::from_secs(30),
            },
        }
    }
}

/// Backoff strategy for retries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed(Duration),
    /// Exponential backoff with base and max duration
    Exponential { base: Duration, max: Duration },
    /// Linear backoff with increment and max duration
    Linear { increment: Duration, max: Duration },
}

/// Action to take when a step fails
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureAction {
    /// Stop the entire workflow
    FailWorkflow,
    /// Skip this step and continue to the next
    ContinueNextStep,
    /// Keep retrying indefinitely until manual intervention
    RetryIndefinitely,
}

/// Workflow execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// Step execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Retrying,
    Skipped,
}

/// State of a workflow step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepState {
    pub status: StepStatus,
    pub attempt: u32,
    pub last_error: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl Default for StepState {
    fn default() -> Self {
        Self {
            status: StepStatus::Pending,
            attempt: 0,
            last_error: None,
            started_at: None,
            completed_at: None,
        }
    }
}

/// Workflow instance state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowState {
    pub instance_id: WorkflowInstanceId,
    pub definition_id: WorkflowId,
    pub status: WorkflowStatus,
    pub current_step: Option<StepId>,
    pub step_states: HashMap<StepId, StepState>,
    pub context: WorkflowContext,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl WorkflowState {
    pub fn new(instance_id: WorkflowInstanceId, definition_id: WorkflowId) -> Self {
        let now = Utc::now();
        Self {
            instance_id,
            definition_id,
            status: WorkflowStatus::Pending,
            current_step: None,
            step_states: HashMap::new(),
            context: WorkflowContext::new(instance_id),
            created_at: now,
            updated_at: now,
        }
    }
}

/// Shared context passed between workflow steps
///
/// # Serialization Warning
///
/// The `data` field contains type-erased values that cannot be serialized.
/// This means workflow context is **not preserved** across:
/// - Process restarts
/// - State persistence to disk
/// - Network serialization
///
/// The workflow engine only supports **in-memory execution**. If you need
/// durable workflows, consider implementing a custom serializable context type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowContext {
    pub instance_id: WorkflowInstanceId,
    #[serde(skip)]
    data: HashMap<String, Arc<dyn std::any::Any + Send + Sync>>,
}

impl WorkflowContext {
    pub fn new(instance_id: WorkflowInstanceId) -> Self {
        Self {
            instance_id,
            data: HashMap::new(),
        }
    }

    /// Store a value in the context (will be wrapped in Arc)
    pub fn set<T: Send + Sync + 'static>(&mut self, key: impl Into<String>, value: T) {
        self.data.insert(key.into(), Arc::new(value));
    }

    /// Store an Arc directly without double-wrapping
    pub fn set_arc<T: Send + Sync + 'static>(&mut self, key: impl Into<String>, value: Arc<T>) {
        self.data.insert(key.into(), value);
    }

    /// Retrieve a value from the context
    pub fn get<T: Send + Sync + 'static>(&self, key: &str) -> Option<Arc<T>> {
        self.data
            .get(key)
            .and_then(|v| v.clone().downcast::<T>().ok())
    }

    /// Check if the context has any data that would be lost during serialization
    pub fn has_unserializable_data(&self) -> bool {
        !self.data.is_empty()
    }

    /// Get the number of context entries (useful for debugging)
    pub fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Result returned by a step execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    Success,
    Failure,
    Skip,
}

/// Error kinds for workflow operations
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorkflowError {
    #[error("Workflow not found: {0}")]
    NotFound(WorkflowInstanceId),

    #[error("Workflow definition not found: {0}")]
    DefinitionNotFound(WorkflowId),

    #[error("Step failed: {step_id} - {message}")]
    StepFailed { step_id: StepId, message: String },

    #[error("Step timeout: {step_id}")]
    StepTimeout { step_id: StepId },

    #[error("Workflow cancelled: {0}")]
    Cancelled(WorkflowInstanceId),

    #[error("Invalid state transition: {from:?} -> {to:?}")]
    InvalidStateTransition {
        from: WorkflowStatus,
        to: WorkflowStatus,
    },

    #[error("Context value not found: {0}")]
    ContextValueNotFound(String),

    #[error("Context value type mismatch: {0}")]
    ContextTypeMismatch(String),
}

pub type WorkflowResult<T> = Result<T, WorkflowError>;
