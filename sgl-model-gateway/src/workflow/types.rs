//! Core workflow types and definitions

use std::{collections::HashMap, fmt, time::Duration};

use chrono::{DateTime, Utc};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use uuid::Uuid;

/// Trait for workflow data that can be passed through workflow steps.
///
/// Implementing this trait allows your data type to be used as the typed
/// context for a workflow. The data must be serializable for state persistence.
///
/// # Example
///
/// ```ignore
/// #[derive(Debug, Clone, Serialize, Deserialize)]
/// pub struct MyWorkflowData {
///     pub config: MyConfig,
///     pub result: Option<MyResult>,
///     #[serde(skip, default)]
///     pub app_context: Option<Arc<AppContext>>,
/// }
///
/// impl WorkflowData for MyWorkflowData {
///     fn workflow_type() -> &'static str { "my_workflow" }
/// }
/// ```
pub trait WorkflowData: Serialize + DeserializeOwned + Send + Sync + Clone + 'static {
    /// Human-readable name for logging and identification
    fn workflow_type() -> &'static str;
}

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
#[serde(bound(
    serialize = "D: Serialize",
    deserialize = "D: serde::de::DeserializeOwned"
))]
pub struct WorkflowState<D: WorkflowData> {
    pub instance_id: WorkflowInstanceId,
    pub definition_id: WorkflowId,
    pub status: WorkflowStatus,
    pub current_step: Option<StepId>,
    pub step_states: HashMap<StepId, StepState>,
    pub context: WorkflowContext<D>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl<D: WorkflowData> WorkflowState<D> {
    pub fn new(instance_id: WorkflowInstanceId, definition_id: WorkflowId, data: D) -> Self {
        let now = Utc::now();
        Self {
            instance_id,
            definition_id,
            status: WorkflowStatus::Pending,
            current_step: None,
            step_states: HashMap::new(),
            context: WorkflowContext::new(instance_id, data),
            created_at: now,
            updated_at: now,
        }
    }
}

/// Shared context passed between workflow steps.
///
/// The context contains typed workflow data that is fully serializable,
/// enabling state persistence and workflow recovery.
///
/// # Type Parameter
///
/// `D` - The workflow-specific data type implementing `WorkflowData`.
/// This type holds all the state needed by workflow steps and must be
/// serializable (except for fields marked with `#[serde(skip)]`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "D: Serialize",
    deserialize = "D: serde::de::DeserializeOwned"
))]
pub struct WorkflowContext<D: WorkflowData> {
    pub instance_id: WorkflowInstanceId,
    pub data: D,
}

impl<D: WorkflowData> WorkflowContext<D> {
    pub fn new(instance_id: WorkflowInstanceId, data: D) -> Self {
        Self { instance_id, data }
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
#[derive(Debug, Clone, thiserror::Error)]
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

    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },

    #[error("Engine is shutting down, not accepting new workflows")]
    ShuttingDown,
}

pub type WorkflowResult<T> = Result<T, WorkflowError>;
