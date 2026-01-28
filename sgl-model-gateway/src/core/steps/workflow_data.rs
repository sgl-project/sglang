//! Typed workflow data structures
//!
//! This module defines the typed data structures for all workflows, enabling
//! compile-time type safety and state persistence. Each workflow has its own
//! strongly-typed data structure, and steps are typed to their specific workflow.
//!
//! # Shared Step Trait
//!
//! For steps that are shared between local and external worker workflows,
//! we use the `WorkerRegistrationData` trait. This trait provides a common
//! interface for accessing worker data while maintaining full type safety.

use std::{collections::HashMap, sync::Arc};

use serde::{Deserialize, Serialize};

use super::{
    mcp_registration::McpServerConfigRequest, tokenizer_registration::TokenizerConfigRequest,
    wasm_module_registration::WasmModuleConfigRequest,
    wasm_module_removal::WasmModuleRemovalRequest, worker::local::WorkerRemovalRequest,
};
/// Re-export the protocol types for convenience
pub use crate::protocols::worker_spec::{
    WorkerConfigRequest, WorkerUpdateRequest as ProtocolUpdateRequest,
};
use crate::{
    app_context::AppContext,
    core::{model_card::ModelCard, Worker},
    protocols::worker_spec::{
        WorkerConfigRequest as ProtocolWorkerConfigRequest,
        WorkerUpdateRequest as ProtocolWorkerUpdateRequest,
    },
    workflow::{WorkflowData, WorkflowError},
};

// ============================================================================
// Shared trait for worker registration workflows
// ============================================================================

/// Trait for workflow data that supports worker registration operations.
///
/// This trait is implemented by both `LocalWorkerWorkflowData` and
/// `ExternalWorkerWorkflowData`, allowing shared steps to work with either
/// workflow type while maintaining full type safety.
pub trait WorkerRegistrationData: WorkflowData {
    /// Get the application context (transient, not serialized).
    fn get_app_context(&self) -> Option<&Arc<AppContext>>;

    /// Get the actual worker objects (transient, not serialized).
    fn get_actual_workers(&self) -> Option<&Vec<Arc<dyn Worker>>>;

    /// Get the labels for policy registration.
    fn get_labels(&self) -> Option<&HashMap<String, String>>;
}

/// Wrapper for worker list that can be serialized
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkerList {
    /// Worker URLs (we can't serialize Arc<dyn Worker>, so we store URLs)
    pub worker_urls: Vec<String>,
}

impl WorkerList {
    pub fn new() -> Self {
        Self {
            worker_urls: Vec::new(),
        }
    }

    pub fn from_workers(workers: &[Arc<dyn Worker>]) -> Self {
        Self {
            worker_urls: workers.iter().map(|w| w.url().to_string()).collect(),
        }
    }
}

// ============================================================================
// Workflow-specific data types
// ============================================================================

/// Data for tokenizer registration workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerWorkflowData {
    pub config: TokenizerConfigRequest,
    pub vocab_size: Option<usize>,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
}

impl WorkflowData for TokenizerWorkflowData {
    fn workflow_type() -> &'static str {
        "tokenizer_registration"
    }
}

impl TokenizerWorkflowData {
    /// Validate that all transient fields are properly initialized.
    ///
    /// Call this after deserializing workflow state to ensure runtime fields
    /// have been repopulated.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}

/// Data for local worker registration workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalWorkerWorkflowData {
    pub config: ProtocolWorkerConfigRequest,
    pub connection_mode: Option<crate::core::ConnectionMode>,
    pub discovered_labels: HashMap<String, String>,
    pub dp_info: Option<super::worker::local::DpInfo>,
    pub workers: Option<WorkerList>,
    pub final_labels: HashMap<String, String>,
    /// Detected runtime type (for gRPC workers)
    pub detected_runtime_type: Option<String>,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
    /// Actual worker objects (transient, not serialized)
    #[serde(skip, default)]
    pub actual_workers: Option<Vec<Arc<dyn Worker>>>,
}

impl WorkflowData for LocalWorkerWorkflowData {
    fn workflow_type() -> &'static str {
        "local_worker_registration"
    }
}

impl LocalWorkerWorkflowData {
    /// Validate that all transient fields are properly initialized.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}

impl WorkerRegistrationData for LocalWorkerWorkflowData {
    fn get_app_context(&self) -> Option<&Arc<AppContext>> {
        self.app_context.as_ref()
    }

    fn get_actual_workers(&self) -> Option<&Vec<Arc<dyn Worker>>> {
        self.actual_workers.as_ref()
    }

    fn get_labels(&self) -> Option<&HashMap<String, String>> {
        Some(&self.final_labels)
    }
}

/// Data for external worker registration workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalWorkerWorkflowData {
    pub config: ProtocolWorkerConfigRequest,
    /// Discovered model cards from /v1/models endpoint
    pub model_cards: Vec<ModelCard>,
    pub workers: Option<WorkerList>,
    /// Labels for policies (derived from config)
    pub labels: HashMap<String, String>,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
    /// Actual worker objects (transient, not serialized)
    #[serde(skip, default)]
    pub actual_workers: Option<Vec<Arc<dyn Worker>>>,
}

impl WorkflowData for ExternalWorkerWorkflowData {
    fn workflow_type() -> &'static str {
        "external_worker_registration"
    }
}

impl ExternalWorkerWorkflowData {
    /// Validate that all transient fields are properly initialized.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}

impl WorkerRegistrationData for ExternalWorkerWorkflowData {
    fn get_app_context(&self) -> Option<&Arc<AppContext>> {
        self.app_context.as_ref()
    }

    fn get_actual_workers(&self) -> Option<&Vec<Arc<dyn Worker>>> {
        self.actual_workers.as_ref()
    }

    fn get_labels(&self) -> Option<&HashMap<String, String>> {
        Some(&self.labels)
    }
}

/// Data for worker removal workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRemovalWorkflowData {
    pub config: WorkerRemovalRequest,
    pub workers_to_remove: Option<WorkerList>,
    /// URLs of workers being removed
    pub worker_urls: Vec<String>,
    /// Model IDs affected by the removal
    pub affected_models: std::collections::HashSet<String>,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
    /// Actual worker objects to remove (transient, not serialized)
    #[serde(skip, default)]
    pub actual_workers_to_remove: Option<Vec<Arc<dyn Worker>>>,
}

impl WorkflowData for WorkerRemovalWorkflowData {
    fn workflow_type() -> &'static str {
        "worker_removal"
    }
}

impl WorkerRemovalWorkflowData {
    /// Validate that all transient fields are properly initialized.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}

/// Data for worker update workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerUpdateWorkflowData {
    pub config: ProtocolWorkerUpdateRequest,
    /// URL of worker(s) to update
    pub worker_url: String,
    /// Whether to update all DP-aware workers with matching prefix
    pub dp_aware: bool,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
    /// Workers to update (transient, not serialized)
    #[serde(skip, default)]
    pub workers_to_update: Option<Vec<Arc<dyn Worker>>>,
    /// Updated worker objects (transient, not serialized)
    #[serde(skip, default)]
    pub updated_workers: Option<Vec<Arc<dyn Worker>>>,
}

impl WorkflowData for WorkerUpdateWorkflowData {
    fn workflow_type() -> &'static str {
        "worker_update"
    }
}

impl WorkerUpdateWorkflowData {
    /// Validate that all transient fields are properly initialized.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}

/// Data for MCP server registration workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpWorkflowData {
    pub config: McpServerConfigRequest,
    pub validated: bool,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
    /// Connected MCP client (transient, not serialized)
    #[serde(skip, default)]
    pub mcp_client: Option<Arc<rmcp::service::RunningService<rmcp::RoleClient, ()>>>,
}

impl WorkflowData for McpWorkflowData {
    fn workflow_type() -> &'static str {
        "mcp_registration"
    }
}

impl McpWorkflowData {
    /// Validate that all transient fields are properly initialized.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}

/// Data for WASM module registration workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRegistrationWorkflowData {
    pub config: WasmModuleConfigRequest,
    pub wasm_bytes: Option<Vec<u8>>,
    /// SHA256 hash of the module file (32 bytes)
    pub sha256_hash: Option<[u8; 32]>,
    /// File size in bytes
    pub file_size_bytes: Option<u64>,
    /// UUID assigned to the registered module
    pub module_uuid: Option<uuid::Uuid>,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
}

impl WorkflowData for WasmRegistrationWorkflowData {
    fn workflow_type() -> &'static str {
        "wasm_module_registration"
    }
}

impl WasmRegistrationWorkflowData {
    /// Validate that all transient fields are properly initialized.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}

/// Data for WASM module removal workflow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRemovalWorkflowData {
    pub config: WasmModuleRemovalRequest,
    pub module_id: Option<String>,
    /// Application context (transient, must be re-initialized after deserialization)
    #[serde(skip, default)]
    pub app_context: Option<Arc<AppContext>>,
}

impl WorkflowData for WasmRemovalWorkflowData {
    fn workflow_type() -> &'static str {
        "wasm_module_removal"
    }
}

impl WasmRemovalWorkflowData {
    /// Validate that all transient fields are properly initialized.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        if self.app_context.is_none() {
            return Err(WorkflowError::ContextValueNotFound(
                "app_context not initialized after deserialization".into(),
            ));
        }
        Ok(())
    }
}
