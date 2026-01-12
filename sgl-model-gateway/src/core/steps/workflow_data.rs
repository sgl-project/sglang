//! Typed workflow data structures
//!
//! This module defines the typed data structures for all workflows, enabling
//! compile-time type safety and state persistence.

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

// ============================================================================
// Unified enum for all workflow types
// ============================================================================

/// Macro to generate type-safe accessor methods for AnyWorkflowData variants.
///
/// This reduces boilerplate and ensures consistent error handling across all accessors.
macro_rules! impl_workflow_accessor {
    ($fn_name:ident, $fn_name_mut:ident, $variant:ident, $ty:ty, $type_name:expr) => {
        /// Extract the inner data, returning an error if this is a different variant.
        #[must_use = "this returns the result of the operation, without modifying the original"]
        pub fn $fn_name(&self) -> Result<&$ty, WorkflowError> {
            match self {
                AnyWorkflowData::$variant(data) => Ok(data),
                _ => Err(WorkflowError::TypeMismatch {
                    expected: $type_name,
                    actual: self.concrete_type(),
                }),
            }
        }

        /// Extract the inner data mutably, returning an error if this is a different variant.
        pub fn $fn_name_mut(&mut self) -> Result<&mut $ty, WorkflowError> {
            // Store the type name before the mutable borrow
            let actual = self.concrete_type();
            match self {
                AnyWorkflowData::$variant(data) => Ok(data),
                _ => Err(WorkflowError::TypeMismatch {
                    expected: $type_name,
                    actual,
                }),
            }
        }
    };
}

/// Macro to generate From implementations for AnyWorkflowData variants.
macro_rules! impl_from_workflow_data {
    ($variant:ident, $ty:ty) => {
        impl From<$ty> for AnyWorkflowData {
            fn from(data: $ty) -> Self {
                AnyWorkflowData::$variant(data)
            }
        }
    };
}

/// Unified workflow data enum covering all workflow types.
///
/// This allows a single `WorkflowEngine<AnyWorkflowData>` to handle all workflows
/// while maintaining type safety at the step level.
///
/// # Type Erasure
///
/// `AnyWorkflowData` implements `WorkflowData` with `workflow_type()` returning `"any"`.
/// This is intentional: the static method cannot know the runtime variant. Use
/// [`concrete_type()`](Self::concrete_type) to get the actual workflow type at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnyWorkflowData {
    Tokenizer(TokenizerWorkflowData),
    LocalWorker(LocalWorkerWorkflowData),
    ExternalWorker(ExternalWorkerWorkflowData),
    WorkerRemoval(WorkerRemovalWorkflowData),
    WorkerUpdate(WorkerUpdateWorkflowData),
    Mcp(McpWorkflowData),
    WasmRegistration(WasmRegistrationWorkflowData),
    WasmRemoval(WasmRemovalWorkflowData),
}

impl WorkflowData for AnyWorkflowData {
    /// Returns `"any"` as this is a type-erased container.
    ///
    /// Use [`concrete_type()`](Self::concrete_type) to get the actual workflow type at runtime.
    fn workflow_type() -> &'static str {
        "any"
    }
}

// Generate From implementations for ergonomic construction
impl_from_workflow_data!(Tokenizer, TokenizerWorkflowData);
impl_from_workflow_data!(LocalWorker, LocalWorkerWorkflowData);
impl_from_workflow_data!(ExternalWorker, ExternalWorkerWorkflowData);
impl_from_workflow_data!(WorkerRemoval, WorkerRemovalWorkflowData);
impl_from_workflow_data!(WorkerUpdate, WorkerUpdateWorkflowData);
impl_from_workflow_data!(Mcp, McpWorkflowData);
impl_from_workflow_data!(WasmRegistration, WasmRegistrationWorkflowData);
impl_from_workflow_data!(WasmRemoval, WasmRemovalWorkflowData);

impl AnyWorkflowData {
    /// Get the concrete workflow type name at runtime.
    ///
    /// Unlike the static `workflow_type()` method, this returns the actual
    /// type of the contained workflow data.
    #[must_use]
    pub fn concrete_type(&self) -> &'static str {
        match self {
            AnyWorkflowData::Tokenizer(_) => TokenizerWorkflowData::workflow_type(),
            AnyWorkflowData::LocalWorker(_) => LocalWorkerWorkflowData::workflow_type(),
            AnyWorkflowData::ExternalWorker(_) => ExternalWorkerWorkflowData::workflow_type(),
            AnyWorkflowData::WorkerRemoval(_) => WorkerRemovalWorkflowData::workflow_type(),
            AnyWorkflowData::WorkerUpdate(_) => WorkerUpdateWorkflowData::workflow_type(),
            AnyWorkflowData::Mcp(_) => McpWorkflowData::workflow_type(),
            AnyWorkflowData::WasmRegistration(_) => WasmRegistrationWorkflowData::workflow_type(),
            AnyWorkflowData::WasmRemoval(_) => WasmRemovalWorkflowData::workflow_type(),
        }
    }

    // Generate all accessor methods using the macro
    impl_workflow_accessor!(
        as_tokenizer,
        as_tokenizer_mut,
        Tokenizer,
        TokenizerWorkflowData,
        "tokenizer_registration"
    );
    impl_workflow_accessor!(
        as_local_worker,
        as_local_worker_mut,
        LocalWorker,
        LocalWorkerWorkflowData,
        "local_worker_registration"
    );
    impl_workflow_accessor!(
        as_external_worker,
        as_external_worker_mut,
        ExternalWorker,
        ExternalWorkerWorkflowData,
        "external_worker_registration"
    );
    impl_workflow_accessor!(
        as_worker_removal,
        as_worker_removal_mut,
        WorkerRemoval,
        WorkerRemovalWorkflowData,
        "worker_removal"
    );
    impl_workflow_accessor!(
        as_worker_update,
        as_worker_update_mut,
        WorkerUpdate,
        WorkerUpdateWorkflowData,
        "worker_update"
    );
    impl_workflow_accessor!(as_mcp, as_mcp_mut, Mcp, McpWorkflowData, "mcp_registration");
    impl_workflow_accessor!(
        as_wasm_registration,
        as_wasm_registration_mut,
        WasmRegistration,
        WasmRegistrationWorkflowData,
        "wasm_module_registration"
    );
    impl_workflow_accessor!(
        as_wasm_removal,
        as_wasm_removal_mut,
        WasmRemoval,
        WasmRemovalWorkflowData,
        "wasm_module_removal"
    );

    // ========================================================================
    // Helper methods for shared worker steps
    // ========================================================================

    /// Get app_context from any workflow data type that has it.
    #[must_use]
    pub fn get_app_context(&self) -> Option<&Arc<AppContext>> {
        match self {
            AnyWorkflowData::Tokenizer(d) => d.app_context.as_ref(),
            AnyWorkflowData::LocalWorker(d) => d.app_context.as_ref(),
            AnyWorkflowData::ExternalWorker(d) => d.app_context.as_ref(),
            AnyWorkflowData::WorkerRemoval(d) => d.app_context.as_ref(),
            AnyWorkflowData::WorkerUpdate(d) => d.app_context.as_ref(),
            AnyWorkflowData::Mcp(d) => d.app_context.as_ref(),
            AnyWorkflowData::WasmRegistration(d) => d.app_context.as_ref(),
            AnyWorkflowData::WasmRemoval(d) => d.app_context.as_ref(),
        }
    }

    /// Get actual workers from local or external worker workflows.
    #[must_use]
    pub fn get_actual_workers(&self) -> Option<&Vec<Arc<dyn Worker>>> {
        match self {
            AnyWorkflowData::LocalWorker(d) => d.actual_workers.as_ref(),
            AnyWorkflowData::ExternalWorker(d) => d.actual_workers.as_ref(),
            _ => None,
        }
    }

    /// Set actual workers for local or external worker workflows.
    pub fn set_actual_workers(
        &mut self,
        workers: Vec<Arc<dyn Worker>>,
    ) -> Result<(), WorkflowError> {
        match self {
            AnyWorkflowData::LocalWorker(d) => {
                d.workers = Some(WorkerList::from_workers(&workers));
                d.actual_workers = Some(workers);
                Ok(())
            }
            AnyWorkflowData::ExternalWorker(d) => {
                d.workers = Some(WorkerList::from_workers(&workers));
                d.actual_workers = Some(workers);
                Ok(())
            }
            _ => Err(WorkflowError::TypeMismatch {
                expected: "LocalWorker or ExternalWorker",
                actual: self.concrete_type(),
            }),
        }
    }

    /// Get labels for policy configuration (from local or external worker workflows).
    #[must_use]
    pub fn get_labels(&self) -> Option<&HashMap<String, String>> {
        match self {
            AnyWorkflowData::LocalWorker(d) => Some(&d.final_labels),
            AnyWorkflowData::ExternalWorker(d) => Some(&d.labels),
            _ => None,
        }
    }

    /// Validate that all transient fields are properly initialized.
    ///
    /// Call this after deserializing workflow state to ensure runtime fields
    /// have been repopulated.
    pub fn validate_initialized(&self) -> Result<(), WorkflowError> {
        match self {
            AnyWorkflowData::Tokenizer(d) => d.validate_initialized(),
            AnyWorkflowData::LocalWorker(d) => d.validate_initialized(),
            AnyWorkflowData::ExternalWorker(d) => d.validate_initialized(),
            AnyWorkflowData::WorkerRemoval(d) => d.validate_initialized(),
            AnyWorkflowData::WorkerUpdate(d) => d.validate_initialized(),
            AnyWorkflowData::Mcp(d) => d.validate_initialized(),
            AnyWorkflowData::WasmRegistration(d) => d.validate_initialized(),
            AnyWorkflowData::WasmRemoval(d) => d.validate_initialized(),
        }
    }
}
