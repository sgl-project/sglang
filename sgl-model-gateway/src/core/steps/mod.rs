//! Workflow step implementations
//!
//! This module contains concrete step implementations for various workflows:
//! - Worker management (registration, removal, updates)
//! - MCP server registration
//! - WASM module registration and removal
//! - Tokenizer registration

pub mod mcp_registration;
pub mod tokenizer_registration;
pub mod wasm_module_registration;
pub mod wasm_module_removal;
pub mod worker;
pub mod workflow_data;
pub mod workflow_engines;

// Worker management (registration, removal)
pub use mcp_registration::{
    create_mcp_registration_workflow, create_mcp_workflow_data, ConnectMcpServerStep,
    DiscoverMcpInventoryStep, McpServerConfigRequest, RegisterMcpServerStep,
    ValidateRegistrationStep,
};
pub use tokenizer_registration::{
    create_tokenizer_registration_workflow, create_tokenizer_workflow_data, LoadTokenizerStep,
    TokenizerConfigRequest, TokenizerRemovalRequest,
};
pub use wasm_module_registration::{
    create_wasm_module_registration_workflow, create_wasm_registration_workflow_data,
    CalculateHashStep, CheckDuplicateStep, LoadWasmBytesStep, RegisterModuleStep,
    ValidateDescriptorStep, ValidateWasmComponentStep, WasmModuleConfigRequest,
};
pub use wasm_module_removal::{
    create_wasm_module_removal_workflow, create_wasm_removal_workflow_data, FindModuleToRemoveStep,
    RemoveModuleStep, WasmModuleRemovalRequest,
};
pub use worker::{
    // Workflow builders
    create_external_worker_workflow,
    // Workflow data helpers
    create_external_worker_workflow_data,
    create_local_worker_workflow,
    create_local_worker_workflow_data,
    create_worker_removal_workflow,
    create_worker_removal_workflow_data,
    create_worker_update_workflow,
    create_worker_update_workflow_data,
    // Utility functions
    group_models_into_cards,
    infer_model_type_from_id,
    // Shared steps
    ActivateWorkersStep,
    // External registration steps
    CreateExternalWorkersStep,
    // Local registration steps
    CreateLocalWorkerStep,
    DetectConnectionModeStep,
    DiscoverDPInfoStep,
    DiscoverMetadataStep,
    DiscoverModelsStep,
    DpInfo,
    // Update steps
    FindWorkerToUpdateStep,
    // Removal steps
    FindWorkersToRemoveStep,
    ModelInfo,
    ModelsResponse,
    RegisterWorkersStep,
    RemoveFromPolicyRegistryStep,
    RemoveFromWorkerRegistryStep,
    UpdatePoliciesForWorkerStep,
    UpdatePoliciesStep,
    UpdateRemainingPoliciesStep,
    UpdateWorkerPropertiesStep,
    WorkerList,
    WorkerRemovalRequest,
};
// Typed workflow data structures
pub use workflow_data::{
    ExternalWorkerWorkflowData, LocalWorkerWorkflowData, McpWorkflowData, ProtocolUpdateRequest,
    TokenizerWorkflowData, WasmRegistrationWorkflowData, WasmRemovalWorkflowData,
    WorkerConfigRequest, WorkerList as WorkflowWorkerList, WorkerRegistrationData,
    WorkerRemovalWorkflowData, WorkerUpdateWorkflowData,
};
// Typed workflow engines
pub use workflow_engines::WorkflowEngines;

pub use crate::config::TokenizerCacheConfig;
