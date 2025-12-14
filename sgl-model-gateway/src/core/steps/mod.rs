//! Workflow step implementations
//!
//! This module contains concrete step implementations for various workflows:
//! - Worker management (registration, removal, updates)
//! - MCP server registration
//! - WASM module registration and removal
//! - Future: Tokenizer fetching, LoRA updates, etc.

pub mod mcp_registration;
pub mod wasm_module_registration;
pub mod wasm_module_removal;
pub mod worker;

// Worker management (registration, removal)
#[allow(deprecated)]
pub use worker::create_external_worker_registration_workflow;
// Backward compatibility aliases
#[allow(deprecated)]
pub use worker::create_worker_registration_workflow;
pub use worker::{
    // Workflow builders
    create_external_worker_workflow,
    create_local_worker_workflow,
    create_worker_removal_workflow,
    create_worker_update_workflow,
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

// Legacy type aliases for backward compatibility
pub type ActivateWorkerStep = ActivateWorkersStep;
pub type RegisterWorkerStep = RegisterWorkersStep;
pub type CreateWorkerStep = CreateLocalWorkerStep;
pub type ActivateExternalWorkersStep = ActivateWorkersStep;
pub type RegisterExternalWorkersStep = RegisterWorkersStep;
pub type UpdateExternalPoliciesStep = UpdatePoliciesStep;

pub use mcp_registration::{
    create_mcp_registration_workflow, ConnectMcpServerStep, DiscoverMcpInventoryStep,
    McpServerConfigRequest, RegisterMcpServerStep, ValidateRegistrationStep,
};
pub use wasm_module_registration::{
    create_wasm_module_registration_workflow, CalculateHashStep, CheckDuplicateStep,
    LoadWasmBytesStep, RegisterModuleStep, ValidateDescriptorStep, ValidateWasmComponentStep,
    WasmModuleConfigRequest,
};
pub use wasm_module_removal::{
    create_wasm_module_removal_workflow, FindModuleToRemoveStep, RemoveModuleStep,
    WasmModuleRemovalRequest,
};
