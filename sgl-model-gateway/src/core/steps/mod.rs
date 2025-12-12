//! Workflow step implementations
//!
//! This module contains concrete step implementations for various workflows:
//! - Local worker registration (SGLang, vLLM - gRPC or HTTP modes)
//! - External worker registration (OpenAI, xAI, Anthropic, etc. - HTTPS only)
//! - Worker removal
//! - MCP server registration
//! - WASM module registration and removal
//! - Future: Tokenizer fetching, LoRA updates, etc.

pub mod external_worker_registration;
pub mod mcp_registration;
pub mod wasm_module_registration;
pub mod wasm_module_removal;
pub mod worker_registration;
pub mod worker_removal;

pub use external_worker_registration::{
    create_external_worker_registration_workflow, group_models_into_cards,
    infer_model_type_from_id, ActivateExternalWorkersStep, CreateExternalWorkersStep,
    DiscoverModelsStep, ModelInfo, ModelsResponse, RegisterExternalWorkersStep,
    UpdateExternalPoliciesStep,
};
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
pub use worker_registration::{
    create_worker_registration_workflow, ActivateWorkerStep, CreateWorkerStep,
    DetectConnectionModeStep, DiscoverDPInfoStep, DiscoverMetadataStep, RegisterWorkerStep,
    UpdatePoliciesStep,
};
pub use worker_removal::{
    create_worker_removal_workflow, FindWorkersToRemoveStep, RemoveFromPolicyRegistryStep,
    RemoveFromWorkerRegistryStep, UpdateRemainingPoliciesStep, WorkerRemovalRequest,
};
