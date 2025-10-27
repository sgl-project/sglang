//! Workflow step implementations
//!
//! This module contains concrete step implementations for various workflows:
//! - Worker registration and activation
//! - Worker removal
//! - MCP server registration
//! - Future: Tokenizer fetching, LoRA updates, etc.

pub mod mcp_registration;
pub mod worker_registration;
pub mod worker_removal;

pub use mcp_registration::{
    create_mcp_registration_workflow, ConnectMcpServerStep, DiscoverMcpInventoryStep,
    McpServerConfigRequest, RegisterMcpServerStep, ValidateRegistrationStep,
};
pub use worker_registration::{
    create_worker_registration_workflow, ActivateWorkerStep, CreateWorkerStep,
    DetectConnectionModeStep, DiscoverMetadataStep, RegisterWorkerStep, UpdatePoliciesStep,
};
pub use worker_removal::{
    create_worker_removal_workflow, FindWorkersToRemoveStep, RemoveFromPolicyRegistryStep,
    RemoveFromWorkerRegistryStep, UpdateRemainingPoliciesStep, WorkerRemovalRequest,
};
