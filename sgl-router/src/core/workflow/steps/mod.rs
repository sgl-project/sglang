//! Workflow step implementations
//!
//! This module contains concrete step implementations for various workflows:
//! - Worker registration and activation
//! - Worker removal
//! - Future: Tokenizer fetching, LoRA updates, etc.

pub mod worker_registration;
pub mod worker_removal;

pub use worker_registration::{
    create_worker_registration_workflow, ActivateWorkerStep, CreateWorkerStep,
    DetectConnectionModeStep, DiscoverMetadataStep, RegisterWorkerStep, UpdatePoliciesStep,
};
pub use worker_removal::{
    create_worker_removal_workflow, FindWorkersToRemoveStep, RemoveFromPolicyRegistryStep,
    RemoveFromWorkerRegistryStep, UpdateRemainingPoliciesStep, WorkerRemovalRequest,
};
