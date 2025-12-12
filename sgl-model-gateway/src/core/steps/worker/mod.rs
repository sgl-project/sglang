//! Worker management workflows.
//!
//! This module provides workflows for worker lifecycle management:
//! - **Registration**: Adding new workers (local SGLang/vLLM or external APIs)
//! - **Removal**: Removing workers from the system
//! - **Update**: (Future) Updating worker configurations
//!
//! # Architecture
//!
//! ```text
//! worker/
//! ├── shared/           # Steps shared across workflows
//! │   ├── register.rs   # RegisterWorkersStep
//! │   ├── update_policies.rs
//! │   └── activate.rs   # ActivateWorkersStep
//! ├── local/            # Local SGLang/vLLM worker management
//! │   ├── detect_connection.rs
//! │   ├── discover_metadata.rs
//! │   ├── discover_dp.rs
//! │   ├── create_worker.rs
//! │   └── removal.rs    # Worker removal steps
//! └── external/         # External API worker registration
//!     ├── discover_models.rs
//!     └── create_workers.rs
//! ```
//!
//! # Workflows
//!
//! ## Local Worker Registration
//!
//! For SGLang/vLLM workers with parallel metadata and DP discovery:
//!
//! ```text
//!           detect_connection_mode
//!                    │
//!       ┌────────────┴────────────┐
//!       │                         │
//!  discover_metadata       discover_dp_info
//!       │                         │
//!       └────────────┬────────────┘
//!                    │
//!              create_worker
//!                    │
//!             register_workers
//!                    │
//!       ┌────────────┴────────────┐
//!       │                         │
//!  update_policies         activate_workers
//! ```
//!
//! ## External Worker Registration
//!
//! For OpenAI-compatible API endpoints:
//!
//! ```text
//!              discover_models
//!                    │
//!              create_workers
//!                    │
//!             register_workers
//!                    │
//!       ┌────────────┴────────────┐
//!       │                         │
//!  update_policies         activate_workers
//! ```
//!
//! ## Worker Removal
//!
//! ```text
//!     find_workers_to_remove
//!              │
//!     remove_from_policy_registry
//!              │
//!     remove_from_worker_registry
//!              │
//!     update_remaining_policies
//! ```

pub mod external;
pub mod local;
pub mod shared;

// Re-export shared steps
#[deprecated(since = "0.2.0", note = "Use create_external_worker_workflow instead")]
pub use external::create_external_worker_workflow as create_external_worker_registration_workflow;
// Re-export external steps and workflow builder
pub use external::{
    create_external_worker_workflow, group_models_into_cards, infer_model_type_from_id,
    CreateExternalWorkersStep, DiscoverModelsStep, ModelInfo, ModelsResponse,
};
// Backward compatibility aliases for registration workflows
#[deprecated(since = "0.2.0", note = "Use create_local_worker_workflow instead")]
pub use local::create_local_worker_workflow as create_worker_registration_workflow;
// Re-export local steps and workflow builders (registration + removal)
pub use local::{
    // Registration workflow
    create_local_worker_workflow,
    // Removal workflow
    create_worker_removal_workflow,
    CreateLocalWorkerStep,
    DetectConnectionModeStep,
    DiscoverDPInfoStep,
    DiscoverMetadataStep,
    DpInfo,
    FindWorkersToRemoveStep,
    RemoveFromPolicyRegistryStep,
    RemoveFromWorkerRegistryStep,
    UpdateRemainingPoliciesStep,
    WorkerRemovalRequest,
};
pub use shared::{ActivateWorkersStep, RegisterWorkersStep, UpdatePoliciesStep, WorkerList};
