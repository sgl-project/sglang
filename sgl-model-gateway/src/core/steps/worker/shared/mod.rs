//! Shared worker registration steps used by both local and external workflows.
//!
//! These steps are designed to work with any worker type and can be composed
//! into different workflows using the DAG-based workflow engine.

mod activate;
mod register;
mod update_policies;

use std::sync::Arc;

pub use activate::ActivateWorkersStep;
pub use register::RegisterWorkersStep;
pub use update_policies::UpdatePoliciesStep;

use crate::core::Worker;

/// Type alias for a collection of workers in workflow context.
/// Both local (single/DP-aware) and external (multi-model) workflows
/// use this unified type for consistency.
pub type WorkerList = Vec<Arc<dyn Worker>>;
