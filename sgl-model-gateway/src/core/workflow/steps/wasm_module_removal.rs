//! WASM Module Removal Workflow Steps
//!
//! Each step is atomic and performs a single operation in the WASM module removal process.
//!
//! Workflow order:
//! 1. FindModuleToRemove - Find the module to remove by UUID
//! 2. RemoveModule - Remove module from WasmModuleManager

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use tracing::{debug, info};
use uuid::Uuid;

use crate::{app_context::AppContext, core::workflow::*};

/// WASM module removal request
#[derive(Debug, Clone)]
pub struct WasmModuleRemovalRequest {
    /// Module UUID to remove
    pub module_uuid: Uuid,
    /// Cached UUID string for worker_url() method
    pub(crate) uuid_string: String,
}

impl WasmModuleRemovalRequest {
    pub fn new(module_uuid: Uuid) -> Self {
        Self {
            module_uuid,
            uuid_string: module_uuid.to_string(),
        }
    }
}

/// Step 1: Find module to remove
///
/// Verifies that the module exists before attempting removal.
pub struct FindModuleToRemoveStep;

#[async_trait]
impl StepExecutor for FindModuleToRemoveStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let removal_request: Arc<WasmModuleRemovalRequest> =
            context.get_or_err("wasm_module_removal_request")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;

        debug!("Finding module to remove: {}", removal_request.module_uuid);

        // Get WASM module manager from app context
        let wasm_manager =
            app_context
                .wasm_manager
                .as_ref()
                .ok_or_else(|| WorkflowError::StepFailed {
                    step_id: StepId::new("find_module_to_remove"),
                    message: "WASM module manager not initialized".to_string(),
                })?;

        // Check if module exists
        let module = wasm_manager
            .get_module(removal_request.module_uuid)
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("find_module_to_remove"),
                message: format!("Failed to get module: {}", e),
            })?;

        if module.is_none() {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("find_module_to_remove"),
                message: format!("Module with UUID {} not found", removal_request.module_uuid),
            });
        }

        info!("Module found for removal: {}", removal_request.module_uuid);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Module not found is not retryable
    }
}

/// Step 2: Remove module from WasmModuleManager
///
/// Removes the module from the manager's module map.
pub struct RemoveModuleStep;

#[async_trait]
impl StepExecutor for RemoveModuleStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let removal_request: Arc<WasmModuleRemovalRequest> =
            context.get_or_err("wasm_module_removal_request")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;

        debug!("Removing WASM module: {}", removal_request.module_uuid);

        // Get WASM module manager from app context
        let wasm_manager =
            app_context
                .wasm_manager
                .as_ref()
                .ok_or_else(|| WorkflowError::StepFailed {
                    step_id: StepId::new("remove_module"),
                    message: "WASM module manager not initialized".to_string(),
                })?;

        // Remove module from manager
        wasm_manager
            .remove_module_internal(removal_request.module_uuid)
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("remove_module"),
                message: format!("Failed to remove module: {}", e),
            })?;

        info!(
            "WASM module removed successfully: {}",
            removal_request.module_uuid
        );
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Removal is not retryable
    }
}

/// Create WASM module removal workflow
///
/// This workflow handles the process of removing a WASM module:
/// - Finds the module to remove
/// - Removes it from the manager
///
/// Workflow configuration:
/// - FindModuleToRemove: No retry, 5s timeout (fast lookup)
/// - RemoveModule: No retry, 5s timeout (fast removal)
pub fn create_wasm_module_removal_workflow() -> WorkflowDefinition {
    WorkflowDefinition::new("wasm_module_removal", "WASM Module Removal")
        .add_step(
            StepDefinition::new(
                "find_module_to_remove",
                "Find Module to Remove",
                Arc::new(FindModuleToRemoveStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new("remove_module", "Remove Module", Arc::new(RemoveModuleStep))
                .with_timeout(Duration::from_secs(5))
                .with_failure_action(FailureAction::FailWorkflow),
        )
}
