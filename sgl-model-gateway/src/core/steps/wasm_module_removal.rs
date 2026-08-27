use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use tracing::{debug, info};
use uuid::Uuid;

use super::workflow_data::WasmRemovalWorkflowData;
use crate::{
    app_context::AppContext,
    workflow::{
        FailureAction, StepDefinition, StepExecutor, StepId, StepResult, WorkflowContext,
        WorkflowDefinition, WorkflowError, WorkflowResult,
    },
};

/// WASM module removal request
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
impl StepExecutor<WasmRemovalWorkflowData> for FindModuleToRemoveStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WasmRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let removal_request = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

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

        // Clone uuid for logging before mutable borrow
        let module_uuid = removal_request.module_uuid;

        // Store the module ID in typed data
        context.data.module_id = Some(module_uuid.to_string());

        info!("Module found for removal: {}", module_uuid);
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
impl StepExecutor<WasmRemovalWorkflowData> for RemoveModuleStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WasmRemovalWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let removal_request = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;

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
pub fn create_wasm_module_removal_workflow() -> WorkflowDefinition<WasmRemovalWorkflowData> {
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
                .with_failure_action(FailureAction::FailWorkflow)
                .depends_on(&["find_module_to_remove"]),
        )
}

/// Helper to create initial workflow data for WASM module removal
pub fn create_wasm_removal_workflow_data(
    config: WasmModuleRemovalRequest,
    app_context: Arc<AppContext>,
) -> WasmRemovalWorkflowData {
    WasmRemovalWorkflowData {
        config,
        module_id: None,
        app_context: Some(app_context),
    }
}
