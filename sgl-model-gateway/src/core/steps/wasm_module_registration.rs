use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};
use uuid::Uuid;
use wasmtime::{component::Component, Config, Engine};
use wfaas::{
    BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, StepExecutor, StepId, StepResult,
    WorkflowContext, WorkflowDefinition, WorkflowError, WorkflowResult,
};

use super::workflow_data::WasmRegistrationWorkflowData;
use crate::{
    app_context::AppContext,
    wasm::module::{WasmModule, WasmModuleDescriptor, WasmModuleMeta},
};

/// WASM module registration request
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WasmModuleConfigRequest {
    /// Module descriptor containing name, file_path, attach_points, etc.
    pub descriptor: WasmModuleDescriptor,
}

/// Step 1: Validate module descriptor
///
/// Checks the descriptor's own fields, then resolves the requested file path
/// against the configured module roots. Confinement to those roots is the
/// security boundary; see [`crate::wasm::module_roots`] for why an allow-list
/// replaced the deny-list of sensitive directories that used to live here.
pub struct ValidateDescriptorStep;

#[async_trait]
impl StepExecutor<WasmRegistrationWorkflowData> for ValidateDescriptorStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WasmRegistrationWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let step_id = || StepId::new("validate_descriptor");

        // Take owned copies so the shared borrow of `context.data` ends before
        // the resolved path is written back to it.
        let (module_name, requested_path) = {
            let descriptor = &context.data.config.descriptor;
            (descriptor.name.clone(), descriptor.file_path.clone())
        };

        debug!("Validating WASM module descriptor: {}", module_name);

        if module_name.is_empty() {
            return Err(WorkflowError::StepFailed {
                step_id: step_id(),
                message: "Module name cannot be empty".to_string(),
            });
        }

        if requested_path.is_empty() {
            return Err(WorkflowError::StepFailed {
                step_id: step_id(),
                message: "Module file path cannot be empty".to_string(),
            });
        }

        let roots = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .wasm_module_roots
            .clone()
            .ok_or_else(|| WorkflowError::StepFailed {
                step_id: step_id(),
                message: "WASM module roots are not configured".to_string(),
            })?;

        // One call replaces the former absolute-path check, `..`/`.` scan,
        // deny-list, symlink re-check and existence probe. The error deliberately
        // says nothing about where the path led; that detail is logged instead.
        let canonical_path = roots.resolve(&requested_path).await.map_err(|error| {
            // `?error` (Debug) so the log keeps the precise cause; the message
            // handed back to the caller uses Display, which by construction
            // cannot say whether the path existed. See `ModulePathError`.
            warn!(
                requested = %requested_path,
                ?error,
                "Rejected WASM module path"
            );
            WorkflowError::StepFailed {
                step_id: step_id(),
                message: error.to_string(),
            }
        })?;

        context.data.canonical_path = Some(canonical_path.to_string_lossy().into_owned());

        info!(
            "Descriptor validated successfully for module: {}",
            module_name
        );
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Validation errors are not retryable (invalid input)
    }
}

/// Step 2: Read the module and derive its size and hash
///
/// Hashing and loading were once separate steps, which meant the file was read
/// twice — and the hash could, in principle, describe a different revision of
/// the file than the bytes that were ultimately registered. Reading once and
/// deriving both from the same buffer removes the redundant I/O and makes that
/// disagreement impossible.
///
/// Streaming the hash in chunks bought nothing either: the whole file is held
/// in memory afterwards regardless, since the bytes are what get registered.
pub struct AcquireModuleStep;

#[async_trait]
impl StepExecutor<WasmRegistrationWorkflowData> for AcquireModuleStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WasmRegistrationWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let step_id = || StepId::new("acquire_module");

        let path = context
            .data
            .canonical_path
            .clone()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("canonical_path".to_string()))?;

        debug!("Reading WASM module from: {}", path);

        let wasm_bytes = tokio::fs::read(&path)
            .await
            .map_err(|e| WorkflowError::StepFailed {
                step_id: step_id(),
                message: format!("Failed to read WASM module: {}", e),
            })?;

        if wasm_bytes.is_empty() {
            return Err(WorkflowError::StepFailed {
                step_id: step_id(),
                message: "Module file size cannot be 0".to_string(),
            });
        }

        let hash: [u8; 32] = Sha256::digest(&wasm_bytes).into();

        context.data.file_size_bytes = Some(wasm_bytes.len() as u64);
        context.data.sha256_hash = Some(hash);
        context.data.wasm_bytes = Some(wasm_bytes);

        info!("WASM module read and hashed: {}", path);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // File I/O errors are retryable (network filesystem, etc.)
    }
}

/// Step 3: Check for duplicate SHA256 hash
///
/// Checks if a module with the same SHA256 hash already exists in the manager.
/// This prevents duplicate modules from being registered.
pub struct CheckDuplicateStep;

#[async_trait]
impl StepExecutor<WasmRegistrationWorkflowData> for CheckDuplicateStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WasmRegistrationWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let sha256_hash = context
            .data
            .sha256_hash
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("sha256_hash".to_string()))?;

        debug!(
            "Checking for duplicate SHA256 hash for module: {}",
            context.data.config.descriptor.name
        );

        // Get WASM module manager from app context
        let wasm_manager =
            app_context
                .wasm_manager
                .as_ref()
                .ok_or_else(|| WorkflowError::StepFailed {
                    step_id: StepId::new("check_duplicate"),
                    message: "WASM module manager not initialized".to_string(),
                })?;

        // Check for duplicate hash using manager's internal method
        wasm_manager
            .check_duplicate_sha256_hash(sha256_hash)
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("check_duplicate"),
                message: format!("Duplicate SHA256 hash detected: {}", e),
            })?;

        info!(
            "No duplicate found for module: {}",
            context.data.config.descriptor.name
        );
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Duplicate check failures are not retryable
    }
}

/// Step 4: Validate WASM component format
///
/// Validates that the loaded WASM bytes represent a valid component.
/// This catches format errors early during registration rather than during execution.
pub struct ValidateWasmComponentStep;

#[async_trait]
impl StepExecutor<WasmRegistrationWorkflowData> for ValidateWasmComponentStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WasmRegistrationWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let wasm_bytes = context
            .data
            .wasm_bytes
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("wasm_bytes".to_string()))?;

        debug!(
            "Validating WASM component format for module: {}",
            context.data.config.descriptor.name
        );

        // Create a temporary engine to validate the component
        let mut config = Config::new();
        config.async_support(true);
        config.wasm_component_model(true);

        let engine = Engine::new(&config).map_err(|e| WorkflowError::StepFailed {
            step_id: StepId::new("validate_wasm_component"),
            message: format!("Failed to create WASM engine: {}", e),
        })?;

        // Attempt to compile the component to validate it
        Component::new(&engine, wasm_bytes)
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("validate_wasm_component"),
                message: format!(
                    "Invalid WASM component: {}. \
                     Hint: The WASM file must be in component format. \
                     If you're using wit-bindgen, use 'wasm-tools component new' to wrap the WASM module into a component.",
                    e
                ),
            })?;

        info!(
            "WASM component validated successfully for module: {}",
            context.data.config.descriptor.name
        );
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Validation errors are not retryable (invalid format)
    }
}

/// Step 5: Register module in WasmModuleManager
///
/// Creates the WasmModule object and registers it in the manager's module map.
/// This is the final step that makes the module available for execution.
pub struct RegisterModuleStep;

#[async_trait]
impl StepExecutor<WasmRegistrationWorkflowData> for RegisterModuleStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WasmRegistrationWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let sha256_hash = context
            .data
            .sha256_hash
            .ok_or_else(|| WorkflowError::ContextValueNotFound("sha256_hash".to_string()))?;
        let file_size_bytes = context
            .data
            .file_size_bytes
            .ok_or_else(|| WorkflowError::ContextValueNotFound("file_size_bytes".to_string()))?;
        let wasm_bytes = context
            .data
            .wasm_bytes
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("wasm_bytes".to_string()))?
            .clone();

        // Record the path the bytes actually came from, not the caller's spelling
        // of it, so the module listing reflects what was loaded.
        let canonical_path = context
            .data
            .canonical_path
            .clone()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("canonical_path".to_string()))?;

        let descriptor = &context.data.config.descriptor;

        debug!("Registering WASM module in manager: {}", descriptor.name);

        // Get WASM module manager from app context
        let wasm_manager =
            app_context
                .wasm_manager
                .as_ref()
                .ok_or_else(|| WorkflowError::StepFailed {
                    step_id: StepId::new("register_module"),
                    message: "WASM module manager not initialized".to_string(),
                })?;

        // Create module metadata
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::from_nanos(0))
            .as_nanos() as u64;

        let module_uuid = Uuid::new_v4();

        let module = WasmModule {
            module_uuid,
            module_meta: WasmModuleMeta {
                name: descriptor.name.clone(),
                file_path: canonical_path,
                sha256_hash,
                size_bytes: file_size_bytes,
                created_at: now,
                last_accessed_at: now,
                access_count: 0,
                attach_points: descriptor.attach_points.clone(),
                wasm_bytes,
            },
        };

        // Clone name for logging before mutable borrow
        let module_name = descriptor.name.clone();

        // Register module in manager
        wasm_manager
            .register_module_internal(module)
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("register_module"),
                message: format!("Failed to register module: {}", e),
            })?;

        // Store module UUID in typed data
        context.data.module_uuid = Some(module_uuid);

        info!(
            "WASM module registered successfully: {} (UUID: {})",
            module_name, module_uuid
        );

        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Registration is a simple operation, not retryable
    }
}

/// Create WASM module registration workflow
///
/// This workflow handles the complete process of registering a WASM module:
/// - Validates the descriptor and confines the path to a configured module root
/// - Reads the module once, deriving its size and SHA256 hash from that read
/// - Checks for duplicates
/// - Validates WASM component format
/// - Registers the module in the manager
///
/// Workflow configuration:
/// - ValidateDescriptor: No retry, 5s timeout (rejects bad input; retrying cannot help)
/// - AcquireModule: 3 retries, 60s timeout (I/O intensive, may need retry)
/// - CheckDuplicate: No retry, 5s timeout (fast check)
/// - ValidateWasmComponent: No retry, 30s timeout (CPU intensive validation)
/// - RegisterModule: No retry, 5s timeout (fast registration)
pub fn create_wasm_module_registration_workflow() -> WorkflowDefinition<WasmRegistrationWorkflowData>
{
    WorkflowDefinition::new("wasm_module_registration", "WASM Module Registration")
        .add_step(
            StepDefinition::new(
                "validate_descriptor",
                "Validate Descriptor",
                Arc::new(ValidateDescriptorStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "acquire_module",
                "Read Module and Compute Hash",
                Arc::new(AcquireModuleStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(60))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["validate_descriptor"]),
        )
        .add_step(
            StepDefinition::new(
                "check_duplicate",
                "Check Duplicate Hash",
                Arc::new(CheckDuplicateStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["acquire_module"]),
        )
        .add_step(
            StepDefinition::new(
                "validate_wasm_component",
                "Validate WASM Component",
                Arc::new(ValidateWasmComponentStep),
            )
            .with_timeout(Duration::from_secs(30))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["check_duplicate"]),
        )
        .add_step(
            StepDefinition::new(
                "register_module",
                "Register Module",
                Arc::new(RegisterModuleStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["validate_wasm_component"]),
        )
}

/// Helper to create initial workflow data for WASM module registration
pub fn create_wasm_registration_workflow_data(
    config: WasmModuleConfigRequest,
    app_context: Arc<AppContext>,
) -> WasmRegistrationWorkflowData {
    WasmRegistrationWorkflowData {
        config,
        canonical_path: None,
        wasm_bytes: None,
        sha256_hash: None,
        file_size_bytes: None,
        module_uuid: None,
        app_context: Some(app_context),
    }
}
