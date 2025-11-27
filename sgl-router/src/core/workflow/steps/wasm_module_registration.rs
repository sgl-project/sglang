//! WASM Module Registration Workflow Steps
//!
//! Each step is atomic and performs a single operation in the WASM module registration process.
//!
//! Workflow order:
//! 1. ValidateDescriptor - Validate module descriptor (name, file_path, file existence)
//! 2. CalculateHash - Calculate SHA256 hash of the module file
//! 3. CheckDuplicate - Check for duplicate SHA256 hash
//! 4. LoadWasmBytes - Load WASM bytes into memory
//! 5. ValidateWasmComponent - Validate WASM component format
//! 6. RegisterModule - Register module in WasmModuleManager

use std::{sync::Arc, time::Duration};

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use tracing::{debug, info};
use uuid::Uuid;
use wasmtime::{component::Component, Config, Engine};

use crate::{
    app_context::AppContext,
    core::workflow::*,
    wasm::module::{WasmModule, WasmModuleDescriptor, WasmModuleMeta},
};

/// WASM module registration request
#[derive(Debug, Clone)]
pub struct WasmModuleConfigRequest {
    /// Module descriptor containing name, file_path, attach_points, etc.
    pub descriptor: WasmModuleDescriptor,
}

/// Step 1: Validate module descriptor
///
/// Validates that the module descriptor has all required fields:
/// - Module name is not empty
/// - File path is not empty
/// - File exists and is readable
/// - File size is not zero
pub struct ValidateDescriptorStep;

#[async_trait]
impl StepExecutor for ValidateDescriptorStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config_request: Arc<WasmModuleConfigRequest> = context
            .get("wasm_module_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("wasm_module_config".to_string()))?;

        let descriptor = &config_request.descriptor;

        debug!("Validating WASM module descriptor: {}", descriptor.name);

        // Validate name
        if descriptor.name.is_empty() {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("validate_descriptor"),
                message: "Module name cannot be empty".to_string(),
            });
        }

        // Validate file path
        if descriptor.file_path.is_empty() {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("validate_descriptor"),
                message: "Module file path cannot be empty".to_string(),
            });
        }

        // Check if file exists and get size
        let metadata = tokio::fs::metadata(&descriptor.file_path)
            .await
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("validate_descriptor"),
                message: format!("Failed to access file {}: {}", descriptor.file_path, e),
            })?;

        if metadata.len() == 0 {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("validate_descriptor"),
                message: "Module file size cannot be 0".to_string(),
            });
        }

        // Store file size in context for later steps
        context.set("file_size_bytes", metadata.len());

        info!(
            "Descriptor validated successfully for module: {}",
            descriptor.name
        );
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Validation errors are not retryable (invalid input)
    }
}

/// Step 2: Calculate SHA256 hash of the module file
///
/// Reads the file and calculates its SHA256 hash for deduplication.
/// This step is I/O intensive and may take time for large files.
pub struct CalculateHashStep;

#[async_trait]
impl StepExecutor for CalculateHashStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config_request: Arc<WasmModuleConfigRequest> = context
            .get("wasm_module_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("wasm_module_config".to_string()))?;

        let file_path = &config_request.descriptor.file_path;

        debug!("Calculating SHA256 hash for: {}", file_path);

        // Read file in chunks to handle large files efficiently
        let mut file =
            tokio::fs::File::open(file_path)
                .await
                .map_err(|e| WorkflowError::StepFailed {
                    step_id: StepId::new("calculate_hash"),
                    message: format!("Failed to open file {}: {}", file_path, e),
                })?;

        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024]; // 1MB buffer

        loop {
            use tokio::io::AsyncReadExt;
            let bytes_read =
                file.read(&mut buffer)
                    .await
                    .map_err(|e| WorkflowError::StepFailed {
                        step_id: StepId::new("calculate_hash"),
                        message: format!("Failed to read file {}: {}", file_path, e),
                    })?;

            if bytes_read == 0 {
                break;
            }

            hasher.update(&buffer[..bytes_read]);
        }

        let hash: [u8; 32] = hasher.finalize().into();

        // Store hash in context
        context.set("sha256_hash", hash);

        info!("SHA256 hash calculated for: {}", file_path);
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
impl StepExecutor for CheckDuplicateStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config_request: Arc<WasmModuleConfigRequest> = context
            .get("wasm_module_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("wasm_module_config".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let sha256_hash: Arc<[u8; 32]> = context
            .get("sha256_hash")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("sha256_hash".to_string()))?;

        debug!(
            "Checking for duplicate SHA256 hash for module: {}",
            config_request.descriptor.name
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
            .check_duplicate_sha256_hash(sha256_hash.as_ref())
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("check_duplicate"),
                message: format!("Duplicate SHA256 hash detected: {}", e),
            })?;

        info!(
            "No duplicate found for module: {}",
            config_request.descriptor.name
        );
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Duplicate check failures are not retryable
    }
}

/// Step 4: Load WASM bytes into memory
///
/// Reads the entire WASM file into memory for faster execution.
/// This is an I/O operation that may take time for large files.
pub struct LoadWasmBytesStep;

#[async_trait]
impl StepExecutor for LoadWasmBytesStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config_request: Arc<WasmModuleConfigRequest> = context
            .get("wasm_module_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("wasm_module_config".to_string()))?;

        let file_path = &config_request.descriptor.file_path;

        debug!("Loading WASM bytes from: {}", file_path);

        let wasm_bytes =
            tokio::fs::read(file_path)
                .await
                .map_err(|e| WorkflowError::StepFailed {
                    step_id: StepId::new("load_wasm_bytes"),
                    message: format!("Failed to read WASM file {}: {}", file_path, e),
                })?;

        // Store WASM bytes in context
        context.set("wasm_bytes", wasm_bytes);

        info!("WASM bytes loaded from: {}", file_path);
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        true // File read errors are retryable
    }
}

/// Step 5: Validate WASM component format
///
/// Validates that the loaded WASM bytes represent a valid component.
/// This catches format errors early during registration rather than during execution.
pub struct ValidateWasmComponentStep;

#[async_trait]
impl StepExecutor for ValidateWasmComponentStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config_request: Arc<WasmModuleConfigRequest> = context
            .get("wasm_module_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("wasm_module_config".to_string()))?;
        let wasm_bytes: Arc<Vec<u8>> = context
            .get("wasm_bytes")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("wasm_bytes".to_string()))?;

        debug!(
            "Validating WASM component format for module: {}",
            config_request.descriptor.name
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
        Component::new(&engine, wasm_bytes.as_ref())
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
            config_request.descriptor.name
        );
        Ok(StepResult::Success)
    }

    fn is_retryable(&self, _error: &WorkflowError) -> bool {
        false // Validation errors are not retryable (invalid format)
    }
}

/// Step 6: Register module in WasmModuleManager
///
/// Creates the WasmModule object and registers it in the manager's module map.
/// This is the final step that makes the module available for execution.
pub struct RegisterModuleStep;

#[async_trait]
impl StepExecutor for RegisterModuleStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config_request: Arc<WasmModuleConfigRequest> = context
            .get("wasm_module_config")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("wasm_module_config".to_string()))?;
        let app_context: Arc<AppContext> = context
            .get("app_context")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?;
        let sha256_hash: Arc<[u8; 32]> = context
            .get("sha256_hash")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("sha256_hash".to_string()))?;
        let file_size_bytes: Arc<u64> = context
            .get("file_size_bytes")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("file_size_bytes".to_string()))?;
        let wasm_bytes: Arc<Vec<u8>> = context
            .get("wasm_bytes")
            .ok_or_else(|| WorkflowError::ContextValueNotFound("wasm_bytes".to_string()))?;

        debug!(
            "Registering WASM module in manager: {}",
            config_request.descriptor.name
        );

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
                name: config_request.descriptor.name.clone(),
                file_path: config_request.descriptor.file_path.clone(),
                sha256_hash: *sha256_hash.as_ref(),
                size_bytes: *file_size_bytes.as_ref(),
                created_at: now,
                last_accessed_at: now,
                access_count: 0,
                attach_points: config_request.descriptor.attach_points.clone(),
                wasm_bytes: wasm_bytes.as_ref().clone(),
            },
        };

        // Register module in manager
        wasm_manager
            .register_module_internal(module)
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("register_module"),
                message: format!("Failed to register module: {}", e),
            })?;

        // Store module UUID in context for return value
        context.set("module_uuid", module_uuid);

        info!(
            "WASM module registered successfully: {} (UUID: {})",
            config_request.descriptor.name, module_uuid
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
/// - Validates the module descriptor
/// - Calculates SHA256 hash for deduplication
/// - Checks for duplicates
/// - Loads WASM bytes into memory
/// - Validates WASM component format
/// - Registers the module in the manager
///
/// Workflow configuration:
/// - ValidateDescriptor: No retry, 5s timeout (fast validation)
/// - CalculateHash: 3 retries, 60s timeout (I/O intensive, may need retry)
/// - CheckDuplicate: No retry, 5s timeout (fast check)
/// - LoadWasmBytes: 3 retries, 60s timeout (I/O intensive)
/// - ValidateWasmComponent: No retry, 30s timeout (CPU intensive validation)
/// - RegisterModule: No retry, 5s timeout (fast registration)
pub fn create_wasm_module_registration_workflow() -> WorkflowDefinition {
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
                "calculate_hash",
                "Calculate SHA256 Hash",
                Arc::new(CalculateHashStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(60))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "check_duplicate",
                "Check Duplicate Hash",
                Arc::new(CheckDuplicateStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "load_wasm_bytes",
                "Load WASM Bytes",
                Arc::new(LoadWasmBytesStep),
            )
            .with_retry(RetryPolicy {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            })
            .with_timeout(Duration::from_secs(60))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "validate_wasm_component",
                "Validate WASM Component",
                Arc::new(ValidateWasmComponentStep),
            )
            .with_timeout(Duration::from_secs(30))
            .with_failure_action(FailureAction::FailWorkflow),
        )
        .add_step(
            StepDefinition::new(
                "register_module",
                "Register Module",
                Arc::new(RegisterModuleStep),
            )
            .with_timeout(Duration::from_secs(5))
            .with_failure_action(FailureAction::FailWorkflow),
        )
}
