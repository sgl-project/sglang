use std::{
    path::{Component as PathComponent, Path},
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};
use uuid::Uuid;
use wasmtime::{component::Component, Config, Engine};

use super::workflow_data::WasmRegistrationWorkflowData;
use crate::{
    app_context::AppContext,
    wasm::module::{WasmModule, WasmModuleDescriptor, WasmModuleMeta},
    workflow::{
        BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, StepExecutor, StepId,
        StepResult, WorkflowContext, WorkflowDefinition, WorkflowError, WorkflowResult,
    },
};

/// WASM module registration request
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WasmModuleConfigRequest {
    /// Module descriptor containing name, file_path, attach_points, etc.
    pub descriptor: WasmModuleDescriptor,
}

/// Sensitive system directories that WASM modules cannot be loaded from.
/// These are blocked to prevent information disclosure attacks.
const BLOCKED_PATH_PREFIXES: &[&str] = &[
    "/etc/",
    "/proc/",
    "/sys/",
    "/dev/",
    "/boot/",
    "/root/",
    "/var/log/",
    "/var/run/",
];

/// Check if a path starts with any blocked prefix.
/// Returns the matched prefix if found, None otherwise.
fn find_blocked_prefix(path: &str) -> Option<&'static str> {
    BLOCKED_PATH_PREFIXES
        .iter()
        .find(|&&prefix| path.starts_with(prefix))
        .copied()
}

/// Check if a path has a .wasm extension (case-insensitive).
fn has_wasm_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wasm"))
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
impl StepExecutor<WasmRegistrationWorkflowData> for ValidateDescriptorStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WasmRegistrationWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let descriptor = &context.data.config.descriptor;

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

        // Security: Validate path to prevent path traversal attacks
        let path = Path::new(&descriptor.file_path);

        // Must be an absolute path
        if !path.is_absolute() {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("validate_descriptor"),
                message: format!(
                    "Module file path must be absolute, got: {}",
                    descriptor.file_path
                ),
            });
        }

        // Check for path traversal components (.. or symbolic links that could escape)
        for component in path.components() {
            match component {
                PathComponent::ParentDir => {
                    warn!(
                        "Path traversal attempt detected in WASM module path: {}",
                        descriptor.file_path
                    );
                    return Err(WorkflowError::StepFailed {
                        step_id: StepId::new("validate_descriptor"),
                        message: "Path traversal (..) not allowed in module file path".to_string(),
                    });
                }
                PathComponent::CurDir => {
                    return Err(WorkflowError::StepFailed {
                        step_id: StepId::new("validate_descriptor"),
                        message: "Current directory (.) not allowed in module file path"
                            .to_string(),
                    });
                }
                _ => {}
            }
        }

        // Require .wasm extension to prevent loading arbitrary files
        if !has_wasm_extension(path) {
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("validate_descriptor"),
                message: "Module file must have .wasm extension".to_string(),
            });
        }

        // Block access to sensitive system directories
        if let Some(prefix) = find_blocked_prefix(&descriptor.file_path) {
            warn!(
                "Attempt to access blocked directory in WASM module path: {}",
                descriptor.file_path
            );
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("validate_descriptor"),
                message: format!("Access to {} directory is not allowed", prefix),
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

        // Canonicalize the path to resolve symlinks and verify final location is safe
        let canonical_path = tokio::fs::canonicalize(&descriptor.file_path)
            .await
            .map_err(|e| WorkflowError::StepFailed {
                step_id: StepId::new("validate_descriptor"),
                message: format!(
                    "Failed to canonicalize path {}: {}",
                    descriptor.file_path, e
                ),
            })?;

        // Re-check blocked directories after symlink resolution
        let canonical_str = canonical_path.to_string_lossy();
        if let Some(prefix) = find_blocked_prefix(&canonical_str) {
            warn!(
                "Symlink resolved to blocked directory: {} -> {}",
                descriptor.file_path, canonical_str
            );
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("validate_descriptor"),
                message: format!(
                    "Path resolves to blocked directory {} (via symlink)",
                    prefix
                ),
            });
        }

        // Ensure canonicalized path still has .wasm extension (symlink target check)
        if !has_wasm_extension(&canonical_path) {
            warn!(
                "Symlink target is not a .wasm file: {} -> {}",
                descriptor.file_path, canonical_str
            );
            return Err(WorkflowError::StepFailed {
                step_id: StepId::new("validate_descriptor"),
                message: "Symlink target must be a .wasm file".to_string(),
            });
        }

        // Clone name for logging before mutable borrow
        let module_name = descriptor.name.clone();

        // Store file size in typed data
        context.data.file_size_bytes = Some(metadata.len());

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

/// Step 2: Calculate SHA256 hash of the module file
///
/// Reads the file and calculates its SHA256 hash for deduplication.
/// This step is I/O intensive and may take time for large files.
pub struct CalculateHashStep;

#[async_trait]
impl StepExecutor<WasmRegistrationWorkflowData> for CalculateHashStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WasmRegistrationWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let file_path = &context.data.config.descriptor.file_path;

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

        // Clone path for logging before mutable borrow
        let path_for_log = file_path.clone();

        // Store hash in typed data
        context.data.sha256_hash = Some(hash);

        info!("SHA256 hash calculated for: {}", path_for_log);
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

/// Step 4: Load WASM bytes into memory
///
/// Reads the entire WASM file into memory for faster execution.
/// This is an I/O operation that may take time for large files.
pub struct LoadWasmBytesStep;

#[async_trait]
impl StepExecutor<WasmRegistrationWorkflowData> for LoadWasmBytesStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<WasmRegistrationWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let file_path = &context.data.config.descriptor.file_path;

        debug!("Loading WASM bytes from: {}", file_path);

        // Clone path for logging before mutable borrow
        let path_for_log = file_path.clone();

        let wasm_bytes =
            tokio::fs::read(file_path)
                .await
                .map_err(|e| WorkflowError::StepFailed {
                    step_id: StepId::new("load_wasm_bytes"),
                    message: format!("Failed to read WASM file {}: {}", file_path, e),
                })?;

        // Store WASM bytes in typed data
        context.data.wasm_bytes = Some(wasm_bytes);

        info!("WASM bytes loaded from: {}", path_for_log);
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

/// Step 6: Register module in WasmModuleManager
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
                file_path: descriptor.file_path.clone(),
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
                "calculate_hash",
                "Calculate SHA256 Hash",
                Arc::new(CalculateHashStep),
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
            .depends_on(&["calculate_hash"]),
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
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["check_duplicate"]),
        )
        .add_step(
            StepDefinition::new(
                "validate_wasm_component",
                "Validate WASM Component",
                Arc::new(ValidateWasmComponentStep),
            )
            .with_timeout(Duration::from_secs(30))
            .with_failure_action(FailureAction::FailWorkflow)
            .depends_on(&["load_wasm_bytes"]),
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
        wasm_bytes: None,
        sha256_hash: None,
        file_size_bytes: None,
        module_uuid: None,
        app_context: Some(app_context),
    }
}
