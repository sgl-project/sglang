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
impl StepExecutor for ValidateDescriptorStep {
    async fn execute(&self, context: &mut WorkflowContext) -> WorkflowResult<StepResult> {
        let config_request: Arc<WasmModuleConfigRequest> =
            context.get_or_err("wasm_module_config")?;

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
        let config_request: Arc<WasmModuleConfigRequest> =
            context.get_or_err("wasm_module_config")?;

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
        let config_request: Arc<WasmModuleConfigRequest> =
            context.get_or_err("wasm_module_config")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;
        let sha256_hash: Arc<[u8; 32]> = context.get_or_err("sha256_hash")?;

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
        let config_request: Arc<WasmModuleConfigRequest> =
            context.get_or_err("wasm_module_config")?;

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
        let config_request: Arc<WasmModuleConfigRequest> =
            context.get_or_err("wasm_module_config")?;
        let wasm_bytes: Arc<Vec<u8>> = context.get_or_err("wasm_bytes")?;

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
        let config_request: Arc<WasmModuleConfigRequest> =
            context.get_or_err("wasm_module_config")?;
        let app_context: Arc<AppContext> = context.get_or_err("app_context")?;
        let sha256_hash: Arc<[u8; 32]> = context.get_or_err("sha256_hash")?;
        let file_size_bytes: Arc<u64> = context.get_or_err("file_size_bytes")?;
        let wasm_bytes: Arc<Vec<u8>> = context.get_or_err("wasm_bytes")?;

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
