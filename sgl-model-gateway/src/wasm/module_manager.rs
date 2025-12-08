//! WASM Module Manager

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
};

use uuid::Uuid;

use crate::wasm::{
    config::WasmRuntimeConfig,
    errors::{Result, WasmError, WasmManagerError, WasmModuleError},
    module::{WasmModule, WasmModuleAttachPoint},
    runtime::WasmRuntime,
    types::{WasmComponentInput, WasmComponentOutput},
};

pub struct WasmModuleManager {
    modules: Arc<RwLock<HashMap<Uuid, WasmModule>>>,
    runtime: Arc<WasmRuntime>,
    // Metrics
    total_executions: AtomicU64,
    successful_executions: AtomicU64,
    failed_executions: AtomicU64,
    total_execution_time_ms: AtomicU64,
    max_execution_time_ms: AtomicU64,
}

impl WasmModuleManager {
    pub fn new(config: WasmRuntimeConfig) -> Result<Self> {
        let runtime = Arc::new(WasmRuntime::new(config)?);
        Ok(Self {
            modules: Arc::new(RwLock::new(HashMap::new())),
            runtime,
            total_executions: AtomicU64::new(0),
            successful_executions: AtomicU64::new(0),
            failed_executions: AtomicU64::new(0),
            total_execution_time_ms: AtomicU64::new(0),
            max_execution_time_ms: AtomicU64::new(0),
        })
    }

    pub fn with_default_config() -> Result<Self> {
        Self::new(WasmRuntimeConfig::default())
    }

    /// Register a module (for workflow steps)
    pub(crate) fn register_module_internal(&self, module: WasmModule) -> Result<()> {
        let mut modules = self
            .modules
            .write()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        modules.insert(module.module_uuid, module);
        Ok(())
    }

    /// Remove a module (for workflow steps)
    pub(crate) fn remove_module_internal(&self, module_uuid: Uuid) -> Result<()> {
        let mut modules = self
            .modules
            .write()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        if !modules.contains_key(&module_uuid) {
            return Err(WasmManagerError::ModuleNotFound(module_uuid).into());
        }
        modules.remove(&module_uuid);
        Ok(())
    }

    pub(crate) fn check_duplicate_sha256_hash(&self, sha256_hash: &[u8; 32]) -> Result<()> {
        let modules = self
            .modules
            .read()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        if modules
            .values()
            .any(|module: &WasmModule| module.module_meta.sha256_hash == *sha256_hash)
        {
            return Err(WasmModuleError::DuplicateSha256((*sha256_hash).into()).into());
        }
        Ok(())
    }

    pub fn get_all_modules(&self) -> Result<Vec<WasmModule>> {
        let modules = self
            .modules
            .read()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        Ok(modules.values().cloned().collect())
    }

    pub fn get_module(&self, module_uuid: Uuid) -> Result<Option<WasmModule>> {
        let modules = self
            .modules
            .read()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        Ok(modules.get(&module_uuid).cloned())
    }

    pub fn get_modules(&self) -> Result<Vec<WasmModule>> {
        let modules = self
            .modules
            .read()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        Ok(modules.values().cloned().collect())
    }

    /// get modules by attach point
    pub fn get_modules_by_attach_point(
        &self,
        attach_point: WasmModuleAttachPoint,
    ) -> Result<Vec<WasmModule>> {
        let modules = self
            .modules
            .read()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        Ok(modules
            .values()
            .filter(|module| module.module_meta.attach_points.contains(&attach_point))
            .cloned()
            .collect())
    }

    pub fn get_runtime(&self) -> &Arc<WasmRuntime> {
        &self.runtime
    }

    /// Get the configured maximum body size for HTTP request/response processing
    pub fn get_max_body_size(&self) -> usize {
        self.runtime.get_config().max_body_size
    }

    /// Execute WASM module using WebAssembly component model based on attach_point
    pub async fn execute_module_interface(
        &self,
        module_uuid: Uuid,
        attach_point: WasmModuleAttachPoint,
        input: WasmComponentInput,
    ) -> Result<WasmComponentOutput> {
        let start_time = std::time::Instant::now();

        // First, get the WASM bytes with a read lock (faster)
        let wasm_bytes = {
            let modules = self
                .modules
                .read()
                .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
            let module = modules
                .get(&module_uuid)
                .ok_or_else(|| WasmError::from(WasmManagerError::ModuleNotFound(module_uuid)))?;

            // Clone the pre-loaded WASM bytes (already in memory, no file I/O)
            module.module_meta.wasm_bytes.clone()
        };

        {
            let mut modules = self
                .modules
                .write()
                .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
            if let Some(module) = modules.get_mut(&module_uuid) {
                // SystemTime::duration_since only fails if the system time is before UNIX_EPOCH,
                // which should never happen in normal operation. If it does, use current time as fallback.
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_else(|_| {
                        // Fallback to a reasonable timestamp if system time is invalid
                        // This should never occur in practice, but provides a safe fallback
                        std::time::Duration::from_nanos(0)
                    })
                    .as_nanos() as u64;
                module.module_meta.last_accessed_at = now;
                module.module_meta.access_count += 1;
            }
        }

        let result = self
            .runtime
            .execute_component_async(wasm_bytes, attach_point, input)
            .await;

        // Record metrics
        let execution_time_ms = start_time.elapsed().as_millis() as u64;
        self.total_executions.fetch_add(1, Ordering::Relaxed);
        self.total_execution_time_ms
            .fetch_add(execution_time_ms, Ordering::Relaxed);
        // Update max execution time
        self.max_execution_time_ms
            .fetch_max(execution_time_ms, Ordering::Relaxed);

        if result.is_ok() {
            self.successful_executions.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_executions.fetch_add(1, Ordering::Relaxed);
        }

        result
    }

    /// Execute WASM module using WebAssembly component model (sync version)
    pub fn execute_module_interface_sync(
        &self,
        module_uuid: Uuid,
        attach_point: WasmModuleAttachPoint,
        input: WasmComponentInput,
    ) -> Result<WasmComponentOutput> {
        let handle = tokio::runtime::Handle::current();
        handle.block_on(self.execute_module_interface(module_uuid, attach_point, input))
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.total_executions.load(Ordering::Relaxed),
            self.successful_executions.load(Ordering::Relaxed),
            self.failed_executions.load(Ordering::Relaxed),
            self.total_execution_time_ms.load(Ordering::Relaxed),
            self.max_execution_time_ms.load(Ordering::Relaxed),
        )
    }

    /// Execute a WASM module for a given attach point
    /// Returns the Action if successful, or None if execution failed
    ///
    /// This is a convenience method that wraps execute_module_interface and handles
    /// error logging automatically.
    pub async fn execute_module_for_attach_point(
        &self,
        module: &WasmModule,
        attach_point: WasmModuleAttachPoint,
        input: WasmComponentInput,
    ) -> Option<crate::wasm::spec::sgl::model_gateway::middleware_types::Action> {
        use tracing::error;

        let action_result = self
            .execute_module_interface(module.module_uuid, attach_point, input)
            .await;

        match action_result {
            Ok(output) => match output {
                WasmComponentOutput::MiddlewareAction(action) => Some(action),
            },
            Err(e) => {
                error!(
                    "Failed to execute WASM module {}: {}",
                    module.module_meta.name, e
                );
                None
            }
        }
    }
}

impl Default for WasmModuleManager {
    fn default() -> Self {
        // with_default_config() should always succeed with default configuration.
        // If it fails, it indicates a critical system configuration error.
        Self::with_default_config()
            .expect("Failed to create WasmModuleManager with default config. This should never happen with valid default configuration.")
    }
}
