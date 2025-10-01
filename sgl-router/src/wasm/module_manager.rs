//! WASM Module Manager

use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
};

use sha2::{Digest, Sha256};
use uuid::Uuid;
use wasmtime::{component::Component, Config, Engine};

use crate::wasm::{
    config::WasmRuntimeConfig,
    errors::{Result, WasmError, WasmManagerError, WasmModuleError, WasmRuntimeError},
    module::{WasmModule, WasmModuleAttachPoint, WasmModuleDescriptor, WasmModuleMeta},
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

    fn check_duplicate_sha256_hash(&self, sha256_hash: &[u8; 32]) -> Result<()> {
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

    fn calculate_size_bytes(&self, file_path: &str) -> Result<u64> {
        let file = File::open(file_path).map_err(|e| WasmError::from(e))?;
        let metadata = file.metadata().map_err(|e| WasmError::from(e))?;
        Ok(metadata.len())
    }

    fn calculate_sha256_hash(&self, file_path: &str) -> Result<[u8; 32]> {
        let mut file = File::open(file_path).map_err(|e| WasmError::from(e))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0; 1024];
        loop {
            let bytes_read = file.read(&mut buffer).map_err(|e| WasmError::from(e))?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }
        Ok(hasher.finalize().into())
    }

    fn validate_module_descriptor(&self, descriptor: WasmModuleDescriptor) -> Result<()> {
        if descriptor.name.is_empty() {
            return Err(WasmModuleError::InvalidDescriptor(
                "Module name cannot be empty".to_string(),
            )
            .into());
        }
        if descriptor.file_path.is_empty() {
            return Err(WasmModuleError::InvalidDescriptor(
                "Module file path cannot be empty".to_string(),
            )
            .into());
        }
        if self.calculate_size_bytes(&descriptor.file_path)? == 0 {
            return Err(WasmModuleError::ValidationFailed(
                "Module file size cannot be 0".to_string(),
            )
            .into());
        }
        Ok(())
    }

    pub fn add_module(&self, descriptor: WasmModuleDescriptor) -> Result<Uuid> {
        // validate the module descriptor
        self.validate_module_descriptor(descriptor.clone())?;

        // calculate the sha256 hash of the module file
        let sha256_hash = self.calculate_sha256_hash(&descriptor.file_path)?;
        self.check_duplicate_sha256_hash(&sha256_hash)?;

        // calculate size before moving descriptor
        let size_bytes = self.calculate_size_bytes(&descriptor.file_path)?;

        // Pre-load WASM bytes into memory for faster execution
        let wasm_bytes = std::fs::read(&descriptor.file_path).map_err(|e| {
            WasmError::from(WasmModuleError::FileRead(format!(
                "Failed to read WASM file: {}",
                e
            )))
        })?;

        // Validate that the WASM file is a valid component by attempting to compile it
        // This catches errors early during module addition rather than during execution
        self.validate_wasm_component(&wasm_bytes)?;

        // now safe, insert the module into the manager
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
        let module_uuid = Uuid::new_v4();
        let module = WasmModule {
            module_uuid,
            module_meta: WasmModuleMeta {
                name: descriptor.name,
                file_path: descriptor.file_path,
                sha256_hash,
                size_bytes,
                created_at: now,
                last_accessed_at: now,
                access_count: 0,
                attach_points: descriptor.attach_points.clone(),
                wasm_bytes,
            },
        };

        let mut modules = self
            .modules
            .write()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        modules.insert(module_uuid, module);
        Ok(module_uuid)
    }

    /// Validate that WASM bytes represent a valid component
    fn validate_wasm_component(&self, wasm_bytes: &[u8]) -> Result<()> {
        // Create a temporary engine to validate the component
        let mut config = Config::new();
        config.async_support(true);
        config.wasm_component_model(true);
        let engine = Engine::new(&config)
            .map_err(|e| WasmError::from(WasmRuntimeError::EngineCreateFailed(e.to_string())))?;

        // Attempt to compile the component to validate it
        Component::new(&engine, wasm_bytes)
            .map_err(|e| WasmError::from(WasmRuntimeError::CompileFailed(format!(
                "Invalid WASM component: {}. \
                 Hint: The WASM file must be in component format. \
                 If you're using wit-bindgen, use 'wasm-tools component new' to wrap the WASM module into a component.",
                e
            ))))?;

        Ok(())
    }

    pub fn remove_module(&self, module_uuid: Uuid) -> Result<()> {
        let mut modules = self
            .modules
            .write()
            .map_err(|e| WasmManagerError::LockFailed(e.to_string()))?;
        if !modules.contains_key(&module_uuid) {
            return Err(WasmManagerError::ModuleNotFound(module_uuid).into());
        }
        // Remove the module - the wasm_bytes Vec will be dropped automatically,
        // releasing the memory
        modules.remove(&module_uuid);
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

    /// Execute WASM module using WIT component model based on attach_point
    pub async fn execute_module_wit(
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

    /// Execute WASM module using WIT component model (sync version)
    pub fn execute_module_wit_sync(
        &self,
        module_uuid: Uuid,
        attach_point: WasmModuleAttachPoint,
        input: WasmComponentInput,
    ) -> Result<WasmComponentOutput> {
        let handle = tokio::runtime::Handle::current();
        handle.block_on(self.execute_module_wit(module_uuid, attach_point, input))
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
}

impl Default for WasmModuleManager {
    fn default() -> Self {
        // with_default_config() should always succeed with default configuration.
        // If it fails, it indicates a critical system configuration error.
        Self::with_default_config()
            .expect("Failed to create WasmModuleManager with default config. This should never happen with valid default configuration.")
    }
}
