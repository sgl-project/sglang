use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WasmRuntimeConfig {
    /// Maximum memory size in pages (64KB per page)
    pub max_memory_pages: u32,
    /// Maximum execution time in milliseconds
    pub max_execution_time_ms: u64,
    /// Enable WASI support
    pub enable_wasi: bool,
    /// Maximum stack size in bytes
    pub max_stack_size: usize,
    /// Number of worker threads in the pool
    pub thread_pool_size: usize,
    /// Maximum number of modules to cache per worker
    pub module_cache_size: usize,
}

impl Default for WasmRuntimeConfig {
    fn default() -> Self {
        let default_thread_pool_size = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);

        Self {
            max_memory_pages: 1024,      // 64MB
            max_execution_time_ms: 1000, // 1 seconds
            enable_wasi: true,
            max_stack_size: 1024 * 1024,                // 1MB
            thread_pool_size: default_thread_pool_size, // based on cpu count
            module_cache_size: 10,                      // Cache up to 10 modules per worker
        }
    }
}
