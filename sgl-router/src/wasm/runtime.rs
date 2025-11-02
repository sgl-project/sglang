//! WASM Runtime
//!
//! Manages WASM component execution using wasmtime with async support.
//! Provides a thread pool for concurrent WASM execution and metrics tracking.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use tokio::sync::oneshot;
use tracing::{debug, error, info};
use wasmtime::{
    component::{Component, Linker, ResourceTable},
    Config, Engine, Store,
};
use wasmtime_wasi::WasiCtx;

use crate::wasm::{
    config::WasmRuntimeConfig,
    errors::{Result, WasmError, WasmRuntimeError},
    module::{MiddlewareAttachPoint, WasmModuleAttachPoint},
    spec::SglRouter,
    types::{WasiState, WasmComponentInput, WasmComponentOutput},
};

pub struct WasmRuntime {
    config: WasmRuntimeConfig,
    thread_pool: Arc<WasmThreadPool>,
    // Metrics
    total_executions: AtomicU64,
    successful_executions: AtomicU64,
    failed_executions: AtomicU64,
    total_execution_time_ms: AtomicU64,
    max_execution_time_ms: AtomicU64,
}

pub struct WasmThreadPool {
    sender: async_channel::Sender<WasmTask>,
    receiver: async_channel::Receiver<WasmTask>,
    workers: Vec<std::thread::JoinHandle<()>>,
    // Metrics
    total_tasks: AtomicU64,
    completed_tasks: AtomicU64,
    failed_tasks: AtomicU64,
}

pub enum WasmTask {
    ExecuteComponent {
        wasm_bytes: Vec<u8>,
        attach_point: WasmModuleAttachPoint,
        input: WasmComponentInput,
        response: oneshot::Sender<Result<WasmComponentOutput>>,
    },
}

impl WasmRuntime {
    pub fn new(config: WasmRuntimeConfig) -> Result<Self> {
        let thread_pool = Arc::new(WasmThreadPool::new(config.clone())?);

        Ok(Self {
            config,
            thread_pool,
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

    pub fn get_config(&self) -> &WasmRuntimeConfig {
        &self.config
    }

    /// get available cpu count and max recommended cpu count
    pub fn get_cpu_info() -> (usize, usize) {
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let max_recommended = cpu_count.max(1);
        (cpu_count, max_recommended)
    }

    /// get current thread pool status
    pub fn get_thread_pool_info(&self) -> (usize, usize) {
        let (_cpu_count, max_recommended) = Self::get_cpu_info();
        let current_workers = self.thread_pool.workers.len();
        (current_workers, max_recommended)
    }

    /// Execute WASM component using WIT interface based on attach_point
    pub async fn execute_component_async(
        &self,
        wasm_bytes: Vec<u8>,
        attach_point: WasmModuleAttachPoint,
        input: WasmComponentInput,
    ) -> Result<WasmComponentOutput> {
        let start_time = std::time::Instant::now();
        let (response_tx, response_rx) = oneshot::channel();

        let task = WasmTask::ExecuteComponent {
            wasm_bytes,
            attach_point,
            input,
            response: response_tx,
        };

        self.thread_pool.sender.send(task).await.map_err(|e| {
            WasmRuntimeError::CallFailed(format!("Failed to send task to thread pool: {}", e))
        })?;

        let result = response_rx.await.map_err(|e| {
            WasmRuntimeError::CallFailed(format!(
                "Failed to receive response from thread pool: {}",
                e
            ))
        })?;

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

impl WasmThreadPool {
    pub fn new(config: WasmRuntimeConfig) -> Result<Self> {
        let (sender, receiver) = async_channel::unbounded();

        let mut workers = Vec::new();
        // set thread pool size based on cpu count
        let max_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);
        let num_workers = config.thread_pool_size.clamp(1, max_workers);

        info!(
            target: "sglang_router_rs::wasm::runtime",
            "Initializing WASM runtime with {} workers",
            num_workers
        );

        for worker_id in 0..num_workers {
            let receiver = receiver.clone();
            let config = config.clone();

            let worker = std::thread::spawn(move || {
                // create independent tokio runtime for this thread
                let rt = match tokio::runtime::Runtime::new() {
                    Ok(rt) => rt,
                    Err(e) => {
                        error!(
                            target: "sglang_router_rs::wasm::runtime",
                            worker_id = worker_id,
                            "Failed to create tokio runtime: {}",
                            e
                        );
                        return;
                    }
                };

                rt.block_on(async {
                    Self::worker_loop(worker_id, receiver, config).await;
                });
            });

            workers.push(worker);
        }

        Ok(Self {
            sender,
            receiver,
            workers,
            total_tasks: AtomicU64::new(0),
            completed_tasks: AtomicU64::new(0),
            failed_tasks: AtomicU64::new(0),
        })
    }

    /// Get current thread pool metrics
    pub fn get_metrics(&self) -> (u64, u64, u64) {
        (
            self.total_tasks.load(Ordering::Relaxed),
            self.completed_tasks.load(Ordering::Relaxed),
            self.failed_tasks.load(Ordering::Relaxed),
        )
    }

    async fn worker_loop(
        worker_id: usize,
        receiver: async_channel::Receiver<WasmTask>,
        config: WasmRuntimeConfig,
    ) {
        debug!(
            target: "sglang_router_rs::wasm::runtime",
            worker_id = worker_id,
            thread_id = ?std::thread::current().id(),
            "Worker started"
        );

        let mut wasmtime_config = Config::new();
        wasmtime_config.async_stack_size(config.max_stack_size);
        wasmtime_config.async_support(true);
        wasmtime_config.wasm_component_model(true); // Enable component model

        let engine = match Engine::new(&wasmtime_config) {
            Ok(engine) => engine,
            Err(e) => {
                error!(
                    target: "sglang_router_rs::wasm::runtime",
                    worker_id = worker_id,
                    "Failed to create engine: {}",
                    e
                );
                return;
            }
        };

        loop {
            let task = match receiver.recv().await {
                Ok(task) => task,
                Err(_) => {
                    debug!(
                        target: "sglang_router_rs::wasm::runtime",
                        worker_id = worker_id,
                        "Worker shutting down"
                    );
                    break; // channel closed, exit loop
                }
            };

            match task {
                WasmTask::ExecuteComponent {
                    wasm_bytes,
                    attach_point,
                    input,
                    response,
                } => {
                    let result = Self::execute_component_in_worker(
                        &engine,
                        wasm_bytes,
                        attach_point,
                        input,
                        &config,
                    )
                    .await;

                    let _ = response.send(result);
                }
            }
        }
    }

    async fn execute_component_in_worker(
        engine: &Engine,
        wasm_bytes: Vec<u8>,
        attach_point: WasmModuleAttachPoint,
        input: WasmComponentInput,
        _config: &WasmRuntimeConfig,
    ) -> Result<WasmComponentOutput> {
        // Compile component from bytes
        // Note: The WASM file must be in component format (not plain WASM module)
        // Use `wasm-tools component new` to wrap a WASM module into a component if needed
        let component = Component::new(engine, &wasm_bytes).map_err(|e| {
            WasmRuntimeError::CompileFailed(format!(
                "failed to parse WebAssembly component: {}. \
                 Hint: The WASM file must be in component format. \
                 If you're using wit-bindgen, use 'wasm-tools component new' to wrap the WASM module into a component.",
                e
            ))
        })?;

        let mut linker = Linker::<WasiState>::new(engine);
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
        let mut builder = WasiCtx::builder();
        let mut store = Store::new(
            engine,
            WasiState {
                ctx: builder.build(),
                table: ResourceTable::new(),
            },
        );

        let output = match attach_point {
            WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnRequest) => {
                let request = match input {
                    WasmComponentInput::MiddlewareRequest(req) => req,
                    _ => {
                        return Err(WasmError::from(WasmRuntimeError::CallFailed(
                            "Expected MiddlewareRequest input for OnRequest attach point"
                                .to_string(),
                        )));
                    }
                };

                // Instantiate component (must use async instantiation when async support is enabled)
                let bindings = SglRouter::instantiate_async(&mut store, &component, &linker)
                    .await
                    .map_err(|e| {
                        WasmError::from(WasmRuntimeError::InstanceCreateFailed(e.to_string()))
                    })?;

                // Call on-request (async call when async support is enabled)
                let action_result = bindings
                    .sgl_router_middleware_on_request()
                    .call_on_request(&mut store, &request)
                    .await
                    .map_err(|e| WasmError::from(WasmRuntimeError::CallFailed(e.to_string())))?;

                WasmComponentOutput::MiddlewareAction(action_result)
            }
            WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnResponse) => {
                // Extract Response input
                let response = match input {
                    WasmComponentInput::MiddlewareResponse(resp) => resp,
                    _ => {
                        return Err(WasmError::from(WasmRuntimeError::CallFailed(
                            "Expected MiddlewareResponse input for OnResponse attach point"
                                .to_string(),
                        )));
                    }
                };

                // Instantiate component (must use async instantiation when async support is enabled)
                let bindings = SglRouter::instantiate_async(&mut store, &component, &linker)
                    .await
                    .map_err(|e| {
                        WasmError::from(WasmRuntimeError::InstanceCreateFailed(e.to_string()))
                    })?;

                // Call on-response (async call when async support is enabled)
                let action_result = bindings
                    .sgl_router_middleware_on_response()
                    .call_on_response(&mut store, &response)
                    .await
                    .map_err(|e| WasmError::from(WasmRuntimeError::CallFailed(e.to_string())))?;

                WasmComponentOutput::MiddlewareAction(action_result)
            }
            WasmModuleAttachPoint::Middleware(MiddlewareAttachPoint::OnError) => {
                return Err(WasmError::from(WasmRuntimeError::CallFailed(
                    "OnError attach point not yet implemented".to_string(),
                )));
            }
        };

        Ok(output)
    }
}

impl Drop for WasmThreadPool {
    fn drop(&mut self) {
        // close sender and receiver
        self.sender.close();
        self.receiver.close();

        // wait for all workers to complete
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::config::WasmRuntimeConfig;

    #[test]
    fn test_get_cpu_info() {
        let (cpu_count, max_recommended) = WasmRuntime::get_cpu_info();
        assert!(cpu_count > 0);
        assert!(max_recommended > 0);
        assert!(max_recommended >= cpu_count);
    }

    #[test]
    fn test_config_default_values() {
        let config = WasmRuntimeConfig::default();

        assert_eq!(config.max_memory_pages, 1024);
        assert_eq!(config.max_execution_time_ms, 1000);
        assert_eq!(config.max_stack_size, 1024 * 1024);
        assert!(config.thread_pool_size > 0);
        assert_eq!(config.module_cache_size, 10);
    }

    #[test]
    fn test_config_clone() {
        let config = WasmRuntimeConfig::default();
        let cloned_config = config.clone();

        assert_eq!(config.max_memory_pages, cloned_config.max_memory_pages);
        assert_eq!(
            config.max_execution_time_ms,
            cloned_config.max_execution_time_ms
        );
        assert_eq!(config.max_stack_size, cloned_config.max_stack_size);
        assert_eq!(config.thread_pool_size, cloned_config.thread_pool_size);
        assert_eq!(config.module_cache_size, cloned_config.module_cache_size);
    }
}
