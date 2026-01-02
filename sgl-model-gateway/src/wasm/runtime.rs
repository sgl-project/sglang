//! WASM Runtime
//!
//! Manages WASM component execution using wasmtime with async support.
//! Provides a thread pool for concurrent WASM execution and metrics tracking.

use std::{
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use lru::LruCache;
use tokio::sync::oneshot;
use tracing::{debug, error};
use wasmtime::{
    component::{Component, Linker, ResourceTable},
    Config, Engine, InstanceAllocationStrategy, PoolingAllocationConfig, Store, StoreLimitsBuilder,
    UpdateDeadline,
};
use wasmtime_wasi::WasiCtx;

/// Epoch increment interval in milliseconds.
/// Epochs are used for cooperative timeout enforcement in WASM execution.
/// A smaller interval gives finer-grained timeout control but slightly more overhead.
const EPOCH_INTERVAL_MS: u64 = 100;

use crate::wasm::{
    config::WasmRuntimeConfig,
    errors::{Result, WasmError, WasmRuntimeError},
    module::{MiddlewareAttachPoint, WasmModuleAttachPoint},
    spec::SglModelGateway,
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

    /// Execute WASM component using WASM interface based on attach_point
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

/// Maps a wasmtime error to a WasmError, detecting epoch interruption (timeout) traps.
fn map_wasm_error(e: wasmtime::Error, timeout_ms: u64) -> WasmError {
    // Use proper trap code detection instead of brittle string matching.
    // Wasmtime uses Trap::Interrupt for epoch-based interruptions.
    if e.downcast_ref::<wasmtime::Trap>() == Some(&wasmtime::Trap::Interrupt) {
        WasmError::from(WasmRuntimeError::Timeout(timeout_ms))
    } else {
        WasmError::from(WasmRuntimeError::CallFailed(e.to_string()))
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

        debug!(
            target: "sgl_model_gateway::wasm::runtime",
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
                            target: "sgl_model_gateway::wasm::runtime",
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
            target: "sgl_model_gateway::wasm::runtime",
            worker_id = worker_id,
            thread_id = ?std::thread::current().id(),
            "Worker started"
        );

        let mut pool_config = PoolingAllocationConfig::default();
        let max_memory_bytes = (config.max_memory_pages as usize) * 65536;

        // Since this thread handles tasks sequentially, we don't need a large pool per thread.
        // A pool size of 20 allows for efficient reuse without hogging memory.
        pool_config.total_core_instances(20);
        pool_config.max_memory_size(max_memory_bytes);
        pool_config.max_component_instance_size(max_memory_bytes);
        pool_config.max_tables_per_component(5);

        let mut wasmtime_config = Config::new();
        wasmtime_config.allocation_strategy(InstanceAllocationStrategy::Pooling(pool_config));

        wasmtime_config.async_stack_size(config.max_stack_size);
        wasmtime_config.async_support(true);
        wasmtime_config.wasm_component_model(true); // Enable component model
        wasmtime_config.epoch_interruption(true); // Enable epoch-based timeout interruption

        let engine = match Engine::new(&wasmtime_config) {
            Ok(engine) => engine,
            Err(e) => {
                error!(
                    target: "sgl_model_gateway::wasm::runtime",
                    worker_id = worker_id,
                    "Failed to create engine: {}",
                    e
                );
                return;
            }
        };

        let cache_capacity =
            NonZeroUsize::new(config.module_cache_size).unwrap_or(NonZeroUsize::new(10).unwrap());
        let mut component_cache: LruCache<Vec<u8>, Component> = LruCache::new(cache_capacity);

        // Start epoch incrementer for timeout enforcement.
        // The engine's epoch counter is incremented periodically, and each Store
        // can set a deadline (number of epochs). When the deadline is reached,
        // WASM execution is interrupted with a trap.
        let engine_for_epoch = engine.clone();
        let epoch_handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(EPOCH_INTERVAL_MS));
            loop {
                interval.tick().await;
                engine_for_epoch.increment_epoch();
            }
        });

        debug!(
            target: "sgl_model_gateway::wasm::runtime",
            worker_id = worker_id,
            epoch_interval_ms = EPOCH_INTERVAL_MS,
            "Epoch incrementer started for timeout enforcement"
        );

        loop {
            let task = match receiver.recv().await {
                Ok(task) => task,
                Err(_) => {
                    debug!(
                        target: "sgl_model_gateway::wasm::runtime",
                        worker_id = worker_id,
                        "Worker shutting down"
                    );
                    epoch_handle.abort(); // Stop the epoch incrementer
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
                        &mut component_cache, // Pass the cache
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
        cache: &mut LruCache<Vec<u8>, Component>, //  cache argument
        wasm_bytes: Vec<u8>,
        attach_point: WasmModuleAttachPoint,
        input: WasmComponentInput,
        config: &WasmRuntimeConfig,
    ) -> Result<WasmComponentOutput> {
        // Compile component from bytes OR retrieve from cache
        // Note: The WASM file must be in component format (not plain WASM module)
        let component = if let Some(comp) = cache.get(&wasm_bytes) {
            comp.clone() // Component is just a handle (cheap clone)
        } else {
            // Compile new component
            let comp = Component::new(engine, &wasm_bytes).map_err(|e| {
                WasmRuntimeError::CompileFailed(format!(
                    "failed to parse WebAssembly component: {}. \
                     Hint: The WASM file must be in component format. \
                     If you're using wit-bindgen, use 'wasm-tools component new' to wrap the WASM module into a component.",
                    e
                ))
            })?;

            cache.push(wasm_bytes, comp.clone());
            comp
        };

        let mut linker = Linker::<WasiState>::new(engine);
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)?;
        let mut builder = WasiCtx::builder();

        // Create memory limits from config.
        // Use the config helper to get total bytes, then safely convert to usize.
        let memory_limit_bytes =
            usize::try_from(config.get_total_memory_bytes()).map_err(|_| {
                WasmError::from(WasmRuntimeError::CallFailed(
                    "Configured WASM memory limit exceeds addressable space on this platform."
                        .to_string(),
                ))
            })?;
        let limits = StoreLimitsBuilder::new()
            .memory_size(memory_limit_bytes)
            .trap_on_grow_failure(true) // Trap instead of returning -1 for easier debugging
            .build();

        let mut store = Store::new(
            engine,
            WasiState {
                ctx: builder.build(),
                table: ResourceTable::new(),
                limits,
            },
        );

        // Apply resource limits to the store.
        // This enforces max_memory_pages by preventing memory.grow beyond the limit.
        store.limiter(|state| &mut state.limits);

        // Set epoch deadline for timeout enforcement.
        // The deadline is the number of epoch ticks before execution is interrupted.
        // With EPOCH_INTERVAL_MS=100ms and max_execution_time_ms=1000ms, deadline=10 epochs.
        let deadline_epochs = (config.max_execution_time_ms / EPOCH_INTERVAL_MS).max(1);
        store.set_epoch_deadline(deadline_epochs);

        // Configure what happens when the deadline is reached during async yields
        store.epoch_deadline_callback(|_store| Ok(UpdateDeadline::Yield(1)));

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
                let bindings = SglModelGateway::instantiate_async(&mut store, &component, &linker)
                    .await
                    .map_err(|e| {
                        WasmError::from(WasmRuntimeError::InstanceCreateFailed(e.to_string()))
                    })?;

                // Call on-request (async call when async support is enabled)
                let action_result = bindings
                    .sgl_model_gateway_middleware_on_request()
                    .call_on_request(&mut store, &request)
                    .await
                    .map_err(|e| map_wasm_error(e, config.max_execution_time_ms))?;

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
                let bindings = SglModelGateway::instantiate_async(&mut store, &component, &linker)
                    .await
                    .map_err(|e| {
                        WasmError::from(WasmRuntimeError::InstanceCreateFailed(e.to_string()))
                    })?;

                // Call on-response (async call when async support is enabled)
                let action_result = bindings
                    .sgl_model_gateway_middleware_on_response()
                    .call_on_response(&mut store, &response)
                    .await
                    .map_err(|e| map_wasm_error(e, config.max_execution_time_ms))?;

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
    use std::{num::NonZeroUsize, time::Instant};

    use lru::LruCache;

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
    #[test]
    fn test_wasm_instantiation_performance_threshold() -> Result<()> {
        // A simple WASM module forcing memory allocation
        const WASM_WAT: &str = r#"
            (module
                (memory (export "memory") 1)
                (func (export "run") (param i32 i32) (result i32)
                    local.get 0
                    local.get 1
                    i32.add)
            )
        "#;

        let iterations = 1000;

        //  Scenario A: Baseline (No Pool, No Cache)
        let engine_standard = Engine::default();
        let start_standard = Instant::now();
        for _ in 0..iterations {
            // Simulate compilation + instantiation overhead
            let module = wasmtime::Module::new(&engine_standard, WASM_WAT).unwrap();
            let mut store = Store::new(&engine_standard, ());
            let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
            let run_func = instance
                .get_typed_func::<(i32, i32), i32>(&mut store, "run")
                .unwrap();
            let _ = run_func.call(&mut store, (10, 20)).unwrap();
        }
        let duration_standard = start_standard.elapsed();

        // --- Scenario B: Optimized (Pool + Cache)
        let mut pool_config = PoolingAllocationConfig::default();

        pool_config.total_core_instances(100);

        let mut config = Config::new();
        config.allocation_strategy(InstanceAllocationStrategy::Pooling(pool_config));

        let engine_pooled = Engine::new(&config).unwrap();

        // Setup LRU Cache
        let cache_capacity = NonZeroUsize::new(100).unwrap();
        let mut cache: LruCache<Vec<u8>, wasmtime::Module> = LruCache::new(cache_capacity);

        // Pre-warm cache (simulating the "cached" state)
        let key = WASM_WAT.as_bytes().to_vec();
        let module_compiled = wasmtime::Module::new(&engine_pooled, WASM_WAT).unwrap();
        cache.push(key.clone(), module_compiled);

        let start_pooled = Instant::now();
        for _ in 0..iterations {
            let module = cache.get(&key).unwrap().clone();
            let mut store = Store::new(&engine_pooled, ());
            let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
            let run_func = instance
                .get_typed_func::<(i32, i32), i32>(&mut store, "run")
                .unwrap();
            let _ = run_func.call(&mut store, (10, 20)).unwrap();
        }
        let duration_pooled = start_pooled.elapsed();

        // Verify Speedup
        let standard_secs = duration_standard.as_secs_f64();
        let pooled_secs = duration_pooled.as_secs_f64();

        if pooled_secs > 0.0 {
            let speedup = standard_secs / pooled_secs;

            assert!(
                speedup > 5.0,
                "Optimization regression: Pooling+Caching was only {:.2}x faster",
                speedup
            );
        }

        Ok(())
    }
}
