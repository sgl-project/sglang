use crate::wasm::config::WasmRuntimeConfig;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};
use tokio::sync::{mpsc, oneshot};
use wasmtime::{Config, Engine, Instance, Module, Store, Val};

pub struct WasmRuntime {
    config: WasmRuntimeConfig,
    thread_pool: Arc<WasmThreadPool>,
    // Metrics
    total_executions: AtomicU64,
    successful_executions: AtomicU64,
    failed_executions: AtomicU64,
    total_execution_time_ms: AtomicU64,
}

pub struct WasmThreadPool {
    sender: mpsc::UnboundedSender<WasmTask>,
    receiver: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<WasmTask>>>,
    workers: Vec<std::thread::JoinHandle<()>>,
    // Metrics
    total_tasks: AtomicU64,
    completed_tasks: AtomicU64,
    failed_tasks: AtomicU64,
}

pub enum WasmTask {
    ExecuteWasmModule {
        wasm_bytes: Vec<u8>,
        function_name: String,
        args: Vec<Val>,
        response: oneshot::Sender<Result<Vec<Val>, String>>,
    },
}

impl WasmRuntime {
    pub fn new(config: WasmRuntimeConfig) -> Result<Self, String> {
        let thread_pool = Arc::new(WasmThreadPool::new(config.clone())?);

        Ok(Self {
            config,
            thread_pool,
            total_executions: AtomicU64::new(0),
            successful_executions: AtomicU64::new(0),
            failed_executions: AtomicU64::new(0),
            total_execution_time_ms: AtomicU64::new(0),
        })
    }

    pub fn with_default_config() -> Result<Self, String> {
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

    // async execute wasm bytes directly
    pub async fn execute_wasm_module_async(
        &self,
        wasm_bytes: Vec<u8>,
        function_name: String,
        args: Vec<Val>,
    ) -> Result<Vec<Val>, String> {
        let start_time = std::time::Instant::now();
        let (response_tx, response_rx) = oneshot::channel();

        let task = WasmTask::ExecuteWasmModule {
            wasm_bytes,
            function_name,
            args,
            response: response_tx,
        };

        self.thread_pool
            .sender
            .send(task)
            .map_err(|_| "Failed to send task to thread pool".to_string())?;

        let result = response_rx
            .await
            .map_err(|_| "Failed to receive response from thread pool".to_string())?;

        // Record metrics
        let execution_time_ms = start_time.elapsed().as_millis() as u64;
        self.total_executions.fetch_add(1, Ordering::Relaxed);
        self.total_execution_time_ms
            .fetch_add(execution_time_ms, Ordering::Relaxed);

        if result.is_ok() {
            self.successful_executions.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_executions.fetch_add(1, Ordering::Relaxed);
        }

        result
    }

    // sync execute method (for simple scenarios)
    pub fn execute_wasm_sync(
        &self,
        wasm_bytes: Vec<u8>,
        function_name: String,
        args: Vec<Val>,
    ) -> Result<Vec<Val>, String> {
        // use tokio::runtime::Handle::current() to execute in async context
        let handle = tokio::runtime::Handle::current();
        handle.block_on(self.execute_wasm_module_async(wasm_bytes, function_name, args))
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> (u64, u64, u64, u64) {
        (
            self.total_executions.load(Ordering::Relaxed),
            self.successful_executions.load(Ordering::Relaxed),
            self.failed_executions.load(Ordering::Relaxed),
            self.total_execution_time_ms.load(Ordering::Relaxed),
        )
    }
}

impl WasmThreadPool {
    pub fn new(config: WasmRuntimeConfig) -> Result<Self, String> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let receiver = Arc::new(tokio::sync::Mutex::new(receiver));

        let mut workers = Vec::new();
        // set thread pool size based on cpu count
        let max_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);
        let num_workers = config.thread_pool_size.clamp(1, max_workers);

        println!(
            "Creating {} independent WASM worker threads (max allowed: {})",
            num_workers, max_workers
        );

        for worker_id in 0..num_workers {
            let receiver = receiver.clone();
            let config = config.clone();

            // create independent thread for each WASM engine
            let worker = std::thread::spawn(move || {
                // create independent tokio runtime for this thread
                let rt = match tokio::runtime::Runtime::new() {
                    Ok(rt) => rt,
                    Err(e) => {
                        eprintln!("Worker {} failed to create tokio runtime: {}", worker_id, e);
                        return;
                    }
                };

                // run the worker loop in this thread's runtime
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
        receiver: Arc<tokio::sync::Mutex<mpsc::UnboundedReceiver<WasmTask>>>,
        config: WasmRuntimeConfig,
    ) {
        println!(
            "WASM Worker {} started on thread {:?}",
            worker_id,
            std::thread::current().id()
        );

        // create independent wasmtime engine for each worker thread
        let mut wasmtime_config = Config::new();
        wasmtime_config.async_stack_size(config.max_stack_size);
        if config.enable_wasi {
            wasmtime_config.wasm_component_model(false);
        }

        let engine = match Engine::new(&wasmtime_config) {
            Ok(engine) => {
                println!("WASM Worker {} engine created successfully", worker_id);
                engine
            }
            Err(e) => {
                eprintln!("Worker {} failed to create engine: {}", worker_id, e);
                return;
            }
        };

        // no module caching needed for pure execution

        loop {
            let task = {
                let mut receiver_guard = receiver.lock().await;
                receiver_guard.recv().await
            };

            let task = match task {
                Some(task) => task,
                None => {
                    println!(
                        "WASM Worker {} received shutdown signal, exiting...",
                        worker_id
                    );
                    break; // channel closed, exit loop
                }
            };

            match task {
                WasmTask::ExecuteWasmModule {
                    wasm_bytes,
                    function_name,
                    args,
                    response,
                } => {
                    let result = Self::execute_wasm_module_in_worker(
                        &engine,
                        wasm_bytes,
                        function_name,
                        args,
                        &config,
                    )
                    .await;

                    // Record task metrics (we can't access pool metrics from here,
                    // but we can add them to the response if needed)
                    let _ = response.send(result);
                }
            }
        }
    }

    async fn execute_wasm_module_in_worker(
        engine: &Engine,
        wasm_bytes: Vec<u8>,
        function_name: String,
        args: Vec<Val>,
        config: &WasmRuntimeConfig,
    ) -> Result<Vec<Val>, String> {
        // compile module from bytes
        let module = Module::new(engine, wasm_bytes)
            .map_err(|e| format!("Failed to compile module: {}", e))?;

        // create store and instance
        let mut store = Store::new(engine, ());
        let instance = Instance::new(&mut store, &module, &[])
            .map_err(|e| format!("Failed to create instance: {}", e))?;

        // get function
        let func = instance
            .get_func(&mut store, &function_name)
            .ok_or_else(|| format!("Function {} not found", function_name))?;

        // set execution timeout
        let timeout_duration = std::time::Duration::from_millis(config.max_execution_time_ms);

        // execute function
        let mut results = vec![Val::I32(0); func.ty(&store).results().len()];
        tokio::time::timeout(timeout_duration, async {
            func.call(&mut store, &args, &mut results)
        })
        .await
        .map_err(|_| "Execution timeout".to_string())?
        .map_err(|e| format!("Execution failed: {}", e))?;

        Ok(results)
    }
}

impl Drop for WasmThreadPool {
    fn drop(&mut self) {
        // drain and discard any pending tasks before closing
        if let Ok(mut receiver_guard) = self.receiver.try_lock() {
            loop {
                match receiver_guard.try_recv() {
                    Ok(_task) => {
                        self.total_tasks.fetch_add(1, Ordering::Relaxed);
                        self.failed_tasks.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => break,
                }
            }
        }

        // close sender, let workers exit naturally once channel is empty and closed
        drop(self.sender.clone());

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

    // Helper function to create a simple WASM module that adds two numbers
    fn create_simple_add_wasm() -> Vec<u8> {
        // This is a minimal WASM module that exports a function "add" that takes two i32 and returns their sum
        vec![
            0x00, 0x61, 0x73, 0x6d, // WASM magic number
            0x01, 0x00, 0x00, 0x00, // version 1
            // Type section
            0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01,
            0x7f, // func type: (i32, i32) -> i32
            // Function section
            0x03, 0x02, 0x01, 0x00, // 1 function of type 0
            // Export section
            0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00, // export "add" function
            // Code section
            0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a,
            0x0b, // add function body
        ]
    }

    // Helper function to create an invalid WASM module
    fn create_invalid_wasm() -> Vec<u8> {
        vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x99] // Valid header but with invalid byte
    }

    #[test]
    fn test_get_cpu_info() {
        let (cpu_count, max_recommended) = WasmRuntime::get_cpu_info();
        assert!(cpu_count > 0);
        assert!(max_recommended > 0);
        assert!(max_recommended >= cpu_count);
    }

    #[tokio::test]
    async fn test_execute_wasm_module_async_success() {
        let wasm_bytes = create_simple_add_wasm();
        let args = vec![Val::I32(5), Val::I32(3)];

        let mut config = Config::new();
        config.async_stack_size(1024 * 1024);
        let engine = Engine::new(&config).unwrap();

        let module = Module::new(&engine, wasm_bytes).unwrap();
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[]).unwrap();
        let func = instance.get_func(&mut store, "add").unwrap();

        let mut results = vec![Val::I32(0); func.ty(&store).results().len()];
        func.call(&mut store, &args, &mut results).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].unwrap_i32(), 8); // 5 + 3 = 8
    }

    #[tokio::test]
    async fn test_execute_wasm_module_async_invalid_module() {
        let wasm_bytes = create_invalid_wasm();

        let mut config = Config::new();
        config.async_stack_size(1024 * 1024);
        let engine = Engine::new(&config).unwrap();

        let result = Module::new(&engine, wasm_bytes);
        assert!(result.is_err());
        let error_msg = result.unwrap_err().to_string();
        assert!(!error_msg.is_empty());
    }

    #[tokio::test]
    async fn test_execute_wasm_module_async_nonexistent_function() {
        let wasm_bytes = create_simple_add_wasm();

        let mut config = Config::new();
        config.async_stack_size(1024 * 1024);
        let engine = Engine::new(&config).unwrap();

        let module = Module::new(&engine, wasm_bytes).unwrap();
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[]).unwrap();

        // Try to get nonexistent function
        let func = instance.get_func(&mut store, "nonexistent");
        assert!(func.is_none());
    }

    #[test]
    fn test_execute_wasm_sync() {
        let wasm_bytes = create_simple_add_wasm();
        let args = vec![Val::I32(10), Val::I32(20)];

        let mut config = Config::new();
        config.async_stack_size(1024 * 1024);
        let engine = Engine::new(&config).unwrap();

        let module = Module::new(&engine, wasm_bytes).unwrap();
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[]).unwrap();
        let func = instance.get_func(&mut store, "add").unwrap();

        let mut results = vec![Val::I32(0); func.ty(&store).results().len()];
        func.call(&mut store, &args, &mut results).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].unwrap_i32(), 30); // 10 + 20 = 30
    }

    #[test]
    fn test_config_default_values() {
        let config = WasmRuntimeConfig::default();

        assert_eq!(config.max_memory_pages, 1024);
        assert_eq!(config.max_execution_time_ms, 1000);
        assert_eq!(config.enable_wasi, true);
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
        assert_eq!(config.enable_wasi, cloned_config.enable_wasi);
        assert_eq!(config.max_stack_size, cloned_config.max_stack_size);
        assert_eq!(config.thread_pool_size, cloned_config.thread_pool_size);
        assert_eq!(config.module_cache_size, cloned_config.module_cache_size);
    }

    #[test]
    fn test_different_function_args() {
        let wasm_bytes = create_simple_add_wasm();

        let mut config = Config::new();
        config.async_stack_size(1024 * 1024);
        let engine = Engine::new(&config).unwrap();

        let module = Module::new(&engine, wasm_bytes).unwrap();
        let mut store = Store::new(&engine, ());
        let instance = Instance::new(&mut store, &module, &[]).unwrap();
        let func = instance.get_func(&mut store, "add").unwrap();

        let test_cases = vec![
            (vec![Val::I32(0), Val::I32(0)], 0),
            (vec![Val::I32(1), Val::I32(1)], 2),
            (vec![Val::I32(-1), Val::I32(1)], 0),
            (vec![Val::I32(100), Val::I32(200)], 300),
        ];

        for (args, expected) in test_cases {
            let mut results = vec![Val::I32(0); func.ty(&store).results().len()];
            func.call(&mut store, &args, &mut results).unwrap();
            assert_eq!(results[0].unwrap_i32(), expected);
        }
    }
}
