use crate::wasm::config::WasmRuntimeConfig;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use wasmtime::{Config, Engine, Instance, Module, Store, Val};

pub struct WasmRuntime {
    config: WasmRuntimeConfig,
    thread_pool: Arc<WasmThreadPool>,
}

pub struct WasmThreadPool {
    sender: mpsc::UnboundedSender<WasmTask>,
    workers: Vec<std::thread::JoinHandle<()>>,
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

        response_rx
            .await
            .map_err(|_| "Failed to receive response from thread pool".to_string())?
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

        Ok(Self { sender, workers })
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
                    let result = Self::execute_wasm_in_worker(
                        &engine,
                        wasm_bytes,
                        function_name,
                        args,
                        &config,
                    )
                    .await;

                    let _ = response.send(result);
                }
            }
        }
    }

    async fn execute_wasm_in_worker(
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
        // close sender, let workers exit naturally
        drop(self.sender.clone());

        // wait for all workers to complete
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
