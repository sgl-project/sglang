//! Async job queue for control plane operations
//!
//! Provides non-blocking worker management by queuing operations and processing
//! them asynchronously in background worker tasks.

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Weak},
    time::{Duration, SystemTime},
};

use dashmap::DashMap;
use smg_mcp::McpConfig;
use tokio::sync::{mpsc, Semaphore};
use tracing::{debug, error, info, warn};
use wfaas::WorkflowId;

use crate::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    core::steps::{
        create_external_worker_workflow_data, create_local_worker_workflow_data,
        create_mcp_workflow_data, create_tokenizer_workflow_data,
        create_wasm_registration_workflow_data, create_wasm_removal_workflow_data,
        create_worker_removal_workflow_data, create_worker_update_workflow_data,
        worker::local::find_workers_by_url, McpServerConfigRequest, TokenizerConfigRequest,
        TokenizerRemovalRequest, WasmModuleConfigRequest, WasmModuleRemovalRequest,
    },
    protocols::worker_spec::{JobStatus, WorkerConfigRequest, WorkerUpdateRequest},
};

/// Job types for control plane operations
#[derive(Debug, Clone)]
pub enum Job {
    AddWorker {
        config: Box<WorkerConfigRequest>,
    },
    UpdateWorker {
        url: String,
        update: Box<WorkerUpdateRequest>,
    },
    RemoveWorker {
        url: String,
    },
    InitializeWorkersFromConfig {
        router_config: Box<RouterConfig>,
    },
    InitializeMcpServers {
        mcp_config: Box<McpConfig>,
    },
    RegisterMcpServer {
        config: Box<McpServerConfigRequest>,
    },
    AddWasmModule {
        config: Box<WasmModuleConfigRequest>,
    },
    RemoveWasmModule {
        request: Box<WasmModuleRemovalRequest>,
    },
    AddTokenizer {
        config: Box<TokenizerConfigRequest>,
    },
    RemoveTokenizer {
        request: Box<TokenizerRemovalRequest>,
    },
}

impl Job {
    /// Get job type as string for logging
    pub fn job_type(&self) -> &'static str {
        match self {
            Job::AddWorker { .. } => "AddWorker",
            Job::UpdateWorker { .. } => "UpdateWorker",
            Job::RemoveWorker { .. } => "RemoveWorker",
            Job::InitializeWorkersFromConfig { .. } => "InitializeWorkersFromConfig",
            Job::InitializeMcpServers { .. } => "InitializeMcpServers",
            Job::RegisterMcpServer { .. } => "RegisterMcpServer",
            Job::AddWasmModule { .. } => "AddWasmModule",
            Job::RemoveWasmModule { .. } => "RemoveWasmModule",
            Job::AddTokenizer { .. } => "AddTokenizer",
            Job::RemoveTokenizer { .. } => "RemoveTokenizer",
        }
    }

    /// Get worker URL, MCP server name, WASM module, or tokenizer identifier for logging and status tracking
    pub fn worker_url(&self) -> &str {
        match self {
            Job::AddWorker { config } => &config.url,
            Job::UpdateWorker { url, .. } => url,
            Job::RemoveWorker { url } => url,
            Job::InitializeWorkersFromConfig { .. } => "startup",
            Job::InitializeMcpServers { .. } => "startup",
            Job::RegisterMcpServer { config } => &config.name,
            Job::AddWasmModule { config } => &config.descriptor.name,
            Job::RemoveWasmModule { request } => &request.uuid_string,
            Job::AddTokenizer { config } => &config.id,
            Job::RemoveTokenizer { request } => &request.id,
        }
    }
}

/// Job queue configuration
#[derive(Clone, Debug)]
pub struct JobQueueConfig {
    /// Maximum pending jobs in queue
    pub queue_capacity: usize,
    /// Maximum number of jobs executing concurrently
    pub max_concurrent_jobs: usize,
}

impl Default for JobQueueConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 1000,
            max_concurrent_jobs: 200,
        }
    }
}

/// Job queue manager for worker validation and removal operations
pub struct JobQueue {
    /// Channel for submitting jobs
    tx: mpsc::Sender<Job>,
    /// Weak reference to AppContext to avoid circular dependencies
    context: Weak<AppContext>,
    /// Job status tracking by worker URL
    status_map: Arc<DashMap<String, JobStatus>>,
    /// Semaphore to limit concurrent job execution
    concurrency_limit: Arc<Semaphore>,
}

impl std::fmt::Debug for JobQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobQueue")
            .field("status_count", &self.status_map.len())
            .finish()
    }
}

impl JobQueue {
    /// Create a new job queue with semaphore-based concurrency control
    ///
    /// Takes a Weak reference to AppContext to avoid circular strong references.
    /// Spawns a single dispatcher task that spawns individual job tasks with semaphore control.
    pub fn new(config: JobQueueConfig, context: Weak<AppContext>) -> Arc<Self> {
        let (tx, mut rx) = mpsc::channel(config.queue_capacity);

        debug!(
            "Initializing job queue: capacity={}, max_concurrent={}",
            config.queue_capacity, config.max_concurrent_jobs
        );

        let status_map = Arc::new(DashMap::new());
        let concurrency_limit = Arc::new(Semaphore::new(config.max_concurrent_jobs));

        let queue = Arc::new(Self {
            tx,
            context: context.clone(),
            status_map: status_map.clone(),
            concurrency_limit: concurrency_limit.clone(),
        });

        // Single dispatcher task
        let ctx = context.clone();
        let status = status_map.clone();
        let sem = concurrency_limit.clone();

        tokio::spawn(async move {
            while let Some(job) = rx.recv().await {
                // Acquire permit (blocks if at concurrency limit)
                let Ok(permit) = sem.clone().acquire_owned().await else {
                    error!("Semaphore closed, stopping dispatcher");
                    break;
                };

                let ctx_clone = ctx.clone();
                let status_clone = status.clone();

                tokio::spawn(async move {
                    Self::process_job(job, ctx_clone, status_clone, permit).await;
                });
            }

            debug!("Job dispatcher stopped");
        });

        // Spawn cleanup task for old job statuses (TTL 5 minutes)
        let cleanup_status_map = status_map.clone();
        tokio::spawn(async move {
            Self::cleanup_old_statuses(cleanup_status_map).await;
        });

        queue
    }

    /// Get current queue and concurrency status
    pub fn get_load_info(&self) -> (usize, usize) {
        let queue_depth = self.tx.max_capacity() - self.tx.capacity();
        let available_permits = self.concurrency_limit.available_permits();
        (queue_depth, available_permits)
    }

    /// Submit a job with detailed queue status
    pub async fn submit(&self, job: Job) -> Result<(), String> {
        // Check if context is still alive before accepting jobs
        if self.context.upgrade().is_none() {
            return Err("Job queue shutting down: AppContext dropped".to_string());
        }

        // Extract values before moving job
        let job_type = job.job_type();
        let worker_url = job.worker_url().to_string();

        // Record pending status
        self.status_map.insert(
            worker_url.clone(),
            JobStatus::pending(job_type, &worker_url),
        );

        match self.tx.send(job).await {
            Ok(_) => {
                let (queue_depth, available_permits) = self.get_load_info();
                debug!(
                    "Job submitted: type={}, worker={}, queue_depth={}, available_slots={}",
                    job_type, worker_url, queue_depth, available_permits
                );
                Ok(())
            }
            Err(_) => {
                self.status_map.remove(&worker_url);
                let (queue_depth, _) = self.get_load_info();
                Err(format!(
                    "Job queue full: {} jobs pending (capacity: {})",
                    queue_depth,
                    self.tx.max_capacity()
                ))
            }
        }
    }

    /// Get job status by worker URL
    pub fn get_status(&self, worker_url: &str) -> Option<JobStatus> {
        self.status_map.get(worker_url).map(|entry| entry.clone())
    }

    /// Whether any AddWorker job is currently pending or processing.
    ///
    /// Used by the removal path to detect an in-flight add/remove handoff
    /// (e.g. a rolling update replacing one worker with another).
    ///
    /// This is deliberately global rather than per-model, because the model
    /// is not knowable here: `JobStatus` records only job type, URL, status
    /// and timestamp, and k8s service discovery submits AddWorker with
    /// `model_id: None` (the model is discovered once the add pipeline
    /// reaches the engine). The queue also cannot inspect the queued `Job`
    /// itself — it has been moved into the channel or a spawned task, leaving
    /// `status_map` as the only observable state.
    ///
    /// The cost of the approximation is bounded: an unrelated model's add
    /// makes this return true, which can delay one removal by up to
    /// `HANDOFF_WAIT_MAX`. Nothing is lost, only deferred, and the TTL
    /// cleanup of `status_map` caps how long a single add can keep returning
    /// true.
    ///
    /// The `"AddWorker"` literal must match `Job::job_type()`, and the status
    /// literals must match the `JobStatus` constructors. Both are plain
    /// strings, so a rename on either side would disable this check silently.
    pub fn has_add_worker_in_flight(&self) -> bool {
        self.status_map.iter().any(|entry| {
            entry.job_type == "AddWorker"
                && matches!(entry.status.as_str(), "pending" | "processing")
        })
    }

    /// Remove job status (called when worker is deleted)
    pub fn remove_status(&self, worker_url: &str) {
        self.status_map.remove(worker_url);
    }

    /// Process a single job with status tracking and error handling
    async fn process_job(
        job: Job,
        context: Weak<AppContext>,
        status_map: Arc<DashMap<String, JobStatus>>,
        _permit: tokio::sync::OwnedSemaphorePermit,
    ) {
        let job_type = job.job_type();
        let worker_url = job.worker_url().to_string();
        let start = std::time::Instant::now();

        // Update to processing
        status_map.insert(
            worker_url.clone(),
            JobStatus::processing(job_type, &worker_url),
        );

        debug!("Processing job: type={}, worker={}", job_type, worker_url);

        // Execute job
        match context.upgrade() {
            Some(ctx) => {
                let result = Self::execute_job(&job, &ctx).await;
                let duration = start.elapsed();
                Self::record_job_completion(job_type, &worker_url, duration, &result, &status_map);
            }
            None => {
                let error_msg = "AppContext dropped".to_string();
                status_map.insert(
                    worker_url.clone(),
                    JobStatus::failed(job_type, &worker_url, error_msg),
                );
                error!(
                    "AppContext dropped, cannot process job: type={}, worker={}",
                    job_type, worker_url
                );
            }
        }

        // Permit automatically released when dropped
    }

    /// Bounded wait before removing a worker that is the last healthy worker
    /// of its model while an AddWorker job is in flight.
    ///
    /// Returns as soon as one of these holds:
    /// - the worker is already gone from the registry (nothing to protect),
    /// - the model retains at least one OTHER healthy worker,
    /// - no AddWorker job is pending/processing (e.g. intentional
    ///   scale-to-zero), or
    /// - the deadline expires (an in-flight add that cannot activate must not
    ///   block removals indefinitely).
    ///
    /// The wait is held while this job owns a job-queue concurrency permit,
    /// and it delays deregistering the departing worker — which keeps sending
    /// it new traffic for the duration. Both argue for a deadline only as
    /// large as the handoff it protects: an observed activation takes about a
    /// second, and the wait also stacks on top of the discovery resync
    /// interval against a pod's termination-drain budget.
    async fn wait_for_handoff_before_removal(url: &str, context: &Arc<AppContext>) {
        const HANDOFF_WAIT_MAX: Duration = Duration::from_secs(5);
        const POLL_INTERVAL: Duration = Duration::from_millis(200);

        let dp_aware = context.router_config.dp_aware;
        let start = std::time::Instant::now();
        loop {
            // Resolve the target exactly the way the removal workflow does. In
            // dp-aware mode a single pod URL is registered as one worker per
            // rank ("<url>@<dp_rank>"), so an exact-URL registry lookup never
            // matches and this guard would silently do nothing while the
            // removal itself still succeeds via prefix matching.
            let removing = find_workers_by_url(&context.worker_registry, url, dp_aware);
            let Some(target) = removing.first() else {
                return;
            };
            let model_id = target.model_id().to_string();
            // Every rank of the pod being removed is leaving, so none of them
            // counts as a worker the model retains.
            let removing_urls: HashSet<&str> = removing.iter().map(|w| w.url()).collect();
            let model_retains_healthy_worker = context
                .worker_registry
                .get_by_model(&model_id)
                .iter()
                .any(|w| !removing_urls.contains(w.url()) && w.is_healthy());
            if model_retains_healthy_worker {
                return;
            }

            let add_in_flight = context
                .worker_job_queue
                .get()
                .map(|queue| queue.has_add_worker_in_flight())
                .unwrap_or(false);
            if !add_in_flight {
                return;
            }

            if start.elapsed() >= HANDOFF_WAIT_MAX {
                warn!(
                    "Removing last healthy worker {} for model {} after waiting {:?} \
                     for an in-flight AddWorker to activate",
                    url, model_id, HANDOFF_WAIT_MAX
                );
                return;
            }

            debug!(
                "Delaying removal of {} (last healthy worker for model {}) while an \
                 AddWorker job is in flight",
                url, model_id
            );
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    /// Execute a specific job
    async fn execute_job(job: &Job, context: &Arc<AppContext>) -> Result<String, String> {
        match job {
            Job::AddWorker { config } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let timeout_duration =
                    Duration::from_secs(context.router_config.worker_startup_timeout_secs + 30);

                // Select workflow based on runtime field
                match config.runtime.as_deref() {
                    Some("external") => {
                        let workflow_data = create_external_worker_workflow_data(
                            (**config).clone(),
                            Arc::clone(context),
                        );
                        let instance_id = engines
                            .external_worker
                            .start_workflow(
                                WorkflowId::new("external_worker_registration"),
                                workflow_data,
                            )
                            .await
                            .map_err(|e| {
                                format!(
                                    "Failed to start external worker registration workflow: {:?}",
                                    e
                                )
                            })?;

                        debug!(
                            "Started external worker registration workflow for {} (instance: {})",
                            config.url, instance_id
                        );

                        engines
                            .external_worker
                            .wait_for_completion(instance_id, &config.url, timeout_duration)
                            .await
                    }
                    _ => {
                        let workflow_data = create_local_worker_workflow_data(
                            (**config).clone(),
                            Arc::clone(context),
                        );
                        let instance_id = engines
                            .local_worker
                            .start_workflow(
                                WorkflowId::new("local_worker_registration"),
                                workflow_data,
                            )
                            .await
                            .map_err(|e| {
                                format!(
                                    "Failed to start local worker registration workflow: {:?}",
                                    e
                                )
                            })?;

                        debug!(
                            "Started local worker registration workflow for {} (instance: {})",
                            config.url, instance_id
                        );

                        engines
                            .local_worker
                            .wait_for_completion(instance_id, &config.url, timeout_duration)
                            .await
                    }
                }
            }
            Job::UpdateWorker { url, update } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data = create_worker_update_workflow_data(
                    url.to_string(),
                    (**update).clone(),
                    Arc::clone(context),
                );

                let instance_id = engines
                    .worker_update
                    .start_workflow(WorkflowId::new("worker_update"), workflow_data)
                    .await
                    .map_err(|e| format!("Failed to start worker update workflow: {:?}", e))?;

                debug!(
                    "Started worker update workflow for {} (instance: {})",
                    url, instance_id
                );

                let timeout_duration = Duration::from_secs(30);

                engines
                    .worker_update
                    .wait_for_completion(instance_id, url, timeout_duration)
                    .await
            }
            Job::RemoveWorker { url } => {
                // Zero-downtime handoff: jobs run as independent tasks, so
                // submission order does not imply completion order. When a
                // rolling update submits AddWorker(new) and RemoveWorker(old)
                // back to back, the removal (fast) can take effect before the
                // add's multi-step activation (~1s) completes, leaving the
                // model with zero routable workers — requests fail with 404
                // during that window. If this removal would leave the model
                // without any other healthy worker while an AddWorker is in
                // flight, give the add a bounded head start. Intentional
                // scale-to-zero (no add in flight) is not delayed.
                Self::wait_for_handoff_before_removal(url, context).await;

                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data = create_worker_removal_workflow_data(
                    url.to_string(),
                    context.router_config.dp_aware,
                    Arc::clone(context),
                );

                let instance_id = engines
                    .worker_removal
                    .start_workflow(WorkflowId::new("worker_removal"), workflow_data)
                    .await
                    .map_err(|e| format!("Failed to start worker removal workflow: {:?}", e))?;

                debug!(
                    "Started worker removal workflow for {} (instance: {})",
                    url, instance_id
                );

                let timeout_duration = Duration::from_secs(30);

                let result = engines
                    .worker_removal
                    .wait_for_completion(instance_id, url, timeout_duration)
                    .await;

                // Clean up job status when removing worker
                if let Some(queue) = context.worker_job_queue.get() {
                    queue.remove_status(url);
                }

                result
            }
            Job::AddWasmModule { config } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data =
                    create_wasm_registration_workflow_data(*config.clone(), Arc::clone(context));

                let instance_id = engines
                    .wasm_registration
                    .start_workflow(WorkflowId::new("wasm_module_registration"), workflow_data)
                    .await
                    .map_err(|e| {
                        format!("Failed to start WASM module registration workflow: {:?}", e)
                    })?;

                debug!(
                    "Started WASM module registration workflow for {} (instance: {})",
                    config.descriptor.name, instance_id
                );

                let timeout_duration = Duration::from_secs(300); // 5 minutes

                engines
                    .wasm_registration
                    .wait_for_completion(instance_id, &config.descriptor.name, timeout_duration)
                    .await
            }
            Job::RemoveWasmModule { request } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data =
                    create_wasm_removal_workflow_data(*request.clone(), Arc::clone(context));

                let instance_id = engines
                    .wasm_removal
                    .start_workflow(WorkflowId::new("wasm_module_removal"), workflow_data)
                    .await
                    .map_err(|e| {
                        format!("Failed to start WASM module removal workflow: {:?}", e)
                    })?;

                debug!(
                    "Started WASM module removal workflow for {} (instance: {})",
                    request.module_uuid, instance_id
                );

                let timeout_duration = Duration::from_secs(60); // 1 minute

                engines
                    .wasm_removal
                    .wait_for_completion(
                        instance_id,
                        &request.module_uuid.to_string(),
                        timeout_duration,
                    )
                    .await
            }
            Job::InitializeWorkersFromConfig { router_config } => {
                let api_key = router_config.api_key.clone();
                let mut worker_count = 0;

                // Create iterator of (url, worker_type, bootstrap_port) tuples based on mode
                let workers: Vec<(String, &str, Option<u16>)> = match &router_config.mode {
                    RoutingMode::Regular { worker_urls } => worker_urls
                        .iter()
                        .map(|url| (url.clone(), "regular", None))
                        .collect(),
                    RoutingMode::PrefillDecode {
                        prefill_urls,
                        decode_urls,
                        ..
                    } => {
                        let prefill_workers = prefill_urls
                            .iter()
                            .map(|(url, port)| (url.clone(), "prefill", *port));

                        let decode_workers =
                            decode_urls.iter().map(|url| (url.clone(), "decode", None));

                        prefill_workers.chain(decode_workers).collect()
                    }
                    RoutingMode::OpenAI { worker_urls } => {
                        // OpenAI mode: submit AddWorker jobs with runtime: "external"
                        // The external_worker_registration workflow handles model discovery
                        let api_key = router_config.api_key.clone();
                        let mut submitted_count = 0;

                        for url in worker_urls {
                            let url_for_error = url.clone();
                            let config = WorkerConfigRequest {
                                url: url.clone(),
                                api_key: api_key.clone(),
                                worker_type: Some("regular".to_string()),
                                labels: HashMap::new(),
                                model_id: None,
                                priority: None,
                                cost: None,
                                runtime: Some("external".to_string()),
                                tokenizer_path: None,
                                reasoning_parser: None,
                                tool_parser: None,
                                chat_template: router_config.chat_template.clone(),
                                bootstrap_port: None,
                                health_check_timeout_secs: router_config.health_check.timeout_secs,
                                health_check_interval_secs: router_config
                                    .health_check
                                    .check_interval_secs,
                                health_success_threshold: router_config
                                    .health_check
                                    .success_threshold,
                                health_failure_threshold: router_config
                                    .health_check
                                    .failure_threshold,
                                disable_health_check: router_config
                                    .health_check
                                    .disable_health_check,
                                max_connection_attempts: router_config
                                    .health_check
                                    .success_threshold
                                    * 10,
                                dp_aware: false,
                            };

                            let job = Job::AddWorker {
                                config: Box::new(config),
                            };

                            if let Some(queue) = context.worker_job_queue.get() {
                                queue.submit(job).await.map_err(|e| {
                                    format!(
                                        "Failed to submit AddWorker job for external endpoint {}: {}",
                                        url_for_error, e
                                    )
                                })?;
                                submitted_count += 1;
                            } else {
                                return Err("JobQueue not available".to_string());
                            }
                        }

                        if submitted_count == 0 {
                            info!("OpenAI mode: no worker URLs provided");
                            return Ok("OpenAI mode: no worker URLs to initialize".to_string());
                        }

                        return Ok(format!(
                            "Submitted {} AddWorker jobs for external endpoints",
                            submitted_count
                        ));
                    }
                };

                debug!(
                    "Creating AddWorker jobs for {} workers from config",
                    workers.len()
                );

                // Process all workers with unified loop
                for (url, worker_type, bootstrap_port) in workers {
                    let url_for_error = url.clone(); // Clone for error message
                    let config = WorkerConfigRequest {
                        url,
                        api_key: api_key.clone(),
                        worker_type: Some(worker_type.to_string()),
                        labels: HashMap::new(),
                        model_id: None,
                        priority: None,
                        cost: None,
                        runtime: None,
                        tokenizer_path: None,
                        reasoning_parser: None,
                        tool_parser: None,
                        chat_template: router_config.chat_template.clone(),
                        bootstrap_port,
                        health_check_timeout_secs: router_config.health_check.timeout_secs,
                        health_check_interval_secs: router_config.health_check.check_interval_secs,
                        health_success_threshold: router_config.health_check.success_threshold,
                        health_failure_threshold: router_config.health_check.failure_threshold,
                        disable_health_check: router_config.health_check.disable_health_check,
                        max_connection_attempts: router_config.health_check.success_threshold * 10,
                        dp_aware: router_config.dp_aware,
                    };

                    let job = Job::AddWorker {
                        config: Box::new(config),
                    };

                    if let Some(queue) = context.worker_job_queue.get() {
                        queue.submit(job).await.map_err(|e| {
                            format!(
                                "Failed to submit AddWorker job for {} worker {}: {}",
                                worker_type, url_for_error, e
                            )
                        })?;
                        worker_count += 1;
                    } else {
                        return Err("JobQueue not available".to_string());
                    }
                }

                Ok(format!("Submitted {} AddWorker jobs", worker_count))
            }
            Job::InitializeMcpServers { mcp_config } => {
                let mut server_count = 0;

                debug!(
                    "Creating RegisterMcpServer jobs for {} MCP servers from config",
                    mcp_config.servers.len()
                );

                // Submit RegisterMcpServer jobs for each server in the config
                for server_config in &mcp_config.servers {
                    let mcp_server_request = McpServerConfigRequest {
                        name: server_config.name.clone(),
                        config: server_config.clone(),
                    };

                    let job = Job::RegisterMcpServer {
                        config: Box::new(mcp_server_request),
                    };

                    if let Some(queue) = context.worker_job_queue.get() {
                        queue.submit(job).await.map_err(|e| {
                            format!(
                                "Failed to submit RegisterMcpServer job for '{}': {}",
                                server_config.name, e
                            )
                        })?;
                        server_count += 1;
                    } else {
                        return Err("JobQueue not available".to_string());
                    }
                }

                Ok(format!("Submitted {} RegisterMcpServer jobs", server_count))
            }
            Job::RegisterMcpServer { config } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data =
                    create_mcp_workflow_data((**config).clone(), Arc::clone(context));

                let instance_id = engines
                    .mcp
                    .start_workflow(WorkflowId::new("mcp_registration"), workflow_data)
                    .await
                    .map_err(|e| format!("Failed to start MCP registration workflow: {:?}", e))?;

                debug!(
                    "Started MCP registration workflow for {} (instance: {})",
                    config.name, instance_id
                );

                let timeout_duration = Duration::from_secs(7200 + 30); // 2hr + margin

                engines
                    .mcp
                    .wait_for_completion(instance_id, &config.name, timeout_duration)
                    .await
            }
            Job::AddTokenizer { config } => {
                let engines = context
                    .workflow_engines
                    .get()
                    .ok_or_else(|| "Workflow engines not initialized".to_string())?;

                let workflow_data =
                    create_tokenizer_workflow_data(*config.clone(), Arc::clone(context));

                let instance_id = engines
                    .tokenizer
                    .start_workflow(WorkflowId::new("tokenizer_registration"), workflow_data)
                    .await
                    .map_err(|e| {
                        format!("Failed to start tokenizer registration workflow: {:?}", e)
                    })?;

                debug!(
                    "Started tokenizer registration workflow for '{}' id={} (instance: {})",
                    config.name, config.id, instance_id
                );

                // Allow up to 10 minutes for HuggingFace downloads
                let timeout_duration = Duration::from_secs(600);

                engines
                    .tokenizer
                    .wait_for_completion(instance_id, &config.id, timeout_duration)
                    .await
            }
            Job::RemoveTokenizer { request } => {
                // Tokenizer removal is synchronous and fast
                if let Some(entry) = context.tokenizer_registry.remove_by_id(&request.id) {
                    info!(
                        "Successfully removed tokenizer '{}' (id: {})",
                        entry.name, entry.id
                    );
                    Ok(format!("Tokenizer '{}' removed successfully", entry.name))
                } else {
                    Err(format!("Tokenizer with id '{}' not found", request.id))
                }
            }
        }
    }

    /// Update job status on completion
    fn record_job_completion(
        job_type: &'static str,
        worker_url: &str,
        _duration: Duration,
        result: &Result<String, String>,
        status_map: &Arc<DashMap<String, JobStatus>>,
    ) {
        match result {
            Ok(message) => {
                status_map.remove(worker_url);
                debug!(
                    "Completed job: type={}, worker={}, result={}",
                    job_type, worker_url, message
                );
            }
            Err(error) => {
                status_map.insert(
                    worker_url.to_string(),
                    JobStatus::failed(job_type, worker_url, error.clone()),
                );
                warn!(
                    "Failed job: type={}, worker={}, error={}",
                    job_type, worker_url, error
                );
            }
        }
    }

    /// Cleanup old job statuses (TTL 5 minutes)
    async fn cleanup_old_statuses(status_map: Arc<DashMap<String, JobStatus>>) {
        const CLEANUP_INTERVAL: Duration = Duration::from_secs(60); // Run every minute
        const STATUS_TTL: u64 = 300; // 5 minutes in seconds

        loop {
            tokio::time::sleep(CLEANUP_INTERVAL).await;

            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // Remove statuses older than TTL
            status_map.retain(|_key, value| now - value.timestamp < STATUS_TTL);

            debug!(
                "Cleaned up old job statuses, remaining: {}",
                status_map.len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_queue() -> Arc<JobQueue> {
        JobQueue::new(JobQueueConfig::default(), Weak::new())
    }

    #[tokio::test]
    async fn test_has_add_worker_in_flight_empty() {
        let queue = test_queue();
        assert!(!queue.has_add_worker_in_flight());
    }

    #[tokio::test]
    async fn test_has_add_worker_in_flight_pending_and_processing() {
        let queue = test_queue();
        queue.status_map.insert(
            "http://w1:8000".to_string(),
            JobStatus::pending("AddWorker", "http://w1:8000"),
        );
        assert!(queue.has_add_worker_in_flight());

        queue.status_map.insert(
            "http://w1:8000".to_string(),
            JobStatus::processing("AddWorker", "http://w1:8000"),
        );
        assert!(queue.has_add_worker_in_flight());
    }

    #[tokio::test]
    async fn test_has_add_worker_in_flight_ignores_terminal_and_other_jobs() {
        let queue = test_queue();
        // A successfully completed AddWorker is removed from the map entirely;
        // a failed one remains with a terminal status — neither is in flight.
        queue.status_map.insert(
            "http://w1:8000".to_string(),
            JobStatus::failed("AddWorker", "http://w1:8000", "boom".to_string()),
        );
        queue.status_map.insert(
            "http://w2:8000".to_string(),
            JobStatus::pending("RemoveWorker", "http://w2:8000"),
        );
        assert!(!queue.has_add_worker_in_flight());
    }
}
