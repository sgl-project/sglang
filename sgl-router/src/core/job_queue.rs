//! Async job queue for control plane operations
//!
//! Provides non-blocking worker management by queuing operations and processing
//! them asynchronously in background worker tasks.

use std::{
    collections::HashMap,
    sync::{Arc, Weak},
    time::{Duration, SystemTime},
};

use dashmap::DashMap;
use tokio::sync::{mpsc, Semaphore};
use tracing::{debug, error, info, warn};

use crate::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    core::workflow::{
        steps::{McpServerConfigRequest, WorkerRemovalRequest},
        WorkflowContext, WorkflowEngine, WorkflowId, WorkflowInstanceId, WorkflowStatus,
    },
    mcp::McpConfig,
    metrics::RouterMetrics,
    protocols::worker_spec::{JobStatus, WorkerConfigRequest},
};

/// Job types for control plane operations
#[derive(Debug, Clone)]
pub enum Job {
    AddWorker { config: Box<WorkerConfigRequest> },
    RemoveWorker { url: String },
    InitializeWorkersFromConfig { router_config: Box<RouterConfig> },
    InitializeMcpServers { mcp_config: Box<McpConfig> },
    RegisterMcpServer { config: Box<McpServerConfigRequest> },
}

impl Job {
    /// Get job type as string for logging
    pub fn job_type(&self) -> &str {
        match self {
            Job::AddWorker { .. } => "AddWorker",
            Job::RemoveWorker { .. } => "RemoveWorker",
            Job::InitializeWorkersFromConfig { .. } => "InitializeWorkersFromConfig",
            Job::InitializeMcpServers { .. } => "InitializeMcpServers",
            Job::RegisterMcpServer { .. } => "RegisterMcpServer",
        }
    }

    /// Get worker URL or MCP server name for logging
    pub fn worker_url(&self) -> &str {
        match self {
            Job::AddWorker { config } => &config.url,
            Job::RemoveWorker { url } => url,
            Job::InitializeWorkersFromConfig { .. } => "startup",
            Job::InitializeMcpServers { .. } => "startup",
            Job::RegisterMcpServer { config } => &config.name,
        }
    }
}

impl JobStatus {
    fn pending(job_type: &str, worker_url: &str) -> Self {
        Self {
            job_type: job_type.to_string(),
            worker_url: worker_url.to_string(),
            status: "pending".to_string(),
            message: None,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    fn processing(job_type: &str, worker_url: &str) -> Self {
        Self {
            job_type: job_type.to_string(),
            worker_url: worker_url.to_string(),
            status: "processing".to_string(),
            message: None,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    fn failed(job_type: &str, worker_url: &str, error: String) -> Self {
        Self {
            job_type: job_type.to_string(),
            worker_url: worker_url.to_string(),
            status: "failed".to_string(),
            message: Some(error),
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
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
            max_concurrent_jobs: 10,
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
            RouterMetrics::record_job_shutdown_rejected();
            return Err("Job queue shutting down: AppContext dropped".to_string());
        }

        // Extract values before moving job
        let job_type = job.job_type().to_string();
        let worker_url = job.worker_url().to_string();

        // Record pending status
        self.status_map.insert(
            worker_url.clone(),
            JobStatus::pending(&job_type, &worker_url),
        );

        match self.tx.send(job).await {
            Ok(_) => {
                let (queue_depth, available_permits) = self.get_load_info();
                RouterMetrics::set_job_queue_depth(queue_depth);
                debug!(
                    "Job submitted: type={}, worker={}, queue_depth={}, available_slots={}",
                    job_type, worker_url, queue_depth, available_permits
                );
                Ok(())
            }
            Err(_) => {
                RouterMetrics::record_job_queue_full();
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
        let job_type = job.job_type().to_string();
        let worker_url = job.worker_url().to_string();
        let start = std::time::Instant::now();

        // Update to processing
        status_map.insert(
            worker_url.clone(),
            JobStatus::processing(&job_type, &worker_url),
        );

        debug!("Processing job: type={}, worker={}", job_type, worker_url);

        // Execute job
        match context.upgrade() {
            Some(ctx) => {
                let result = Self::execute_job(&job, &ctx).await;
                let duration = start.elapsed();
                Self::record_job_completion(&job_type, &worker_url, duration, &result, &status_map);
            }
            None => {
                let error_msg = "AppContext dropped".to_string();
                status_map.insert(
                    worker_url.clone(),
                    JobStatus::failed(&job_type, &worker_url, error_msg),
                );
                error!(
                    "AppContext dropped, cannot process job: type={}, worker={}",
                    job_type, worker_url
                );
            }
        }

        // Permit automatically released when dropped
    }

    /// Execute a specific job
    async fn execute_job(job: &Job, context: &Arc<AppContext>) -> Result<String, String> {
        match job {
            Job::AddWorker { config } => {
                let engine = context
                    .workflow_engine
                    .get()
                    .ok_or_else(|| "Workflow engine not initialized".to_string())?;

                let instance_id = Self::start_worker_workflow(engine, config, context).await?;

                debug!(
                    "Started worker registration workflow for {} (instance: {})",
                    config.url, instance_id
                );

                let timeout_duration =
                    Duration::from_secs(context.router_config.worker_startup_timeout_secs + 30);

                Self::wait_for_workflow_completion(
                    engine,
                    instance_id,
                    &config.url,
                    timeout_duration,
                )
                .await
            }
            Job::RemoveWorker { url } => {
                let engine = context
                    .workflow_engine
                    .get()
                    .ok_or_else(|| "Workflow engine not initialized".to_string())?;

                let instance_id = Self::start_worker_removal_workflow(engine, url, context).await?;

                debug!(
                    "Started worker removal workflow for {} (instance: {})",
                    url, instance_id
                );

                let timeout_duration = Duration::from_secs(30);

                let result =
                    Self::wait_for_workflow_completion(engine, instance_id, url, timeout_duration)
                        .await;

                // Clean up job status when removing worker
                if let Some(queue) = context.worker_job_queue.get() {
                    queue.remove_status(url);
                }

                result
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
                    RoutingMode::OpenAI { .. } => {
                        info!("OpenAI mode: no workers to initialize");
                        return Ok("OpenAI mode: no workers to initialize".to_string());
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
                        chat_template: None,
                        bootstrap_port,
                        health_check_timeout_secs: router_config.health_check.timeout_secs,
                        health_check_interval_secs: router_config.health_check.check_interval_secs,
                        health_success_threshold: router_config.health_check.success_threshold,
                        health_failure_threshold: router_config.health_check.failure_threshold,
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
                let engine = context
                    .workflow_engine
                    .get()
                    .ok_or_else(|| "Workflow engine not initialized".to_string())?;

                let instance_id =
                    Self::start_mcp_registration_workflow(engine, config, context).await?;

                debug!(
                    "Started MCP registration workflow for {} (instance: {})",
                    config.name, instance_id
                );

                let timeout_duration = Duration::from_secs(7200 + 30); // 2hr + margin

                Self::wait_for_workflow_completion(
                    engine,
                    instance_id,
                    &config.name,
                    timeout_duration,
                )
                .await
            }
        }
    }

    /// Start a workflow and return its instance ID
    async fn start_worker_workflow(
        engine: &Arc<WorkflowEngine>,
        config: &WorkerConfigRequest,
        context: &Arc<AppContext>,
    ) -> Result<WorkflowInstanceId, String> {
        let mut workflow_context = WorkflowContext::new(WorkflowInstanceId::new());
        workflow_context.set("worker_config", config.clone());
        workflow_context.set_arc("app_context", Arc::clone(context));

        engine
            .start_workflow(WorkflowId::new("worker_registration"), workflow_context)
            .await
            .map_err(|e| format!("Failed to start worker registration workflow: {:?}", e))
    }

    /// Start worker removal workflow
    async fn start_worker_removal_workflow(
        engine: &Arc<WorkflowEngine>,
        url: &str,
        context: &Arc<AppContext>,
    ) -> Result<WorkflowInstanceId, String> {
        let removal_request = WorkerRemovalRequest {
            url: url.to_string(),
            dp_aware: context.router_config.dp_aware,
        };

        let mut workflow_context = WorkflowContext::new(WorkflowInstanceId::new());
        workflow_context.set("removal_request", removal_request);
        workflow_context.set_arc("app_context", Arc::clone(context));

        engine
            .start_workflow(WorkflowId::new("worker_removal"), workflow_context)
            .await
            .map_err(|e| format!("Failed to start worker removal workflow: {:?}", e))
    }

    /// Start MCP server registration workflow
    async fn start_mcp_registration_workflow(
        engine: &Arc<WorkflowEngine>,
        config: &McpServerConfigRequest,
        context: &Arc<AppContext>,
    ) -> Result<WorkflowInstanceId, String> {
        let mut workflow_context = WorkflowContext::new(WorkflowInstanceId::new());
        workflow_context.set("mcp_server_config", config.clone());
        workflow_context.set_arc("app_context", Arc::clone(context));

        engine
            .start_workflow(WorkflowId::new("mcp_registration"), workflow_context)
            .await
            .map_err(|e| format!("Failed to start MCP registration workflow: {:?}", e))
    }

    /// Wait for workflow completion with adaptive polling
    async fn wait_for_workflow_completion(
        engine: &Arc<WorkflowEngine>,
        instance_id: WorkflowInstanceId,
        worker_url: &str,
        timeout_duration: Duration,
    ) -> Result<String, String> {
        let start = std::time::Instant::now();
        let mut poll_interval = Duration::from_millis(100);
        let max_poll_interval = Duration::from_millis(2000);
        let poll_backoff = Duration::from_millis(200);

        loop {
            // Check timeout
            if start.elapsed() > timeout_duration {
                return Err(format!(
                    "Workflow timeout after {}s for worker {}",
                    timeout_duration.as_secs(),
                    worker_url
                ));
            }

            // Get workflow status
            let state = engine
                .get_status(instance_id)
                .map_err(|e| format!("Failed to get workflow status: {:?}", e))?;

            let result = match state.status {
                WorkflowStatus::Completed => Ok(format!(
                    "Worker {} registered and activated successfully via workflow",
                    worker_url
                )),
                WorkflowStatus::Failed => {
                    let current_step = state.current_step.as_ref();
                    let step_name = current_step
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    let error_msg = current_step
                        .and_then(|step_id| state.step_states.get(step_id))
                        .and_then(|s| s.last_error.as_deref())
                        .unwrap_or("Unknown error");
                    Err(format!(
                        "Workflow failed at step {}: {}",
                        step_name, error_msg
                    ))
                }
                WorkflowStatus::Cancelled => {
                    Err(format!("Workflow cancelled for worker {}", worker_url))
                }
                WorkflowStatus::Pending | WorkflowStatus::Paused | WorkflowStatus::Running => {
                    tokio::time::sleep(poll_interval).await;
                    poll_interval = (poll_interval + poll_backoff).min(max_poll_interval);
                    continue;
                }
            };

            // Clean up terminal workflow states
            engine.state_store().cleanup_if_terminal(instance_id);
            return result;
        }
    }

    /// Record job completion metrics and update status
    fn record_job_completion(
        job_type: &str,
        worker_url: &str,
        duration: Duration,
        result: &Result<String, String>,
        status_map: &Arc<DashMap<String, JobStatus>>,
    ) {
        RouterMetrics::record_job_duration(job_type, duration);

        match result {
            Ok(message) => {
                RouterMetrics::record_job_success(job_type);
                status_map.remove(worker_url);
                debug!(
                    "Completed job: type={}, worker={}, duration={:.3}s, result={}",
                    job_type,
                    worker_url,
                    duration.as_secs_f64(),
                    message
                );
            }
            Err(error) => {
                RouterMetrics::record_job_failure(job_type);
                status_map.insert(
                    worker_url.to_string(),
                    JobStatus::failed(job_type, worker_url, error.clone()),
                );
                warn!(
                    "Failed job: type={}, worker={}, duration={:.3}s, error={}",
                    job_type,
                    worker_url,
                    duration.as_secs_f64(),
                    error
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
