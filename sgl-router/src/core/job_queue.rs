//! Async job queue for control plane operations
//!
//! Provides non-blocking worker management by queuing operations and processing
//! them asynchronously in background worker tasks.

use crate::core::WorkerManager;
use crate::protocols::worker_spec::{JobStatus, WorkerConfigRequest};
use crate::server::AppContext;
use dashmap::DashMap;
use metrics::{counter, gauge, histogram};
use std::sync::{Arc, Weak};
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Job types for control plane operations
#[derive(Debug, Clone)]
pub enum Job {
    AddWorker { config: Box<WorkerConfigRequest> },
    RemoveWorker { url: String },
}

impl Job {
    /// Get job type as string for logging
    pub fn job_type(&self) -> &str {
        match self {
            Job::AddWorker { .. } => "AddWorker",
            Job::RemoveWorker { .. } => "RemoveWorker",
        }
    }

    /// Get worker URL for logging
    pub fn worker_url(&self) -> &str {
        match self {
            Job::AddWorker { config } => &config.url,
            Job::RemoveWorker { url } => url,
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
    /// Number of worker tasks processing jobs
    pub worker_count: usize,
}

impl Default for JobQueueConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 1000,
            worker_count: 2,
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
}

impl std::fmt::Debug for JobQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobQueue")
            .field("status_count", &self.status_map.len())
            .finish()
    }
}

impl JobQueue {
    /// Create a new job queue with background workers (spawns tasks)
    ///
    /// Takes a Weak reference to AppContext to avoid circular strong references.
    /// Spawns background worker tasks that will process jobs asynchronously.
    pub fn new(config: JobQueueConfig, context: Weak<AppContext>) -> Arc<Self> {
        let (tx, rx) = mpsc::channel(config.queue_capacity);

        info!(
            "Initializing worker job queue: capacity={}, workers={}",
            config.queue_capacity, config.worker_count
        );

        let rx = Arc::new(tokio::sync::Mutex::new(rx));
        let status_map = Arc::new(DashMap::new());

        let queue = Arc::new(Self {
            tx,
            context: context.clone(),
            status_map: status_map.clone(),
        });

        for i in 0..config.worker_count {
            let rx = Arc::clone(&rx);
            let context = context.clone();
            let status_map = status_map.clone();

            tokio::spawn(async move {
                Self::worker_loop(i, rx, context, status_map).await;
            });
        }

        // Spawn cleanup task for old job statuses (TTL 5 minutes)
        let cleanup_status_map = status_map.clone();
        tokio::spawn(async move {
            Self::cleanup_old_statuses(cleanup_status_map).await;
        });

        queue
    }

    /// Submit a job
    pub async fn submit(&self, job: Job) -> Result<(), String> {
        // Check if context is still alive before accepting jobs
        if self.context.upgrade().is_none() {
            counter!("sgl_router_job_shutdown_rejected_total").increment(1);
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
                let queue_depth = self.tx.max_capacity() - self.tx.capacity();
                gauge!("sgl_router_job_queue_depth").set(queue_depth as f64);

                info!(
                    "Job submitted: type={}, worker={}, queue_depth={}",
                    job_type, worker_url, queue_depth
                );
                Ok(())
            }
            Err(_) => {
                counter!("sgl_router_job_queue_full_total").increment(1);
                // Remove status on failure
                self.status_map.remove(&worker_url);
                Err("Worker job queue full".to_string())
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

    /// Worker loop that processes jobs
    async fn worker_loop(
        worker_id: usize,
        rx: Arc<tokio::sync::Mutex<mpsc::Receiver<Job>>>,
        context: Weak<AppContext>,
        status_map: Arc<DashMap<String, JobStatus>>,
    ) {
        info!("Worker job queue worker {} started", worker_id);

        loop {
            // Lock the receiver and try to receive a job
            let job = {
                let mut rx_guard = rx.lock().await;
                rx_guard.recv().await
            };

            match job {
                Some(job) => {
                    let job_type = job.job_type().to_string();
                    let worker_url = job.worker_url().to_string();
                    let start = std::time::Instant::now();

                    // Update status to processing
                    status_map.insert(
                        worker_url.clone(),
                        JobStatus::processing(&job_type, &worker_url),
                    );

                    info!(
                        "Worker {} processing job: type={}, worker={}",
                        worker_id, job_type, worker_url
                    );

                    // Upgrade weak reference to process job
                    match context.upgrade() {
                        Some(ctx) => {
                            // Execute job
                            let result = Self::execute_job(&job, &ctx).await;
                            let duration = start.elapsed();

                            // Record metrics
                            histogram!("sgl_router_job_duration_seconds", "job_type" => job_type.clone())
                                .record(duration.as_secs_f64());

                            match result {
                                Ok(message) => {
                                    counter!("sgl_router_job_success_total", "job_type" => job_type.clone())
                                        .increment(1);
                                    // Remove status on success - worker in registry is sufficient
                                    status_map.remove(&worker_url);
                                    info!(
                                        "Worker {} completed job: type={}, worker={}, duration={:.3}s, result={}",
                                        worker_id, job_type, worker_url, duration.as_secs_f64(), message
                                    );
                                }
                                Err(error) => {
                                    counter!("sgl_router_job_failure_total", "job_type" => job_type.clone())
                                        .increment(1);
                                    // Keep failed status for API to report error details
                                    status_map.insert(
                                        worker_url.clone(),
                                        JobStatus::failed(&job_type, &worker_url, error.clone()),
                                    );
                                    warn!(
                                        "Worker {} failed job: type={}, worker={}, duration={:.3}s, error={}",
                                        worker_id, job_type, worker_url, duration.as_secs_f64(), error
                                    );
                                }
                            }
                        }
                        None => {
                            let error_msg = "AppContext dropped".to_string();
                            status_map.insert(
                                worker_url.clone(),
                                JobStatus::failed(&job_type, &worker_url, error_msg),
                            );
                            error!(
                                "Worker {}: AppContext dropped, cannot process job: type={}, worker={}",
                                worker_id, job_type, worker_url
                            );
                            break;
                        }
                    }
                }
                None => {
                    warn!(
                        "Worker job queue worker {} channel closed, stopping",
                        worker_id
                    );
                    break;
                }
            }
        }

        warn!("Worker job queue worker {} stopped", worker_id);
    }

    /// Execute a specific job
    async fn execute_job(job: &Job, context: &Arc<AppContext>) -> Result<String, String> {
        match job {
            Job::AddWorker { config } => {
                // Register worker with is_healthy=false
                let worker =
                    WorkerManager::add_worker_from_config(config.as_ref(), context).await?;

                // Validate and activate
                WorkerManager::validate_and_activate_worker(&worker, context).await
            }
            Job::RemoveWorker { url } => {
                let result = WorkerManager::remove_worker(url, context);
                // Clean up job status when removing worker
                if let Some(queue) = context.worker_job_queue.get() {
                    queue.remove_status(url);
                }
                result
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

    #[test]
    fn test_job_queue_config_default() {
        let config = JobQueueConfig::default();
        assert_eq!(config.queue_capacity, 1000);
        assert_eq!(config.worker_count, 2);
    }
}
