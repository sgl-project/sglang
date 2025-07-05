use super::{WorkerError, WorkerResult};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Core worker abstraction that represents a backend service
pub trait Worker: Send + Sync + fmt::Debug {
    /// Get the worker's URL
    fn url(&self) -> &str;

    /// Get the worker's type (Regular, Prefill, or Decode)
    fn worker_type(&self) -> WorkerType;

    /// Check if the worker is currently healthy
    fn is_healthy(&self) -> bool;

    /// Set the worker's health status
    fn set_healthy(&self, healthy: bool);

    /// Perform a health check on the worker (synchronous for dyn compatibility)
    fn check_health(&self) -> WorkerResult<()>;

    /// Get the current load (number of active requests)
    fn load(&self) -> usize;

    /// Increment the load counter
    fn increment_load(&self);

    /// Decrement the load counter
    fn decrement_load(&self);

    /// Get the number of processed requests
    fn processed_requests(&self) -> usize;

    /// Increment the processed requests counter
    fn increment_processed(&self);

    /// Get worker-specific metadata
    fn metadata(&self) -> &WorkerMetadata;

    /// Clone the worker (for trait objects)
    fn clone_worker(&self) -> Box<dyn Worker>;
}

/// Worker type classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorkerType {
    /// Regular worker for standard routing
    Regular,
    /// Prefill worker for PD disaggregated mode
    Prefill {
        /// Bootstrap port for communication with decode workers
        bootstrap_port: Option<u16>,
    },
    /// Decode worker for PD disaggregated mode
    Decode,
}

impl fmt::Display for WorkerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerType::Regular => write!(f, "Regular"),
            WorkerType::Prefill { bootstrap_port } => match bootstrap_port {
                Some(port) => write!(f, "Prefill(bootstrap:{})", port),
                None => write!(f, "Prefill"),
            },
            WorkerType::Decode => write!(f, "Decode"),
        }
    }
}

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Timeout for health checks in seconds
    pub timeout_secs: u64,
    /// Interval between health checks in seconds
    pub check_interval_secs: u64,
    /// Health check endpoint path
    pub endpoint: String,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 5,
            check_interval_secs: 30,
            endpoint: "/health".to_string(),
        }
    }
}

/// Metadata associated with a worker
#[derive(Debug, Clone)]
pub struct WorkerMetadata {
    /// Worker URL
    pub url: String,
    /// Worker type
    pub worker_type: WorkerType,
    /// Additional labels/tags
    pub labels: std::collections::HashMap<String, String>,
    /// Health check configuration
    pub health_config: HealthConfig,
}

/// Basic worker implementation
#[derive(Debug, Clone)]
pub struct BasicWorker {
    metadata: WorkerMetadata,
    load_counter: Arc<AtomicUsize>,
    processed_counter: Arc<AtomicUsize>,
    healthy: Arc<AtomicBool>,
}

impl BasicWorker {
    pub fn new(url: String, worker_type: WorkerType) -> Self {
        let metadata = WorkerMetadata {
            url: url.clone(),
            worker_type,
            labels: std::collections::HashMap::new(),
            health_config: HealthConfig::default(),
        };

        Self {
            metadata,
            load_counter: Arc::new(AtomicUsize::new(0)),
            processed_counter: Arc::new(AtomicUsize::new(0)),
            healthy: Arc::new(AtomicBool::new(true)),
        }
    }

    pub fn with_labels(mut self, labels: std::collections::HashMap<String, String>) -> Self {
        self.metadata.labels = labels;
        self
    }

    pub fn with_health_config(mut self, config: HealthConfig) -> Self {
        self.metadata.health_config = config;
        self
    }
}

impl Worker for BasicWorker {
    fn url(&self) -> &str {
        &self.metadata.url
    }

    fn worker_type(&self) -> WorkerType {
        self.metadata.worker_type.clone()
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Release);
    }

    fn check_health(&self) -> WorkerResult<()> {
        // This will be implemented with actual HTTP health check logic
        // For now, just return the current health status
        if self.is_healthy() {
            Ok(())
        } else {
            Err(WorkerError::HealthCheckFailed {
                url: self.url().to_string(),
                reason: "Worker marked as unhealthy".to_string(),
            })
        }
    }

    fn load(&self) -> usize {
        self.load_counter.load(Ordering::Relaxed)
    }

    fn increment_load(&self) {
        self.load_counter.fetch_add(1, Ordering::Relaxed);
    }

    fn decrement_load(&self) {
        self.load_counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(1)
            })
            .ok();
    }

    fn processed_requests(&self) -> usize {
        self.processed_counter.load(Ordering::Relaxed)
    }

    fn increment_processed(&self) {
        self.processed_counter.fetch_add(1, Ordering::Relaxed);
    }

    fn metadata(&self) -> &WorkerMetadata {
        &self.metadata
    }

    fn clone_worker(&self) -> Box<dyn Worker> {
        Box::new(self.clone())
    }
}

/// Worker factory for creating workers of different types
pub struct WorkerFactory;

impl WorkerFactory {
    /// Create a regular worker
    pub fn create_regular(url: String) -> Box<dyn Worker> {
        Box::new(BasicWorker::new(url, WorkerType::Regular))
    }

    /// Create a prefill worker with optional bootstrap port
    pub fn create_prefill(url: String, bootstrap_port: Option<u16>) -> Box<dyn Worker> {
        Box::new(BasicWorker::new(
            url,
            WorkerType::Prefill { bootstrap_port },
        ))
    }

    /// Create a decode worker
    pub fn create_decode(url: String) -> Box<dyn Worker> {
        Box::new(BasicWorker::new(url, WorkerType::Decode))
    }

    /// Create workers from URLs with automatic type detection
    pub fn create_from_urls(
        regular_urls: Vec<String>,
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
    ) -> (
        Vec<Box<dyn Worker>>,
        Vec<Box<dyn Worker>>,
        Vec<Box<dyn Worker>>,
    ) {
        let regular_workers: Vec<Box<dyn Worker>> =
            regular_urls.into_iter().map(Self::create_regular).collect();

        let prefill_workers: Vec<Box<dyn Worker>> = prefill_urls
            .into_iter()
            .map(|(url, port)| Self::create_prefill(url, port))
            .collect();

        let decode_workers: Vec<Box<dyn Worker>> =
            decode_urls.into_iter().map(Self::create_decode).collect();

        (regular_workers, prefill_workers, decode_workers)
    }
}

/// Helper trait for collections of workers
pub trait WorkerCollection {
    fn healthy_workers(&self) -> Vec<&dyn Worker>;
    fn total_load(&self) -> usize;
    fn find_worker(&self, url: &str) -> Option<&dyn Worker>;
    fn find_worker_mut(&mut self, url: &str) -> Option<&mut Box<dyn Worker>>;
}

impl WorkerCollection for Vec<Box<dyn Worker>> {
    fn healthy_workers(&self) -> Vec<&dyn Worker> {
        self.iter()
            .filter(|w| w.is_healthy())
            .map(|w| w.as_ref())
            .collect()
    }

    fn total_load(&self) -> usize {
        self.iter().map(|w| w.load()).sum()
    }

    fn find_worker(&self, url: &str) -> Option<&dyn Worker> {
        self.iter().find(|w| w.url() == url).map(|w| w.as_ref())
    }

    fn find_worker_mut(&mut self, url: &str) -> Option<&mut Box<dyn Worker>> {
        self.iter_mut().find(|w| w.url() == url)
    }
}

/// Convert a list of worker URLs to worker trait objects
pub fn urls_to_workers(urls: Vec<String>) -> Vec<Box<dyn Worker>> {
    urls.into_iter()
        .map(WorkerFactory::create_regular)
        .collect()
}

/// Convert worker trait objects back to URLs
pub fn workers_to_urls(workers: &[Box<dyn Worker>]) -> Vec<String> {
    workers.iter().map(|w| w.url().to_string()).collect()
}
