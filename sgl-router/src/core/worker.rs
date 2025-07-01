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
        self.healthy.load(Ordering::Relaxed)
    }

    fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Relaxed);
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
        self.load_counter.fetch_sub(1, Ordering::Relaxed);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_basic_worker_creation_and_metadata() {
        let worker = BasicWorker::new(
            "http://localhost:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: Some(9000),
            },
        );

        assert_eq!(worker.url(), "http://localhost:8000");
        assert_eq!(
            worker.worker_type(),
            WorkerType::Prefill {
                bootstrap_port: Some(9000)
            }
        );
        assert!(worker.is_healthy());
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.processed_requests(), 0);
    }

    #[test]
    fn test_worker_load_tracking() {
        let worker = BasicWorker::new("http://test".to_string(), WorkerType::Regular);

        // Test multiple increments and decrements
        worker.increment_load();
        worker.increment_load();
        worker.increment_load();
        assert_eq!(worker.load(), 3);

        worker.decrement_load();
        assert_eq!(worker.load(), 2);

        // Test processed requests counter
        worker.increment_processed();
        worker.increment_processed();
        assert_eq!(worker.processed_requests(), 2);
    }

    #[test]
    fn test_worker_health_management() {
        let worker = BasicWorker::new("http://test".to_string(), WorkerType::Decode);

        // Initially healthy
        assert!(worker.is_healthy());
        assert!(worker.check_health().is_ok());

        // Set unhealthy and verify behavior
        worker.set_healthy(false);
        assert!(!worker.is_healthy());

        let health_result = worker.check_health();
        assert!(health_result.is_err());
        if let Err(WorkerError::HealthCheckFailed { url, reason: _ }) = health_result {
            assert_eq!(url, "http://test");
        }

        // Restore health
        worker.set_healthy(true);
        assert!(worker.is_healthy());
        assert!(worker.check_health().is_ok());
    }

    #[test]
    fn test_worker_factory_different_types() {
        let regular = WorkerFactory::create_regular("http://regular".to_string());
        let prefill = WorkerFactory::create_prefill("http://prefill".to_string(), Some(8080));
        let decode = WorkerFactory::create_decode("http://decode".to_string());

        assert_eq!(regular.worker_type(), WorkerType::Regular);
        assert_eq!(
            prefill.worker_type(),
            WorkerType::Prefill {
                bootstrap_port: Some(8080)
            }
        );
        assert_eq!(decode.worker_type(), WorkerType::Decode);

        assert_eq!(regular.url(), "http://regular");
        assert_eq!(prefill.url(), "http://prefill");
        assert_eq!(decode.url(), "http://decode");
    }

    #[test]
    fn test_worker_factory_batch_creation() {
        let regular_urls = vec!["http://r1".to_string(), "http://r2".to_string()];
        let prefill_urls = vec![
            ("http://p1".to_string(), Some(8080)),
            ("http://p2".to_string(), None),
        ];
        let decode_urls = vec!["http://d1".to_string()];

        let (regular_workers, prefill_workers, decode_workers) =
            WorkerFactory::create_from_urls(regular_urls, prefill_urls, decode_urls);

        assert_eq!(regular_workers.len(), 2);
        assert_eq!(prefill_workers.len(), 2);
        assert_eq!(decode_workers.len(), 1);

        assert_eq!(regular_workers[0].url(), "http://r1");
        assert_eq!(
            prefill_workers[1].worker_type(),
            WorkerType::Prefill {
                bootstrap_port: None
            }
        );
        assert_eq!(decode_workers[0].worker_type(), WorkerType::Decode);
    }

    #[test]
    fn test_worker_collection_operations() {
        let mut workers: Vec<Box<dyn Worker>> = vec![
            WorkerFactory::create_regular("http://healthy1".to_string()),
            WorkerFactory::create_regular("http://unhealthy".to_string()),
            WorkerFactory::create_regular("http://healthy2".to_string()),
        ];

        // Set different health states
        workers[1].set_healthy(false);

        // Add some load
        workers[0].increment_load();
        workers[0].increment_load();
        workers[2].increment_load();

        // Test healthy_workers
        let healthy = workers.healthy_workers();
        assert_eq!(healthy.len(), 2);
        assert_eq!(healthy[0].url(), "http://healthy1");
        assert_eq!(healthy[1].url(), "http://healthy2");

        // Test total_load
        assert_eq!(workers.total_load(), 3);

        // Test find_worker
        let found = workers.find_worker("http://unhealthy");
        assert!(found.is_some());
        assert!(!found.unwrap().is_healthy());

        let not_found = workers.find_worker("http://nonexistent");
        assert!(not_found.is_none());

        // Test find_worker_mut
        let found_mut = workers.find_worker_mut("http://healthy1");
        assert!(found_mut.is_some());
    }

    #[test]
    fn test_worker_type_display() {
        assert_eq!(format!("{}", WorkerType::Regular), "Regular");
        assert_eq!(format!("{}", WorkerType::Decode), "Decode");
        assert_eq!(
            format!(
                "{}",
                WorkerType::Prefill {
                    bootstrap_port: Some(8080)
                }
            ),
            "Prefill(bootstrap:8080)"
        );
        assert_eq!(
            format!(
                "{}",
                WorkerType::Prefill {
                    bootstrap_port: None
                }
            ),
            "Prefill"
        );
    }

    #[test]
    fn test_worker_with_custom_configuration() {
        let mut labels = HashMap::new();
        labels.insert("env".to_string(), "production".to_string());
        labels.insert("region".to_string(), "us-west".to_string());

        let health_config = HealthConfig {
            timeout_secs: 10,
            check_interval_secs: 60,
            endpoint: "/custom-health".to_string(),
        };

        let worker = BasicWorker::new("http://configured".to_string(), WorkerType::Regular)
            .with_labels(labels.clone())
            .with_health_config(health_config.clone());

        let metadata = worker.metadata();
        assert_eq!(metadata.labels.get("env").unwrap(), "production");
        assert_eq!(metadata.labels.get("region").unwrap(), "us-west");
        assert_eq!(metadata.health_config.timeout_secs, 10);
        assert_eq!(metadata.health_config.endpoint, "/custom-health");
    }

    #[test]
    fn test_worker_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let worker = Arc::new(BasicWorker::new(
            "http://concurrent".to_string(),
            WorkerType::Regular,
        ));
        let num_threads = 10;
        let operations_per_thread = 100;

        let mut handles = vec![];

        // Spawn threads that concurrently increment load and processed counters
        for thread_id in 0..num_threads {
            let worker_clone = Arc::clone(&worker);
            let handle = thread::spawn(move || {
                for _ in 0..operations_per_thread {
                    worker_clone.increment_load();
                    worker_clone.increment_processed();

                    // Occasionally toggle health status to test concurrent health management
                    if thread_id % 3 == 0 {
                        worker_clone.set_healthy(false);
                        worker_clone.set_healthy(true);
                    }
                }

                // Decrement some load at the end
                for _ in 0..(operations_per_thread / 2) {
                    worker_clone.decrement_load();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        // Verify final state
        let expected_load =
            num_threads * operations_per_thread - num_threads * (operations_per_thread / 2);
        let expected_processed = num_threads * operations_per_thread;

        assert_eq!(worker.load(), expected_load);
        assert_eq!(worker.processed_requests(), expected_processed);
        assert!(worker.is_healthy()); // Should end up healthy

        // Verify basic functionality still works after concurrent access
        assert_eq!(worker.url(), "http://concurrent");
        assert_eq!(worker.worker_type(), WorkerType::Regular);
        assert!(worker.check_health().is_ok());
    }

    #[test]
    fn test_urls_conversion_functions() {
        let urls = vec![
            "http://worker1".to_string(),
            "http://worker2".to_string(),
            "http://worker3".to_string(),
        ];

        let workers = urls_to_workers(urls.clone());
        assert_eq!(workers.len(), 3);

        let converted_urls = workers_to_urls(&workers);
        assert_eq!(converted_urls, urls);

        // Verify all workers are Regular type
        for worker in &workers {
            assert_eq!(worker.worker_type(), WorkerType::Regular);
        }
    }
}
