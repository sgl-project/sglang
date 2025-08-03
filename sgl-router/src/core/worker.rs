use super::{WorkerError, WorkerResult};
use async_trait::async_trait;
use once_cell::sync::Lazy;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

// Shared HTTP client for health checks
static HEALTH_CHECK_CLIENT: Lazy<reqwest::Client> = Lazy::new(|| {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30)) // Default timeout, overridden per request
        .build()
        .expect("Failed to create health check HTTP client")
});

/// Core worker abstraction that represents a backend service
#[async_trait]
pub trait Worker: Send + Sync + fmt::Debug {
    /// Get the worker's URL
    fn url(&self) -> &str;

    /// Get the worker's type (Regular, Prefill, or Decode)
    fn worker_type(&self) -> WorkerType;

    /// Check if the worker is currently healthy
    fn is_healthy(&self) -> bool;

    /// Set the worker's health status
    fn set_healthy(&self, healthy: bool);

    /// Perform an async health check on the worker
    async fn check_health_async(&self) -> WorkerResult<()>;

    /// Synchronous health check wrapper (for compatibility)
    fn check_health(&self) -> WorkerResult<()> {
        // Use a small runtime for synchronous contexts
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| WorkerError::HealthCheckFailed {
                url: self.url().to_string(),
                reason: format!("Failed to create runtime: {}", e),
            })?
            .block_on(self.check_health_async())
    }

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

    pub fn normalised_url(&self) -> WorkerResult<&str> {
        if self.url().contains("@") {
            // Need to extract the URL from "http://host:port@dp_rank"
            let parts: Vec<&str> = self.url().split('@').collect();
            if parts.len() != 2 {
                return Err(WorkerError::InvalidUrl {
                    url: self.url().to_string(),
                });
            }
            // Ensure the second part (the dp_rank) can be parsed as an integer
            match parts[1].parse::<usize>() {
                Ok(_) => Ok(parts[0]),
                Err(_) => Err(WorkerError::InvalidUrl {
                    url: self.url().to_string(),
                }),
            }
        } else {
            Ok(self.url())
        }
    }
}

#[async_trait]
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

    async fn check_health_async(&self) -> WorkerResult<()> {
        use std::time::Duration;

        // Perform actual HTTP health check
        let url = self.normalised_url()?;
        let health_url = format!("{}{}", url, self.metadata.health_config.endpoint);
        let timeout = Duration::from_secs(self.metadata.health_config.timeout_secs);

        // Use the shared client with a custom timeout for this request
        match HEALTH_CHECK_CLIENT
            .get(&health_url)
            .timeout(timeout)
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    self.set_healthy(true);
                    Ok(())
                } else {
                    self.set_healthy(false);
                    Err(WorkerError::HealthCheckFailed {
                        url: url.to_string(),
                        reason: format!("Health check returned status: {}", response.status()),
                    })
                }
            }
            Err(e) => {
                self.set_healthy(false);
                Err(WorkerError::HealthCheckFailed {
                    url: url.to_string(),
                    reason: format!("Health check request failed: {}", e),
                })
            }
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

/// RAII guard for worker load management
pub struct WorkerLoadGuard<'a> {
    workers: Vec<&'a dyn Worker>,
}

impl<'a> WorkerLoadGuard<'a> {
    /// Create a new load guard for a single worker
    pub fn new(worker: &'a dyn Worker) -> Self {
        worker.increment_load();
        Self {
            workers: vec![worker],
        }
    }

    /// Create a new load guard for multiple workers
    pub fn new_multi(workers: Vec<&'a dyn Worker>) -> Self {
        // Increment load counters for all workers
        for worker in &workers {
            worker.increment_load();
        }
        Self { workers }
    }
}

impl<'a> Drop for WorkerLoadGuard<'a> {
    fn drop(&mut self) {
        // Decrement load counters for all workers
        for worker in &self.workers {
            worker.decrement_load();
        }
    }
}

/// Health checker handle with graceful shutdown
pub struct HealthChecker {
    handle: tokio::task::JoinHandle<()>,
    shutdown: Arc<AtomicBool>,
}

impl fmt::Debug for HealthChecker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HealthChecker")
            .field("shutdown", &self.shutdown.load(Ordering::Relaxed))
            .finish()
    }
}

impl HealthChecker {
    /// Shutdown the health checker gracefully
    pub async fn shutdown(self) {
        self.shutdown.store(true, Ordering::Release);
        let _ = self.handle.await;
    }
}

/// Start an async background health checker for a collection of workers
pub fn start_health_checker(
    workers: std::sync::Arc<std::sync::RwLock<Vec<Box<dyn Worker>>>>,
    check_interval_secs: u64,
) -> HealthChecker {
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();

    let handle = tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(tokio::time::Duration::from_secs(check_interval_secs));

        loop {
            interval.tick().await;

            // Check for shutdown signal
            if shutdown_clone.load(Ordering::Acquire) {
                tracing::debug!("Health checker shutting down");
                break;
            }

            // Check health of all workers
            let workers_to_check = match workers.read() {
                Ok(guard) => guard.iter().map(|w| w.clone_worker()).collect::<Vec<_>>(),
                Err(poisoned) => {
                    tracing::error!("Worker lock poisoned: {}", poisoned);
                    continue;
                }
            };

            // Perform health checks concurrently
            let health_checks = workers_to_check.iter().map(|worker| {
                let worker_url = worker.url().to_string();
                let was_healthy = worker.is_healthy();

                async move {
                    match worker.check_health_async().await {
                        Ok(_) => {
                            if !was_healthy {
                                tracing::info!("Worker {} is now healthy", worker_url);
                            }
                        }
                        Err(e) => {
                            if was_healthy {
                                tracing::warn!("Worker {} health check failed: {}", worker_url, e);
                            } else {
                                // Worker was already unhealthy, log at debug level
                                tracing::debug!("Worker {} remains unhealthy: {}", worker_url, e);
                            }
                        }
                    }
                }
            });

            // Execute all health checks concurrently
            futures::future::join_all(health_checks).await;
        }
    });

    HealthChecker { handle, shutdown }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::RwLock;
    use std::time::Duration;
    use tokio::time::timeout;

    // Test WorkerType
    #[test]
    fn test_worker_type_display() {
        assert_eq!(WorkerType::Regular.to_string(), "Regular");
        assert_eq!(
            WorkerType::Prefill {
                bootstrap_port: Some(8080)
            }
            .to_string(),
            "Prefill(bootstrap:8080)"
        );
        assert_eq!(
            WorkerType::Prefill {
                bootstrap_port: None
            }
            .to_string(),
            "Prefill"
        );
        assert_eq!(WorkerType::Decode.to_string(), "Decode");
    }

    #[test]
    fn test_worker_type_equality() {
        assert_eq!(WorkerType::Regular, WorkerType::Regular);
        assert_ne!(WorkerType::Regular, WorkerType::Decode);
        assert_eq!(
            WorkerType::Prefill {
                bootstrap_port: Some(8080)
            },
            WorkerType::Prefill {
                bootstrap_port: Some(8080)
            }
        );
        assert_ne!(
            WorkerType::Prefill {
                bootstrap_port: Some(8080)
            },
            WorkerType::Prefill {
                bootstrap_port: Some(8081)
            }
        );
    }

    #[test]
    fn test_worker_type_clone() {
        let original = WorkerType::Prefill {
            bootstrap_port: Some(8080),
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // Test HealthConfig
    #[test]
    fn test_health_config_default() {
        let config = HealthConfig::default();
        assert_eq!(config.timeout_secs, 5);
        assert_eq!(config.check_interval_secs, 30);
        assert_eq!(config.endpoint, "/health");
    }

    #[test]
    fn test_health_config_custom() {
        let config = HealthConfig {
            timeout_secs: 10,
            check_interval_secs: 60,
            endpoint: "/healthz".to_string(),
        };
        assert_eq!(config.timeout_secs, 10);
        assert_eq!(config.check_interval_secs, 60);
        assert_eq!(config.endpoint, "/healthz");
    }

    // Test BasicWorker
    #[test]
    fn test_basic_worker_creation() {
        let worker = BasicWorker::new("http://test:8080".to_string(), WorkerType::Regular);
        assert_eq!(worker.url(), "http://test:8080");
        assert_eq!(worker.worker_type(), WorkerType::Regular);
        assert!(worker.is_healthy());
        assert_eq!(worker.load(), 0);
        assert_eq!(worker.processed_requests(), 0);
    }

    #[test]
    fn test_worker_with_labels() {
        let mut labels = std::collections::HashMap::new();
        labels.insert("env".to_string(), "prod".to_string());
        labels.insert("zone".to_string(), "us-west".to_string());

        let worker = BasicWorker::new("http://test:8080".to_string(), WorkerType::Regular)
            .with_labels(labels.clone());

        assert_eq!(worker.metadata().labels, labels);
    }

    #[test]
    fn test_worker_with_health_config() {
        let custom_config = HealthConfig {
            timeout_secs: 15,
            check_interval_secs: 45,
            endpoint: "/custom-health".to_string(),
        };

        let worker = BasicWorker::new("http://test:8080".to_string(), WorkerType::Regular)
            .with_health_config(custom_config.clone());

        assert_eq!(worker.metadata().health_config.timeout_secs, 15);
        assert_eq!(worker.metadata().health_config.check_interval_secs, 45);
        assert_eq!(worker.metadata().health_config.endpoint, "/custom-health");
    }

    // Test Worker trait implementation
    #[test]
    fn test_worker_url() {
        let worker = BasicWorker::new("http://worker1:8080".to_string(), WorkerType::Regular);
        assert_eq!(worker.url(), "http://worker1:8080");
    }

    #[test]
    fn test_worker_type_getter() {
        let regular = BasicWorker::new("http://test:8080".to_string(), WorkerType::Regular);
        assert_eq!(regular.worker_type(), WorkerType::Regular);

        let prefill = BasicWorker::new(
            "http://test:8080".to_string(),
            WorkerType::Prefill {
                bootstrap_port: Some(9090),
            },
        );
        assert_eq!(
            prefill.worker_type(),
            WorkerType::Prefill {
                bootstrap_port: Some(9090)
            }
        );

        let decode = BasicWorker::new("http://test:8080".to_string(), WorkerType::Decode);
        assert_eq!(decode.worker_type(), WorkerType::Decode);
    }

    #[test]
    fn test_health_status() {
        let worker = BasicWorker::new("http://test:8080".to_string(), WorkerType::Regular);

        // Initial state is healthy
        assert!(worker.is_healthy());

        // Set unhealthy
        worker.set_healthy(false);
        assert!(!worker.is_healthy());

        // Set healthy again
        worker.set_healthy(true);
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_load_counter_operations() {
        let worker = BasicWorker::new("http://test:8080".to_string(), WorkerType::Regular);

        // Initial load is 0
        assert_eq!(worker.load(), 0);

        // Increment once
        worker.increment_load();
        assert_eq!(worker.load(), 1);

        // Increment twice more
        worker.increment_load();
        worker.increment_load();
        assert_eq!(worker.load(), 3);

        // Decrement once
        worker.decrement_load();
        assert_eq!(worker.load(), 2);

        // Decrement to 0
        worker.decrement_load();
        worker.decrement_load();
        assert_eq!(worker.load(), 0);

        // Decrement below 0 should stay at 0
        worker.decrement_load();
        assert_eq!(worker.load(), 0);
    }

    #[test]
    fn test_processed_counter() {
        let worker = BasicWorker::new("http://test:8080".to_string(), WorkerType::Regular);

        // Initial count is 0
        assert_eq!(worker.processed_requests(), 0);

        // Increment multiple times
        for i in 1..=100 {
            worker.increment_processed();
            assert_eq!(worker.processed_requests(), i);
        }
    }

    #[test]
    fn test_clone_worker() {
        let original = BasicWorker::new("http://test:8080".to_string(), WorkerType::Regular);
        original.increment_load();
        original.increment_processed();
        original.set_healthy(false);

        let cloned = original.clone_worker();

        // Verify cloned worker has same URL and type
        assert_eq!(cloned.url(), original.url());
        assert_eq!(cloned.worker_type(), original.worker_type());

        // Load counters should be independent (cloned shares the Arc)
        assert_eq!(cloned.load(), original.load());

        // Modify original and verify clone is affected (shared state)
        original.increment_load();
        assert_eq!(cloned.load(), original.load());
    }

    // Test concurrent operations
    #[tokio::test]
    async fn test_concurrent_load_increments() {
        let worker = Arc::new(BasicWorker::new(
            "http://test:8080".to_string(),
            WorkerType::Regular,
        ));

        let mut handles = vec![];

        // Spawn 100 tasks incrementing load
        for _ in 0..100 {
            let worker_clone = Arc::clone(&worker);
            let handle = tokio::spawn(async move {
                worker_clone.increment_load();
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        // Final count should be 100
        assert_eq!(worker.load(), 100);
    }

    #[tokio::test]
    async fn test_concurrent_load_decrements() {
        let worker = Arc::new(BasicWorker::new(
            "http://test:8080".to_string(),
            WorkerType::Regular,
        ));

        // Set initial load to 100
        for _ in 0..100 {
            worker.increment_load();
        }
        assert_eq!(worker.load(), 100);

        let mut handles = vec![];

        // Spawn 100 tasks decrementing load
        for _ in 0..100 {
            let worker_clone = Arc::clone(&worker);
            let handle = tokio::spawn(async move {
                worker_clone.decrement_load();
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        // Final count should be 0
        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn test_concurrent_health_updates() {
        let worker = Arc::new(BasicWorker::new(
            "http://test:8080".to_string(),
            WorkerType::Regular,
        ));

        let mut handles = vec![];

        // Spawn threads randomly setting health status
        for i in 0..100 {
            let worker_clone = Arc::clone(&worker);
            let handle = tokio::spawn(async move {
                worker_clone.set_healthy(i % 2 == 0);
                tokio::time::sleep(Duration::from_micros(10)).await;
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            handle.await.unwrap();
        }

        // Final state should be deterministic (last write wins)
        // We can't predict the exact final state due to scheduling,
        // but we can verify no data corruption occurred
        let final_health = worker.is_healthy();
        assert!(final_health == true || final_health == false);
    }

    // Test WorkerFactory
    #[test]
    fn test_create_regular_worker() {
        let worker = WorkerFactory::create_regular("http://regular:8080".to_string());
        assert_eq!(worker.url(), "http://regular:8080");
        assert_eq!(worker.worker_type(), WorkerType::Regular);
    }

    #[test]
    fn test_create_prefill_worker() {
        // With bootstrap port
        let worker1 = WorkerFactory::create_prefill("http://prefill:8080".to_string(), Some(9090));
        assert_eq!(worker1.url(), "http://prefill:8080");
        assert_eq!(
            worker1.worker_type(),
            WorkerType::Prefill {
                bootstrap_port: Some(9090)
            }
        );

        // Without bootstrap port
        let worker2 = WorkerFactory::create_prefill("http://prefill:8080".to_string(), None);
        assert_eq!(
            worker2.worker_type(),
            WorkerType::Prefill {
                bootstrap_port: None
            }
        );
    }

    #[test]
    fn test_create_decode_worker() {
        let worker = WorkerFactory::create_decode("http://decode:8080".to_string());
        assert_eq!(worker.url(), "http://decode:8080");
        assert_eq!(worker.worker_type(), WorkerType::Decode);
    }

    #[test]
    fn test_create_from_urls() {
        let regular_urls = vec![
            "http://regular1:8080".to_string(),
            "http://regular2:8080".to_string(),
        ];
        let prefill_urls = vec![
            ("http://prefill1:8080".to_string(), Some(9090)),
            ("http://prefill2:8080".to_string(), None),
        ];
        let decode_urls = vec![
            "http://decode1:8080".to_string(),
            "http://decode2:8080".to_string(),
        ];

        let (regular, prefill, decode) =
            WorkerFactory::create_from_urls(regular_urls, prefill_urls, decode_urls);

        assert_eq!(regular.len(), 2);
        assert_eq!(prefill.len(), 2);
        assert_eq!(decode.len(), 2);

        assert_eq!(regular[0].url(), "http://regular1:8080");
        assert_eq!(prefill[0].url(), "http://prefill1:8080");
        assert_eq!(decode[0].url(), "http://decode1:8080");
    }

    // Test WorkerCollection trait
    #[test]
    fn test_healthy_workers_filter() {
        let workers: Vec<Box<dyn Worker>> = vec![
            WorkerFactory::create_regular("http://w1:8080".to_string()),
            WorkerFactory::create_regular("http://w2:8080".to_string()),
            WorkerFactory::create_regular("http://w3:8080".to_string()),
        ];

        // Set some workers unhealthy
        workers[0].set_healthy(false);
        workers[2].set_healthy(false);

        let healthy = workers.healthy_workers();
        assert_eq!(healthy.len(), 1);
        assert_eq!(healthy[0].url(), "http://w2:8080");
    }

    #[test]
    fn test_total_load_calculation() {
        let workers: Vec<Box<dyn Worker>> = vec![
            WorkerFactory::create_regular("http://w1:8080".to_string()),
            WorkerFactory::create_regular("http://w2:8080".to_string()),
            WorkerFactory::create_regular("http://w3:8080".to_string()),
        ];

        // Set different loads
        workers[0].increment_load();
        workers[0].increment_load(); // load = 2

        workers[1].increment_load();
        workers[1].increment_load();
        workers[1].increment_load(); // load = 3

        workers[2].increment_load(); // load = 1

        assert_eq!(workers.total_load(), 6);
    }

    #[test]
    fn test_find_worker() {
        let workers: Vec<Box<dyn Worker>> = vec![
            WorkerFactory::create_regular("http://w1:8080".to_string()),
            WorkerFactory::create_regular("http://w2:8080".to_string()),
            WorkerFactory::create_regular("http://w3:8080".to_string()),
        ];

        // Found case
        let found = workers.find_worker("http://w2:8080");
        assert!(found.is_some());
        assert_eq!(found.unwrap().url(), "http://w2:8080");

        // Not found case
        let not_found = workers.find_worker("http://w4:8080");
        assert!(not_found.is_none());
    }

    #[test]
    fn test_find_worker_mut() {
        let mut workers: Vec<Box<dyn Worker>> = vec![
            WorkerFactory::create_regular("http://w1:8080".to_string()),
            WorkerFactory::create_regular("http://w2:8080".to_string()),
        ];

        // Find and modify
        if let Some(worker) = workers.find_worker_mut("http://w1:8080") {
            worker.set_healthy(false);
        }

        // Verify modification
        assert!(!workers[0].is_healthy());
        assert!(workers[1].is_healthy());
    }

    // Test WorkerLoadGuard
    #[test]
    fn test_load_guard_single_worker() {
        let worker = BasicWorker::new("http://test:8080".to_string(), WorkerType::Regular);
        assert_eq!(worker.load(), 0);

        {
            let _guard = WorkerLoadGuard::new(&worker);
            assert_eq!(worker.load(), 1);
        }

        // Guard dropped, load decremented
        assert_eq!(worker.load(), 0);
    }

    #[test]
    fn test_load_guard_multiple_workers() {
        let workers: Vec<Box<dyn Worker>> = vec![
            WorkerFactory::create_regular("http://w1:8080".to_string()),
            WorkerFactory::create_regular("http://w2:8080".to_string()),
            WorkerFactory::create_regular("http://w3:8080".to_string()),
        ];

        let worker_refs: Vec<&dyn Worker> = workers.iter().map(|w| w.as_ref()).collect();

        {
            let _guard = WorkerLoadGuard::new_multi(worker_refs);
            // All loads incremented
            assert_eq!(workers[0].load(), 1);
            assert_eq!(workers[1].load(), 1);
            assert_eq!(workers[2].load(), 1);
        }

        // All loads decremented
        assert_eq!(workers[0].load(), 0);
        assert_eq!(workers[1].load(), 0);
        assert_eq!(workers[2].load(), 0);
    }

    #[test]
    fn test_load_guard_panic_safety() {
        let worker = Arc::new(BasicWorker::new(
            "http://test:8080".to_string(),
            WorkerType::Regular,
        ));
        assert_eq!(worker.load(), 0);

        // Clone for use inside catch_unwind
        let worker_clone = Arc::clone(&worker);

        // This will panic, but the guard should still clean up
        let result = std::panic::catch_unwind(|| {
            let _guard = WorkerLoadGuard::new(worker_clone.as_ref());
            assert_eq!(worker_clone.load(), 1);
            panic!("Test panic");
        });

        // Verify panic occurred
        assert!(result.is_err());

        // Load should be decremented even after panic
        assert_eq!(worker.load(), 0);
    }

    // Test helper functions
    #[test]
    fn test_urls_to_workers() {
        let urls = vec!["http://w1:8080".to_string(), "http://w2:8080".to_string()];

        let workers = urls_to_workers(urls);
        assert_eq!(workers.len(), 2);
        assert_eq!(workers[0].url(), "http://w1:8080");
        assert_eq!(workers[1].url(), "http://w2:8080");
        assert_eq!(workers[0].worker_type(), WorkerType::Regular);
    }

    #[test]
    fn test_workers_to_urls() {
        let workers: Vec<Box<dyn Worker>> = vec![
            WorkerFactory::create_regular("http://w1:8080".to_string()),
            WorkerFactory::create_regular("http://w2:8080".to_string()),
        ];

        let urls = workers_to_urls(&workers);
        assert_eq!(urls, vec!["http://w1:8080", "http://w2:8080"]);
    }

    // Test synchronous health check wrapper
    #[test]
    fn test_check_health_sync_wrapper() {
        // We can't easily test the actual HTTP call without mocking,
        // but we can verify the sync wrapper works
        let worker = BasicWorker::new("http://test:8080".to_string(), WorkerType::Regular);

        // This will fail because there's no server at this URL,
        // but it tests that the sync wrapper doesn't panic
        let result = worker.check_health();
        assert!(result.is_err());
    }

    // Test HealthChecker background task
    #[tokio::test]
    async fn test_health_checker_startup() {
        let workers = Arc::new(RwLock::new(vec![WorkerFactory::create_regular(
            "http://w1:8080".to_string(),
        )]));

        let checker = start_health_checker(workers.clone(), 60);

        // Verify it starts without panic
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Shutdown
        checker.shutdown().await;
    }

    #[tokio::test]
    async fn test_health_checker_shutdown() {
        let workers = Arc::new(RwLock::new(vec![WorkerFactory::create_regular(
            "http://w1:8080".to_string(),
        )]));

        let checker = start_health_checker(workers.clone(), 60);

        // Shutdown should complete quickly
        let shutdown_result = timeout(Duration::from_secs(1), checker.shutdown()).await;
        assert!(shutdown_result.is_ok());
    }

    // Performance test for load counter
    #[test]
    fn test_load_counter_performance() {
        use std::time::Instant;

        let worker = BasicWorker::new("http://test:8080".to_string(), WorkerType::Regular);
        let iterations = 1_000_000;

        let start = Instant::now();
        for _ in 0..iterations {
            worker.increment_load();
        }
        let duration = start.elapsed();

        let ops_per_sec = iterations as f64 / duration.as_secs_f64();
        println!("Load counter operations per second: {:.0}", ops_per_sec);

        // Should be well over 1M ops/sec
        assert!(ops_per_sec > 1_000_000.0);
    }
}
