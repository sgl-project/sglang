//! Core worker trait and implementations for unified worker management

// Standard library imports
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// Third-party crate imports
use dyn_clone::DynClone;
use futures::future::{join_all, BoxFuture};
use once_cell::sync::Lazy;
use tokio::runtime::Runtime;
use tokio::time::timeout;
use tracing::{debug, error, info};

// Local crate imports
use super::error::WorkerError;
use crate::utils::api_path;

// ============================================================================
// CONSTANTS AND GLOBAL STATE
// ============================================================================

/// Global Tokio runtime for sync wrappers
static TOKIO_RT: Lazy<Runtime> =
    Lazy::new(|| Runtime::new().expect("Failed to create global Tokio runtime"));

/// Default TTL for health check caching
const DEFAULT_HEALTH_CHECK_CACHE_TTL: Duration = Duration::from_secs(30);

/// API endpoint constants
const HEALTH_ENDPOINT: &str = "/health";
const GET_LOAD_ENDPOINT: &str = "/get_load";
const FLUSH_CACHE_ENDPOINT: &str = "/flush_cache";
const SERVER_INFO_ENDPOINT: &str = "/get_server_info";
const MODELS_ENDPOINT: &str = "/v1/models";
const MODEL_INFO_ENDPOINT: &str = "/get_model_info";

// ============================================================================
// CORE TYPES AND TRAITS
// ============================================================================

/// Supported worker types in the routing system
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorkerType {
    /// Regular worker for standard routing
    Regular,
    /// Decode worker for disaggregated prefill-decode systems
    Decode,
    /// Prefill worker for disaggregated prefill-decode systems
    Prefill(Option<u16>),
}

impl fmt::Display for WorkerType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WorkerType::Regular => write!(f, "Regular"),
            WorkerType::Decode => write!(f, "Decode"),
            WorkerType::Prefill(port) => {
                if let Some(port) = port {
                    write!(f, "Prefill(bootstrap_port={})", port)
                } else {
                    write!(f, "Prefill")
                }
            }
        }
    }
}

/// Configuration for worker API endpoints
pub struct WorkerEndpoints {
    pub health: &'static str,
    pub load: &'static str,
    pub flush_cache: &'static str,
    pub server_info: &'static str,
    pub models: &'static str,
    pub model_info: &'static str,
}

impl WorkerType {
    pub fn get_endpoints(&self) -> WorkerEndpoints {
        match self {
            _ => WorkerEndpoints {
                health: HEALTH_ENDPOINT,
                load: GET_LOAD_ENDPOINT,
                flush_cache: FLUSH_CACHE_ENDPOINT,
                server_info: SERVER_INFO_ENDPOINT,
                models: MODELS_ENDPOINT,
                model_info: MODEL_INFO_ENDPOINT,
            },
        }
    }
}

/// Core worker trait defining the unified interface for all worker types
///
/// This trait provides a common interface for health checking, load management,
/// and other operations across different worker implementations (Regular, Decode, Prefill).
pub trait Worker: Send + Sync + DynClone + fmt::Display + fmt::Debug {
    /// Get the worker's URL
    fn url(&self) -> &str;

    /// Get the worker's type
    fn worker_type(&self) -> WorkerType;

    /// Check if the worker is currently healthy
    fn is_healthy(&self) -> bool;

    /// Perform a health check on the worker
    fn check_health(&self) -> BoxFuture<'_, Result<(), WorkerError>>;

    /// Get the current load counter for this worker
    fn load(&self) -> Arc<AtomicUsize>;

    /// Update the health status of the worker
    fn update_health(&self, healthy: bool);
}
dyn_clone::clone_trait_object!(Worker);

// ============================================================================
// COMMON IMPLEMENTATION INFRASTRUCTURE
// ============================================================================

/// Shared state and functionality for most worker implementations
#[derive(Debug, Clone)]
struct WorkerCommon {
    url: String,
    healthy: Arc<AtomicBool>,
    load: Arc<AtomicUsize>,
    last_health_check: Arc<RwLock<Instant>>,
    health_check_ttl: Duration,
}

impl WorkerCommon {
    fn new(url: String, health_check_ttl: Duration) -> Self {
        Self {
            url,
            // start with false to force health check
            healthy: Arc::new(AtomicBool::new(false)),
            load: Arc::new(AtomicUsize::new(0)),
            last_health_check: Arc::new(RwLock::new(Instant::now())),
            health_check_ttl,
        }
    }
}

macro_rules! common_worker_methods {
    () => {
        fn url(&self) -> &str {
            &self.common.url
        }

        fn is_healthy(&self) -> bool {
            self.common
                .healthy
                .load(std::sync::atomic::Ordering::Relaxed)
        }

        fn check_health(&self) -> futures::future::BoxFuture<'_, Result<(), WorkerError>> {
            let url = self.common.url.clone();
            let health_url = api_path(&self.common.url, self.worker_type().get_endpoints().health);
            let last_health_check = self.common.last_health_check.clone();
            let healthy = self.common.healthy.clone();
            Box::pin(async move {
                // We can serve from the cache if the worker was healthy last time and the cache
                // has not expired yet.
                let serve_from_cache = self.is_healthy() && {
                    if let Ok(lock) = last_health_check.read() {
                        lock.elapsed() < self.common.health_check_ttl
                    } else {
                        false // If we can't read lock, assume we need a check
                    }
                };

                if serve_from_cache {
                    info!("Health check serving from cache for worker: {}", self);
                    return Ok(());
                }

                // Perform actual health check
                let client = reqwest::Client::new();
                let result = client.get(health_url).send().await;
                // Update last check time
                if let Ok(mut lock) = last_health_check.write() {
                    *lock = Instant::now();
                }

                match result {
                    Ok(response) => {
                        let is_healthy = response.status().is_success();
                        healthy.store(is_healthy, Ordering::Relaxed);
                        if !is_healthy {
                            info!(
                                "Health check failed for {} with status: {}",
                                self,
                                response.status()
                            );
                            return Err(WorkerError::HealthCheckFailed {
                                url,
                                reason: format!(
                                    "Health check returned status: {}",
                                    response.status()
                                ),
                            });
                        }
                    }
                    Err(e) => {
                        info!("Health check failed for {} with error: {}", self, e);
                        return Err(WorkerError::HealthCheckFailed {
                            url,
                            reason: format!("Health check failed with error: {}", e),
                        });
                    }
                }

                Ok(())
            })
        }

        fn load(&self) -> std::sync::Arc<std::sync::atomic::AtomicUsize> {
            self.common.load.clone()
        }

        fn update_health(&self, healthy: bool) {
            self.common
                .healthy
                .store(healthy, std::sync::atomic::Ordering::Relaxed);
        }
    };
}

// ============================================================================
// CONCRETE WORKER IMPLEMENTATIONS
// ============================================================================

/// Regular worker implementation for standard routing scenarios
#[derive(Debug, Clone)]
pub struct RegularWorker {
    common: WorkerCommon,
}

impl RegularWorker {
    fn new(url: String, health_check_ttl: Duration) -> Self {
        Self {
            common: WorkerCommon::new(url, health_check_ttl),
        }
    }
}

impl fmt::Display for RegularWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegularWorker({})", self.common.url)
    }
}

impl Worker for RegularWorker {
    fn worker_type(&self) -> WorkerType {
        WorkerType::Regular
    }
    common_worker_methods!();
}

/// Decode worker implementation for disaggregated prefill-decode systems
#[derive(Debug, Clone)]
pub struct DecodeWorker {
    common: WorkerCommon,
}

impl DecodeWorker {
    fn new(url: String, health_check_ttl: Duration) -> Self {
        Self {
            common: WorkerCommon::new(url, health_check_ttl),
        }
    }
}

impl fmt::Display for DecodeWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DecodeWorker({})", self.common.url)
    }
}

impl Worker for DecodeWorker {
    fn worker_type(&self) -> WorkerType {
        WorkerType::Decode
    }
    common_worker_methods!();
}

/// Prefill worker implementation for disaggregated prefill-decode systems
///
/// Supports optional bootstrap port configuration for specialized routing.
#[derive(Debug, Clone)]
pub struct PrefillWorker {
    common: WorkerCommon,
    bootstrap_port: Option<u16>,
}

impl PrefillWorker {
    fn new(url: String, health_check_ttl: Duration, bootstrap_port: Option<u16>) -> Self {
        Self {
            common: WorkerCommon::new(url, health_check_ttl),
            bootstrap_port,
        }
    }
}

impl fmt::Display for PrefillWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.bootstrap_port {
            Some(port) => write!(
                f,
                "PrefillWorker({}, bootstrap_port={})",
                self.common.url, port
            ),
            None => write!(f, "PrefillWorker({})", self.common.url),
        }
    }
}

impl Worker for PrefillWorker {
    fn worker_type(&self) -> WorkerType {
        WorkerType::Prefill(self.bootstrap_port)
    }
    common_worker_methods!();
}

// ============================================================================
// FACTORY AND ADAPTER PATTERNS
// ============================================================================

/// Factory for creating worker instances with consistent configuration
pub struct WorkerFactory;

impl WorkerFactory {
    /// Create a new Regular worker with default health check TTL
    pub fn create_regular(url: String) -> Arc<dyn Worker> {
        Arc::new(RegularWorker::new(url, DEFAULT_HEALTH_CHECK_CACHE_TTL))
    }

    /// Create a new Prefill worker with optional bootstrap port and default health check TTL
    pub fn create_prefill(url: String, bootstrap_port: Option<u16>) -> Arc<dyn Worker> {
        Arc::new(PrefillWorker::new(
            url,
            DEFAULT_HEALTH_CHECK_CACHE_TTL,
            bootstrap_port,
        ))
    }

    /// Create a new Decode worker with default health check TTL
    pub fn create_decode(url: String) -> Arc<dyn Worker> {
        Arc::new(DecodeWorker::new(url, DEFAULT_HEALTH_CHECK_CACHE_TTL))
    }

    /// Create a new Regular worker with custom health check TTL (for testing)
    #[cfg(test)]
    pub fn create_regular_with_ttl(url: String, health_check_ttl: Duration) -> Arc<dyn Worker> {
        Arc::new(RegularWorker::new(url, health_check_ttl))
    }

    /// Create a new Prefill worker with custom health check TTL (for testing)
    #[cfg(test)]
    pub fn create_prefill_with_ttl(
        url: String,
        bootstrap_port: Option<u16>,
        health_check_ttl: Duration,
    ) -> Arc<dyn Worker> {
        Arc::new(PrefillWorker::new(url, health_check_ttl, bootstrap_port))
    }

    /// Create a new Decode worker with custom health check TTL (for testing)
    #[cfg(test)]
    pub fn create_decode_with_ttl(url: String, health_check_ttl: Duration) -> Arc<dyn Worker> {
        Arc::new(DecodeWorker::new(url, health_check_ttl))
    }
}

/// Adapter functions for batch worker creation from URL collections
pub(crate) mod worker_adapter {
    use super::*;

    /// Convert a vector of URLs into Regular workers
    pub fn from_regular_vec(urls: Vec<String>) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| WorkerFactory::create_regular(url.clone()))
            .collect()
    }

    /// Convert a vector of (URL, bootstrap_port) tuples into Prefill workers
    pub fn from_prefill_vec(urls: Vec<(String, Option<u16>)>) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|(url, bootstrap_port)| {
                WorkerFactory::create_prefill(url.clone(), *bootstrap_port)
            })
            .collect()
    }

    /// Convert a vector of URLs into Decode workers
    pub fn from_decode_vec(urls: Vec<String>) -> Vec<Arc<dyn Worker>> {
        urls.iter()
            .map(|url| WorkerFactory::create_decode(url.clone()))
            .collect()
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Utility functions for worker health checking and load management
pub(crate) mod utils {
    use super::*;

    /// Wait for all workers to become healthy (synchronous wrapper)
    ///
    /// This function blocks until all workers pass their health checks or the timeout is reached.
    ///
    /// # Arguments
    /// * `workers` - Slice of workers to check
    /// * `timeout_secs` - Maximum time to wait in seconds
    /// * `interval_secs` - Time between health check attempts in seconds
    pub(crate) fn wait_for_healthy_workers_sync(
        workers: &[Arc<dyn Worker>],
        timeout_secs: u64,
        interval_secs: u64,
    ) -> Result<(), String> {
        TOKIO_RT.block_on(wait_for_healthy_workers(
            workers,
            interval_secs,
            timeout_secs,
        ))
    }

    /// Wait for all workers to become healthy (asynchronous version)
    ///
    /// Continuously attempts health checks with specified intervals until all workers
    /// are healthy or the timeout is reached.
    ///
    /// # Arguments
    /// * `workers` - Slice of workers to check
    /// * `interval_secs` - Time between health check attempts in seconds
    /// * `timeout_secs` - Maximum time to wait in seconds
    pub(crate) async fn wait_for_healthy_workers(
        workers: &[Arc<dyn Worker>],
        interval_secs: u64,
        timeout_secs: u64,
    ) -> Result<(), String> {
        use tokio::time::sleep;
        timeout(Duration::from_secs(timeout_secs), async {
            loop {
                let health_futures: Vec<_> =
                    workers.iter().map(|worker| worker.check_health()).collect();
                let health_results = join_all(health_futures).await;

                let all_healthy = health_results.iter().all(|r| r.is_ok());

                if all_healthy {
                    info!("All workers are healthy");
                    return Ok(());
                } else {
                    info!("Health checking workers:");
                    for (worker, result) in workers.iter().zip(health_results.into_iter()) {
                        if let Err(err) = result {
                            info!("  {} - {}", worker, err);
                        }
                    }
                    sleep(Duration::from_secs(interval_secs)).await;
                }
            }
        })
        .await
        .unwrap_or_else(|_| {
            error!(
                "Timeout {}s waiting for workers {:?} to become healthy. Please set --router-worker-startup-timeout-secs or --worker-startup-timeout-secs to a larger value",
                timeout_secs, workers
            );
            Err(format!(
                "Timeout {}s waiting for workers {:?} to become healthy.",
                timeout_secs, workers
            ))
        })
    }

    /// Retrieve the current load from a worker
    ///
    /// Makes an HTTP request to the worker's load endpoint and parses the response.
    /// Returns None if the request fails or the response is invalid.
    ///
    /// # Arguments
    /// * `client` - HTTP client to use for the request
    /// * `worker` - Worker to query for load information
    pub(crate) async fn get_worker_load(
        client: &reqwest::Client,
        worker: &Arc<dyn Worker>,
    ) -> Option<isize> {
        let get_load_api = api_path(worker.url(), worker.worker_type().get_endpoints().load);

        match client.get(get_load_api).send().await {
            Ok(res) if res.status().is_success() => match res.bytes().await {
                Ok(bytes) => match serde_json::from_slice::<serde_json::Value>(&bytes) {
                    Ok(data) => data
                        .get("load")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as isize),
                    Err(e) => {
                        debug!("Failed to parse load response from {}: {}", worker, e);
                        None
                    }
                },
                Err(e) => {
                    debug!("Failed to read load response from {}: {}", worker, e);
                    None
                }
            },
            Ok(res) => {
                debug!(
                    "Worker {} returned non-success status: {}",
                    worker,
                    res.status()
                );
                None
            }
            Err(e) => {
                debug!("Failed to get load from {}: {}", worker, e);
                None
            }
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::OnceCell;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::time::{sleep, Duration};
    use tracing_subscriber;

    static TRACING_INIT: OnceCell<()> = OnceCell::new();

    #[ctor::ctor]
    fn init_tracing() {
        TRACING_INIT.get_or_init(|| {
            tracing_subscriber::fmt()
                .with_max_level(tracing::Level::DEBUG)
                .with_test_writer() // directs logs to test output
                .init();
        });
    }

    // Mock worker for testing that can simulate different health states
    #[derive(Debug, Clone)]
    struct MockWorker {
        url: String,
        healthy: Arc<AtomicBool>,
        load_counter: Arc<AtomicUsize>,
        should_fail_health_check: Arc<AtomicBool>,
    }

    impl MockWorker {
        fn new(url: String, initially_healthy: bool) -> Self {
            Self {
                url,
                healthy: Arc::new(AtomicBool::new(initially_healthy)),
                load_counter: Arc::new(AtomicUsize::new(0)),
                should_fail_health_check: Arc::new(AtomicBool::new(false)),
            }
        }

        fn set_should_fail_health_check(&self, should_fail: bool) {
            self.should_fail_health_check
                .store(should_fail, Ordering::Relaxed);
        }

        fn set_healthy(&self, healthy: bool) {
            self.healthy.store(healthy, Ordering::Relaxed);
        }
    }

    impl fmt::Display for MockWorker {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "MockWorker({})", self.url)
        }
    }

    impl Worker for MockWorker {
        fn url(&self) -> &str {
            &self.url
        }

        fn worker_type(&self) -> WorkerType {
            WorkerType::Regular
        }

        fn is_healthy(&self) -> bool {
            self.healthy.load(Ordering::Relaxed)
        }

        fn check_health(&self) -> BoxFuture<'_, Result<(), WorkerError>> {
            let url = self.url.clone();
            let healthy = self.healthy.clone();
            let should_fail = self.should_fail_health_check.clone();

            Box::pin(async move {
                // Simulate some async work
                sleep(Duration::from_millis(10)).await;

                if should_fail.load(Ordering::Relaxed) {
                    healthy.store(false, Ordering::Relaxed);
                    return Err(WorkerError::HealthCheckFailed {
                        url,
                        reason: "Mock health check failure".to_string(),
                    });
                }

                healthy.store(true, Ordering::Relaxed);
                Ok(())
            })
        }

        fn load(&self) -> Arc<AtomicUsize> {
            self.load_counter.clone()
        }

        fn update_health(&self, healthy: bool) {
            self.healthy.store(healthy, Ordering::Relaxed);
        }
    }

    #[test]
    fn test_regular_worker() {
        let worker = RegularWorker::new(
            "http://localhost:8080".to_string(),
            DEFAULT_HEALTH_CHECK_CACHE_TTL,
        );
        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), WorkerType::Regular);
        assert!(!worker.is_healthy()); // starts as false to force health check
    }

    #[test]
    fn test_prefill_worker() {
        let worker = DecodeWorker::new(
            "http://localhost:8080".to_string(),
            DEFAULT_HEALTH_CHECK_CACHE_TTL,
        );
        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), WorkerType::Decode);
        assert!(!worker.is_healthy()); // starts as false to force health check
    }

    #[test]
    fn test_decode_worker() {
        let worker = PrefillWorker::new(
            "http://localhost:8080".to_string(),
            DEFAULT_HEALTH_CHECK_CACHE_TTL,
            Some(9000),
        );
        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), WorkerType::Prefill(Some(9000)));
        assert!(!worker.is_healthy()); // starts as false to force health check
    }

    // Tests for utility functions
    mod utils_tests {
        use super::*;

        #[tokio::test]
        async fn test_wait_for_healthy_workers_async_success() {
            let workers: Vec<Arc<dyn Worker>> = vec![
                Arc::new(MockWorker::new("http://worker1:8080".to_string(), true)),
                Arc::new(MockWorker::new("http://worker2:8080".to_string(), true)),
            ];

            let result = utils::wait_for_healthy_workers(&workers, 1, 5).await;
            assert!(result.is_ok());
        }

        #[tokio::test]
        async fn test_wait_for_healthy_workers_async_timeout() {
            let mock_worker = MockWorker::new("http://worker1:8080".to_string(), false);
            mock_worker.set_should_fail_health_check(true);

            let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(mock_worker)];

            let result = utils::wait_for_healthy_workers(&workers, 1, 2).await;
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Timeout"));
        }

        #[tokio::test]
        async fn test_wait_for_healthy_workers_async_eventually_healthy() {
            let mock_worker = MockWorker::new("http://worker1:8080".to_string(), false);
            mock_worker.set_should_fail_health_check(true);

            let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(mock_worker.clone())];

            // Start a task that will make the worker healthy after some time
            let mock_worker_clone = mock_worker.clone();
            tokio::spawn(async move {
                sleep(Duration::from_millis(500)).await;
                mock_worker_clone.set_should_fail_health_check(false);
            });

            let result = utils::wait_for_healthy_workers(&workers, 1, 5).await;
            assert!(result.is_ok());
        }

        #[test]
        fn test_wait_for_healthy_workers_sync_success() {
            let workers: Vec<Arc<dyn Worker>> = vec![
                Arc::new(MockWorker::new("http://worker1:8080".to_string(), true)),
                Arc::new(MockWorker::new("http://worker2:8080".to_string(), true)),
            ];

            let result = utils::wait_for_healthy_workers_sync(&workers, 5, 1);
            assert!(result.is_ok());
        }

        #[test]
        fn test_wait_for_healthy_workers_sync_timeout() {
            let mock_worker = MockWorker::new("http://worker1:8080".to_string(), false);
            mock_worker.set_should_fail_health_check(true);

            let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(mock_worker)];

            let result = utils::wait_for_healthy_workers_sync(&workers, 2, 1);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Timeout"));
        }

        #[tokio::test]
        async fn test_get_worker_load_with_mock_server() {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            use tokio::net::TcpListener;

            // Start a mock HTTP server
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();

            // Spawn server task
            tokio::spawn(async move {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let mut buffer = [0; 1024];
                    let _ = stream.read(&mut buffer).await;

                    let json_body = r#"{"load": 42}"#;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        json_body.len(),
                        json_body
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                    // Explicitly shutdown the write side to signal end of response
                    let _ = stream.shutdown().await;
                }
            });

            // Give server time to start
            sleep(Duration::from_millis(100)).await;

            let mock_worker = MockWorker::new(format!("http://127.0.0.1:{}", addr.port()), true);
            let worker: Arc<dyn Worker> = Arc::new(mock_worker);
            let client = reqwest::Client::new();

            let load = utils::get_worker_load(&client, &worker).await;
            assert_eq!(load, Some(42));
        }

        #[tokio::test]
        async fn test_get_worker_load_connection_error() {
            // Use a non-existent port to simulate connection error
            let mock_worker = MockWorker::new("http://127.0.0.1:1".to_string(), true);
            let worker: Arc<dyn Worker> = Arc::new(mock_worker);
            let client = reqwest::Client::new();

            let load = utils::get_worker_load(&client, &worker).await;
            assert_eq!(load, None);
        }

        #[tokio::test]
        async fn test_get_worker_load_invalid_json() {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            use tokio::net::TcpListener;

            // Start a mock HTTP server that returns invalid JSON
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();

            tokio::spawn(async move {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let mut buffer = [0; 1024];
                    let _ = stream.read(&mut buffer).await;

                    let invalid_body = "invalid json";
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        invalid_body.len(),
                        invalid_body
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                    let _ = stream.shutdown().await;
                }
            });

            sleep(Duration::from_millis(100)).await;

            let mock_worker = MockWorker::new(format!("http://127.0.0.1:{}", addr.port()), true);
            let worker: Arc<dyn Worker> = Arc::new(mock_worker);
            let client = reqwest::Client::new();

            let load = utils::get_worker_load(&client, &worker).await;
            assert_eq!(load, None);
        }

        #[tokio::test]
        async fn test_get_worker_load_missing_load_field() {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            use tokio::net::TcpListener;

            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();

            tokio::spawn(async move {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let mut buffer = [0; 1024];
                    let _ = stream.read(&mut buffer).await;

                    let json_body = r#"{"other": 123}"#;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        json_body.len(),
                        json_body
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                    let _ = stream.shutdown().await;
                }
            });

            sleep(Duration::from_millis(100)).await;

            let mock_worker = MockWorker::new(format!("http://127.0.0.1:{}", addr.port()), true);
            let worker: Arc<dyn Worker> = Arc::new(mock_worker);
            let client = reqwest::Client::new();

            let load = utils::get_worker_load(&client, &worker).await;
            assert_eq!(load, None);
        }

        #[tokio::test]
        async fn test_get_worker_load_http_error() {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            use tokio::net::TcpListener;

            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();

            tokio::spawn(async move {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let mut buffer = [0; 1024];
                    let _ = stream.read(&mut buffer).await;

                    let response = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                    let _ = stream.shutdown().await;
                }
            });

            sleep(Duration::from_millis(100)).await;

            let mock_worker = MockWorker::new(format!("http://127.0.0.1:{}", addr.port()), true);
            let worker: Arc<dyn Worker> = Arc::new(mock_worker);
            let client = reqwest::Client::new();

            let load = utils::get_worker_load(&client, &worker).await;
            assert_eq!(load, None);
        }
    }

    // Tests for worker implementations
    mod worker_impl_tests {
        use super::*;

        #[test]
        fn test_regular_worker_creation() {
            let worker =
                RegularWorker::new("http://localhost:8080".to_string(), Duration::from_secs(60));
            assert_eq!(worker.url(), "http://localhost:8080");
            assert_eq!(worker.worker_type(), WorkerType::Regular);
            assert!(!worker.is_healthy()); // starts as false to force health check
            assert_eq!(worker.load().load(Ordering::Relaxed), 0);
        }

        #[test]
        fn test_decode_worker_creation() {
            let worker =
                DecodeWorker::new("http://localhost:9090".to_string(), Duration::from_secs(45));
            assert_eq!(worker.url(), "http://localhost:9090");
            assert_eq!(worker.worker_type(), WorkerType::Decode);
            assert!(!worker.is_healthy());
            assert_eq!(worker.load().load(Ordering::Relaxed), 0);
        }

        #[test]
        fn test_prefill_worker_creation_with_bootstrap_port() {
            let worker = PrefillWorker::new(
                "http://localhost:7070".to_string(),
                Duration::from_secs(30),
                Some(8888),
            );
            assert_eq!(worker.url(), "http://localhost:7070");
            assert_eq!(worker.worker_type(), WorkerType::Prefill(Some(8888)));
            assert!(!worker.is_healthy());
            assert_eq!(worker.load().load(Ordering::Relaxed), 0);
        }

        #[test]
        fn test_prefill_worker_creation_without_bootstrap_port() {
            let worker = PrefillWorker::new(
                "http://localhost:7070".to_string(),
                Duration::from_secs(30),
                None,
            );
            assert_eq!(worker.url(), "http://localhost:7070");
            assert_eq!(worker.worker_type(), WorkerType::Prefill(None));
            assert!(!worker.is_healthy());
            assert_eq!(worker.load().load(Ordering::Relaxed), 0);
        }

        #[test]
        fn test_worker_update_health() {
            let worker = RegularWorker::new(
                "http://localhost:8080".to_string(),
                DEFAULT_HEALTH_CHECK_CACHE_TTL,
            );

            // Initially unhealthy
            assert!(!worker.is_healthy());

            // Update to healthy
            worker.update_health(true);
            assert!(worker.is_healthy());

            // Update back to unhealthy
            worker.update_health(false);
            assert!(!worker.is_healthy());
        }

        #[test]
        fn test_worker_load_counter() {
            let worker = DecodeWorker::new(
                "http://localhost:8080".to_string(),
                DEFAULT_HEALTH_CHECK_CACHE_TTL,
            );

            let load = worker.load();
            assert_eq!(load.load(Ordering::Relaxed), 0);

            // Simulate load changes
            load.store(5, Ordering::Relaxed);
            assert_eq!(worker.load().load(Ordering::Relaxed), 5);

            load.fetch_add(3, Ordering::Relaxed);
            assert_eq!(worker.load().load(Ordering::Relaxed), 8);
        }

        #[test]
        fn test_worker_display_formatting() {
            let regular_worker = RegularWorker::new(
                "http://worker1:8080".to_string(),
                DEFAULT_HEALTH_CHECK_CACHE_TTL,
            );
            assert_eq!(
                format!("{}", regular_worker),
                "RegularWorker(http://worker1:8080)"
            );

            let decode_worker = DecodeWorker::new(
                "http://worker2:9090".to_string(),
                DEFAULT_HEALTH_CHECK_CACHE_TTL,
            );
            assert_eq!(
                format!("{}", decode_worker),
                "DecodeWorker(http://worker2:9090)"
            );

            let prefill_worker_with_port = PrefillWorker::new(
                "http://worker3:7070".to_string(),
                DEFAULT_HEALTH_CHECK_CACHE_TTL,
                Some(8888),
            );
            assert_eq!(
                format!("{}", prefill_worker_with_port),
                "PrefillWorker(http://worker3:7070, bootstrap_port=8888)"
            );

            let prefill_worker_without_port = PrefillWorker::new(
                "http://worker4:7070".to_string(),
                DEFAULT_HEALTH_CHECK_CACHE_TTL,
                None,
            );
            assert_eq!(
                format!("{}", prefill_worker_without_port),
                "PrefillWorker(http://worker4:7070)"
            );
        }

        #[test]
        fn test_worker_type_display() {
            assert_eq!(format!("{}", WorkerType::Regular), "Regular");
            assert_eq!(format!("{}", WorkerType::Decode), "Decode");
            assert_eq!(
                format!("{}", WorkerType::Prefill(Some(8888))),
                "Prefill(bootstrap_port=8888)"
            );
            assert_eq!(format!("{}", WorkerType::Prefill(None)), "Prefill");
        }

        #[test]
        fn test_worker_type_equality() {
            assert_eq!(WorkerType::Regular, WorkerType::Regular);
            assert_eq!(WorkerType::Decode, WorkerType::Decode);
            assert_eq!(
                WorkerType::Prefill(Some(8888)),
                WorkerType::Prefill(Some(8888))
            );
            assert_eq!(WorkerType::Prefill(None), WorkerType::Prefill(None));

            assert_ne!(WorkerType::Regular, WorkerType::Decode);
            assert_ne!(
                WorkerType::Prefill(Some(8888)),
                WorkerType::Prefill(Some(9999))
            );
            assert_ne!(WorkerType::Prefill(Some(8888)), WorkerType::Prefill(None));
        }

        #[test]
        fn test_worker_endpoints() {
            let regular_endpoints = WorkerType::Regular.get_endpoints();
            assert_eq!(regular_endpoints.health, "/health");
            assert_eq!(regular_endpoints.load, "/get_load");
            assert_eq!(regular_endpoints.flush_cache, "/flush_cache");
            assert_eq!(regular_endpoints.server_info, "/get_server_info");
            assert_eq!(regular_endpoints.models, "/v1/models");
            assert_eq!(regular_endpoints.model_info, "/get_model_info");

            let decode_endpoints = WorkerType::Decode.get_endpoints();
            assert_eq!(decode_endpoints.health, "/health");
            assert_eq!(decode_endpoints.load, "/get_load");

            let prefill_endpoints = WorkerType::Prefill(Some(8888)).get_endpoints();
            assert_eq!(prefill_endpoints.health, "/health");
            assert_eq!(prefill_endpoints.load, "/get_load");
        }

        #[tokio::test]
        async fn test_worker_health_check_success() {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            use tokio::net::TcpListener;

            // Start a mock HTTP server that returns 200 OK
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();

            tokio::spawn(async move {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let mut buffer = [0; 1024];
                    let _ = stream.read(&mut buffer).await;

                    let response =
                        "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                    let _ = stream.shutdown().await;
                }
            });

            sleep(Duration::from_millis(100)).await;

            let worker = RegularWorker::new(
                format!("http://127.0.0.1:{}", addr.port()),
                Duration::from_millis(100), // short TTL for testing
            );

            let result = worker.check_health().await;
            assert!(result.is_ok());
            assert!(worker.is_healthy());
        }

        #[tokio::test]
        async fn test_worker_health_check_failure() {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            use tokio::net::TcpListener;

            // Start a mock HTTP server that returns 500 error
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();

            tokio::spawn(async move {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let mut buffer = [0; 1024];
                    let _ = stream.read(&mut buffer).await;

                    let response = "HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                    let _ = stream.shutdown().await;
                }
            });

            sleep(Duration::from_millis(100)).await;

            let worker = DecodeWorker::new(
                format!("http://127.0.0.1:{}", addr.port()),
                Duration::from_millis(100),
            );

            let result = worker.check_health().await;
            assert!(result.is_err());
            assert!(!worker.is_healthy());

            if let Err(WorkerError::HealthCheckFailed { url, reason }) = result {
                assert!(url.contains(&addr.port().to_string()));
                assert!(reason.contains("Health check returned status: 500"));
            } else {
                panic!("Expected HealthCheckFailed error");
            }
        }

        #[tokio::test]
        async fn test_worker_health_check_connection_error() {
            // Use a non-existent port to simulate connection error
            let worker = PrefillWorker::new(
                "http://127.0.0.1:1".to_string(),
                Duration::from_millis(100),
                Some(8888),
            );

            let result = worker.check_health().await;
            assert!(result.is_err());
            assert!(!worker.is_healthy());

            if let Err(WorkerError::HealthCheckFailed { url, reason }) = result {
                assert_eq!(url, "http://127.0.0.1:1");
                assert!(reason.contains("Health check failed with error"));
            } else {
                panic!("Expected HealthCheckFailed error");
            }
        }

        #[tokio::test]
        async fn test_worker_health_check_caching() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            use std::sync::Arc;
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            use tokio::net::TcpListener;

            let request_count = Arc::new(AtomicUsize::new(0));
            let request_count_clone = request_count.clone();

            // Start a mock HTTP server that counts requests
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();

            tokio::spawn(async move {
                while let Ok((mut stream, _)) = listener.accept().await {
                    request_count_clone.fetch_add(1, Ordering::Relaxed);

                    let mut buffer = [0; 1024];
                    let _ = stream.read(&mut buffer).await;

                    let response =
                        "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                    let _ = stream.shutdown().await;
                }
            });

            sleep(Duration::from_millis(100)).await;

            let worker = RegularWorker::new(
                format!("http://127.0.0.1:{}", addr.port()),
                Duration::from_secs(1), // 1 second TTL
            );

            // First health check should make a request
            let result1 = worker.check_health().await;
            assert!(result1.is_ok());
            assert!(worker.is_healthy());

            // Second health check should use cache (no new request)
            let result2 = worker.check_health().await;
            assert!(result2.is_ok());
            assert!(worker.is_healthy());

            // Should have made only one request due to caching
            assert_eq!(request_count.load(Ordering::Relaxed), 1);
        }

        // Factory tests
        mod factory_tests {
            use super::*;

            #[test]
            fn test_worker_factory_create_regular() {
                let worker = WorkerFactory::create_regular("http://worker1:8080".to_string());
                assert_eq!(worker.url(), "http://worker1:8080");
                assert_eq!(worker.worker_type(), WorkerType::Regular);
                assert!(!worker.is_healthy());
            }

            #[test]
            fn test_worker_factory_create_decode() {
                let worker = WorkerFactory::create_decode("http://worker2:9090".to_string());
                assert_eq!(worker.url(), "http://worker2:9090");
                assert_eq!(worker.worker_type(), WorkerType::Decode);
                assert!(!worker.is_healthy());
            }

            #[test]
            fn test_worker_factory_create_prefill_with_port() {
                let worker =
                    WorkerFactory::create_prefill("http://worker3:7070".to_string(), Some(8888));
                assert_eq!(worker.url(), "http://worker3:7070");
                assert_eq!(worker.worker_type(), WorkerType::Prefill(Some(8888)));
                assert!(!worker.is_healthy());
            }

            #[test]
            fn test_worker_factory_create_prefill_without_port() {
                let worker = WorkerFactory::create_prefill("http://worker4:7070".to_string(), None);
                assert_eq!(worker.url(), "http://worker4:7070");
                assert_eq!(worker.worker_type(), WorkerType::Prefill(None));
                assert!(!worker.is_healthy());
            }
        }

        // Worker adapter tests
        mod adapter_tests {
            use super::*;

            #[test]
            fn test_from_regular_vec() {
                let urls = vec![
                    "http://worker1:8080".to_string(),
                    "http://worker2:8080".to_string(),
                    "http://worker3:8080".to_string(),
                ];

                let workers = worker_adapter::from_regular_vec(urls.clone());

                assert_eq!(workers.len(), 3);
                for (i, worker) in workers.iter().enumerate() {
                    assert_eq!(worker.url(), urls[i]);
                    assert_eq!(worker.worker_type(), WorkerType::Regular);
                    assert!(!worker.is_healthy());
                }
            }

            #[test]
            fn test_from_regular_vec_empty() {
                let workers = worker_adapter::from_regular_vec(vec![]);
                assert_eq!(workers.len(), 0);
            }

            #[test]
            fn test_from_decode_vec() {
                let urls = vec![
                    "http://decode1:9090".to_string(),
                    "http://decode2:9090".to_string(),
                ];

                let workers = worker_adapter::from_decode_vec(urls.clone());

                assert_eq!(workers.len(), 2);
                for (i, worker) in workers.iter().enumerate() {
                    assert_eq!(worker.url(), urls[i]);
                    assert_eq!(worker.worker_type(), WorkerType::Decode);
                    assert!(!worker.is_healthy());
                }
            }

            #[test]
            fn test_from_decode_vec_empty() {
                let workers = worker_adapter::from_decode_vec(vec![]);
                assert_eq!(workers.len(), 0);
            }

            #[test]
            fn test_from_prefill_vec() {
                let urls_with_ports = vec![
                    ("http://prefill1:7070".to_string(), Some(8888)),
                    ("http://prefill2:7070".to_string(), None),
                    ("http://prefill3:7070".to_string(), Some(9999)),
                ];

                let workers = worker_adapter::from_prefill_vec(urls_with_ports.clone());

                assert_eq!(workers.len(), 3);
                for (i, worker) in workers.iter().enumerate() {
                    assert_eq!(worker.url(), urls_with_ports[i].0);
                    assert_eq!(
                        worker.worker_type(),
                        WorkerType::Prefill(urls_with_ports[i].1)
                    );
                    assert!(!worker.is_healthy());
                }
            }

            #[test]
            fn test_from_prefill_vec_empty() {
                let workers = worker_adapter::from_prefill_vec(vec![]);
                assert_eq!(workers.len(), 0);
            }

            #[test]
            fn test_mixed_worker_types_in_collection() {
                let mut workers: Vec<Arc<dyn Worker>> = Vec::new();

                workers.push(WorkerFactory::create_regular(
                    "http://regular:8080".to_string(),
                ));
                workers.push(WorkerFactory::create_decode(
                    "http://decode:9090".to_string(),
                ));
                workers.push(WorkerFactory::create_prefill(
                    "http://prefill:7070".to_string(),
                    Some(8888),
                ));

                assert_eq!(workers.len(), 3);
                assert_eq!(workers[0].worker_type(), WorkerType::Regular);
                assert_eq!(workers[1].worker_type(), WorkerType::Decode);
                assert_eq!(workers[2].worker_type(), WorkerType::Prefill(Some(8888)));
            }
        }

        #[test]
        fn test_worker_debug_formatting() {
            let regular_worker = RegularWorker::new(
                "http://worker1:8080".to_string(),
                DEFAULT_HEALTH_CHECK_CACHE_TTL,
            );

            let debug_output = format!("{:?}", regular_worker);
            assert!(debug_output.contains("RegularWorker"));
            assert!(debug_output.contains("http://worker1:8080"));

            let decode_worker = DecodeWorker::new(
                "http://worker2:9090".to_string(),
                DEFAULT_HEALTH_CHECK_CACHE_TTL,
            );

            let debug_output = format!("{:?}", decode_worker);
            assert!(debug_output.contains("DecodeWorker"));
            assert!(debug_output.contains("http://worker2:9090"));
        }

        // Enhanced health check tests using shared mock infrastructure
        mod enhanced_health_tests {
            use super::*;
            use crate::test_utils::mock_servers::create_enhanced_mock_health_server;
            use std::sync::atomic::Ordering;
            use std::time::Instant;

            #[tokio::test]
            async fn test_worker_health_check_success_scenario() {
                let (mock_url, call_count) = create_enhanced_mock_health_server(
                    vec![(200, r#"{"status": "healthy"}"#.to_string())],
                    vec![Duration::from_millis(0)],
                    Some(5),
                )
                .await;

                // Test actual worker health check
                let worker = WorkerFactory::create_regular(mock_url);
                let health_result = worker.check_health().await;

                assert!(health_result.is_ok());
                assert!(worker.is_healthy());
                assert_eq!(call_count.load(Ordering::SeqCst), 1);
            }

            #[tokio::test]
            async fn test_worker_health_check_error_then_success() {
                let (mock_url, call_count) = create_enhanced_mock_health_server(
                    vec![
                        (
                            503,
                            r#"{"status": "unhealthy", "error": "initializing"}"#.to_string(),
                        ),
                        (
                            503,
                            r#"{"status": "unhealthy", "error": "loading model"}"#.to_string(),
                        ),
                        (200, r#"{"status": "healthy"}"#.to_string()),
                    ],
                    vec![Duration::from_millis(0)],
                    Some(10),
                )
                .await;

                let worker = WorkerFactory::create_regular(mock_url);

                // First health check should fail
                let result1 = worker.check_health().await;
                assert!(result1.is_err());
                assert!(!worker.is_healthy());

                // Second health check should fail
                let result2 = worker.check_health().await;
                assert!(result2.is_err());
                assert!(!worker.is_healthy());

                // Third health check should succeed
                let result3 = worker.check_health().await;
                assert!(result3.is_ok());
                assert!(worker.is_healthy());

                assert_eq!(call_count.load(Ordering::SeqCst), 3);
            }

            #[tokio::test]
            async fn test_worker_health_check_slow_response() {
                let (mock_url, call_count) = create_enhanced_mock_health_server(
                    vec![(200, r#"{"status": "healthy"}"#.to_string())],
                    vec![Duration::from_millis(500)], // Slow response
                    Some(5),
                )
                .await;

                let worker = WorkerFactory::create_regular(mock_url);
                let start = Instant::now();
                let result = worker.check_health().await;
                let duration = start.elapsed();

                assert!(result.is_ok());
                assert!(worker.is_healthy());
                assert!(duration >= Duration::from_millis(400)); // Should take at least 400ms
                assert_eq!(call_count.load(Ordering::SeqCst), 1);
            }

            #[tokio::test]
            async fn test_worker_health_check_caching_with_mock() {
                let (mock_url, call_count) = create_enhanced_mock_health_server(
                    vec![(200, r#"{"status": "healthy"}"#.to_string())],
                    vec![Duration::from_millis(0)],
                    Some(10),
                )
                .await;

                let worker = WorkerFactory::create_regular(mock_url);

                // First health check should make an HTTP request
                let result1 = worker.check_health().await;
                assert!(result1.is_ok());
                assert!(worker.is_healthy());
                assert_eq!(call_count.load(Ordering::SeqCst), 1);

                // Second health check should use cache (no additional HTTP request)
                // since the default TTL is 5 seconds
                let result2 = worker.check_health().await;
                assert!(result2.is_ok());
                assert!(worker.is_healthy());
                assert_eq!(call_count.load(Ordering::SeqCst), 1); // Still 1, cached

                // Third health check should also use cache
                let result3 = worker.check_health().await;
                assert!(result3.is_ok());
                assert!(worker.is_healthy());
                assert_eq!(call_count.load(Ordering::SeqCst), 1); // Still 1, cached
            }

            #[tokio::test]
            async fn test_worker_health_check_different_worker_types() {
                // Create mock servers for different worker types
                let (regular_url, regular_count) = create_enhanced_mock_health_server(
                    vec![(200, r#"{"status": "healthy"}"#.to_string())],
                    vec![Duration::from_millis(0)],
                    Some(5),
                )
                .await;

                let (prefill_url, prefill_count) = create_enhanced_mock_health_server(
                    vec![(200, r#"{"status": "healthy"}"#.to_string())],
                    vec![Duration::from_millis(0)],
                    Some(5),
                )
                .await;

                let (decode_url, decode_count) = create_enhanced_mock_health_server(
                    vec![(200, r#"{"status": "healthy"}"#.to_string())],
                    vec![Duration::from_millis(0)],
                    Some(5),
                )
                .await;

                // Test different types of workers
                let regular_worker = WorkerFactory::create_regular(regular_url);
                let prefill_worker = WorkerFactory::create_prefill(prefill_url, Some(8081));
                let decode_worker = WorkerFactory::create_decode(decode_url);

                // Test health checks for all worker types
                let regular_result = regular_worker.check_health().await;
                let prefill_result = prefill_worker.check_health().await;
                let decode_result = decode_worker.check_health().await;

                assert!(regular_result.is_ok());
                assert!(prefill_result.is_ok());
                assert!(decode_result.is_ok());

                assert!(regular_worker.is_healthy());
                assert!(prefill_worker.is_healthy());
                assert!(decode_worker.is_healthy());

                // Verify worker types
                assert_eq!(regular_worker.worker_type(), WorkerType::Regular);
                assert_eq!(
                    prefill_worker.worker_type(),
                    WorkerType::Prefill(Some(8081))
                );
                assert_eq!(decode_worker.worker_type(), WorkerType::Decode);

                // All mock servers should have been called
                assert_eq!(regular_count.load(Ordering::SeqCst), 1);
                assert_eq!(prefill_count.load(Ordering::SeqCst), 1);
                assert_eq!(decode_count.load(Ordering::SeqCst), 1);
            }

            #[tokio::test]
            async fn test_wait_for_healthy_workers_timeout() {
                let (mock_url, call_count) = create_enhanced_mock_health_server(
                    vec![(503, r#"{"status": "unhealthy"}"#.to_string())],
                    vec![Duration::from_millis(0)],
                    Some(20),
                )
                .await;

                let workers = vec![WorkerFactory::create_regular(mock_url)];

                // This should timeout after 2 seconds
                let result = utils::wait_for_healthy_workers(
                    &workers, 1, // interval_secs
                    2, // timeout_secs
                )
                .await;

                assert!(result.is_err());
                assert!(result.unwrap_err().contains("Timeout"));
                // Should have made at least 2 health check attempts
                assert!(call_count.load(Ordering::SeqCst) >= 2);
            }

            #[tokio::test]
            async fn test_wait_for_healthy_workers_eventually_success() {
                let (mock_url, call_count) = create_enhanced_mock_health_server(
                    vec![
                        (503, r#"{"status": "initializing"}"#.to_string()),
                        (503, r#"{"status": "loading"}"#.to_string()),
                        (200, r#"{"status": "healthy"}"#.to_string()),
                    ],
                    vec![
                        Duration::from_millis(100), // First call slow
                        Duration::from_millis(50),  // Second call medium
                        Duration::from_millis(10),  // Third call fast
                    ],
                    Some(10),
                )
                .await;

                let workers = vec![WorkerFactory::create_regular(mock_url)];

                let start = Instant::now();
                let result = utils::wait_for_healthy_workers(
                    &workers, 1,  // interval_secs
                    10, // timeout_secs
                )
                .await;
                let duration = start.elapsed();

                assert!(result.is_ok());
                assert!(duration >= Duration::from_millis(150)); // Should take at least 160ms total
                assert_eq!(call_count.load(Ordering::SeqCst), 3);

                // Verify worker is now healthy
                assert!(workers[0].is_healthy());
            }

            #[tokio::test]
            async fn test_wait_for_healthy_workers_multiple_workers_mixed() {
                let (healthy_url, healthy_count) = create_enhanced_mock_health_server(
                    vec![(200, r#"{"status": "healthy"}"#.to_string())],
                    vec![Duration::from_millis(0)],
                    Some(5),
                )
                .await;

                let (unhealthy_url, unhealthy_count) = create_enhanced_mock_health_server(
                    vec![
                        (503, r#"{"status": "unhealthy"}"#.to_string()),
                        (503, r#"{"status": "unhealthy"}"#.to_string()),
                        (200, r#"{"status": "healthy"}"#.to_string()),
                    ],
                    vec![Duration::from_millis(0)],
                    Some(10),
                )
                .await;

                let workers = vec![
                    WorkerFactory::create_regular(healthy_url),
                    WorkerFactory::create_regular(unhealthy_url),
                ];

                let result = utils::wait_for_healthy_workers(
                    &workers, 1,  // interval_secs
                    10, // timeout_secs
                )
                .await;
                assert!(result.is_ok());

                // Verify both workers are now healthy
                assert!(workers[0].is_healthy());
                assert!(workers[1].is_healthy());

                // Both workers should have been checked
                assert!(healthy_count.load(Ordering::SeqCst) >= 1);
                assert!(unhealthy_count.load(Ordering::SeqCst) >= 3);
            }
        }
    }
}
