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
pub const DEFAULT_HEALTH_CHECK_CACHE_TTL: Duration = Duration::from_secs(30);

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
                    write!(f, "Prefill(bootstrap_port={port})")
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
        WorkerEndpoints {
            health: HEALTH_ENDPOINT,
            load: GET_LOAD_ENDPOINT,
            flush_cache: FLUSH_CACHE_ENDPOINT,
            server_info: SERVER_INFO_ENDPOINT,
            models: MODELS_ENDPOINT,
            model_info: MODEL_INFO_ENDPOINT,
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

    /// Create a new Regular worker with custom health check TTL
    pub fn create_regular_with_ttl(url: String, health_check_ttl: Duration) -> Arc<dyn Worker> {
        Arc::new(RegularWorker::new(url, health_check_ttl))
    }

    /// Create a new Prefill worker with custom health check TTL
    pub fn create_prefill_with_ttl(
        url: String,
        bootstrap_port: Option<u16>,
        health_check_ttl: Duration,
    ) -> Arc<dyn Worker> {
        Arc::new(PrefillWorker::new(url, health_check_ttl, bootstrap_port))
    }

    /// Create a new Decode worker with custom health check TTL
    pub fn create_decode_with_ttl(url: String, health_check_ttl: Duration) -> Arc<dyn Worker> {
        Arc::new(DecodeWorker::new(url, health_check_ttl))
    }
}

/// Adapter functions for batch worker creation from URL collections
pub mod worker_adapter {
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
pub mod utils {
    use super::*;

    /// Wait for all workers to become healthy (synchronous wrapper)
    ///
    /// This function blocks until all workers pass their health checks or the timeout is reached.
    ///
    /// # Arguments
    /// * `workers` - Slice of workers to check
    /// * `timeout_secs` - Maximum time to wait in seconds
    /// * `interval_secs` - Time between health check attempts in seconds
    pub fn wait_for_healthy_workers_sync(
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
    pub async fn wait_for_healthy_workers(
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
                "Timeout {timeout_secs}s waiting for workers {workers:?} to become healthy."
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
    pub async fn get_worker_load(
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
