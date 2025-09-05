use super::{CircuitBreaker, CircuitBreakerConfig, WorkerError, WorkerResult};
use crate::grpc::SglangSchedulerClient;
use crate::metrics::RouterMetrics;
use async_trait::async_trait;
use futures;
use serde_json;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};
use tokio::sync::Mutex;

// Shared HTTP client for worker operations (health checks, server info, etc.)
static WORKER_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30)) // Default timeout, overridden per request
        .build()
        .expect("Failed to create worker HTTP client")
});

/// Core worker abstraction that represents a backend service
#[async_trait]
pub trait Worker: Send + Sync + fmt::Debug {
    /// Get the worker's URL
    fn url(&self) -> &str;

    /// Get the worker's type (Regular, Prefill, or Decode)
    fn worker_type(&self) -> WorkerType;

    /// Get the worker's connection mode (HTTP or gRPC)
    fn connection_mode(&self) -> ConnectionMode;

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

    /// Reset the load counter to 0 (for sync/recovery)
    fn reset_load(&self) {
        // Default implementation - does nothing
        // Workers that track load should override this
    }

    /// Get the number of processed requests
    fn processed_requests(&self) -> usize;

    /// Increment the processed requests counter
    fn increment_processed(&self);

    /// Get worker-specific metadata
    fn metadata(&self) -> &WorkerMetadata;

    /// Clone the worker (for trait objects)
    fn clone_worker(&self) -> Box<dyn Worker>;

    /// Get the circuit breaker for this worker
    fn circuit_breaker(&self) -> &CircuitBreaker;

    /// Check if the worker is available (healthy + circuit closed/half-open)
    fn is_available(&self) -> bool {
        self.is_healthy() && self.circuit_breaker().can_execute()
    }

    /// Record the outcome of a request to this worker
    fn record_outcome(&self, success: bool) {
        // Record outcome-level metric with worker label
        let outcome_str = if success { "success" } else { "failure" };
        RouterMetrics::record_cb_outcome(self.url(), outcome_str);

        // Record into circuit breaker and infer state change for metrics
        let before = self.circuit_breaker().state();
        self.circuit_breaker().record_outcome(success);
        let after = self.circuit_breaker().state();

        if before != after {
            let from = match before {
                crate::core::CircuitState::Closed => "closed",
                crate::core::CircuitState::Open => "open",
                crate::core::CircuitState::HalfOpen => "half_open",
            };
            let to = match after {
                crate::core::CircuitState::Closed => "closed",
                crate::core::CircuitState::Open => "open",
                crate::core::CircuitState::HalfOpen => "half_open",
            };
            RouterMetrics::record_cb_state_transition(self.url(), from, to);
        }

        let state_code = match self.circuit_breaker().state() {
            crate::core::CircuitState::Closed => 0u8,
            crate::core::CircuitState::Open => 1u8,
            crate::core::CircuitState::HalfOpen => 2u8,
        };
        RouterMetrics::set_cb_state(self.url(), state_code);
    }

    // === DP-aware methods ===

    /// Check if this worker is DP-aware
    fn is_dp_aware(&self) -> bool {
        false
    }

    /// Get the base URL without any DP rank suffix
    fn base_url(&self) -> &str {
        self.url()
    }

    /// Get DP rank if this is a DP-aware worker
    fn dp_rank(&self) -> Option<usize> {
        None
    }

    /// Get DP size if this worker is part of a DP group
    fn dp_size(&self) -> Option<usize> {
        None
    }

    /// Transform a request for DP-aware routing
    async fn prepare_request(&self, req: serde_json::Value) -> WorkerResult<serde_json::Value> {
        Ok(req)
    }

    /// Get the actual endpoint URL for requests
    fn endpoint_url(&self, route: &str) -> String {
        format!("{}{}", self.base_url(), route)
    }

    /// Check if this worker can handle a specific request
    fn can_handle(&self, _req: &serde_json::Value) -> bool {
        true
    }
}

/// Connection mode for worker communication
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConnectionMode {
    /// HTTP/REST connection
    Http,
    /// gRPC connection
    Grpc {
        /// Optional port for gRPC endpoint (if different from URL)
        port: Option<u16>,
    },
}

impl fmt::Display for ConnectionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConnectionMode::Http => write!(f, "HTTP"),
            ConnectionMode::Grpc { port } => match port {
                Some(p) => write!(f, "gRPC(port:{})", p),
                None => write!(f, "gRPC"),
            },
        }
    }
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
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy
    pub success_threshold: u32,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 5,
            check_interval_secs: 30,
            endpoint: "/health".to_string(),
            failure_threshold: 3,
            success_threshold: 2,
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
    /// Connection mode
    pub connection_mode: ConnectionMode,
    /// Additional labels/tags
    pub labels: std::collections::HashMap<String, String>,
    /// Health check configuration
    pub health_config: HealthConfig,
}

/// Basic worker implementation
#[derive(Clone)]
pub struct BasicWorker {
    metadata: WorkerMetadata,
    load_counter: Arc<AtomicUsize>,
    processed_counter: Arc<AtomicUsize>,
    healthy: Arc<AtomicBool>,
    consecutive_failures: Arc<AtomicUsize>,
    consecutive_successes: Arc<AtomicUsize>,
    circuit_breaker: CircuitBreaker,
    /// Optional gRPC client for gRPC workers
    grpc_client: Option<Arc<Mutex<SglangSchedulerClient>>>,
}

impl fmt::Debug for BasicWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BasicWorker")
            .field("metadata", &self.metadata)
            .field("healthy", &self.healthy.load(Ordering::Relaxed))
            .field("circuit_breaker", &self.circuit_breaker)
            .field("has_grpc_client", &self.grpc_client.is_some())
            .finish()
    }
}

impl BasicWorker {
    pub fn new(url: String, worker_type: WorkerType) -> Self {
        Self::with_connection_mode(url, worker_type, ConnectionMode::Http)
    }

    pub fn with_connection_mode(
        url: String,
        worker_type: WorkerType,
        connection_mode: ConnectionMode,
    ) -> Self {
        let metadata = WorkerMetadata {
            url: url.clone(),
            worker_type,
            connection_mode,
            labels: std::collections::HashMap::new(),
            health_config: HealthConfig::default(),
        };

        Self {
            metadata,
            load_counter: Arc::new(AtomicUsize::new(0)),
            processed_counter: Arc::new(AtomicUsize::new(0)),
            healthy: Arc::new(AtomicBool::new(true)),
            consecutive_failures: Arc::new(AtomicUsize::new(0)),
            consecutive_successes: Arc::new(AtomicUsize::new(0)),
            circuit_breaker: CircuitBreaker::new(),
            grpc_client: None,
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

    pub fn with_circuit_breaker_config(mut self, config: CircuitBreakerConfig) -> Self {
        self.circuit_breaker = CircuitBreaker::with_config(config);
        self
    }

    /// Set the gRPC client for gRPC workers
    pub fn with_grpc_client(mut self, client: SglangSchedulerClient) -> Self {
        self.grpc_client = Some(Arc::new(Mutex::new(client)));
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

    fn connection_mode(&self) -> ConnectionMode {
        self.metadata.connection_mode.clone()
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Release);
        RouterMetrics::set_worker_health(self.url(), healthy);
    }

    async fn check_health_async(&self) -> WorkerResult<()> {
        use std::time::Duration;

        let health_result = match &self.metadata.connection_mode {
            ConnectionMode::Http => {
                // Perform HTTP health check
                let url = self.normalised_url()?;
                let health_url = format!("{}{}", url, self.metadata.health_config.endpoint);
                let timeout = Duration::from_secs(self.metadata.health_config.timeout_secs);

                // Use the shared client with a custom timeout for this request
                match WORKER_CLIENT.get(&health_url).timeout(timeout).send().await {
                    Ok(response) => response.status().is_success(),
                    Err(_) => false,
                }
            }
            ConnectionMode::Grpc { .. } => {
                // Perform gRPC health check
                if let Some(grpc_client) = &self.grpc_client {
                    let mut client = grpc_client.lock().await;
                    match client.health_check().await {
                        Ok(response) => {
                            tracing::debug!(
                                "gRPC health check succeeded for {}: healthy={}",
                                self.metadata.url,
                                response.healthy
                            );
                            response.healthy
                        }
                        Err(e) => {
                            tracing::warn!(
                                "gRPC health check RPC failed for {}: {:?}",
                                self.metadata.url,
                                e
                            );
                            false
                        }
                    }
                } else {
                    tracing::error!("No gRPC client available for worker {}", self.metadata.url);
                    false
                }
            }
        };

        if health_result {
            // Health check succeeded
            self.consecutive_failures.store(0, Ordering::Release);
            let successes = self.consecutive_successes.fetch_add(1, Ordering::AcqRel) + 1;

            // Mark healthy if we've reached the success threshold
            if !self.is_healthy()
                && successes >= self.metadata.health_config.success_threshold as usize
            {
                self.set_healthy(true);
                self.consecutive_successes.store(0, Ordering::Release);
            }
            Ok(())
        } else {
            // Health check failed
            self.consecutive_successes.store(0, Ordering::Release);
            let failures = self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;

            // Mark unhealthy if we've reached the failure threshold
            if self.is_healthy()
                && failures >= self.metadata.health_config.failure_threshold as usize
            {
                self.set_healthy(false);
                self.consecutive_failures.store(0, Ordering::Release);
            }

            Err(WorkerError::HealthCheckFailed {
                url: self.metadata.url.clone(),
                reason: format!("Health check failed (consecutive failures: {})", failures),
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

    fn reset_load(&self) {
        self.load_counter.store(0, Ordering::Relaxed);
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

    fn circuit_breaker(&self) -> &CircuitBreaker {
        &self.circuit_breaker
    }
}

/// A DP-aware worker that handles data-parallel routing
#[derive(Debug, Clone)]
pub struct DPAwareWorker {
    /// The underlying basic worker
    base_worker: BasicWorker,
    /// DP rank for this worker
    dp_rank: usize,
    /// Total DP size
    dp_size: usize,
    /// Base URL without DP suffix
    base_url: String,
}

impl DPAwareWorker {
    /// Create a new DP-aware worker of any type
    pub fn new(base_url: String, dp_rank: usize, dp_size: usize, worker_type: WorkerType) -> Self {
        // Create URL with DP rank suffix for identification
        let worker_url = format!("{}@{}", base_url, dp_rank);
        let base_worker = BasicWorker::new(worker_url, worker_type);

        Self {
            base_worker,
            dp_rank,
            dp_size,
            base_url,
        }
    }
}

#[async_trait]
impl Worker for DPAwareWorker {
    fn url(&self) -> &str {
        self.base_worker.url()
    }

    fn worker_type(&self) -> WorkerType {
        self.base_worker.worker_type()
    }

    fn connection_mode(&self) -> ConnectionMode {
        self.base_worker.connection_mode()
    }

    fn is_healthy(&self) -> bool {
        self.base_worker.is_healthy()
    }

    fn set_healthy(&self, healthy: bool) {
        self.base_worker.set_healthy(healthy);
    }

    async fn check_health_async(&self) -> WorkerResult<()> {
        // Delegate to the base worker's health check logic
        self.base_worker.check_health_async().await
    }

    fn load(&self) -> usize {
        self.base_worker.load()
    }

    fn increment_load(&self) {
        self.base_worker.increment_load();
    }

    fn decrement_load(&self) {
        self.base_worker.decrement_load();
    }

    fn reset_load(&self) {
        self.base_worker.reset_load();
    }

    fn processed_requests(&self) -> usize {
        self.base_worker.processed_requests()
    }

    fn increment_processed(&self) {
        self.base_worker.increment_processed();
    }

    fn metadata(&self) -> &WorkerMetadata {
        self.base_worker.metadata()
    }

    fn clone_worker(&self) -> Box<dyn Worker> {
        Box::new(self.clone())
    }

    fn circuit_breaker(&self) -> &CircuitBreaker {
        self.base_worker.circuit_breaker()
    }

    // DP-aware specific implementations

    fn is_dp_aware(&self) -> bool {
        true
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn dp_rank(&self) -> Option<usize> {
        Some(self.dp_rank)
    }

    fn dp_size(&self) -> Option<usize> {
        Some(self.dp_size)
    }

    async fn prepare_request(&self, mut req: serde_json::Value) -> WorkerResult<serde_json::Value> {
        // Inject data_parallel_rank into the request
        if let Some(map) = req.as_object_mut() {
            map.insert(
                "data_parallel_rank".to_string(),
                serde_json::json!(self.dp_rank),
            );
            Ok(req)
        } else {
            Err(WorkerError::InvalidConfiguration {
                message: "Request must be a JSON object for DP-aware routing".to_string(),
            })
        }
    }

    fn endpoint_url(&self, route: &str) -> String {
        // Use base URL for actual requests
        format!("{}{}", self.base_url, route)
    }
}

/// Worker factory for creating workers of different types
pub struct WorkerFactory;

impl WorkerFactory {
    /// Create a regular worker
    pub fn create_regular(url: String) -> Box<dyn Worker> {
        Box::new(BasicWorker::new(url, WorkerType::Regular))
    }

    /// Create a regular worker with custom circuit breaker configuration
    pub fn create_regular_with_config(
        url: String,
        circuit_breaker_config: CircuitBreakerConfig,
    ) -> Box<dyn Worker> {
        Box::new(
            BasicWorker::new(url, WorkerType::Regular)
                .with_circuit_breaker_config(circuit_breaker_config),
        )
    }

    /// Create a prefill worker with optional bootstrap port
    pub fn create_prefill(url: String, bootstrap_port: Option<u16>) -> Box<dyn Worker> {
        Box::new(BasicWorker::new(
            url,
            WorkerType::Prefill { bootstrap_port },
        ))
    }

    /// Create a prefill worker with custom circuit breaker configuration
    pub fn create_prefill_with_config(
        url: String,
        bootstrap_port: Option<u16>,
        circuit_breaker_config: CircuitBreakerConfig,
    ) -> Box<dyn Worker> {
        Box::new(
            BasicWorker::new(url, WorkerType::Prefill { bootstrap_port })
                .with_circuit_breaker_config(circuit_breaker_config),
        )
    }

    /// Create a decode worker
    pub fn create_decode(url: String) -> Box<dyn Worker> {
        Box::new(BasicWorker::new(url, WorkerType::Decode))
    }

    /// Create a decode worker with custom circuit breaker configuration
    pub fn create_decode_with_config(
        url: String,
        circuit_breaker_config: CircuitBreakerConfig,
    ) -> Box<dyn Worker> {
        Box::new(
            BasicWorker::new(url, WorkerType::Decode)
                .with_circuit_breaker_config(circuit_breaker_config),
        )
    }

    /// Create workers from URLs with automatic type detection
    #[allow(clippy::type_complexity)]
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

    /// Create a gRPC worker
    pub fn create_grpc(url: String, worker_type: WorkerType, port: Option<u16>) -> Box<dyn Worker> {
        Box::new(BasicWorker::with_connection_mode(
            url,
            worker_type,
            ConnectionMode::Grpc { port },
        ))
    }

    /// Create a gRPC worker with custom circuit breaker configuration
    pub fn create_grpc_with_config(
        url: String,
        worker_type: WorkerType,
        port: Option<u16>,
        circuit_breaker_config: CircuitBreakerConfig,
    ) -> Box<dyn Worker> {
        Box::new(
            BasicWorker::with_connection_mode(url, worker_type, ConnectionMode::Grpc { port })
                .with_circuit_breaker_config(circuit_breaker_config),
        )
    }

    /// Create a DP-aware worker of specified type
    pub fn create_dp_aware(
        base_url: String,
        dp_rank: usize,
        dp_size: usize,
        worker_type: WorkerType,
    ) -> Box<dyn Worker> {
        Box::new(DPAwareWorker::new(base_url, dp_rank, dp_size, worker_type))
    }

    /// Get DP size from a worker
    async fn get_worker_dp_size(url: &str, api_key: &Option<String>) -> WorkerResult<usize> {
        let mut req_builder = WORKER_CLIENT.get(format!("{}/get_server_info", url));

        if let Some(key) = api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        let response = req_builder
            .send()
            .await
            .map_err(|e| WorkerError::NetworkError {
                url: url.to_string(),
                error: e.to_string(),
            })?;

        if !response.status().is_success() {
            return Err(WorkerError::NetworkError {
                url: url.to_string(),
                error: format!("Server returned: {}", response.status()),
            });
        }

        let info: serde_json::Value =
            response
                .json()
                .await
                .map_err(|e| WorkerError::NetworkError {
                    url: url.to_string(),
                    error: format!("Failed to parse JSON: {}", e),
                })?;

        let dp_size = info
            .get("dp_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| WorkerError::InvalidConfiguration {
                message: "dp_size not found in server info".to_string(),
            })?;

        if dp_size > usize::MAX as u64 {
            return Err(WorkerError::InvalidConfiguration {
                message: format!("dp_size is too large: {}", dp_size),
            });
        }

        Ok(dp_size as usize)
    }

    /// Private helper to create DP-aware workers of any type
    async fn create_dp_aware_workers_of_type(
        url: &str,
        api_key: &Option<String>,
        worker_type: WorkerType,
    ) -> WorkerResult<Vec<Box<dyn Worker>>> {
        let dp_size = Self::get_worker_dp_size(url, api_key).await?;

        let workers = (0..dp_size)
            .map(|rank| Self::create_dp_aware(url.to_string(), rank, dp_size, worker_type.clone()))
            .collect();

        Ok(workers)
    }

    /// Create DP-aware regular workers from a single URL
    pub async fn create_dp_aware_regular_workers(
        url: &str,
        api_key: &Option<String>,
    ) -> WorkerResult<Vec<Box<dyn Worker>>> {
        Self::create_dp_aware_workers_of_type(url, api_key, WorkerType::Regular).await
    }

    /// Create DP-aware prefill workers from a single URL
    pub async fn create_dp_aware_prefill_workers(
        url: &str,
        bootstrap_port: Option<u16>,
        api_key: &Option<String>,
    ) -> WorkerResult<Vec<Box<dyn Worker>>> {
        Self::create_dp_aware_workers_of_type(url, api_key, WorkerType::Prefill { bootstrap_port })
            .await
    }

    /// Create DP-aware decode workers from a single URL
    pub async fn create_dp_aware_decode_workers(
        url: &str,
        api_key: &Option<String>,
    ) -> WorkerResult<Vec<Box<dyn Worker>>> {
        Self::create_dp_aware_workers_of_type(url, api_key, WorkerType::Decode).await
    }

    /// Create workers based on configuration (for regular router)
    pub async fn create_workers(
        urls: Vec<String>,
        dp_aware: bool,
        api_key: &Option<String>,
    ) -> WorkerResult<Vec<Box<dyn Worker>>> {
        if dp_aware {
            // Create futures for all worker creations
            let worker_futs = urls
                .iter()
                .map(|url| Self::create_dp_aware_regular_workers(url, api_key));

            // Execute all futures concurrently and flatten results
            let all_workers = futures::future::try_join_all(worker_futs)
                .await?
                .into_iter()
                .flatten()
                .collect();

            Ok(all_workers)
        } else {
            Ok(urls
                .into_iter()
                .map(|url| Self::create_regular(url))
                .collect())
        }
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

        // Counter for periodic load reset (every 10 health check cycles)
        let mut check_count = 0u64;
        const LOAD_RESET_INTERVAL: u64 = 10;

        loop {
            interval.tick().await;

            // Check for shutdown signal
            if shutdown_clone.load(Ordering::Acquire) {
                tracing::debug!("Health checker shutting down");
                break;
            }

            check_count += 1;

            // Check health of all workers
            let workers_to_check = match workers.read() {
                Ok(guard) => guard.iter().map(|w| w.clone_worker()).collect::<Vec<_>>(),
                Err(poisoned) => {
                    tracing::error!("Worker lock poisoned: {}", poisoned);
                    continue;
                }
            };

            // Periodically reset load counters to prevent drift
            // Only do this when we believe all workers should be idle
            if check_count.is_multiple_of(LOAD_RESET_INTERVAL) {
                let max_load = workers_to_check.iter().map(|w| w.load()).max().unwrap_or(0);
                // Only reset if load appears to be very low (likely drift)
                if max_load <= 2 {
                    tracing::debug!(
                        "Resetting load counters to prevent drift (max_load: {})",
                        max_load
                    );
                    for worker in &workers_to_check {
                        worker.reset_load();
                    }
                }
            }

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
    use std::thread;
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
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.success_threshold, 2);
    }

    #[test]
    fn test_health_config_custom() {
        let config = HealthConfig {
            timeout_secs: 10,
            check_interval_secs: 60,
            endpoint: "/healthz".to_string(),
            failure_threshold: 5,
            success_threshold: 3,
        };
        assert_eq!(config.timeout_secs, 10);
        assert_eq!(config.check_interval_secs, 60);
        assert_eq!(config.endpoint, "/healthz");
        assert_eq!(config.failure_threshold, 5);
        assert_eq!(config.success_threshold, 3);
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
            failure_threshold: 4,
            success_threshold: 2,
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

        // Use AssertUnwindSafe wrapper for the test
        // This is safe because we're only testing the load counter behavior,
        // not the grpc_client which is None for HTTP workers
        use std::panic::AssertUnwindSafe;

        // This will panic, but the guard should still clean up
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _guard = WorkerLoadGuard::new(worker_clone.as_ref());
            assert_eq!(worker_clone.load(), 1);
            panic!("Test panic");
        }));

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

    // ===== Tests for DPAwareWorker =====

    #[test]
    fn test_dp_aware_worker_creation() {
        let dp_worker =
            DPAwareWorker::new("http://worker1:8080".to_string(), 2, 4, WorkerType::Regular);

        assert_eq!(dp_worker.url(), "http://worker1:8080@2");
        assert_eq!(dp_worker.base_url(), "http://worker1:8080");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(dp_worker.dp_rank(), Some(2));
        assert_eq!(dp_worker.dp_size(), Some(4));
        assert_eq!(dp_worker.worker_type(), WorkerType::Regular);
    }

    #[test]
    fn test_dp_aware_worker_creation_prefill() {
        let dp_worker = DPAwareWorker::new(
            "http://worker1:8080".to_string(),
            1,
            2,
            WorkerType::Prefill {
                bootstrap_port: Some(9090),
            },
        );

        assert_eq!(dp_worker.url(), "http://worker1:8080@1");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(
            dp_worker.worker_type(),
            WorkerType::Prefill {
                bootstrap_port: Some(9090)
            }
        );
    }

    #[test]
    fn test_dp_aware_worker_creation_decode() {
        let dp_worker =
            DPAwareWorker::new("http://worker1:8080".to_string(), 0, 4, WorkerType::Decode);

        assert_eq!(dp_worker.url(), "http://worker1:8080@0");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(dp_worker.worker_type(), WorkerType::Decode);
    }

    #[tokio::test]
    async fn test_dp_aware_prepare_request() {
        let dp_worker =
            DPAwareWorker::new("http://worker1:8080".to_string(), 3, 8, WorkerType::Regular);

        let original_req = serde_json::json!({
            "prompt": "Hello",
            "max_tokens": 100
        });

        let prepared_req = dp_worker.prepare_request(original_req).await.unwrap();

        assert_eq!(prepared_req["prompt"], "Hello");
        assert_eq!(prepared_req["max_tokens"], 100);
        assert_eq!(prepared_req["data_parallel_rank"], 3);
    }

    #[tokio::test]
    async fn test_dp_aware_prepare_request_invalid() {
        let dp_worker =
            DPAwareWorker::new("http://worker1:8080".to_string(), 0, 4, WorkerType::Regular);

        // Non-object JSON should fail
        let invalid_req = serde_json::json!("not an object");
        let result = dp_worker.prepare_request(invalid_req).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            WorkerError::InvalidConfiguration { message } => {
                assert!(message.contains("JSON object"));
            }
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }

    #[test]
    fn test_dp_aware_endpoint_url() {
        let dp_worker =
            DPAwareWorker::new("http://worker1:8080".to_string(), 1, 4, WorkerType::Regular);

        assert_eq!(
            dp_worker.endpoint_url("/generate"),
            "http://worker1:8080/generate"
        );
        assert_eq!(
            dp_worker.endpoint_url("/health"),
            "http://worker1:8080/health"
        );
    }

    #[test]
    fn test_dp_aware_worker_delegated_methods() {
        let dp_worker =
            DPAwareWorker::new("http://worker1:8080".to_string(), 0, 2, WorkerType::Regular);

        // Test health status
        assert!(dp_worker.is_healthy());
        dp_worker.set_healthy(false);
        assert!(!dp_worker.is_healthy());

        // Test load tracking
        assert_eq!(dp_worker.load(), 0);
        dp_worker.increment_load();
        assert_eq!(dp_worker.load(), 1);
        dp_worker.decrement_load();
        assert_eq!(dp_worker.load(), 0);

        // Test processed tracking
        assert_eq!(dp_worker.processed_requests(), 0);
        dp_worker.increment_processed();
        assert_eq!(dp_worker.processed_requests(), 1);
    }

    // ===== Tests for WorkerFactory async methods =====

    #[tokio::test]
    async fn test_factory_create_dp_aware() {
        let worker = WorkerFactory::create_dp_aware(
            "http://worker1:8080".to_string(),
            1,
            4,
            WorkerType::Regular,
        );

        assert_eq!(worker.url(), "http://worker1:8080@1");
        assert!(worker.is_dp_aware());
        assert_eq!(worker.dp_rank(), Some(1));
        assert_eq!(worker.dp_size(), Some(4));
        assert_eq!(worker.worker_type(), WorkerType::Regular);
    }

    #[tokio::test]
    async fn test_factory_create_dp_aware_prefill() {
        let worker = WorkerFactory::create_dp_aware(
            "http://worker1:8080".to_string(),
            0,
            2,
            WorkerType::Prefill {
                bootstrap_port: Some(8090),
            },
        );

        assert_eq!(worker.url(), "http://worker1:8080@0");
        assert!(worker.is_dp_aware());
        assert_eq!(
            worker.worker_type(),
            WorkerType::Prefill {
                bootstrap_port: Some(8090)
            }
        );
    }

    #[tokio::test]
    async fn test_factory_create_workers_regular() {
        let urls = vec!["http://w1:8080".to_string(), "http://w2:8080".to_string()];

        let workers = WorkerFactory::create_workers(urls, false, &None)
            .await
            .unwrap();

        assert_eq!(workers.len(), 2);
        assert!(!workers[0].is_dp_aware());
        assert!(!workers[1].is_dp_aware());
        assert_eq!(workers[0].url(), "http://w1:8080");
        assert_eq!(workers[1].url(), "http://w2:8080");
    }

    // ===== Circuit Breaker Integration Tests =====

    #[test]
    fn test_worker_circuit_breaker() {
        let worker = BasicWorker::new("http://test:8080".to_string(), WorkerType::Regular);

        // Initial state should be available
        assert!(worker.is_available());
        assert_eq!(
            worker.circuit_breaker().state(),
            crate::core::CircuitState::Closed
        );

        // Record some failures
        worker.record_outcome(false);
        worker.record_outcome(false);

        // Still available (default threshold is 5)
        assert!(worker.is_available());

        // Record more failures to open circuit
        worker.record_outcome(false);
        worker.record_outcome(false);
        worker.record_outcome(false);

        // Circuit should be open, worker not available
        assert!(!worker.is_available());
        assert!(worker.is_healthy()); // Still healthy
        assert!(!worker.circuit_breaker().can_execute()); // But circuit is open
    }

    #[test]
    fn test_worker_with_circuit_breaker_config() {
        let config = crate::core::CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout_duration: Duration::from_millis(100),
            window_duration: Duration::from_secs(60),
        };

        let worker = BasicWorker::new("http://test:8080".to_string(), WorkerType::Regular)
            .with_circuit_breaker_config(config);

        // Should open after 2 failures
        worker.record_outcome(false);
        assert!(worker.is_available());
        worker.record_outcome(false);
        assert!(!worker.is_available());

        // Wait for timeout
        thread::sleep(Duration::from_millis(150));

        // Should be half-open
        assert!(worker.is_available());
        assert_eq!(
            worker.circuit_breaker().state(),
            crate::core::CircuitState::HalfOpen
        );

        // Success should close it
        worker.record_outcome(true);
        assert_eq!(
            worker.circuit_breaker().state(),
            crate::core::CircuitState::Closed
        );
    }

    #[test]
    fn test_dp_aware_worker_circuit_breaker() {
        let dp_worker =
            DPAwareWorker::new("http://worker:8080".to_string(), 0, 2, WorkerType::Regular);

        // Should have circuit breaker
        assert!(dp_worker.is_available());

        // Record failures
        for _ in 0..5 {
            dp_worker.record_outcome(false);
        }

        // Should not be available
        assert!(!dp_worker.is_available());
        assert_eq!(
            dp_worker.circuit_breaker().state(),
            crate::core::CircuitState::Open
        );
    }

    // ===== Integration tests =====

    #[tokio::test]
    async fn test_mixed_worker_types() {
        // Create a mix of worker types
        let regular = WorkerFactory::create_regular("http://regular:8080".to_string());
        let prefill = WorkerFactory::create_prefill("http://prefill:8080".to_string(), Some(9090));
        let decode = WorkerFactory::create_decode("http://decode:8080".to_string());
        let dp_aware_regular =
            WorkerFactory::create_dp_aware("http://dp:8080".to_string(), 0, 2, WorkerType::Regular);
        let dp_aware_prefill = WorkerFactory::create_dp_aware(
            "http://dp-prefill:8080".to_string(),
            1,
            2,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        );
        let dp_aware_decode = WorkerFactory::create_dp_aware(
            "http://dp-decode:8080".to_string(),
            0,
            4,
            WorkerType::Decode,
        );

        let workers: Vec<Box<dyn Worker>> = vec![
            regular,
            prefill,
            decode,
            dp_aware_regular,
            dp_aware_prefill,
            dp_aware_decode,
        ];

        // Test that they all implement Worker trait properly
        for worker in &workers {
            assert!(worker.is_healthy());
            assert_eq!(worker.load(), 0);
            assert_eq!(worker.processed_requests(), 0);
        }

        // Test specific behaviors
        assert!(!workers[0].is_dp_aware()); // regular
        assert!(!workers[1].is_dp_aware()); // prefill
        assert!(!workers[2].is_dp_aware()); // decode
        assert!(workers[3].is_dp_aware()); // dp_aware_regular
        assert!(workers[4].is_dp_aware()); // dp_aware_prefill
        assert!(workers[5].is_dp_aware()); // dp_aware_decode

        // Test worker types
        assert_eq!(workers[0].worker_type(), WorkerType::Regular);
        assert_eq!(
            workers[1].worker_type(),
            WorkerType::Prefill {
                bootstrap_port: Some(9090)
            }
        );
        assert_eq!(workers[2].worker_type(), WorkerType::Decode);
        assert_eq!(workers[3].worker_type(), WorkerType::Regular);
        assert_eq!(
            workers[4].worker_type(),
            WorkerType::Prefill {
                bootstrap_port: None
            }
        );
        assert_eq!(workers[5].worker_type(), WorkerType::Decode);
    }
}
