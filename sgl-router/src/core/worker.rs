use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, LazyLock,
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json;
use tokio::{sync::RwLock, time};

use super::{CircuitBreaker, WorkerError, WorkerResult};
use crate::{
    core::{BasicWorkerBuilder, CircuitState, DPAwareWorkerBuilder},
    metrics::RouterMetrics,
    protocols::worker_spec::WorkerInfo,
    routers::grpc::client::GrpcClient,
};

static WORKER_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create worker HTTP client")
});

/// Core worker abstraction that represents a backend service
#[async_trait]
pub trait Worker: Send + Sync + fmt::Debug {
    /// Get the worker's URL
    fn url(&self) -> &str;
    /// Get the worker's API key
    fn api_key(&self) -> &Option<String>;
    /// Get the worker's type (Regular, Prefill, or Decode)
    fn worker_type(&self) -> WorkerType;

    /// Get the worker's connection mode (HTTP or gRPC)
    fn connection_mode(&self) -> ConnectionMode;

    /// Get the bootstrap hostname for PD mode
    /// Returns cached hostname parsed from URL at construction time
    fn bootstrap_host(&self) -> &str {
        &self.metadata().bootstrap_host
    }

    /// Get the bootstrap port for PD mode
    /// Returns cached port from WorkerType::Prefill
    fn bootstrap_port(&self) -> Option<u16> {
        self.metadata().bootstrap_port
    }

    /// Check if the worker is currently healthy
    fn is_healthy(&self) -> bool;

    /// Set the worker's health status
    fn set_healthy(&self, healthy: bool);

    /// Perform an async health check on the worker
    async fn check_health_async(&self) -> WorkerResult<()>;

    /// Synchronous health check wrapper (for compatibility)
    fn check_health(&self) -> WorkerResult<()> {
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
    fn reset_load(&self) {}

    /// Get the number of processed requests
    fn processed_requests(&self) -> usize;

    /// Increment the processed requests counter
    fn increment_processed(&self);

    /// Get worker-specific metadata
    fn metadata(&self) -> &WorkerMetadata;

    /// Get the circuit breaker for this worker
    fn circuit_breaker(&self) -> &CircuitBreaker;

    /// Check if the worker is available (healthy + circuit closed/half-open)
    fn is_available(&self) -> bool {
        self.is_healthy() && self.circuit_breaker().can_execute()
    }

    /// Record the outcome of a request to this worker
    fn record_outcome(&self, success: bool) {
        let outcome_str = if success { "success" } else { "failure" };
        RouterMetrics::record_cb_outcome(self.url(), outcome_str);

        let before = self.circuit_breaker().state();
        self.circuit_breaker().record_outcome(success);
        let after = self.circuit_breaker().state();

        if before != after {
            let from = match before {
                CircuitState::Closed => "closed",
                CircuitState::Open => "open",
                CircuitState::HalfOpen => "half_open",
            };
            let to = match after {
                CircuitState::Closed => "closed",
                CircuitState::Open => "open",
                CircuitState::HalfOpen => "half_open",
            };
            RouterMetrics::record_cb_state_transition(self.url(), from, to);
        }

        let state_code = match self.circuit_breaker().state() {
            CircuitState::Closed => 0u8,
            CircuitState::Open => 1u8,
            CircuitState::HalfOpen => 2u8,
        };
        RouterMetrics::set_cb_state(self.url(), state_code);
    }

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

    /// Get the model ID this worker serves
    fn model_id(&self) -> &str {
        self.metadata()
            .labels
            .get("model_id")
            .map(|s| s.as_str())
            .unwrap_or("unknown")
    }

    /// Get the priority of this worker (higher value = higher priority)
    fn priority(&self) -> u32 {
        self.metadata()
            .labels
            .get("priority")
            .and_then(|s| s.parse().ok())
            .unwrap_or(50) // Default priority is 50 (mid-range)
    }

    /// Get the cost factor of this worker (1.0 = baseline)
    fn cost(&self) -> f32 {
        self.metadata()
            .labels
            .get("cost")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.0)
    }

    /// Get the tokenizer path for this worker (gRPC mode only)
    fn tokenizer_path(&self) -> Option<&str> {
        self.metadata()
            .labels
            .get("tokenizer_path")
            .map(|s| s.as_str())
    }

    /// Get the reasoning parser type for this worker (gRPC mode only)
    fn reasoning_parser(&self) -> Option<&str> {
        self.metadata()
            .labels
            .get("reasoning_parser")
            .map(|s| s.as_str())
    }

    /// Get the tool parser type for this worker (gRPC mode only)
    fn tool_parser(&self) -> Option<&str> {
        self.metadata()
            .labels
            .get("tool_parser")
            .map(|s| s.as_str())
    }

    /// Get the chat template for this worker (gRPC mode only)
    fn chat_template(&self) -> Option<&str> {
        self.metadata()
            .labels
            .get("chat_template")
            .map(|s| s.as_str())
    }

    /// Get or create a gRPC client for this worker
    /// Returns None for HTTP workers, Some(client) for gRPC workers
    async fn get_grpc_client(&self) -> WorkerResult<Option<Arc<GrpcClient>>>;

    /// Reset the gRPC client connection (for reconnection scenarios)
    /// No-op for HTTP workers
    async fn reset_grpc_client(&self) -> WorkerResult<()> {
        Ok(())
    }
    async fn grpc_health_check(&self) -> WorkerResult<bool>;
    async fn http_health_check(&self) -> WorkerResult<bool>;
}

/// Connection mode for worker communication
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ConnectionMode {
    /// HTTP/REST connection
    #[default]
    Http,
    /// gRPC connection
    Grpc {
        /// Optional port for gRPC endpoint (if different from URL)
        #[serde(skip_serializing_if = "Option::is_none")]
        #[serde(default)]
        port: Option<u16>,
    },
}

impl ConnectionMode {
    /// Check if this connection mode matches another, with special handling for gRPC
    /// This allows matching any gRPC connection regardless of port when comparing
    /// Grpc { port: None } as a wildcard
    pub fn matches(&self, filter: &ConnectionMode) -> bool {
        match (self, filter) {
            (ConnectionMode::Http, ConnectionMode::Http) => true,
            (ConnectionMode::Grpc { .. }, ConnectionMode::Grpc { port: None }) => true,
            (ConnectionMode::Grpc { port: p1 }, ConnectionMode::Grpc { port: p2 }) => p1 == p2,
            _ => false,
        }
    }
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

/// Runtime implementation type for gRPC workers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeType {
    /// SGLang runtime (default)
    #[default]
    Sglang,
    /// vLLM runtime
    Vllm,
}

impl fmt::Display for RuntimeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeType::Sglang => write!(f, "sglang"),
            RuntimeType::Vllm => write!(f, "vllm"),
        }
    }
}

impl std::str::FromStr for RuntimeType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sglang" => Ok(RuntimeType::Sglang),
            "vllm" => Ok(RuntimeType::Vllm),
            _ => Err(format!("Unknown runtime type: {}", s)),
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
    /// Runtime type (for gRPC workers)
    pub runtime_type: RuntimeType,
    /// Additional labels/tags
    pub labels: std::collections::HashMap<String, String>,
    /// Health check configuration
    pub health_config: HealthConfig,
    /// API key
    pub api_key: Option<String>,
    /// Cached bootstrap hostname (parsed from URL at construction time)
    pub bootstrap_host: String,
    /// Cached bootstrap port (from WorkerType::Prefill)
    pub bootstrap_port: Option<u16>,
}

/// Basic worker implementation
#[derive(Clone)]
pub struct BasicWorker {
    pub metadata: WorkerMetadata,
    pub load_counter: Arc<AtomicUsize>,
    pub processed_counter: Arc<AtomicUsize>,
    pub healthy: Arc<AtomicBool>,
    pub consecutive_failures: Arc<AtomicUsize>,
    pub consecutive_successes: Arc<AtomicUsize>,
    pub circuit_breaker: CircuitBreaker,
    /// Lazily initialized gRPC client for gRPC workers
    pub grpc_client: Arc<RwLock<Option<Arc<GrpcClient>>>>,
}

impl fmt::Debug for BasicWorker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BasicWorker")
            .field("metadata", &self.metadata)
            .field("healthy", &self.healthy.load(Ordering::Relaxed))
            .field("circuit_breaker", &self.circuit_breaker)
            .field("grpc_client", &"<RwLock>")
            .finish()
    }
}

impl BasicWorker {
    pub fn normalised_url(&self) -> WorkerResult<&str> {
        if self.url().contains("@") {
            // Use rfind to split from the right, handling IPv6 addresses with brackets
            // e.g., "http://[::1]:8080@0" -> "http://[::1]:8080" and "0"
            if let Some(at_pos) = self.url().rfind('@') {
                let base_url = &self.url()[..at_pos];
                let rank_str = &self.url()[at_pos + 1..];

                // Validate that the rank part is actually a number
                match rank_str.parse::<usize>() {
                    Ok(_) => Ok(base_url),
                    Err(_) => {
                        // The '@' is not a DP rank separator, return full URL
                        Ok(self.url())
                    }
                }
            } else {
                Ok(self.url())
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

    fn api_key(&self) -> &Option<String> {
        &self.metadata.api_key
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
        let health_result = match &self.metadata.connection_mode {
            ConnectionMode::Http => self.http_health_check().await?,
            ConnectionMode::Grpc { .. } => self.grpc_health_check().await?,
        };

        if health_result {
            self.consecutive_failures.store(0, Ordering::Release);
            let successes = self.consecutive_successes.fetch_add(1, Ordering::AcqRel) + 1;

            if !self.is_healthy()
                && successes >= self.metadata.health_config.success_threshold as usize
            {
                self.set_healthy(true);
                self.consecutive_successes.store(0, Ordering::Release);
            }
            Ok(())
        } else {
            self.consecutive_successes.store(0, Ordering::Release);
            let failures = self.consecutive_failures.fetch_add(1, Ordering::AcqRel) + 1;

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

    fn circuit_breaker(&self) -> &CircuitBreaker {
        &self.circuit_breaker
    }

    async fn get_grpc_client(&self) -> WorkerResult<Option<Arc<GrpcClient>>> {
        match self.metadata.connection_mode {
            ConnectionMode::Http => Ok(None),
            ConnectionMode::Grpc { .. } => {
                {
                    let client_guard = self.grpc_client.read().await;
                    if let Some(ref client) = *client_guard {
                        return Ok(Some(client.clone()));
                    }
                }

                let mut client_guard = self.grpc_client.write().await;

                if let Some(ref client) = *client_guard {
                    return Ok(Some(client.clone()));
                }

                let runtime_str = self.metadata.runtime_type.to_string();
                tracing::info!(
                    "Lazily initializing gRPC client ({}) for worker: {}",
                    runtime_str,
                    self.metadata.url
                );
                match GrpcClient::connect(&self.metadata.url, &runtime_str).await {
                    Ok(client) => {
                        let client_arc = Arc::new(client);
                        *client_guard = Some(client_arc.clone());
                        tracing::info!(
                            "Successfully connected gRPC client ({}) for worker: {}",
                            runtime_str,
                            self.metadata.url
                        );
                        Ok(Some(client_arc))
                    }
                    Err(e) => {
                        tracing::error!(
                            "Failed to connect gRPC client for worker {}: {}",
                            self.metadata.url,
                            e
                        );
                        Err(WorkerError::ConnectionFailed {
                            url: self.metadata.url.clone(),
                            reason: format!("Failed to connect to gRPC server: {}", e),
                        })
                    }
                }
            }
        }
    }

    async fn reset_grpc_client(&self) -> WorkerResult<()> {
        match self.metadata.connection_mode {
            ConnectionMode::Http => Ok(()),
            ConnectionMode::Grpc { .. } => {
                let mut client_guard = self.grpc_client.write().await;
                if client_guard.is_some() {
                    tracing::info!("Resetting gRPC client for worker: {}", self.metadata.url);
                    *client_guard = None;
                }
                Ok(())
            }
        }
    }

    async fn grpc_health_check(&self) -> WorkerResult<bool> {
        let timeout = Duration::from_secs(self.metadata.health_config.timeout_secs);
        let maybe = self.get_grpc_client().await?;
        let Some(grpc_client) = maybe else {
            tracing::error!(
                "Worker {} is not a gRPC worker but connection mode is gRPC",
                self.metadata.url
            );
            return Ok(false);
        };

        match time::timeout(timeout, grpc_client.health_check()).await {
            Ok(Ok(resp)) => {
                tracing::debug!(
                    "gRPC health OK for {}: healthy={}",
                    self.metadata.url,
                    resp.healthy
                );
                Ok(resp.healthy)
            }
            Ok(Err(err)) => {
                tracing::warn!("gRPC health RPC error for {}: {err:?}", self.metadata.url);
                Ok(false)
            }
            Err(_) => {
                tracing::warn!("gRPC health timed out for {}", self.metadata.url);
                Ok(false)
            }
        }
    }

    async fn http_health_check(&self) -> WorkerResult<bool> {
        let timeout = Duration::from_secs(self.metadata.health_config.timeout_secs);

        let url = self.normalised_url()?;
        let health_url = format!("{}{}", url, self.metadata.health_config.endpoint);

        let mut req = WORKER_CLIENT.get(health_url).timeout(timeout);
        if let Some(api_key) = &self.metadata.api_key {
            req = req.bearer_auth(api_key);
        }

        match req.send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(err) => {
                tracing::warn!(
                    "HTTP health check failed for {}: {err:?}",
                    self.metadata.url
                );
                Ok(false)
            }
        }
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
    /// Create a new DP-aware worker with a pre-configured base worker
    /// This is primarily used by the builder pattern
    pub fn with_base_worker(
        base_worker: BasicWorker,
        base_url: String,
        dp_rank: usize,
        dp_size: usize,
    ) -> Self {
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

    fn api_key(&self) -> &Option<String> {
        self.base_worker.api_key()
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

    fn circuit_breaker(&self) -> &CircuitBreaker {
        self.base_worker.circuit_breaker()
    }

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
        format!("{}{}", self.base_url, route)
    }

    async fn get_grpc_client(&self) -> WorkerResult<Option<Arc<GrpcClient>>> {
        self.base_worker.get_grpc_client().await
    }

    async fn reset_grpc_client(&self) -> WorkerResult<()> {
        self.base_worker.reset_grpc_client().await
    }

    async fn grpc_health_check(&self) -> WorkerResult<bool> {
        self.base_worker.grpc_health_check().await
    }

    async fn http_health_check(&self) -> WorkerResult<bool> {
        self.base_worker.http_health_check().await
    }
}

/// Worker factory for creating workers of different types
pub struct WorkerFactory;

impl WorkerFactory {
    /// Create a DP-aware worker of specified type
    pub fn create_dp_aware(
        base_url: String,
        dp_rank: usize,
        dp_size: usize,
        worker_type: WorkerType,
        api_key: Option<String>,
    ) -> Box<dyn Worker> {
        let mut builder =
            DPAwareWorkerBuilder::new(base_url, dp_rank, dp_size).worker_type(worker_type);
        if let Some(api_key) = api_key {
            builder = builder.api_key(api_key);
        }
        Box::new(builder.build())
    }

    /// Static health validation before creating a worker
    /// This replaces wait_for_worker_health in handlers
    pub async fn validate_health(url: &str, timeout_secs: u64) -> WorkerResult<()> {
        let start_time = Instant::now();
        let timeout = Duration::from_secs(timeout_secs);

        loop {
            if start_time.elapsed() > timeout {
                return Err(WorkerError::HealthCheckFailed {
                    url: url.to_string(),
                    reason: format!(
                        "Timeout {}s waiting for worker to become healthy",
                        timeout_secs
                    ),
                });
            }

            // Note: This static function doesn't have access to worker's API key
            // API key authentication is handled in the worker instance's check_health_async method
            match WORKER_CLIENT
                .get(format!("{}/health", url))
                .timeout(Duration::from_secs(5))
                .send()
                .await
            {
                Ok(res) if res.status().is_success() => {
                    tracing::info!("Worker {} is healthy", url);
                    return Ok(());
                }
                Ok(res) => {
                    tracing::warn!(
                        "Worker {} health check failed with status: {}",
                        url,
                        res.status()
                    );
                }
                Err(e) => {
                    tracing::warn!("Failed to contact worker {}: {}", url, e);
                }
            }

            time::sleep(Duration::from_secs(1)).await;
        }
    }
}

/// Convert a list of worker URLs to worker trait objects
pub fn urls_to_workers(urls: Vec<String>, api_key: Option<String>) -> Vec<Box<dyn Worker>> {
    urls.into_iter()
        .map(|url| {
            let worker_builder = BasicWorkerBuilder::new(url).worker_type(WorkerType::Regular);

            let worker = if let Some(ref api_key) = api_key {
                worker_builder.api_key(api_key.clone()).build()
            } else {
                worker_builder.build()
            };

            Box::new(worker) as Box<dyn Worker>
        })
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
    /// Create a new HealthChecker
    pub fn new(handle: tokio::task::JoinHandle<()>, shutdown: Arc<AtomicBool>) -> Self {
        Self { handle, shutdown }
    }

    /// Shutdown the health checker gracefully
    pub async fn shutdown(self) {
        self.shutdown.store(true, Ordering::Release);
        let _ = self.handle.await;
    }
}

/// Helper to convert Worker trait object to WorkerInfo struct
pub fn worker_to_info(worker: &Arc<dyn Worker>) -> WorkerInfo {
    let worker_type_str = match worker.worker_type() {
        WorkerType::Regular => "regular",
        WorkerType::Prefill { .. } => "prefill",
        WorkerType::Decode => "decode",
    };

    let bootstrap_port = match worker.worker_type() {
        WorkerType::Prefill { bootstrap_port } => bootstrap_port,
        _ => None,
    };

    let runtime_type = match worker.connection_mode() {
        ConnectionMode::Grpc { .. } => Some(worker.metadata().runtime_type.to_string()),
        ConnectionMode::Http => None,
    };

    WorkerInfo {
        id: worker.url().to_string(),
        url: worker.url().to_string(),
        model_id: worker.model_id().to_string(),
        priority: worker.priority(),
        cost: worker.cost(),
        worker_type: worker_type_str.to_string(),
        is_healthy: worker.is_healthy(),
        load: worker.load(),
        connection_mode: format!("{:?}", worker.connection_mode()),
        runtime_type,
        tokenizer_path: worker.tokenizer_path().map(String::from),
        reasoning_parser: worker.reasoning_parser().map(String::from),
        tool_parser: worker.tool_parser().map(String::from),
        chat_template: worker.chat_template().map(String::from),
        bootstrap_port,
        metadata: worker.metadata().labels.clone(),
        job_status: None,
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use super::*;
    use crate::core::CircuitBreakerConfig;

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

    #[test]
    fn test_basic_worker_creation() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();
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

        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .labels(labels.clone())
            .build();

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

        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .health_config(custom_config.clone())
            .build();

        assert_eq!(worker.metadata().health_config.timeout_secs, 15);
        assert_eq!(worker.metadata().health_config.check_interval_secs, 45);
        assert_eq!(worker.metadata().health_config.endpoint, "/custom-health");
    }

    #[test]
    fn test_worker_url() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(worker.url(), "http://worker1:8080");
    }

    #[test]
    fn test_worker_type_getter() {
        use crate::core::BasicWorkerBuilder;
        let regular = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(regular.worker_type(), WorkerType::Regular);

        let prefill = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(9090),
            })
            .build();
        assert_eq!(
            prefill.worker_type(),
            WorkerType::Prefill {
                bootstrap_port: Some(9090)
            }
        );

        let decode = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Decode)
            .build();
        assert_eq!(decode.worker_type(), WorkerType::Decode);
    }

    #[test]
    fn test_health_status() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert!(worker.is_healthy());

        worker.set_healthy(false);
        assert!(!worker.is_healthy());

        worker.set_healthy(true);
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_load_counter_operations() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(worker.load(), 0);

        worker.increment_load();
        assert_eq!(worker.load(), 1);

        worker.increment_load();
        worker.increment_load();
        assert_eq!(worker.load(), 3);

        worker.decrement_load();
        assert_eq!(worker.load(), 2);

        worker.decrement_load();
        worker.decrement_load();
        assert_eq!(worker.load(), 0);

        worker.decrement_load();
        assert_eq!(worker.load(), 0);
    }

    #[test]
    fn test_processed_counter() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(worker.processed_requests(), 0);

        for i in 1..=100 {
            worker.increment_processed();
            assert_eq!(worker.processed_requests(), i);
        }
    }

    #[tokio::test]
    async fn test_concurrent_load_increments() {
        use crate::core::BasicWorkerBuilder;
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut handles = vec![];

        for _ in 0..100 {
            let worker_clone = Arc::clone(&worker);
            let handle = tokio::spawn(async move {
                worker_clone.increment_load();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(worker.load(), 100);
    }

    #[tokio::test]
    async fn test_concurrent_load_decrements() {
        use crate::core::BasicWorkerBuilder;
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        for _ in 0..100 {
            worker.increment_load();
        }
        assert_eq!(worker.load(), 100);

        let mut handles = vec![];

        for _ in 0..100 {
            let worker_clone = Arc::clone(&worker);
            let handle = tokio::spawn(async move {
                worker_clone.decrement_load();
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(worker.load(), 0);
    }

    #[tokio::test]
    async fn test_concurrent_health_updates() {
        use crate::core::BasicWorkerBuilder;
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        let mut handles = vec![];

        for i in 0..100 {
            let worker_clone = Arc::clone(&worker);
            let handle = tokio::spawn(async move {
                worker_clone.set_healthy(i % 2 == 0);
                time::sleep(Duration::from_micros(10)).await;
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }
    }

    #[test]
    fn test_create_regular_worker() {
        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://regular:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        assert_eq!(worker.url(), "http://regular:8080");
        assert_eq!(worker.worker_type(), WorkerType::Regular);
    }

    #[test]
    fn test_create_prefill_worker() {
        let worker1: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(9090),
                })
                .build(),
        );
        assert_eq!(worker1.url(), "http://prefill:8080");
        assert_eq!(
            worker1.worker_type(),
            WorkerType::Prefill {
                bootstrap_port: Some(9090)
            }
        );

        let worker2: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: None,
                })
                .build(),
        );
        assert_eq!(
            worker2.worker_type(),
            WorkerType::Prefill {
                bootstrap_port: None
            }
        );
    }

    #[test]
    fn test_create_decode_worker() {
        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://decode:8080")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        assert_eq!(worker.url(), "http://decode:8080");
        assert_eq!(worker.worker_type(), WorkerType::Decode);
    }

    #[test]
    fn test_load_guard_single_worker() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();
        assert_eq!(worker.load(), 0);

        {
            let _guard = WorkerLoadGuard::new(&worker);
            assert_eq!(worker.load(), 1);
        }

        assert_eq!(worker.load(), 0);
    }

    #[test]
    fn test_load_guard_multiple_workers() {
        let workers: Vec<Box<dyn Worker>> = vec![
            Box::new(
                BasicWorkerBuilder::new("http://w1:8080")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Box::new(
                BasicWorkerBuilder::new("http://w2:8080")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Box::new(
                BasicWorkerBuilder::new("http://w3:8080")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let worker_refs: Vec<&dyn Worker> = workers.iter().map(|w| w.as_ref()).collect();

        {
            let _guard = WorkerLoadGuard::new_multi(worker_refs);
            assert_eq!(workers[0].load(), 1);
            assert_eq!(workers[1].load(), 1);
            assert_eq!(workers[2].load(), 1);
        }

        assert_eq!(workers[0].load(), 0);
        assert_eq!(workers[1].load(), 0);
        assert_eq!(workers[2].load(), 0);
    }

    #[test]
    fn test_load_guard_panic_safety() {
        use crate::core::BasicWorkerBuilder;
        let worker = Arc::new(
            BasicWorkerBuilder::new("http://test:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        assert_eq!(worker.load(), 0);

        let worker_clone = Arc::clone(&worker);

        use std::panic::AssertUnwindSafe;

        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _guard = WorkerLoadGuard::new(worker_clone.as_ref());
            assert_eq!(worker_clone.load(), 1);
            panic!("Test panic");
        }));

        assert!(result.is_err());

        assert_eq!(worker.load(), 0);
    }

    #[test]
    fn test_urls_to_workers() {
        let urls = vec!["http://w1:8080".to_string(), "http://w2:8080".to_string()];

        let workers = urls_to_workers(urls, Some("test_api_key".to_string()));
        assert_eq!(workers.len(), 2);
        assert_eq!(workers[0].url(), "http://w1:8080");
        assert_eq!(workers[1].url(), "http://w2:8080");
        assert_eq!(workers[0].worker_type(), WorkerType::Regular);
    }

    #[test]
    fn test_workers_to_urls() {
        let workers: Vec<Box<dyn Worker>> = vec![
            Box::new(
                BasicWorkerBuilder::new("http://w1:8080")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Box::new(
                BasicWorkerBuilder::new("http://w2:8080")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let urls = workers_to_urls(&workers);
        assert_eq!(urls, vec!["http://w1:8080", "http://w2:8080"]);
    }

    #[test]
    fn test_check_health_sync_wrapper() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        let result = worker.check_health();
        assert!(result.is_err());
    }

    #[test]
    fn test_load_counter_performance() {
        use std::time::Instant;

        use crate::core::BasicWorkerBuilder;

        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();
        let iterations = 1_000_000;

        let start = Instant::now();
        for _ in 0..iterations {
            worker.increment_load();
        }
        let duration = start.elapsed();

        let ops_per_sec = iterations as f64 / duration.as_secs_f64();
        println!("Load counter operations per second: {:.0}", ops_per_sec);

        assert!(ops_per_sec > 1_000_000.0);
    }

    #[test]
    fn test_dp_aware_worker_creation() {
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 2, 4)
            .worker_type(WorkerType::Regular)
            .build();

        assert_eq!(dp_worker.url(), "http://worker1:8080@2");
        assert_eq!(dp_worker.base_url(), "http://worker1:8080");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(dp_worker.dp_rank(), Some(2));
        assert_eq!(dp_worker.dp_size(), Some(4));
        assert_eq!(dp_worker.worker_type(), WorkerType::Regular);
    }

    #[test]
    fn test_dp_aware_worker_creation_prefill() {
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 1, 2)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(9090),
            })
            .build();

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
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 0, 4)
            .worker_type(WorkerType::Decode)
            .build();

        assert_eq!(dp_worker.url(), "http://worker1:8080@0");
        assert!(dp_worker.is_dp_aware());
        assert_eq!(dp_worker.worker_type(), WorkerType::Decode);
    }

    #[tokio::test]
    async fn test_dp_aware_prepare_request() {
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 3, 8)
            .worker_type(WorkerType::Regular)
            .build();

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
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 0, 4)
            .worker_type(WorkerType::Regular)
            .build();

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
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 1, 4)
            .worker_type(WorkerType::Regular)
            .build();

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
        let dp_worker = DPAwareWorkerBuilder::new("http://worker1:8080", 0, 2)
            .worker_type(WorkerType::Regular)
            .build();

        assert!(dp_worker.is_healthy());
        dp_worker.set_healthy(false);
        assert!(!dp_worker.is_healthy());

        assert_eq!(dp_worker.load(), 0);
        dp_worker.increment_load();
        assert_eq!(dp_worker.load(), 1);
        dp_worker.decrement_load();
        assert_eq!(dp_worker.load(), 0);

        assert_eq!(dp_worker.processed_requests(), 0);
        dp_worker.increment_processed();
        assert_eq!(dp_worker.processed_requests(), 1);
    }

    #[tokio::test]
    async fn test_factory_create_dp_aware() {
        let worker = WorkerFactory::create_dp_aware(
            "http://worker1:8080".to_string(),
            1,
            4,
            WorkerType::Regular,
            Some("test_api_key".to_string()),
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
            Some("test_api_key".to_string()),
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

    #[test]
    fn test_worker_circuit_breaker() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        assert!(worker.is_available());
        assert_eq!(worker.circuit_breaker().state(), CircuitState::Closed);

        worker.record_outcome(false);
        worker.record_outcome(false);

        assert!(worker.is_available());

        worker.record_outcome(false);
        worker.record_outcome(false);
        worker.record_outcome(false);

        assert!(!worker.is_available());
        assert!(worker.is_healthy());
        assert!(!worker.circuit_breaker().can_execute());
    }

    #[test]
    fn test_worker_with_circuit_breaker_config() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout_duration: Duration::from_millis(100),
            window_duration: Duration::from_secs(60),
        };

        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .circuit_breaker_config(config)
            .build();

        worker.record_outcome(false);
        assert!(worker.is_available());
        worker.record_outcome(false);
        assert!(!worker.is_available());

        thread::sleep(Duration::from_millis(150));

        assert!(worker.is_available());
        assert_eq!(worker.circuit_breaker().state(), CircuitState::HalfOpen);

        worker.record_outcome(true);
        assert_eq!(worker.circuit_breaker().state(), CircuitState::Closed);
    }

    #[test]
    fn test_dp_aware_worker_circuit_breaker() {
        let dp_worker = DPAwareWorkerBuilder::new("http://worker:8080", 0, 2)
            .worker_type(WorkerType::Regular)
            .build();

        assert!(dp_worker.is_available());

        for _ in 0..5 {
            dp_worker.record_outcome(false);
        }

        assert!(!dp_worker.is_available());
        assert_eq!(dp_worker.circuit_breaker().state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn test_mixed_worker_types() {
        let regular: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://regular:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        let prefill: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://prefill:8080")
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(9090),
                })
                .build(),
        );
        let decode: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://decode:8080")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        let dp_aware_regular = WorkerFactory::create_dp_aware(
            "http://dp:8080".to_string(),
            0,
            2,
            WorkerType::Regular,
            Some("test_api_key".to_string()),
        );
        let dp_aware_prefill = WorkerFactory::create_dp_aware(
            "http://dp-prefill:8080".to_string(),
            1,
            2,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            Some("test_api_key".to_string()),
        );
        let dp_aware_decode = WorkerFactory::create_dp_aware(
            "http://dp-decode:8080".to_string(),
            0,
            4,
            WorkerType::Decode,
            Some("test_api_key".to_string()),
        );

        let workers: Vec<Box<dyn Worker>> = vec![
            regular,
            prefill,
            decode,
            dp_aware_regular,
            dp_aware_prefill,
            dp_aware_decode,
        ];

        for worker in &workers {
            assert!(worker.is_healthy());
            assert_eq!(worker.load(), 0);
            assert_eq!(worker.processed_requests(), 0);
        }

        assert!(!workers[0].is_dp_aware());
        assert!(!workers[1].is_dp_aware());
        assert!(!workers[2].is_dp_aware());
        assert!(workers[3].is_dp_aware());
        assert!(workers[4].is_dp_aware());
        assert!(workers[5].is_dp_aware());

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
