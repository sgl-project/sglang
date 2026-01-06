use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, LazyLock, RwLock as StdRwLock,
    },
    time::Duration,
};

use async_trait::async_trait;
use axum::body::Body;
use serde::{Deserialize, Serialize};
use tokio::{sync::OnceCell, time};

use super::{
    model_card::{ModelCard, ProviderType},
    model_type::{Endpoint, ModelType},
    CircuitBreaker, WorkerError, WorkerResult, UNKNOWN_MODEL_ID,
};
use crate::{
    observability::metrics::{metrics_labels, Metrics},
    protocols::worker_spec::WorkerInfo,
    routers::grpc::client::GrpcClient,
};

/// Default worker priority (mid-range on 0-100 scale)
pub const DEFAULT_WORKER_PRIORITY: u32 = 50;

/// Default worker cost factor (baseline cost)
pub const DEFAULT_WORKER_COST: f32 = 1.0;

/// Default HTTP client timeout for worker requests (in seconds)
pub const DEFAULT_WORKER_HTTP_TIMEOUT_SECS: u64 = 30;

static WORKER_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(DEFAULT_WORKER_HTTP_TIMEOUT_SECS))
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
    /// Returns a reference to avoid cloning on every access
    fn worker_type(&self) -> &WorkerType;

    /// Get the worker's connection mode (HTTP or gRPC)
    /// Returns a reference to avoid cloning on every access
    fn connection_mode(&self) -> &ConnectionMode;

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
    ///
    /// # Deprecation Notice
    /// This method creates a new Tokio runtime for each call, which is expensive.
    /// Prefer using `check_health_async()` within an async context instead.
    ///
    /// # Performance Warning
    /// Creating a runtime per call has significant overhead. Only use this
    /// method when you cannot use the async version.
    #[deprecated(
        since = "0.4.6",
        note = "Use check_health_async() instead. This method creates a new Tokio runtime per call."
    )]
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
        self.circuit_breaker().record_outcome(success);
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
    /// Checks ModelCards first, then falls back to labels
    fn model_id(&self) -> &str {
        // Check ModelCards first
        self.metadata()
            .models
            .first()
            .map(|m| m.id.as_str())
            .or_else(|| {
                // Fall back to labels
                self.metadata().labels.get("model_id").map(|s| s.as_str())
            })
            .unwrap_or(UNKNOWN_MODEL_ID)
    }

    /// Get the priority of this worker (higher value = higher priority)
    fn priority(&self) -> u32 {
        self.metadata()
            .labels
            .get("priority")
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_WORKER_PRIORITY)
    }

    /// Get the cost factor of this worker (baseline = 1.0)
    fn cost(&self) -> f32 {
        self.metadata()
            .labels
            .get("cost")
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_WORKER_COST)
    }

    /// Get tokenizer path for a specific model.
    fn tokenizer_path(&self, model_id: &str) -> Option<&str> {
        self.metadata()
            .find_model(model_id)
            .and_then(|m| m.tokenizer_path.as_deref())
    }

    /// Get reasoning parser for a specific model.
    fn reasoning_parser(&self, model_id: &str) -> Option<&str> {
        self.metadata()
            .find_model(model_id)
            .and_then(|m| m.reasoning_parser.as_deref())
    }

    /// Get tool parser for a specific model.
    fn tool_parser(&self, model_id: &str) -> Option<&str> {
        self.metadata()
            .find_model(model_id)
            .and_then(|m| m.tool_parser.as_deref())
    }

    /// Get chat template for a specific model.
    fn chat_template(&self, model_id: &str) -> Option<&str> {
        self.metadata()
            .find_model(model_id)
            .and_then(|m| m.chat_template.as_deref())
    }

    /// Get the default provider type for this worker.
    /// `None` means native/passthrough.
    fn default_provider(&self) -> Option<&ProviderType> {
        self.metadata().default_provider.as_ref()
    }

    /// Get provider for a specific model.
    /// Priority: ModelCard.provider > worker.default_provider
    fn provider_for_model(&self, model_id: &str) -> Option<&ProviderType> {
        self.metadata().provider_for_model(model_id)
    }

    /// Check if a model is a classifier (has id2label mapping).
    fn is_classifier(&self, model_id: &str) -> bool {
        self.metadata()
            .find_model(model_id)
            .map(|m| m.is_classifier())
            .unwrap_or(false)
    }

    /// Get the id2label mapping for a classification model.
    /// Returns None if model is not a classifier or not found.
    fn id2label(&self, model_id: &str) -> Option<&std::collections::HashMap<u32, String>> {
        self.metadata()
            .find_model(model_id)
            .filter(|m| m.is_classifier())
            .map(|m| &m.id2label)
    }

    /// Get the number of classification labels for a model.
    fn num_labels(&self, model_id: &str) -> u32 {
        self.metadata()
            .find_model(model_id)
            .map(|m| m.num_labels)
            .unwrap_or(0)
    }

    /// Get label for a class index from a classification model.
    /// Returns generic label (LABEL_N) if model not found or index not in mapping.
    fn get_label(&self, model_id: &str, class_idx: u32) -> String {
        self.metadata()
            .find_model(model_id)
            .map(|m| m.get_label(class_idx))
            .unwrap_or_else(|| format!("LABEL_{}", class_idx))
    }

    /// Check if this worker supports a specific model.
    /// If models list is empty, worker accepts any model.
    fn supports_model(&self, model_id: &str) -> bool {
        self.metadata().supports_model(model_id)
    }

    /// Check if this worker supports an endpoint for a given model.
    /// Falls back to default_model_type if model not found.
    fn supports_endpoint(&self, model_id: &str, endpoint: Endpoint) -> bool {
        self.metadata().supports_endpoint(model_id, endpoint)
    }

    /// Get all models this worker can serve.
    fn models(&self) -> &[ModelCard] {
        &self.metadata().models
    }

    /// Set models for this worker (for lazy discovery).
    /// Default implementation does nothing - only BasicWorker supports this.
    fn set_models(&self, _models: Vec<ModelCard>) {
        // Default: no-op. BasicWorker overrides this.
    }

    /// Check if models have been discovered for this worker.
    /// Returns true if models were set via set_models() or if metadata has models.
    fn has_models_discovered(&self) -> bool {
        !self.metadata().models.is_empty()
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

    /// Get the metric label for this connection mode
    pub fn as_metric_label(&self) -> &'static str {
        match self {
            ConnectionMode::Http => metrics_labels::CONNECTION_HTTP,
            ConnectionMode::Grpc { .. } => metrics_labels::CONNECTION_GRPC,
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

/// Runtime implementation type for workers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeType {
    /// SGLang runtime (default)
    #[default]
    Sglang,
    /// vLLM runtime
    Vllm,
    /// External OpenAI-compatible API (not local inference)
    /// Used for routing to external providers like OpenAI, Azure OpenAI, xAI, etc.
    External,
}

impl fmt::Display for RuntimeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeType::Sglang => write!(f, "sglang"),
            RuntimeType::Vllm => write!(f, "vllm"),
            RuntimeType::External => write!(f, "external"),
        }
    }
}

impl std::str::FromStr for RuntimeType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Use eq_ignore_ascii_case to avoid to_lowercase() allocation
        if s.eq_ignore_ascii_case("sglang") {
            Ok(RuntimeType::Sglang)
        } else if s.eq_ignore_ascii_case("vllm") {
            Ok(RuntimeType::Vllm)
        } else if s.eq_ignore_ascii_case("external") {
            Ok(RuntimeType::External)
        } else {
            Err(format!("Unknown runtime type: {}", s))
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

impl WorkerType {
    /// Get the metric label for this worker type
    pub fn as_metric_label(&self) -> &'static str {
        match self {
            WorkerType::Regular => metrics_labels::WORKER_REGULAR,
            WorkerType::Prefill { .. } => metrics_labels::WORKER_PREFILL,
            WorkerType::Decode => metrics_labels::WORKER_DECODE,
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
    /// Models this worker can serve.
    /// If empty, worker accepts any model (backward compatible behavior).
    pub models: Vec<ModelCard>,
    /// Default provider for this worker (used when model doesn't specify one).
    /// `None` means native/passthrough.
    pub default_provider: Option<ProviderType>,
    /// Default model type for unknown models (defaults to LLM capabilities).
    pub default_model_type: ModelType,
}

impl WorkerMetadata {
    /// Find a model card by ID (including aliases)
    pub fn find_model(&self, model_id: &str) -> Option<&ModelCard> {
        self.models.iter().find(|m| m.matches(model_id))
    }

    /// Check if this worker can serve a given model.
    /// If models list is empty, worker accepts any model (backward compatible).
    pub fn supports_model(&self, model_id: &str) -> bool {
        self.models.is_empty() || self.find_model(model_id).is_some()
    }

    /// Check if this worker supports an endpoint for a given model.
    /// Falls back to default_model_type if model not found.
    pub fn supports_endpoint(&self, model_id: &str, endpoint: Endpoint) -> bool {
        if let Some(model) = self.find_model(model_id) {
            model.supports_endpoint(endpoint)
        } else {
            self.default_model_type.supports_endpoint(endpoint)
        }
    }

    /// Get the provider for a given model.
    /// Returns the model's provider if found, otherwise the worker's default provider.
    pub fn provider_for_model(&self, model_id: &str) -> Option<&ProviderType> {
        self.find_model(model_id)
            .and_then(|m| m.provider.as_ref())
            .or(self.default_provider.as_ref())
    }

    /// Get all model IDs this worker can serve
    pub fn model_ids(&self) -> impl Iterator<Item = &str> {
        self.models.iter().map(|m| m.id.as_str())
    }
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
    /// Lazily initialized gRPC client for gRPC workers.
    /// Uses OnceCell for lock-free reads after initialization.
    pub grpc_client: Arc<OnceCell<Arc<GrpcClient>>>,
    /// Runtime-mutable models override (for lazy discovery)
    /// When set, overrides metadata.models for routing decisions.
    /// Uses std::sync::RwLock for synchronous access in supports_model().
    pub models_override: Arc<StdRwLock<Option<Vec<ModelCard>>>>,
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
        // Use rfind directly - no need for redundant contains() check
        // rfind already returns None if '@' is not found
        // e.g., "http://[::1]:8080@0" -> "http://[::1]:8080" and "0"
        if let Some(at_pos) = self.url().rfind('@') {
            let base_url = &self.url()[..at_pos];
            let rank_str = &self.url()[at_pos + 1..];

            // Validate that the rank part is actually a number
            if rank_str.parse::<usize>().is_ok() {
                Ok(base_url)
            } else {
                // The '@' is not a DP rank separator, return full URL
                Ok(self.url())
            }
        } else {
            Ok(self.url())
        }
    }

    fn update_running_requests_metrics(&self) {
        let load = self.load();
        Metrics::set_worker_requests_active(self.url(), load);
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

    fn worker_type(&self) -> &WorkerType {
        &self.metadata.worker_type
    }

    fn connection_mode(&self) -> &ConnectionMode {
        &self.metadata.connection_mode
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Release);
        Metrics::set_worker_health(self.url(), healthy);
    }

    async fn check_health_async(&self) -> WorkerResult<()> {
        let health_result = match &self.metadata.connection_mode {
            ConnectionMode::Http => self.http_health_check().await?,
            ConnectionMode::Grpc { .. } => self.grpc_health_check().await?,
        };

        // Get worker type label for metrics
        let worker_type_str = self.metadata.worker_type.as_metric_label();

        if health_result {
            self.consecutive_failures.store(0, Ordering::Release);
            let successes = self.consecutive_successes.fetch_add(1, Ordering::AcqRel) + 1;

            // Record health check success metric
            Metrics::record_worker_health_check(worker_type_str, metrics_labels::CB_SUCCESS);

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

            // Record health check failure metric
            Metrics::record_worker_health_check(worker_type_str, metrics_labels::CB_FAILURE);

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
        self.update_running_requests_metrics();
    }

    fn decrement_load(&self) {
        if self
            .load_counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_sub(1)
            })
            .is_err()
        {
            tracing::warn!(
                worker_url = %self.metadata.url,
                "Attempted to decrement load counter that is already at 0"
            );
        }
        self.update_running_requests_metrics();
    }

    fn reset_load(&self) {
        self.load_counter.store(0, Ordering::Relaxed);
        self.update_running_requests_metrics();
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

    fn supports_model(&self, model_id: &str) -> bool {
        // Check models_override first (for lazy discovery)
        if let Ok(guard) = self.models_override.read() {
            if let Some(ref models) = *guard {
                // Models were discovered - check if this model is supported
                return models.iter().any(|m| m.matches(model_id));
            }
        }
        // Fall back to metadata.models (empty = wildcard = supports nothing until discovery)
        self.metadata.supports_model(model_id)
    }

    fn set_models(&self, models: Vec<ModelCard>) {
        if let Ok(mut guard) = self.models_override.write() {
            tracing::debug!(
                "Setting {} models for worker {} via lazy discovery",
                models.len(),
                self.metadata.url
            );
            *guard = Some(models);
        }
    }

    fn has_models_discovered(&self) -> bool {
        // Check if models_override has been set
        if let Ok(guard) = self.models_override.read() {
            if guard.is_some() {
                return true;
            }
        }
        // Fall back to checking metadata.models
        !self.metadata.models.is_empty()
    }

    async fn get_grpc_client(&self) -> WorkerResult<Option<Arc<GrpcClient>>> {
        match self.metadata.connection_mode {
            ConnectionMode::Http => Ok(None),
            ConnectionMode::Grpc { .. } => {
                // OnceCell provides lock-free reads after initialization.
                // get_or_try_init only acquires internal lock on first call.
                let client = self
                    .grpc_client
                    .get_or_try_init(|| async {
                        let runtime_str = self.metadata.runtime_type.to_string();
                        tracing::info!(
                            "Lazily initializing gRPC client ({}) for worker: {}",
                            runtime_str,
                            self.metadata.url
                        );
                        match GrpcClient::connect(&self.metadata.url, &runtime_str).await {
                            Ok(client) => {
                                tracing::info!(
                                    "Successfully connected gRPC client ({}) for worker: {}",
                                    runtime_str,
                                    self.metadata.url
                                );
                                Ok(Arc::new(client))
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
                    })
                    .await?;
                Ok(Some(Arc::clone(client)))
            }
        }
    }

    async fn reset_grpc_client(&self) -> WorkerResult<()> {
        // OnceCell doesn't support resetting. This is intentional for lock-free performance.
        // If a connection fails, the worker should be removed and re-added.
        tracing::debug!(
            "reset_grpc_client called for {} (no-op with OnceCell)",
            self.metadata.url
        );
        Ok(())
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

    fn worker_type(&self) -> &WorkerType {
        self.base_worker.worker_type()
    }

    fn connection_mode(&self) -> &ConnectionMode {
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

/// RAII guard for worker load management
///
/// Automatically decrements worker load when dropped. Can be attached to
/// an axum Response to tie the guard's lifetime to the response body,
/// which is essential for streaming responses where the function returns
/// immediately but the stream continues in the background.
pub struct WorkerLoadGuard {
    worker: Arc<dyn Worker>,
}

impl WorkerLoadGuard {
    pub fn new(worker: Arc<dyn Worker>) -> Self {
        worker.increment_load();
        Self { worker }
    }

    /// Attach this guard to a Response, tying the guard's lifetime to the response body.
    ///
    /// When the response body is fully consumed or dropped (e.g., client disconnects),
    /// the guard is dropped and worker load is decremented automatically.
    ///
    /// This is the proper RAII pattern for SSE/streaming responses where the handler
    /// returns immediately but the stream continues in a background task.
    pub fn attach_to_response(
        self,
        response: axum::response::Response,
    ) -> axum::response::Response {
        let (parts, body) = response.into_parts();

        // Wrap body with guard - guard drops when body drops
        let guarded_body = GuardedBody {
            inner: body,
            _guard: self,
        };

        axum::response::Response::from_parts(parts, Body::new(guarded_body))
    }
}

impl Drop for WorkerLoadGuard {
    fn drop(&mut self) {
        self.worker.decrement_load();
    }
}

/// Attach multiple guards to a Response (for dual prefill/decode workers)
pub fn attach_guards_to_response(
    guards: Vec<WorkerLoadGuard>,
    response: axum::response::Response,
) -> axum::response::Response {
    let (parts, body) = response.into_parts();

    let guarded_body = MultiGuardedBody {
        inner: body,
        _guards: guards,
    };

    axum::response::Response::from_parts(parts, Body::new(guarded_body))
}

/// Body wrapper that holds a WorkerLoadGuard
///
/// When this body is dropped (stream ends or client disconnects),
/// the guard is dropped, decrementing worker load.
struct GuardedBody {
    inner: Body,
    _guard: WorkerLoadGuard,
}

/// Body wrapper that holds multiple WorkerLoadGuards (for dual prefill/decode)
struct MultiGuardedBody {
    inner: Body,
    _guards: Vec<WorkerLoadGuard>,
}

impl http_body::Body for GuardedBody {
    type Data = bytes::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        std::pin::Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

impl http_body::Body for MultiGuardedBody {
    type Data = bytes::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        std::pin::Pin::new(&mut self.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// Health checker handle with graceful shutdown
pub(crate) struct HealthChecker {
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub async fn shutdown(self) {
        self.shutdown.store(true, Ordering::Release);
        let _ = self.handle.await;
    }
}

/// Helper to convert Worker trait object to WorkerInfo struct
pub fn worker_to_info(worker: &Arc<dyn Worker>) -> WorkerInfo {
    // Cache references that are used multiple times to avoid redundant method calls
    let worker_type = worker.worker_type();
    let connection_mode = worker.connection_mode();
    let url = worker.url();
    let model_id = worker.model_id();

    let worker_type_str = match worker_type {
        WorkerType::Regular => "regular",
        WorkerType::Prefill { .. } => "prefill",
        WorkerType::Decode => "decode",
    };

    let bootstrap_port = match worker_type {
        WorkerType::Prefill { bootstrap_port } => *bootstrap_port,
        _ => None,
    };

    let runtime_type = match connection_mode {
        ConnectionMode::Grpc { .. } => Some(worker.metadata().runtime_type.to_string()),
        ConnectionMode::Http => None,
    };

    WorkerInfo {
        id: url.to_string(),
        url: url.to_string(),
        model_id: model_id.to_string(),
        priority: worker.priority(),
        cost: worker.cost(),
        worker_type: worker_type_str.to_string(),
        is_healthy: worker.is_healthy(),
        load: worker.load(),
        connection_mode: connection_mode.to_string(),
        runtime_type,
        tokenizer_path: worker.tokenizer_path(model_id).map(String::from),
        reasoning_parser: worker.reasoning_parser(model_id).map(String::from),
        tool_parser: worker.tool_parser(model_id).map(String::from),
        chat_template: worker.chat_template(model_id).map(String::from),
        bootstrap_port,
        metadata: worker.metadata().labels.clone(),
        job_status: None,
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use super::*;
    use crate::core::{
        circuit_breaker::{CircuitBreakerConfig, CircuitState},
        DPAwareWorkerBuilder,
    };

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
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
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
        assert_eq!(regular.worker_type(), &WorkerType::Regular);

        let prefill = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(9090),
            })
            .build();
        assert_eq!(
            prefill.worker_type(),
            &WorkerType::Prefill {
                bootstrap_port: Some(9090)
            }
        );

        let decode = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Decode)
            .build();
        assert_eq!(decode.worker_type(), &WorkerType::Decode);
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
        use crate::core::BasicWorkerBuilder;
        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://regular:8080")
                .worker_type(WorkerType::Regular)
                .build(),
        );
        assert_eq!(worker.url(), "http://regular:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
    }

    #[test]
    fn test_create_prefill_worker() {
        use crate::core::BasicWorkerBuilder;
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
            &WorkerType::Prefill {
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
            &WorkerType::Prefill {
                bootstrap_port: None
            }
        );
    }

    #[test]
    fn test_create_decode_worker() {
        use crate::core::BasicWorkerBuilder;
        let worker: Box<dyn Worker> = Box::new(
            BasicWorkerBuilder::new("http://decode:8080")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        assert_eq!(worker.url(), "http://decode:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Decode);
    }

    #[tokio::test]
    async fn test_check_health_async() {
        use crate::core::BasicWorkerBuilder;
        let worker = BasicWorkerBuilder::new("http://test:8080")
            .worker_type(WorkerType::Regular)
            .build();

        // Health check should fail since there's no actual server
        let result = worker.check_health_async().await;
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
        assert_eq!(dp_worker.worker_type(), &WorkerType::Regular);
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
            &WorkerType::Prefill {
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
        assert_eq!(dp_worker.worker_type(), &WorkerType::Decode);
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
        use crate::core::BasicWorkerBuilder;
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
        let dp_aware_regular: Box<dyn Worker> = Box::new(
            DPAwareWorkerBuilder::new("http://dp:8080", 0, 2)
                .worker_type(WorkerType::Regular)
                .api_key("test_api_key")
                .build(),
        );
        let dp_aware_prefill: Box<dyn Worker> = Box::new(
            DPAwareWorkerBuilder::new("http://dp-prefill:8080", 1, 2)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: None,
                })
                .api_key("test_api_key")
                .build(),
        );
        let dp_aware_decode: Box<dyn Worker> = Box::new(
            DPAwareWorkerBuilder::new("http://dp-decode:8080", 0, 4)
                .worker_type(WorkerType::Decode)
                .api_key("test_api_key")
                .build(),
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

        assert_eq!(workers[0].worker_type(), &WorkerType::Regular);
        assert_eq!(
            workers[1].worker_type(),
            &WorkerType::Prefill {
                bootstrap_port: Some(9090)
            }
        );
        assert_eq!(workers[2].worker_type(), &WorkerType::Decode);
        assert_eq!(workers[3].worker_type(), &WorkerType::Regular);
        assert_eq!(
            workers[4].worker_type(),
            &WorkerType::Prefill {
                bootstrap_port: None
            }
        );
        assert_eq!(workers[5].worker_type(), &WorkerType::Decode);
    }

    // === Phase 1.3: WorkerMetadata model methods tests ===

    #[test]
    fn test_worker_metadata_empty_models_accepts_all() {
        let metadata = WorkerMetadata {
            url: "http://test:8080".to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
            runtime_type: RuntimeType::default(),
            labels: std::collections::HashMap::new(),
            health_config: HealthConfig::default(),
            api_key: None,
            bootstrap_host: "test".to_string(),
            bootstrap_port: None,
            models: Vec::new(), // Empty = accepts any model
            default_provider: None,
            default_model_type: ModelType::LLM,
        };

        // Empty models list should accept any model
        assert!(metadata.supports_model("any-model"));
        assert!(metadata.supports_model("gpt-4"));
        assert!(metadata.supports_model("llama-3.1"));
    }

    #[test]
    fn test_worker_metadata_find_model() {
        use super::ModelCard;

        let model1 = ModelCard::new("meta-llama/Llama-3.1-8B")
            .with_alias("llama-3.1-8b")
            .with_alias("llama3.1");
        let model2 = ModelCard::new("gpt-4o");

        let metadata = WorkerMetadata {
            url: "http://test:8080".to_string(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
            runtime_type: RuntimeType::default(),
            labels: std::collections::HashMap::new(),
            health_config: HealthConfig::default(),
            api_key: None,
            bootstrap_host: "test".to_string(),
            bootstrap_port: None,
            models: vec![model1, model2],
            default_provider: None,
            default_model_type: ModelType::LLM,
        };

        // Find by primary ID
        assert!(metadata.find_model("meta-llama/Llama-3.1-8B").is_some());
        assert!(metadata.find_model("gpt-4o").is_some());

        // Find by alias
        assert!(metadata.find_model("llama-3.1-8b").is_some());
        assert!(metadata.find_model("llama3.1").is_some());

        // Not found
        assert!(metadata.find_model("unknown-model").is_none());
    }
}
