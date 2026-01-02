use std::collections::HashMap;

use super::{
    circuit_breaker::{CircuitBreaker, CircuitBreakerConfig},
    model_card::ModelCard,
    model_type::ModelType,
    worker::{
        BasicWorker, ConnectionMode, DPAwareWorker, HealthConfig, RuntimeType, WorkerMetadata,
        WorkerType,
    },
};
use crate::routers::grpc::client::GrpcClient;

/// Builder for creating BasicWorker instances with fluent API
pub struct BasicWorkerBuilder {
    url: String,
    api_key: Option<String>,
    worker_type: WorkerType,
    connection_mode: ConnectionMode,
    runtime_type: RuntimeType,
    labels: HashMap<String, String>,
    models: Vec<ModelCard>,
    health_config: HealthConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    grpc_client: Option<GrpcClient>,
}

impl BasicWorkerBuilder {
    /// Create a new builder with only the URL
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            api_key: None,
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
            runtime_type: RuntimeType::default(),
            labels: HashMap::new(),
            models: Vec::new(),
            health_config: HealthConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            grpc_client: None,
        }
    }

    /// Create a new builder with URL and worker type (for backwards compatibility)
    pub fn new_with_type(url: impl Into<String>, worker_type: WorkerType) -> Self {
        Self {
            url: url.into(),
            api_key: None,
            worker_type,
            connection_mode: ConnectionMode::Http,
            runtime_type: RuntimeType::default(),
            labels: HashMap::new(),
            models: Vec::new(),
            health_config: HealthConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            grpc_client: None,
        }
    }

    /// Set the API key
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the worker type (Regular, Prefill, or Decode)
    pub fn worker_type(mut self, worker_type: WorkerType) -> Self {
        self.worker_type = worker_type;
        self
    }

    /// Set the connection mode (HTTP or gRPC)
    pub fn connection_mode(mut self, mode: ConnectionMode) -> Self {
        self.connection_mode = mode;
        self
    }

    /// Set the runtime type (SGLang or vLLM)
    pub fn runtime_type(mut self, runtime_type: RuntimeType) -> Self {
        self.runtime_type = runtime_type;
        self
    }

    /// Set labels for worker identification
    pub fn labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels = labels;
        self
    }

    /// Add a single label
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Set health check configuration
    pub fn health_config(mut self, config: HealthConfig) -> Self {
        self.health_config = config;
        self
    }

    /// Set circuit breaker configuration
    pub fn circuit_breaker_config(mut self, config: CircuitBreakerConfig) -> Self {
        self.circuit_breaker_config = config;
        self
    }

    /// Set gRPC client for gRPC workers
    pub fn grpc_client(mut self, client: GrpcClient) -> Self {
        self.grpc_client = Some(client);
        self
    }

    /// Set models this worker can serve
    pub fn models(mut self, models: Vec<ModelCard>) -> Self {
        self.models = models;
        self
    }

    /// Add a single model this worker can serve
    pub fn model(mut self, model: ModelCard) -> Self {
        self.models.push(model);
        self
    }

    /// Build the BasicWorker instance
    pub fn build(self) -> BasicWorker {
        use std::sync::{
            atomic::{AtomicBool, AtomicUsize},
            Arc, RwLock as StdRwLock,
        };

        use tokio::sync::RwLock;

        let bootstrap_host = match url::Url::parse(&self.url) {
            Ok(parsed) => parsed.host_str().unwrap_or("localhost").to_string(),
            Err(_) if !self.url.contains("://") => {
                match url::Url::parse(&format!("http://{}", self.url)) {
                    Ok(parsed) => parsed.host_str().unwrap_or("localhost").to_string(),
                    Err(_) => {
                        tracing::warn!(
                            "Failed to parse URL '{}', defaulting to localhost",
                            self.url
                        );
                        "localhost".to_string()
                    }
                }
            }
            Err(_) => {
                tracing::warn!(
                    "Failed to parse URL '{}', defaulting to localhost",
                    self.url
                );
                "localhost".to_string()
            }
        };

        let bootstrap_port = match self.worker_type {
            WorkerType::Prefill { bootstrap_port } => bootstrap_port,
            _ => None,
        };

        let metadata = WorkerMetadata {
            url: self.url.clone(),
            api_key: self.api_key,
            worker_type: self.worker_type,
            connection_mode: self.connection_mode,
            runtime_type: self.runtime_type,
            labels: self.labels,
            health_config: self.health_config,
            bootstrap_host,
            bootstrap_port,
            models: self.models,                // Empty = accepts any model
            default_provider: None,             // Native/passthrough
            default_model_type: ModelType::LLM, // Standard LLM capabilities
        };

        let grpc_client = Arc::new(RwLock::new(self.grpc_client.map(Arc::new)));

        BasicWorker {
            metadata,
            load_counter: Arc::new(AtomicUsize::new(0)),
            processed_counter: Arc::new(AtomicUsize::new(0)),
            healthy: Arc::new(AtomicBool::new(true)),
            consecutive_failures: Arc::new(AtomicUsize::new(0)),
            consecutive_successes: Arc::new(AtomicUsize::new(0)),
            circuit_breaker: CircuitBreaker::with_config(self.circuit_breaker_config),
            grpc_client,
            models_override: Arc::new(StdRwLock::new(None)),
        }
    }
}

/// Builder for creating DPAwareWorker instances with fluent API
pub struct DPAwareWorkerBuilder {
    base_url: String,
    api_key: Option<String>,
    dp_rank: usize,
    dp_size: usize,
    worker_type: WorkerType,
    connection_mode: ConnectionMode,
    runtime_type: RuntimeType,
    labels: HashMap<String, String>,
    models: Vec<ModelCard>,
    health_config: HealthConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    grpc_client: Option<GrpcClient>,
}

impl DPAwareWorkerBuilder {
    /// Create a new DP-aware worker builder
    pub fn new(base_url: impl Into<String>, dp_rank: usize, dp_size: usize) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            dp_rank,
            dp_size,
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
            runtime_type: RuntimeType::default(),
            labels: HashMap::new(),
            models: Vec::new(),
            health_config: HealthConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            grpc_client: None,
        }
    }

    /// Create a new DP-aware worker builder with worker type (for backwards compatibility)
    pub fn new_with_type(
        base_url: impl Into<String>,
        dp_rank: usize,
        dp_size: usize,
        worker_type: WorkerType,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: None,
            dp_rank,
            dp_size,
            worker_type,
            connection_mode: ConnectionMode::Http,
            runtime_type: RuntimeType::default(),
            labels: HashMap::new(),
            models: Vec::new(),
            health_config: HealthConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            grpc_client: None,
        }
    }

    /// Set the API key
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the worker type (Regular, Prefill, or Decode)
    pub fn worker_type(mut self, worker_type: WorkerType) -> Self {
        self.worker_type = worker_type;
        self
    }

    /// Set the connection mode (HTTP or gRPC)
    pub fn connection_mode(mut self, mode: ConnectionMode) -> Self {
        self.connection_mode = mode;
        self
    }

    /// Set the runtime type (SGLang or vLLM)
    pub fn runtime_type(mut self, runtime_type: RuntimeType) -> Self {
        self.runtime_type = runtime_type;
        self
    }

    /// Set labels for worker identification
    pub fn labels(mut self, labels: HashMap<String, String>) -> Self {
        self.labels = labels;
        self
    }

    /// Add a single label
    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Set health check configuration
    pub fn health_config(mut self, config: HealthConfig) -> Self {
        self.health_config = config;
        self
    }

    /// Set circuit breaker configuration
    pub fn circuit_breaker_config(mut self, config: CircuitBreakerConfig) -> Self {
        self.circuit_breaker_config = config;
        self
    }

    /// Set gRPC client for gRPC workers
    pub fn grpc_client(mut self, client: GrpcClient) -> Self {
        self.grpc_client = Some(client);
        self
    }

    /// Set models this worker can serve
    pub fn models(mut self, models: Vec<ModelCard>) -> Self {
        self.models = models;
        self
    }

    /// Add a single model this worker can serve
    pub fn model(mut self, model: ModelCard) -> Self {
        self.models.push(model);
        self
    }

    /// Build the DPAwareWorker instance
    pub fn build(self) -> DPAwareWorker {
        let worker_url = format!("{}@{}", self.base_url, self.dp_rank);
        let mut builder = BasicWorkerBuilder::new(worker_url)
            .models(self.models)
            .worker_type(self.worker_type)
            .connection_mode(self.connection_mode)
            .runtime_type(self.runtime_type)
            .labels(self.labels)
            .health_config(self.health_config)
            .circuit_breaker_config(self.circuit_breaker_config);

        if let Some(client) = self.grpc_client {
            builder = builder.grpc_client(client);
        }
        if let Some(api_key) = self.api_key {
            builder = builder.api_key(api_key);
        }

        let base_worker = builder.build();
        DPAwareWorker::with_base_worker(base_worker, self.base_url, self.dp_rank, self.dp_size)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::core::worker::Worker;

    #[test]
    fn test_basic_worker_builder_minimal() {
        let worker = BasicWorkerBuilder::new("http://localhost:8080").build();

        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
        assert_eq!(worker.connection_mode(), &ConnectionMode::Http);
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_basic_worker_builder_with_type() {
        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .worker_type(WorkerType::Decode)
            .build();

        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), &WorkerType::Decode);
        assert_eq!(worker.connection_mode(), &ConnectionMode::Http);
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_basic_worker_builder_full() {
        let mut labels = HashMap::new();
        labels.insert("env".to_string(), "prod".to_string());
        labels.insert("region".to_string(), "us-east".to_string());

        let health_config = HealthConfig {
            endpoint: "/health".to_string(),
            timeout_secs: 30,
            check_interval_secs: 60,
            failure_threshold: 3,
            success_threshold: 2,
        };

        let cb_config = CircuitBreakerConfig {
            failure_threshold: 10,
            success_threshold: 5,
            timeout_duration: Duration::from_millis(2000),
            window_duration: Duration::from_millis(30000),
        };

        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .worker_type(WorkerType::Prefill {
                bootstrap_port: None,
            })
            .connection_mode(ConnectionMode::Grpc { port: Some(50051) })
            .labels(labels.clone())
            .health_config(health_config.clone())
            .circuit_breaker_config(cb_config)
            .build();

        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(
            worker.worker_type(),
            &WorkerType::Prefill {
                bootstrap_port: None
            }
        );
        assert_eq!(
            worker.connection_mode(),
            &ConnectionMode::Grpc { port: Some(50051) }
        );
        assert_eq!(worker.metadata().labels, labels);
        assert_eq!(
            worker.metadata().health_config.endpoint,
            health_config.endpoint
        );
        assert_eq!(
            worker.metadata().health_config.timeout_secs,
            health_config.timeout_secs
        );
        assert_eq!(
            worker.metadata().health_config.check_interval_secs,
            health_config.check_interval_secs
        );
        assert_eq!(
            worker.metadata().health_config.failure_threshold,
            health_config.failure_threshold
        );
        assert_eq!(
            worker.metadata().health_config.success_threshold,
            health_config.success_threshold
        );
    }

    #[test]
    fn test_basic_worker_builder_with_single_label() {
        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .worker_type(WorkerType::Decode)
            .label("env", "staging")
            .label("version", "v1.2.3")
            .build();

        assert_eq!(
            worker.metadata().labels.get("env"),
            Some(&"staging".to_string())
        );
        assert_eq!(
            worker.metadata().labels.get("version"),
            Some(&"v1.2.3".to_string())
        );
    }

    #[test]
    fn test_dp_aware_worker_builder_minimal() {
        let worker = DPAwareWorkerBuilder::new("http://localhost:8080", 2, 8).build();

        assert_eq!(worker.url(), "http://localhost:8080@2");
        assert_eq!(worker.dp_rank(), Some(2));
        assert_eq!(worker.dp_size(), Some(8));
        assert_eq!(worker.worker_type(), &WorkerType::Regular);
    }

    #[test]
    fn test_dp_aware_worker_builder_full() {
        let mut labels = HashMap::new();
        labels.insert("cluster".to_string(), "main".to_string());

        let health_config = HealthConfig {
            endpoint: "/status".to_string(),
            timeout_secs: 20,
            check_interval_secs: 45,
            failure_threshold: 5,
            success_threshold: 3,
        };

        let worker = DPAwareWorkerBuilder::new("http://localhost:8080", 3, 16)
            .worker_type(WorkerType::Prefill {
                bootstrap_port: Some(9090),
            })
            .connection_mode(ConnectionMode::Http)
            .labels(labels.clone())
            .health_config(health_config.clone())
            .api_key("test_api_key")
            .build();

        assert_eq!(worker.url(), "http://localhost:8080@3");
        assert_eq!(worker.dp_rank(), Some(3));
        assert_eq!(worker.dp_size(), Some(16));
        assert_eq!(worker.metadata().labels, labels);
        assert_eq!(
            worker.metadata().health_config.endpoint,
            health_config.endpoint
        );
        assert_eq!(
            worker.metadata().health_config.timeout_secs,
            health_config.timeout_secs
        );
        assert_eq!(
            worker.metadata().health_config.check_interval_secs,
            health_config.check_interval_secs
        );
        assert_eq!(
            worker.metadata().health_config.failure_threshold,
            health_config.failure_threshold
        );
        assert_eq!(
            worker.metadata().health_config.success_threshold,
            health_config.success_threshold
        );
    }

    #[test]
    fn test_dp_aware_worker_with_grpc() {
        let worker = DPAwareWorkerBuilder::new("grpc://cluster.local", 1, 4)
            .worker_type(WorkerType::Decode)
            .connection_mode(ConnectionMode::Grpc { port: Some(50051) })
            .label("transport", "grpc")
            .build();

        assert_eq!(worker.url(), "grpc://cluster.local@1");
        assert_eq!(worker.dp_rank(), Some(1));
        assert_eq!(worker.dp_size(), Some(4));
        assert_eq!(worker.worker_type(), &WorkerType::Decode);
        assert_eq!(
            worker.connection_mode(),
            &ConnectionMode::Grpc { port: Some(50051) }
        );
        assert_eq!(
            worker.metadata().labels.get("transport"),
            Some(&"grpc".to_string())
        );
    }
}
