use super::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
use super::worker::{
    BasicWorker, ConnectionMode, DPAwareWorker, HealthConfig, WorkerMetadata, WorkerType,
};
use crate::grpc::client::SglangSchedulerClient;
use std::collections::HashMap;

/// Builder for creating BasicWorker instances with fluent API
pub struct BasicWorkerBuilder {
    // Required fields
    url: String,

    // Optional fields with defaults
    worker_type: WorkerType,
    connection_mode: ConnectionMode,
    labels: HashMap<String, String>,
    health_config: HealthConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    grpc_client: Option<SglangSchedulerClient>,
}

impl BasicWorkerBuilder {
    /// Create a new builder with only the URL (defaults to Regular worker type)
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
            labels: HashMap::new(),
            health_config: HealthConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            grpc_client: None,
        }
    }

    /// Create a new builder with URL and worker type (for backwards compatibility)
    pub fn new_with_type(url: impl Into<String>, worker_type: WorkerType) -> Self {
        Self {
            url: url.into(),
            worker_type,
            connection_mode: ConnectionMode::Http,
            labels: HashMap::new(),
            health_config: HealthConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            grpc_client: None,
        }
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
    pub fn grpc_client(mut self, client: SglangSchedulerClient) -> Self {
        self.grpc_client = Some(client);
        self
    }

    /// Build the BasicWorker instance
    pub fn build(self) -> BasicWorker {
        use std::sync::{
            atomic::{AtomicBool, AtomicUsize},
            Arc,
        };
        use tokio::sync::Mutex;

        let metadata = WorkerMetadata {
            url: self.url.clone(),
            worker_type: self.worker_type,
            connection_mode: self.connection_mode,
            labels: self.labels,
            health_config: self.health_config,
        };

        BasicWorker {
            metadata,
            load_counter: Arc::new(AtomicUsize::new(0)),
            processed_counter: Arc::new(AtomicUsize::new(0)),
            healthy: Arc::new(AtomicBool::new(true)),
            consecutive_failures: Arc::new(AtomicUsize::new(0)),
            consecutive_successes: Arc::new(AtomicUsize::new(0)),
            circuit_breaker: CircuitBreaker::with_config(self.circuit_breaker_config),
            grpc_client: self.grpc_client.map(|client| Arc::new(Mutex::new(client))),
        }
    }
}

/// Builder for creating DPAwareWorker instances with fluent API
pub struct DPAwareWorkerBuilder {
    // Required fields
    base_url: String,
    dp_rank: usize,
    dp_size: usize,

    // Optional fields with defaults
    worker_type: WorkerType,
    connection_mode: ConnectionMode,
    labels: HashMap<String, String>,
    health_config: HealthConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    grpc_client: Option<SglangSchedulerClient>,
}

impl DPAwareWorkerBuilder {
    /// Create a new DP-aware worker builder (defaults to Regular worker type)
    pub fn new(base_url: impl Into<String>, dp_rank: usize, dp_size: usize) -> Self {
        Self {
            base_url: base_url.into(),
            dp_rank,
            dp_size,
            worker_type: WorkerType::Regular,
            connection_mode: ConnectionMode::Http,
            labels: HashMap::new(),
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
            dp_rank,
            dp_size,
            worker_type,
            connection_mode: ConnectionMode::Http,
            labels: HashMap::new(),
            health_config: HealthConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            grpc_client: None,
        }
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
    pub fn grpc_client(mut self, client: SglangSchedulerClient) -> Self {
        self.grpc_client = Some(client);
        self
    }

    /// Build the DPAwareWorker instance
    pub fn build(self) -> DPAwareWorker {
        // Create URL with DP rank suffix for identification
        let worker_url = format!("{}@{}", self.base_url, self.dp_rank);

        // Use BasicWorkerBuilder to create a properly configured base worker
        let mut builder = BasicWorkerBuilder::new(worker_url)
            .worker_type(self.worker_type)
            .connection_mode(self.connection_mode)
            .labels(self.labels)
            .health_config(self.health_config)
            .circuit_breaker_config(self.circuit_breaker_config);

        // Add gRPC client if provided
        if let Some(client) = self.grpc_client {
            builder = builder.grpc_client(client);
        }

        let base_worker = builder.build();

        // Create the DPAwareWorker with the configured base worker
        DPAwareWorker::with_base_worker(base_worker, self.base_url, self.dp_rank, self.dp_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::worker::Worker;
    use std::time::Duration;

    #[test]
    fn test_basic_worker_builder_minimal() {
        // Using new API - defaults to Regular type
        let worker = BasicWorkerBuilder::new("http://localhost:8080").build();

        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), WorkerType::Regular);
        assert_eq!(worker.connection_mode(), ConnectionMode::Http);
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_basic_worker_builder_with_type() {
        // Test setting worker type explicitly
        let worker = BasicWorkerBuilder::new("http://localhost:8080")
            .worker_type(WorkerType::Decode)
            .build();

        assert_eq!(worker.url(), "http://localhost:8080");
        assert_eq!(worker.worker_type(), WorkerType::Decode);
        assert_eq!(worker.connection_mode(), ConnectionMode::Http);
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
            WorkerType::Prefill {
                bootstrap_port: None
            }
        );
        assert_eq!(
            worker.connection_mode(),
            ConnectionMode::Grpc { port: Some(50051) }
        );
        assert_eq!(worker.metadata().labels, labels);
        // Can't directly compare HealthConfig without PartialEq, so check individual fields
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
        // Using new API - defaults to Regular type
        let worker = DPAwareWorkerBuilder::new("http://localhost:8080", 2, 8).build();

        assert_eq!(worker.url(), "http://localhost:8080@2");
        assert_eq!(worker.dp_rank(), Some(2));
        assert_eq!(worker.dp_size(), Some(8));
        // Note: base_url is a private field, we can only test through the url() method
        assert_eq!(worker.worker_type(), WorkerType::Regular);
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
            .build();

        assert_eq!(worker.url(), "http://localhost:8080@3");
        assert_eq!(worker.dp_rank(), Some(3));
        assert_eq!(worker.dp_size(), Some(16));
        assert_eq!(worker.metadata().labels, labels);
        // Can't directly compare HealthConfig without PartialEq, so check individual fields
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
        // Test that DPAwareWorkerBuilder can set a gRPC client
        let worker = DPAwareWorkerBuilder::new("grpc://cluster.local", 1, 4)
            .worker_type(WorkerType::Decode)
            .connection_mode(ConnectionMode::Grpc { port: Some(50051) })
            .label("transport", "grpc")
            .build();

        assert_eq!(worker.url(), "grpc://cluster.local@1");
        assert_eq!(worker.dp_rank(), Some(1));
        assert_eq!(worker.dp_size(), Some(4));
        assert_eq!(worker.worker_type(), WorkerType::Decode);
        assert_eq!(
            worker.connection_mode(),
            ConnectionMode::Grpc { port: Some(50051) }
        );
        assert_eq!(
            worker.metadata().labels.get("transport"),
            Some(&"grpc".to_string())
        );

        // Note: We can't directly test the grpc_client as it's private,
        // but the fact that the worker builds successfully with grpc connection mode
        // validates that the configuration is properly passed through
    }
}
