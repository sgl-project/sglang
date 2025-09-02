// gRPC Router Implementation

use crate::config::types::{
    CircuitBreakerConfig as ConfigCircuitBreakerConfig,
    HealthCheckConfig as ConfigHealthCheckConfig, RetryConfig,
};
use crate::core::{
    BasicWorker, CircuitBreakerConfig, HealthChecker, HealthConfig, Worker, WorkerType,
};
use crate::grpc::SglangSchedulerClient;
use crate::metrics::RouterMetrics;
use crate::policies::LoadBalancingPolicy;
use crate::reasoning_parser::ParserFactory;
use crate::routers::{RouterTrait, WorkerManagement};
use crate::tokenizer::{factory, traits::Tokenizer};
use crate::tool_parser::ParserRegistry;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tracing::{info, warn};

/// gRPC router implementation for SGLang
#[allow(dead_code)] // Fields will be used once implementation is complete
pub struct GrpcRouter {
    /// Worker connections
    workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
    /// gRPC clients for each worker
    grpc_clients: Arc<RwLock<HashMap<String, SglangSchedulerClient>>>,
    /// Load balancing policy
    policy: Arc<dyn LoadBalancingPolicy>,
    /// Tokenizer for handling text encoding/decoding
    tokenizer: Arc<dyn Tokenizer>,
    /// Reasoning parser factory for structured reasoning outputs
    reasoning_parser_factory: ParserFactory,
    /// Tool parser registry for function/tool calls
    tool_parser_registry: &'static ParserRegistry,
    /// Worker health checker
    _health_checker: Option<HealthChecker>,
    /// Configuration
    timeout_secs: u64,
    interval_secs: u64,
    dp_aware: bool,
    api_key: Option<String>,
    retry_config: RetryConfig,
    circuit_breaker_config: CircuitBreakerConfig,
}

impl GrpcRouter {
    /// Create a new gRPC router
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        worker_urls: Vec<String>,
        policy: Arc<dyn LoadBalancingPolicy>,
        timeout_secs: u64,
        interval_secs: u64,
        dp_aware: bool,
        api_key: Option<String>,
        retry_config: RetryConfig,
        circuit_breaker_config: ConfigCircuitBreakerConfig,
        health_check_config: ConfigHealthCheckConfig,
        tokenizer_path_or_model: String,
    ) -> Result<Self, String> {
        // Update metrics
        RouterMetrics::set_active_workers(worker_urls.len());

        // Initialize tokenizer
        let tokenizer = factory::create_tokenizer(&tokenizer_path_or_model)
            .map_err(|e| format!("Failed to create tokenizer: {}", e))?;

        // Initialize reasoning parser factory
        let reasoning_parser_factory = ParserFactory::new();

        // Get tool parser registry
        let tool_parser_registry = ParserRegistry::new();

        // Convert config CircuitBreakerConfig to core CircuitBreakerConfig
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Create gRPC clients for each worker
        let mut grpc_clients = HashMap::new();
        for url in &worker_urls {
            match SglangSchedulerClient::connect(url).await {
                Ok(client) => {
                    grpc_clients.insert(url.clone(), client);
                    info!("Connected to gRPC worker at {}", url);
                }
                Err(e) => {
                    warn!("Failed to connect to gRPC worker at {}: {}", url, e);
                    // Continue with other workers
                }
            }
        }

        if grpc_clients.is_empty() {
            return Err("Failed to connect to any gRPC workers".to_string());
        }

        // Create Worker trait objects with gRPC connection mode
        let workers: Vec<Box<dyn Worker>> = worker_urls
            .iter()
            .map(|url| {
                let worker = BasicWorker::with_connection_mode(
                    url.clone(),
                    WorkerType::Regular,
                    crate::core::ConnectionMode::Grpc { port: None },
                )
                .with_circuit_breaker_config(core_cb_config.clone())
                .with_health_config(HealthConfig {
                    timeout_secs: health_check_config.timeout_secs,
                    check_interval_secs: health_check_config.check_interval_secs,
                    endpoint: health_check_config.endpoint.clone(),
                    failure_threshold: health_check_config.failure_threshold,
                    success_threshold: health_check_config.success_threshold,
                });
                Box::new(worker) as Box<dyn Worker>
            })
            .collect();

        // Initialize policy with workers if needed
        if let Some(cache_aware) = policy
            .as_any()
            .downcast_ref::<crate::policies::CacheAwarePolicy>()
        {
            cache_aware.init_workers(&workers);
        }

        let workers = Arc::new(RwLock::new(workers));
        let health_checker = crate::core::start_health_checker(Arc::clone(&workers), interval_secs);

        Ok(GrpcRouter {
            workers,
            grpc_clients: Arc::new(RwLock::new(grpc_clients)),
            policy,
            tokenizer,
            reasoning_parser_factory,
            tool_parser_registry,
            _health_checker: Some(health_checker),
            timeout_secs,
            interval_secs,
            dp_aware,
            api_key,
            retry_config,
            circuit_breaker_config: core_cb_config,
        })
    }
}

impl std::fmt::Debug for GrpcRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrpcRouter")
            .field("workers_count", &self.workers.read().unwrap().len())
            .field("timeout_secs", &self.timeout_secs)
            .field("interval_secs", &self.interval_secs)
            .field("dp_aware", &self.dp_aware)
            .finish()
    }
}

#[async_trait]
impl RouterTrait for GrpcRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::GenerateRequest,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::ChatCompletionRequest,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::CompletionRequest,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_embeddings(&self, _headers: Option<&HeaderMap>, _body: Body) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_rerank(&self, _headers: Option<&HeaderMap>, _body: Body) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn flush_cache(&self) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_worker_loads(&self) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    fn router_type(&self) -> &'static str {
        "grpc"
    }

    fn readiness(&self) -> Response {
        (StatusCode::SERVICE_UNAVAILABLE).into_response()
    }
}

#[async_trait]
impl WorkerManagement for GrpcRouter {
    async fn add_worker(&self, _worker_url: &str) -> Result<String, String> {
        Err("Not implemented".to_string())
    }

    fn remove_worker(&self, _worker_url: &str) {}

    fn get_worker_urls(&self) -> Vec<String> {
        vec![]
    }
}
