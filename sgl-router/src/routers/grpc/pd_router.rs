// PD (Prefill-Decode) gRPC Router Implementation

use crate::config::types::RetryConfig;
use crate::core::{
    BasicWorker, CircuitBreakerConfig, HealthChecker, HealthConfig, Worker, WorkerType,
};
use crate::grpc::SglangSchedulerClient;
use crate::metrics::RouterMetrics;
use crate::policies::LoadBalancingPolicy;
use crate::reasoning_parser::ParserFactory;
use crate::routers::{RouterTrait, WorkerManagement};
use crate::tokenizer::traits::Tokenizer;
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

/// gRPC PD (Prefill-Decode) router implementation for SGLang
#[allow(dead_code)] // Fields will be used once implementation is complete
pub struct GrpcPDRouter {
    /// Prefill worker connections
    prefill_workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
    /// Decode worker connections
    decode_workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
    /// gRPC clients for prefill workers
    prefill_grpc_clients: Arc<RwLock<HashMap<String, SglangSchedulerClient>>>,
    /// gRPC clients for decode workers
    decode_grpc_clients: Arc<RwLock<HashMap<String, SglangSchedulerClient>>>,
    /// Load balancing policy for prefill
    prefill_policy: Arc<dyn LoadBalancingPolicy>,
    /// Load balancing policy for decode
    decode_policy: Arc<dyn LoadBalancingPolicy>,
    /// Tokenizer for handling text encoding/decoding
    tokenizer: Arc<dyn Tokenizer>,
    /// Reasoning parser factory for structured reasoning outputs
    reasoning_parser_factory: ParserFactory,
    /// Tool parser registry for function/tool calls
    tool_parser_registry: &'static ParserRegistry,
    /// Worker health checkers
    _prefill_health_checker: Option<HealthChecker>,
    _decode_health_checker: Option<HealthChecker>,
    /// Configuration
    timeout_secs: u64,
    interval_secs: u64,
    dp_aware: bool,
    api_key: Option<String>,
    retry_config: RetryConfig,
    circuit_breaker_config: CircuitBreakerConfig,
}

impl GrpcPDRouter {
    /// Create a new gRPC PD router
    pub async fn new(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        prefill_policy: Arc<dyn LoadBalancingPolicy>,
        decode_policy: Arc<dyn LoadBalancingPolicy>,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        // Update metrics
        RouterMetrics::set_active_workers(prefill_urls.len() + decode_urls.len());

        // Extract necessary components from context
        let tokenizer = ctx
            .tokenizer
            .as_ref()
            .ok_or_else(|| "gRPC PD router requires tokenizer".to_string())?
            .clone();
        let reasoning_parser_factory = ctx
            .reasoning_parser_factory
            .as_ref()
            .ok_or_else(|| "gRPC PD router requires reasoning parser factory".to_string())?
            .clone();
        let tool_parser_registry = ctx
            .tool_parser_registry
            .ok_or_else(|| "gRPC PD router requires tool parser registry".to_string())?;

        // Convert config CircuitBreakerConfig to core CircuitBreakerConfig
        let circuit_breaker_config = ctx.router_config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Create gRPC clients for prefill workers
        let mut prefill_grpc_clients = HashMap::new();
        for (url, _bootstrap_port) in &prefill_urls {
            match SglangSchedulerClient::connect(url).await {
                Ok(client) => {
                    prefill_grpc_clients.insert(url.clone(), client);
                    info!("Connected to gRPC prefill worker at {}", url);
                }
                Err(e) => {
                    warn!("Failed to connect to gRPC prefill worker at {}: {}", url, e);
                    // Continue with other workers
                }
            }
        }

        // Create gRPC clients for decode workers
        let mut decode_grpc_clients = HashMap::new();
        for url in &decode_urls {
            match SglangSchedulerClient::connect(url).await {
                Ok(client) => {
                    decode_grpc_clients.insert(url.clone(), client);
                    info!("Connected to gRPC decode worker at {}", url);
                }
                Err(e) => {
                    warn!("Failed to connect to gRPC decode worker at {}: {}", url, e);
                    // Continue with other workers
                }
            }
        }

        if prefill_grpc_clients.is_empty() && decode_grpc_clients.is_empty() {
            return Err("Failed to connect to any gRPC workers".to_string());
        }

        // Create Prefill Worker trait objects with gRPC connection mode
        let prefill_workers: Vec<Box<dyn Worker>> = prefill_urls
            .iter()
            .map(|(url, bootstrap_port)| {
                let worker = BasicWorker::with_connection_mode(
                    url.clone(),
                    WorkerType::Prefill {
                        bootstrap_port: *bootstrap_port,
                    },
                    crate::core::ConnectionMode::Grpc {
                        port: *bootstrap_port,
                    },
                )
                .with_circuit_breaker_config(core_cb_config.clone())
                .with_health_config(HealthConfig {
                    timeout_secs: ctx.router_config.health_check.timeout_secs,
                    check_interval_secs: ctx.router_config.health_check.check_interval_secs,
                    endpoint: ctx.router_config.health_check.endpoint.clone(),
                    failure_threshold: ctx.router_config.health_check.failure_threshold,
                    success_threshold: ctx.router_config.health_check.success_threshold,
                });
                Box::new(worker) as Box<dyn Worker>
            })
            .collect();

        // Create Decode Worker trait objects with gRPC connection mode
        let decode_workers: Vec<Box<dyn Worker>> = decode_urls
            .iter()
            .map(|url| {
                let worker = BasicWorker::with_connection_mode(
                    url.clone(),
                    WorkerType::Decode,
                    crate::core::ConnectionMode::Grpc { port: None },
                )
                .with_circuit_breaker_config(core_cb_config.clone())
                .with_health_config(HealthConfig {
                    timeout_secs: ctx.router_config.health_check.timeout_secs,
                    check_interval_secs: ctx.router_config.health_check.check_interval_secs,
                    endpoint: ctx.router_config.health_check.endpoint.clone(),
                    failure_threshold: ctx.router_config.health_check.failure_threshold,
                    success_threshold: ctx.router_config.health_check.success_threshold,
                });
                Box::new(worker) as Box<dyn Worker>
            })
            .collect();

        // Initialize policies with workers if needed
        if let Some(cache_aware) = prefill_policy
            .as_any()
            .downcast_ref::<crate::policies::CacheAwarePolicy>()
        {
            cache_aware.init_workers(&prefill_workers);
        }

        if let Some(cache_aware) = decode_policy
            .as_any()
            .downcast_ref::<crate::policies::CacheAwarePolicy>()
        {
            cache_aware.init_workers(&decode_workers);
        }

        let prefill_workers = Arc::new(RwLock::new(prefill_workers));
        let decode_workers = Arc::new(RwLock::new(decode_workers));

        let prefill_health_checker = crate::core::start_health_checker(
            Arc::clone(&prefill_workers),
            ctx.router_config.worker_startup_check_interval_secs,
        );
        let decode_health_checker = crate::core::start_health_checker(
            Arc::clone(&decode_workers),
            ctx.router_config.worker_startup_check_interval_secs,
        );

        Ok(GrpcPDRouter {
            prefill_workers,
            decode_workers,
            prefill_grpc_clients: Arc::new(RwLock::new(prefill_grpc_clients)),
            decode_grpc_clients: Arc::new(RwLock::new(decode_grpc_clients)),
            prefill_policy,
            decode_policy,
            tokenizer,
            reasoning_parser_factory,
            tool_parser_registry,
            _prefill_health_checker: Some(prefill_health_checker),
            _decode_health_checker: Some(decode_health_checker),
            timeout_secs: ctx.router_config.worker_startup_timeout_secs,
            interval_secs: ctx.router_config.worker_startup_check_interval_secs,
            dp_aware: ctx.router_config.dp_aware,
            api_key: ctx.router_config.api_key.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            circuit_breaker_config: core_cb_config,
        })
    }
}

impl std::fmt::Debug for GrpcPDRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GrpcPDRouter")
            .field(
                "prefill_workers_count",
                &self.prefill_workers.read().unwrap().len(),
            )
            .field(
                "decode_workers_count",
                &self.decode_workers.read().unwrap().len(),
            )
            .field("timeout_secs", &self.timeout_secs)
            .field("interval_secs", &self.interval_secs)
            .field("dp_aware", &self.dp_aware)
            .finish()
    }
}

#[async_trait]
impl RouterTrait for GrpcPDRouter {
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
        "grpc_pd"
    }

    fn readiness(&self) -> Response {
        (StatusCode::SERVICE_UNAVAILABLE).into_response()
    }
}

#[async_trait]
impl WorkerManagement for GrpcPDRouter {
    async fn add_worker(&self, _worker_url: &str) -> Result<String, String> {
        Err("Not implemented".to_string())
    }

    fn remove_worker(&self, _worker_url: &str) {}

    fn get_worker_urls(&self) -> Vec<String> {
        vec![]
    }
}
