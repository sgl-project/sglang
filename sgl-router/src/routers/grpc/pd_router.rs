// PD (Prefill-Decode) gRPC Router Implementation

use crate::config::types::RetryConfig;
use crate::core::{
    BasicWorkerBuilder, CircuitBreakerConfig, HealthConfig, WorkerRegistry, WorkerType,
};
use crate::grpc_client::SglangSchedulerClient;
use crate::metrics::RouterMetrics;
use crate::policies::{LoadBalancingPolicy, PolicyRegistry};
use crate::reasoning_parser::ParserFactory;
use crate::routers::RouterTrait;
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
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

/// gRPC PD (Prefill-Decode) router implementation for SGLang
#[allow(dead_code)] // Fields will be used once implementation is complete
pub struct GrpcPDRouter {
    /// Centralized worker registry
    worker_registry: Arc<WorkerRegistry>,
    /// Centralized policy registry
    policy_registry: Arc<PolicyRegistry>,
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
        // Get registries from context
        let worker_registry = ctx.worker_registry.clone();
        let policy_registry = ctx.policy_registry.clone();

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

        // Create Prefill Worker trait objects with gRPC connection mode and register them
        for (url, bootstrap_port) in &prefill_urls {
            if let Some(client) = prefill_grpc_clients.remove(url) {
                let worker = BasicWorkerBuilder::new(url.clone())
                    .worker_type(WorkerType::Prefill {
                        bootstrap_port: *bootstrap_port,
                    })
                    .connection_mode(crate::core::ConnectionMode::Grpc {
                        port: *bootstrap_port,
                    })
                    .circuit_breaker_config(core_cb_config.clone())
                    .health_config(HealthConfig {
                        timeout_secs: ctx.router_config.health_check.timeout_secs,
                        check_interval_secs: ctx.router_config.health_check.check_interval_secs,
                        endpoint: ctx.router_config.health_check.endpoint.clone(),
                        failure_threshold: ctx.router_config.health_check.failure_threshold,
                        success_threshold: ctx.router_config.health_check.success_threshold,
                    })
                    .grpc_client(client)
                    .build();

                // Register worker in the centralized registry
                worker_registry.register(Arc::new(worker));
            }
        }

        // Create Decode Worker trait objects with gRPC connection mode and register them
        for url in &decode_urls {
            if let Some(client) = decode_grpc_clients.remove(url) {
                let worker = BasicWorkerBuilder::new(url.clone())
                    .worker_type(WorkerType::Decode)
                    .connection_mode(crate::core::ConnectionMode::Grpc { port: None })
                    .circuit_breaker_config(core_cb_config.clone())
                    .health_config(HealthConfig {
                        timeout_secs: ctx.router_config.health_check.timeout_secs,
                        check_interval_secs: ctx.router_config.health_check.check_interval_secs,
                        endpoint: ctx.router_config.health_check.endpoint.clone(),
                        failure_threshold: ctx.router_config.health_check.failure_threshold,
                        success_threshold: ctx.router_config.health_check.success_threshold,
                    })
                    .grpc_client(client)
                    .build();

                // Register worker in the centralized registry
                worker_registry.register(Arc::new(worker));
            }
        }

        // Initialize policies with workers if needed - filter for gRPC workers only
        let prefill_workers = worker_registry.get_workers_filtered(
            None, // any model
            Some(WorkerType::Prefill {
                bootstrap_port: None,
            }),
            Some(crate::core::ConnectionMode::Grpc { port: None }),
            false, // include unhealthy workers during initialization
        );
        if let Some(cache_aware) = prefill_policy
            .as_any()
            .downcast_ref::<crate::policies::CacheAwarePolicy>()
        {
            cache_aware.init_workers(&prefill_workers);
        }

        let decode_workers = worker_registry.get_workers_filtered(
            None, // any model
            Some(WorkerType::Decode),
            Some(crate::core::ConnectionMode::Grpc { port: None }),
            false, // include unhealthy workers during initialization
        );
        if let Some(cache_aware) = decode_policy
            .as_any()
            .downcast_ref::<crate::policies::CacheAwarePolicy>()
        {
            cache_aware.init_workers(&decode_workers);
        }

        // No need for local health checkers - WorkerRegistry handles health checking

        Ok(GrpcPDRouter {
            worker_registry,
            policy_registry,
            prefill_policy,
            decode_policy,
            tokenizer,
            reasoning_parser_factory,
            tool_parser_registry,
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
        let prefill_workers = self.worker_registry.get_workers_filtered(
            None,
            Some(WorkerType::Prefill {
                bootstrap_port: None,
            }),
            Some(crate::core::ConnectionMode::Grpc { port: None }),
            false,
        );
        let decode_workers = self.worker_registry.get_workers_filtered(
            None,
            Some(WorkerType::Decode),
            Some(crate::core::ConnectionMode::Grpc { port: None }),
            false,
        );
        f.debug_struct("GrpcPDRouter")
            .field("prefill_workers_count", &prefill_workers.len())
            .field("decode_workers_count", &decode_workers.len())
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

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // TODO: Implement actual generation test for gRPC PD mode
        (
            StatusCode::NOT_IMPLEMENTED,
            "Health generate not yet implemented for gRPC PD",
        )
            .into_response()
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
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
        _params: &crate::protocols::spec::ResponsesGetParams,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    async fn route_rerank(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::RerankRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (StatusCode::NOT_IMPLEMENTED).into_response()
    }

    fn router_type(&self) -> &'static str {
        "grpc_pd"
    }
}
