use crate::{
    config::{ConnectionMode, HistoryBackend, RouterConfig},
    core::{WorkerRegistry, WorkerType},
    data_connector::{MemoryResponseStorage, NoOpResponseStorage, SharedResponseStorage},
    logging::{self, LoggingConfig},
    metrics::{self, PrometheusConfig},
    middleware::{self, QueuedRequest, TokenBucket},
    policies::PolicyRegistry,
    protocols::{
        spec::{
            ChatCompletionRequest, CompletionRequest, EmbeddingRequest, GenerateRequest,
            RerankRequest, ResponsesRequest, V1RerankReqInput,
        },
        worker_spec::{WorkerApiResponse, WorkerConfigRequest, WorkerErrorResponse},
    },
    reasoning_parser::ParserFactory,
    routers::WorkerInitializer,
    routers::{
        router_manager::{RouterId, RouterManager},
        RouterFactory, RouterTrait,
    },
    service_discovery::{start_service_discovery, ServiceDiscoveryConfig},
    tokenizer::{factory as tokenizer_factory, traits::Tokenizer},
    tool_parser::ParserRegistry,
};
use axum::{
    extract::{Path, Query, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    serve, Json, Router,
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    sync::Arc,
    time::Duration,
};
use tokio::{net::TcpListener, signal, spawn};
use tracing::{error, info, warn, Level};

#[derive(Clone)]
pub struct AppContext {
    pub client: Client,
    pub router_config: RouterConfig,
    pub rate_limiter: Arc<TokenBucket>,
    pub tokenizer: Option<Arc<dyn Tokenizer>>,
    pub reasoning_parser_factory: Option<ParserFactory>,
    pub tool_parser_registry: Option<&'static ParserRegistry>,
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub router_manager: Option<Arc<RouterManager>>,
    pub response_storage: SharedResponseStorage,
}

impl AppContext {
    pub fn new(
        router_config: RouterConfig,
        client: Client,
        max_concurrent_requests: usize,
        rate_limit_tokens_per_second: Option<usize>,
    ) -> Result<Self, String> {
        let rate_limit_tokens = rate_limit_tokens_per_second.unwrap_or(max_concurrent_requests);
        let rate_limiter = Arc::new(TokenBucket::new(max_concurrent_requests, rate_limit_tokens));

        // Initialize gRPC-specific components only when in gRPC mode
        let (tokenizer, reasoning_parser_factory, tool_parser_registry) =
            if router_config.connection_mode == ConnectionMode::Grpc {
                // Get tokenizer path (required for gRPC mode)
                let tokenizer_path = router_config
                    .tokenizer_path
                    .clone()
                    .or_else(|| router_config.model_path.clone())
                    .ok_or_else(|| {
                        "gRPC mode requires either --tokenizer-path or --model-path to be specified"
                            .to_string()
                    })?;

                // Initialize all gRPC components
                let tokenizer = Some(
                    tokenizer_factory::create_tokenizer(&tokenizer_path)
                        .map_err(|e| format!("Failed to create tokenizer: {e}"))?,
                );
                let reasoning_parser_factory = Some(ParserFactory::new());
                let tool_parser_registry = Some(ParserRegistry::new());

                (tokenizer, reasoning_parser_factory, tool_parser_registry)
            } else {
                // HTTP mode doesn't need these components
                (None, None, None)
            };

        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(router_config.policy.clone()));

        let router_manager = None;

        // Initialize response storage based on configuration
        let response_storage: SharedResponseStorage = match router_config.history_backend {
            HistoryBackend::Memory => Arc::new(MemoryResponseStorage::new()),
            HistoryBackend::None => Arc::new(NoOpResponseStorage::new()),
        };

        Ok(Self {
            client,
            router_config,
            rate_limiter,
            tokenizer,
            reasoning_parser_factory,
            tool_parser_registry,
            worker_registry,
            policy_registry,
            router_manager,
            response_storage,
        })
    }
}

#[derive(Clone)]
pub struct AppState {
    pub router: Arc<dyn RouterTrait>,
    pub context: Arc<AppContext>,
    pub concurrency_queue_tx: Option<tokio::sync::mpsc::Sender<QueuedRequest>>,
    pub router_manager: Option<Arc<RouterManager>>,
}

// Fallback handler for unmatched routes
async fn sink_handler() -> Response {
    StatusCode::NOT_FOUND.into_response()
}

// Health check endpoints
async fn liveness(State(state): State<Arc<AppState>>) -> Response {
    state.router.liveness()
}

async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    state.router.readiness()
}

async fn health(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.health(req).await
}

async fn health_generate(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.health_generate(req).await
}

async fn get_server_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_server_info(req).await
}

async fn v1_models(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_models(req).await
}

async fn get_model_info(State(state): State<Arc<AppState>>, req: Request) -> Response {
    state.router.get_model_info(req).await
}

// Generation endpoints
// The RouterTrait now accepts optional headers and typed body directly
async fn generate(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<GenerateRequest>,
) -> Response {
    state
        .router
        .route_generate(Some(&headers), &body, None)
        .await
}

async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<ChatCompletionRequest>,
) -> Response {
    state.router.route_chat(Some(&headers), &body, None).await
}

async fn v1_completions(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<CompletionRequest>,
) -> Response {
    state
        .router
        .route_completion(Some(&headers), &body, None)
        .await
}

async fn rerank(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<RerankRequest>,
) -> Response {
    state.router.route_rerank(Some(&headers), &body, None).await
}

async fn v1_rerank(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<V1RerankReqInput>,
) -> Response {
    state
        .router
        .route_rerank(Some(&headers), &body.into(), None)
        .await
}

async fn v1_responses(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<ResponsesRequest>,
) -> Response {
    state
        .router
        .route_responses(Some(&headers), &body, None)
        .await
}

async fn v1_embeddings(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<EmbeddingRequest>,
) -> Response {
    state
        .router
        .route_embeddings(Some(&headers), &body, None)
        .await
}

async fn v1_responses_get(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    state
        .router
        .get_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_cancel(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    state
        .router
        .cancel_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_delete(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    // Python server does not support this yet
    state
        .router
        .delete_response(Some(&headers), &response_id)
        .await
}

async fn v1_responses_list_input_items(
    State(state): State<Arc<AppState>>,
    Path(response_id): Path<String>,
    headers: http::HeaderMap,
) -> Response {
    // Python server does not support this yet
    state
        .router
        .list_response_input_items(Some(&headers), &response_id)
        .await
}

// ---------- Worker management endpoints (Legacy) ----------

#[derive(Deserialize)]
struct UrlQuery {
    url: String,
}

async fn add_worker(
    State(state): State<Arc<AppState>>,
    Query(UrlQuery { url }): Query<UrlQuery>,
) -> Response {
    match state.router.add_worker(&url).await {
        Ok(message) => (StatusCode::OK, message).into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, error).into_response(),
    }
}

async fn list_workers(State(state): State<Arc<AppState>>) -> Response {
    let worker_list = state.router.get_worker_urls();
    Json(serde_json::json!({ "urls": worker_list })).into_response()
}

async fn remove_worker(
    State(state): State<Arc<AppState>>,
    Query(UrlQuery { url }): Query<UrlQuery>,
) -> Response {
    state.router.remove_worker(&url);
    (
        StatusCode::OK,
        format!("Successfully removed worker: {url}"),
    )
        .into_response()
}

async fn flush_cache(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    state.router.flush_cache().await
}

async fn get_loads(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    state.router.get_worker_loads().await
}

// ---------- Worker management endpoints (RESTful) ----------

/// POST /workers - Add a new worker with full configuration
async fn create_worker(
    State(state): State<Arc<AppState>>,
    Json(config): Json<WorkerConfigRequest>,
) -> Response {
    // Check if we have a RouterManager (enable_igw=true)
    if let Some(router_manager) = &state.router_manager {
        // Call RouterManager's add_worker method directly with the full config
        match router_manager.add_worker(config).await {
            Ok(response) => (StatusCode::OK, Json(response)).into_response(),
            Err(error) => (StatusCode::BAD_REQUEST, Json(error)).into_response(),
        }
    } else {
        // In single router mode, use the router's add_worker with basic config
        match state.router.add_worker(&config.url).await {
            Ok(message) => {
                let response = WorkerApiResponse {
                    success: true,
                    message,
                    worker: None,
                };
                (StatusCode::OK, Json(response)).into_response()
            }
            Err(error) => {
                let error_response = WorkerErrorResponse {
                    error,
                    code: "ADD_WORKER_FAILED".to_string(),
                };
                (StatusCode::BAD_REQUEST, Json(error_response)).into_response()
            }
        }
    }
}

/// GET /workers - List all workers with details
async fn list_workers_rest(State(state): State<Arc<AppState>>) -> Response {
    if let Some(router_manager) = &state.router_manager {
        let response = router_manager.list_workers();
        Json(response).into_response()
    } else {
        // In single router mode, get detailed worker info from registry
        let workers = state.context.worker_registry.get_all();
        let response = serde_json::json!({
            "workers": workers.iter().map(|worker| {
                let mut worker_info = serde_json::json!({
                    "url": worker.url(),
                    "model_id": worker.model_id(),
                    "worker_type": match worker.worker_type() {
                        WorkerType::Regular => "regular",
                        WorkerType::Prefill { .. } => "prefill",
                        WorkerType::Decode => "decode",
                    },
                    "is_healthy": worker.is_healthy(),
                    "load": worker.load(),
                    "connection_mode": format!("{:?}", worker.connection_mode()),
                    "priority": worker.priority(),
                    "cost": worker.cost(),
                });

                // Add bootstrap_port for Prefill workers
                if let WorkerType::Prefill { bootstrap_port } = worker.worker_type() {
                    worker_info["bootstrap_port"] = serde_json::json!(bootstrap_port);
                }

                worker_info
            }).collect::<Vec<_>>(),
            "total": workers.len(),
            "stats": {
                "prefill_count": state.context.worker_registry.get_prefill_workers().len(),
                "decode_count": state.context.worker_registry.get_decode_workers().len(),
                "regular_count": state.context.worker_registry.get_by_type(&WorkerType::Regular).len(),
            }
        });
        Json(response).into_response()
    }
}

/// GET /workers/{url} - Get specific worker info
async fn get_worker(State(state): State<Arc<AppState>>, Path(url): Path<String>) -> Response {
    if let Some(router_manager) = &state.router_manager {
        if let Some(worker) = router_manager.get_worker(&url) {
            Json(worker).into_response()
        } else {
            let error = WorkerErrorResponse {
                error: format!("Worker {url} not found"),
                code: "WORKER_NOT_FOUND".to_string(),
            };
            (StatusCode::NOT_FOUND, Json(error)).into_response()
        }
    } else {
        let workers = state.router.get_worker_urls();
        if workers.contains(&url) {
            Json(json!({
                "url": url,
                "model_id": "unknown",
                "is_healthy": true
            }))
            .into_response()
        } else {
            let error = WorkerErrorResponse {
                error: format!("Worker {url} not found"),
                code: "WORKER_NOT_FOUND".to_string(),
            };
            (StatusCode::NOT_FOUND, Json(error)).into_response()
        }
    }
}

/// DELETE /workers/{url} - Remove a worker
async fn delete_worker(State(state): State<Arc<AppState>>, Path(url): Path<String>) -> Response {
    if let Some(router_manager) = &state.router_manager {
        match router_manager.remove_worker_from_registry(&url) {
            Ok(response) => (StatusCode::OK, Json(response)).into_response(),
            Err(error) => (StatusCode::BAD_REQUEST, Json(error)).into_response(),
        }
    } else {
        // In single router mode, use router's remove_worker
        state.router.remove_worker(&url);
        let response = WorkerApiResponse {
            success: true,
            message: format!("Worker {url} removed successfully"),
            worker: None,
        };
        (StatusCode::OK, Json(response)).into_response()
    }
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub router_config: RouterConfig,
    pub max_payload_size: usize,
    pub log_dir: Option<String>,
    pub log_level: Option<String>,
    pub service_discovery_config: Option<ServiceDiscoveryConfig>,
    pub prometheus_config: Option<PrometheusConfig>,
    pub request_timeout_secs: u64,
    pub request_id_headers: Option<Vec<String>>,
}

/// Build the Axum application with all routes and middleware
pub fn build_app(
    app_state: Arc<AppState>,
    max_payload_size: usize,
    request_id_headers: Vec<String>,
    cors_allowed_origins: Vec<String>,
) -> Router {
    // Create routes
    let protected_routes = Router::new()
        .route("/generate", post(generate))
        .route("/v1/chat/completions", post(v1_chat_completions))
        .route("/v1/completions", post(v1_completions))
        .route("/rerank", post(rerank))
        .route("/v1/rerank", post(v1_rerank))
        .route("/v1/responses", post(v1_responses))
        .route("/v1/embeddings", post(v1_embeddings))
        .route("/v1/responses/{response_id}", get(v1_responses_get))
        .route(
            "/v1/responses/{response_id}/cancel",
            post(v1_responses_cancel),
        )
        .route("/v1/responses/{response_id}", delete(v1_responses_delete))
        .route(
            "/v1/responses/{response_id}/input",
            get(v1_responses_list_input_items),
        )
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            middleware::concurrency_limit_middleware,
        ));

    let public_routes = Router::new()
        .route("/liveness", get(liveness))
        .route("/readiness", get(readiness))
        .route("/health", get(health))
        .route("/health_generate", get(health_generate))
        .route("/v1/models", get(v1_models))
        .route("/get_model_info", get(get_model_info))
        .route("/get_server_info", get(get_server_info));

    let admin_routes = Router::new()
        .route("/add_worker", post(add_worker))
        .route("/remove_worker", post(remove_worker))
        .route("/list_workers", get(list_workers))
        .route("/flush_cache", post(flush_cache))
        .route("/get_loads", get(get_loads));

    // Worker management routes
    let worker_routes = Router::new()
        .route("/workers", post(create_worker))
        .route("/workers", get(list_workers_rest))
        .route("/workers/{url}", get(get_worker))
        .route("/workers/{url}", delete(delete_worker));

    // Build app with all routes and middleware
    Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .merge(admin_routes)
        .merge(worker_routes)
        // Request body size limiting
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            max_payload_size,
        ))
        .layer(middleware::create_logging_layer())
        .layer(middleware::RequestIdLayer::new(request_id_headers))
        .layer(create_cors_layer(cors_allowed_origins))
        .fallback(sink_handler)
        .with_state(app_state)
}

pub async fn startup(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Only initialize logging if not already done (for Python bindings support)
    static LOGGING_INITIALIZED: AtomicBool = AtomicBool::new(false);

    let _log_guard = if !LOGGING_INITIALIZED.swap(true, Ordering::SeqCst) {
        Some(logging::init_logging(LoggingConfig {
            level: config
                .log_level
                .as_deref()
                .and_then(|s| match s.to_uppercase().parse::<Level>() {
                    Ok(l) => Some(l),
                    Err(_) => {
                        warn!("Invalid log level string: '{s}'. Defaulting to INFO.");
                        None
                    }
                })
                .unwrap_or(Level::INFO),
            json_format: false,
            log_dir: config.log_dir.clone(),
            colorize: true,
            log_file_name: "sgl-router".to_string(),
            log_targets: None,
        }))
    } else {
        None
    };

    // Initialize prometheus metrics exporter
    if let Some(prometheus_config) = config.prometheus_config {
        metrics::start_prometheus(prometheus_config);
    }

    info!(
        "Starting router on {}:{} | mode: {:?} | policy: {:?} | max_payload: {}MB",
        config.host,
        config.port,
        config.router_config.mode,
        config.router_config.policy,
        config.max_payload_size / (1024 * 1024)
    );

    let client = Client::builder()
        .pool_idle_timeout(Some(Duration::from_secs(50)))
        .pool_max_idle_per_host(500)
        .timeout(Duration::from_secs(config.request_timeout_secs))
        .connect_timeout(Duration::from_secs(10))
        .tcp_nodelay(true)
        .tcp_keepalive(Some(Duration::from_secs(30)))
        .build()
        .expect("Failed to create HTTP client");

    // Create the application context with all dependencies
    let app_context = AppContext::new(
        config.router_config.clone(),
        client.clone(),
        config.router_config.max_concurrent_requests,
        config.router_config.rate_limit_tokens_per_second,
    )?;

    let app_context = Arc::new(app_context);

    info!(
        "Initializing workers for routing mode: {:?}",
        config.router_config.mode
    );
    WorkerInitializer::initialize_workers(
        &config.router_config,
        &app_context.worker_registry,
        Some(&app_context.policy_registry),
    )
    .await
    .map_err(|e| format!("Failed to initialize workers: {}", e))?;

    let worker_stats = app_context.worker_registry.stats();
    info!(
        "Workers initialized: {} total, {} healthy",
        worker_stats.total_workers, worker_stats.healthy_workers
    );

    // Create the appropriate router based on enable_igw flag
    let (router, router_manager): (Arc<dyn RouterTrait>, Option<Arc<RouterManager>>) =
        if config.router_config.enable_igw {
            info!("Multi-router mode enabled (enable_igw=true)");

            // Create RouterManager with shared registries from AppContext
            let router_manager = Arc::new(RouterManager::new(
                config.router_config.clone(),
                client.clone(),
                app_context.worker_registry.clone(),
                app_context.policy_registry.clone(),
            ));

            // 1. HTTP Regular Router
            match RouterFactory::create_regular_router(&app_context).await {
                Ok(http_regular) => {
                    info!("Created HTTP Regular router");
                    router_manager.register_router(
                        RouterId::new("http-regular".to_string()),
                        Arc::from(http_regular),
                    );
                }
                Err(e) => {
                    warn!("Failed to create HTTP Regular router: {e}");
                }
            }

            // 2. HTTP PD Router
            match RouterFactory::create_pd_router(
                None,
                None,
                &config.router_config.policy,
                &app_context,
            )
            .await
            {
                Ok(http_pd) => {
                    info!("Created HTTP PD router");
                    router_manager
                        .register_router(RouterId::new("http-pd".to_string()), Arc::from(http_pd));
                }
                Err(e) => {
                    warn!("Failed to create HTTP PD router: {e}");
                }
            }

            // TODO: Add gRPC routers once we have dynamic tokenizer loading

            info!(
                "RouterManager initialized with {} routers",
                router_manager.router_count()
            );
            (
                router_manager.clone() as Arc<dyn RouterTrait>,
                Some(router_manager),
            )
        } else {
            info!("Single router mode (enable_igw=false)");
            // Create single router with the context
            (
                Arc::from(RouterFactory::create_router(&app_context).await?),
                None,
            )
        };

    // Start health checker for all workers in the registry
    let _health_checker = app_context
        .worker_registry
        .start_health_checker(config.router_config.health_check.check_interval_secs);
    info!(
        "Started health checker for workers with {}s interval",
        config.router_config.health_check.check_interval_secs
    );

    // Set up concurrency limiter with queue if configured
    let (limiter, processor) = middleware::ConcurrencyLimiter::new(
        app_context.rate_limiter.clone(),
        config.router_config.queue_size,
        Duration::from_secs(config.router_config.queue_timeout_secs),
    );

    // Start queue processor if enabled
    if let Some(processor) = processor {
        spawn(processor.run());
        info!(
            "Started request queue with size: {}, timeout: {}s",
            config.router_config.queue_size, config.router_config.queue_timeout_secs
        );
    }

    // Create app state with router and context
    let app_state = Arc::new(AppState {
        router,
        context: app_context.clone(),
        concurrency_queue_tx: limiter.queue_tx.clone(),
        router_manager,
    });
    let router_arc = Arc::clone(&app_state.router);

    // Start the service discovery if enabled
    if let Some(service_discovery_config) = config.service_discovery_config {
        if service_discovery_config.enabled {
            match start_service_discovery(service_discovery_config, router_arc).await {
                Ok(handle) => {
                    info!("Service discovery started");
                    // Spawn a task to handle the service discovery thread
                    spawn(async move {
                        if let Err(e) = handle.await {
                            error!("Service discovery task failed: {:?}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to start service discovery: {e}");
                    warn!("Continuing without service discovery");
                }
            }
        }
    }

    info!(
        "Router ready | workers: {:?}",
        app_state.router.get_worker_urls()
    );

    let request_id_headers = config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

    // Build the application
    let app = build_app(
        app_state,
        config.max_payload_size,
        request_id_headers,
        config.router_config.cors_allowed_origins.clone(),
    );

    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr).await?;
    info!("Starting server on {}", addr);
    serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;

    Ok(())
}

// Graceful shutdown handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, starting graceful shutdown");
        },
        _ = terminate => {
            info!("Received terminate signal, starting graceful shutdown");
        },
    }
}

// CORS Layer Creation
fn create_cors_layer(allowed_origins: Vec<String>) -> tower_http::cors::CorsLayer {
    use tower_http::cors::Any;

    let cors = if allowed_origins.is_empty() {
        // Allow all origins if none specified
        tower_http::cors::CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
            .expose_headers(Any)
    } else {
        // Restrict to specific origins
        let origins: Vec<http::HeaderValue> = allowed_origins
            .into_iter()
            .filter_map(|origin| origin.parse().ok())
            .collect();

        tower_http::cors::CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([http::Method::GET, http::Method::POST, http::Method::OPTIONS])
            .allow_headers([http::header::CONTENT_TYPE, http::header::AUTHORIZATION])
            .expose_headers([http::header::HeaderName::from_static("x-request-id")])
    };

    cors.max_age(Duration::from_secs(3600))
}
