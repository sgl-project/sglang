<<<<<<< HEAD
use crate::logging::{self, LoggingConfig};
use crate::openai_api_types::{ChatCompletionRequest, CompletionRequest, GenerateRequest};
use crate::prometheus::{self, PrometheusConfig};
use crate::request_adapter::ToPdRequest;
use crate::router::PolicyConfig;
use crate::router::Router;
use crate::service_discovery::{start_service_discovery, ServiceDiscoveryConfig};
use actix_web::{
    error, get, post, web, App, Error, HttpRequest, HttpResponse, HttpServer, Responder,
};
use futures_util::StreamExt;
=======
use crate::config::RouterConfig;
use crate::logging::{self, LoggingConfig};
use crate::metrics::{self, PrometheusConfig};
use crate::middleware::TokenBucket;
use crate::protocols::spec::{ChatCompletionRequest, CompletionRequest, GenerateRequest};
use crate::routers::{RouterFactory, RouterTrait};
use crate::service_discovery::{start_service_discovery, ServiceDiscoveryConfig};
use axum::{
    extract::{Query, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
>>>>>>> origin/main
use reqwest::Client;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
<<<<<<< HEAD
use tokio::spawn;
use tracing::{error, info, warn, Level};

#[derive(Debug)]
pub struct AppState {
    router: Arc<Router>,
    client: Client,
    is_pd_mode: bool, // Add flag to track PD mode
}

impl AppState {
    pub fn new(
        worker_urls: Vec<String>,
        client: Client,
        policy_config: PolicyConfig,
    ) -> Result<Self, String> {
        // Check if this is PD mode from policy config
        let is_pd_mode = matches!(policy_config, PolicyConfig::PrefillDecodeConfig { .. });

        // Create router based on policy
        let router = Arc::new(Router::new(worker_urls, policy_config)?);
        Ok(Self {
            router,
            client,
            is_pd_mode,
        })
    }
}

async fn sink_handler(_req: HttpRequest, mut payload: web::Payload) -> Result<HttpResponse, Error> {
    // Drain the payload
    while let Some(chunk) = payload.next().await {
        if let Err(err) = chunk {
            println!("Error while draining payload: {:?}", err);
            break;
        }
    }
    Ok(HttpResponse::NotFound().finish())
}

// Custom error handler for JSON payload errors.
fn json_error_handler(err: error::JsonPayloadError, _req: &HttpRequest) -> Error {
    error!("JSON payload error: {:?}", err);
    match &err {
        error::JsonPayloadError::OverflowKnownLength { length, limit } => {
            error!(
                "Payload too large: {} bytes exceeds limit of {} bytes",
                length, limit
            );
            error::ErrorPayloadTooLarge(format!(
                "Payload too large: {} bytes exceeds limit of {} bytes",
                length, limit
            ))
        }
        error::JsonPayloadError::Overflow { limit } => {
            error!("Payload overflow: exceeds limit of {} bytes", limit);
            error::ErrorPayloadTooLarge(format!("Payload exceeds limit of {} bytes", limit))
        }
        _ => error::ErrorBadRequest(format!("Invalid JSON payload: {}", err)),
    }
}

#[get("/health")]
async fn health(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/health", &req)
        .await
}

#[get("/health_generate")]
async fn health_generate(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    // Check if we're in PD mode
    if data.is_pd_mode {
        // For PD mode, check health on all servers
        data.router
            .route_pd_health_generate(&data.client, &req)
            .await
    } else {
        // Regular mode
        data.router
            .route_to_first(&data.client, "/health_generate", &req)
            .await
    }
}

#[get("/get_server_info")]
async fn get_server_info(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    if data.is_pd_mode {
        // For PD mode, aggregate info from both prefill and decode servers
        data.router.get_pd_server_info(&data.client, &req).await
    } else {
        // Regular mode - return first server's info
        data.router
            .route_to_first(&data.client, "/get_server_info", &req)
            .await
    }
}

#[get("/v1/models")]
async fn v1_models(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    if data.is_pd_mode {
        // For PD mode, return models from the first prefill server
        data.router.get_pd_models(&data.client, &req).await
    } else {
        // Regular mode
        data.router
            .route_to_first(&data.client, "/v1/models", &req)
            .await
    }
}

#[get("/get_model_info")]
async fn get_model_info(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    if data.is_pd_mode {
        // For PD mode, get model info from the first prefill server
        data.router.get_pd_model_info(&data.client, &req).await
    } else {
        data.router
            .route_to_first(&data.client, "/get_model_info", &req)
            .await
    }
}

#[post("/generate")]
async fn generate(
    req: HttpRequest,
    body: web::Json<GenerateRequest>,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let client = &state.client;
    let router = &state.router;

    // Use typed request directly for both PD and regular routing
    if state.is_pd_mode {
        // For PD mode, convert to PD request with bootstrap
        let pd_request = body.into_inner().to_pd_request();

        Ok(router
            .route_pd_generate_typed(&client, &req, pd_request, "/generate")
            .await)
    } else {
        // For regular mode, use typed request directly
        let request = body.into_inner();
        Ok(router
            .route_typed_request(&client, &req, &request, "/generate")
            .await)
    }
}

#[post("/v1/chat/completions")]
async fn v1_chat_completions(
    req: HttpRequest,
    body: web::Json<ChatCompletionRequest>,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let client = &state.client;
    let router = &state.router;

    // Use typed request directly for both PD and regular routing
    if state.is_pd_mode {
        // For PD mode, convert to PD request with bootstrap
        let pd_request = body.into_inner().to_pd_request();

        Ok(router
            .route_pd_chat_typed(&client, &req, pd_request, "/v1/chat/completions")
            .await)
    } else {
        // For regular mode, use typed request directly
        let request = body.into_inner();
        Ok(router
            .route_typed_request(&client, &req, &request, "/v1/chat/completions")
            .await)
    }
}

#[post("/v1/completions")]
async fn v1_completions(
    req: HttpRequest,
    body: web::Json<CompletionRequest>,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let client = &state.client;
    let router = &state.router;

    // Use typed request directly for both PD and regular routing
    if state.is_pd_mode {
        // For PD mode, convert to PD request with bootstrap
        let pd_request = body.into_inner().to_pd_request();

        Ok(router
            .route_pd_generate_typed(&client, &req, pd_request, "/v1/completions")
            .await)
    } else {
        // For regular mode, use typed request directly
        let request = body.into_inner();
        Ok(router
            .route_typed_request(&client, &req, &request, "/v1/completions")
            .await)
    }
}

#[post("/add_worker")]
async fn add_worker(
    query: web::Query<HashMap<String, String>>,
    data: web::Data<AppState>,
) -> impl Responder {
    let worker_url = match query.get("url") {
        Some(url) => url.to_string(),
        None => {
            return HttpResponse::BadRequest()
                .body("Worker URL required. Provide 'url' query parameter")
        }
    };

    match data.router.add_worker(&worker_url).await {
        Ok(message) => HttpResponse::Ok().body(message),
        Err(error) => HttpResponse::BadRequest().body(error),
    }
}

#[get("/list_workers")]
async fn list_workers(data: web::Data<AppState>) -> impl Responder {
    let workers = data.router.get_worker_urls();
    let worker_list = workers.read().unwrap().clone();
    HttpResponse::Ok().json(serde_json::json!({ "urls": worker_list }))
}

#[post("/remove_worker")]
async fn remove_worker(
    query: web::Query<HashMap<String, String>>,
    data: web::Data<AppState>,
) -> impl Responder {
    let worker_url = match query.get("url") {
        Some(url) => url.to_string(),
        None => return HttpResponse::BadRequest().finish(),
    };
    data.router.remove_worker(&worker_url);
    HttpResponse::Ok().body(format!("Successfully removed worker: {}", worker_url))
}

#[post("/flush_cache")]
async fn flush_cache(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    if data.is_pd_mode {
        // For PD mode, flush cache on both prefill and decode servers
        data.router.route_pd_flush_cache(&data.client).await
    } else {
        // Route to all workers for cache flushing
        data.router
            .route_to_all(&data.client, "/flush_cache", &req)
            .await
    }
}

#[get("/get_loads")]
async fn get_loads(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    // Get loads from all workers
    data.router.get_all_loads(&data.client, &req).await
=======
use tokio::net::TcpListener;
use tokio::signal;
use tokio::spawn;
use tracing::{error, info, warn, Level};

#[derive(Clone)]
pub struct AppContext {
    pub client: Client,
    pub router_config: RouterConfig,
    pub rate_limiter: Arc<TokenBucket>,
    // Future dependencies can be added here
}

impl AppContext {
    pub fn new(
        router_config: RouterConfig,
        client: Client,
        max_concurrent_requests: usize,
        rate_limit_tokens_per_second: Option<usize>,
    ) -> Self {
        let rate_limit_tokens = rate_limit_tokens_per_second.unwrap_or(max_concurrent_requests);
        let rate_limiter = Arc::new(TokenBucket::new(max_concurrent_requests, rate_limit_tokens));
        Self {
            client,
            router_config,
            rate_limiter,
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub router: Arc<dyn RouterTrait>,
    pub context: Arc<AppContext>,
    pub concurrency_queue_tx: Option<tokio::sync::mpsc::Sender<crate::middleware::QueuedRequest>>,
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
    state.router.route_generate(Some(&headers), &body).await
}

async fn v1_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<ChatCompletionRequest>,
) -> Response {
    state.router.route_chat(Some(&headers), &body).await
}

async fn v1_completions(
    State(state): State<Arc<AppState>>,
    headers: http::HeaderMap,
    Json(body): Json<CompletionRequest>,
) -> Response {
    state.router.route_completion(Some(&headers), &body).await
}

// Worker management endpoints
async fn add_worker(
    State(state): State<Arc<AppState>>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let worker_url = match params.get("url") {
        Some(url) => url.to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                "Worker URL required. Provide 'url' query parameter",
            )
                .into_response();
        }
    };

    match state.router.add_worker(&worker_url).await {
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
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let worker_url = match params.get("url") {
        Some(url) => url.to_string(),
        None => return StatusCode::BAD_REQUEST.into_response(),
    };

    state.router.remove_worker(&worker_url);
    (
        StatusCode::OK,
        format!("Successfully removed worker: {}", worker_url),
    )
        .into_response()
}

async fn flush_cache(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    state.router.flush_cache().await
}

async fn get_loads(State(state): State<Arc<AppState>>, _req: Request) -> Response {
    state.router.get_worker_loads().await
>>>>>>> origin/main
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
<<<<<<< HEAD
    pub worker_urls: Vec<String>,
    pub policy_config: PolicyConfig,
=======
    pub router_config: RouterConfig,
>>>>>>> origin/main
    pub max_payload_size: usize,
    pub log_dir: Option<String>,
    pub log_level: Option<String>,
    pub service_discovery_config: Option<ServiceDiscoveryConfig>,
    pub prometheus_config: Option<PrometheusConfig>,
    pub request_timeout_secs: u64,
<<<<<<< HEAD
}

pub async fn startup(config: ServerConfig) -> std::io::Result<()> {
=======
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
        .route_layer(axum::middleware::from_fn_with_state(
            app_state.clone(),
            crate::middleware::concurrency_limit_middleware,
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

    // Build app with all routes and middleware
    Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .merge(admin_routes)
        // Request body size limiting
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            max_payload_size,
        ))
        // Request ID layer - must be added AFTER logging layer in the code
        // so it executes BEFORE logging layer at runtime (layers execute bottom-up)
        .layer(crate::middleware::RequestIdLayer::new(request_id_headers))
        // Custom logging layer that can now see request IDs from extensions
        .layer(crate::middleware::create_logging_layer())
        // CORS (should be outermost)
        .layer(create_cors_layer(cors_allowed_origins))
        // Fallback
        .fallback(sink_handler)
        // State - apply last to get Router<Arc<AppState>>
        .with_state(app_state)
}

pub async fn startup(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
>>>>>>> origin/main
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
                        warn!("Invalid log level string: '{}'. Defaulting to INFO.", s);
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
<<<<<<< HEAD
        info!(
            "🚧 Initializing Prometheus metrics on {}:{}",
            prometheus_config.host, prometheus_config.port
        );
        prometheus::start_prometheus(prometheus_config);
    } else {
        info!("🚧 Prometheus metrics disabled");
    }

    info!("🚧 Initializing router on {}:{}", config.host, config.port);
    info!("🚧 Initializing workers on {:?}", config.worker_urls);
    info!("🚧 Policy Config: {:?}", config.policy_config);
    info!(
        "🚧 Max payload size: {} MB",
        config.max_payload_size / (1024 * 1024)
    );

    // Log service discovery status
    if let Some(service_discovery_config) = &config.service_discovery_config {
        info!("🚧 Service discovery enabled");
        info!("🚧 Selector: {:?}", service_discovery_config.selector);
    } else {
        info!("🚧 Service discovery disabled");
    }

    let client = Client::builder()
        .pool_idle_timeout(Some(Duration::from_secs(50)))
        .timeout(Duration::from_secs(config.request_timeout_secs)) // Use configurable timeout
        .build()
        .expect("Failed to create HTTP client");

    let app_state_init = AppState::new(
        config.worker_urls.clone(),
        client.clone(),
        config.policy_config.clone(),
    )
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let router_arc = Arc::clone(&app_state_init.router);
    let app_state = web::Data::new(app_state_init);
=======
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
        .pool_max_idle_per_host(500) // Increase to 500 connections per host
        .timeout(Duration::from_secs(config.request_timeout_secs))
        .connect_timeout(Duration::from_secs(10)) // Separate connection timeout
        .tcp_nodelay(true)
        .tcp_keepalive(Some(Duration::from_secs(30))) // Keep connections alive
        .build()
        .expect("Failed to create HTTP client");

    // Create the application context with all dependencies
    let app_context = Arc::new(AppContext::new(
        config.router_config.clone(),
        client.clone(),
        config.router_config.max_concurrent_requests,
        config.router_config.rate_limit_tokens_per_second,
    ));

    // Create router with the context
    let router = RouterFactory::create_router(&app_context).await?;

    // Set up concurrency limiter with queue if configured
    let (limiter, processor) = crate::middleware::ConcurrencyLimiter::new(
        app_context.rate_limiter.clone(),
        config.router_config.queue_size,
        Duration::from_secs(config.router_config.queue_timeout_secs),
    );

    // Start queue processor if enabled
    if let Some(processor) = processor {
        tokio::spawn(processor.run());
        info!(
            "Started request queue with size: {}, timeout: {}s",
            config.router_config.queue_size, config.router_config.queue_timeout_secs
        );
    }

    // Create app state with router and context
    let app_state = Arc::new(AppState {
        router: Arc::from(router),
        context: app_context.clone(),
        concurrency_queue_tx: limiter.queue_tx.clone(),
    });
    let router_arc = Arc::clone(&app_state.router);
>>>>>>> origin/main

    // Start the service discovery if enabled
    if let Some(service_discovery_config) = config.service_discovery_config {
        if service_discovery_config.enabled {
<<<<<<< HEAD
            info!("🚧 Initializing Kubernetes service discovery");
            // Pass the Arc<Router> directly
            match start_service_discovery(service_discovery_config, router_arc).await {
                Ok(handle) => {
                    info!("✅ Service discovery started successfully");
=======
            match start_service_discovery(service_discovery_config, router_arc).await {
                Ok(handle) => {
                    info!("Service discovery started");
>>>>>>> origin/main
                    // Spawn a task to handle the service discovery thread
                    spawn(async move {
                        if let Err(e) = handle.await {
                            error!("Service discovery task failed: {:?}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to start service discovery: {}", e);
                    warn!("Continuing without service discovery");
                }
            }
        }
    }

<<<<<<< HEAD
    info!("✅ Serving router on {}:{}", config.host, config.port);
    info!(
        "✅ Serving workers on {:?}",
        app_state.router.get_worker_urls().read().unwrap()
    );

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .app_data(
                web::JsonConfig::default()
                    .limit(config.max_payload_size)
                    .error_handler(json_error_handler),
            )
            .app_data(web::PayloadConfig::default().limit(config.max_payload_size))
            .service(generate)
            .service(v1_chat_completions)
            .service(v1_completions)
            .service(v1_models)
            .service(get_model_info)
            .service(health)
            .service(health_generate)
            .service(get_server_info)
            .service(add_worker)
            .service(remove_worker)
            .service(list_workers)
            .service(flush_cache)
            .service(get_loads)
            .default_service(web::route().to(sink_handler))
    })
    .bind_auto_h2c((config.host, config.port))?
    .run()
    .await
=======
    info!(
        "Router ready | workers: {:?}",
        app_state.router.get_worker_urls()
    );

    // Configure request ID headers
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

    // Create TCP listener - use the configured host
    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr).await?;

    // Start server with graceful shutdown
    info!("Starting server on {}", addr);

    // Serve the application with graceful shutdown
    axum::serve(listener, app)
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
>>>>>>> origin/main
}
