use crate::logging::{self, LoggingConfig};
use crate::pd_types::{ChatReqInput, GenerateReqInput};
use crate::prometheus::{self, PrometheusConfig};
use crate::router::PolicyConfig;
use crate::router::Router;
use crate::service_discovery::{start_service_discovery, ServiceDiscoveryConfig};
use actix_web::{
    error, get, post, web, App, Error, HttpRequest, HttpResponse, HttpServer, Responder,
};
use bytes::Bytes;
use futures_util::StreamExt;
use reqwest::Client;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::spawn;
use tracing::{error, info, warn, Level};

#[derive(Debug)]
pub struct AppState {
    router: Arc<Router>,
    client: Client,
}

impl AppState {
    pub fn new(
        worker_urls: Vec<String>,
        client: Client,
        policy_config: PolicyConfig,
    ) -> Result<Self, String> {
        // Create router based on policy
        let router = Arc::new(Router::new(worker_urls, policy_config)?);
        Ok(Self { router, client })
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
    if data.router.is_prefill_decode() {
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
    if data.router.is_prefill_decode() {
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
    if data.router.is_prefill_decode() {
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
    if data.router.is_prefill_decode() {
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
    typed_req: web::Json<GenerateReqInput>,
    data: web::Data<AppState>,
) -> impl Responder {
    // For PD mode, use typed request handling with proper JSON extraction
    if data.router.is_prefill_decode() {
        let typed_req = typed_req.into_inner();

        // Debug logging to understand the request structure
        info!(
            "PD generate request - batch_size: {:?}",
            typed_req.get_batch_size()
        );
        if let Some(input_ids) = &typed_req.input_ids {
            match input_ids {
                crate::pd_types::SingleOrBatch::Single(ids) => {
                    info!("Single request with {} tokens", ids.len());
                }
                crate::pd_types::SingleOrBatch::Batch(batch) => {
                    info!("Batch request with {} sequences", batch.len());
                }
            }
        }

        data.router
            .route_pd_generate_typed(&data.client, &req, typed_req, "/generate")
            .await
    } else {
        // Regular mode - convert back to bytes for compatibility
        let body = match serde_json::to_vec(&typed_req.into_inner()) {
            Ok(bytes) => Bytes::from(bytes),
            Err(e) => {
                return HttpResponse::InternalServerError()
                    .body(format!("Failed to serialize request: {}", e))
            }
        };

        data.router
            .route_generate_request(&data.client, &req, &body, "/generate")
            .await
    }
}

#[post("/v1/chat/completions")]
async fn v1_chat_completions(
    req: HttpRequest,
    typed_req: web::Json<ChatReqInput>,
    data: web::Data<AppState>,
) -> impl Responder {
    // For PD mode, use typed request handling with proper JSON extraction
    if data.router.is_prefill_decode() {
        let typed_req = typed_req.into_inner();

        data.router
            .route_pd_chat_typed(&data.client, &req, typed_req, "/v1/chat/completions")
            .await
    } else {
        // Regular mode - convert back to bytes for compatibility
        let body = match serde_json::to_vec(&typed_req.into_inner()) {
            Ok(bytes) => Bytes::from(bytes),
            Err(e) => {
                return HttpResponse::InternalServerError()
                    .body(format!("Failed to serialize request: {}", e))
            }
        };

        data.router
            .route_generate_request(&data.client, &req, &body, "/v1/chat/completions")
            .await
    }
}

#[post("/v1/completions")]
async fn v1_completions(
    req: HttpRequest,
    typed_req: web::Json<GenerateReqInput>,
    data: web::Data<AppState>,
) -> impl Responder {
    // For PD mode, use typed request handling with proper JSON extraction
    if data.router.is_prefill_decode() {
        let typed_req = typed_req.into_inner();

        data.router
            .route_pd_generate_typed(&data.client, &req, typed_req, "/v1/completions")
            .await
    } else {
        // Regular mode - convert back to bytes for compatibility
        let body = match serde_json::to_vec(&typed_req.into_inner()) {
            Ok(bytes) => Bytes::from(bytes),
            Err(e) => {
                return HttpResponse::InternalServerError()
                    .body(format!("Failed to serialize request: {}", e))
            }
        };

        data.router
            .route_generate_request(&data.client, &req, &body, "/v1/completions")
            .await
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
    if data.router.is_prefill_decode() {
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
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub worker_urls: Vec<String>,
    pub policy_config: PolicyConfig,
    pub verbose: bool,
    pub max_payload_size: usize,
    pub log_dir: Option<String>,
    pub service_discovery_config: Option<ServiceDiscoveryConfig>,
    pub prometheus_config: Option<PrometheusConfig>,
    pub request_timeout_secs: u64,
}

pub async fn startup(config: ServerConfig) -> std::io::Result<()> {
    // Only initialize logging if not already done (for Python bindings support)
    static LOGGING_INITIALIZED: AtomicBool = AtomicBool::new(false);

    let _log_guard = if !LOGGING_INITIALIZED.swap(true, Ordering::SeqCst) {
        Some(logging::init_logging(LoggingConfig {
            level: if config.verbose {
                Level::DEBUG
            } else {
                Level::INFO
            },
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

    // Start the service discovery if enabled
    if let Some(service_discovery_config) = config.service_discovery_config {
        if service_discovery_config.enabled {
            info!("🚧 Initializing Kubernetes service discovery");
            // Pass the Arc<Router> directly
            match start_service_discovery(service_discovery_config, router_arc).await {
                Ok(handle) => {
                    info!("✅ Service discovery started successfully");
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
            // Note: /register endpoint for dynamic PD server registration not implemented yet
            // This would require modifying PDRouter to support dynamic worker addition
            // Default handler for unmatched routes.
            .default_service(web::route().to(sink_handler))
    })
    .bind_auto_h2c((config.host, config.port))?
    .run()
    .await
}
