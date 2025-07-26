use crate::config::RouterConfig;
use crate::logging::{self, LoggingConfig};
use crate::metrics::{self, PrometheusConfig};
use crate::openai_api_types::{ChatCompletionRequest, CompletionRequest, GenerateRequest};
use crate::routers::{RouterFactory, RouterTrait};
use crate::service_discovery::{start_service_discovery, ServiceDiscoveryConfig};
use actix_web::{
    error, get, post, web, App, Error, HttpRequest, HttpResponse, HttpServer, Responder,
};
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
    router: Arc<dyn RouterTrait>,
    client: Client,
}

impl AppState {
    pub fn new(router_config: RouterConfig, client: Client) -> Result<Self, String> {
        // Use RouterFactory to create the appropriate router type
        let router = RouterFactory::create_router(&router_config)?;

        // Convert Box<dyn RouterTrait> to Arc<dyn RouterTrait>
        let router = Arc::from(router);

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

#[get("/liveness")]
async fn liveness(_req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router.liveness()
}

#[get("/readiness")]
async fn readiness(_req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router.readiness()
}

#[get("/health")]
async fn health(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router.health(&data.client, &req).await
}

#[get("/health_generate")]
async fn health_generate(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router.health_generate(&data.client, &req).await
}

#[get("/get_server_info")]
async fn get_server_info(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router.get_server_info(&data.client, &req).await
}

#[get("/v1/models")]
async fn v1_models(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router.get_models(&data.client, &req).await
}

#[get("/get_model_info")]
async fn get_model_info(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router.get_model_info(&data.client, &req).await
}

#[post("/generate")]
async fn generate(
    req: HttpRequest,
    body: web::Json<GenerateRequest>,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let json_body = serde_json::to_value(body.into_inner())
        .map_err(|e| error::ErrorBadRequest(format!("Invalid JSON: {}", e)))?;
    Ok(state
        .router
        .route_generate(&state.client, &req, json_body)
        .await)
}

#[post("/v1/chat/completions")]
async fn v1_chat_completions(
    req: HttpRequest,
    body: web::Json<ChatCompletionRequest>,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let json_body = serde_json::to_value(body.into_inner())
        .map_err(|e| error::ErrorBadRequest(format!("Invalid JSON: {}", e)))?;
    Ok(state
        .router
        .route_chat(&state.client, &req, json_body)
        .await)
}

#[post("/v1/completions")]
async fn v1_completions(
    req: HttpRequest,
    body: web::Json<CompletionRequest>,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let json_body = serde_json::to_value(body.into_inner())
        .map_err(|e| error::ErrorBadRequest(format!("Invalid JSON: {}", e)))?;
    Ok(state
        .router
        .route_completion(&state.client, &req, json_body)
        .await)
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
    let worker_list = data.router.get_worker_urls();
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
async fn flush_cache(_req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router.flush_cache(&data.client).await
}

#[get("/get_loads")]
async fn get_loads(_req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router.get_worker_loads(&data.client).await
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
}

pub async fn startup(config: ServerConfig) -> std::io::Result<()> {
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
        info!(
            "🚧 Initializing Prometheus metrics on {}:{}",
            prometheus_config.host, prometheus_config.port
        );
        metrics::start_prometheus(prometheus_config);
    } else {
        info!("🚧 Prometheus metrics disabled");
    }

    info!("🚧 Initializing router on {}:{}", config.host, config.port);
    info!("🚧 Router mode: {:?}", config.router_config.mode);
    info!("🚧 Policy: {:?}", config.router_config.policy);
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

    let app_state_init = AppState::new(config.router_config.clone(), client.clone())
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
        app_state.router.get_worker_urls()
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
            .service(liveness)
            .service(readiness)
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
}
