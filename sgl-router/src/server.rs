use crate::router::PolicyConfig;
use crate::router::Router;
use actix_web::{
    error, get, post, web, App, Error, HttpRequest, HttpResponse, HttpServer, Responder,
};
use bytes::Bytes;
use env_logger::Builder;
use futures_util::StreamExt;
use log::{info, LevelFilter};
use std::collections::HashMap;
use std::io::Write;
use std::time::Duration;

#[derive(Debug)]
pub struct AppState {
    router: Router,
    client: reqwest::Client,
}

impl AppState {
    pub fn new(
        worker_urls: Vec<String>,
        client: reqwest::Client,
        policy_config: PolicyConfig,
    ) -> Result<Self, String> {
        // Create router based on policy
        let router = Router::new(worker_urls, policy_config)?;
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
fn json_error_handler(_err: error::JsonPayloadError, _req: &HttpRequest) -> Error {
    error::ErrorPayloadTooLarge("Payload too large")
}

#[get("/health")]
async fn health(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/health", &req)
        .await
}

#[get("/health_generate")]
async fn health_generate(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/health_generate", &req)
        .await
}

#[get("/get_server_info")]
async fn get_server_info(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/get_server_info", &req)
        .await
}

#[get("/v1/models")]
async fn v1_models(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/v1/models", &req)
        .await
}

#[get("/get_model_info")]
async fn get_model_info(req: HttpRequest, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_to_first(&data.client, "/get_model_info", &req)
        .await
}

#[post("/generate")]
async fn generate(req: HttpRequest, body: Bytes, data: web::Data<AppState>) -> impl Responder {
    data.router
        .route_generate_request(&data.client, &req, &body, "/generate")
        .await
}

#[post("/v1/chat/completions")]
async fn v1_chat_completions(
    req: HttpRequest,
    body: Bytes,
    data: web::Data<AppState>,
) -> impl Responder {
    data.router
        .route_generate_request(&data.client, &req, &body, "/v1/chat/completions")
        .await
}

#[post("/v1/completions")]
async fn v1_completions(
    req: HttpRequest,
    body: Bytes,
    data: web::Data<AppState>,
) -> impl Responder {
    data.router
        .route_generate_request(&data.client, &req, &body, "/v1/completions")
        .await
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

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub worker_urls: Vec<String>,
    pub policy_config: PolicyConfig,
    pub verbose: bool,
    pub max_payload_size: usize,
}

pub async fn startup(config: ServerConfig) -> std::io::Result<()> {
    // Initialize logger
    Builder::new()
        .format(|buf, record| {
            use chrono::Local;
            writeln!(
                buf,
                "[Router (Rust)] {} - {} - {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter(
            None,
            if config.verbose {
                LevelFilter::Debug
            } else {
                LevelFilter::Info
            },
        )
        .init();

    info!("🚧 Initializing router on {}:{}", config.host, config.port);
    info!("🚧 Initializing workers on {:?}", config.worker_urls);
    info!("🚧 Policy Config: {:?}", config.policy_config);
    info!(
        "🚧 Max payload size: {} MB",
        config.max_payload_size / (1024 * 1024)
    );

    let client = reqwest::Client::builder()
        .pool_idle_timeout(Some(Duration::from_secs(50)))
        .build()
        .expect("Failed to create HTTP client");

    let app_state = web::Data::new(
        AppState::new(
            config.worker_urls.clone(),
            client,
            config.policy_config.clone(),
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?,
    );

    info!("✅ Serving router on {}:{}", config.host, config.port);
    info!("✅ Serving workers on {:?}", config.worker_urls);

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
            // Default handler for unmatched routes.
            .default_service(web::route().to(sink_handler))
    })
    .bind((config.host, config.port))?
    .run()
    .await
}
