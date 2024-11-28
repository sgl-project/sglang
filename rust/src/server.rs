use crate::router::PolicyConfig;
use crate::router::Router;
use actix_web::{get, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use bytes::Bytes;
use env_logger::Builder;
use log::{info, LevelFilter};
use std::io::Write;

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
    ) -> Self {
        // Create router based on policy
        let router = Router::new(worker_urls, policy_config);

        Self { router, client }
    }
}

async fn forward_request(
    client: &reqwest::Client,
    worker_url: String,
    route: String,
) -> HttpResponse {
    match client.get(format!("{}{}", worker_url, route)).send().await {
        Ok(res) => {
            let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

            // print the status
            println!(
                "Forwarding Request Worker URL: {}, Route: {}, Status: {}",
                worker_url, route, status
            );
            match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(_) => HttpResponse::InternalServerError().finish(),
            }
        }
        Err(_) => HttpResponse::InternalServerError().finish(),
    }
}

#[get("/health")]
async fn health(data: web::Data<AppState>) -> impl Responder {
    let worker_url = match data.router.get_first() {
        Some(url) => url,
        None => return HttpResponse::InternalServerError().finish(),
    };

    forward_request(&data.client, worker_url, "/health".to_string()).await
}

#[get("/health_generate")]
async fn health_generate(data: web::Data<AppState>) -> impl Responder {
    let worker_url = match data.router.get_first() {
        Some(url) => url,
        None => return HttpResponse::InternalServerError().finish(),
    };

    forward_request(&data.client, worker_url, "/health_generate".to_string()).await
}

#[get("/get_server_info")]
async fn get_server_info(data: web::Data<AppState>) -> impl Responder {
    let worker_url = match data.router.get_first() {
        Some(url) => url,
        None => return HttpResponse::InternalServerError().finish(),
    };

    forward_request(&data.client, worker_url, "/get_server_info".to_string()).await
}

#[get("/v1/models")]
async fn v1_models(data: web::Data<AppState>) -> impl Responder {
    let worker_url = match data.router.get_first() {
        Some(url) => url,
        None => return HttpResponse::InternalServerError().finish(),
    };

    forward_request(&data.client, worker_url, "/v1/models".to_string()).await
}

#[get("/get_model_info")]
async fn get_model_info(data: web::Data<AppState>) -> impl Responder {
    let worker_url = match data.router.get_first() {
        Some(url) => url,
        None => return HttpResponse::InternalServerError().finish(),
    };

    forward_request(&data.client, worker_url, "/get_model_info".to_string()).await
}

#[post("/generate")]
async fn generate(req: HttpRequest, body: Bytes, data: web::Data<AppState>) -> impl Responder {
    data.router
        .dispatch(&data.client, req, body, "generate")
        .await
}

#[post("/v1/chat/completions")]
async fn v1_chat_completions(
    req: HttpRequest,
    body: Bytes,
    data: web::Data<AppState>,
) -> impl Responder {
    data.router
        .dispatch(&data.client, req, body, "v1/chat/completions")
        .await
}

#[post("/v1/completions")]
async fn v1_completions(
    req: HttpRequest,
    body: Bytes,
    data: web::Data<AppState>,
) -> impl Responder {
    data.router
        .dispatch(&data.client, req, body, "v1/completions")
        .await
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub worker_urls: Vec<String>,
    pub policy_config: PolicyConfig,
    pub verbose: bool,
}

pub async fn startup(config: ServerConfig) -> std::io::Result<()> {
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

    info!("Starting server on {}:{}", config.host, config.port);
    info!("Worker URLs: {:?}", config.worker_urls);
    info!("Policy Config: {:?}", config.policy_config);

    let client = reqwest::Client::builder()
        .build()
        .expect("Failed to create HTTP client");

    let app_state = web::Data::new(AppState::new(
        config.worker_urls,
        client,
        config.policy_config,
    ));

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(generate)
            .service(v1_chat_completions)
            .service(v1_completions)
            .service(v1_models)
            .service(get_model_info)
            .service(health)
            .service(health_generate)
            .service(get_server_info)
    })
    .bind((config.host, config.port))?
    .run()
    .await
}
