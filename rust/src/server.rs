use crate::router::PolicyConfig;
use crate::router::Router;
use actix_web::{get, post, web, App, HttpRequest, HttpResponse, HttpServer, Responder};
use bytes::Bytes;

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
            println!("Worker URL: {}, Status: {}", worker_url, status);
            match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(_) => HttpResponse::InternalServerError().finish(),
            }
        }
        Err(_) => HttpResponse::InternalServerError().finish(),
    }
}

async fn mock_forward_request(route: &str) -> HttpResponse {
    match route {
        "/v1/models" => {
            let mock_response = serde_json::json!({
                "object": "list",
                "data": [{
                    "id": "mock_model",
                    "object": "model",
                    "created": (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() as i64),
                    "owned_by": "sglang",
                    "root": "mock_model"
                }]
            });

            HttpResponse::Ok()
                .content_type("application/json")
                .body(serde_json::to_vec(&mock_response).unwrap_or_default())
        }
        "/get_model_info" => {
            let mock_response = serde_json::json!({
                "model_path": "/path/to/mock/model",
                "is_generation": true
            });

            HttpResponse::Ok()
                .content_type("application/json")
                .body(serde_json::to_vec(&mock_response).unwrap_or_default())
        }
        _ => HttpResponse::NotFound().finish()
    }
}


#[get("/v1/models")]
async fn v1_model(data: web::Data<AppState>) -> impl Responder {
    let worker_url = match data.router.get_first() {
        Some(url) => url,
        None => return HttpResponse::InternalServerError().finish(),
    };

    forward_request(&data.client, worker_url, "/v1/models".to_string()).await
    // mock_forward_request("/v1/models").await
}

#[get("/get_model_info")]
async fn get_model_info(data: web::Data<AppState>) -> impl Responder {
    let worker_url = match data.router.get_first() {
        Some(url) => url,
        None => return HttpResponse::InternalServerError().finish(),
    };

    forward_request(&data.client, worker_url, "/get_model_info".to_string()).await
    // mock_forward_request("/get_model_info").await
}

#[post("/generate")]
async fn generate(req: HttpRequest, body: Bytes, data: web::Data<AppState>) -> impl Responder {
    data.router.dispatch(&data.client, req, body).await
}

pub async fn startup(
    host: String,
    port: u16,
    worker_urls: Vec<String>,
    policy_config: PolicyConfig,
) -> std::io::Result<()> {
    println!("Starting server on {}:{}", host, port);
    println!("Worker URLs: {:?}", worker_urls);

    // Create client once with configuration
    let client = reqwest::Client::builder()
        .build()
        .expect("Failed to create HTTP client");

    // Store both worker_urls and client in AppState
    let app_state = web::Data::new(AppState::new(worker_urls, client, policy_config));

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(generate)
            .service(v1_model)
            .service(get_model_info)
    })
    .bind((host, port))?
    .run()
    .await
}
