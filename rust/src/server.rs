use actix_web::{get, post, web, App, HttpServer, HttpResponse, HttpRequest, Responder};
use bytes::Bytes;
use futures_util::StreamExt;
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use crate::router::Router;
use crate::router::create_router;


#[derive(Debug)]
pub struct AppState {
    router: Box<dyn Router>,
    client: reqwest::Client,
}


impl AppState
{
    pub fn new(worker_urls: Vec<String>, policy: String, client: reqwest::Client) -> Self {
        // Create router based on policy
        let router = create_router(worker_urls, policy);
        
        Self {
            router,
            client,
        }
    }
}

#[get("/v1/models")]
async fn v1_model(
    data: web::Data<AppState>,
) -> impl Responder {
    let worker_url= match data.router.get_first() {
        Some(url) => url,
        None => return HttpResponse::InternalServerError().finish(),
    };
    // Use the shared client
    match data.client
        .get(&format!("{}/v1/models", worker_url))
        .send()
        .await 
    {
        Ok(res) => {
            let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
        
            // print the status
            println!("Worker URL: {}, Status: {}", worker_url, status);
            match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(_) => HttpResponse::InternalServerError().finish(),
            }
        },
        Err(_) => HttpResponse::InternalServerError().finish(),
    }
}

#[get("/get_model_info")]
async fn get_model_info(
    data: web::Data<AppState>,
) -> impl Responder {
    let worker_url= match data.router.get_first() {
        Some(url) => url,
        None => return HttpResponse::InternalServerError().finish(),
    };
    // Use the shared client
    match data.client
        .get(&format!("{}/get_model_info", worker_url))
        .send()
        .await 
    {
        Ok(res) => {
            let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
        
            // print the status
            println!("Worker URL: {}, Status: {}", worker_url, status);
            match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(_) => HttpResponse::InternalServerError().finish(),
            }
        },
        Err(_) => HttpResponse::InternalServerError().finish(),
    }
}

// no deser and ser, just forward and return
#[post("/generate")]
async fn generate(
    req: HttpRequest,
    body: Bytes,
    data: web::Data<AppState>,
) -> impl Responder {

    // create a router struct
    // TODO: use router abstraction for different policy
    let worker_url= match data.router.select() {
        Some(url) => url,
        None => return HttpResponse::InternalServerError().finish(),
    };

    // Check if client requested streaming
    let is_stream = serde_json::from_slice::<serde_json::Value>(&body)
        .map(|v| v.get("stream").and_then(|s| s.as_bool()).unwrap_or(false))
        .unwrap_or(false);

    let res = match data.client
        .post(&format!("{}/generate", worker_url))
        .header(
            "Content-Type", 
            req.headers()
                .get("Content-Type")
                .and_then(|h| h.to_str().ok())
                .unwrap_or("application/json")
        )
        .body(body.to_vec())
        .send()
        .await 
    {
        Ok(res) => res,
        Err(_) => return HttpResponse::InternalServerError().finish(),
    };

    let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
        .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

    if !is_stream {
        match res.bytes().await {
            Ok(body) => HttpResponse::build(status).body(body.to_vec()),
            Err(_) => HttpResponse::InternalServerError().finish(),
        } 
    } else {
        HttpResponse::build(status)
            .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
            .streaming(res.bytes_stream().map(|b| match b {
                Ok(b) => Ok::<_, actix_web::Error>(b),
                Err(_) => Err(actix_web::Error::from(actix_web::error::ErrorInternalServerError("Failed to read stream"))),
            }))
    }
}

pub async fn startup(host: String, port: u16, worker_urls: Vec<String>, routing_policy: String) -> std::io::Result<()> {
    println!("Starting server on {}:{}", host, port);
    println!("Worker URLs: {:?}", worker_urls);

    // Create client once with configuration
    let client = reqwest::Client::builder()
        .build()
        .expect("Failed to create HTTP client");

    // Store both worker_urls and client in AppState
    let app_state = web::Data::new(AppState::new(
        worker_urls,
        routing_policy,
        client,
    ));

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