use crate::io_struct::{ChatReqInput, GenerateReqInput};
use crate::lb_state::{LBConfig, LBState};
use crate::strategy_lb::EngineType;
use actix_web::{HttpRequest, HttpResponse, HttpServer, get, post, web};
use reqwest::Method;
use serde_json::json;
use std::io::Write;

#[get("/health")]
pub async fn health(_req: HttpRequest, _: web::Data<LBState>) -> HttpResponse {
    HttpResponse::Ok().body("Ok")
}

#[get("/health_generate")]
pub async fn health_generate(
    _req: HttpRequest,
    app_state: web::Data<LBState>,
) -> Result<HttpResponse, actix_web::Error> {
    let servers = app_state.strategy_lb.get_all_servers();
    app_state
        .route_collect(&servers, Method::GET, "/health_generate", None)
        .await?;
    // FIXME: log the response
    Ok(HttpResponse::Ok().body("Health check passed on all servers"))
}

#[post("/flush_cache")]
pub async fn flush_cache(
    _req: HttpRequest,
    app_state: web::Data<LBState>,
) -> Result<HttpResponse, actix_web::Error> {
    let servers = app_state.strategy_lb.get_all_servers();
    app_state
        .route_collect(&servers, Method::POST, "/flush_cache", None)
        .await?;
    Ok(HttpResponse::Ok().body("Cache flushed on all servers"))
}

#[get("/get_model_info")]
pub async fn get_model_info(
    _req: HttpRequest,
    app_state: web::Data<LBState>,
) -> Result<HttpResponse, actix_web::Error> {
    // Return the first server's model info
    let engine = app_state.strategy_lb.get_one_server();
    app_state
        .route_one(&engine, Method::GET, "/get_model_info", None, false)
        .await?
        .into()
}

#[post("/generate")]
pub async fn generate(
    _req: HttpRequest,
    req: web::Json<GenerateReqInput>,
    app_state: web::Data<LBState>,
) -> Result<HttpResponse, actix_web::Error> {
    app_state
        .generate("/generate", Box::new(req.into_inner()))
        .await
}

#[post("/v1/completions")]
pub async fn completions(
    _req: HttpRequest,
    req: web::Json<GenerateReqInput>,
    app_state: web::Data<LBState>,
) -> Result<HttpResponse, actix_web::Error> {
    app_state
        .generate("/v1/completions", Box::new(req.into_inner()))
        .await
}

#[post("/v1/chat/completions")]
pub async fn chat_completions(
    _req: HttpRequest,
    req: web::Json<ChatReqInput>,
    app_state: web::Data<LBState>,
) -> Result<HttpResponse, actix_web::Error> {
    app_state
        .generate("/v1/chat/completions", Box::new(req.into_inner()))
        .await
}

#[get("/get_server_info")]
pub async fn get_server_info(
    _req: HttpRequest,
    app_state: web::Data<LBState>,
) -> Result<HttpResponse, actix_web::Error> {
    let servers = app_state.strategy_lb.get_all_servers();
    let responses = app_state
        .route_collect(&servers, Method::GET, "/get_server_info", None)
        .await?;
    let mut prefill_infos = Vec::new();
    let mut decode_infos = Vec::new();
    for (i, resp) in responses.iter().enumerate() {
        let json = resp.to_json()?;
        match servers[i].engine_type {
            EngineType::Prefill => prefill_infos.push(json),
            EngineType::Decode => decode_infos.push(json),
        }
    }
    Ok(HttpResponse::Ok().json(json!({
        "prefill": prefill_infos,
        "decode": decode_infos,
    })))
}

#[get("/get_loads")]
pub async fn get_loads(
    _req: HttpRequest,
    app_state: web::Data<LBState>,
) -> Result<HttpResponse, actix_web::Error> {
    let (prefill_loads, decode_loads) = app_state.get_engine_loads().await?;
    Ok(HttpResponse::Ok().json(json!({
        "prefill": prefill_loads.into_iter().map(|l| l.to_json()).collect::<Vec<_>>(),
        "decode": decode_loads.into_iter().map(|l| l.to_json()).collect::<Vec<_>>()
    })))
}

pub async fn periodic_logging(lb_state: LBState) {
    // FIXME: currently we can just clone the lb_state to log as the lb is stateless
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(lb_state.log_interval)).await;
        let (prefill_loads, decode_loads) = match lb_state.get_engine_loads().await {
            Ok((prefill_loads, decode_loads)) => (prefill_loads, decode_loads),
            Err(e) => {
                log::error!("Failed to get engine loads: {}", e);
                continue;
            }
        };
        let prefill_loads = prefill_loads
            .into_iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>();
        let decode_loads = decode_loads
            .into_iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>();
        log::info!("Prefill loads: {}", prefill_loads.join(", "));
        log::info!("Decode loads: {}", decode_loads.join(", "));
    }
}

pub async fn startup(lb_config: LBConfig, lb_state: LBState) -> std::io::Result<()> {
    let app_state = web::Data::new(lb_state);

    println!("Starting server at {}:{}", lb_config.host, lb_config.port);

    // default level is info
    env_logger::Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} - {} - {}",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .filter(None, log::LevelFilter::Info)
        .init();

    HttpServer::new(move || {
        actix_web::App::new()
            .wrap(actix_web::middleware::Logger::default())
            .app_data(app_state.clone())
            .service(health)
            .service(health_generate)
            .service(flush_cache)
            .service(get_model_info)
            .service(get_server_info)
            .service(get_loads)
            .service(generate)
            .service(chat_completions)
            .service(completions)
    })
    .bind((lb_config.host, lb_config.port))?
    .run()
    .await?;

    std::io::Result::Ok(())
}
