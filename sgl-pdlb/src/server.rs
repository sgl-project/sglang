use crate::strategy_lb::{EngineInfo, StrategyLB};
use actix_web::{HttpRequest, HttpResponse, HttpServer, get, post, web};
use bytes::Bytes;
use futures::StreamExt;
use futures::future::join_all;
use serde_json::json;
use std::io::Write;

#[derive(Debug, Clone)]
pub struct LBState {
    strategy_lb: StrategyLB,
    client: reqwest::Client,
    log_interval: u64,
}

impl LBState {
    pub fn new(lb_config: LBConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(50))
            .build()
            .expect("Failed to build HTTP client");

        let lb = StrategyLB::new(
            lb_config.policy,
            lb_config.prefill_infos,
            lb_config.decode_infos,
        );
        Self {
            strategy_lb: lb,
            client,
            log_interval: lb_config.log_interval,
        }
    }

    pub fn modify_request(
        request: &serde_json::Value,
        prefill_info: &EngineInfo,
    ) -> serde_json::Value {
        let mut modified_request = request.clone();
        modified_request["bootstrap_host"] = prefill_info.get_hostname().into();
        modified_request["bootstrap_room"] = rand::random::<u64>().into();
        modified_request["bootstrap_port"] = prefill_info.boostrap_port.into();
        modified_request
    }

    async fn route_one(
        &self,
        engine_info: EngineInfo,
        api_path: &str,
        request: serde_json::Value,
    ) -> HttpResponse {
        let url = engine_info.api_path(api_path);
        let response = self.client.post(&url).json(&request).send().await;
        match response {
            Ok(response) => HttpResponse::Ok().body(response.bytes().await.unwrap()),
            Err(e) => HttpResponse::InternalServerError()
                .body(format!("Failed to send request to worker {}: {}", url, e)),
        }
    }

    async fn route_one_stream(
        &self,
        engine_info: EngineInfo,
        api_path: &str,
        request: serde_json::Value,
    ) -> HttpResponse {
        let url = engine_info.api_path(api_path);
        let response = self.client.post(&url).json(&request).send().await;
        match response {
            Ok(response) => {
                let stream = response.bytes_stream();
                let response_stream = futures::stream::unfold(stream, |mut stream| async move {
                    match stream.next().await {
                        Some(Ok(chunk)) => {
                            let chunk = Bytes::from(chunk);
                            Some((Ok::<Bytes, reqwest::Error>(chunk), stream))
                        }
                        Some(Err(_)) => None,
                        None => None,
                    }
                });
                HttpResponse::Ok()
                    .content_type("application/octet-stream")
                    .streaming(response_stream)
            }
            Err(e) => HttpResponse::InternalServerError()
                .body(format!("Failed to send request to worker {}: {}", url, e)),
        }
    }

    async fn route_collect(
        &self,
        engines: Vec<EngineInfo>,
        method: &str,
        api_path: &str,
        request: serde_json::Value,
    ) -> Vec<Result<reqwest::Response, reqwest::Error>> {
        let mut tasks = Vec::new();
        for engine in engines {
            let url = engine.api_path(api_path);
            let task = match method {
                "POST" => self.client.post(&url).json(&request).send(),
                "GET" => self.client.get(&url).send(),
                _ => panic!("Unsupported method: {}", method),
            };
            tasks.push(task);
        }
        join_all(tasks).await
    }

    async fn route_collect_json(
        &self,
        engines: Vec<EngineInfo>,
        method: &str,
        api_path: &str,
        request: serde_json::Value,
        discard_errors: bool,
    ) -> Vec<serde_json::Value> {
        // This will discard error responses
        let responses = self.route_collect(engines, method, api_path, request).await;
        let mut results = Vec::new();
        for response in responses {
            match response {
                Ok(response) => {
                    if response.status().is_success() {
                        results.push(response.json::<serde_json::Value>().await.unwrap());
                    } else {
                        results.push(json!({"error": "Failed to get response from server"}));
                    }
                }
                Err(_) => {
                    if !discard_errors {
                        results.push(json!({"error": "Failed to get response from server"}));
                    }
                }
            }
        }
        results
    }

    async fn generate(
        &self,
        prefill: &EngineInfo,
        decode: &EngineInfo,
        modified_json: serde_json::Value,
    ) -> HttpResponse {
        let prefill_task = self.route_one(prefill.clone(), "/generate", modified_json.clone());
        let decode_task = self.route_one(decode.clone(), "/generate", modified_json.clone());
        let (_, decode_response) = tokio::join!(prefill_task, decode_task);
        decode_response
    }

    async fn generate_stream(
        &self,
        prefill: &EngineInfo,
        decode: &EngineInfo,
        modified_json: serde_json::Value,
    ) -> HttpResponse {
        let prefill_task = self.route_one(prefill.clone(), "/generate", modified_json.clone());
        let decode_task = self.route_one_stream(decode.clone(), "/generate", modified_json.clone());
        let (_, decode_response) = tokio::join!(prefill_task, decode_task);
        decode_response
    }

    async fn get_engine_loads(
        &self,
        remove_offline: bool,
    ) -> (Vec<(String, isize)>, Vec<(String, isize)>) {
        let prefill_servers = self.strategy_lb.prefill_servers.clone();
        let decode_servers = self.strategy_lb.decode_servers.clone();
        let prefill_loads = self
            .route_collect_json(
                prefill_servers.clone(),
                "GET",
                "/get_load",
                serde_json::Value::Null,
                remove_offline,
            )
            .await;
        let decode_loads = self
            .route_collect_json(
                decode_servers.clone(),
                "GET",
                "/get_load",
                serde_json::Value::Null,
                remove_offline,
            )
            .await;
        let mut prefill_loads: Vec<(String, isize)> = prefill_loads
            .into_iter()
            .enumerate()
            .map(|(i, load)| {
                (
                    prefill_servers[i].to_string(),
                    load["load"].as_i64().unwrap_or(-1) as isize,
                )
            })
            .collect();
        let mut decode_loads: Vec<(String, isize)> = decode_loads
            .into_iter()
            .enumerate()
            .map(|(i, load)| {
                (
                    decode_servers[i].to_string(),
                    load["load"].as_i64().unwrap_or(-1) as isize,
                )
            })
            .collect();
        if remove_offline {
            prefill_loads.retain(|(_, load)| *load != -1);
            decode_loads.retain(|(_, load)| *load != -1);
        }
        (prefill_loads, decode_loads)
    }
}

#[get("/health")]
pub async fn health(_req: HttpRequest, _: web::Data<LBState>) -> HttpResponse {
    HttpResponse::Ok().body("Ok")
}

#[get("/health_generate")]
pub async fn health_generate(_req: HttpRequest, app_state: web::Data<LBState>) -> HttpResponse {
    let all_servers = app_state.strategy_lb.get_all_servers();
    let responses = app_state
        .route_collect(
            all_servers,
            "GET",
            "/health_generate",
            serde_json::Value::Null,
        )
        .await;
    if responses.iter().any(|r| r.is_err()) {
        HttpResponse::InternalServerError().body("Failed to get health from some servers")
    } else {
        HttpResponse::Ok().body("Health check passed on all servers")
    }
}

#[post("/flush_cache")]
pub async fn flush_cache(_req: HttpRequest, app_state: web::Data<LBState>) -> HttpResponse {
    let all_servers = app_state.strategy_lb.get_all_servers();
    let responses = app_state
        .route_collect(all_servers, "POST", "/flush_cache", serde_json::Value::Null)
        .await;
    if responses.iter().any(|r| r.is_err()) {
        HttpResponse::InternalServerError().body("Failed to flush cache on some servers")
    } else {
        HttpResponse::Ok().body("Cache flushed on all servers")
    }
}

#[get("/get_model_info")]
pub async fn get_model_info(_req: HttpRequest, app_state: web::Data<LBState>) -> HttpResponse {
    let model_infos = app_state
        .route_collect_json(
            app_state.strategy_lb.get_all_servers(),
            "GET",
            "/get_model_info",
            serde_json::Value::Null,
            false,
        )
        .await;
    HttpResponse::Ok().json(model_infos)
}

#[post("/generate")]
pub async fn generate(
    _req: HttpRequest,
    json: web::Json<serde_json::Value>,
    app_state: web::Data<LBState>,
) -> HttpResponse {
    let (prefill, decode) = app_state.strategy_lb.select_pair(&app_state.client).await;
    let modified_json = LBState::modify_request(&json, &prefill);

    if modified_json.get("stream").is_none() || modified_json["stream"] == false {
        app_state.generate(&prefill, &decode, modified_json).await
    } else {
        app_state
            .generate_stream(&prefill, &decode, modified_json)
            .await
    }
}

#[post("/v1/chat/completions")]
pub async fn chat_completions(
    _req: HttpRequest,
    json: web::Json<serde_json::Value>,
    app_state: web::Data<LBState>,
) -> HttpResponse {
    let (prefill, decode) = app_state.strategy_lb.select_pair(&app_state.client).await;
    let modified_json = LBState::modify_request(&json, &prefill);

    if modified_json.get("stream").is_none() || modified_json["stream"] == false {
        app_state.generate(&prefill, &decode, modified_json).await
    } else {
        app_state
            .generate_stream(&prefill, &decode, modified_json)
            .await
    }
}

#[get("/get_server_info")]
pub async fn get_server_info(_req: HttpRequest, app_state: web::Data<LBState>) -> HttpResponse {
    let prefill_server_infos = app_state
        .route_collect_json(
            app_state
                .strategy_lb
                .prefill_servers
                .iter()
                .cloned()
                .collect(),
            "GET",
            "/get_server_info",
            serde_json::Value::Null,
            false,
        )
        .await;
    let decode_server_infos = app_state
        .route_collect_json(
            app_state
                .strategy_lb
                .decode_servers
                .iter()
                .cloned()
                .collect(),
            "GET",
            "/get_server_info",
            serde_json::Value::Null,
            false,
        )
        .await;
    HttpResponse::Ok().json(json!({
        "prefill": prefill_server_infos,
        "decode": decode_server_infos,
    }))
}

#[get("/get_loads")]
pub async fn get_loads(_req: HttpRequest, app_state: web::Data<LBState>) -> HttpResponse {
    // return invalid load as -1
    let (prefill_loads, decode_loads) = app_state.get_engine_loads(false).await;
    let res = json!({
        "prefill": prefill_loads,
        "decode": decode_loads,
    });
    HttpResponse::Ok().json(res)
}

#[derive(Debug, Clone)]
pub struct LBConfig {
    pub host: String,
    pub port: u16,
    pub policy: String,
    pub prefill_infos: Vec<(String, Option<u16>)>,
    pub decode_infos: Vec<String>,
    pub log_interval: u64,
}

pub async fn periodic_logging(lb_state: LBState) {
    // FIXME: currently we can just clone the lb_state to log as the lb is stateless
    loop {
        tokio::time::sleep(std::time::Duration::from_secs(lb_state.log_interval)).await;
        let (prefill_loads, decode_loads) = lb_state.get_engine_loads(true).await;
        log::info!("Prefill loads: {:?}", prefill_loads);
        log::info!("Decode loads: {:?}", decode_loads);
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
    })
    .bind((lb_config.host, lb_config.port))?
    .run()
    .await?;

    std::io::Result::Ok(())
}
