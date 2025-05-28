use crate::io_struct::{Bootstrap, ChatReqInput, GenerateReqInput};
use crate::strategy_lb::{EngineInfo, StrategyLB};
use actix_web::{HttpRequest, HttpResponse, HttpServer, get, post, web};
use bytes::Bytes;
use futures::{Stream, StreamExt, future::join_all};
use reqwest::StatusCode;
use serde_json::json;
use std::{io::Write, pin::Pin};

pub enum ProxyResponseType {
    Full(Bytes),
    Stream(Pin<Box<dyn Stream<Item = Result<Bytes, actix_web::Error>> + Send>>),
}

pub struct ProxyResponse {
    pub status: StatusCode,
    pub body: ProxyResponseType,
}

impl Into<Result<HttpResponse, actix_web::Error>> for ProxyResponse {
    fn into(self) -> Result<HttpResponse, actix_web::Error> {
        let status = actix_web::http::StatusCode::from_u16(self.status.as_u16()).map_err(|e| {
            actix_web::error::ErrorBadGateway(format!("Invalid status code: {}", e))
        })?;
        match self.body {
            ProxyResponseType::Full(body) => Ok(HttpResponse::Ok().status(status).body(body)),
            ProxyResponseType::Stream(body) => Ok(HttpResponse::Ok()
                .status(status)
                .content_type("application/octet-stream")
                .streaming(body)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LBState {
    strategy_lb: StrategyLB,
    client: reqwest::Client,
    log_interval: u64,
}

impl LBState {
    pub fn new(lb_config: LBConfig) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(lb_config.timeout))
            .build()?;

        let lb = StrategyLB::new(
            lb_config.policy,
            lb_config.prefill_infos,
            lb_config.decode_infos,
        );
        Ok(Self {
            strategy_lb: lb,
            client,
            log_interval: lb_config.log_interval,
        })
    }

    async fn route_one(
        &self,
        engine_info: EngineInfo,
        method: &str,
        api_path: &str,
        request: serde_json::Value,
        stream: bool,
    ) -> Result<ProxyResponse, actix_web::Error> {
        let url = engine_info.api_path(api_path);
        let task = match method {
            "POST" => self.client.post(&url).json(&request).send(),
            "GET" => self.client.get(&url).send(),
            _ => panic!("Unsupported method: {}", method),
        };
        let resp = task
            .await
            .map_err(|e| actix_web::error::ErrorBadGateway(e))?;
        let status = resp.status();

        if stream {
            let resp_stream = resp.bytes_stream().map(|r| {
                r.map_err(|e| actix_web::error::ErrorBadGateway(e))
                    .map(Bytes::from)
            });
            Ok(ProxyResponse {
                status,
                body: ProxyResponseType::Stream(Box::pin(resp_stream)),
            })
        } else {
            let body = resp
                .bytes()
                .await
                .map_err(|e| actix_web::error::ErrorBadGateway(e))?;
            Ok(ProxyResponse {
                status,
                body: ProxyResponseType::Full(body),
            })
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
        api_path: &str,
        mut req: Box<dyn Bootstrap>,
    ) -> Result<HttpResponse, actix_web::Error> {
        let (prefill, decode) = self.strategy_lb.select_pair(&self.client).await;
        let stream = req.is_stream();
        req.add_bootstrap_info(&prefill)?;
        let json = serde_json::to_value(req)?;
        let prefill_task = self.route_one(prefill.clone(), "POST", api_path, json.clone(), false);
        let decode_task = self.route_one(decode.clone(), "POST", api_path, json.clone(), stream);
        let (_, decode_response) = tokio::join!(prefill_task, decode_task);
        decode_response?.into()
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
pub async fn get_model_info(
    _req: HttpRequest,
    app_state: web::Data<LBState>,
) -> Result<HttpResponse, actix_web::Error> {
    // Return the first server's model info
    let engine = app_state.strategy_lb.get_one_server();
    app_state
        .route_one(
            engine,
            "GET",
            "/get_model_info",
            serde_json::Value::Null,
            false,
        )
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
    pub timeout: u64,
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
