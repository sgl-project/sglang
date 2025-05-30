use crate::io_struct::Bootstrap;
use crate::strategy_lb::{EngineInfo, EngineLoad, EngineType, LBPolicy, StrategyLB};
use actix_web::HttpResponse;
use bytes::Bytes;
use futures::{Stream, StreamExt, future::join_all};
use reqwest::{Method, StatusCode};
use std::pin::Pin;

pub enum ProxyResponseBody {
    Full(Bytes),
    Stream(Pin<Box<dyn Stream<Item = Result<Bytes, actix_web::Error>> + Send>>),
}

pub struct ProxyResponse {
    pub status: StatusCode,
    pub body: ProxyResponseBody,
}

impl ProxyResponse {
    pub fn to_json(&self) -> Result<serde_json::Value, actix_web::Error> {
        match &self.body {
            ProxyResponseBody::Full(body) => Ok(serde_json::from_slice(&body)?),
            ProxyResponseBody::Stream(_) => Err(actix_web::error::ErrorBadRequest(
                "Stream response is not supported",
            )),
        }
    }
}

impl Into<Result<HttpResponse, actix_web::Error>> for ProxyResponse {
    fn into(self) -> Result<HttpResponse, actix_web::Error> {
        let status = actix_web::http::StatusCode::from_u16(self.status.as_u16()).map_err(|e| {
            actix_web::error::ErrorBadGateway(format!("Invalid status code: {}", e))
        })?;
        match self.body {
            ProxyResponseBody::Full(body) => Ok(HttpResponse::Ok().status(status).body(body)),
            ProxyResponseBody::Stream(body) => Ok(HttpResponse::Ok()
                .status(status)
                .content_type("application/octet-stream")
                .streaming(body)),
        }
    }
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

#[derive(Debug, Clone)]
pub struct LBState {
    pub strategy_lb: StrategyLB,
    pub client: reqwest::Client,
    pub log_interval: u64,
}

impl LBState {
    pub fn new(lb_config: LBConfig) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(lb_config.timeout))
            .build()?;
        let policy = match lb_config.policy.as_str() {
            "random" => LBPolicy::Random,
            "po2" => LBPolicy::PowerOfTwo,
            _ => anyhow::bail!("Invalid policy"),
        };
        let prefill_servers = lb_config
            .prefill_infos
            .into_iter()
            .map(|(url, port)| EngineInfo::new_prefill(url, port))
            .collect();
        let decode_servers = lb_config
            .decode_infos
            .into_iter()
            .map(|url| EngineInfo::new_decode(url))
            .collect();
        let lb = StrategyLB::new(policy, prefill_servers, decode_servers);
        Ok(Self {
            strategy_lb: lb,
            client,
            log_interval: lb_config.log_interval,
        })
    }

    pub async fn route_one(
        &self,
        engine_info: &EngineInfo,
        method: Method,
        api_path: &str,
        request: Option<&serde_json::Value>,
        stream: bool,
    ) -> Result<ProxyResponse, actix_web::Error> {
        let url = engine_info.api_path(api_path);
        let request = request.unwrap_or(&serde_json::Value::Null);
        let task = self.client.request(method, url).json(request).send();
        let resp = task.await.map_err(actix_web::error::ErrorBadGateway)?;
        // FIXME: handle error status code (map status code to error)
        let status = resp.status();
        let body = if stream {
            let resp_stream = resp.bytes_stream().map(|r| {
                r.map_err(actix_web::error::ErrorBadGateway)
                    .map(Bytes::from)
            });
            ProxyResponseBody::Stream(Box::pin(resp_stream))
        } else {
            let body = resp
                .bytes()
                .await
                .map_err(actix_web::error::ErrorBadGateway)?;
            ProxyResponseBody::Full(body)
        };
        Ok(ProxyResponse { status, body })
    }

    pub async fn route_collect(
        &self,
        engines: &Vec<EngineInfo>,
        method: Method,
        api_path: &str,
        request: Option<&serde_json::Value>,
    ) -> Result<Vec<ProxyResponse>, actix_web::Error> {
        let tasks = engines
            .iter()
            .map(|engine| self.route_one(engine, method.clone(), api_path, request, false));
        let responses = join_all(tasks).await;
        responses
            .into_iter()
            .map(|r| r.map_err(actix_web::error::ErrorBadGateway))
            .collect()
    }

    pub async fn generate(
        &self,
        api_path: &str,
        mut req: Box<dyn Bootstrap>,
    ) -> Result<HttpResponse, actix_web::Error> {
        let (prefill, decode) = self.strategy_lb.select_pair(&self.client).await;
        let stream = req.is_stream();
        req.add_bootstrap_info(&prefill)?;
        let json = serde_json::to_value(req)?;
        let prefill_task = self.route_one(&prefill, Method::POST, api_path, Some(&json), false);
        let decode_task = self.route_one(&decode, Method::POST, api_path, Some(&json), stream);
        let (_, decode_response) = tokio::join!(prefill_task, decode_task);
        decode_response?.into()
    }

    pub async fn get_engine_loads(
        &self,
    ) -> Result<(Vec<EngineLoad>, Vec<EngineLoad>), actix_web::Error> {
        let servers = self.strategy_lb.get_all_servers();
        let responses = self
            .route_collect(&servers, Method::GET, "/get_load", None)
            .await?;
        let loads = responses
            .into_iter()
            .enumerate()
            .map(|(i, r)| Ok(EngineLoad::from_json(&servers[i], &r.to_json()?)))
            .collect::<Result<Vec<EngineLoad>, actix_web::Error>>()?;
        let mut prefill_loads = Vec::new();
        let mut decode_loads = Vec::new();
        for load in loads {
            match load.engine_info.engine_type {
                EngineType::Prefill => prefill_loads.push(load),
                EngineType::Decode => decode_loads.push(load),
            }
        }
        Ok((prefill_loads, decode_loads))
    }
}
