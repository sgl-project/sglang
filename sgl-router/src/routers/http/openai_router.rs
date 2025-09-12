//! OpenAI router implementation (reqwest-based)

use crate::config::CircuitBreakerConfig;
use crate::core::{CircuitBreaker, CircuitBreakerConfig as CoreCircuitBreakerConfig};
use crate::protocols::spec::{ChatCompletionRequest, CompletionRequest, GenerateRequest};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use std::{
    any::Any,
    sync::atomic::{AtomicBool, Ordering},
};

/// Router for OpenAI backend
#[derive(Debug)]
pub struct OpenAIRouter {
    /// HTTP client for upstream OpenAI-compatible API
    client: reqwest::Client,
    /// Base URL for identification (no trailing slash)
    base_url: String,
    /// Circuit breaker
    circuit_breaker: CircuitBreaker,
    /// Health status
    healthy: AtomicBool,
}

impl OpenAIRouter {
    /// Create a new OpenAI router
    pub async fn new(
        base_url: String,
        circuit_breaker_config: Option<CircuitBreakerConfig>,
    ) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        let base_url = base_url.trim_end_matches('/').to_string();

        // Convert circuit breaker config
        let core_cb_config = circuit_breaker_config
            .map(|cb| CoreCircuitBreakerConfig {
                failure_threshold: cb.failure_threshold,
                success_threshold: cb.success_threshold,
                timeout_duration: std::time::Duration::from_secs(cb.timeout_duration_secs),
                window_duration: std::time::Duration::from_secs(cb.window_duration_secs),
            })
            .unwrap_or_default();

        let circuit_breaker = CircuitBreaker::with_config(core_cb_config);

        Ok(Self {
            client,
            base_url,
            circuit_breaker,
            healthy: AtomicBool::new(true),
        })
    }
}

#[async_trait]
impl super::super::WorkerManagement for OpenAIRouter {
    async fn add_worker(&self, _worker_url: &str) -> Result<String, String> {
        Err("Cannot add workers to OpenAI router".to_string())
    }

    fn remove_worker(&self, _worker_url: &str) {
        // No-op for OpenAI router
    }

    fn get_worker_urls(&self) -> Vec<String> {
        vec![self.base_url.clone()]
    }
}

#[async_trait]
impl super::super::RouterTrait for OpenAIRouter {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn health(&self, _req: Request<Body>) -> Response {
        // Simple upstream probe: GET {base}/v1/models without auth
        let url = format!("{}/v1/models", self.base_url);
        match self
            .client
            .get(&url)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await
        {
            Ok(resp) => {
                let code = resp.status();
                // Treat success and auth-required as healthy (endpoint reachable)
                if code.is_success() || code.as_u16() == 401 || code.as_u16() == 403 {
                    (StatusCode::OK, "OK").into_response()
                } else {
                    (
                        StatusCode::SERVICE_UNAVAILABLE,
                        format!("Upstream status: {}", code),
                    )
                        .into_response()
                }
            }
            Err(e) => (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Upstream error: {}", e),
            )
                .into_response(),
        }
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // For OpenAI, health_generate is the same as health
        self.health(_req).await
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        let info = serde_json::json!({
            "router_type": "openai",
            "workers": 1,
            "base_url": &self.base_url
        });
        (StatusCode::OK, info.to_string()).into_response()
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Proxy to upstream /v1/models; forward Authorization header if provided
        let headers = req.headers();

        let mut upstream = self.client.get(format!("{}/v1/models", self.base_url));

        if let Some(auth) = headers
            .get("authorization")
            .or_else(|| headers.get("Authorization"))
        {
            upstream = upstream.header("Authorization", auth);
        }

        match upstream.send().await {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                let content_type = res.headers().get(CONTENT_TYPE).cloned();
                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(axum::body::Body::from(body));
                        *response.status_mut() = status;
                        if let Some(ct) = content_type {
                            response.headers_mut().insert(CONTENT_TYPE, ct);
                        }
                        response
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to read upstream response: {}", e),
                    )
                        .into_response(),
                }
            }
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                format!("Failed to contact upstream: {}", e),
            )
                .into_response(),
        }
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        // Not directly supported without model param; return 501
        (
            StatusCode::NOT_IMPLEMENTED,
            "get_model_info not implemented for OpenAI router",
        )
            .into_response()
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &GenerateRequest,
    ) -> Response {
        // Generate endpoint is SGLang-specific, not supported for OpenAI backend
        (
            StatusCode::NOT_IMPLEMENTED,
            "Generate endpoint not supported for OpenAI backend",
        )
            .into_response()
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
    ) -> Response {
        if !self.circuit_breaker.can_execute() {
            return (StatusCode::SERVICE_UNAVAILABLE, "Circuit breaker open").into_response();
        }

        // Serialize request body, removing SGLang-only fields
        let mut payload = match serde_json::to_value(body) {
            Ok(v) => v,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to serialize request: {}", e),
                )
                    .into_response();
            }
        };
        if let Some(obj) = payload.as_object_mut() {
            for key in [
                "top_k",
                "min_p",
                "min_tokens",
                "regex",
                "ebnf",
                "stop_token_ids",
                "no_stop_trim",
                "ignore_eos",
                "continue_final_message",
                "skip_special_tokens",
                "lora_path",
                "session_params",
                "separate_reasoning",
                "stream_reasoning",
                "chat_template_kwargs",
                "return_hidden_states",
                "repetition_penalty",
            ] {
                obj.remove(key);
            }
        }

        let url = format!("{}/v1/chat/completions", self.base_url);
        let mut req = self.client.post(&url).json(&payload);

        // Forward Authorization header if provided
        if let Some(h) = headers {
            if let Some(auth) = h.get("authorization").or_else(|| h.get("Authorization")) {
                req = req.header("Authorization", auth);
            }
        }

        // Accept SSE when stream=true
        if body.stream {
            req = req.header("Accept", "text/event-stream");
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                self.circuit_breaker.record_failure();
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("Failed to contact upstream: {}", e),
                )
                    .into_response();
            }
        };

        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !body.stream {
            // Capture Content-Type before consuming response body
            let content_type = resp.headers().get(CONTENT_TYPE).cloned();
            match resp.bytes().await {
                Ok(body) => {
                    self.circuit_breaker.record_success();
                    let mut response = Response::new(axum::body::Body::from(body));
                    *response.status_mut() = status;
                    if let Some(ct) = content_type {
                        response.headers_mut().insert(CONTENT_TYPE, ct);
                    }
                    response
                }
                Err(e) => {
                    self.circuit_breaker.record_failure();
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to read response: {}", e),
                    )
                        .into_response()
                }
            }
        } else {
            // Stream SSE bytes to client
            let stream = resp.bytes_stream();
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            tokio::spawn(async move {
                let mut s = stream;
                while let Some(chunk) = s.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if tx.send(Ok(bytes)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(format!("Stream error: {}", e)));
                            break;
                        }
                    }
                }
            });
            let mut response = Response::new(Body::from_stream(
                tokio_stream::wrappers::UnboundedReceiverStream::new(rx),
            ));
            *response.status_mut() = status;
            response
                .headers_mut()
                .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
            response
        }
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &CompletionRequest,
    ) -> Response {
        // Completion endpoint not implemented for OpenAI backend
        (
            StatusCode::NOT_IMPLEMENTED,
            "Completion endpoint not implemented for OpenAI backend",
        )
            .into_response()
    }

    async fn flush_cache(&self) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "flush_cache not supported for OpenAI router",
        )
            .into_response()
    }

    async fn get_worker_loads(&self) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "get_worker_loads not supported for OpenAI router",
        )
            .into_response()
    }

    fn router_type(&self) -> &'static str {
        "openai"
    }

    fn readiness(&self) -> Response {
        if self.healthy.load(Ordering::Acquire) && self.circuit_breaker.can_execute() {
            (StatusCode::OK, "Ready").into_response()
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "Not ready").into_response()
        }
    }

    async fn route_embeddings(&self, _headers: Option<&HeaderMap>, _body: Body) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Embeddings endpoint not implemented for OpenAI backend",
        )
            .into_response()
    }

    async fn route_rerank(&self, _headers: Option<&HeaderMap>, _body: Body) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Rerank endpoint not implemented for OpenAI backend",
        )
            .into_response()
    }
}
