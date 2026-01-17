use std::{sync::Arc, time::Instant};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::{stream, StreamExt};
use reqwest::Client;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error};

use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    core::{
        is_retryable_status, AttachedBody, ConnectionMode, RetryExecutor, Worker, WorkerLoadGuard,
        WorkerRegistry, WorkerType, UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, Metrics},
        otel_trace::inject_trace_context_http,
    },
    policies::{PolicyRegistry, SelectWorkerInfo},
    protocols::{
        chat::ChatCompletionRequest,
        classify::ClassifyRequest,
        common::GenerationRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::{RerankRequest, RerankResponse, RerankResult},
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::{
        error::{self, extract_error_code_from_response},
        grpc::utils::{error_type_from_status, route_to_endpoint},
        header_utils, RouterTrait,
    },
};

/// Regular router that uses injected load balancing policies
pub struct Router {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: Client,
    dp_aware: bool,
    enable_igw: bool,
    retry_config: RetryConfig,
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Router")
            .field("worker_registry", &self.worker_registry)
            .field("policy_registry", &self.policy_registry)
            .field("client", &self.client)
            .field("dp_aware", &self.dp_aware)
            .field("enable_igw", &self.enable_igw)
            .field("retry_config", &self.retry_config)
            .finish()
    }
}

impl Router {
    /// Create a new router with injected policy and client
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        Ok(Router {
            worker_registry: ctx.worker_registry.clone(),
            policy_registry: ctx.policy_registry.clone(),
            client: ctx.client.clone(),
            dp_aware: ctx.router_config.dp_aware,
            enable_igw: ctx.router_config.enable_igw,
            retry_config: ctx.router_config.effective_retry_config(),
        })
    }

    fn select_first_worker(&self) -> Result<String, String> {
        let workers = self.worker_registry.get_all();
        let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();
        if healthy_workers.is_empty() {
            Err("No workers are available".to_string())
        } else {
            Ok(healthy_workers[0].url().to_string())
        }
    }

    async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
        let headers = header_utils::copy_request_headers(&req);

        match self.select_first_worker() {
            Ok(worker_url) => {
                let mut request_builder = self.client.get(format!("{}/{}", worker_url, endpoint));
                for (name, value) in headers {
                    if header_utils::should_forward_request_header(&name) {
                        request_builder = request_builder.header(name, value);
                    }
                }

                match request_builder.send().await {
                    Ok(res) => {
                        let status = StatusCode::from_u16(res.status().as_u16())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                        // Preserve headers from backend
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(body) => {
                                let mut response = Response::new(Body::from(body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {}", e),
                            ),
                        }
                    }
                    Err(e) => convert_reqwest_error(e),
                }
            }
            Err(e) => error::service_unavailable("no_workers", e),
        }
    }

    /// Select worker for a specific model considering circuit breaker state
    async fn select_worker_for_model(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Option<Arc<dyn Worker>> {
        let effective_model_id = if !self.enable_igw { None } else { model_id };

        // Get workers for the specified model O(1), filtered by connection mode
        let workers = self.worker_registry.get_workers_filtered(
            effective_model_id,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Http),
            None,  // any runtime type
            false, // get all workers, we'll filter by is_available() next
        );

        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();
        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self
            .worker_registry
            .get_hash_ring(effective_model_id.unwrap_or(UNKNOWN_MODEL_ID));

        let idx = policy
            .select_worker(
                &available,
                &SelectWorkerInfo {
                    request_text: text,
                    tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                    headers,
                    hash_ring,
                },
            )
            .await?;

        // Record worker selection metric (Layer 3)
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model_id.unwrap_or(UNKNOWN_MODEL_ID),
            policy.name(),
        );

        Some(available[idx].clone())
    }

    pub async fn route_typed_request<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: Option<&str>,
    ) -> Response {
        let start = Instant::now();
        let is_stream = typed_req.is_stream();
        let text = typed_req.extract_text_for_routing();
        let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(is_stream),
        );

        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // operation per attempt
            |_: u32| async {
                let res = self
                    .route_typed_request_once(headers, typed_req, route, model_id, is_stream, &text)
                    .await;

                // Need to be outside `route_typed_request_once` because that function has multiple return paths
                Metrics::record_router_upstream_response(
                    metrics_labels::ROUTER_HTTP,
                    res.status().as_u16(),
                    extract_error_code_from_response(&res),
                );

                res
            },
            // should_retry predicate
            |res, _attempt| is_retryable_status(res.status()),
            // on_backoff hook
            |delay, attempt| {
                // Layer 3 worker metrics
                Metrics::record_worker_retry(metrics_labels::WORKER_REGULAR, endpoint);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            // on_exhausted hook
            || {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_REGULAR, endpoint);
            },
        )
        .await;

        if response.status().is_success() {
            let duration = start.elapsed();
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn route_typed_request_once<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: Option<&str>,
        is_stream: bool,
        text: &str,
    ) -> Response {
        let worker = match self
            .select_worker_for_model(model_id, Some(text), headers)
            .await
        {
            Some(w) => w,
            None => {
                return error::service_unavailable(
                    "no_available_workers",
                    "No available workers (all circuits open or unhealthy)",
                );
            }
        };

        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        let load_guard = ["cache_aware", "manual"]
            .contains(&policy.name())
            .then(|| WorkerLoadGuard::new(worker.clone(), headers));

        // Note: Using borrowed reference avoids heap allocation
        events::RequestSentEvent { url: worker.url() }.emit();
        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        let response = self
            .send_typed_request(
                headers,
                typed_req,
                route,
                worker.url(),
                is_stream,
                load_guard,
            )
            .await;

        events::RequestReceivedEvent {}.emit();

        let status = response.status();
        worker.record_outcome(status.is_success());

        // Record worker errors for server errors (5xx)
        if status.is_server_error() {
            Metrics::record_worker_error(
                metrics_labels::WORKER_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                error_type_from_status(status),
            );
        }

        response
    }

    // Helper: return base worker URL (strips DP suffix when enabled)
    fn worker_base_url(&self, worker_url: &str) -> String {
        if self.dp_aware {
            if let Ok((prefix, _)) = Self::extract_dp_rank(worker_url) {
                return prefix.to_string();
            }
        }
        worker_url.to_string()
    }

    // Generic simple routing for GET/POST without JSON body
    async fn route_simple_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
        method: Method,
    ) -> Response {
        // TODO: currently the sglang worker is using in-memory state management, so this implementation has to fan out to all workers.
        // Eventually, we need to have router to manage the chat history with a proper database, will update this implementation accordingly.
        let workers = self.worker_registry.get_all();
        if workers.is_empty() {
            return error::service_unavailable("no_workers", "No available workers");
        }

        let filtered_headers: Vec<_> = headers
            .map(|hdrs| {
                hdrs.iter()
                    .filter(|(name, _)| header_utils::should_forward_request_header(name.as_str()))
                    .collect()
            })
            .unwrap_or_default();

        let futures: Vec<_> = workers
            .into_iter()
            .map(|worker| {
                let worker_url = worker.url();
                let base = self.worker_base_url(worker_url);
                let url = format!("{}/{}", base, endpoint);
                let client = self.client.clone();
                let method = method.clone();

                let headers = filtered_headers.clone();

                let api_key = worker.api_key().clone();

                async move {
                    let mut request_builder = match method {
                        Method::GET => client.get(url),
                        Method::POST => client.post(url),
                        _ => {
                            return Err(error::method_not_allowed(
                                "unsupported_method",
                                "Unsupported method for simple routing",
                            ))
                        }
                    };

                    if let Some(key) = api_key {
                        let mut auth_header = String::with_capacity(7 + key.len());
                        auth_header.push_str("Bearer ");
                        auth_header.push_str(&key);
                        request_builder = request_builder.header("Authorization", auth_header);
                    }

                    for (name, value) in headers {
                        request_builder = request_builder.header(name.clone(), value.clone());
                    }

                    request_builder.send().await.map_err(convert_reqwest_error)
                }
            })
            .collect();

        // Now execute the collected futures concurrently
        let mut stream = stream::iter(futures).buffer_unordered(32);
        let mut last_response: Option<Response> = None;

        while let Some(result) = stream.next().await {
            match result {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                    let response_headers = header_utils::preserve_response_headers(res.headers());

                    match res.bytes().await {
                        Ok(body) => {
                            let mut response = Response::new(Body::from(body));
                            *response.status_mut() = status;
                            *response.headers_mut() = response_headers;

                            if status.is_success() {
                                return response;
                            }
                            last_response = Some(response);
                        }
                        Err(e) => {
                            last_response = Some(error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {}", e),
                            ));
                        }
                    }
                }
                Err(e) => {
                    last_response = Some(e);
                }
            }
        }

        last_response
            .unwrap_or_else(|| error::bad_gateway("no_worker_response", "No worker response"))
    }

    // Route a GET request with provided headers to a specific endpoint
    async fn route_get_request(&self, headers: Option<&HeaderMap>, endpoint: &str) -> Response {
        self.route_simple_request(headers, endpoint, Method::GET)
            .await
    }

    // Route a POST request with empty body to a specific endpoint
    async fn route_post_empty_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
    ) -> Response {
        self.route_simple_request(headers, endpoint, Method::POST)
            .await
    }

    // TODO (rui): Better accommodate to the Worker abstraction
    fn extract_dp_rank(worker_url: &str) -> Result<(&str, usize), String> {
        let parts: Vec<&str> = worker_url.split('@').collect();
        if parts.len() != 2 {
            return Err(format!("invalid worker_url format: {}", worker_url));
        }

        // Parse the second part (dp_rank) into an integer
        match parts[1].parse::<usize>() {
            Ok(dp_rank) => Ok((parts[0], dp_rank)),
            Err(_) => Err(format!(
                "failed to parse dp_rank from worker_url: {}",
                worker_url
            )),
        }
    }

    // Send typed request directly without conversion
    async fn send_typed_request<T: serde::Serialize>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        worker_url: &str,
        is_stream: bool,
        load_guard: Option<WorkerLoadGuard>,
    ) -> Response {
        // Get the worker once and reuse for API key and load tracking
        let worker = self.worker_registry.get_by_url(worker_url);
        let api_key = worker.as_ref().and_then(|w| w.api_key().clone());

        // Static key string to avoid per-request allocations
        const DP_RANK_KEY: &str = "data_parallel_rank";

        let mut request_builder = if self.dp_aware {
            let (worker_url_prefix, dp_rank) = match Self::extract_dp_rank(worker_url) {
                Ok(tup) => tup,
                Err(e) => {
                    error!("Failed to extract dp_rank: {}", e);
                    return error::internal_error(
                        "dp_rank_extraction_failed",
                        format!("Failed to extract dp_rank: {}", e),
                    );
                }
            };

            let mut json_val = match serde_json::to_value(typed_req) {
                Ok(j) => j,
                Err(e) => {
                    return error::bad_request(
                        "serialization_failed",
                        format!("Convert into serde_json::Value failed: {}", e),
                    );
                }
            };

            if let Some(map) = json_val.as_object_mut() {
                // Use static key string to avoid allocation
                map.insert(DP_RANK_KEY.to_string(), serde_json::json!(dp_rank));
                // Only serialize if debug logging is enabled to avoid CPU overhead
                if tracing::enabled!(tracing::Level::DEBUG) {
                    debug!(
                        "Modified request body: {}",
                        serde_json::to_string(&json_val).unwrap_or_else(|_| String::from("ERR"))
                    );
                }
            } else {
                return error::bad_request(
                    "dp_rank_insertion_failed",
                    "Failed to insert the data_parallel_rank field into the request body",
                );
            }

            self.client
                .post(format!("{}{}", worker_url_prefix, route))
                .json(&json_val)
        } else {
            self.client
                .post(format!("{}{}", worker_url, route))
                .json(typed_req) // Use json() directly with typed request
        };

        if let Some(key) = api_key {
            // Pre-allocate string with capacity to avoid reallocation
            let mut auth_header = String::with_capacity(7 + key.len());
            auth_header.push_str("Bearer ");
            auth_header.push_str(&key);
            request_builder = request_builder.header("Authorization", auth_header);
        }

        if let Some(headers) = headers {
            for (name, value) in headers {
                if header_utils::should_forward_request_header(name.as_str()) {
                    request_builder = request_builder.header(name, value);
                }
            }
        }

        let res = match request_builder.send().await {
            Ok(res) => res,
            Err(e) => {
                error!(
                    "Failed to send typed request worker_url={} route={} error={}",
                    worker_url, route, e
                );

                return convert_reqwest_error(e);
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // For non-streaming requests, preserve headers
            let response_headers = header_utils::preserve_response_headers(res.headers());

            let response = match res.bytes().await {
                Ok(body) => {
                    let mut response = Response::new(Body::from(body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => {
                    let error_msg = format!("Failed to get response body: {}", e);
                    error::internal_error("read_response_body_failed", error_msg)
                }
            };

            // load_guard dropped here automatically after response body is read
            response
        } else {
            // Preserve headers for streaming response
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            // Ensure we set the correct content-type for SSE
            response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            let stream = res.bytes_stream();
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

            // Spawn task to forward stream
            tokio::spawn(async move {
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
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

            let stream = UnboundedReceiverStream::new(rx);
            let body = Body::from_stream(stream);

            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = response_headers;

            // Attach load guard to response body for proper RAII lifecycle
            // Guard is dropped when response body is consumed or client disconnects
            if let Some(guard) = load_guard {
                response = AttachedBody::wrap_response(response, guard);
            }
            response
        }
    }

    async fn build_rerank_response(
        req: &RerankRequest,
        response: Response,
    ) -> anyhow::Result<Response> {
        let (_, response_body) = response.into_parts();
        let body_bytes = to_bytes(response_body, usize::MAX).await?;
        let rerank_results = serde_json::from_slice::<Vec<RerankResult>>(&body_bytes)?;
        let mut rerank_response =
            RerankResponse::new(rerank_results, req.model.clone(), req.rid.clone());
        // Sorting is handled by Python worker (serving_rerank.py)
        if let Some(top_k) = req.top_k {
            rerank_response.apply_top_k(top_k);
        }
        if !req.return_documents {
            rerank_response.drop_documents();
        }
        Ok(Json(rerank_response).into_response())
    }
}

fn convert_reqwest_error(e: reqwest::Error) -> Response {
    let url = e
        .url()
        .map(|u| u.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let message = format!("{}. URL: {}", e, url);

    // TODO improve error status code
    let (status, code) = if let Some(upstream_status) = e.status() {
        (upstream_status, "call_upstream_status_error")
    } else if e.is_builder() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_builder_error",
        )
    } else if e.is_request() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_error",
        )
    } else if e.is_redirect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_redirect_error",
        )
    } else if e.is_body() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_body_error",
        )
    } else if e.is_decode() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_decode_error",
        )
    } else if e.is_timeout() {
        (StatusCode::GATEWAY_TIMEOUT, "call_upstream_timeout")
    } else if e.is_connect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_connection_failed",
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_failed",
        )
    };

    error::create_error(status, code, message)
}

use async_trait::async_trait;

#[async_trait]
impl RouterTrait for Router {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "health_generate").await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_server_info").await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "v1/models").await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_model_info").await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/generate", model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/chat/completions", model_id)
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/completions", model_id)
            .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/responses", model_id)
            .await
    }

    async fn get_response(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        let endpoint = format!("v1/responses/{}", response_id);
        self.route_get_request(headers, &endpoint).await
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{}/cancel", response_id);
        self.route_post_empty_request(headers, &endpoint).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/embeddings", model_id)
            .await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/classify", model_id)
            .await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        let response = self
            .route_typed_request(headers, body, "/v1/rerank", model_id)
            .await;
        if response.status().is_success() {
            match Self::build_rerank_response(body, response).await {
                Ok(rerank_response) => rerank_response,
                Err(e) => {
                    error!("Failed to build rerank response: {}", e);
                    return error::internal_error(
                        "rerank_response_build_failed",
                        "Failed to build rerank response",
                    );
                }
            }
        } else {
            response
        }
    }

    fn router_type(&self) -> &'static str {
        "regular"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorkerBuilder;

    fn create_test_regular_router() -> Router {
        // Create registries
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::RoundRobin,
        ));

        // Register test workers
        let worker1 = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://worker2:8080")
            .worker_type(WorkerType::Regular)
            .build();
        worker_registry.register(Arc::new(worker1));
        worker_registry.register(Arc::new(worker2));

        Router {
            worker_registry,
            policy_registry,
            dp_aware: false,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            enable_igw: false,
        }
    }

    fn create_test_unhealthy_router() -> Router {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        workers[0].set_healthy(false);
        router
    }

    #[test]
    fn test_router_get_worker_urls_regular() {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        let urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));
    }

    #[test]
    fn test_select_first_worker_regular() {
        let router = create_test_regular_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();
        // DashMap doesn't guarantee order, so just check we get one of the workers
        assert!(url == "http://worker1:8080" || url == "http://worker2:8080");
    }

    #[test]
    fn test_select_first_worker_with_unhealthy_worker() {
        let router = create_test_unhealthy_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();

        let worker = router.worker_registry.get_by_url(&url).unwrap();
        assert!(worker.is_healthy());
    }
}
