use std::{sync::Arc, time::Instant};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{
        header::{CONTENT_LENGTH, CONTENT_TYPE},
        HeaderMap, HeaderValue, Method, StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use futures_util::StreamExt;
use memchr::memmem;
use reqwest::Client;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error};

use crate::{
    config::types::RetryConfig,
    core::{
        is_retryable_status, ConnectionMode, RetryExecutor, Worker, WorkerRegistry, WorkerType,
    },
    observability::{
        events::{self, Event},
        metrics::RouterMetrics,
        otel_trace::inject_trace_context_http,
    },
    policies::PolicyRegistry,
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
    routers::{header_utils, RouterTrait},
};

/// Regular router that uses injected load balancing policies
#[derive(Debug)]
pub struct Router {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: Client,
    dp_aware: bool,
    enable_igw: bool,
    retry_config: RetryConfig,
}

impl Router {
    /// Create a new router with injected policy and client
    pub async fn new(ctx: &Arc<crate::app_context::AppContext>) -> Result<Self, String> {
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

    // Helper method to proxy GET requests to the first available worker
    async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
        let headers = header_utils::copy_request_headers(&req);

        match self.select_first_worker() {
            Ok(worker_url) => {
                let mut request_builder = self.client.get(format!("{}/{}", worker_url, endpoint));
                for (name, value) in headers {
                    // Use eq_ignore_ascii_case to avoid string allocation
                    if !name.eq_ignore_ascii_case("content-type")
                        && !name.eq_ignore_ascii_case("content-length")
                    {
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
                            Err(e) => (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!("Failed to read response: {}", e),
                            )
                                .into_response(),
                        }
                    }
                    Err(e) => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Request failed: {}", e),
                    )
                        .into_response(),
                }
            }
            Err(e) => (StatusCode::SERVICE_UNAVAILABLE, e).into_response(),
        }
    }

    /// Select worker for a specific model considering circuit breaker state
    fn select_worker_for_model(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
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

        let idx = policy.select_worker(&available, text)?;
        Some(available[idx].clone())
    }

    pub async fn route_typed_request<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &str,
        model_id: Option<&str>,
    ) -> Response {
        let start = Instant::now();
        let is_stream = typed_req.is_stream();
        let text = typed_req.extract_text_for_routing();

        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // operation per attempt
            |_: u32| async {
                let worker = match self.select_worker_for_model(model_id, Some(&text)) {
                    Some(w) => w,
                    None => {
                        RouterMetrics::record_request_error(route, "no_available_workers");
                        return (
                            StatusCode::SERVICE_UNAVAILABLE,
                            "No available workers (all circuits open or unhealthy)",
                        )
                            .into_response();
                    }
                };

                // Optional load tracking for cache-aware policy
                // Get the policy for this model to check if it's cache-aware
                let policy = match model_id {
                    Some(model) => self.policy_registry.get_policy_or_default(model),
                    None => self.policy_registry.get_default_policy(),
                };

                let load_incremented = if policy.name() == "cache_aware" {
                    increment_load(&worker);
                    true
                } else {
                    false
                };

                // Keep a clone for potential cleanup on retry
                let worker_for_cleanup = if load_incremented {
                    Some(worker.clone())
                } else {
                    None
                };

                events::RequestSentEvent {
                    url: worker.url().to_string(),
                }
                .emit();
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
                        load_incremented,
                    )
                    .await;

                events::RequestReceivedEvent {}.emit();

                worker.record_outcome(response.status().is_success());

                // For retryable failures, we need to decrement load since send_typed_request
                // won't have done it (it only decrements on success or non-retryable failures)
                if is_retryable_status(response.status()) && load_incremented {
                    if let Some(cleanup_worker) = worker_for_cleanup {
                        decrement_load(&cleanup_worker);
                    }
                }

                response
            },
            // should_retry predicate
            |res, _attempt| is_retryable_status(res.status()),
            // on_backoff hook
            |delay, attempt| {
                RouterMetrics::record_retry(route);
                RouterMetrics::record_retry_backoff_duration(delay, attempt);
            },
            // on_exhausted hook
            || RouterMetrics::record_retries_exhausted(route),
        )
        .await;

        if response.status().is_success() {
            let duration = start.elapsed();
            RouterMetrics::record_request(route);
            RouterMetrics::record_generate_duration(duration);
        } else if !is_retryable_status(response.status()) {
            RouterMetrics::record_request_error(route, "non_retryable_error");
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
            return (StatusCode::SERVICE_UNAVAILABLE, "No available workers").into_response();
        }

        // Pre-filter headers once before the loop to avoid repeated lowercasing
        let filtered_headers: Vec<_> = headers
            .map(|hdrs| {
                hdrs.iter()
                    .filter(|(name, _)| {
                        !name.as_str().eq_ignore_ascii_case("content-type")
                            && !name.as_str().eq_ignore_ascii_case("content-length")
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut last_response: Option<Response> = None;
        for worker in workers {
            let worker_url = worker.url();
            let base = self.worker_base_url(worker_url);

            let url = format!("{}/{}", base, endpoint);
            let mut request_builder = match method {
                Method::GET => self.client.get(url),
                Method::POST => self.client.post(url),
                _ => {
                    return (
                        StatusCode::METHOD_NOT_ALLOWED,
                        "Unsupported method for simple routing",
                    )
                        .into_response()
                }
            };

            if let Some(api_key) = worker.api_key() {
                // Pre-allocate string with capacity to avoid reallocation
                let mut auth_header = String::with_capacity(7 + api_key.len());
                auth_header.push_str("Bearer ");
                auth_header.push_str(api_key);
                request_builder = request_builder.header("Authorization", auth_header);
            }

            // Apply pre-filtered headers
            for (name, value) in &filtered_headers {
                request_builder = request_builder.header(*name, *value);
            }

            match request_builder.send().await {
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
                            last_response = Some(
                                (
                                    StatusCode::INTERNAL_SERVER_ERROR,
                                    format!("Failed to read response: {}", e),
                                )
                                    .into_response(),
                            );
                        }
                    }
                }
                Err(e) => {
                    last_response = Some(
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Request failed: {}", e),
                        )
                            .into_response(),
                    );
                }
            }
        }

        last_response
            .unwrap_or_else(|| (StatusCode::BAD_GATEWAY, "No worker response").into_response())
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
        route: &str,
        worker_url: &str,
        is_stream: bool,
        load_incremented: bool, // Whether load was incremented for this request
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
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to extract dp_rank: {}", e),
                    )
                        .into_response();
                }
            };

            let mut json_val = match serde_json::to_value(typed_req) {
                Ok(j) => j,
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        format!("Convert into serde_json::Value failed: {}", e),
                    )
                        .into_response();
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
                return (
                    StatusCode::BAD_REQUEST,
                    "Failed to insert the data_parallel_rank field into the request body",
                )
                    .into_response();
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

        // Copy all headers from original request if provided
        if let Some(headers) = headers {
            for (name, value) in headers {
                // Skip Content-Type and Content-Length as .json() sets them
                if *name != CONTENT_TYPE && *name != CONTENT_LENGTH {
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

                // Decrement load on error if it was incremented
                if load_incremented {
                    if let Some(ref w) = worker {
                        decrement_load(w);
                    }
                }

                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Request failed: {}", e),
                )
                    .into_response();
            }
        };

        RouterMetrics::record_upstream_http_response(route, res.status().as_u16());

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
                    // IMPORTANT: Decrement load on error before returning
                    if load_incremented {
                        if let Some(ref w) = worker {
                            decrement_load(w);
                        }
                    }

                    let error_msg = format!("Failed to get response body: {}", e);
                    (StatusCode::INTERNAL_SERVER_ERROR, error_msg).into_response()
                }
            };

            // Decrement load counter for non-streaming requests if it was incremented
            if load_incremented {
                if let Some(ref w) = worker {
                    decrement_load(w);
                }
            }

            response
        } else if load_incremented {
            // For streaming with load tracking, we need to manually decrement when done
            // Clone the worker Arc for the async block instead of looking it up again
            let stream_worker = worker.clone();

            // Preserve headers for streaming response
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            // Ensure we set the correct content-type for SSE
            response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            let stream = res.bytes_stream();
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

            // Spawn task to forward stream and detect completion
            tokio::spawn(async move {
                let mut stream = stream;
                let mut decremented = false;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            // Check for stream end marker using memmem for efficiency
                            if memmem::find(&bytes, b"data: [DONE]").is_some() {
                                if let Some(ref w) = stream_worker {
                                    decrement_load(w);
                                    decremented = true;
                                }
                            }
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
                if !decremented {
                    if let Some(ref w) = stream_worker {
                        decrement_load(w);
                    }
                }
            });

            let stream = UnboundedReceiverStream::new(rx);
            let body = Body::from_stream(stream);

            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = response_headers;
            response
        } else {
            // For requests without load tracking, just stream
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

fn increment_load(w: &Arc<dyn Worker>) {
    w.increment_load();
    RouterMetrics::set_running_requests(w.url(), w.load());
}

fn decrement_load(w: &Arc<dyn Worker>) {
    w.decrement_load();
    RouterMetrics::set_running_requests(w.url(), w.load());
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
        // Record embeddings-specific metrics in addition to general request metrics
        let start = Instant::now();
        let res = self
            .route_typed_request(headers, body, "/v1/embeddings", model_id)
            .await;

        // Embedding specific metrics
        if res.status().is_success() {
            RouterMetrics::record_embeddings_request();
            RouterMetrics::record_embeddings_duration(start.elapsed());
        } else {
            let error_type = format!("http_{}", res.status().as_u16());
            RouterMetrics::record_embeddings_error(&error_type);
        }

        res
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Record classification-specific metrics in addition to general request metrics
        let start = Instant::now();
        let res = self
            .route_typed_request(headers, body, "/v1/classify", model_id)
            .await;

        // Classification specific metrics
        if res.status().is_success() {
            RouterMetrics::record_classify_request();
            RouterMetrics::record_classify_duration(start.elapsed());
        } else {
            let error_type = format!("http_{}", res.status().as_u16());
            RouterMetrics::record_classify_error(&error_type);
        }

        res
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
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to build rerank response".to_string(),
                    )
                        .into_response();
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
