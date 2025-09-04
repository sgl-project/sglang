use crate::config::types::RetryConfig;
use crate::core::{
    is_retryable_status, BasicWorker, CircuitBreakerConfig, HealthChecker, HealthConfig,
    RetryExecutor, Worker, WorkerFactory, WorkerType,
};
use crate::metrics::RouterMetrics;
use crate::policies::LoadBalancingPolicy;
use crate::protocols::spec::{
    ChatCompletionRequest, CompletionRequest, GenerateRequest, GenerationRequest,
};
use crate::routers::header_utils;
use crate::routers::{RouterTrait, WorkerManagement};
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_LENGTH, header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::StreamExt;
use reqwest::Client;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn};

/// Regular router that uses injected load balancing policies
#[derive(Debug)]
pub struct Router {
    workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
    policy: Arc<dyn LoadBalancingPolicy>,
    client: Client,
    worker_startup_timeout_secs: u64,
    worker_startup_check_interval_secs: u64,
    dp_aware: bool,
    api_key: Option<String>,
    retry_config: RetryConfig,
    circuit_breaker_config: CircuitBreakerConfig,
    _worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, isize>>>,
    _load_monitor_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    _health_checker: Option<HealthChecker>,
}

impl Router {
    /// Create a new router with injected policy and client
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        worker_urls: Vec<String>,
        policy: Arc<dyn LoadBalancingPolicy>,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        // Update active workers gauge
        RouterMetrics::set_active_workers(worker_urls.len());

        // Wait for workers to be healthy (skip if empty - for service discovery mode)
        if !worker_urls.is_empty() {
            Self::wait_for_healthy_workers(
                &worker_urls,
                ctx.router_config.worker_startup_timeout_secs,
                ctx.router_config.worker_startup_check_interval_secs,
            )
            .await?;
        }

        let worker_urls = if ctx.router_config.dp_aware {
            // worker address now in the format of "http://host:port@dp_rank"
            Self::get_dp_aware_workers(&worker_urls, &ctx.router_config.api_key)
                .map_err(|e| format!("Failed to get dp-aware workers: {}", e))?
        } else {
            worker_urls
        };

        // Convert config CircuitBreakerConfig to core CircuitBreakerConfig
        let circuit_breaker_config = ctx.router_config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Create Worker trait objects from URLs with health check config
        let workers: Vec<Box<dyn Worker>> = worker_urls
            .iter()
            .map(|url| {
                let worker = BasicWorker::new(url.clone(), WorkerType::Regular)
                    .with_circuit_breaker_config(core_cb_config.clone())
                    .with_health_config(HealthConfig {
                        timeout_secs: ctx.router_config.health_check.timeout_secs,
                        check_interval_secs: ctx.router_config.health_check.check_interval_secs,
                        endpoint: ctx.router_config.health_check.endpoint.clone(),
                        failure_threshold: ctx.router_config.health_check.failure_threshold,
                        success_threshold: ctx.router_config.health_check.success_threshold,
                    });
                Box::new(worker) as Box<dyn Worker>
            })
            .collect();

        // Initialize policy with workers if needed (e.g., for cache-aware)
        if let Some(cache_aware) = policy
            .as_any()
            .downcast_ref::<crate::policies::CacheAwarePolicy>()
        {
            cache_aware.init_workers(&workers);
        }

        let workers = Arc::new(RwLock::new(workers));
        let health_checker = crate::core::start_health_checker(
            Arc::clone(&workers),
            ctx.router_config.worker_startup_check_interval_secs,
        );

        // Setup load monitoring for PowerOfTwo policy
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let worker_loads = Arc::new(rx);

        let load_monitor_handle = if policy.name() == "power_of_two" {
            let monitor_urls = worker_urls.clone();
            let monitor_interval = ctx.router_config.worker_startup_check_interval_secs;
            let policy_clone = Arc::clone(&policy);
            let client_clone = ctx.client.clone();

            Some(Arc::new(tokio::spawn(async move {
                Self::monitor_worker_loads(
                    monitor_urls,
                    tx,
                    monitor_interval,
                    policy_clone,
                    client_clone,
                )
                .await;
            })))
        } else {
            None
        };

        Ok(Router {
            workers,
            policy,
            client: ctx.client.clone(),
            worker_startup_timeout_secs: ctx.router_config.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: ctx
                .router_config
                .worker_startup_check_interval_secs,
            dp_aware: ctx.router_config.dp_aware,
            api_key: ctx.router_config.api_key.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
            circuit_breaker_config: core_cb_config,
            _worker_loads: worker_loads,
            _load_monitor_handle: load_monitor_handle,
            _health_checker: Some(health_checker),
        })
    }

    /// Get the current list of worker URLs
    pub fn get_worker_urls(&self) -> Vec<String> {
        self.workers
            .read()
            .unwrap()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    pub async fn wait_for_healthy_workers(
        worker_urls: &[String],
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval_secs: u64,
    ) -> Result<(), String> {
        if worker_urls.is_empty() {
            return Err(
                "Timeout waiting for workers to become healthy: no workers provided".to_string(),
            );
        }

        // Perform health check asynchronously
        Self::wait_for_healthy_workers_async(
            worker_urls,
            worker_startup_timeout_secs,
            worker_startup_check_interval_secs,
        )
        .await
    }

    async fn wait_for_healthy_workers_async(
        worker_urls: &[String],
        worker_startup_timeout_secs: u64,
        worker_startup_check_interval_secs: u64,
    ) -> Result<(), String> {
        info!(
            "Waiting for {} workers to become healthy (timeout: {}s)",
            worker_urls.len(),
            worker_startup_timeout_secs
        );

        let start_time = std::time::Instant::now();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        loop {
            if start_time.elapsed() > Duration::from_secs(worker_startup_timeout_secs) {
                error!(
                    "Timeout {}s waiting for workers {:?} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    worker_startup_timeout_secs, worker_urls
                );
                return Err(format!(
                    "Timeout {}s waiting for workers {:?} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    worker_startup_timeout_secs, worker_urls
                ));
            }

            // Perform all health checks concurrently
            let mut health_checks = Vec::new();
            for url in worker_urls {
                let client_clone = client.clone();
                let url_clone = url.clone();

                let check_health = tokio::spawn(async move {
                    let health_url = format!("{}/health", url_clone);
                    match client_clone.get(&health_url).send().await {
                        Ok(res) => {
                            if res.status().is_success() {
                                None
                            } else {
                                Some((url_clone, format!("status: {}", res.status())))
                            }
                        }
                        Err(_) => Some((url_clone, "not ready".to_string())),
                    }
                });

                health_checks.push(check_health);
            }

            // Wait for all health checks to complete
            let results = futures::future::join_all(health_checks).await;

            let mut all_healthy = true;
            let mut unhealthy_workers = Vec::new();

            for result in results {
                match result {
                    Ok(None) => {
                        // Worker is healthy
                    }
                    Ok(Some((url, reason))) => {
                        all_healthy = false;
                        unhealthy_workers.push((url, reason));
                    }
                    Err(e) => {
                        all_healthy = false;
                        unhealthy_workers
                            .push(("unknown".to_string(), format!("task error: {}", e)));
                    }
                }
            }

            if all_healthy {
                info!("All {} workers are healthy", worker_urls.len());
                return Ok(());
            } else {
                debug!(
                    "Waiting for {} workers to become healthy ({} unhealthy: {:?})",
                    worker_urls.len(),
                    unhealthy_workers.len(),
                    unhealthy_workers
                );
                tokio::time::sleep(Duration::from_secs(worker_startup_check_interval_secs)).await;
            }
        }
    }

    fn get_worker_dp_size(worker_url: &str, api_key: &Option<String>) -> Result<usize, String> {
        let sync_client = reqwest::blocking::Client::new();
        let mut req_builder = sync_client.get(format!("{}/get_server_info", worker_url));
        if let Some(key) = api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        match req_builder.send() {
            Ok(res) => {
                if res.status().is_success() {
                    let server_info = res
                        .text()
                        .map_err(|e| format!("failed to read text from response: {}", e))?;

                    let server_info: serde_json::Value = serde_json::from_str(&server_info)
                        .map_err(|e| format!("failed to decode JSON: {}", e))?;

                    let dp_size = server_info
                        .get("dp_size")
                        .and_then(|v| v.as_u64())
                        .ok_or_else(|| String::from("dp_size not found or not an u64"))?;

                    Ok(if dp_size > usize::MAX as u64 {
                        return Err(format!("dp_size is too large: {}", dp_size));
                    } else {
                        dp_size as usize
                    })
                } else {
                    Err(format!("unexpected status code: {}", res.status()))
                }
            }
            Err(e) => Err(format!("error response: {}", e)),
        }
    }

    // Given a list of workers, return a list of workers with dp_rank as suffix
    fn get_dp_aware_workers(
        worker_urls: &[String],
        api_key: &Option<String>,
    ) -> Result<Vec<String>, String> {
        let mut dp_aware_workers: Vec<String> = Vec::new();

        for url in worker_urls {
            match Self::get_worker_dp_size(url, api_key) {
                Ok(dp_size) => {
                    for i in 0..dp_size {
                        dp_aware_workers.push(format!("{}@{}", url, i));
                    }
                }
                Err(e) => return Err(format!("Failed to get DP size for {}: {}", url, e)),
            }
        }

        Ok(dp_aware_workers)
    }

    fn select_first_worker(&self) -> Result<String, String> {
        let workers_guard = self.workers.read().unwrap();
        if workers_guard.is_empty() {
            Err("No workers are available".to_string())
        } else {
            Ok(workers_guard[0].url().to_string())
        }
    }

    pub async fn send_health_check(&self, worker_url: &str) -> Response {
        let health_url = if self.dp_aware {
            // Need to extract the URL from "http://host:port@dp_rank"
            match Self::extract_dp_rank(worker_url) {
                Ok((worker_url_prefix, _dp_rank)) => worker_url_prefix,
                Err(e) => {
                    error!("Failed to extract dp_rank for health check: {}", e);
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to extract dp_rank: {}", e),
                    )
                        .into_response();
                }
            }
        } else {
            worker_url
        };

        let request_builder = self.client.get(format!("{}/health", health_url));

        let response = match request_builder.send().await {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                match res.bytes().await {
                    Ok(body) => (status, body).into_response(),
                    Err(e) => {
                        error!(
                            worker_url = %health_url,
                            error = %e,
                            "Failed to read health response body"
                        );
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read response body: {}", e),
                        )
                            .into_response()
                    }
                }
            }
            Err(e) => {
                error!(
                    worker_url = %health_url,
                    error = %e,
                    "Failed to send health request to worker"
                );
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to send request to worker {}: {}", health_url, e),
                )
                    .into_response()
            }
        };

        // Don't record metrics for health checks
        response
    }

    // Helper method to proxy GET requests to the first available worker
    async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
        let headers = header_utils::copy_request_headers(&req);

        match self.select_first_worker() {
            Ok(worker_url) => {
                let mut request_builder = self.client.get(format!("{}/{}", worker_url, endpoint));
                for (name, value) in headers {
                    let name_lc = name.to_lowercase();
                    if name_lc != "content-type" && name_lc != "content-length" {
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
                                let mut response = Response::new(axum::body::Body::from(body));
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

    // New method to route typed requests directly
    /// Select worker considering circuit breaker state
    fn select_worker_with_circuit_breaker(&self, text: Option<&str>) -> Option<Box<dyn Worker>> {
        let workers = self.workers.read().ok()?;
        let available: Vec<Box<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .map(|w| w.clone_worker())
            .collect();
        if available.is_empty() {
            return None;
        }
        let idx = self.policy.select_worker(&available, text)?;
        Some(available[idx].clone_worker())
    }

    pub async fn route_typed_request<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &str,
    ) -> Response {
        let start = Instant::now();
        let is_stream = typed_req.is_stream();
        let text = typed_req.extract_text_for_routing();

        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // operation per attempt
            |_: u32| async {
                let worker = match self.select_worker_with_circuit_breaker(Some(&text)) {
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
                let load_incremented = if self.policy.name() == "cache_aware" {
                    worker.increment_load();
                    RouterMetrics::set_running_requests(worker.url(), worker.load());
                    true
                } else {
                    false
                };

                // Keep a clone for potential cleanup on retry
                let worker_for_cleanup = if load_incremented {
                    Some(worker.clone_worker())
                } else {
                    None
                };

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

                worker.record_outcome(response.status().is_success());

                // For retryable failures, we need to decrement load since send_typed_request
                // won't have done it (it only decrements on success or non-retryable failures)
                if is_retryable_status(response.status()) && load_incremented {
                    if let Some(cleanup_worker) = worker_for_cleanup {
                        cleanup_worker.decrement_load();
                        RouterMetrics::set_running_requests(
                            cleanup_worker.url(),
                            cleanup_worker.load(),
                        );
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

            // Parse the request body
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

            // Insert the data_parallel_rank field
            if let Some(map) = json_val.as_object_mut() {
                map.insert(
                    String::from("data_parallel_rank"),
                    serde_json::json!(dp_rank),
                );
                debug!(
                    "Modified request body: {}",
                    serde_json::to_string(&json_val).unwrap_or(String::from("ERR"))
                );
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
                    if let Ok(workers_guard) = self.workers.read() {
                        if let Some(worker) = workers_guard.iter().find(|w| w.url() == worker_url) {
                            worker.decrement_load();
                            RouterMetrics::set_running_requests(worker_url, worker.load());
                        }
                    }
                }

                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Request failed: {}", e),
                )
                    .into_response();
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // For non-streaming requests, preserve headers
            let response_headers = header_utils::preserve_response_headers(res.headers());

            let response = match res.bytes().await {
                Ok(body) => {
                    let mut response = Response::new(axum::body::Body::from(body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => {
                    // IMPORTANT: Decrement load on error before returning
                    if load_incremented {
                        if let Ok(workers_guard) = self.workers.read() {
                            if let Some(worker) =
                                workers_guard.iter().find(|w| w.url() == worker_url)
                            {
                                worker.decrement_load();
                                RouterMetrics::set_running_requests(worker_url, worker.load());
                            }
                        }
                    }

                    let error_msg = format!("Failed to get response body: {}", e);
                    (StatusCode::INTERNAL_SERVER_ERROR, error_msg).into_response()
                }
            };

            // Decrement load counter for non-streaming requests if it was incremented
            if load_incremented {
                if let Ok(workers_guard) = self.workers.read() {
                    if let Some(worker) = workers_guard.iter().find(|w| w.url() == worker_url) {
                        worker.decrement_load();
                        RouterMetrics::set_running_requests(worker_url, worker.load());
                    }
                }
            }

            response
        } else if load_incremented {
            // For streaming with load tracking, we need to manually decrement when done
            let workers = Arc::clone(&self.workers);
            let worker_url = worker_url.to_string();

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
                            // Check for stream end marker
                            if bytes
                                .as_ref()
                                .windows(12)
                                .any(|window| window == b"data: [DONE]")
                            {
                                if let Ok(workers_guard) = workers.read() {
                                    if let Some(worker) =
                                        workers_guard.iter().find(|w| w.url() == worker_url)
                                    {
                                        worker.decrement_load();
                                        RouterMetrics::set_running_requests(
                                            &worker_url,
                                            worker.load(),
                                        );
                                        decremented = true;
                                    }
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
                    if let Ok(workers_guard) = workers.read() {
                        if let Some(worker) = workers_guard.iter().find(|w| w.url() == worker_url) {
                            worker.decrement_load();
                            RouterMetrics::set_running_requests(&worker_url, worker.load());
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

    pub async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        let start_time = std::time::Instant::now();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(self.worker_startup_timeout_secs))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        loop {
            if start_time.elapsed() > Duration::from_secs(self.worker_startup_timeout_secs) {
                error!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    self.worker_startup_timeout_secs, worker_url
                );
                return Err(format!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    self.worker_startup_timeout_secs, worker_url
                ));
            }

            match client.get(format!("{}/health", worker_url)).send().await {
                Ok(res) => {
                    if res.status().is_success() {
                        let mut workers_guard = self.workers.write().unwrap();
                        if self.dp_aware {
                            // Need to contact the worker to extract the dp_size,
                            // and add them as multiple workers
                            let url_vec = vec![String::from(worker_url)];
                            let dp_url_vec = Self::get_dp_aware_workers(&url_vec, &self.api_key)
                                .map_err(|e| format!("Failed to get dp-aware workers: {}", e))?;
                            let mut worker_added: bool = false;
                            for dp_url in &dp_url_vec {
                                if workers_guard.iter().any(|w| w.url() == dp_url) {
                                    warn!("Worker {} already exists", dp_url);
                                    continue;
                                }
                                info!("Added worker: {}", dp_url);
                                let new_worker = WorkerFactory::create_regular_with_config(
                                    dp_url.to_string(),
                                    self.circuit_breaker_config.clone(),
                                );
                                workers_guard.push(new_worker);
                                worker_added = true;
                            }
                            if !worker_added {
                                return Err(format!("No worker added for {}", worker_url));
                            }
                        } else {
                            if workers_guard.iter().any(|w| w.url() == worker_url) {
                                return Err(format!("Worker {} already exists", worker_url));
                            }
                            info!("Added worker: {}", worker_url);
                            let new_worker = WorkerFactory::create_regular_with_config(
                                worker_url.to_string(),
                                self.circuit_breaker_config.clone(),
                            );
                            workers_guard.push(new_worker);
                        }

                        RouterMetrics::set_active_workers(workers_guard.len());

                        // If cache aware policy, initialize the worker in the tree
                        if let Some(cache_aware) =
                            self.policy
                                .as_any()
                                .downcast_ref::<crate::policies::CacheAwarePolicy>()
                        {
                            // Get updated workers after adding
                            drop(workers_guard);
                            let workers_guard = self.workers.read().unwrap();
                            cache_aware.init_workers(&workers_guard);
                        }

                        return Ok(format!("Successfully added worker: {}", worker_url));
                    } else {
                        debug!(
                            "Worker {} health check pending - status: {}",
                            worker_url,
                            res.status()
                        );
                        // if the url does not have http or https prefix, warn users
                        if !worker_url.starts_with("http://") && !worker_url.starts_with("https://")
                        {
                            warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                        }

                        tokio::time::sleep(Duration::from_secs(
                            self.worker_startup_check_interval_secs,
                        ))
                        .await;
                        continue;
                    }
                }
                Err(e) => {
                    debug!("Worker {} health check pending - error: {}", worker_url, e);

                    // if the url does not have http or https prefix, warn users
                    if !worker_url.starts_with("http://") && !worker_url.starts_with("https://") {
                        warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                    }

                    tokio::time::sleep(Duration::from_secs(
                        self.worker_startup_check_interval_secs,
                    ))
                    .await;
                    continue;
                }
            }
        }
    }

    pub fn remove_worker(&self, worker_url: &str) {
        if self.dp_aware {
            // remove dp-aware workers in a prefix-matching fashion
            // without contacting the remote worker
            let mut candidate_workers: Vec<String> = Vec::new();
            let mut removed_workers: Vec<String> = Vec::new();
            let worker_url_prefix = format!("{}@", worker_url);

            {
                // find the candidate workers to be removed
                let workers_guard = self.workers.read().unwrap();
                for w in workers_guard.iter() {
                    if w.url().starts_with(&worker_url_prefix) {
                        candidate_workers.push(w.url().to_string());
                    }
                }
            }

            {
                // do the removing on the worker_urls
                let mut workers_guard = self.workers.write().unwrap();
                for dp_url in candidate_workers.iter() {
                    if let Some(index) = workers_guard.iter().position(|w| w.url() == dp_url) {
                        workers_guard.remove(index);
                        info!("Removed worker: {}", dp_url);
                        removed_workers.push(dp_url.to_string());
                    } else {
                        warn!("Worker {} not found, skipping removal", dp_url);
                        continue;
                    }
                }
                RouterMetrics::set_active_workers(workers_guard.len());
            }

            // If cache aware policy, remove the workers from the tree
            if let Some(cache_aware) = self
                .policy
                .as_any()
                .downcast_ref::<crate::policies::CacheAwarePolicy>()
            {
                for dp_url in removed_workers.iter() {
                    cache_aware.remove_worker(dp_url);
                    info!("Removed worker from tree: {}", dp_url);
                }
            }
        } else {
            let mut workers_guard = self.workers.write().unwrap();
            if let Some(index) = workers_guard.iter().position(|w| w.url() == worker_url) {
                workers_guard.remove(index);
                info!("Removed worker: {}", worker_url);
                RouterMetrics::set_active_workers(workers_guard.len());
            } else {
                warn!("Worker {} not found, skipping removal", worker_url);
                return;
            }

            // If cache aware policy, remove the workers from the tree
            if let Some(cache_aware) = self
                .policy
                .as_any()
                .downcast_ref::<crate::policies::CacheAwarePolicy>()
            {
                cache_aware.remove_worker(worker_url);
                info!("Removed worker from tree: {}", worker_url);
            }
        }
    }

    async fn get_worker_load(&self, worker_url: &str) -> Option<isize> {
        let worker_url = if self.dp_aware {
            // Need to extract the URL from "http://host:port@dp_rank"
            let (worker_url_prefix, _dp_rank) = match Self::extract_dp_rank(worker_url) {
                Ok(tup) => tup,
                Err(e) => {
                    error!("Failed to extract dp_rank: {}", e);
                    return None;
                }
            };
            worker_url_prefix
        } else {
            worker_url
        };

        match self
            .client
            .get(format!("{}/get_load", worker_url))
            .send()
            .await
        {
            Ok(res) if res.status().is_success() => match res.bytes().await {
                Ok(bytes) => match serde_json::from_slice::<serde_json::Value>(&bytes) {
                    Ok(data) => data
                        .get("load")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as isize),
                    Err(e) => {
                        debug!("Failed to parse load response from {}: {}", worker_url, e);
                        None
                    }
                },
                Err(e) => {
                    debug!("Failed to read load response from {}: {}", worker_url, e);
                    None
                }
            },
            Ok(res) => {
                debug!(
                    "Worker {} returned non-success status: {}",
                    worker_url,
                    res.status()
                );
                None
            }
            Err(e) => {
                debug!("Failed to get load from {}: {}", worker_url, e);
                None
            }
        }
    }

    // Background task to monitor worker loads
    async fn monitor_worker_loads(
        worker_urls: Vec<String>,
        tx: tokio::sync::watch::Sender<HashMap<String, isize>>,
        interval_secs: u64,
        policy: Arc<dyn LoadBalancingPolicy>,
        client: Client,
    ) {
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));

        loop {
            interval.tick().await;

            let mut loads = HashMap::new();
            for url in &worker_urls {
                if let Some(load) = Self::get_worker_load_static(&client, url).await {
                    loads.insert(url.clone(), load);
                }
            }

            if !loads.is_empty() {
                // Update policy with new loads
                policy.update_loads(&loads);

                // Send to watchers
                if let Err(e) = tx.send(loads) {
                    error!("Failed to send load update: {}", e);
                }
            }
        }
    }

    // Static version of get_worker_load for use in monitoring task
    async fn get_worker_load_static(client: &reqwest::Client, worker_url: &str) -> Option<isize> {
        let worker_url = if worker_url.contains("@") {
            // Need to extract the URL from "http://host:port@dp_rank"
            let (worker_url_prefix, _dp_rank) = match Self::extract_dp_rank(worker_url) {
                Ok(tup) => tup,
                Err(e) => {
                    debug!("Failed to extract dp_rank: {}", e);
                    return None;
                }
            };
            worker_url_prefix
        } else {
            worker_url
        };

        match client.get(format!("{}/get_load", worker_url)).send().await {
            Ok(res) if res.status().is_success() => match res.bytes().await {
                Ok(bytes) => match serde_json::from_slice::<serde_json::Value>(&bytes) {
                    Ok(data) => data
                        .get("load")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as isize),
                    Err(e) => {
                        debug!("Failed to parse load response from {}: {}", worker_url, e);
                        None
                    }
                },
                Err(e) => {
                    debug!("Failed to read load response from {}: {}", worker_url, e);
                    None
                }
            },
            Ok(res) => {
                debug!(
                    "Worker {} returned non-success status: {}",
                    worker_url,
                    res.status()
                );
                None
            }
            Err(e) => {
                debug!("Failed to get load from {}: {}", worker_url, e);
                None
            }
        }
    }
}

use async_trait::async_trait;

#[async_trait]
impl WorkerManagement for Router {
    async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        Router::add_worker(self, worker_url).await
    }

    fn remove_worker(&self, worker_url: &str) {
        Router::remove_worker(self, worker_url)
    }

    fn get_worker_urls(&self) -> Vec<String> {
        Router::get_worker_urls(self)
    }
}

#[async_trait]
impl RouterTrait for Router {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health(&self, _req: Request<Body>) -> Response {
        let workers = self.workers.read().unwrap();
        let unhealthy_servers: Vec<_> = workers
            .iter()
            .filter(|w| !w.is_healthy())
            .map(|w| w.url().to_string())
            .collect();

        if unhealthy_servers.is_empty() {
            (StatusCode::OK, "All servers healthy").into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Unhealthy servers: {:?}", unhealthy_servers),
            )
                .into_response()
        }
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
    ) -> Response {
        self.route_typed_request(headers, body, "/generate").await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/chat/completions")
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/completions")
            .await
    }

    async fn route_embeddings(&self, _headers: Option<&HeaderMap>, _body: Body) -> Response {
        todo!()
    }

    async fn route_rerank(&self, _headers: Option<&HeaderMap>, _body: Body) -> Response {
        todo!()
    }

    async fn flush_cache(&self) -> Response {
        // Get all worker URLs
        let worker_urls = self.get_worker_urls();

        // Send requests to all workers concurrently without headers
        let mut tasks = Vec::new();
        for worker_url in &worker_urls {
            let worker_url = if self.dp_aware {
                // Need to extract the URL from "http://host:port@dp_rank"
                let (worker_url_prefix, _dp_rank) = match Self::extract_dp_rank(worker_url) {
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
                worker_url_prefix
            } else {
                worker_url
            };
            let request_builder = self.client.post(format!("{}/flush_cache", worker_url));
            tasks.push(request_builder.send());
        }

        // Wait for all responses
        let results = futures_util::future::join_all(tasks).await;

        // Check if all succeeded
        let all_success = results.iter().all(|r| {
            r.as_ref()
                .map(|res| res.status().is_success())
                .unwrap_or(false)
        });

        if all_success {
            (StatusCode::OK, "Cache flushed on all servers").into_response()
        } else {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Cache flush failed on one or more servers",
            )
                .into_response()
        }
    }

    async fn get_worker_loads(&self) -> Response {
        let urls = self.get_worker_urls();
        let mut loads = Vec::new();

        // Get loads from all workers
        for url in &urls {
            let load = self.get_worker_load(url).await.unwrap_or(-1);
            loads.push(serde_json::json!({
                "worker": url,
                "load": load
            }));
        }

        Json(serde_json::json!({
            "workers": loads
        }))
        .into_response()
    }

    fn router_type(&self) -> &'static str {
        "regular"
    }

    fn readiness(&self) -> Response {
        // Regular router is ready if it has at least one healthy worker
        let healthy_count = self
            .workers
            .read()
            .unwrap()
            .iter()
            .filter(|w| w.is_healthy())
            .count();

        if healthy_count > 0 {
            Json(serde_json::json!({
                "status": "ready",
                "healthy_workers": healthy_count,
                "total_workers": self.workers.read().unwrap().len()
            }))
            .into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "status": "not_ready",
                    "reason": "no healthy workers available",
                    "total_workers": self.workers.read().unwrap().len()
                })),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::RandomPolicy;
    use std::collections::HashMap;

    fn create_test_regular_router() -> Router {
        let workers = vec![
            WorkerFactory::create_regular("http://worker1:8080".to_string()),
            WorkerFactory::create_regular("http://worker2:8080".to_string()),
        ];
        let (_, rx) = tokio::sync::watch::channel(HashMap::new());
        Router {
            workers: Arc::new(RwLock::new(workers)),
            policy: Arc::new(RandomPolicy::new()),
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            dp_aware: false,
            api_key: None,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
            _worker_loads: Arc::new(rx),
            _load_monitor_handle: None,
            _health_checker: None,
        }
    }

    #[test]
    fn test_router_get_worker_urls_regular() {
        let router = create_test_regular_router();
        let urls = router.get_worker_urls();

        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));
    }

    #[test]
    fn test_select_first_worker_regular() {
        let router = create_test_regular_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "http://worker1:8080");
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_empty_list() {
        // Empty list will return error immediately
        let result = Router::wait_for_healthy_workers(&[], 1, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("no workers provided"));
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_invalid_urls() {
        // This test will timeout quickly since the URLs are invalid
        let result =
            Router::wait_for_healthy_workers(&["http://nonexistent:8080".to_string()], 1, 1).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Timeout"));
    }
}
