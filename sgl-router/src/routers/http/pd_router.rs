// PD (Prefill-Decode) Router Implementation
// This module handles routing for disaggregated prefill-decode systems
use super::pd_types::{api_path, PDRouterError};
use crate::config::types::RetryConfig;
use crate::core::{
    is_retryable_status, BasicWorkerBuilder, CircuitBreakerConfig, HealthConfig, RetryExecutor,
    Worker, WorkerLoadGuard, WorkerRegistry, WorkerType,
};
use crate::metrics::RouterMetrics;
use crate::policies::{LoadBalancingPolicy, PolicyRegistry};
use crate::protocols::spec::{
    ChatCompletionRequest, ChatMessage, CompletionRequest, GenerateRequest, RerankRequest,
    ResponsesRequest, StringOrArray, UserMessageContent,
};
use crate::routers::header_utils;
use crate::routers::{RouterTrait, WorkerManagement};
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn};

#[derive(Debug)]
pub struct PDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub worker_startup_timeout_secs: u64,
    pub worker_startup_check_interval_secs: u64,
    pub worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, isize>>>,
    pub load_monitor_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    pub client: Client,
    // Dedicated client for prefill fire-and-forget (non-logprob) requests
    pub prefill_client: Client,
    pub retry_config: RetryConfig,
    pub circuit_breaker_config: CircuitBreakerConfig,
    // Channel for sending prefill responses to background workers for draining
    prefill_drain_tx: mpsc::Sender<reqwest::Response>,
}

// Request context for PD router operations
#[derive(Clone)]
struct PDRequestContext<'a> {
    route: &'static str,
    batch_size: Option<usize>,
    is_stream: bool,
    return_logprob: bool,
    request_text: Option<String>,
    model_id: Option<&'a str>,
}

impl PDRouter {
    // Private helper method to perform health check on a new server
    async fn wait_for_server_health(&self, url: &str) -> Result<(), PDRouterError> {
        crate::routers::http::router::Router::wait_for_healthy_workers(
            &[url.to_string()],
            self.worker_startup_timeout_secs,
            self.worker_startup_check_interval_secs,
        )
        .await
        .map_err(|_| PDRouterError::HealthCheckFailed {
            url: url.to_string(),
        })
    }

    // Generic helper for processing all workers with an endpoint
    async fn process_workers(
        &self,
        worker_type_enum: WorkerType,
        worker_type: &str,
        endpoint: &str,
    ) -> (Vec<String>, Vec<String>) {
        let mut results = Vec::new();
        let mut errors = Vec::new();

        // Get workers from registry based on type
        let workers = self.worker_registry.get_by_type(&worker_type_enum);
        let urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        // Process each worker
        for worker_url in urls {
            let url = format!("{}/{}", worker_url, endpoint);
            match self.client.post(&url).send().await {
                Ok(res) if res.status().is_success() => {
                    results.push(format!("{} {}: OK", worker_type, worker_url));
                }
                Ok(res) => {
                    errors.push(format!(
                        "{} {} returned status: {}",
                        worker_type,
                        worker_url,
                        res.status()
                    ));
                }
                Err(e) => {
                    errors.push(format!("{} {} error: {}", worker_type, worker_url, e));
                }
            }
        }

        (results, errors)
    }

    // Helper to get prefill worker URLs
    fn get_prefill_worker_urls(&self) -> Vec<String> {
        self.worker_registry
            .get_prefill_workers()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    // Helper to get decode worker URLs
    fn get_decode_worker_urls(&self) -> Vec<String> {
        self.worker_registry
            .get_decode_workers()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    // Helper for proxying requests to the first prefill worker
    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();
        let first_worker_url = workers.first().map(|w| w.url().to_string());

        if let Some(worker_url) = first_worker_url {
            self.proxy_to_worker(worker_url, endpoint, headers).await
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No prefill servers available".to_string(),
            )
                .into_response()
        }
    }

    // Generic helper for proxying to a specific worker
    async fn proxy_to_worker(
        &self,
        worker_url: String,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let url = format!("{}/{}", worker_url, endpoint);
        let mut request_builder = self.client.get(&url);

        // Add headers if provided
        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(axum::body::Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read response body: {}", e),
                        )
                            .into_response()
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                (status, format!("{} server returned status: ", res.status())).into_response()
            }
            Err(e) => {
                error!("Failed to proxy request server: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to proxy request: {}", e),
                )
                    .into_response()
            }
        }
    }

    pub async fn add_prefill_server(
        &self,
        url: String,
        bootstrap_port: Option<u16>,
    ) -> Result<String, PDRouterError> {
        // Wait for the new server to be healthy
        self.wait_for_server_health(&url).await?;

        // Check if already exists
        if self.worker_registry.get_by_url(&url).is_some() {
            return Err(PDRouterError::WorkerAlreadyExists { url: url.clone() });
        }

        // Create Worker for the new prefill server with circuit breaker configuration
        // TODO: In IGW mode, fetch model_id from worker's /get_model_info endpoint
        let worker = BasicWorkerBuilder::new(url.clone())
            .worker_type(WorkerType::Prefill { bootstrap_port })
            .circuit_breaker_config(self.circuit_breaker_config.clone())
            .build();

        let worker_arc: Arc<dyn Worker> = Arc::new(worker);

        // Register the worker in the registry
        self.worker_registry.register(worker_arc.clone());

        // Notify PolicyRegistry about the new worker
        let model_id = worker_arc.model_id();
        let policy = self.policy_registry.on_worker_added(model_id, None);

        // If this is a cache-aware policy, update it with all workers for this model
        if policy.name() == "cache_aware" {
            if let Some(cache_aware) = policy
                .as_any()
                .downcast_ref::<crate::policies::CacheAwarePolicy>()
            {
                let model_workers = self.worker_registry.get_by_model_fast(model_id);
                cache_aware.init_workers(&model_workers);
            }
        }

        info!("Added prefill server: {}", url);
        Ok(format!("Successfully added prefill server: {}", url))
    }

    pub async fn add_decode_server(&self, url: String) -> Result<String, PDRouterError> {
        // Wait for the new server to be healthy
        self.wait_for_server_health(&url).await?;

        // Check if already exists
        if self.worker_registry.get_by_url(&url).is_some() {
            return Err(PDRouterError::WorkerAlreadyExists { url: url.clone() });
        }

        // Create Worker for the new decode server with circuit breaker configuration
        // TODO: In IGW mode, fetch model_id from worker's /get_model_info endpoint
        let worker = BasicWorkerBuilder::new(url.clone())
            .worker_type(WorkerType::Decode)
            .circuit_breaker_config(self.circuit_breaker_config.clone())
            .build();

        let worker_arc: Arc<dyn Worker> = Arc::new(worker);

        // Register the worker in the registry
        self.worker_registry.register(worker_arc.clone());

        // Notify PolicyRegistry about the new worker
        let model_id = worker_arc.model_id();
        let policy = self.policy_registry.on_worker_added(model_id, None);

        // If this is a cache-aware policy, update it with all workers for this model
        if policy.name() == "cache_aware" {
            if let Some(cache_aware) = policy
                .as_any()
                .downcast_ref::<crate::policies::CacheAwarePolicy>()
            {
                let model_workers = self.worker_registry.get_by_model_fast(model_id);
                cache_aware.init_workers(&model_workers);
            }
        }

        info!("Added decode server: {}", url);
        Ok(format!("Successfully added decode server: {}", url))
    }

    pub async fn remove_prefill_server(&self, url: &str) -> Result<String, PDRouterError> {
        // Check if worker exists and get model_id
        let model_id = match self.worker_registry.get_by_url(url) {
            Some(worker) => worker.model_id().to_string(),
            None => {
                return Err(PDRouterError::WorkerNotFound {
                    url: url.to_string(),
                });
            }
        };

        // Remove from registry
        let removed = self.worker_registry.remove_by_url(url);

        if removed.is_some() {
            // Notify PolicyRegistry about the removed worker
            self.policy_registry.on_worker_removed(&model_id);

            // Get the policy for this model to update cache-aware if needed
            if let Some(policy) = self.policy_registry.get_policy(&model_id) {
                if policy.name() == "cache_aware" {
                    if let Some(cache_aware) = policy
                        .as_any()
                        .downcast_ref::<crate::policies::CacheAwarePolicy>()
                    {
                        cache_aware.remove_worker_by_url(url);
                    }
                }
            }
        }

        if removed.is_some() {
            info!("Removed prefill server: {}", url);
            Ok(format!("Successfully removed prefill server: {}", url))
        } else {
            Err(PDRouterError::WorkerNotFound {
                url: url.to_string(),
            })
        }
    }

    pub async fn remove_decode_server(&self, url: &str) -> Result<String, PDRouterError> {
        // Check if worker exists and get model_id
        let model_id = match self.worker_registry.get_by_url(url) {
            Some(worker) => worker.model_id().to_string(),
            None => {
                return Err(PDRouterError::WorkerNotFound {
                    url: url.to_string(),
                });
            }
        };

        // Remove from registry
        let removed = self.worker_registry.remove_by_url(url);

        if removed.is_some() {
            // Notify PolicyRegistry about the removed worker
            self.policy_registry.on_worker_removed(&model_id);

            // Get the policy for this model to update cache-aware if needed
            if let Some(policy) = self.policy_registry.get_policy(&model_id) {
                if policy.name() == "cache_aware" {
                    if let Some(cache_aware) = policy
                        .as_any()
                        .downcast_ref::<crate::policies::CacheAwarePolicy>()
                    {
                        cache_aware.remove_worker_by_url(url);
                    }
                }
            }
        }

        if removed.is_some() {
            info!("Removed decode server: {}", url);
            Ok(format!("Successfully removed decode server: {}", url))
        } else {
            Err(PDRouterError::WorkerNotFound {
                url: url.to_string(),
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        ctx: &Arc<crate::server::AppContext>,
    ) -> Result<Self, String> {
        // Convert config CircuitBreakerConfig to core CircuitBreakerConfig
        let circuit_breaker_config = ctx.router_config.effective_circuit_breaker_config();
        let core_cb_config = CircuitBreakerConfig {
            failure_threshold: circuit_breaker_config.failure_threshold,
            success_threshold: circuit_breaker_config.success_threshold,
            timeout_duration: Duration::from_secs(circuit_breaker_config.timeout_duration_secs),
            window_duration: Duration::from_secs(circuit_breaker_config.window_duration_secs),
        };

        // Register prefill workers in the registry
        for (url, port) in prefill_urls {
            let worker = BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: port,
                })
                .circuit_breaker_config(core_cb_config.clone())
                .health_config(HealthConfig {
                    timeout_secs: ctx.router_config.health_check.timeout_secs,
                    check_interval_secs: ctx.router_config.health_check.check_interval_secs,
                    endpoint: ctx.router_config.health_check.endpoint.clone(),
                    failure_threshold: ctx.router_config.health_check.failure_threshold,
                    success_threshold: ctx.router_config.health_check.success_threshold,
                })
                .build();
            ctx.worker_registry.register(Arc::new(worker));
        }

        // Register decode workers in the registry
        for url in decode_urls {
            let worker = BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Decode)
                .circuit_breaker_config(core_cb_config.clone())
                .health_config(HealthConfig {
                    timeout_secs: ctx.router_config.health_check.timeout_secs,
                    check_interval_secs: ctx.router_config.health_check.check_interval_secs,
                    endpoint: ctx.router_config.health_check.endpoint.clone(),
                    failure_threshold: ctx.router_config.health_check.failure_threshold,
                    success_threshold: ctx.router_config.health_check.success_threshold,
                })
                .build();
            ctx.worker_registry.register(Arc::new(worker));
        }

        // Get all workers from registry for health check
        let all_workers = ctx.worker_registry.get_all();
        let all_urls: Vec<String> = all_workers
            .iter()
            .map(|worker| worker.url().to_string())
            .collect();
        if !all_urls.is_empty() {
            crate::routers::http::router::Router::wait_for_healthy_workers(
                &all_urls,
                ctx.router_config.worker_startup_timeout_secs,
                ctx.router_config.worker_startup_check_interval_secs,
            )
            .await?;
        }

        // Initialize cache-aware policies with workers from registry
        // Note: We need to get workers by type and convert to Box<dyn Worker> for CacheAwarePolicy
        // This is a temporary workaround until CacheAwarePolicy is updated to work with Arc<dyn Worker>
        // TODO: Update CacheAwarePolicy to accept Arc<dyn Worker> instead of Box<dyn Worker>

        // Set up background load monitoring for power-of-two selection
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let worker_loads = Arc::new(rx);

        // Get policies from registry to check if we need load monitoring
        let prefill_policy = ctx.policy_registry.get_prefill_policy();
        let decode_policy = ctx.policy_registry.get_decode_policy();

        let load_monitor_handle =
            if prefill_policy.name() == "power_of_two" || decode_policy.name() == "power_of_two" {
                let monitor_urls = all_urls.clone();
                let monitor_interval = ctx.router_config.worker_startup_check_interval_secs;
                let monitor_client = ctx.client.clone();
                let prefill_policy_clone = Arc::clone(&prefill_policy);
                let decode_policy_clone = Arc::clone(&decode_policy);

                Some(Arc::new(tokio::spawn(async move {
                    Self::monitor_worker_loads_with_client(
                        monitor_urls,
                        tx,
                        monitor_interval,
                        monitor_client,
                        prefill_policy_clone,
                        decode_policy_clone,
                    )
                    .await;
                })))
            } else {
                None
            };

        // Note: Health checking is now handled centrally by RouterManager
        // Individual routers no longer need to manage health checkers

        // Build a dedicated prefill client for fire-and-forget semantics
        let prefill_client = reqwest::Client::builder()
            .pool_max_idle_per_host(0)
            .http1_only()
            .connect_timeout(Duration::from_millis(300))
            .timeout(Duration::from_secs(ctx.router_config.request_timeout_secs))
            .build()
            .map_err(|e| format!("Failed to build prefill client: {}", e))?;

        // Create bounded channel for prefill response draining
        // Larger buffer for high concurrency scenarios
        let (prefill_drain_tx, mut prefill_drain_rx) = mpsc::channel::<reqwest::Response>(2000);

        // Spawn a coordinator with limited concurrent drain tasks
        // This prevents unbounded task spawning under extreme load
        tokio::spawn(async move {
            info!("Prefill drain coordinator started");

            // Use a semaphore to limit concurrent drain operations
            let max_concurrent_drains = 100;
            let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent_drains));

            while let Some(response) = prefill_drain_rx.recv().await {
                let permit = semaphore.clone().acquire_owned().await;

                match permit {
                    Ok(permit) => {
                        // Spawn a task to drain this response
                        tokio::spawn(async move {
                            let url = response.url().to_string();
                            let status = response.status();

                            if !status.is_success() {
                                error!("Prefill drain: error status={} url={}", status, url);
                                RouterMetrics::record_pd_prefill_error(&url);
                            }

                            // Drain the response body efficiently
                            // Use streaming to avoid loading entire body into memory
                            let start = std::time::Instant::now();
                            let mut stream = response.bytes_stream();
                            let mut bytes_drained = 0;

                            while let Some(chunk_result) = stream.next().await {
                                match chunk_result {
                                    Ok(chunk) => bytes_drained += chunk.len(),
                                    Err(e) => {
                                        debug!(
                                            "Prefill drain: error streaming url={} error={}",
                                            url, e
                                        );
                                        break;
                                    }
                                }
                            }

                            let elapsed = start.elapsed();
                            if elapsed > Duration::from_millis(100) {
                                // Only log slow drains
                                debug!(
                                    "Prefill drain: slow drain {} bytes from {} in {:?}",
                                    bytes_drained, url, elapsed
                                );
                            }

                            // Permit is automatically released when dropped
                            drop(permit);
                        });
                    }
                    Err(_) => {
                        // Semaphore closed, shutting down
                        break;
                    }
                }
            }
            info!("Prefill drain coordinator shutting down");
        });

        Ok(PDRouter {
            worker_registry: Arc::clone(&ctx.worker_registry),
            policy_registry: Arc::clone(&ctx.policy_registry),
            worker_startup_timeout_secs: ctx.router_config.worker_startup_timeout_secs,
            worker_startup_check_interval_secs: ctx
                .router_config
                .worker_startup_check_interval_secs,
            worker_loads,
            load_monitor_handle,
            client: ctx.client.clone(),
            prefill_client,
            prefill_drain_tx,
            retry_config: ctx.router_config.effective_retry_config(),
            circuit_breaker_config: core_cb_config,
        })
    }

    // Helper to handle server selection errors
    fn handle_server_selection_error(error: String) -> Response {
        error!("Failed to select PD pair error={}", error);
        RouterMetrics::record_pd_error("server_selection");
        (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("No available servers: {}", error),
        )
            .into_response()
    }

    // Helper to handle serialization errors
    fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        error!("Failed to serialize request error={}", error);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to serialize request",
        )
            .into_response()
    }

    // Helper to determine batch size from a GenerateRequest
    fn get_generate_batch_size(req: &GenerateRequest) -> Option<usize> {
        // Check prompt array
        if let Some(StringOrArray::Array(arr)) = &req.prompt {
            if !arr.is_empty() {
                return Some(arr.len());
            }
        }
        // Check text array
        if let Some(text) = &req.text {
            if text.contains("[") && text.contains("]") {
                // This is a simplified check - in reality we'd need to parse JSON
                return None; // For now, fall back to non-batch
            }
        }
        None
    }

    // Helper to determine batch size from a ChatCompletionRequest
    fn get_chat_batch_size(req: &ChatCompletionRequest) -> Option<usize> {
        // Check 'n' parameter for multiple responses
        if let Some(n) = req.n {
            if n > 1 {
                return Some(n as usize);
            }
        }
        None
    }

    // Helper to determine batch size from a CompletionRequest
    fn get_completion_batch_size(req: &CompletionRequest) -> Option<usize> {
        // Check prompt array
        if let StringOrArray::Array(arr) = &req.prompt {
            if !arr.is_empty() {
                return Some(arr.len());
            }
        }
        None
    }

    // Helper to inject bootstrap fields into an existing JSON request value
    fn inject_bootstrap_into_value(
        mut original: Value,
        prefill_worker: &dyn Worker,
        batch_size: Option<usize>,
    ) -> Result<Value, String> {
        let bootstrap_port = match prefill_worker.worker_type() {
            crate::core::WorkerType::Prefill { bootstrap_port } => bootstrap_port,
            _ => None,
        };
        let hostname = super::pd_types::get_hostname(prefill_worker.url());

        let obj = original
            .as_object_mut()
            .ok_or_else(|| "Request must be a JSON object".to_string())?;

        if let Some(n) = batch_size {
            let mut hosts = Vec::with_capacity(n);
            let mut ports = Vec::with_capacity(n);
            let mut rooms = Vec::with_capacity(n);
            for _ in 0..n {
                hosts.push(hostname.clone());
                ports.push(bootstrap_port);
                rooms.push(super::pd_types::generate_room_id());
            }
            obj.insert(
                "bootstrap_host".to_string(),
                Value::Array(hosts.into_iter().map(serde_json::Value::from).collect()),
            );
            obj.insert(
                "bootstrap_port".to_string(),
                Value::Array(
                    ports
                        .into_iter()
                        .map(|p| match p {
                            Some(v) => serde_json::Value::from(v),
                            None => Value::Null,
                        })
                        .collect(),
                ),
            );
            obj.insert(
                "bootstrap_room".to_string(),
                Value::Array(rooms.into_iter().map(serde_json::Value::from).collect()),
            );
        } else {
            obj.insert(
                "bootstrap_host".to_string(),
                serde_json::Value::from(hostname),
            );
            obj.insert(
                "bootstrap_port".to_string(),
                match bootstrap_port {
                    Some(v) => serde_json::Value::from(v),
                    None => Value::Null,
                },
            );
            obj.insert(
                "bootstrap_room".to_string(),
                serde_json::Value::from(super::pd_types::generate_room_id()),
            );
        }
        Ok(original)
    }

    // Execute the dual dispatch to prefill and decode servers with retries and bootstrap injection
    async fn execute_dual_dispatch<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // Operation per attempt
            {
                let original_request = original_request.clone();
                move |attempt: u32| {
                    let original_request = original_request.clone();
                    let context = context.clone();
                    async move {
                        // Select workers fresh for each attempt
                        let (prefill, decode) = match self
                            .select_pd_pair(context.request_text.as_deref(), context.model_id)
                            .await
                        {
                            Ok(pair) => pair,
                            Err(e) => {
                                RouterMetrics::record_pd_error("server_selection");
                                return Self::handle_server_selection_error(e);
                            }
                        };

                        debug!(
                            "PD retry attempt {} using prefill={} decode={}",
                            attempt,
                            prefill.url(),
                            decode.url()
                        );

                        // Serialize the original request
                        let mut json_request = match serde_json::to_value(&original_request) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        // Inject bootstrap based on current prefill worker
                        json_request = match Self::inject_bootstrap_into_value(
                            json_request,
                            prefill.as_ref(),
                            context.batch_size,
                        ) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        // Execute the actual dual dispatch
                        let response = self
                            .execute_dual_dispatch_internal(
                                headers,
                                json_request,
                                context,
                                prefill.as_ref(),
                                decode.as_ref(),
                                start_time,
                            )
                            .await;

                        // Record outcomes for circuit breakers
                        let _status = response.status();
                        let not_error = _status.is_success() || _status.is_client_error();
                        prefill.record_outcome(not_error);
                        decode.record_outcome(not_error);

                        response
                    }
                }
            },
            // Should retry predicate
            |res, _attempt| is_retryable_status(res.status()),
            // On backoff hook
            |delay, attempt| {
                RouterMetrics::record_retry(route);
                RouterMetrics::record_retry_backoff_duration(delay, attempt);
            },
            // On exhausted hook
            || RouterMetrics::record_retries_exhausted(route),
        )
        .await
    }

    async fn handle_decode_error_response(
        &self,
        res: reqwest::Response,
        context: &PDRequestContext<'_>,
        prefill: &dyn Worker,
        decode: &dyn Worker,
    ) -> Response {
        let status = res.status();

        if context.is_stream {
            // Handle streaming error response
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let error_payload = match res.bytes().await {
                Ok(error_body) => {
                    if let Ok(error_json) = serde_json::from_slice::<Value>(&error_body) {
                        json!({ "message": error_json, "status": status.as_u16() })
                    } else {
                        json!({ "message": String::from_utf8_lossy(&error_body).to_string(), "status": status.as_u16() })
                    }
                }
                Err(e) => {
                    json!({ "message": format!("Decode server error: {}", e), "status": status.as_u16() })
                }
            };

            let sse_data = format!(
                "data: {{'error': {}}}",
                serde_json::to_string(&error_payload).unwrap_or_default()
            );
            let error_stream = tokio_stream::once(Ok(axum::body::Bytes::from(sse_data)));

            let decode_url = decode.url().to_string();
            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
                Some(decode_url),
                Some(response_headers),
                prefill,
                decode,
            )
        } else {
            // Handle non-streaming error response
            match res.bytes().await {
                Ok(error_body) => (status, error_body).into_response(),
                Err(e) => (status, format!("Decode server error: {}", e)).into_response(),
            }
        }
    }

    // Internal method that performs the actual dual dispatch (without retry logic)
    async fn execute_dual_dispatch_internal(
        &self,
        headers: Option<&HeaderMap>,
        json_request: Value,
        context: PDRequestContext<'_>,
        prefill: &dyn Worker,
        decode: &dyn Worker,
        start_time: Instant,
    ) -> Response {
        // For non-streaming: use guard for automatic load management
        // For streaming: load will be managed in create_streaming_response
        let _guard = if !context.is_stream {
            Some(WorkerLoadGuard::new_multi(vec![prefill, decode]))
        } else {
            None
        };

        // Build decode request with shared client
        let decode_request = self.build_post_with_headers(
            &self.client,
            decode.url(),
            context.route,
            &json_request,
            headers,
            false,
        );

        // Send both requests concurrently
        debug!(
            "Sending concurrent requests to prefill={} decode={}",
            prefill.url(),
            decode.url()
        );

        if context.return_logprob {
            // Build prefill request with shared client when we need response body
            let prefill_request = self.build_post_with_headers(
                &self.client,
                prefill.url(),
                context.route,
                &json_request,
                headers,
                false,
            );
            // When we need logprobs, wait for both responses
            let (prefill_result, decode_result) =
                tokio::join!(prefill_request.send(), decode_request.send());
            debug!("Received responses from both servers");

            // Update metrics
            let duration = start_time.elapsed();
            RouterMetrics::record_pd_request_duration(context.route, duration);
            RouterMetrics::record_pd_request(context.route);
            RouterMetrics::record_pd_prefill_request(prefill.url());
            RouterMetrics::record_pd_decode_request(decode.url());

            // Process decode response with prefill for logprobs
            debug!("Processing decode response with logprobs");
            match decode_result {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    debug!("Decode response status: {}", status);

                    if !status.is_success() {
                        RouterMetrics::record_pd_decode_error(decode.url());
                        error!(
                            "Decode server returned error status decode_url={} status={}",
                            decode.url(),
                            status
                        );

                        return self
                            .handle_decode_error_response(res, &context, prefill, decode)
                            .await;
                    }

                    // Process prefill response for logprobs
                    let prefill_body = match self
                        .process_prefill_response(
                            prefill_result,
                            prefill.url(),
                            context.return_logprob,
                        )
                        .await
                    {
                        Ok((_, body)) => body,
                        Err(error_response) => return error_response,
                    };

                    if context.is_stream {
                        // Streaming response with logprobs
                        let prefill_logprobs = prefill_body
                            .as_ref()
                            .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                            .and_then(|json| {
                                json.pointer("/meta_info/input_token_logprobs").cloned()
                            });

                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        self.create_streaming_response(
                            res.bytes_stream(),
                            status,
                            prefill_logprobs,
                            context.return_logprob,
                            None,
                            Some(response_headers),
                            prefill,
                            decode,
                        )
                    } else {
                        // Non-streaming response with logprobs
                        self.process_non_streaming_response(
                            res,
                            status,
                            context.return_logprob,
                            prefill_body,
                        )
                        .await
                    }
                }
                Err(e) => {
                    error!(
                        decode_url = %decode.url(),
                        error = %e,
                        "Decode request failed"
                    );
                    RouterMetrics::record_pd_decode_error(decode.url());
                    (
                        StatusCode::BAD_GATEWAY,
                        format!("Decode server error: {}", e),
                    )
                        .into_response()
                }
            }
        } else {
            // When we don't need logprobs, only wait for decode response
            // Send both requests concurrently but don't wait for prefill
            // Use dedicated prefill client with Connection: close
            let prefill_future = self
                .build_post_with_headers(
                    &self.prefill_client,
                    prefill.url(),
                    context.route,
                    &json_request,
                    headers,
                    true,
                )
                .send();
            let decode_future = decode_request.send();

            // Send prefill response to background worker for draining
            // This ensures HTTP compliance without blocking
            let drain_tx = self.prefill_drain_tx.clone();
            let prefill_url = prefill.url().to_string();
            tokio::spawn(async move {
                if let Ok(response) = prefill_future.await {
                    // Try to send to drain worker
                    // If channel is full (under extreme load), drain inline as fallback
                    match drain_tx.try_send(response) {
                        Ok(_) => {
                            // Successfully queued for draining
                            debug!("Prefill response queued for draining");
                        }
                        Err(mpsc::error::TrySendError::Full(response)) => {
                            // Channel full - drain inline as fallback
                            warn!("Prefill drain channel full (capacity exceeded), draining inline for {}", prefill_url);
                            RouterMetrics::record_pd_prefill_error(&prefill_url);

                            // Drain inline with timeout to prevent blocking too long
                            let drain_future = async {
                                let mut stream = response.bytes_stream();
                                while stream.next().await.is_some() {
                                    // Just drain
                                }
                            };

                            match tokio::time::timeout(Duration::from_secs(1), drain_future).await {
                                Ok(_) => debug!("Inline drain completed for {}", prefill_url),
                                Err(_) => error!("Inline drain timeout for {}", prefill_url),
                            }
                        }
                        Err(mpsc::error::TrySendError::Closed(_)) => {
                            error!("Prefill drain channel closed!");
                        }
                    }
                }
            });

            // Wait only for decode response
            let decode_result = decode_future.await;
            debug!("Received decode response");

            // Update metrics
            let duration = start_time.elapsed();
            RouterMetrics::record_pd_request_duration(context.route, duration);
            RouterMetrics::record_pd_request(context.route);
            RouterMetrics::record_pd_prefill_request(prefill.url());
            RouterMetrics::record_pd_decode_request(decode.url());

            // Process decode response immediately
            debug!("Processing decode response (no logprobs)");
            match decode_result {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    debug!("Decode response status: {}", status);

                    if !status.is_success() {
                        RouterMetrics::record_pd_decode_error(decode.url());
                        error!(
                            "Decode server returned error status decode_url={} status={}",
                            decode.url(),
                            status
                        );

                        self.handle_decode_error_response(res, &context, prefill, decode)
                            .await
                    } else if context.is_stream {
                        // Streaming response without logprobs - direct passthrough
                        let decode_url = decode.url().to_string();
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        self.create_streaming_response(
                            res.bytes_stream(),
                            status,
                            None,
                            false,
                            Some(decode_url),
                            Some(response_headers),
                            prefill,
                            decode,
                        )
                    } else {
                        // Non-streaming response without logprobs - direct passthrough like fast version
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(decode_body) => {
                                let mut response =
                                    Response::new(axum::body::Body::from(decode_body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => {
                                error!("Failed to read decode response: {}", e);
                                (StatusCode::INTERNAL_SERVER_ERROR, "Failed to read response")
                                    .into_response()
                            }
                        }
                    }
                }
                Err(e) => {
                    error!(
                        decode_url = %decode.url(),
                        error = %e,
                        "Decode request failed"
                    );
                    RouterMetrics::record_pd_decode_error(decode.url());
                    (
                        StatusCode::BAD_GATEWAY,
                        format!("Decode server error: {}", e),
                    )
                        .into_response()
                }
            }
        }
    }

    // Check if either prefill or decode policy needs request text
    fn policies_need_request_text(&self) -> bool {
        // Check both prefill and decode policies
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        prefill_policy.needs_request_text() || decode_policy.needs_request_text()
    }

    // Select a pair of prefill and decode servers considering circuit breaker state
    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        // Get workers from registry - filter by model if provided
        let prefill_workers = if let Some(model) = model_id {
            // Get model-specific workers and filter for prefill type
            self.worker_registry
                .get_by_model_fast(model)
                .into_iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Prefill { .. }))
                .collect()
        } else {
            self.worker_registry.get_prefill_workers()
        };

        let decode_workers = if let Some(model) = model_id {
            // Get model-specific workers and filter for decode type
            self.worker_registry
                .get_by_model_fast(model)
                .into_iter()
                .filter(|w| matches!(w.worker_type(), WorkerType::Decode))
                .collect()
        } else {
            self.worker_registry.get_decode_workers()
        };

        // Select workers using helper function
        // Use separate policies for prefill and decode to avoid counter conflicts
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();

        let prefill = Self::pick_worker_by_policy_arc(
            &prefill_workers,
            &*prefill_policy,
            request_text,
            "prefill",
        )?;

        let decode = Self::pick_worker_by_policy_arc(
            &decode_workers,
            &*decode_policy,
            request_text,
            "decode",
        )?;

        Ok((prefill, decode))
    }

    // Helper function to select a worker using the policy (Arc version)
    fn pick_worker_by_policy_arc(
        workers: &[Arc<dyn Worker>],
        policy: &dyn LoadBalancingPolicy,
        request_text: Option<&str>,
        worker_type: &str,
    ) -> Result<Arc<dyn Worker>, String> {
        // Check if we have any workers
        if workers.is_empty() {
            return Err(format!(
                "No {} workers available. Please check if {} servers are configured and healthy.",
                worker_type, worker_type
            ));
        }

        // Filter available workers (healthy + circuit breaker not open)
        let available_workers: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();

        if available_workers.is_empty() {
            return Err(format!(
                "No available {} workers (all circuits open or unhealthy)",
                worker_type
            ));
        }

        // Let policy select from available workers (no conversion needed now!)
        let selected_idx = policy
            .select_worker(&available_workers, request_text)
            .ok_or_else(|| {
                format!(
                    "Policy {} failed to select a {} worker",
                    policy.name(),
                    worker_type
                )
            })?;

        // Return the selected Arc worker
        Ok(available_workers[selected_idx].clone())
    }

    // Background task to monitor worker loads with shared client
    async fn monitor_worker_loads_with_client(
        worker_urls: Vec<String>,
        tx: tokio::sync::watch::Sender<HashMap<String, isize>>,
        interval_secs: u64,
        client: Client,
        prefill_policy: Arc<dyn LoadBalancingPolicy>,
        decode_policy: Arc<dyn LoadBalancingPolicy>,
    ) {
        loop {
            let mut loads = HashMap::new();

            let futures: Vec<_> = worker_urls
                .iter()
                .map(|url| {
                    let client = client.clone();
                    let url = url.clone();
                    async move {
                        let load = get_worker_load(&client, &url).await.unwrap_or(0);
                        (url, load)
                    }
                })
                .collect();

            let results = futures_util::future::join_all(futures).await;

            for (url, load) in results {
                loads.insert(url, load);
            }

            debug!("Worker loads updated: {:?}", loads);

            // Update both policies with current loads
            prefill_policy.update_loads(&loads);
            decode_policy.update_loads(&loads);

            // Check if receiver is still active
            if tx.send(loads).is_err() {
                info!("Load monitor receiver dropped, shutting down monitor task");
                break;
            }

            tokio::time::sleep(Duration::from_secs(interval_secs)).await;
        }
    }

    // Helper to create a streaming response
    #[allow(clippy::too_many_arguments)]
    fn create_streaming_response(
        &self,
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        decode_url: Option<String>,
        headers: Option<HeaderMap>,
        prefill: &dyn Worker,
        decode: &dyn Worker,
    ) -> Response {
        // For streaming, increment load now - will be decremented when streaming completes
        prefill.increment_load();
        decode.increment_load();

        // Store URLs to find workers later for decrementing
        let prefill_url = prefill.url().to_string();
        let decode_url_str = decode.url().to_string();

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Clone the registry for the spawned task
        let registry = self.worker_registry.clone();

        tokio::spawn(async move {
            // Use a flag to track whether stream completed successfully
            let mut stream_completed = false;

            futures_util::pin_mut!(stream);
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        // Check for stream end marker to decrement load early
                        let is_done = chunk
                            .as_ref()
                            .windows(12)
                            .any(|window| window == b"data: [DONE]");

                        let result = if return_logprob && prefill_logprobs.is_some() {
                            // Try to merge logprobs
                            Self::merge_streaming_logprobs(prefill_logprobs.clone(), &chunk)
                                .unwrap_or(chunk)
                        } else {
                            chunk
                        };

                        if tx.send(Ok(result)).is_err() {
                            break;
                        }

                        // If we see the done marker, decrement load immediately
                        if is_done {
                            stream_completed = true;
                            break;
                        }
                    }
                    Err(e) => {
                        if let Some(ref url) = decode_url {
                            error!("Stream error from decode server {}: {}", url, e);
                            RouterMetrics::record_pd_stream_error(url);
                        }
                        let _ = tx.send(Err(format!("Stream error: {}", e)));
                        break;
                    }
                }
            }

            // Always decrement load after streaming (either completes or errors)
            // Find and decrement prefill worker
            if let Some(worker) = registry.get_by_url(&prefill_url) {
                worker.decrement_load();
                debug!(
                    "Decremented load for prefill worker: {} (stream_completed: {})",
                    prefill_url, stream_completed
                );
            }

            // Find and decrement decode worker
            if let Some(worker) = registry.get_by_url(&decode_url_str) {
                worker.decrement_load();
                debug!(
                    "Decremented load for decode worker: {} (stream_completed: {})",
                    decode_url_str, stream_completed
                );
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        let mut response = Response::new(body);
        *response.status_mut() = status;

        // Use provided headers or create new ones, then ensure content-type is set for streaming
        let mut headers = headers.unwrap_or_default();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = headers;

        response
    }

    // Helper to process non-streaming decode response with logprob merging
    async fn process_non_streaming_response(
        &self,
        res: reqwest::Response,
        status: StatusCode,
        return_logprob: bool,
        prefill_body: Option<bytes::Bytes>,
    ) -> Response {
        let response = res.bytes().await;
        let decode_body = match response {
            Ok(decode_body) => decode_body,
            Err(e) => {
                error!("Failed to read decode response: {}", e);
                return (StatusCode::INTERNAL_SERVER_ERROR, "Failed to read response")
                    .into_response();
            }
        };

        if !return_logprob {
            return (status, decode_body).into_response();
        }

        let Some(prefill_body) = prefill_body else {
            return (status, decode_body).into_response();
        };

        // Merge logprobs from prefill and decode
        let (Ok(prefill_json), Ok(mut decode_json)) = (
            serde_json::from_slice::<Value>(&prefill_body),
            serde_json::from_slice::<Value>(&decode_body),
        ) else {
            warn!("Failed to parse responses for logprob merging");
            return (status, decode_body).into_response();
        };

        Self::merge_logprobs_in_json(&prefill_json, &mut decode_json);

        // Return merged response
        match serde_json::to_vec(&decode_json) {
            Ok(body) => (status, body).into_response(),
            Err(e) => {
                error!("Failed to serialize merged response: {}", e);
                (status, decode_body).into_response()
            }
        }
    }

    // Helper to process prefill response and extract body if needed for logprobs
    async fn process_prefill_response(
        &self,
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        prefill_url: &str,
        return_logprob: bool,
    ) -> Result<(StatusCode, Option<bytes::Bytes>), Response> {
        // Check prefill result first - it's critical for disaggregated mode
        let prefill_response = match prefill_result {
            Ok(response) => response,
            Err(e) => {
                RouterMetrics::record_pd_prefill_error(prefill_url);
                error!(
                    "Prefill server failed (CRITICAL) prefill_url={} error={}. Decode will timeout without prefill KV cache.",
                    prefill_url,
                    e
                );

                // Return error immediately - don't wait for decode to timeout
                return Err((
                    StatusCode::BAD_GATEWAY,
                    format!(
                        "Prefill server error: {}. This will cause decode timeout.",
                        e
                    ),
                )
                    .into_response());
            }
        };

        let prefill_status = StatusCode::from_u16(prefill_response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // Check if prefill succeeded
        if !prefill_status.is_success() {
            RouterMetrics::record_pd_prefill_error(prefill_url);

            // Get error body from prefill
            let error_msg = prefill_response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown prefill error".to_string());

            error!(
                "Prefill server returned error status prefill_url={} status={} body={}",
                prefill_url, prefill_status, error_msg
            );

            return Err((
                prefill_status,
                format!("Prefill server error ({}): {}", prefill_status, error_msg),
            )
                .into_response());
        }

        // Read prefill body if needed for logprob merging
        let prefill_body = if return_logprob {
            match prefill_response.bytes().await {
                Ok(body) => Some(body),
                Err(e) => {
                    warn!("Failed to read prefill response body for logprobs: {}", e);
                    None
                }
            }
        } else {
            // For non-logprob requests, just consume the response without storing
            debug!("Consuming prefill response body (non-logprob request)");
            match prefill_response.bytes().await {
                Ok(_) => debug!("Prefill response consumed successfully"),
                Err(e) => warn!("Error consuming prefill response: {}", e),
            }
            None
        };

        Ok((prefill_status, prefill_body))
    }

    fn build_post_with_headers(
        &self,
        client: &Client,
        url: &str,
        route: &str,
        json_request: &Value,
        headers: Option<&HeaderMap>,
        connection_close: bool,
    ) -> reqwest::RequestBuilder {
        let mut request = client.post(api_path(url, route)).json(json_request);
        if connection_close {
            request = request.header("Connection", "close");
        }
        if let Some(headers) = headers {
            for (name, value) in headers.iter() {
                let name_lc = name.as_str().to_ascii_lowercase();
                // Whitelist important end-to-end headers, skip hop-by-hop
                let forward = matches!(
                    name_lc.as_str(),
                    "authorization" | "x-request-id" | "x-correlation-id"
                ) || name_lc.starts_with("x-request-id-");
                if forward {
                    if let Ok(val) = value.to_str() {
                        request = request.header(name, val);
                    }
                }
            }
        }
        request
    }

    // Helper to merge logprobs from prefill and decode responses
    fn merge_logprobs_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
        if let (Some(prefill_meta), Some(decode_meta)) = (
            prefill_json.get("meta_info"),
            decode_json.get_mut("meta_info"),
        ) {
            if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
                prefill_meta.get("input_token_logprobs"),
                decode_meta.get_mut("input_token_logprobs"),
            ) {
                if let (Some(prefill_arr), Some(decode_arr)) =
                    (prefill_logprobs.as_array(), decode_logprobs.as_array_mut())
                {
                    let mut merged = prefill_arr.clone();
                    merged.extend(decode_arr.clone());
                    decode_meta["input_token_logprobs"] = Value::Array(merged);
                    return true;
                }
            }
        }
        false
    }

    // Simple helper to merge logprobs in streaming responses
    fn merge_streaming_logprobs(
        prefill_logprobs: Option<Value>,
        decode_chunk: &[u8],
    ) -> Result<bytes::Bytes, ()> {
        // Skip non-data chunks
        let chunk_str = std::str::from_utf8(decode_chunk).map_err(|_| ())?;
        if !chunk_str.starts_with("data: ") || chunk_str.contains("[DONE]") {
            return Err(());
        }

        // Parse JSON from chunk
        let json_str = chunk_str.trim_start_matches("data: ").trim();
        let mut decode_json: Value = serde_json::from_str(json_str).map_err(|_| ())?;

        // Merge prefill logprobs if available
        if let Some(ref p_logprobs) = prefill_logprobs {
            if let Some(meta) = decode_json.get_mut("meta_info") {
                if let Some(d_logprobs) = meta.get_mut("input_token_logprobs") {
                    if let (Some(p_arr), Some(d_arr)) =
                        (p_logprobs.as_array(), d_logprobs.as_array())
                    {
                        let mut merged = p_arr.clone();
                        merged.extend(d_arr.clone());
                        *d_logprobs = Value::Array(merged);
                    }
                }
            }
        }

        // Re-serialize
        let merged_str = format!(
            "data: {}\n\n",
            serde_json::to_string(&decode_json).unwrap_or_default()
        );
        Ok(bytes::Bytes::from(merged_str))
    }
}

// Helper functions

async fn get_worker_load(client: &Client, worker_url: &str) -> Option<isize> {
    match client.get(format!("{}/get_load", worker_url)).send().await {
        Ok(res) if res.status().is_success() => match res.bytes().await {
            Ok(bytes) => match serde_json::from_slice::<Value>(&bytes) {
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

#[async_trait]
impl WorkerManagement for PDRouter {
    async fn add_worker(&self, _worker_url: &str) -> Result<String, String> {
        // For PD router, we don't support adding workers via this generic method
        Err(
            "PD router requires specific add_prefill_server or add_decode_server methods"
                .to_string(),
        )
    }

    fn remove_worker(&self, worker_url: &str) {
        // Remove from registry
        if let Some(worker) = self.worker_registry.remove_by_url(worker_url) {
            match worker.worker_type() {
                WorkerType::Prefill { .. } => {
                    info!("Removed prefill worker: {}", worker_url);
                }
                WorkerType::Decode => {
                    info!("Removed decode worker: {}", worker_url);
                }
                _ => {
                    info!("Removed worker: {}", worker_url);
                }
            }
        }
    }

    fn get_worker_urls(&self) -> Vec<String> {
        self.worker_registry.get_all_urls()
    }
}

#[async_trait]
impl RouterTrait for PDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health(&self, _req: Request<Body>) -> Response {
        // This is a server readiness check - checking if we have healthy workers
        // Workers handle their own health checks in the background
        let mut all_healthy = true;
        let mut unhealthy_servers = Vec::new();

        // Check all workers
        for worker in self.worker_registry.get_all() {
            if !worker.is_healthy() {
                all_healthy = false;
                let worker_type = match worker.worker_type() {
                    WorkerType::Prefill { .. } => "Prefill",
                    WorkerType::Decode => "Decode",
                    _ => "Worker",
                };
                unhealthy_servers.push(format!("{}: {}", worker_type, worker.url()));
            }
        }

        if all_healthy {
            (StatusCode::OK, "All servers healthy").into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Unhealthy servers: {:?}", unhealthy_servers),
            )
                .into_response()
        }
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Test model generation capability by selecting a random pair and testing them
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        // Select a random worker pair using the policy
        let (prefill, decode) = match self.select_pd_pair(None, None).await {
            Ok(pair) => pair,
            Err(e) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("No healthy worker pair available: {}", e),
                )
                    .into_response();
            }
        };

        // Test prefill server's health_generate
        let prefill_url = format!("{}/health_generate", prefill.url());
        let (prefill_result, decode_result) = tokio::join!(
            self.client.get(&prefill_url).send(),
            self.client
                .get(format!("{}/health_generate", decode.url()))
                .send()
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Health generate failed: {:?}", errors),
            )
                .into_response()
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // Get info from the first decode server to match sglang's server info format
        // Note: We use decode workers for server info to match expected format
        self.proxy_to_first_prefill_worker("get_server_info", None)
            .await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("get_model_info", Some(headers))
            .await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Extract parameters
        let is_stream = body.stream;
        let return_logprob = body.return_logprob;

        // Extract text for cache-aware routing
        let request_text = if self.policies_need_request_text() {
            body.text
                .as_deref()
                .or_else(|| {
                    body.prompt.as_ref().and_then(|p| match p {
                        StringOrArray::String(s) => Some(s.as_str()),
                        StringOrArray::Array(v) => v.first().map(|s| s.as_str()),
                    })
                })
                .map(|s| s.to_string())
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_generate_batch_size(body);

        // Create context
        let context = PDRequestContext {
            route: "/generate",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
        };

        // Execute with retry and bootstrap injection
        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Extract parameters
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        // Extract text for cache-aware routing
        let request_text = if self.policies_need_request_text() {
            body.messages.first().and_then(|msg| match msg {
                ChatMessage::User { content, .. } => match content {
                    UserMessageContent::Text(text) => Some(text.clone()),
                    UserMessageContent::Parts(_) => None,
                },
                ChatMessage::System { content, .. } => Some(content.clone()),
                _ => None,
            })
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_chat_batch_size(body);

        // Create context
        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
        };

        // Execute with retry and bootstrap injection
        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Extract parameters
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        // Extract text for cache-aware routing
        let request_text = if self.policies_need_request_text() {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_completion_batch_size(body);

        // Create context
        let context = PDRequestContext {
            route: "/v1/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
        };

        // Execute with retry and bootstrap injection
        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses endpoint not implemented for PD router",
        )
            .into_response()
    }

    async fn get_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses retrieve endpoint not implemented for PD router",
        )
            .into_response()
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Responses cancel endpoint not implemented for PD router",
        )
            .into_response()
    }

    async fn route_embeddings(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &crate::protocols::spec::EmbeddingRequest,
        _model_id: Option<&str>,
    ) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "Embeddings endpoint not implemented for PD router",
        )
            .into_response()
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        // Extract text for cache-aware routing
        let req_text = if self.policies_need_request_text() {
            Some(body.query.clone())
        } else {
            None
        };

        // Create context
        let context = PDRequestContext {
            route: "/v1/rerank",
            batch_size: None,
            is_stream: false,
            return_logprob: false,
            request_text: req_text,
            model_id,
        };

        // Execute with retry and bootstrap injection
        self.execute_dual_dispatch(headers, body, context).await
    }

    async fn flush_cache(&self) -> Response {
        // Process both prefill and decode workers
        let (prefill_results, prefill_errors) = self
            .process_workers(
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
                "Prefill",
                "flush_cache",
            )
            .await;
        let (decode_results, decode_errors) = self
            .process_workers(WorkerType::Decode, "Decode", "flush_cache")
            .await;

        // Combine results and errors
        let mut results = prefill_results;
        results.extend(decode_results);
        let mut errors = prefill_errors;
        errors.extend(decode_errors);

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!("Cache flushed successfully: {:?}", results),
            )
                .into_response()
        } else {
            (
                StatusCode::PARTIAL_CONTENT,
                format!(
                    "Partial success. Results: {:?}, Errors: {:?}",
                    results, errors
                ),
            )
                .into_response()
        }
    }

    async fn get_worker_loads(&self) -> Response {
        let mut loads = HashMap::new();
        let mut errors = Vec::new();

        // Process prefill workers
        let prefill_urls = self.get_prefill_worker_urls();
        for worker_url in prefill_urls {
            match get_worker_load(&self.client, &worker_url).await {
                Some(load) => {
                    loads.insert(format!("prefill_{}", worker_url), load);
                }
                None => {
                    errors.push(format!("Failed to get load from prefill {}", worker_url));
                }
            }
        }

        // Process decode workers
        let decode_urls = self.get_decode_worker_urls();
        for worker_url in decode_urls {
            match get_worker_load(&self.client, &worker_url).await {
                Some(load) => {
                    loads.insert(format!("decode_{}", worker_url), load);
                }
                None => {
                    errors.push(format!("Failed to get load from decode {}", worker_url));
                }
            }
        }

        let response_data = serde_json::json!({
            "loads": loads,
            "errors": errors
        });

        (StatusCode::OK, Json(response_data)).into_response()
    }

    fn router_type(&self) -> &'static str {
        "pd"
    }

    fn readiness(&self) -> Response {
        // PD router is ready if it has at least one healthy prefill AND one healthy decode worker
        let prefill_workers = self.worker_registry.get_prefill_workers();
        let decode_workers = self.worker_registry.get_decode_workers();

        let healthy_prefill_count = prefill_workers.iter().filter(|w| w.is_healthy()).count();

        let healthy_decode_count = decode_workers.iter().filter(|w| w.is_healthy()).count();

        let total_prefill = prefill_workers.len();
        let total_decode = decode_workers.len();

        if healthy_prefill_count > 0 && healthy_decode_count > 0 {
            Json(json!({
                "status": "ready",
                "prefill": {
                    "healthy": healthy_prefill_count,
                    "total": total_prefill
                },
                "decode": {
                    "healthy": healthy_decode_count,
                    "total": total_decode
                }
            }))
            .into_response()
        } else {
            let mut reasons = Vec::new();
            if healthy_prefill_count == 0 {
                reasons.push("no healthy prefill workers");
            }
            if healthy_decode_count == 0 {
                reasons.push("no healthy decode workers");
            }

            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "status": "not_ready",
                    "reason": reasons.join(", "),
                    "prefill": {
                        "healthy": healthy_prefill_count,
                        "total": total_prefill
                    },
                    "decode": {
                        "healthy": healthy_decode_count,
                        "total": total_decode
                    }
                })),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::WorkerType;

    fn create_test_pd_router() -> PDRouter {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        PDRouter {
            worker_registry,
            policy_registry,
            worker_startup_timeout_secs: 5,
            worker_startup_check_interval_secs: 1,
            worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
            load_monitor_handle: None,
            client: Client::new(),
            prefill_client: Client::new(),
            prefill_drain_tx: mpsc::channel(100).0,
            retry_config: RetryConfig::default(),
            circuit_breaker_config: CircuitBreakerConfig::default(),
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .build();
        worker.set_healthy(healthy);
        Box::new(worker)
    }

    // ============= Worker Management Tests =============

    #[tokio::test]
    async fn test_add_prefill_server_already_exists() {
        let router = create_test_pd_router();

        // Add a worker first
        let worker = create_test_worker(
            "http://localhost:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: Some(8080),
            },
            true,
        );
        router.worker_registry.register(Arc::from(worker));

        // Try to add the same URL again - this would fail during health check in real scenario
        // For unit test, we test the duplicate check logic
        let exists = router
            .worker_registry
            .get_by_url("http://localhost:8000")
            .is_some();
        assert!(exists);
    }

    #[tokio::test]
    async fn test_remove_prefill_server_success() {
        let router = create_test_pd_router();

        // Add servers first
        let worker1 = create_test_worker(
            "http://worker1".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let worker2 = create_test_worker(
            "http://worker2".to_string(),
            WorkerType::Prefill {
                bootstrap_port: Some(8080),
            },
            true,
        );

        router.worker_registry.register(Arc::from(worker1));
        router.worker_registry.register(Arc::from(worker2));

        // Remove one
        let result = router.remove_prefill_server("http://worker1").await;

        assert!(result.is_ok());
        assert!(result.unwrap().contains("Successfully removed"));

        let workers = router.worker_registry.get_prefill_workers();
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].url(), "http://worker2");
    }

    #[tokio::test]
    async fn test_remove_prefill_server_not_found() {
        let router = create_test_pd_router();

        let result = router.remove_prefill_server("http://nonexistent").await;

        assert!(result.is_err());
        match result.unwrap_err() {
            PDRouterError::WorkerNotFound { url } => {
                assert_eq!(url, "http://nonexistent");
            }
            _ => panic!("Expected WorkerNotFound error"),
        }
    }

    #[tokio::test]
    async fn test_remove_decode_server_success() {
        let router = create_test_pd_router();

        // Add server first
        let worker = create_test_worker("http://decode1".to_string(), WorkerType::Decode, true);
        router.worker_registry.register(Arc::from(worker));

        let result = router.remove_decode_server("http://decode1").await;

        assert!(result.is_ok());
        assert!(result.unwrap().contains("Successfully removed"));

        let workers = router.worker_registry.get_decode_workers();
        assert_eq!(workers.len(), 0);
    }

    // ============= Lock Error Handling Tests =============

    #[test]
    fn test_registry_operations() {
        let router = create_test_pd_router();

        // Test registry operations
        let workers = router.worker_registry.get_all();
        assert_eq!(workers.len(), 0);

        // Add a worker
        let worker = create_test_worker(
            "http://test".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        router.worker_registry.register(Arc::from(worker));

        let workers = router.worker_registry.get_all();
        assert_eq!(workers.len(), 1);

        let prefill_workers = router.worker_registry.get_prefill_workers();
        assert_eq!(prefill_workers.len(), 1);
    }

    // ============= Bootstrap Injection Tests =============
    // Note: These tests are commented out as we've moved to the optimized bootstrap injection
    // approach that doesn't use the Bootstrap trait on GenerateReqInput anymore.

    // TODO: Add new tests for the optimized bootstrap injection approach using
    // RequestWithBootstrap and BatchRequestWithBootstrap wrappers

    // ============= Worker Selection Tests =============

    #[tokio::test]
    async fn test_select_healthy_prefill_worker() {
        let router = create_test_pd_router();

        // Add mix of healthy and unhealthy workers
        let healthy_worker = create_test_worker(
            "http://healthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let unhealthy_worker = create_test_worker(
            "http://unhealthy".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            false,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(unhealthy_worker));
        router.worker_registry.register(Arc::from(healthy_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let result = router.select_pd_pair(None, None).await;

        assert!(result.is_ok());
        let (prefill, _decode) = result.unwrap();

        // Should select the healthy worker
        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let result = router.select_pd_pair(None, None).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
    }

    // ============= Health Endpoints Tests =============

    #[tokio::test]
    async fn test_health_endpoints() {
        let router = create_test_pd_router();

        // Add healthy workers - create_test_worker returns Box<dyn Worker>, convert to Arc
        let prefill_worker = create_test_worker(
            "http://localhost:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode_worker = create_test_worker(
            "http://localhost:8001".to_string(),
            WorkerType::Decode,
            true,
        );

        router.worker_registry.register(Arc::from(prefill_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        // Test health endpoint
        let http_req = axum::http::Request::builder()
            .body(axum::body::Body::empty())
            .unwrap();
        let response = router.health(http_req).await;

        assert_eq!(response.status(), 200);

        // Test readiness endpoint
        let response = router.readiness();
        assert_eq!(response.status(), 200);
    }

    // ============= Load Monitoring Tests =============

    #[tokio::test]
    async fn test_load_monitor_updates() {
        let power_of_two_policy = Arc::new(crate::policies::PowerOfTwoPolicy::new());
        let mut router = create_test_pd_router();
        // Set power_of_two policies in the registry
        router
            .policy_registry
            .set_prefill_policy(power_of_two_policy.clone());
        router
            .policy_registry
            .set_decode_policy(power_of_two_policy);

        // Create load channel
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        router.worker_loads = Arc::new(rx);

        // Simulate load updates
        let mut loads = HashMap::new();
        loads.insert("http://worker1".to_string(), 10);
        loads.insert("http://worker2".to_string(), 5);

        let _ = tx.send(loads.clone());

        // Router should receive updates
        let received = router.worker_loads.borrow().clone();
        assert_eq!(received.get("http://worker1"), Some(&10));
        assert_eq!(received.get("http://worker2"), Some(&5));
    }

    // ============= Worker Load Tests =============

    #[test]
    fn test_worker_load_metrics() {
        let prefill_worker = create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        // Create load guard for both workers
        let _guard =
            WorkerLoadGuard::new_multi(vec![prefill_worker.as_ref(), decode_worker.as_ref()]);

        // Load should be incremented
        assert_eq!(prefill_worker.load(), 1);
        assert_eq!(decode_worker.load(), 1);

        // Drop guard - load should decrement
        drop(_guard);

        assert_eq!(prefill_worker.load(), 0);
        assert_eq!(decode_worker.load(), 0);
    }

    #[tokio::test]
    async fn test_streaming_load_tracking() {
        use futures_util::StreamExt;
        use tokio::time::{sleep, Duration};

        let router = create_test_pd_router();

        // Add workers - create_test_worker returns Box<dyn Worker>, convert to Arc
        let prefill_worker = create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(prefill_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        // Get references to the workers from registry
        let prefill_workers = router.worker_registry.get_prefill_workers();
        let decode_workers = router.worker_registry.get_decode_workers();

        let prefill_ref = prefill_workers[0].clone();
        let decode_ref = decode_workers[0].clone();

        // Initially load should be 0
        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);

        // Create a mock streaming response
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

        // Call create_streaming_response which should increment load
        let _response = router.create_streaming_response(
            stream.map(Ok),
            StatusCode::OK,
            None,
            false,
            None,
            None,
            prefill_ref.as_ref(),
            decode_ref.as_ref(),
        );

        // Load should be incremented immediately
        assert_eq!(prefill_ref.load(), 1);
        assert_eq!(decode_ref.load(), 1);

        // Send some data through the stream
        tx.send(bytes::Bytes::from("test data")).unwrap();

        // Give time for the spawned task to process
        sleep(Duration::from_millis(10)).await;

        // Load should still be 1 (streaming in progress)
        assert_eq!(prefill_ref.load(), 1);
        assert_eq!(decode_ref.load(), 1);

        // Close the stream
        drop(tx);

        // Give time for cleanup
        sleep(Duration::from_millis(100)).await;

        // Load should be decremented after streaming completes
        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);
    }

    // ============= Concurrent Operations Tests =============

    #[tokio::test]
    async fn test_concurrent_worker_operations() {
        let router = Arc::new(create_test_pd_router());

        let mut handles = vec![];

        // Spawn tasks to add workers
        for i in 0..5 {
            let router_clone = Arc::clone(&router);
            let url = format!("http://worker{}", i);
            let handle = tokio::spawn(async move {
                let worker = create_test_worker(
                    url,
                    WorkerType::Prefill {
                        bootstrap_port: None,
                    },
                    true,
                );
                router_clone.worker_registry.register(Arc::from(worker));
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            let _ = handle.await;
        }

        // Check final state
        let workers = router.worker_registry.get_prefill_workers();
        assert_eq!(workers.len(), 5);
    }
}
