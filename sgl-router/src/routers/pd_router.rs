// PD (Prefill-Decode) Router Implementation
// This module handles routing for disaggregated prefill-decode systems

use super::bootstrap_injector::inject_bootstrap_fields;
use super::pd_types::{api_path, PDRouterError};
use crate::config::types::RetryConfig;
use crate::core::{HealthChecker, Worker, WorkerFactory, WorkerLoadGuard};
use crate::metrics::RouterMetrics;
use crate::openai_api_types::{ChatCompletionRequest, CompletionRequest, GenerateRequest};
use crate::policies::LoadBalancingPolicy;
use crate::routers::{RouterTrait, WorkerManagement};
use crate::tree::Tree;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::StreamExt;
use rand::Rng;
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn};

#[derive(Debug)]
pub struct PDRouter {
    pub prefill_workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
    pub decode_workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
    pub prefill_policy: Arc<dyn LoadBalancingPolicy>,
    pub decode_policy: Arc<dyn LoadBalancingPolicy>,
    pub prefill_tree: Option<Arc<Mutex<Tree>>>,
    pub decode_tree: Option<Arc<Mutex<Tree>>>,
    pub timeout_secs: u64,
    pub interval_secs: u64,
    pub worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, isize>>>,
    pub load_monitor_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    pub client: Client,
    pub retry_config: RetryConfig,
    _prefill_health_checker: Option<HealthChecker>,
    _decode_health_checker: Option<HealthChecker>,
}

impl PDRouter {
    // Dynamic worker management methods for service discovery

    // Private helper method to perform health check on a new server
    async fn wait_for_server_health(&self, url: &str) -> Result<(), PDRouterError> {
        crate::routers::router::Router::wait_for_healthy_workers(
            &[url.to_string()],
            self.timeout_secs,
            self.interval_secs,
        )
        .map_err(|_| PDRouterError::HealthCheckFailed {
            url: url.to_string(),
        })
    }

    pub async fn add_prefill_server(
        &self,
        url: String,
        bootstrap_port: Option<u16>,
    ) -> Result<String, PDRouterError> {
        // Wait for the new server to be healthy
        self.wait_for_server_health(&url).await?;

        // Create Worker for the new prefill server
        let worker = WorkerFactory::create_prefill(url.clone(), bootstrap_port);

        // Add to prefill workers list
        let mut workers = self
            .prefill_workers
            .write()
            .map_err(|_| PDRouterError::LockError {
                operation: "prefill_workers write".to_string(),
            })?;

        // Check if already exists
        if workers.iter().any(|w| w.url() == &url) {
            return Err(PDRouterError::WorkerAlreadyExists { url: url.clone() });
        }

        workers.push(worker);

        // Add to cache tree if using cache-aware policy for prefill
        if let Some(ref tree) = self.prefill_tree {
            tree.lock().unwrap().insert("", &url);
        }

        info!("Added prefill server: {}", url);
        Ok(format!("Successfully added prefill server: {}", url))
    }

    pub async fn add_decode_server(&self, url: String) -> Result<String, PDRouterError> {
        // Wait for the new server to be healthy
        self.wait_for_server_health(&url).await?;

        // Create Worker for the new decode server
        let worker = WorkerFactory::create_decode(url.clone());

        // Add to decode workers list
        let mut workers = self
            .decode_workers
            .write()
            .map_err(|_| PDRouterError::LockError {
                operation: "decode_workers write".to_string(),
            })?;

        // Check if already exists
        if workers.iter().any(|w| w.url() == &url) {
            return Err(PDRouterError::WorkerAlreadyExists { url: url.clone() });
        }

        workers.push(worker);

        // Add to cache tree if using cache-aware policy for decode
        if let Some(ref tree) = self.decode_tree {
            tree.lock().unwrap().insert("", &url);
        }

        info!("Added decode server: {}", url);
        Ok(format!("Successfully added decode server: {}", url))
    }

    pub async fn remove_prefill_server(&self, url: &str) -> Result<String, PDRouterError> {
        let mut workers = self
            .prefill_workers
            .write()
            .map_err(|_| PDRouterError::LockError {
                operation: "prefill_workers write".to_string(),
            })?;

        // Find and remove the server
        let initial_len = workers.len();
        workers.retain(|w| w.url() != url);

        if workers.len() == initial_len {
            return Err(PDRouterError::WorkerNotFound {
                url: url.to_string(),
            });
        }

        // Remove from cache tree if using cache-aware policy
        if let Some(ref tree) = self.prefill_tree {
            tree.lock().unwrap().remove_tenant(url);
        }

        info!("Removed prefill server: {}", url);
        Ok(format!("Successfully removed prefill server: {}", url))
    }

    pub async fn remove_decode_server(&self, url: &str) -> Result<String, PDRouterError> {
        let mut workers = self
            .decode_workers
            .write()
            .map_err(|_| PDRouterError::LockError {
                operation: "decode_workers write".to_string(),
            })?;

        // Find and remove the server
        let initial_len = workers.len();
        workers.retain(|w| w.url() != url);

        if workers.len() == initial_len {
            return Err(PDRouterError::WorkerNotFound {
                url: url.to_string(),
            });
        }

        // Remove from the cache tree if using cache-aware policy for decode
        if let Some(ref tree) = self.decode_tree {
            tree.lock().unwrap().remove_tenant(url);
        }

        info!("Removed decode server: {}", url);
        Ok(format!("Successfully removed decode server: {}", url))
    }

    pub fn new(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        prefill_policy: Arc<dyn LoadBalancingPolicy>,
        decode_policy: Arc<dyn LoadBalancingPolicy>,
        client: Client,
        timeout_secs: u64,
        interval_secs: u64,
        retry_config: RetryConfig,
    ) -> Result<Self, String> {
        // Convert URLs to Worker trait objects
        let prefill_workers: Vec<Box<dyn Worker>> = prefill_urls
            .into_iter()
            .map(|(url, port)| WorkerFactory::create_prefill(url, port))
            .collect();

        let decode_workers: Vec<Box<dyn Worker>> = decode_urls
            .into_iter()
            .map(WorkerFactory::create_decode)
            .collect();

        // Wait for PD workers to be healthy (skip if empty - for service discovery mode)
        let all_urls: Vec<String> = prefill_workers
            .iter()
            .chain(decode_workers.iter())
            .map(|worker| worker.url().to_string())
            .collect();
        if !all_urls.is_empty() {
            crate::routers::router::Router::wait_for_healthy_workers(
                &all_urls,
                timeout_secs,
                interval_secs,
            )?;
        }

        // Initialize cache-aware components if needed for prefill policy
        let prefill_tree = Self::initialize_radix_tree(&prefill_policy, &prefill_workers)?;

        // Initialize cache-aware components if needed for decode policy
        let decode_tree = Self::initialize_radix_tree(&decode_policy, &decode_workers)?;

        // Set up background load monitoring for power-of-two selection
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let worker_loads = Arc::new(rx);

        let load_monitor_handle =
            if prefill_policy.name() == "power_of_two" || decode_policy.name() == "power_of_two" {
                let monitor_urls = all_urls.clone();
                let monitor_interval = interval_secs;
                let monitor_client = client.clone();
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

        let prefill_workers = Arc::new(RwLock::new(prefill_workers));
        let decode_workers = Arc::new(RwLock::new(decode_workers));

        // Start health checkers for both worker pools
        let prefill_health_checker =
            crate::core::start_health_checker(Arc::clone(&prefill_workers), interval_secs);
        let decode_health_checker =
            crate::core::start_health_checker(Arc::clone(&decode_workers), interval_secs);

        Ok(PDRouter {
            prefill_workers,
            decode_workers,
            prefill_policy,
            decode_policy,
            prefill_tree,
            decode_tree,
            timeout_secs,
            interval_secs,
            worker_loads,
            load_monitor_handle,
            client,
            retry_config,
            _prefill_health_checker: Some(prefill_health_checker),
            _decode_health_checker: Some(decode_health_checker),
        })
    }

    // Helper function to initialize radix tree for cache-aware policies
    fn initialize_radix_tree(
        policy: &Arc<dyn LoadBalancingPolicy>,
        workers: &[Box<dyn Worker>],
    ) -> Result<Option<Arc<Mutex<Tree>>>, String> {
        if let Some(cache_policy) = policy
            .as_any()
            .downcast_ref::<crate::policies::CacheAwarePolicy>()
        {
            // Initialize the policy's internal tree with workers
            cache_policy.init_workers(workers);

            let tree = Arc::new(Mutex::new(Tree::new()));

            {
                let tree_guard = tree
                    .lock()
                    .map_err(|e| format!("Failed to lock tree: {}", e))?;
                for worker in workers {
                    tree_guard.insert("", worker.url());
                }
            }

            Ok(Some(tree))
        } else {
            Ok(None)
        }
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

    // Helper to handle bootstrap injection errors
    fn handle_bootstrap_error(error: impl std::fmt::Display) -> Response {
        error!("Failed to add bootstrap info error={}", error);
        RouterMetrics::record_pd_error("bootstrap_injection");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Bootstrap injection failed: {}", error),
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

    // Execute the dual dispatch to prefill and decode servers with retry logic
    async fn execute_dual_dispatch(
        &self,
        headers: Option<&HeaderMap>,
        json_request: Value,
        route: &str,
        prefill: &dyn Worker,
        decode: &dyn Worker,
        is_stream: bool,
        return_logprob: bool,
        start_time: Instant,
    ) -> Response {
        for attempt in 0..self.retry_config.max_retries {
            if attempt > 0 {
                // Calculate backoff with exponential growth and jitter
                let base_backoff = self.retry_config.initial_backoff_ms as f64
                    * self
                        .retry_config
                        .backoff_multiplier
                        .powf((attempt - 1) as f32) as f64;
                let backoff_ms = base_backoff.min(self.retry_config.max_backoff_ms as f64) as u64;

                // Add jitter to prevent thundering herd
                let jitter = {
                    let mut rng = rand::thread_rng();
                    rng.gen_range(0..backoff_ms / 2)
                };
                let total_backoff = Duration::from_millis(backoff_ms + jitter);

                info!(
                    "Retrying request (attempt {}/{}) after {:?} backoff",
                    attempt + 1,
                    self.retry_config.max_retries,
                    total_backoff
                );

                tokio::time::sleep(total_backoff).await;
            }

            debug!(
                "Executing request attempt {}/{}",
                attempt + 1,
                self.retry_config.max_retries
            );
            let result = self
                .execute_dual_dispatch_inner(
                    headers,
                    json_request.clone(),
                    route,
                    prefill,
                    decode,
                    is_stream,
                    return_logprob,
                    start_time,
                )
                .await;

            // Check if we should retry based on the response status
            let status = result.status();
            debug!(
                "Request attempt {} returned status: {}",
                attempt + 1,
                status
            );

            // Don't retry client errors (4xx) or successful responses
            if status.is_client_error() || status.is_success() {
                debug!(
                    "Returning response with status {} (no retry needed)",
                    status
                );
                return result;
            }

            // Check if this is the last attempt
            if attempt == self.retry_config.max_retries - 1 {
                warn!("Final attempt failed with status {}", status);
                return result;
            }

            // Log retry decision for retryable errors
            if status.is_server_error()
                || status == StatusCode::BAD_GATEWAY
                || status == StatusCode::GATEWAY_TIMEOUT
            {
                warn!(
                    "Retryable error status: {} on attempt {}/{}. Will retry.",
                    status,
                    attempt + 1,
                    self.retry_config.max_retries
                );
            } else {
                // Don't retry other statuses
                debug!("Status {} is not retryable, returning response", status);
                return result;
            }
        }

        // This should never be reached due to the loop logic, but just in case
        unreachable!("Retry loop completed without returning")
    }

    // Inner implementation of dual dispatch (extracted for retry logic)
    async fn execute_dual_dispatch_inner(
        &self,
        headers: Option<&HeaderMap>,
        json_request: Value,
        route: &str,
        prefill: &dyn Worker,
        decode: &dyn Worker,
        is_stream: bool,
        return_logprob: bool,
        start_time: Instant,
    ) -> Response {
        // Update load tracking for both workers
        let _guard = WorkerLoadGuard::new_multi(vec![prefill, decode]);

        // Build requests with headers
        let prefill_request =
            self.build_request_with_headers(prefill.url(), route, &json_request, headers);

        let decode_request =
            self.build_request_with_headers(decode.url(), route, &json_request, headers);

        // Send both requests concurrently
        debug!(
            "Sending concurrent requests to prefill={} decode={}",
            prefill.url(),
            decode.url()
        );
        let (prefill_result, decode_result) =
            tokio::join!(prefill_request.send(), decode_request.send());
        debug!("Received responses from both servers");

        // Update metrics
        let duration = start_time.elapsed();
        RouterMetrics::record_pd_request_duration(route, duration);
        RouterMetrics::record_pd_request(route);
        RouterMetrics::record_pd_prefill_request(prefill.url());
        RouterMetrics::record_pd_decode_request(decode.url());

        // Process prefill response
        let (_prefill_status, prefill_body) = match self
            .process_prefill_response(prefill_result, prefill.url(), return_logprob)
            .await
        {
            Ok(result) => result,
            Err(error_response) => return error_response,
        };

        // Process decode response
        debug!("Processing decode response");
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

                    // Return the error response from decode server
                    match res.bytes().await {
                        Ok(error_body) => {
                            return (status, error_body).into_response();
                        }
                        Err(e) => {
                            return (status, format!("Decode server error: {}", e)).into_response();
                        }
                    }
                }

                if is_stream {
                    // Streaming response
                    let prefill_logprobs = if return_logprob {
                        prefill_body
                            .as_ref()
                            .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                            .and_then(|json| {
                                json.pointer("/meta_info/input_token_logprobs").cloned()
                            })
                    } else {
                        None
                    };

                    let decode_url = if !return_logprob {
                        Some(decode.url().to_string())
                    } else {
                        None
                    };

                    Self::create_streaming_response(
                        res.bytes_stream(),
                        status,
                        prefill_logprobs,
                        return_logprob,
                        decode_url,
                    )
                } else {
                    // Non-streaming response - use helper
                    self.process_non_streaming_response(res, status, return_logprob, prefill_body)
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
    }

    // Select a pair of prefill and decode servers
    async fn select_pd_pair(
        &self,
        request_text: Option<&str>,
    ) -> Result<(Box<dyn Worker>, Box<dyn Worker>), String> {
        // Get read locks for both worker lists
        let prefill_workers = self
            .prefill_workers
            .read()
            .map_err(|e| format!("Failed to acquire prefill workers lock: {}", e))?;
        let decode_workers = self
            .decode_workers
            .read()
            .map_err(|e| format!("Failed to acquire decode workers lock: {}", e))?;

        // Check we have workers
        if prefill_workers.is_empty() {
            return Err("No prefill workers available. Please check if prefill servers are configured and healthy.".to_string());
        }
        if decode_workers.is_empty() {
            return Err("No decode workers available. Please check if decode servers are configured and healthy.".to_string());
        }

        // Select prefill worker using prefill policy
        let prefill_idx = self
            .prefill_policy
            .select_worker(&prefill_workers, request_text)
            .ok_or("Failed to select prefill worker")?;

        // Select decode worker using decode policy
        let decode_idx = self
            .decode_policy
            .select_worker(&decode_workers, request_text)
            .ok_or("Failed to select decode worker")?;

        let prefill = prefill_workers[prefill_idx].clone_worker();
        let decode = decode_workers[decode_idx].clone_worker();
        Ok((prefill, decode))
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
    fn create_streaming_response(
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        decode_url: Option<String>,
    ) -> Response {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            futures_util::pin_mut!(stream);
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
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
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        let mut response = Response::new(body);
        *response.status_mut() = status;
        response
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
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
        match res.bytes().await {
            Ok(decode_body) => {
                if return_logprob && prefill_body.is_some() {
                    // Merge logprobs from prefill and decode
                    let prefill_body = prefill_body.as_ref().unwrap();
                    match (
                        serde_json::from_slice::<Value>(prefill_body),
                        serde_json::from_slice::<Value>(&decode_body),
                    ) {
                        (Ok(prefill_json), Ok(mut decode_json)) => {
                            // Use helper to merge logprobs
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
                        _ => {
                            // If parsing fails, just return decode response
                            warn!("Failed to parse responses for logprob merging");
                            (status, decode_body).into_response()
                        }
                    }
                } else {
                    (status, decode_body).into_response()
                }
            }
            Err(e) => {
                error!("Failed to read decode response: {}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "Failed to read response").into_response()
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

    // Helper to build a request with headers copied from the original request
    fn build_request_with_headers(
        &self,
        url: &str,
        route: &str,
        json_request: &Value,
        headers: Option<&HeaderMap>,
    ) -> reqwest::RequestBuilder {
        let mut request = self.client.post(api_path(url, route)).json(json_request);

        // Copy headers from original request (excluding content-type and content-length which are set by .json())
        if let Some(headers) = headers {
            for (name, value) in headers.iter() {
                let name_str = name.as_str();
                if name_str != "content-type" && name_str != "content-length" {
                    // Skip headers with non-ASCII values
                    if value.to_str().is_ok() {
                        request = request.header(name, value);
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
        // For PD router, we would need to know if it's a prefill or decode server
        // For now, try both
        if let Ok(mut workers) = self.prefill_workers.write() {
            if let Some(index) = workers.iter().position(|w| w.url() == worker_url) {
                workers.remove(index);
                info!("Removed prefill worker: {}", worker_url);
                return;
            }
        }

        if let Ok(mut workers) = self.decode_workers.write() {
            if let Some(index) = workers.iter().position(|w| w.url() == worker_url) {
                workers.remove(index);
                info!("Removed decode worker: {}", worker_url);
            }
        }
    }

    fn get_worker_urls(&self) -> Vec<String> {
        let mut urls = Vec::new();

        // Add prefill worker URLs
        if let Ok(workers) = self.prefill_workers.read() {
            for worker in workers.iter() {
                urls.push(worker.url().to_string());
            }
        }

        // Add decode worker URLs
        if let Ok(workers) = self.decode_workers.read() {
            for worker in workers.iter() {
                urls.push(worker.url().to_string());
            }
        }

        urls
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

        // Check prefill servers
        for worker in self.prefill_workers.read().unwrap().iter() {
            if !worker.is_healthy() {
                all_healthy = false;
                unhealthy_servers.push(format!("Prefill: {}", worker.url()));
            }
        }

        // Check decode servers
        for worker in self.decode_workers.read().unwrap().iter() {
            if !worker.is_healthy() {
                all_healthy = false;
                unhealthy_servers.push(format!("Decode: {}", worker.url()));
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
        let (prefill, decode) = match self.select_pd_pair(None).await {
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
        let prefill_result = self.client.get(&prefill_url).send().await;

        // Test decode server's health_generate
        let decode_url = format!("{}/health_generate", decode.url());
        let decode_result = self.client.get(&decode_url).send().await;

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
        let first_decode_url = if let Ok(workers) = self.decode_workers.read() {
            workers.first().map(|w| w.url().to_string())
        } else {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to access decode workers",
            )
                .into_response();
        };

        if let Some(worker_url) = first_decode_url {
            match self
                .client
                .get(format!("{}/get_server_info", worker_url))
                .send()
                .await
            {
                Ok(res) if res.status().is_success() => {
                    match res.json::<Value>().await {
                        Ok(info) => {
                            // The decode server should already return the proper format
                            // with tokenizer_path and other fields that bench_one_batch_server.py expects
                            Json(info).into_response()
                        }
                        Err(e) => {
                            error!("Failed to parse server info: {}", e);
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                format!("Failed to parse server info: {}", e),
                            )
                                .into_response()
                        }
                    }
                }
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    (
                        status,
                        format!("Decode server returned status: {}", res.status()),
                    )
                        .into_response()
                }
                Err(e) => {
                    error!("Failed to get server info: {}", e);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to get server info: {}", e),
                    )
                        .into_response()
                }
            }
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No decode servers available",
            )
                .into_response()
        }
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = crate::routers::router::copy_request_headers(&req);

        // Get first prefill worker URL to avoid holding lock across await
        let first_worker_url = if let Ok(workers) = self.prefill_workers.read() {
            workers.first().map(|w| w.url().to_string())
        } else {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to access prefill workers",
            )
                .into_response();
        };

        if let Some(worker_url) = first_worker_url {
            let url = format!("{}/v1/models", worker_url);
            let mut request_builder = self.client.get(&url);

            // Add headers
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }

            match request_builder.send().await {
                Ok(res) if res.status().is_success() => match res.bytes().await {
                    Ok(body) => (StatusCode::OK, body).into_response(),
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read response body: {}", e),
                        )
                            .into_response()
                    }
                },
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    (
                        status,
                        format!("Prefill server returned status: {}", res.status()),
                    )
                        .into_response()
                }
                Err(e) => {
                    error!("Failed to get models: {}", e);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to get models: {}", e),
                    )
                        .into_response()
                }
            }
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No prefill servers available",
            )
                .into_response()
        }
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = crate::routers::router::copy_request_headers(&req);

        // Get first prefill worker URL to avoid holding lock across await
        let first_worker_url = if let Ok(workers) = self.prefill_workers.read() {
            workers.first().map(|w| w.url().to_string())
        } else {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to access prefill workers",
            )
                .into_response();
        };

        if let Some(worker_url) = first_worker_url {
            let url = format!("{}/get_model_info", worker_url);
            let mut request_builder = self.client.get(&url);

            // Add headers
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }

            match request_builder.send().await {
                Ok(res) if res.status().is_success() => match res.bytes().await {
                    Ok(body) => (StatusCode::OK, body).into_response(),
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read response body: {}", e),
                        )
                            .into_response()
                    }
                },
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    (
                        status,
                        format!("Prefill server returned status: {}", res.status()),
                    )
                        .into_response()
                }
                Err(e) => {
                    error!("Failed to get model info: {}", e);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to get model info: {}", e),
                    )
                        .into_response()
                }
            }
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No prefill servers available",
            )
                .into_response()
        }
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
    ) -> Response {
        let start = Instant::now();

        // Convert directly to JSON to preserve all fields automatically
        let mut json = match serde_json::to_value(body) {
            Ok(json) => json,
            Err(e) => return Self::handle_serialization_error(e),
        };

        // Extract flags for routing logic
        let is_stream = body.stream;
        let return_logprob = body.return_logprob;

        // Extract text for cache-aware routing
        let request_text = body.text.as_deref().or_else(|| {
            body.prompt.as_ref().and_then(|p| match p {
                crate::openai_api_types::StringOrArray::String(s) => Some(s.as_str()),
                crate::openai_api_types::StringOrArray::Array(v) => v.first().map(|s| s.as_str()),
            })
        });

        // Select servers
        let (prefill, decode) = match self.select_pd_pair(request_text).await {
            Ok(pair) => pair,
            Err(e) => return Self::handle_server_selection_error(e),
        };

        // Log routing decision
        info!(
            "PD routing decision route=/generate prefill_url={} decode_url={}",
            prefill.url(),
            decode.url()
        );

        // Inject bootstrap fields directly into JSON
        if let Err(e) = inject_bootstrap_fields(&mut json, prefill.as_ref()) {
            return Self::handle_bootstrap_error(e);
        }

        // Execute dual dispatch
        self.execute_dual_dispatch(
            headers,
            json,
            "/generate",
            prefill.as_ref(),
            decode.as_ref(),
            is_stream,
            return_logprob,
            start,
        )
        .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
    ) -> Response {
        let start = Instant::now();

        // Convert directly to JSON to preserve all fields automatically
        let mut json = match serde_json::to_value(body) {
            Ok(json) => json,
            Err(e) => return Self::handle_serialization_error(e),
        };

        // Extract flags for routing logic
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        // Extract text for cache-aware routing from chat messages
        let request_text = body.messages.first().and_then(|msg| match msg {
            crate::openai_api_types::ChatMessage::User { content, .. } => {
                match content {
                    crate::openai_api_types::UserMessageContent::Text(text) => Some(text.as_str()),
                    crate::openai_api_types::UserMessageContent::Parts(_) => None, // Skip complex content
                }
            }
            crate::openai_api_types::ChatMessage::System { content, .. } => Some(content.as_str()),
            _ => None,
        });

        // Select servers
        let (prefill, decode) = match self.select_pd_pair(request_text).await {
            Ok(pair) => pair,
            Err(e) => return Self::handle_server_selection_error(e),
        };

        // Log routing decision
        info!(
            "PD routing decision route=/v1/chat/completions prefill_url={} decode_url={}",
            prefill.url(),
            decode.url()
        );

        // Inject bootstrap fields directly into JSON
        if let Err(e) = inject_bootstrap_fields(&mut json, prefill.as_ref()) {
            return Self::handle_bootstrap_error(e);
        }

        // Execute dual dispatch
        self.execute_dual_dispatch(
            headers,
            json,
            "/v1/chat/completions",
            prefill.as_ref(),
            decode.as_ref(),
            is_stream,
            return_logprob,
            start,
        )
        .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
    ) -> Response {
        let start = Instant::now();

        // Convert directly to JSON to preserve all fields automatically
        let mut json = match serde_json::to_value(body) {
            Ok(json) => json,
            Err(e) => return Self::handle_serialization_error(e),
        };

        // Extract flags for routing logic
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        // Extract text for cache-aware routing
        let request_text = match &body.prompt {
            crate::openai_api_types::StringOrArray::String(s) => Some(s.as_str()),
            crate::openai_api_types::StringOrArray::Array(v) => v.first().map(|s| s.as_str()),
        };

        // Select servers
        let (prefill, decode) = match self.select_pd_pair(request_text).await {
            Ok(pair) => pair,
            Err(e) => return Self::handle_server_selection_error(e),
        };

        // Log routing decision
        info!(
            "PD routing decision route=/v1/completions prefill_url={} decode_url={}",
            prefill.url(),
            decode.url()
        );

        // Inject bootstrap fields directly into JSON
        if let Err(e) = inject_bootstrap_fields(&mut json, prefill.as_ref()) {
            return Self::handle_bootstrap_error(e);
        }

        // Execute dual dispatch
        self.execute_dual_dispatch(
            headers,
            json,
            "/v1/completions",
            prefill.as_ref(),
            decode.as_ref(),
            is_stream,
            return_logprob,
            start,
        )
        .await
    }

    async fn flush_cache(&self) -> Response {
        let mut results = Vec::new();
        let mut errors = Vec::new();

        // Get prefill worker URLs first to avoid holding lock across await
        let prefill_urls = if let Ok(workers) = self.prefill_workers.read() {
            workers
                .iter()
                .map(|w| w.url().to_string())
                .collect::<Vec<_>>()
        } else {
            errors.push("Failed to access prefill workers".to_string());
            Vec::new()
        };

        // Flush prefill workers
        for worker_url in prefill_urls {
            let url = format!("{}/flush_cache", worker_url);
            match self.client.post(&url).send().await {
                Ok(res) if res.status().is_success() => {
                    results.push(format!("Prefill {}: OK", worker_url));
                }
                Ok(res) => {
                    errors.push(format!(
                        "Prefill {} returned status: {}",
                        worker_url,
                        res.status()
                    ));
                }
                Err(e) => {
                    errors.push(format!("Prefill {} error: {}", worker_url, e));
                }
            }
        }

        // Get decode worker URLs first to avoid holding lock across await
        let decode_urls = if let Ok(workers) = self.decode_workers.read() {
            workers
                .iter()
                .map(|w| w.url().to_string())
                .collect::<Vec<_>>()
        } else {
            errors.push("Failed to access decode workers".to_string());
            Vec::new()
        };

        // Flush decode workers
        for worker_url in decode_urls {
            let url = format!("{}/flush_cache", worker_url);
            match self.client.post(&url).send().await {
                Ok(res) if res.status().is_success() => {
                    results.push(format!("Decode {}: OK", worker_url));
                }
                Ok(res) => {
                    errors.push(format!(
                        "Decode {} returned status: {}",
                        worker_url,
                        res.status()
                    ));
                }
                Err(e) => {
                    errors.push(format!("Decode {} error: {}", worker_url, e));
                }
            }
        }

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

        // Get prefill worker URLs first to avoid holding lock across await
        let prefill_urls = if let Ok(workers) = self.prefill_workers.read() {
            workers
                .iter()
                .map(|w| w.url().to_string())
                .collect::<Vec<_>>()
        } else {
            errors.push("Failed to access prefill workers".to_string());
            Vec::new()
        };

        // Get loads from prefill workers
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

        // Get decode worker URLs first to avoid holding lock across await
        let decode_urls = if let Ok(workers) = self.decode_workers.read() {
            workers
                .iter()
                .map(|w| w.url().to_string())
                .collect::<Vec<_>>()
        } else {
            errors.push("Failed to access decode workers".to_string());
            Vec::new()
        };

        // Get loads from decode workers
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
        let healthy_prefill_count = self
            .prefill_workers
            .read()
            .unwrap()
            .iter()
            .filter(|w| w.is_healthy())
            .count();

        let healthy_decode_count = self
            .decode_workers
            .read()
            .unwrap()
            .iter()
            .filter(|w| w.is_healthy())
            .count();

        let total_prefill = self.prefill_workers.read().unwrap().len();
        let total_decode = self.decode_workers.read().unwrap().len();

        if healthy_prefill_count > 0 && healthy_decode_count > 0 {
            Json(serde_json::json!({
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
    use crate::core::{BasicWorker, WorkerType};
    use crate::policies::{CacheAwarePolicy, RandomPolicy};

    fn create_test_pd_router() -> PDRouter {
        let prefill_policy = Arc::new(RandomPolicy::new());
        let decode_policy = Arc::new(RandomPolicy::new());

        PDRouter {
            prefill_workers: Arc::new(RwLock::new(vec![])),
            decode_workers: Arc::new(RwLock::new(vec![])),
            prefill_policy,
            decode_policy,
            prefill_tree: None,
            decode_tree: None,
            timeout_secs: 5,
            interval_secs: 1,
            worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
            load_monitor_handle: None,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            _prefill_health_checker: None,
            _decode_health_checker: None,
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorker::new(url, worker_type);
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
        router.prefill_workers.write().unwrap().push(worker);

        // Try to add the same URL again - this would fail during health check in real scenario
        // For unit test, we test the duplicate check logic
        let workers = router.prefill_workers.read().unwrap();
        let exists = workers.iter().any(|w| w.url() == "http://localhost:8000");
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

        router.prefill_workers.write().unwrap().push(worker1);
        router.prefill_workers.write().unwrap().push(worker2);

        // Remove one
        let result = router.remove_prefill_server("http://worker1").await;

        assert!(result.is_ok());
        assert!(result.unwrap().contains("Successfully removed"));

        let workers = router.prefill_workers.read().unwrap();
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
        router.decode_workers.write().unwrap().push(worker);

        let result = router.remove_decode_server("http://decode1").await;

        assert!(result.is_ok());
        assert!(result.unwrap().contains("Successfully removed"));

        let workers = router.decode_workers.read().unwrap();
        assert_eq!(workers.len(), 0);
    }

    // ============= Lock Error Handling Tests =============

    #[test]
    fn test_lock_operations() {
        let router = create_test_pd_router();

        // Test read/write locks work correctly
        {
            let read_guard = router.prefill_workers.read().unwrap();
            assert_eq!(read_guard.len(), 0);
        }

        {
            let mut write_guard = router.prefill_workers.write().unwrap();
            write_guard.push(create_test_worker(
                "http://test".to_string(),
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
                true,
            ));
        }

        {
            let read_guard = router.prefill_workers.read().unwrap();
            assert_eq!(read_guard.len(), 1);
        }
    }

    // ============= Cache Tree Integration Tests =============

    #[tokio::test]
    async fn test_cache_tree_operations() {
        let cache_policy = Arc::new(CacheAwarePolicy::new());
        let mut router = create_test_pd_router();
        router.prefill_policy = cache_policy;

        // Initialize cache tree
        let tree = Arc::new(Mutex::new(Tree::new()));
        router.prefill_tree = Some(Arc::clone(&tree));

        // Manually add worker and update tree
        let worker = create_test_worker(
            "http://worker1".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        router.prefill_workers.write().unwrap().push(worker);

        // Update tree
        tree.lock().unwrap().insert("", "http://worker1");

        // Verify tree contains the worker
        let tree_guard = tree.lock().unwrap();
        let (_matched_text, tenant) = tree_guard.prefix_match("");
        // Since we inserted with empty prefix, we should get a match
        assert_eq!(tenant, "http://worker1");
    }

    #[tokio::test]
    async fn test_cache_tree_rebuild_on_remove() {
        let cache_policy = Arc::new(CacheAwarePolicy::new());
        let mut router = create_test_pd_router();
        router.prefill_policy = cache_policy;

        // Initialize cache tree
        let tree = Arc::new(Mutex::new(Tree::new()));
        router.prefill_tree = Some(Arc::clone(&tree));

        // Add multiple workers
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
                bootstrap_port: None,
            },
            true,
        );

        router.prefill_workers.write().unwrap().push(worker1);
        router.prefill_workers.write().unwrap().push(worker2);

        // Initialize tree with both workers
        {
            let tree_guard = tree.lock().unwrap();
            tree_guard.insert("", "http://worker1");
            tree_guard.insert("", "http://worker2");
        }

        // Remove one worker
        let result = router.remove_prefill_server("http://worker1").await;
        assert!(result.is_ok());

        // Verify tree only contains remaining worker
        let tree_guard = tree.lock().unwrap();
        let (_matched_text, tenant) = tree_guard.prefix_match("");
        // After rebuild, tree should only have worker2
        assert_eq!(tenant, "http://worker2");
    }

    #[tokio::test]
    async fn test_no_cache_tree_operations() {
        let router = create_test_pd_router();
        assert!(router.prefill_tree.is_none());

        // Add a worker without cache tree
        let worker = create_test_worker(
            "http://worker1".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        router.prefill_workers.write().unwrap().push(worker);

        // Remove should work without tree
        let result = router.remove_prefill_server("http://worker1").await;
        assert!(result.is_ok());
    }

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

        router
            .prefill_workers
            .write()
            .unwrap()
            .push(unhealthy_worker);
        router.prefill_workers.write().unwrap().push(healthy_worker);
        router.decode_workers.write().unwrap().push(decode_worker);

        let result = router.select_pd_pair(None).await;

        assert!(result.is_ok());
        let (prefill, _decode) = result.unwrap();

        // Should select the healthy worker
        assert_eq!(prefill.url(), "http://healthy");
        assert!(prefill.is_healthy());
    }

    #[tokio::test]
    async fn test_empty_worker_lists() {
        let router = create_test_pd_router();

        let result = router.select_pd_pair(None).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
    }

    // ============= Health Endpoints Tests =============

    #[tokio::test]
    async fn test_health_endpoints() {
        let router = create_test_pd_router();

        // Add healthy workers
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

        router.prefill_workers.write().unwrap().push(prefill_worker);
        router.decode_workers.write().unwrap().push(decode_worker);

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
        router.prefill_policy = power_of_two_policy.clone();
        router.decode_policy = power_of_two_policy;

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
                router_clone.prefill_workers.write().unwrap().push(worker);
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            let _ = handle.await;
        }

        // Check final state
        let workers = router.prefill_workers.read().unwrap();
        assert_eq!(workers.len(), 5);
    }

    #[tokio::test]
    async fn test_simplified_routing_preserves_sglang_fields() {
        use crate::openai_api_types::GenerateRequest;
        use crate::routers::bootstrap_injector::inject_bootstrap_fields;

        // Create a test worker
        let worker = BasicWorker::new(
            "http://test-server:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: Some(5678),
            },
        );

        // Create a GenerateRequest with SGLang extensions
        let mut session_params = std::collections::HashMap::new();
        session_params.insert("test_key".to_string(), serde_json::json!("test_value"));

        let request = GenerateRequest {
            text: Some("Test prompt".to_string()),
            stream: false,
            return_logprob: true,
            // SGLang extensions
            lora_path: Some(crate::openai_api_types::LoRAPath::Single(Some(
                "test.bin".to_string(),
            ))),
            session_params: Some(session_params.clone()),
            return_hidden_states: true,
            rid: Some("test-request-id".to_string()),
            // Other fields default to None/false
            prompt: None,
            input_ids: None,
            parameters: None,
            sampling_params: None,
        };

        // Convert to JSON (simulating the simplified routing path)
        let mut json = serde_json::to_value(&request).unwrap();

        // Inject bootstrap fields
        let result = inject_bootstrap_fields(&mut json, &worker);
        assert!(result.is_ok());

        // Verify all SGLang fields are preserved
        assert_eq!(json["text"], serde_json::json!("Test prompt"));
        assert_eq!(json["stream"], serde_json::json!(false));
        assert_eq!(json["return_logprob"], serde_json::json!(true));
        assert_eq!(json["lora_path"], serde_json::json!("test.bin")); // LoRAPath::Single serializes as just the inner value
        assert_eq!(
            json["session_params"],
            serde_json::to_value(&session_params).unwrap()
        );
        assert_eq!(json["return_hidden_states"], serde_json::json!(true));
        assert_eq!(json["rid"], serde_json::json!("test-request-id"));

        // Verify bootstrap fields were added
        assert_eq!(json["bootstrap_host"], serde_json::json!("test-server"));
        assert_eq!(json["bootstrap_port"], serde_json::json!(5678));
        assert!(json["bootstrap_room"].is_number());
    }

    #[tokio::test]
    async fn test_simplified_routing_chat_completion() {
        use crate::openai_api_types::{ChatCompletionRequest, ChatMessage, UserMessageContent};
        use crate::routers::bootstrap_injector::inject_bootstrap_fields;

        // Create a test worker
        let worker = BasicWorker::new(
            "http://chat-server:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: Some(9999),
            },
        );

        // Create a ChatCompletionRequest with SGLang extensions
        let request = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage::User {
                role: "user".to_string(),
                content: UserMessageContent::Text("Hello world!".to_string()),
                name: None,
            }],
            stream: false,
            n: Some(2), // This should create batch bootstrap
            // SGLang extensions
            top_k: Some(50),
            separate_reasoning: false,
            stream_reasoning: true,
            // Set all other fields to defaults
            temperature: None,
            top_p: None,
            stream_options: None,
            stop: None,
            max_tokens: None,
            max_completion_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            seed: None,
            logprobs: false,
            top_logprobs: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            functions: None,
            function_call: None,
            min_p: None,
            min_tokens: None,
            repetition_penalty: None,
            regex: None,
            ebnf: None,
            stop_token_ids: None,
            no_stop_trim: false,
            ignore_eos: false,
            continue_final_message: false,
            skip_special_tokens: true,
            lora_path: None,
            session_params: None,
            return_hidden_states: false,
        };

        // Convert to JSON (simulating the simplified routing path)
        let mut json = serde_json::to_value(&request).unwrap();

        // Inject bootstrap fields
        let result = inject_bootstrap_fields(&mut json, &worker);
        assert!(result.is_ok());

        // Verify original fields preserved
        assert_eq!(json["model"], serde_json::json!("gpt-4"));
        assert_eq!(json["stream"], serde_json::json!(false));
        assert_eq!(json["n"], serde_json::json!(2));
        assert_eq!(json["top_k"], serde_json::json!(50));
        assert_eq!(json["separate_reasoning"], serde_json::json!(false));
        assert_eq!(json["stream_reasoning"], serde_json::json!(true));

        // Verify batch bootstrap fields for n=2
        let bootstrap_hosts = json["bootstrap_host"].as_array().unwrap();
        assert_eq!(bootstrap_hosts.len(), 2);
        assert_eq!(bootstrap_hosts[0], serde_json::json!("chat-server"));
        assert_eq!(bootstrap_hosts[1], serde_json::json!("chat-server"));

        let bootstrap_ports = json["bootstrap_port"].as_array().unwrap();
        assert_eq!(bootstrap_ports.len(), 2);
        assert_eq!(bootstrap_ports[0], serde_json::json!(9999));
        assert_eq!(bootstrap_ports[1], serde_json::json!(9999));

        let bootstrap_rooms = json["bootstrap_room"].as_array().unwrap();
        assert_eq!(bootstrap_rooms.len(), 2);
        // Rooms should be different (randomness)
        assert_ne!(bootstrap_rooms[0], bootstrap_rooms[1]);
    }
}
