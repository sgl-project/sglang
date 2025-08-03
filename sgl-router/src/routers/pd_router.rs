// PD (Prefill-Decode) Router Implementation
// This module handles routing for disaggregated prefill-decode systems

use super::pd_types::{api_path, Bootstrap, ChatReqInput, GenerateReqInput, PDRouterError};
use super::request_adapter::ToPdRequest;
use crate::core::{HealthChecker, Worker, WorkerFactory, WorkerLoadGuard};
use crate::metrics::RouterMetrics;
use crate::openai_api_types::{ChatCompletionRequest, CompletionRequest, GenerateRequest};
use crate::policies::LoadBalancingPolicy;
use crate::tree::Tree;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use futures_util::StreamExt;
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
    _prefill_health_checker: Option<HealthChecker>,
    _decode_health_checker: Option<HealthChecker>,
}

impl PDRouter {
    // Dynamic worker management methods for service discovery
    pub async fn add_prefill_server(
        &self,
        url: String,
        bootstrap_port: Option<u16>,
    ) -> Result<String, PDRouterError> {
        // Wait for the new server to be healthy
        crate::routers::router::Router::wait_for_healthy_workers(
            &[url.clone()],
            self.timeout_secs,
            self.interval_secs,
        )
        .map_err(|_| PDRouterError::HealthCheckFailed { url: url.clone() })?;

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
        crate::routers::router::Router::wait_for_healthy_workers(
            &[url.clone()],
            self.timeout_secs,
            self.interval_secs,
        )
        .map_err(|_| PDRouterError::HealthCheckFailed { url: url.clone() })?;

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

    // Route a typed generate request
    pub async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        mut typed_req: GenerateReqInput,
        route: &str,
    ) -> Response {
        let start = Instant::now();

        // Get stream flag and return_logprob flag before moving the request
        let is_stream = typed_req.stream;
        let return_logprob = typed_req
            .other
            .get("return_logprob")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Extract text for cache-aware routing from the typed request
        let request_text = typed_req.text.as_ref().and_then(|t| match t {
            super::pd_types::InputText::Single(s) => Some(s.as_str()),
            super::pd_types::InputText::Batch(v) => v.first().map(|s| s.as_str()),
        });

        // Select servers
        let (prefill, decode) = match self.select_pd_pair(request_text).await {
            Ok(pair) => pair,
            Err(e) => {
                error!("Failed to select PD pair error={}", e);
                RouterMetrics::record_pd_error("server_selection");
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("No available servers: {}", e),
                )
                    .into_response();
            }
        };

        // Log routing decision
        info!(
            "PD routing decision route={} prefill_url={} decode_url={}",
            route,
            prefill.url(),
            decode.url()
        );

        // Add bootstrap info using the trait method
        if let Err(e) = typed_req.add_bootstrap_info(prefill.as_ref()) {
            error!("Failed to add bootstrap info error={}", e);
            RouterMetrics::record_pd_error("bootstrap_injection");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Bootstrap injection failed: {}", e),
            )
                .into_response();
        }

        // Convert to JSON after bootstrap injection
        let json_with_bootstrap = match serde_json::to_value(&typed_req) {
            Ok(json) => json,
            Err(e) => {
                error!("Failed to serialize request error={}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to serialize request",
                )
                    .into_response();
            }
        };

        // Execute dual dispatch
        self.execute_dual_dispatch(
            headers,
            json_with_bootstrap,
            route,
            prefill.as_ref(),
            decode.as_ref(),
            is_stream,
            return_logprob,
            start,
        )
        .await
    }

    // Route a typed chat request
    pub async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        mut typed_req: ChatReqInput,
        route: &str,
    ) -> Response {
        let start = Instant::now();

        // Get stream flag and return_logprob flag before moving the request
        let is_stream = typed_req.stream;
        let return_logprob = typed_req
            .other
            .get("return_logprob")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Extract text for cache-aware routing from chat messages
        let request_text = typed_req
            .other
            .get("messages")
            .and_then(|messages| messages.as_array())
            .and_then(|arr| arr.first())
            .and_then(|msg| msg.get("content"))
            .and_then(|content| content.as_str());

        // Select servers
        let (prefill, decode) = match self.select_pd_pair(request_text).await {
            Ok(pair) => pair,
            Err(e) => {
                error!("Failed to select PD pair error={}", e);
                RouterMetrics::record_pd_error("server_selection");
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("No available servers: {}", e),
                )
                    .into_response();
            }
        };

        // Log routing decision
        info!(
            "PD routing decision route={} prefill_url={} decode_url={}",
            route,
            prefill.url(),
            decode.url()
        );

        // Add bootstrap info using the trait method
        if let Err(e) = typed_req.add_bootstrap_info(prefill.as_ref()) {
            error!("Failed to add bootstrap info error={}", e);
            RouterMetrics::record_pd_error("bootstrap_injection");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Bootstrap injection failed: {}", e),
            )
                .into_response();
        }

        // Convert to JSON after bootstrap injection
        let json_with_bootstrap = match serde_json::to_value(&typed_req) {
            Ok(json) => json,
            Err(e) => {
                error!("Failed to serialize request error={}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to serialize request",
                )
                    .into_response();
            }
        };

        // Execute dual dispatch
        self.execute_dual_dispatch(
            headers,
            json_with_bootstrap,
            route,
            prefill.as_ref(),
            decode.as_ref(),
            is_stream,
            return_logprob,
            start,
        )
        .await
    }

    // Route a completion request while preserving OpenAI format
    pub async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        mut typed_req: CompletionRequest,
        route: &str,
    ) -> Response {
        let start = Instant::now();

        // Get stream flag and return_logprob flag before moving the request
        let is_stream = typed_req.stream;
        let return_logprob = typed_req.logprobs.is_some();

        // Extract text for cache-aware routing from the typed request
        let request_text = match &typed_req.prompt {
            crate::openai_api_types::StringOrArray::String(s) => Some(s.as_str()),
            crate::openai_api_types::StringOrArray::Array(arr) => arr.first().map(|s| s.as_str()),
        };

        // Select servers
        let (prefill, decode) = match self.select_pd_pair(request_text).await {
            Ok(pair) => pair,
            Err(e) => {
                error!("Failed to select PD pair error={}", e);
                RouterMetrics::record_pd_error("server_selection");
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("No available servers: {}", e),
                )
                    .into_response();
            }
        };

        // Log routing decision
        info!(
            "PD routing decision route={} prefill_url={} decode_url={}",
            route,
            prefill.url(),
            decode.url()
        );

        // Add bootstrap info using the trait method
        if let Err(e) = typed_req.add_bootstrap_info(prefill.as_ref()) {
            error!("Failed to add bootstrap info error={}", e);
            RouterMetrics::record_pd_error("bootstrap_injection");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Bootstrap injection failed: {}", e),
            )
                .into_response();
        }

        // Convert to JSON after bootstrap injection
        let json_with_bootstrap = match serde_json::to_value(&typed_req) {
            Ok(json) => json,
            Err(e) => {
                error!("Failed to serialize request error={}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to serialize request",
                )
                    .into_response();
            }
        };

        // Execute dual dispatch
        self.execute_dual_dispatch(
            headers,
            json_with_bootstrap,
            route,
            prefill.as_ref(),
            decode.as_ref(),
            is_stream,
            return_logprob,
            start,
        )
        .await
    }

    // Execute the dual dispatch to prefill and decode servers
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
        // Update load tracking for both workers
        let _guard = WorkerLoadGuard::new_multi(vec![prefill, decode]);

        // Build requests using .json() method
        let mut prefill_request = self
            .client
            .post(api_path(prefill.url(), route))
            .json(&json_request);

        let mut decode_request = self
            .client
            .post(api_path(decode.url(), route))
            .json(&json_request);

        // Copy headers from original request (excluding content-type and content-length which are set by .json())
        if let Some(headers) = headers {
            for (name, value) in headers.iter() {
                let name_str = name.as_str();
                if name_str != "content-type" && name_str != "content-length" {
                    // Skip headers with non-ASCII values
                    if value.to_str().is_ok() {
                        prefill_request = prefill_request.header(name, value);
                        decode_request = decode_request.header(name, value);
                    }
                }
            }
        }

        // Send both requests concurrently
        let (prefill_result, decode_result) =
            tokio::join!(prefill_request.send(), decode_request.send());

        // Update metrics
        let duration = start_time.elapsed();
        RouterMetrics::record_pd_request_duration(route, duration);
        RouterMetrics::record_pd_request(route);
        RouterMetrics::record_pd_prefill_request(prefill.url());
        RouterMetrics::record_pd_decode_request(decode.url());

        // Process decode response
        match decode_result {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

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

                // Log prefill errors for debugging
                if let Err(e) = &prefill_result {
                    error!(
                        "Prefill server failed (non-critical) prefill_url={} error={}",
                        prefill.url(),
                        e
                    );
                    RouterMetrics::record_pd_prefill_error(prefill.url());
                }

                if is_stream {
                    // Streaming response
                    if return_logprob {
                        // Get prefill logprobs for merging
                        let prefill_logprobs =
                            match prefill_result {
                                Ok(prefill_res) => match prefill_res.bytes().await {
                                    Ok(body) => serde_json::from_slice::<Value>(&body)
                                        .ok()
                                        .and_then(|json| {
                                            json.pointer("/meta_info/input_token_logprobs").cloned()
                                        }),
                                    Err(_) => None,
                                },
                                Err(_) => None,
                            };

                        // Stream with logprob merging
                        let stream = res.bytes_stream();
                        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

                        tokio::spawn(async move {
                            let mut stream = stream;
                            while let Some(chunk_result) = stream.next().await {
                                match chunk_result {
                                    Ok(chunk) => {
                                        // Try to merge logprobs
                                        if let Ok(merged) = Self::merge_streaming_logprobs(
                                            prefill_logprobs.clone(),
                                            &chunk,
                                        ) {
                                            if tx.send(Ok(merged)).is_err() {
                                                break;
                                            }
                                        } else {
                                            if tx.send(Ok(chunk)).is_err() {
                                                break;
                                            }
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
                        response
                            .headers_mut()
                            .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
                        response
                    } else {
                        // No logprob merging needed
                        let stream = res.bytes_stream();
                        let decode_url = decode.url().to_string();
                        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

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
                                        error!(
                                            "Stream error from decode server {}: {}",
                                            decode_url, e
                                        );
                                        RouterMetrics::record_pd_stream_error(&decode_url);
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
                } else {
                    // Non-streaming response
                    match res.bytes().await {
                        Ok(decode_body) => {
                            if return_logprob {
                                self.merge_logprobs(prefill_result, decode_body, status)
                                    .await
                            } else {
                                (status, decode_body).into_response()
                            }
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

    // Merge logprobs from prefill and decode responses
    async fn merge_logprobs(
        &self,
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        decode_body: bytes::Bytes,
        status: StatusCode,
    ) -> Response {
        match prefill_result {
            Ok(prefill_res) => {
                match prefill_res.bytes().await {
                    Ok(prefill_body) => {
                        match (
                            serde_json::from_slice::<Value>(&prefill_body),
                            serde_json::from_slice::<Value>(&decode_body),
                        ) {
                            (Ok(prefill_json), Ok(mut decode_json)) => {
                                // Merge input_token_logprobs
                                if let (Some(prefill_meta), Some(decode_meta)) = (
                                    prefill_json.get("meta_info"),
                                    decode_json.get_mut("meta_info"),
                                ) {
                                    if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
                                        prefill_meta.get("input_token_logprobs"),
                                        decode_meta.get_mut("input_token_logprobs"),
                                    ) {
                                        if let (Some(p_arr), Some(d_arr)) = (
                                            prefill_logprobs.as_array(),
                                            decode_logprobs.as_array(),
                                        ) {
                                            let mut merged = p_arr.clone();
                                            merged.extend(d_arr.clone());
                                            decode_meta["input_token_logprobs"] =
                                                Value::Array(merged);
                                        }
                                    }
                                }
                                let mut response = Json(decode_json).into_response();
                                *response.status_mut() = status;
                                response
                            }
                            _ => {
                                warn!("Failed to parse responses for logprob merging");
                                (status, decode_body).into_response()
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to read prefill response: {}", e);
                        (status, decode_body).into_response()
                    }
                }
            }
            Err(_) => (status, decode_body).into_response(),
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

async fn get_worker_load(client: &reqwest::Client, worker_url: &str) -> Option<isize> {
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

// PD-specific endpoints
impl PDRouter {
    pub async fn health_generate(&self) -> Response {
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

    pub async fn get_server_info(&self) -> Response {
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

    pub async fn get_models(&self, req: Request<Body>) -> Response {
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
            // Send request directly without going through Router
            let mut request_builder = self.client.get(format!("{}/v1/models", worker_url));
            for (name, value) in headers {
                if name.to_lowercase() != "content-type" && name.to_lowercase() != "content-length"
                {
                    request_builder = request_builder.header(name, value);
                }
            }
            match request_builder.send().await {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match res.bytes().await {
                        Ok(body) => (status, body).into_response(),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read response body: {}", e),
                        )
                            .into_response(),
                    }
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to send request: {}", e),
                )
                    .into_response(),
            }
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No prefill servers available",
            )
                .into_response()
        }
    }

    pub async fn get_loads(&self, client: &reqwest::Client) -> Response {
        let p_urls: Vec<_> = self
            .prefill_workers
            .read()
            .unwrap()
            .iter()
            .map(|w| w.url().to_string())
            .collect();
        let d_urls: Vec<_> = self
            .decode_workers
            .read()
            .unwrap()
            .iter()
            .map(|w| w.url().to_string())
            .collect();

        let mut prefill_loads = Vec::new();
        let mut decode_loads = Vec::new();

        for url in &p_urls {
            let load = get_worker_load(client, url).await.unwrap_or(-1);
            prefill_loads.push(serde_json::json!({
                "engine": format!("(Prefill@{})", url),
                "load": load as i64
            }));
        }

        for url in &d_urls {
            let load = get_worker_load(client, url).await.unwrap_or(-1);
            decode_loads.push(serde_json::json!({
                "engine": format!("(Decode@{})", url),
                "load": load as i64
            }));
        }

        Json(serde_json::json!({
            "prefill": prefill_loads,
            "decode": decode_loads
        }))
        .into_response()
    }

    pub async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = crate::routers::router::copy_request_headers(&req);

        // Get model info from the first prefill server (matches original Rust PDLB behavior)
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
            let mut request_builder = self.client.get(format!("{}/get_model_info", worker_url));
            for (name, value) in headers {
                if name.to_lowercase() != "content-type" && name.to_lowercase() != "content-length"
                {
                    request_builder = request_builder.header(name, value);
                }
            }
            match request_builder.send().await {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match res.bytes().await {
                        Ok(body) => (status, body).into_response(),
                        Err(e) => (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to read response body: {}", e),
                        )
                            .into_response(),
                    }
                }
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to send request: {}", e),
                )
                    .into_response(),
            }
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No prefill servers available",
            )
                .into_response()
        }
    }

    pub async fn flush_cache(&self, client: &reqwest::Client) -> Response {
        let mut tasks = Vec::new();

        // Flush cache on all prefill servers
        for worker in self.prefill_workers.read().unwrap().iter() {
            let url = format!("{}/flush_cache", worker.url());
            tasks.push(client.post(&url).send());
        }

        // Flush cache on all decode servers
        for worker in self.decode_workers.read().unwrap().iter() {
            let url = format!("{}/flush_cache", worker.url());
            tasks.push(client.post(&url).send());
        }

        let results = futures_util::future::join_all(tasks).await;

        let mut all_success = true;
        for (i, result) in results.into_iter().enumerate() {
            match result {
                Ok(res) if res.status().is_success() => {}
                Ok(res) => {
                    all_success = false;
                    warn!(
                        "Server {} returned status {} for flush_cache",
                        i,
                        res.status()
                    );
                }
                Err(e) => {
                    all_success = false;
                    error!("Server {} error during flush_cache: {}", i, e);
                }
            }
        }

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
}

use crate::routers::{RouterTrait, WorkerManagement};
use async_trait::async_trait;
use reqwest::Client;

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
        // Use the existing PDRouter health_generate method
        PDRouter::health_generate(self).await
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // Use the existing PDRouter get_server_info method
        PDRouter::get_server_info(self).await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Use the existing PDRouter get_models method
        PDRouter::get_models(self, req).await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Use the existing PDRouter get_model_info method
        PDRouter::get_model_info(self, req).await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
    ) -> Response {
        // Convert OpenAI format to PD format
        let pd_req = body.clone().to_pd_request();

        PDRouter::route_generate(self, headers, pd_req, "/generate").await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
    ) -> Response {
        // Convert OpenAI format to PD format
        let pd_req = body.clone().to_pd_request();

        PDRouter::route_chat(self, headers, pd_req, "/v1/chat/completions").await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
    ) -> Response {
        // Use the new method that preserves OpenAI format
        PDRouter::route_completion(self, headers, body.clone(), "/v1/completions").await
    }

    async fn flush_cache(&self) -> Response {
        // Use the existing PDRouter flush_cache method
        PDRouter::flush_cache(self, &self.client).await
    }

    async fn get_worker_loads(&self) -> Response {
        // Use the existing PDRouter get_loads method
        PDRouter::get_loads(self, &self.client).await
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
    use crate::routers::pd_types::SingleOrBatch;

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

    // ============= Bootstrap Injection Tests =============

    #[test]
    fn test_bootstrap_injection_with_existing_fields() {
        let mut req = GenerateReqInput {
            text: Some(SingleOrBatch::Single("Test".to_string())),
            input_ids: None,
            stream: false,
            bootstrap_host: Some(SingleOrBatch::Single("existing-host".to_string())),
            bootstrap_port: Some(SingleOrBatch::Single(Some(9999))),
            bootstrap_room: Some(SingleOrBatch::Single(12345)),
            other: Value::Object(serde_json::Map::new()),
        };

        let prefill_worker = create_test_worker(
            "http://new-host:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: Some(8080),
            },
            true,
        );

        // Bootstrap info is added regardless of existing fields
        let result = req.add_bootstrap_info(prefill_worker.as_ref());
        assert!(result.is_ok());

        // Bootstrap info should be updated with new values
        assert_eq!(
            req.bootstrap_host,
            Some(SingleOrBatch::Single("new-host".to_string()))
        );
        assert_eq!(req.bootstrap_port, Some(SingleOrBatch::Single(Some(8080))));
        // Room should be regenerated (different from original)
        if let Some(SingleOrBatch::Single(room)) = req.bootstrap_room {
            assert_ne!(room, 12345);
        } else {
            panic!("Expected single room ID");
        }
    }

    #[test]
    fn test_bootstrap_room_generation() {
        let mut req1 = GenerateReqInput {
            text: Some(SingleOrBatch::Single("Test".to_string())),
            input_ids: None,
            stream: false,
            bootstrap_host: None,
            bootstrap_port: None,
            bootstrap_room: None,
            other: Value::Object(serde_json::Map::new()),
        };

        let mut req2 = GenerateReqInput {
            text: Some(SingleOrBatch::Single("Test".to_string())),
            input_ids: None,
            stream: false,
            bootstrap_host: None,
            bootstrap_port: None,
            bootstrap_room: None,
            other: Value::Object(serde_json::Map::new()),
        };

        let prefill_worker = create_test_worker(
            "http://host:8000".to_string(),
            WorkerType::Prefill {
                bootstrap_port: Some(8080),
            },
            true,
        );

        // Add bootstrap info to both requests
        let _ = req1.add_bootstrap_info(prefill_worker.as_ref());
        let _ = req2.add_bootstrap_info(prefill_worker.as_ref());

        // Room IDs should be different
        if let (Some(SingleOrBatch::Single(room1)), Some(SingleOrBatch::Single(room2))) =
            (req1.bootstrap_room, req2.bootstrap_room)
        {
            assert_ne!(room1, room2, "Room IDs should be unique");
        } else {
            panic!("Expected single room IDs");
        }
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
}
