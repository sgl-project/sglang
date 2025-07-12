// PD (Prefill-Decode) Router Implementation
// This module handles routing for disaggregated prefill-decode systems

use crate::core::{HealthChecker, Worker, WorkerFactory, WorkerLoadGuard};
use crate::pd_types::{
    api_path, Bootstrap, ChatReqInput, GenerateReqInput, PDRouterError, PDSelectionPolicy,
};
use crate::tree::Tree;
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use futures_util::{StreamExt, TryStreamExt};
use metrics::{counter, histogram};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// Removed over-engineered ProxyResponse - using HttpResponse directly

#[derive(Debug)]
pub struct PDRouter {
    pub prefill_workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
    pub decode_workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
    pub selection_policy: PDSelectionPolicy,
    pub prefill_tree: Option<Arc<Mutex<Tree>>>,
    pub timeout_secs: u64,
    pub interval_secs: u64,
    pub worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, isize>>>,
    pub load_monitor_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    pub http_client: reqwest::Client,
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
        crate::router::Router::wait_for_healthy_workers(
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

        // Add to cache tree if using cache-aware policy
        if let Some(ref tree) = self.prefill_tree {
            tree.lock().unwrap().insert("", &url);
        }

        info!("Added prefill server: {}", url);
        Ok(format!("Successfully added prefill server: {}", url))
    }

    pub async fn add_decode_server(&self, url: String) -> Result<String, PDRouterError> {
        // Wait for the new server to be healthy
        crate::router::Router::wait_for_healthy_workers(
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

        // Initialize load tracking
        // Worker tracks its own load internally

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

        // Remove from load tracking
        // Worker load tracking is internal

        // Remove from cache tree if using cache-aware policy
        if let Some(ref tree) = self.prefill_tree {
            // Note: Tree doesn't have a remove method, so we rebuild it
            let mut tree_guard = tree.lock().unwrap();
            *tree_guard = Tree::new();
            for worker in workers.iter() {
                tree_guard.insert("", worker.url());
            }
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

        info!("Removed decode server: {}", url);
        Ok(format!("Successfully removed decode server: {}", url))
    }

    pub fn new(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        selection_policy: PDSelectionPolicy,
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

        // Wait for PD workers to be healthy
        let all_urls: Vec<String> = prefill_workers
            .iter()
            .chain(decode_workers.iter())
            .map(|worker| worker.url().to_string())
            .collect();
        crate::router::Router::wait_for_healthy_workers(&all_urls, timeout_secs, interval_secs)?;

        // Initialize cache-aware components if needed
        let prefill_tree = match &selection_policy {
            PDSelectionPolicy::CacheAware { .. } => {
                let tree = Arc::new(Mutex::new(Tree::new()));
                // Initialize tree with prefill workers
                for worker in &prefill_workers {
                    tree.lock().unwrap().insert("", worker.url());
                }
                Some(tree)
            }
            _ => None,
        };

        // Set up background load monitoring for power-of-two selection
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let worker_loads = Arc::new(rx);

        // Create a shared HTTP client for all operations
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        let load_monitor_handle = if matches!(selection_policy, PDSelectionPolicy::PowerOfTwo) {
            let monitor_urls = all_urls.clone();
            let monitor_interval = interval_secs;
            let monitor_client = http_client.clone();

            Some(Arc::new(tokio::spawn(async move {
                Self::monitor_worker_loads_with_client(
                    monitor_urls,
                    tx,
                    monitor_interval,
                    monitor_client,
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
            selection_policy,
            prefill_tree,
            timeout_secs,
            interval_secs,
            worker_loads,
            load_monitor_handle,
            http_client,
            _prefill_health_checker: Some(prefill_health_checker),
            _decode_health_checker: Some(decode_health_checker),
        })
    }

    // Route a typed generate request
    pub async fn route_generate(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        mut typed_req: GenerateReqInput,
        route: &str,
    ) -> HttpResponse {
        let start = Instant::now();
        let _request_id = Uuid::new_v4();

        // Get stream flag and return_logprob flag before moving the request
        let is_stream = typed_req.is_stream();
        let return_logprob = typed_req
            .other
            .get("return_logprob")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Select servers
        let (prefill, decode) = match self.select_pd_pair(client).await {
            Ok(pair) => pair,
            Err(e) => {
                error!("Failed to select PD pair: {}", e);
                counter!("sgl_router_pd_errors_total", "error" => "server_selection").increment(1);
                return HttpResponse::ServiceUnavailable()
                    .body(format!("No available servers: {}", e));
            }
        };

        // Log routing decision
        info!(
            "PD routing: {} -> prefill={}, decode={}",
            route,
            prefill.url(),
            decode.url()
        );

        // Add bootstrap info using the trait method
        if let Err(e) = typed_req.add_bootstrap_info(prefill.as_ref()) {
            error!("Failed to add bootstrap info: {}", e);
            counter!("sgl_router_pd_errors_total", "error" => "bootstrap_injection").increment(1);
            return HttpResponse::InternalServerError()
                .body(format!("Bootstrap injection failed: {}", e));
        }

        // Convert to JSON after bootstrap injection
        let json_with_bootstrap = match serde_json::to_value(&typed_req) {
            Ok(json) => json,
            Err(e) => {
                error!("Failed to serialize request: {}", e);
                return HttpResponse::InternalServerError().body("Failed to serialize request");
            }
        };

        // Execute dual dispatch
        self.execute_dual_dispatch(
            client,
            req,
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
        client: &reqwest::Client,
        req: &HttpRequest,
        mut typed_req: ChatReqInput,
        route: &str,
    ) -> HttpResponse {
        let start = Instant::now();

        // Get stream flag and return_logprob flag before moving the request
        let is_stream = typed_req.is_stream();
        let return_logprob = typed_req
            .other
            .get("return_logprob")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Select servers
        let (prefill, decode) = match self.select_pd_pair(client).await {
            Ok(pair) => pair,
            Err(e) => {
                error!("Failed to select PD pair: {}", e);
                counter!("sgl_router_pd_errors_total", "error" => "server_selection").increment(1);
                return HttpResponse::ServiceUnavailable()
                    .body(format!("No available servers: {}", e));
            }
        };

        // Log routing decision
        info!(
            "PD routing: {} -> prefill={}, decode={}",
            route,
            prefill.url(),
            decode.url()
        );

        // Add bootstrap info using the trait method
        if let Err(e) = typed_req.add_bootstrap_info(prefill.as_ref()) {
            error!("Failed to add bootstrap info: {}", e);
            counter!("sgl_router_pd_errors_total", "error" => "bootstrap_injection").increment(1);
            return HttpResponse::InternalServerError()
                .body(format!("Bootstrap injection failed: {}", e));
        }

        // Convert to JSON after bootstrap injection
        let json_with_bootstrap = match serde_json::to_value(&typed_req) {
            Ok(json) => json,
            Err(e) => {
                error!("Failed to serialize request: {}", e);
                return HttpResponse::InternalServerError().body("Failed to serialize request");
            }
        };

        // Execute dual dispatch
        self.execute_dual_dispatch(
            client,
            req,
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
    #[allow(clippy::too_many_arguments)]
    async fn execute_dual_dispatch(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        json_request: serde_json::Value,
        route: &str,
        prefill: &dyn Worker,
        decode: &dyn Worker,
        is_stream: bool,
        return_logprob: bool,
        start_time: Instant,
    ) -> HttpResponse {
        // Update load tracking for both workers
        let _guard = WorkerLoadGuard::new_multi(vec![prefill, decode]);

        // Build requests using .json() method
        let mut prefill_request = client
            .post(api_path(prefill.url(), route))
            .json(&json_request);

        let mut decode_request = client
            .post(api_path(decode.url(), route))
            .json(&json_request);

        // Copy headers from original request
        for (name, value) in crate::router::copy_request_headers(req) {
            if name.to_lowercase() != "content-type" && name.to_lowercase() != "content-length" {
                prefill_request = prefill_request.header(&name, &value);
                decode_request = decode_request.header(&name, &value);
            }
        }

        // Send both requests concurrently
        let (prefill_result, decode_result) =
            tokio::join!(prefill_request.send(), decode_request.send());

        // Update metrics
        let duration = start_time.elapsed();
        histogram!("sgl_router_pd_request_duration_seconds", "route" => route.to_string())
            .record(duration.as_secs_f64());
        counter!("sgl_router_pd_requests_total", "route" => route.to_string()).increment(1);
        counter!("sgl_router_pd_prefill_requests_total", "worker" => prefill.url().to_string())
            .increment(1);
        counter!("sgl_router_pd_decode_requests_total", "worker" => decode.url().to_string())
            .increment(1);

        // Process decode response
        match decode_result {
            Ok(res) => {
                let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

                if !status.is_success() {
                    counter!("sgl_router_pd_decode_errors_total", "worker" => decode.url().to_string()).increment(1);
                    error!(
                        "Decode server {} returned error status: {}",
                        decode.url(),
                        status
                    );

                    // Return the error response from decode server
                    match res.bytes().await {
                        Ok(error_body) => {
                            return HttpResponse::build(status).body(error_body.to_vec());
                        }
                        Err(e) => {
                            return HttpResponse::build(status)
                                .body(format!("Decode server error: {}", e));
                        }
                    }
                }

                // Log prefill errors for debugging
                if let Err(e) = &prefill_result {
                    error!(
                        "Prefill server {} failed (non-critical): {}",
                        prefill.url(),
                        e
                    );
                    counter!("sgl_router_pd_prefill_errors_total", "worker" => prefill.url().to_string()).increment(1);
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
                        HttpResponse::build(status)
                            .insert_header((
                                CONTENT_TYPE,
                                HeaderValue::from_static("text/event-stream"),
                            ))
                            .streaming(res.bytes_stream().map(move |chunk_result| {
                                match chunk_result {
                                    Ok(chunk) => {
                                        // Try to merge logprobs
                                        if let Ok(merged) = Self::merge_streaming_logprobs(
                                            prefill_logprobs.clone(),
                                            &chunk,
                                        ) {
                                            Ok(merged)
                                        } else {
                                            Ok(chunk)
                                        }
                                    }
                                    Err(e) => Err(actix_web::error::ErrorInternalServerError(
                                        format!("Stream error: {}", e),
                                    )),
                                }
                            }))
                    } else {
                        // No logprob merging needed
                        HttpResponse::build(status)
                            .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                            .streaming({
                                let decode_url = decode.url().to_string();
                                res.bytes_stream().map_err(move |e| {
                                    error!("Stream error from decode server {}: {}", decode_url, e);
                                    counter!("sgl_router_pd_stream_errors_total", "worker" => decode_url.to_string()).increment(1);
                                    actix_web::error::ErrorInternalServerError(format!("Stream error: {}", e))
                                })
                            })
                    }
                } else {
                    // Non-streaming response
                    match res.bytes().await {
                        Ok(decode_body) => {
                            if return_logprob {
                                self.merge_logprobs(prefill_result, decode_body, status)
                                    .await
                            } else {
                                HttpResponse::build(status).body(decode_body.to_vec())
                            }
                        }
                        Err(e) => {
                            error!("Failed to read decode response: {}", e);
                            HttpResponse::InternalServerError().body("Failed to read response")
                        }
                    }
                }
            }
            Err(e) => {
                error!("Decode request failed: {}", e);
                counter!("sgl_router_pd_decode_errors_total", "worker" => decode.url().to_string())
                    .increment(1);
                HttpResponse::BadGateway().body(format!("Decode server error: {}", e))
            }
        }
    }

    // Merge logprobs from prefill and decode responses
    async fn merge_logprobs(
        &self,
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        decode_body: bytes::Bytes,
        status: actix_web::http::StatusCode,
    ) -> HttpResponse {
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
                                HttpResponse::build(status).json(&decode_json)
                            }
                            _ => {
                                warn!("Failed to parse responses for logprob merging");
                                HttpResponse::build(status).body(decode_body.to_vec())
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to read prefill response: {}", e);
                        HttpResponse::build(status).body(decode_body.to_vec())
                    }
                }
            }
            Err(_) => HttpResponse::build(status).body(decode_body.to_vec()),
        }
    }

    // Select a pair of prefill and decode servers
    async fn select_pd_pair(
        &self,
        _client: &reqwest::Client,
    ) -> Result<(Box<dyn Worker>, Box<dyn Worker>), String> {
        // Check we have workers
        if self
            .prefill_workers
            .read()
            .map_err(|e| format!("Failed to acquire prefill workers lock: {}", e))?
            .is_empty()
        {
            return Err("No prefill workers available. Please check if prefill servers are configured and healthy.".to_string());
        }
        if self
            .decode_workers
            .read()
            .map_err(|e| format!("Failed to acquire decode workers lock: {}", e))?
            .is_empty()
        {
            return Err("No decode workers available. Please check if decode servers are configured and healthy.".to_string());
        }

        match &self.selection_policy {
            PDSelectionPolicy::Random => self.select_random(),
            PDSelectionPolicy::PowerOfTwo => self.select_power_of_two().await,
            PDSelectionPolicy::CacheAware { .. } => {
                // TODO: Implement cache-aware selection
                self.select_power_of_two().await
            }
        }
    }

    fn select_random(&self) -> Result<(Box<dyn Worker>, Box<dyn Worker>), String> {
        let prefill_list = self.prefill_workers.read().map_err(|_| "Lock error")?;
        let decode_list = self.decode_workers.read().map_err(|_| "Lock error")?;

        let prefill = prefill_list[rand::random::<usize>() % prefill_list.len()].clone_worker();
        let decode = decode_list[rand::random::<usize>() % decode_list.len()].clone_worker();

        Ok((prefill, decode))
    }

    async fn select_power_of_two(&self) -> Result<(Box<dyn Worker>, Box<dyn Worker>), String> {
        let prefill_list = self.prefill_workers.read().map_err(|_| "Lock error")?;
        let decode_list = self.decode_workers.read().map_err(|_| "Lock error")?;

        let (p1_idx, p2_idx) = get_two_random_indices(prefill_list.len());
        let (d1_idx, d2_idx) = get_two_random_indices(decode_list.len());

        let loads = self.worker_loads.borrow();

        let p1_load = loads
            .get(prefill_list[p1_idx].url())
            .copied()
            .unwrap_or(isize::MAX);
        let p2_load = loads
            .get(prefill_list[p2_idx].url())
            .copied()
            .unwrap_or(isize::MAX);
        let d1_load = loads
            .get(decode_list[d1_idx].url())
            .copied()
            .unwrap_or(isize::MAX);
        let d2_load = loads
            .get(decode_list[d2_idx].url())
            .copied()
            .unwrap_or(isize::MAX);

        info!(
            "Power-of-two selection - Prefill: {}={} vs {}={} | Decode: {}={} vs {}={}",
            prefill_list[p1_idx].url(),
            p1_load,
            prefill_list[p2_idx].url(),
            p2_load,
            decode_list[d1_idx].url(),
            d1_load,
            decode_list[d2_idx].url(),
            d2_load
        );

        let selected_prefill = if p1_load <= p2_load {
            prefill_list[p1_idx].clone_worker()
        } else {
            prefill_list[p2_idx].clone_worker()
        };

        let selected_decode = if d1_load <= d2_load {
            decode_list[d1_idx].clone_worker()
        } else {
            decode_list[d2_idx].clone_worker()
        };

        Ok((selected_prefill, selected_decode))
    }

    // Background task to monitor worker loads with shared client
    async fn monitor_worker_loads_with_client(
        worker_urls: Vec<String>,
        tx: tokio::sync::watch::Sender<HashMap<String, isize>>,
        interval_secs: u64,
        client: reqwest::Client,
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
fn get_two_random_indices(len: usize) -> (usize, usize) {
    if len == 1 {
        (0, 0)
    } else {
        let idx1 = rand::random::<usize>() % len;
        let mut idx2 = rand::random::<usize>() % len;
        while idx2 == idx1 {
            idx2 = rand::random::<usize>() % len;
        }
        (idx1, idx2)
    }
}

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
    pub async fn health_generate(&self, client: &reqwest::Client) -> HttpResponse {
        let mut all_healthy = true;
        let mut unhealthy_servers = Vec::new();

        // Collect all worker URLs with their types
        let mut worker_infos = Vec::new();

        for worker in self.prefill_workers.read().unwrap().iter() {
            worker_infos.push((worker.url().to_string(), "prefill"));
        }

        for worker in self.decode_workers.read().unwrap().iter() {
            worker_infos.push((worker.url().to_string(), "decode"));
        }

        // Create tasks with URL tracking
        let tasks: Vec<_> = worker_infos
            .iter()
            .map(|(url, _)| {
                let health_url = format!("{}/health_generate", url);
                client.get(&health_url).send()
            })
            .collect();

        let results = futures_util::future::join_all(tasks).await;

        for ((url, worker_type), result) in worker_infos.iter().zip(results.into_iter()) {
            match result {
                Ok(res) if res.status().is_success() => {
                    debug!("Health check passed for {} server: {}", worker_type, url);
                }
                Ok(res) => {
                    all_healthy = false;
                    let msg = format!(
                        "{} server {} returned status {}",
                        worker_type,
                        url,
                        res.status()
                    );
                    error!("{}", msg);
                    unhealthy_servers.push(msg);
                }
                Err(e) => {
                    all_healthy = false;
                    let msg = format!("{} server {} error: {}", worker_type, url, e);
                    error!("{}", msg);
                    unhealthy_servers.push(msg);
                }
            }
        }

        if all_healthy {
            HttpResponse::Ok().body("Health check passed on all servers")
        } else {
            HttpResponse::ServiceUnavailable()
                .body(format!("Health check failed: {:?}", unhealthy_servers))
        }
    }

    pub async fn get_server_info(&self, client: &reqwest::Client) -> HttpResponse {
        // Get info from the first decode server to match sglang's server info format
        let first_decode_url = if let Ok(workers) = self.decode_workers.read() {
            workers.first().map(|w| w.url().to_string())
        } else {
            return HttpResponse::InternalServerError().body("Failed to access decode workers");
        };

        if let Some(worker_url) = first_decode_url {
            match client
                .get(format!("{}/get_server_info", worker_url))
                .send()
                .await
            {
                Ok(res) if res.status().is_success() => {
                    match res.json::<Value>().await {
                        Ok(info) => {
                            // The decode server should already return the proper format
                            // with tokenizer_path and other fields that bench_one_batch_server.py expects
                            HttpResponse::Ok().json(info)
                        }
                        Err(e) => {
                            error!("Failed to parse server info: {}", e);
                            HttpResponse::InternalServerError()
                                .body(format!("Failed to parse server info: {}", e))
                        }
                    }
                }
                Ok(res) => {
                    let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
                    HttpResponse::build(status)
                        .body(format!("Decode server returned status: {}", res.status()))
                }
                Err(e) => {
                    error!("Failed to get server info: {}", e);
                    HttpResponse::InternalServerError()
                        .body(format!("Failed to get server info: {}", e))
                }
            }
        } else {
            HttpResponse::ServiceUnavailable().body("No decode servers available")
        }
    }

    pub async fn get_models(&self, client: &reqwest::Client, req: &HttpRequest) -> HttpResponse {
        // Get first prefill worker URL to avoid holding lock across await
        let first_worker_url = if let Ok(workers) = self.prefill_workers.read() {
            workers.first().map(|w| w.url().to_string())
        } else {
            return HttpResponse::InternalServerError().body("Failed to access prefill workers");
        };

        if let Some(worker_url) = first_worker_url {
            // Send request directly without going through Router
            let mut request_builder = client.get(format!("{}/v1/models", worker_url));
            for (name, value) in crate::router::copy_request_headers(req) {
                if name.to_lowercase() != "content-type" && name.to_lowercase() != "content-length"
                {
                    request_builder = request_builder.header(name, value);
                }
            }
            match request_builder.send().await {
                Ok(res) => {
                    let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
                    match res.bytes().await {
                        Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                        Err(e) => HttpResponse::InternalServerError()
                            .body(format!("Failed to read response body: {}", e)),
                    }
                }
                Err(e) => HttpResponse::InternalServerError()
                    .body(format!("Failed to send request: {}", e)),
            }
        } else {
            HttpResponse::ServiceUnavailable().body("No prefill servers available")
        }
    }

    pub async fn get_loads(&self, client: &reqwest::Client) -> HttpResponse {
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

        HttpResponse::Ok().json(serde_json::json!({
            "prefill": prefill_loads,
            "decode": decode_loads
        }))
    }

    pub async fn get_model_info(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
    ) -> HttpResponse {
        // Get model info from the first prefill server (matches original Rust PDLB behavior)
        // Get first prefill worker URL to avoid holding lock across await
        let first_worker_url = if let Ok(workers) = self.prefill_workers.read() {
            workers.first().map(|w| w.url().to_string())
        } else {
            return HttpResponse::InternalServerError().body("Failed to access prefill workers");
        };

        if let Some(worker_url) = first_worker_url {
            let mut request_builder = client.get(format!("{}/get_model_info", worker_url));
            for (name, value) in crate::router::copy_request_headers(req) {
                if name.to_lowercase() != "content-type" && name.to_lowercase() != "content-length"
                {
                    request_builder = request_builder.header(name, value);
                }
            }
            match request_builder.send().await {
                Ok(res) => {
                    let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
                    match res.bytes().await {
                        Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                        Err(e) => HttpResponse::InternalServerError()
                            .body(format!("Failed to read response body: {}", e)),
                    }
                }
                Err(e) => HttpResponse::InternalServerError()
                    .body(format!("Failed to send request: {}", e)),
            }
        } else {
            HttpResponse::ServiceUnavailable().body("No prefill servers available")
        }
    }

    pub async fn flush_cache(&self, client: &reqwest::Client) -> HttpResponse {
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
            HttpResponse::Ok().body("Cache flushed on all servers")
        } else {
            HttpResponse::InternalServerError().body("Cache flush failed on one or more servers")
        }
    }
}
