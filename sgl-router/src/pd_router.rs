// PD (Prefill-Decode) Router Implementation
// This module handles routing for disaggregated prefill-decode systems

use crate::core::{Worker, WorkerFactory, WorkerType};
use crate::pd_types::{
    Bootstrap, ChatReqInput, GenerateReqInput, PDRouterError, PDSelectionPolicy,
};
use crate::tree::Tree;
use crate::utils::api_path;
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use futures_util::{StreamExt, TryStreamExt};
use metrics::{counter, histogram};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, RwLock, RwLockWriteGuard};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// Removed over-engineered ProxyResponse - using HttpResponse directly

#[derive(Debug)]
pub struct PDRouter {
    pub prefill_workers: Arc<RwLock<Vec<Arc<dyn Worker>>>>,
    pub decode_workers: Arc<RwLock<Vec<Arc<dyn Worker>>>>,
    pub selection_policy: PDSelectionPolicy,
    pub prefill_tree: Option<Arc<Mutex<Tree>>>,
    pub timeout_secs: u64,
    pub interval_secs: u64,
    pub worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, isize>>>,
    pub load_monitor_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    pub http_client: reqwest::Client,
}

// RAII guard for load tracking to ensure cleanup even on panic
struct LoadGuard {
    workers: Vec<Arc<dyn Worker>>,
}

impl LoadGuard {
    fn new(workers: Vec<Arc<dyn Worker>>) -> Self {
        for worker in workers.iter() {
            worker.load().fetch_add(1, Ordering::Relaxed);
        }
        LoadGuard { workers }
    }
}

impl Drop for LoadGuard {
    fn drop(&mut self) {
        // Guaranteed cleanup even on panic
        for worker in self.workers.iter() {
            worker.load().fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl PDRouter {
    pub async fn add_worker(&self, worker: Arc<dyn Worker>) -> Result<String, PDRouterError> {
        crate::core::worker::utils::wait_for_healthy_workers(
            &[worker.clone()],
            self.timeout_secs,
            self.interval_secs,
        )
        .await
        .map_err(|_| PDRouterError::HealthCheckFailed {
            url: worker.url().to_string(),
        })?;

        let mut target_workers = match worker.worker_type() {
            WorkerType::Decode => {
                self.decode_workers
                    .write()
                    .map_err(|_| PDRouterError::LockError {
                        operation: "decode_workers write".to_string(),
                    })?
            }
            WorkerType::Prefill(_) => {
                self.prefill_workers
                    .write()
                    .map_err(|_| PDRouterError::LockError {
                        operation: "prefill_workers write".to_string(),
                    })?
            }
            other_type => {
                return Err(PDRouterError::InvalidWorkerType {
                    url: worker.url().to_string(),
                    worker_type: other_type.to_string(),
                })
            }
        };

        if target_workers.iter().any(|w| w.url() == worker.url()) {
            return Err(PDRouterError::WorkerAlreadyExists {
                url: worker.url().to_string(),
            });
        }
        target_workers.push(worker.clone());

        // Add to cache tree if using cache-aware policy and worker is a prefill worker
        if matches!(worker.worker_type(), WorkerType::Prefill(_)) && self.prefill_tree.is_some() {
            self.prefill_tree
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .insert("", worker.url());
        }
        info!("Added worker: {}", worker);
        Ok(format!("Successfully added worker: {worker}"))
    }

    // Dynamic worker management methods for service discovery
    pub async fn add_prefill_server(
        &self,
        url: String,
        bootstrap_port: Option<u16>,
    ) -> Result<String, PDRouterError> {
        let worker = WorkerFactory::create_prefill(url.clone(), bootstrap_port);
        return self.add_worker(worker).await;
    }

    pub async fn add_decode_server(&self, url: String) -> Result<String, PDRouterError> {
        let worker = WorkerFactory::create_decode(url.clone());
        return self.add_worker(worker).await;
    }

    pub async fn remove_decode_server(&self, url: &str) -> Result<String, PDRouterError> {
        let target_workers = self
            .decode_workers
            .write()
            .map_err(|_| PDRouterError::LockError {
                operation: "remove_workers write".to_string(),
            })?;
        self.remove_worker(url, target_workers)
    }

    pub fn unified_remove_worker(&self, worker: Arc<dyn Worker>) -> Result<String, PDRouterError> {
        let target_workers = match worker.worker_type() {
            WorkerType::Decode => {
                self.decode_workers
                    .write()
                    .map_err(|_| PDRouterError::LockError {
                        operation: "decode_workers write".to_string(),
                    })?
            }
            WorkerType::Prefill(_) => {
                self.prefill_workers
                    .write()
                    .map_err(|_| PDRouterError::LockError {
                        operation: "prefill_workers write".to_string(),
                    })?
            }
            other_type => {
                return Err(PDRouterError::InvalidWorkerType {
                    url: worker.url().to_string(),
                    worker_type: other_type.to_string(),
                })
            }
        };
        self.remove_worker(worker.url(), target_workers)
    }

    fn remove_worker(
        &self,
        url: &str,
        mut target_workers: RwLockWriteGuard<'_, Vec<Arc<dyn Worker>>>,
    ) -> Result<String, PDRouterError> {
        let worker = target_workers.iter().find(|w| w.url() == url);
        if worker.is_none() {
            return Err(PDRouterError::WorkerNotFound {
                url: url.to_string(),
            });
        }
        let worker = worker.unwrap().clone();
        target_workers.retain(|w| w.url() != url);

        // Remove from cache tree if using cache-aware policy and worker is a prefill worker
        if matches!(worker.worker_type(), WorkerType::Prefill(_)) && self.prefill_tree.is_some() {
            // Note: Tree doesn't have a remove method, so we rebuild it
            let tree = self.prefill_tree.as_ref().unwrap();
            let mut tree_guard = tree.lock().unwrap();
            *tree_guard = Tree::new();
            for worker in target_workers.iter() {
                if matches!(worker.worker_type(), WorkerType::Prefill(_)) {
                    tree_guard.insert("", worker.url());
                }
            }
        }

        info!("Removed worker: {}", worker);
        Ok(format!("Successfully removed worker: {worker}"))
    }

    pub async fn remove_prefill_server(&self, url: &str) -> Result<String, PDRouterError> {
        let target_workers =
            self.prefill_workers
                .write()
                .map_err(|_| PDRouterError::LockError {
                    operation: "prefill_workers write".to_string(),
                })?;
        self.remove_worker(url, target_workers)
    }

    pub fn new(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        selection_policy: PDSelectionPolicy,
        timeout_secs: u64,
        interval_secs: u64,
    ) -> Result<Self, String> {
        // Convert URLs to workers
        let prefill_workers: Vec<_> = prefill_urls
            .into_iter()
            .map(|(url, port)| WorkerFactory::create_prefill(url, port))
            .collect();
        let decode_workers: Vec<_> = decode_urls
            .into_iter()
            .map(|url| WorkerFactory::create_decode(url))
            .collect();

        let all_workers: Vec<_> = prefill_workers
            .iter()
            .chain(decode_workers.iter())
            .cloned()
            .collect();

        // Wait for PD workers to be healthy
        crate::core::worker::utils::wait_for_healthy_workers_sync(
            &all_workers,
            timeout_secs,
            interval_secs,
        )?;

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

        let prefill_workers_with_lock = Arc::new(RwLock::new(prefill_workers));
        let decode_workers_with_lock = Arc::new(RwLock::new(decode_workers));

        // Set up background load monitoring for power-of-two selection
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let worker_loads = Arc::new(rx);

        // Create a shared HTTP client for all operations
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {e}"))?;

        let load_monitor_handle = if matches!(selection_policy, PDSelectionPolicy::PowerOfTwo) {
            let monitor_interval = interval_secs;
            let monitor_client = http_client.clone();
            let monitor_prefill_workers = prefill_workers_with_lock.clone();
            let monitor_decode_workers = decode_workers_with_lock.clone();

            Some(Arc::new(tokio::spawn(async move {
                Self::monitor_worker_loads_with_client(
                    monitor_prefill_workers,
                    monitor_decode_workers,
                    tx,
                    monitor_interval,
                    monitor_client,
                )
                .await;
            })))
        } else {
            None
        };

        Ok(PDRouter {
            prefill_workers: prefill_workers_with_lock,
            decode_workers: decode_workers_with_lock,
            selection_policy,
            prefill_tree,
            timeout_secs,
            interval_secs,
            worker_loads,
            load_monitor_handle,
            http_client,
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
                    .body(format!("No available servers: {e}"));
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
        if let Err(e) = typed_req.add_bootstrap_info(&prefill) {
            error!("Failed to add bootstrap info: {}", e);
            counter!("sgl_router_pd_errors_total", "error" => "bootstrap_injection").increment(1);
            return HttpResponse::InternalServerError()
                .body(format!("Bootstrap injection failed: {e}"));
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
            prefill,
            decode,
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
                    .body(format!("No available servers: {e}"));
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
        if let Err(e) = typed_req.add_bootstrap_info(&prefill) {
            error!("Failed to add bootstrap info: {}", e);
            counter!("sgl_router_pd_errors_total", "error" => "bootstrap_injection").increment(1);
            return HttpResponse::InternalServerError()
                .body(format!("Bootstrap injection failed: {e}"));
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
            prefill,
            decode,
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
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        is_stream: bool,
        return_logprob: bool,
        start_time: Instant,
    ) -> HttpResponse {
        // Update load tracking for both workers
        let _guard = LoadGuard::new(vec![prefill.clone(), decode.clone()]);

        let prefill_url = prefill.url().to_string();
        let decode_url = decode.url().to_string();

        // Build requests using .json() method
        let mut prefill_request = client
            .post(api_path(&prefill_url, route))
            .json(&json_request);

        let mut decode_request = client
            .post(api_path(&decode_url, route))
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
        counter!("sgl_router_pd_prefill_requests_total", "worker" => prefill_url.clone())
            .increment(1);
        counter!("sgl_router_pd_decode_requests_total", "worker" => decode_url.clone())
            .increment(1);

        // Process decode response
        match decode_result {
            Ok(res) => {
                let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

                if !status.is_success() {
                    counter!("sgl_router_pd_decode_errors_total", "worker" => decode_url.clone())
                        .increment(1);
                    error!(
                        "Decode server {} returned error status: {}",
                        &decode_url, status
                    );

                    // Return the error response from decode server
                    match res.bytes().await {
                        Ok(error_body) => {
                            return HttpResponse::build(status).body(error_body.to_vec());
                        }
                        Err(e) => {
                            return HttpResponse::build(status)
                                .body(format!("Decode server error: {e}"));
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
                    counter!("sgl_router_pd_prefill_errors_total", "worker" => prefill_url)
                        .increment(1);
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
                                        format!("Stream error: {e}"),
                                    )),
                                }
                            }))
                    } else {
                        // No logprob merging needed
                        HttpResponse::build(status)
                            .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                            .streaming({
                                res.bytes_stream().map_err(move |e| {
                                    error!("Stream error from decode server {}: {}", &decode_url, e);
                                    counter!("sgl_router_pd_stream_errors_total", "worker" => decode_url.clone()).increment(1);
                                    actix_web::error::ErrorInternalServerError(format!("Stream error: {e}"))
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
                HttpResponse::BadGateway().body(format!("Decode server error: {e}"))
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
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        // Check we have workers
        if self
            .prefill_workers
            .read()
            .map_err(|e| format!("Failed to acquire prefill workers lock: {e}"))?
            .is_empty()
        {
            return Err("No prefill workers available. Please check if prefill servers are configured and healthy.".to_string());
        }
        if self
            .decode_workers
            .read()
            .map_err(|e| format!("Failed to acquire decode workers lock: {e}"))?
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

    fn select_random(&self) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        let prefill_list = self.prefill_workers.read().map_err(|_| "Lock error")?;
        let decode_list = self.decode_workers.read().map_err(|_| "Lock error")?;

        let prefill = prefill_list[rand::random::<usize>() % prefill_list.len()].clone();
        let decode = decode_list[rand::random::<usize>() % decode_list.len()].clone();

        Ok((prefill, decode))
    }

    async fn select_power_of_two(&self) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>), String> {
        let prefill_list = self.prefill_workers.read().map_err(|_| "Lock error")?;
        let decode_list = self.decode_workers.read().map_err(|_| "Lock error")?;

        let (p1_idx, p2_idx) = get_two_random_indices(prefill_list.len());
        let (d1_idx, d2_idx) = get_two_random_indices(decode_list.len());

        let loads = self.worker_loads.borrow();

        let p1_load = loads
            .get(&prefill_list[p1_idx].url().to_string())
            .copied()
            .unwrap_or(0);
        let p2_load = loads
            .get(&prefill_list[p2_idx].url().to_string())
            .copied()
            .unwrap_or(0);
        let d1_load = loads
            .get(&decode_list[d1_idx].url().to_string())
            .copied()
            .unwrap_or(0);
        let d2_load = loads
            .get(&decode_list[d2_idx].url().to_string())
            .copied()
            .unwrap_or(0);

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
            prefill_list[p1_idx].clone()
        } else {
            prefill_list[p2_idx].clone()
        };

        let selected_decode = if d1_load <= d2_load {
            decode_list[d1_idx].clone()
        } else {
            decode_list[d2_idx].clone()
        };

        Ok((selected_prefill, selected_decode))
    }

    // Background task to monitor worker loads with shared client
    async fn monitor_worker_loads_with_client(
        prefill_workers: Arc<RwLock<Vec<Arc<dyn Worker>>>>,
        decode_workers: Arc<RwLock<Vec<Arc<dyn Worker>>>>,
        tx: tokio::sync::watch::Sender<HashMap<String, isize>>,
        interval_secs: u64,
        client: reqwest::Client,
    ) {
        loop {
            let mut loads = HashMap::new();
            let workers: Vec<_> = {
                let prefill_workers_guard = prefill_workers.read().unwrap();
                let decode_workers_guard = decode_workers.read().unwrap();
                prefill_workers_guard
                    .iter()
                    .chain(decode_workers_guard.iter())
                    .cloned()
                    .collect()
            };

            let futures: Vec<_> = workers
                .iter()
                .map(|worker| {
                    let client = client.clone();
                    async move {
                        let load =
                            crate::core::worker::utils::get_worker_load(&client, worker).await;
                        (worker.url().to_string(), load.unwrap_or(0))
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
    match client.get(format!("{worker_url}/get_load")).send().await {
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
                let health_url = format!("{url}/health_generate");
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
                    let msg = format!("{worker_type} server {url} error: {e}");
                    error!("{}", msg);
                    unhealthy_servers.push(msg);
                }
            }
        }

        if all_healthy {
            HttpResponse::Ok().body("Health check passed on all servers")
        } else {
            HttpResponse::ServiceUnavailable()
                .body(format!("Health check failed: {unhealthy_servers:?}"))
        }
    }

    pub async fn get_server_info(&self, client: &reqwest::Client) -> HttpResponse {
        // Get info from all decode servers (where generation happens)
        let mut all_internal_states = Vec::new();
        let mut decode_infos = Vec::new();

        // Clone URLs to avoid holding lock across await
        let worker_urls: Vec<String> = self
            .decode_workers
            .read()
            .unwrap()
            .iter()
            .map(|w| w.url().to_string())
            .collect();

        for worker_url in worker_urls {
            match client
                .get(format!("{worker_url}/get_server_info"))
                .send()
                .await
            {
                Ok(res) if res.status().is_success() => {
                    match res.json::<Value>().await {
                        Ok(info) => {
                            // Extract internal_states from each decode server
                            if let Some(states) = info.get("internal_states") {
                                if let Some(states_array) = states.as_array() {
                                    all_internal_states.extend(states_array.clone());
                                }
                            }
                            decode_infos.push(info);
                        }
                        Err(e) => error!("Failed to parse server info: {}", e),
                    }
                }
                _ => {}
            }
        }

        // If we have internal states, return in the format expected by bench_one_batch_server.py
        if !all_internal_states.is_empty() {
            // Use the first decode server's internal state (they should all be similar)
            HttpResponse::Ok().json(serde_json::json!({
                "internal_states": all_internal_states,
                // Include original format for compatibility
                "decode_servers": decode_infos,
            }))
        } else {
            // Fallback: create a dummy internal_states entry
            HttpResponse::Ok().json(serde_json::json!({
                "internal_states": [{
                    "last_gen_throughput": 0.0,
                    "avg_spec_accept_length": null,
                }],
                "decode_servers": decode_infos,
            }))
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
            let mut request_builder = client.get(format!("{worker_url}/v1/models"));
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
                            .body(format!("Failed to read response body: {e}")),
                    }
                }
                Err(e) => HttpResponse::InternalServerError()
                    .body(format!("Failed to send request: {e}")),
            }
        } else {
            HttpResponse::ServiceUnavailable().body("No prefill servers available")
        }
    }

    pub async fn get_loads(&self, client: &reqwest::Client) -> HttpResponse {
        let prefill_workers = self.prefill_workers.read().unwrap().clone();
        let decode_workers = self.decode_workers.read().unwrap().clone();

        let (prefill_loads, decode_loads) =
            crate::router::get_loads_helper(client, prefill_workers, decode_workers).await;

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
            let mut request_builder = client.get(format!("{worker_url}/get_model_info"));
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
                            .body(format!("Failed to read response body: {e}")),
                    }
                }
                Err(e) => HttpResponse::InternalServerError()
                    .body(format!("Failed to send request: {e}")),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::worker::WorkerType;
    use crate::pd_types::PDSelectionPolicy;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    // ============================================================================
    // Test Helper Functions and Mock Servers
    // ============================================================================

    async fn create_mock_health_server(port: u16) -> String {
        let listener = TcpListener::bind(format!("127.0.0.1:{}", port))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let mut buffer = [0; 1024];
                    let _ = stream.read(&mut buffer).await;

                    let response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 13\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}";
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                    let _ = stream.shutdown().await;
                }
            }
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        format!("http://127.0.0.1:{}", addr.port())
    }

    async fn create_mock_load_server(load_value: isize) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let response_body = format!("{{\"load\": {}}}", load_value);
        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let mut buffer = [0; 1024];
                    let _ = stream.read(&mut buffer).await;

                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                    let _ = stream.shutdown().await;
                }
            }
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        format!("http://127.0.0.1:{}", addr.port())
    }

    async fn create_mock_generate_server(response_body: &str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let response_body = response_body.to_string();
        tokio::spawn(async move {
            loop {
                if let Ok((mut stream, _)) = listener.accept().await {
                    let mut buffer = [0; 1024];
                    let _ = stream.read(&mut buffer).await;

                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        response_body.len(),
                        response_body
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                    let _ = stream.flush().await;
                    let _ = stream.shutdown().await;
                }
            }
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        format!("http://127.0.0.1:{}", addr.port())
    }

    fn create_test_pd_router_with_mocks() -> PDRouter {
        let prefill_workers = vec![
            WorkerFactory::create_prefill("http://prefill1:8080".to_string(), Some(8081)),
            WorkerFactory::create_prefill("http://prefill2:8080".to_string(), Some(8082)),
        ];
        let decode_workers = vec![
            WorkerFactory::create_decode("http://decode1:8080".to_string()),
            WorkerFactory::create_decode("http://decode2:8080".to_string()),
        ];

        let (_tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let worker_loads = Arc::new(rx);

        PDRouter {
            prefill_workers: Arc::new(RwLock::new(prefill_workers)),
            decode_workers: Arc::new(RwLock::new(decode_workers)),
            selection_policy: PDSelectionPolicy::Random,
            prefill_tree: None,
            timeout_secs: 5,
            interval_secs: 1,
            worker_loads,
            load_monitor_handle: None,
            http_client: reqwest::Client::new(),
        }
    }

    fn create_test_cache_aware_pd_router() -> PDRouter {
        let prefill_workers = vec![
            WorkerFactory::create_prefill("http://prefill1:8080".to_string(), Some(8081)),
            WorkerFactory::create_prefill("http://prefill2:8080".to_string(), Some(8082)),
        ];
        let decode_workers = vec![
            WorkerFactory::create_decode("http://decode1:8080".to_string()),
            WorkerFactory::create_decode("http://decode2:8080".to_string()),
        ];

        let tree = Arc::new(Mutex::new(Tree::new()));
        for worker in &prefill_workers {
            tree.lock().unwrap().insert("", worker.url());
        }

        let (_tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let worker_loads = Arc::new(rx);

        PDRouter {
            prefill_workers: Arc::new(RwLock::new(prefill_workers)),
            decode_workers: Arc::new(RwLock::new(decode_workers)),
            selection_policy: PDSelectionPolicy::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 10,
                balance_rel_threshold: 1.5,
            },
            prefill_tree: Some(tree),
            timeout_secs: 5,
            interval_secs: 1,
            worker_loads,
            load_monitor_handle: None,
            http_client: reqwest::Client::new(),
        }
    }

    // ============================================================================
    // Basic PDRouter Tests
    // ============================================================================

    #[test]
    fn test_pd_router_creation() {
        let router = create_test_pd_router_with_mocks();

        assert!(router.prefill_workers.read().unwrap().len() == 2);
        assert!(router.decode_workers.read().unwrap().len() == 2);
        assert!(matches!(router.selection_policy, PDSelectionPolicy::Random));
        assert!(router.prefill_tree.is_none());
    }

    #[test]
    fn test_cache_aware_pd_router_creation() {
        let router = create_test_cache_aware_pd_router();

        assert!(router.prefill_workers.read().unwrap().len() == 2);
        assert!(router.decode_workers.read().unwrap().len() == 2);
        assert!(matches!(
            router.selection_policy,
            PDSelectionPolicy::CacheAware { .. }
        ));
        assert!(router.prefill_tree.is_some());
    }

    // ============================================================================
    // Worker Selection Tests
    // ============================================================================

    #[tokio::test]
    async fn test_select_random_pd_pair() {
        let router = create_test_pd_router_with_mocks();
        let client = reqwest::Client::new();

        let result = router.select_pd_pair(&client).await;
        assert!(result.is_ok());

        let (prefill, decode) = result.unwrap();
        assert!(matches!(prefill.worker_type(), WorkerType::Prefill(_)));
        assert!(matches!(decode.worker_type(), WorkerType::Decode));
    }

    #[tokio::test]
    async fn test_select_pd_pair_empty_prefill_workers() {
        let router = create_test_pd_router_with_mocks();
        *router.prefill_workers.write().unwrap() = vec![];

        let client = reqwest::Client::new();
        let result = router.select_pd_pair(&client).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No prefill workers available"));
    }

    #[tokio::test]
    async fn test_select_pd_pair_empty_decode_workers() {
        let router = create_test_pd_router_with_mocks();
        *router.decode_workers.write().unwrap() = vec![];

        let client = reqwest::Client::new();
        let result = router.select_pd_pair(&client).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No decode workers available"));
    }

    #[tokio::test]
    async fn test_select_power_of_two() {
        let mut router = create_test_pd_router_with_mocks();
        router.selection_policy = PDSelectionPolicy::PowerOfTwo;

        // Set up load data
        let mut load_data = HashMap::new();
        load_data.insert("http://prefill1:8080".to_string(), 5);
        load_data.insert("http://prefill2:8080".to_string(), 10);
        load_data.insert("http://decode1:8080".to_string(), 3);
        load_data.insert("http://decode2:8080".to_string(), 7);

        // Update the watch channel
        let (_tx, rx) = tokio::sync::watch::channel(load_data);
        router.worker_loads = Arc::new(rx);

        let result = router.select_power_of_two().await;
        assert!(result.is_ok());

        let (prefill, decode) = result.unwrap();
        assert!(matches!(prefill.worker_type(), WorkerType::Prefill(_)));
        assert!(matches!(decode.worker_type(), WorkerType::Decode));
    }

    // ============================================================================
    // Worker Management Tests
    // ============================================================================

    #[tokio::test]
    async fn test_add_worker_health_check_failure() {
        let router = create_test_pd_router_with_mocks();

        // Create a worker with non-existent URL - should fail health check
        let failing_worker =
            WorkerFactory::create_prefill("http://nonexistent:8080".to_string(), Some(8081));

        let result = router.add_worker(failing_worker).await;

        // Should fail because health check fails
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PDRouterError::HealthCheckFailed { .. }
        ));
    }

    #[tokio::test]
    async fn test_add_worker_duplicate_url() {
        // Create a mock server that will pass health checks
        let mock_url = create_mock_health_server(0).await;
        let router = create_test_pd_router_with_mocks();

        // First, add a worker with the mock URL to the router manually
        let new_worker = WorkerFactory::create_prefill(mock_url.clone(), Some(8081));
        router.prefill_workers.write().unwrap().push(new_worker);

        // Now try to add another worker with the same URL - should fail as duplicate
        let duplicate_worker = WorkerFactory::create_prefill(mock_url, Some(8081));
        let result = router.add_worker(duplicate_worker).await;

        // Should fail because worker already exists (after health check passes)
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PDRouterError::WorkerAlreadyExists { .. }
        ));
    }

    #[test]
    fn test_worker_creation_and_management() {
        let router = create_test_pd_router_with_mocks();

        // Test initial state
        assert_eq!(router.prefill_workers.read().unwrap().len(), 2);
        assert_eq!(router.decode_workers.read().unwrap().len(), 2);

        // Test that we can create workers with proper types
        let prefill_worker =
            WorkerFactory::create_prefill("http://new-prefill:8080".to_string(), Some(8083));
        let decode_worker = WorkerFactory::create_decode("http://new-decode:8080".to_string());

        assert!(matches!(
            prefill_worker.worker_type(),
            WorkerType::Prefill(_)
        ));
        assert!(matches!(decode_worker.worker_type(), WorkerType::Decode));

        // Test URL generation
        assert_eq!(prefill_worker.url(), "http://new-prefill:8080");
        assert_eq!(decode_worker.url(), "http://new-decode:8080");
    }

    #[tokio::test]
    async fn test_remove_prefill_server() {
        let router = create_test_pd_router_with_mocks();

        let result = router.remove_prefill_server("http://prefill1:8080").await;
        assert!(result.is_ok());

        // Verify worker was removed
        let workers = router.prefill_workers.read().unwrap();
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].url(), "http://prefill2:8080");
    }

    #[tokio::test]
    async fn test_remove_decode_server() {
        let router = create_test_pd_router_with_mocks();

        let result = router.remove_decode_server("http://decode1:8080").await;
        assert!(result.is_ok());

        // Verify worker was removed
        let workers = router.decode_workers.read().unwrap();
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0].url(), "http://decode2:8080");
    }

    #[tokio::test]
    async fn test_remove_nonexistent_worker() {
        let router = create_test_pd_router_with_mocks();

        let result = router
            .remove_prefill_server("http://nonexistent:8080")
            .await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PDRouterError::WorkerNotFound { .. }
        ));
    }

    // ============================================================================
    // Load Guard Tests
    // ============================================================================

    #[test]
    fn test_load_guard_creation_and_drop() {
        let workers = vec![
            WorkerFactory::create_decode("http://worker1:8080".to_string()),
            WorkerFactory::create_prefill("http://worker2:8080".to_string(), Some(8081)),
        ];

        // Check initial load values
        assert_eq!(workers[0].load().load(Ordering::Relaxed), 0);
        assert_eq!(workers[1].load().load(Ordering::Relaxed), 0);

        {
            let _guard = LoadGuard::new(workers.clone());
            // Load should be incremented
            assert_eq!(workers[0].load().load(Ordering::Relaxed), 1);
            assert_eq!(workers[1].load().load(Ordering::Relaxed), 1);
        } // Guard drops here

        // Load should be decremented back to 0
        assert_eq!(workers[0].load().load(Ordering::Relaxed), 0);
        assert_eq!(workers[1].load().load(Ordering::Relaxed), 0);
    }

    // ============================================================================
    // Helper Function Tests
    // ============================================================================

    #[test]
    fn test_get_two_random_indices() {
        // Test with multiple elements
        let (idx1, idx2) = get_two_random_indices(5);
        assert!(idx1 < 5);
        assert!(idx2 < 5);
        assert_ne!(idx1, idx2);

        // Test with single element
        let (idx1, idx2) = get_two_random_indices(1);
        assert_eq!(idx1, 0);
        assert_eq!(idx2, 0);
    }

    #[tokio::test]
    async fn test_get_worker_load_success() {
        let mock_url = create_mock_load_server(42).await;
        let client = reqwest::Client::new();

        let load = get_worker_load(&client, &mock_url).await;
        assert_eq!(load, Some(42));
    }

    #[tokio::test]
    async fn test_get_worker_load_failure() {
        let client = reqwest::Client::new();

        let load = get_worker_load(&client, "http://nonexistent:8080").await;
        assert_eq!(load, None);
    }

    // ============================================================================
    // Logprob Merging Tests
    // ============================================================================

    #[test]
    fn test_merge_streaming_logprobs_success() {
        let prefill_logprobs = Some(serde_json::json!([1.0, 2.0, 3.0]));
        let decode_chunk = b"data: {\"meta_info\": {\"input_token_logprobs\": [4.0, 5.0]}}\n\n";

        let result = PDRouter::merge_streaming_logprobs(prefill_logprobs, decode_chunk);
        assert!(result.is_ok());

        let merged = result.unwrap();
        let merged_str = std::str::from_utf8(&merged).unwrap();
        assert!(merged_str.contains("data:"));
        assert!(merged_str.contains("1.0"));
        assert!(merged_str.contains("4.0"));
    }

    #[test]
    fn test_merge_streaming_logprobs_skip_non_data() {
        let prefill_logprobs = Some(serde_json::json!([1.0, 2.0, 3.0]));
        let non_data_chunk = b"event: ping\n\n";

        let result = PDRouter::merge_streaming_logprobs(prefill_logprobs, non_data_chunk);
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_streaming_logprobs_done_chunk() {
        let prefill_logprobs = Some(serde_json::json!([1.0, 2.0, 3.0]));
        let done_chunk = b"data: [DONE]\n\n";

        let result = PDRouter::merge_streaming_logprobs(prefill_logprobs, done_chunk);
        assert!(result.is_err());
    }

    // ============================================================================
    // Bootstrap and Request Tests
    // ============================================================================

    #[test]
    fn test_generate_request_bootstrap() {
        use crate::pd_types::{GenerateReqInput, InputText};

        let prefill_worker =
            WorkerFactory::create_prefill("http://prefill:8080".to_string(), Some(8081));
        let mut req = GenerateReqInput {
            text: Some(InputText::Single("test input".to_string())),
            input_ids: None,
            stream: false,
            bootstrap_host: None,
            bootstrap_port: None,
            bootstrap_room: None,
            other: serde_json::Value::Object(serde_json::Map::new()),
        };

        let result = req.add_bootstrap_info(&prefill_worker);
        assert!(result.is_ok());

        // Check that bootstrap info was added
        assert!(req.bootstrap_host.is_some());
        assert!(req.bootstrap_port.is_some());
        assert!(req.bootstrap_room.is_some());
    }

    #[test]
    fn test_chat_request_bootstrap() {
        use crate::pd_types::ChatReqInput;

        let prefill_worker =
            WorkerFactory::create_prefill("http://prefill:8080".to_string(), Some(8081));
        let mut req = ChatReqInput {
            stream: false,
            bootstrap_host: None,
            bootstrap_port: None,
            bootstrap_room: None,
            other: serde_json::Value::Object(serde_json::Map::new()),
        };

        let result = req.add_bootstrap_info(&prefill_worker);
        assert!(result.is_ok());

        // Check that bootstrap info was added
        assert!(req.bootstrap_host.is_some());
        assert!(req.bootstrap_port.is_some());
        assert!(req.bootstrap_room.is_some());
    }

    // ============================================================================
    // Integration Tests with Mock Servers
    // ============================================================================

    #[tokio::test]
    async fn test_health_generate_all_healthy() {
        let prefill_url = create_mock_health_server(0).await;
        let decode_url = create_mock_health_server(0).await;

        let router = PDRouter {
            prefill_workers: Arc::new(RwLock::new(vec![WorkerFactory::create_prefill(
                prefill_url,
                Some(8081),
            )])),
            decode_workers: Arc::new(RwLock::new(vec![WorkerFactory::create_decode(decode_url)])),
            selection_policy: PDSelectionPolicy::Random,
            prefill_tree: None,
            timeout_secs: 5,
            interval_secs: 1,
            worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
            load_monitor_handle: None,
            http_client: reqwest::Client::new(),
        };

        let client = reqwest::Client::new();
        let response = router.health_generate(&client).await;

        assert_eq!(response.status(), actix_web::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_loads() {
        let prefill_url = create_mock_load_server(10).await;
        let decode_url = create_mock_load_server(20).await;

        let router = PDRouter {
            prefill_workers: Arc::new(RwLock::new(vec![WorkerFactory::create_prefill(
                prefill_url,
                Some(8081),
            )])),
            decode_workers: Arc::new(RwLock::new(vec![WorkerFactory::create_decode(decode_url)])),
            selection_policy: PDSelectionPolicy::Random,
            prefill_tree: None,
            timeout_secs: 5,
            interval_secs: 1,
            worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
            load_monitor_handle: None,
            http_client: reqwest::Client::new(),
        };

        let client = reqwest::Client::new();
        let response = router.get_loads(&client).await;

        assert_eq!(response.status(), actix_web::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_models() {
        let model_response = r#"{"data": [{"id": "test-model"}]}"#;
        let mock_url = create_mock_generate_server(model_response).await;

        let router = PDRouter {
            prefill_workers: Arc::new(RwLock::new(vec![WorkerFactory::create_prefill(
                mock_url,
                Some(8081),
            )])),
            decode_workers: Arc::new(RwLock::new(vec![])),
            selection_policy: PDSelectionPolicy::Random,
            prefill_tree: None,
            timeout_secs: 5,
            interval_secs: 1,
            worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
            load_monitor_handle: None,
            http_client: reqwest::Client::new(),
        };

        let client = reqwest::Client::new();
        let req = actix_web::test::TestRequest::get().to_http_request();
        let response = router.get_models(&client, &req).await;

        assert_eq!(response.status(), actix_web::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_server_info() {
        let server_info_response = r#"{"internal_states": [{"last_gen_throughput": 100.0}]}"#;
        let mock_url = create_mock_generate_server(server_info_response).await;

        let router = PDRouter {
            prefill_workers: Arc::new(RwLock::new(vec![])),
            decode_workers: Arc::new(RwLock::new(vec![WorkerFactory::create_decode(mock_url)])),
            selection_policy: PDSelectionPolicy::Random,
            prefill_tree: None,
            timeout_secs: 5,
            interval_secs: 1,
            worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
            load_monitor_handle: None,
            http_client: reqwest::Client::new(),
        };

        let client = reqwest::Client::new();
        let response = router.get_server_info(&client).await;

        assert_eq!(response.status(), actix_web::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_model_info() {
        let model_info_response = r#"{"model_name": "test-model", "vocab_size": 50000}"#;
        let mock_url = create_mock_generate_server(model_info_response).await;

        let router = PDRouter {
            prefill_workers: Arc::new(RwLock::new(vec![WorkerFactory::create_prefill(
                mock_url,
                Some(8081),
            )])),
            decode_workers: Arc::new(RwLock::new(vec![])),
            selection_policy: PDSelectionPolicy::Random,
            prefill_tree: None,
            timeout_secs: 5,
            interval_secs: 1,
            worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
            load_monitor_handle: None,
            http_client: reqwest::Client::new(),
        };

        let client = reqwest::Client::new();
        let req = actix_web::test::TestRequest::get().to_http_request();
        let response = router.get_model_info(&client, &req).await;

        assert_eq!(response.status(), actix_web::http::StatusCode::OK);
    }

    // ============================================================================
    // Edge Cases
    // ============================================================================

    #[tokio::test]
    async fn test_empty_workers_health_check() {
        let router = PDRouter {
            prefill_workers: Arc::new(RwLock::new(vec![])),
            decode_workers: Arc::new(RwLock::new(vec![])),
            selection_policy: PDSelectionPolicy::Random,
            prefill_tree: None,
            timeout_secs: 5,
            interval_secs: 1,
            worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
            load_monitor_handle: None,
            http_client: reqwest::Client::new(),
        };

        let client = reqwest::Client::new();
        let response = router.health_generate(&client).await;

        // Should still return OK even with no workers
        assert_eq!(response.status(), actix_web::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_models_no_prefill_workers() {
        let router = PDRouter {
            prefill_workers: Arc::new(RwLock::new(vec![])),
            decode_workers: Arc::new(RwLock::new(vec![])),
            selection_policy: PDSelectionPolicy::Random,
            prefill_tree: None,
            timeout_secs: 5,
            interval_secs: 1,
            worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
            load_monitor_handle: None,
            http_client: reqwest::Client::new(),
        };

        let client = reqwest::Client::new();
        let req = actix_web::test::TestRequest::get().to_http_request();
        let response = router.get_models(&client, &req).await;

        assert_eq!(
            response.status(),
            actix_web::http::StatusCode::SERVICE_UNAVAILABLE
        );
    }

    // ============================================================================
    // Cache-Aware Operations Tests
    // ============================================================================

    #[tokio::test]
    async fn test_add_prefill_worker_cache_aware() {
        let router = create_test_cache_aware_pd_router();
        let mock_url = create_mock_health_server(0).await;

        // Get initial tree state
        let initial_tree_size = if let Some(tree) = &router.prefill_tree {
            tree.lock().unwrap().get_used_size_per_tenant().len()
        } else {
            0
        };

        let initial_worker_count = router.prefill_workers.read().unwrap().len();

        // Add a new prefill worker
        let result = router
            .add_prefill_server(mock_url.clone(), Some(8084))
            .await;
        assert!(result.is_ok());

        // Verify worker was added
        let workers = router.prefill_workers.read().unwrap();
        assert_eq!(workers.len(), initial_worker_count + 1);
        assert!(workers.iter().any(|w| w.url() == mock_url));

        // Verify worker was added to cache tree
        if let Some(tree) = &router.prefill_tree {
            let tree_tenants = tree.lock().unwrap().get_used_size_per_tenant();
            assert_eq!(tree_tenants.len(), initial_tree_size + 1);
            assert!(tree_tenants.contains_key(&mock_url));
        }
    }

    #[tokio::test]
    async fn test_remove_prefill_worker_cache_aware() {
        let router = create_test_cache_aware_pd_router();

        // Get initial state
        let initial_worker_count = router.prefill_workers.read().unwrap().len();
        let worker_to_remove = router.prefill_workers.read().unwrap()[0].url().to_string();

        // Remove the prefill worker
        let result = router.remove_prefill_server(&worker_to_remove).await;
        assert!(result.is_ok());

        // Verify worker was removed
        let workers = router.prefill_workers.read().unwrap();
        assert_eq!(workers.len(), initial_worker_count - 1);
        assert!(!workers.iter().any(|w| w.url() == worker_to_remove));

        // Verify worker was removed from cache tree
        if let Some(tree) = &router.prefill_tree {
            let tree_tenants = tree.lock().unwrap().get_used_size_per_tenant();
            // Tree should still have the remaining prefill worker
            assert_eq!(tree_tenants.len(), workers.len());
            assert!(!tree_tenants.contains_key(&worker_to_remove));
        }
    }

    #[tokio::test]
    async fn test_add_decode_worker_no_cache_impact() {
        let router = create_test_cache_aware_pd_router();
        let mock_url = create_mock_health_server(0).await;

        // Get initial tree state
        let initial_tree_size = if let Some(tree) = &router.prefill_tree {
            tree.lock().unwrap().get_used_size_per_tenant().len()
        } else {
            0
        };

        let initial_worker_count = router.decode_workers.read().unwrap().len();

        // Add a new decode worker
        let result = router.add_decode_server(mock_url.clone()).await;
        assert!(result.is_ok());

        // Verify decode worker was added
        let workers = router.decode_workers.read().unwrap();
        assert_eq!(workers.len(), initial_worker_count + 1);
        assert!(workers.iter().any(|w| w.url() == mock_url));

        // Verify cache tree was NOT affected (decode workers don't use cache)
        if let Some(tree) = &router.prefill_tree {
            let tree_tenants = tree.lock().unwrap().get_used_size_per_tenant();
            assert_eq!(tree_tenants.len(), initial_tree_size);
            assert!(!tree_tenants.contains_key(&mock_url));
        }
    }

    #[tokio::test]
    async fn test_cache_aware_selection_with_multiple_workers() {
        let mut router = create_test_cache_aware_pd_router();

        // Set up power-of-two selection for comparison
        router.selection_policy = PDSelectionPolicy::PowerOfTwo;

        // Set up load data to test selection
        let mut load_data = HashMap::new();
        load_data.insert("http://prefill1:8080".to_string(), 5);
        load_data.insert("http://prefill2:8080".to_string(), 10);
        load_data.insert("http://decode1:8080".to_string(), 3);
        load_data.insert("http://decode2:8080".to_string(), 7);

        let (_tx, rx) = tokio::sync::watch::channel(load_data);
        router.worker_loads = Arc::new(rx);

        // Test that power-of-two selection works
        let result = router.select_power_of_two().await;
        assert!(result.is_ok());

        let (prefill, decode) = result.unwrap();

        // Should select workers with lower load
        assert!(prefill.url() == "http://prefill1:8080"); // load 5 < 10
        assert!(decode.url() == "http://decode1:8080"); // load 3 < 7
    }

    #[test]
    fn test_cache_aware_pd_router_tree_initialization() {
        let router = create_test_cache_aware_pd_router();

        // Verify cache tree exists and is properly initialized
        assert!(router.prefill_tree.is_some());

        if let Some(tree) = &router.prefill_tree {
            let tree_tenants = tree.lock().unwrap().get_used_size_per_tenant();

            // Should have entries for both prefill workers
            assert_eq!(tree_tenants.len(), 2);
            assert!(tree_tenants.contains_key("http://prefill1:8080"));
            assert!(tree_tenants.contains_key("http://prefill2:8080"));
        }
    }

    #[tokio::test]
    async fn test_cache_aware_worker_selection_policy() {
        let router = create_test_cache_aware_pd_router();

        // Verify the selection policy is properly set
        if let PDSelectionPolicy::CacheAware {
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
        } = &router.selection_policy
        {
            assert_eq!(*cache_threshold, 0.5);
            assert_eq!(*balance_abs_threshold, 10);
            assert_eq!(*balance_rel_threshold, 1.5);
        } else {
            panic!("Expected CacheAware selection policy");
        }
    }

    // ============================================================================
    // Error Handling Tests
    // ============================================================================

    #[test]
    fn test_pd_router_error_types() {
        let health_check_error = PDRouterError::HealthCheckFailed {
            url: "http://test:8080".to_string(),
        };
        assert!(format!("{:?}", health_check_error).contains("HealthCheckFailed"));

        let worker_not_found_error = PDRouterError::WorkerNotFound {
            url: "http://test:8080".to_string(),
        };
        assert!(format!("{:?}", worker_not_found_error).contains("WorkerNotFound"));

        let lock_error = PDRouterError::LockError {
            operation: "test operation".to_string(),
        };
        assert!(format!("{:?}", lock_error).contains("LockError"));

        let worker_already_exists_error = PDRouterError::WorkerAlreadyExists {
            url: "http://test:8080".to_string(),
        };
        assert!(format!("{:?}", worker_already_exists_error).contains("WorkerAlreadyExists"));

        let invalid_worker_type_error = PDRouterError::InvalidWorkerType {
            url: "http://test:8080".to_string(),
            worker_type: "Regular".to_string(),
        };
        assert!(format!("{:?}", invalid_worker_type_error).contains("InvalidWorkerType"));
    }

    #[tokio::test]
    async fn test_flush_cache() {
        let mock_prefill_url = create_mock_generate_server("{}").await;
        let mock_decode_url = create_mock_generate_server("{}").await;

        let router = PDRouter {
            prefill_workers: Arc::new(RwLock::new(vec![WorkerFactory::create_prefill(
                mock_prefill_url,
                Some(8081),
            )])),
            decode_workers: Arc::new(RwLock::new(vec![WorkerFactory::create_decode(
                mock_decode_url,
            )])),
            selection_policy: PDSelectionPolicy::Random,
            prefill_tree: None,
            timeout_secs: 5,
            interval_secs: 1,
            worker_loads: Arc::new(tokio::sync::watch::channel(HashMap::new()).1),
            load_monitor_handle: None,
            http_client: reqwest::Client::new(),
        };

        let client = reqwest::Client::new();
        let response = router.flush_cache(&client).await;

        assert_eq!(response.status(), actix_web::http::StatusCode::OK);
    }

    // ============================================================================
    // Load Balancing Integration Tests
    // ============================================================================

    #[tokio::test]
    async fn test_cache_aware_with_load_balancing() {
        let router = create_test_cache_aware_pd_router();

        // Test that cache-aware selection falls back to power-of-two when needed
        // This simulates the behavior in the actual cache-aware implementation
        let client = reqwest::Client::new();

        // Test multiple selections to ensure they don't panic
        for _ in 0..5 {
            let result = router.select_pd_pair(&client).await;
            assert!(result.is_ok());

            let (prefill, decode) = result.unwrap();
            assert!(matches!(prefill.worker_type(), WorkerType::Prefill(_)));
            assert!(matches!(decode.worker_type(), WorkerType::Decode));
        }
    }
}
