// PD (Prefill-Decode) Router Implementation
// This module handles routing for disaggregated prefill-decode systems

use crate::pd_types::{Bootstrap, ChatReqInput, EngineInfo, GenerateReqInput, PDSelectionPolicy};
use crate::tree::Tree;
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use futures_util::{StreamExt, TryStreamExt};
use metrics::{counter, histogram};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// Removed over-engineered ProxyResponse - using HttpResponse directly

#[derive(Debug)]
pub struct PDRouter {
    pub prefill_workers: Arc<RwLock<Vec<EngineInfo>>>,
    pub decode_workers: Arc<RwLock<Vec<EngineInfo>>>,
    pub selection_policy: PDSelectionPolicy,
    pub load_tracking: Arc<dashmap::DashMap<String, Arc<AtomicUsize>>>,
    pub prefill_tree: Option<Arc<Mutex<Tree>>>,
    pub timeout_secs: u64,
    pub interval_secs: u64,
    pub worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, isize>>>,
    pub load_monitor_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    pub http_client: reqwest::Client,
}

// RAII guard for load tracking to ensure cleanup even on panic
struct LoadGuard<'a> {
    tracking: &'a Arc<dashmap::DashMap<String, Arc<AtomicUsize>>>,
    urls: Vec<String>,
}

impl<'a> LoadGuard<'a> {
    fn new(
        tracking: &'a Arc<dashmap::DashMap<String, Arc<AtomicUsize>>>,
        urls: Vec<String>,
    ) -> Self {
        // Increment counters
        for url in &urls {
            let counter = tracking
                .entry(url.clone())
                .or_insert_with(|| Arc::new(AtomicUsize::new(0)));
            counter.fetch_add(1, Ordering::Relaxed);
        }
        LoadGuard { tracking, urls }
    }
}

impl Drop for LoadGuard<'_> {
    fn drop(&mut self) {
        // Guaranteed cleanup even on panic
        for url in &self.urls {
            if let Some(counter) = self.tracking.get(url) {
                counter.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

impl PDRouter {
    // TODO: Add methods for dynamic worker management to support /register endpoint:
    // - add_prefill_server(url: String, bootstrap_port: Option<u16>)
    // - add_decode_server(url: String)
    // - remove_prefill_server(url: &str)
    // - remove_decode_server(url: &str)
    // These methods will be used when service discovery is implemented for PD mode

    pub fn new(
        prefill_urls: Vec<(String, Option<u16>)>,
        decode_urls: Vec<String>,
        selection_policy: PDSelectionPolicy,
        timeout_secs: u64,
        interval_secs: u64,
    ) -> Result<Self, String> {
        // Convert URLs to EngineInfo
        let prefill_workers: Vec<EngineInfo> = prefill_urls
            .into_iter()
            .map(|(url, port)| EngineInfo::new_prefill(url, port))
            .collect();

        let decode_workers: Vec<EngineInfo> = decode_urls
            .into_iter()
            .map(EngineInfo::new_decode)
            .collect();

        // Wait for PD workers to be healthy
        let all_urls: Vec<String> = prefill_workers
            .iter()
            .chain(decode_workers.iter())
            .map(|engine| engine.url.clone())
            .collect();
        crate::router::Router::wait_for_healthy_workers(&all_urls, timeout_secs, interval_secs)?;

        // Initialize load tracking with atomic counters
        let load_tracking = Arc::new(dashmap::DashMap::new());
        for engine in &prefill_workers {
            load_tracking.insert(engine.url.clone(), Arc::new(AtomicUsize::new(0)));
        }
        for engine in &decode_workers {
            load_tracking.insert(engine.url.clone(), Arc::new(AtomicUsize::new(0)));
        }

        // Initialize cache-aware components if needed
        let prefill_tree = match &selection_policy {
            PDSelectionPolicy::CacheAware { .. } => {
                let tree = Arc::new(Mutex::new(Tree::new()));
                // Initialize tree with prefill workers
                for engine in &prefill_workers {
                    tree.lock().unwrap().insert("", &engine.url);
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

        Ok(PDRouter {
            prefill_workers: Arc::new(RwLock::new(prefill_workers)),
            decode_workers: Arc::new(RwLock::new(decode_workers)),
            selection_policy,
            load_tracking,
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
                    .body(format!("No available servers: {}", e));
            }
        };

        // Log routing decision
        info!(
            "PD routing: {} -> prefill={}, decode={}",
            route, prefill.url, decode.url
        );

        // Add bootstrap info using the trait method
        if let Err(e) = typed_req.add_bootstrap_info(&prefill) {
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
            &prefill,
            &decode,
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
            route, prefill.url, decode.url
        );

        // Add bootstrap info using the trait method
        if let Err(e) = typed_req.add_bootstrap_info(&prefill) {
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
            &prefill,
            &decode,
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
        prefill: &EngineInfo,
        decode: &EngineInfo,
        is_stream: bool,
        return_logprob: bool,
        start_time: Instant,
    ) -> HttpResponse {
        // Update load tracking for both workers
        let _guard = LoadGuard::new(
            &self.load_tracking,
            vec![prefill.url.clone(), decode.url.clone()],
        );

        // Build requests using .json() method
        let mut prefill_request = client.post(prefill.api_path(route)).json(&json_request);

        let mut decode_request = client.post(decode.api_path(route)).json(&json_request);

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
        counter!("sgl_router_pd_prefill_requests_total", "worker" => prefill.url.to_string())
            .increment(1);
        counter!("sgl_router_pd_decode_requests_total", "worker" => decode.url.to_string())
            .increment(1);

        // Process decode response
        match decode_result {
            Ok(res) => {
                let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

                if !status.is_success() {
                    counter!("sgl_router_pd_decode_errors_total", "worker" => decode.url.to_string()).increment(1);
                    error!(
                        "Decode server {} returned error status: {}",
                        decode.url, status
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
                        prefill.url, e
                    );
                    counter!("sgl_router_pd_prefill_errors_total", "worker" => prefill.url.to_string()).increment(1);
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
                                let decode_url = decode.url.clone();
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
                counter!("sgl_router_pd_decode_errors_total", "worker" => decode.url.to_string())
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
    ) -> Result<(EngineInfo, EngineInfo), String> {
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

    fn select_random(&self) -> Result<(EngineInfo, EngineInfo), String> {
        let prefill_list = self.prefill_workers.read().map_err(|_| "Lock error")?;
        let decode_list = self.decode_workers.read().map_err(|_| "Lock error")?;

        let prefill = prefill_list[rand::random::<usize>() % prefill_list.len()].clone();
        let decode = decode_list[rand::random::<usize>() % decode_list.len()].clone();

        Ok((prefill, decode))
    }

    async fn select_power_of_two(&self) -> Result<(EngineInfo, EngineInfo), String> {
        let prefill_list = self.prefill_workers.read().map_err(|_| "Lock error")?;
        let decode_list = self.decode_workers.read().map_err(|_| "Lock error")?;

        let (p1_idx, p2_idx) = get_two_random_indices(prefill_list.len());
        let (d1_idx, d2_idx) = get_two_random_indices(decode_list.len());

        let loads = self.worker_loads.borrow();

        let p1_load = loads.get(&prefill_list[p1_idx].url).copied().unwrap_or(0);
        let p2_load = loads.get(&prefill_list[p2_idx].url).copied().unwrap_or(0);
        let d1_load = loads.get(&decode_list[d1_idx].url).copied().unwrap_or(0);
        let d2_load = loads.get(&decode_list[d2_idx].url).copied().unwrap_or(0);

        info!(
            "Power-of-two selection - Prefill: {}={} vs {}={} | Decode: {}={} vs {}={}",
            prefill_list[p1_idx].url,
            p1_load,
            prefill_list[p2_idx].url,
            p2_load,
            decode_list[d1_idx].url,
            d1_load,
            decode_list[d2_idx].url,
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
            worker_infos.push((worker.url.clone(), "prefill"));
        }

        for worker in self.decode_workers.read().unwrap().iter() {
            worker_infos.push((worker.url.clone(), "decode"));
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
        // Get info from all decode servers (where generation happens)
        let mut all_internal_states = Vec::new();
        let mut decode_infos = Vec::new();

        // Clone URLs to avoid holding lock across await
        let worker_urls: Vec<String> = self
            .decode_workers
            .read()
            .unwrap()
            .iter()
            .map(|w| w.url.clone())
            .collect();

        for worker_url in worker_urls {
            match client
                .get(format!("{}/get_server_info", worker_url))
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
            workers.first().map(|w| w.url.clone())
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
            .map(|w| w.url.clone())
            .collect();
        let d_urls: Vec<_> = self
            .decode_workers
            .read()
            .unwrap()
            .iter()
            .map(|w| w.url.clone())
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
            workers.first().map(|w| w.url.clone())
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
            let url = format!("{}/flush_cache", worker.url);
            tasks.push(client.post(&url).send());
        }

        // Flush cache on all decode servers
        for worker in self.decode_workers.read().unwrap().iter() {
            let url = format!("{}/flush_cache", worker.url);
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
