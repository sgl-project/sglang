use crate::core::worker::{worker_adapter, Worker, WorkerFactory};
use crate::pd_router::PDRouter;
use crate::pd_types::PDSelectionPolicy;
use crate::tree::Tree;
// use crate::utils::api_path; // Currently unused
use ::metrics::{counter, gauge, histogram};
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use futures_util::{StreamExt, TryStreamExt};
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;
use std::time::Instant;
use tokio;
use tracing::{debug, error, info, warn};

pub fn copy_request_headers(req: &HttpRequest) -> Vec<(String, String)> {
    req.headers()
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|v| (name.to_string(), v.to_string()))
        })
        .collect()
}

#[derive(Debug)]
pub enum Router {
    RoundRobin {
        workers: Arc<RwLock<Vec<Arc<dyn Worker>>>>,
        current_index: AtomicUsize,
        timeout_secs: u64,
        interval_secs: u64,
    },
    Random {
        workers: Arc<RwLock<Vec<Arc<dyn Worker>>>>,
        timeout_secs: u64,
        interval_secs: u64,
    },
    PrefillDecode {
        pd_router: Arc<PDRouter>,
    },
    CacheAware {
        /*
            Cache-Aware Load Balancing Router

            This router combines two strategies to optimize both cache utilization and request distribution:

            1. Cache-Aware Routing (Approximate Tree)
            2. Load Balancing (Shortest Queue with Balance Thresholds)

            The router dynamically switches between these strategies based on load conditions:
            - Uses load balancing when the system is imbalanced
            - Uses cache-aware routing when the system is balanced

            A system is considered imbalanced if both conditions are met:
            1. (max - min) > abs_threshold
            2. max > rel_threshold * min

            Strategy Details:

            1. Cache-Aware Routing (Approximate Tree)
            -------------------------------------------
            This strategy maintains an approximate radix tree for each worker based on request history,
            eliminating the need for direct cache state queries. The tree stores raw text characters
            instead of token IDs to avoid tokenization overhead.

            Process:
            a. For each request, find the worker with the highest prefix match
            b. If match rate > cache_threshold:
            Route to the worker with highest match (likely has relevant data cached)
            c. If match rate ≤ cache_threshold:
            Route to the worker with smallest tree size (most available cache capacity)
            d. Background maintenance:
            Periodically evict least recently used leaf nodes to prevent memory overflow

            2. Load Balancing (Shortest Queue)
            -------------------------------------------
            This strategy tracks pending request counts per worker and routes new requests
            to the least busy worker when the system is detected to be imbalanced.

            Configuration Parameters:
            ------------------------
            1. cache_threshold: (float, 0.0 to 1.0)
            Minimum prefix match ratio to use highest-match routing.
            Below this threshold, routes to worker with most available cache space.

            2. balance_abs_threshold: (integer)
            Absolute difference threshold for load imbalance detection.
            System is potentially imbalanced if (max_load - min_load) > abs_threshold

            3. balance_rel_threshold: (float)
            Relative ratio threshold for load imbalance detection.
            System is potentially imbalanced if max_load > min_load * rel_threshold
            Used in conjunction with abs_threshold to determine final imbalance state.

            4. eviction_interval_secs: (integer)
            Interval between LRU eviction cycles for the approximate trees.

            5. max_tree_size: (integer)
            Maximum nodes per tree. When exceeded, LRU leaf nodes are evicted
            during the next eviction cycle.
        */
        workers: Arc<RwLock<Vec<Arc<dyn Worker>>>>,
        tree: Arc<Mutex<Tree>>,
        running_queue: Arc<Mutex<HashMap<String, usize>>>,
        processed_queue: Arc<Mutex<HashMap<String, usize>>>,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        timeout_secs: u64,
        interval_secs: u64,
        _eviction_thread: Option<thread::JoinHandle<()>>,
    },
}

#[derive(Debug, Clone)]
pub enum PolicyConfig {
    RandomConfig {
        timeout_secs: u64,
        interval_secs: u64,
    },
    RoundRobinConfig {
        timeout_secs: u64,
        interval_secs: u64,
    },
    CacheAwareConfig {
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
        timeout_secs: u64,
        interval_secs: u64,
    },
    PrefillDecodeConfig {
        selection_policy: PDSelectionPolicy,
        prefill_urls: Vec<(String, Option<u16>)>, // (url, bootstrap_port)
        decode_urls: Vec<String>,
        timeout_secs: u64,
        interval_secs: u64,
    },
}

impl Router {
    pub fn new(worker_urls: Vec<String>, policy_config: PolicyConfig) -> Result<Self, String> {
        // Update active workers gauge
        let workers = worker_adapter::from_regular_vec(worker_urls.clone());
        gauge!("sgl_router_active_workers").set(workers.len() as f64);

        // Get timeout and interval from policy config
        let (timeout_secs, interval_secs) = match &policy_config {
            PolicyConfig::RandomConfig {
                timeout_secs,
                interval_secs,
            } => (*timeout_secs, *interval_secs),
            PolicyConfig::RoundRobinConfig {
                timeout_secs,
                interval_secs,
            } => (*timeout_secs, *interval_secs),
            PolicyConfig::CacheAwareConfig {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            PolicyConfig::PrefillDecodeConfig {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
        };

        crate::core::worker::utils::wait_for_healthy_workers_sync(
            &workers,
            timeout_secs,
            interval_secs,
        )?;

        // Create router based on policy...
        Ok(match policy_config {
            PolicyConfig::RandomConfig {
                timeout_secs,
                interval_secs,
            } => Router::Random {
                workers: Arc::new(RwLock::new(workers)),
                timeout_secs,
                interval_secs,
            },
            PolicyConfig::RoundRobinConfig {
                timeout_secs,
                interval_secs,
            } => Router::RoundRobin {
                workers: Arc::new(RwLock::new(workers)),
                current_index: std::sync::atomic::AtomicUsize::new(0),
                timeout_secs,
                interval_secs,
            },
            PolicyConfig::CacheAwareConfig {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
                timeout_secs,
                interval_secs,
            } => {
                let mut running_queue = HashMap::new();
                for worker in &workers {
                    running_queue.insert(worker.url().to_string(), 0);
                }

                let mut processed_queue = HashMap::new();
                for worker in &workers {
                    processed_queue.insert(worker.url().to_string(), 0);
                }

                let tree = Arc::new(Mutex::new(Tree::new()));
                let running_queue = Arc::new(Mutex::new(running_queue));
                let processed_queue = Arc::new(Mutex::new(processed_queue));

                // Create background eviction thread
                let tree_clone = Arc::clone(&tree);
                let processed_queue_clone = Arc::clone(&processed_queue);
                let running_queue_clone = Arc::clone(&running_queue);
                let eviction_thread = thread::spawn(move || {
                    loop {
                        // Sleep for the specified interval
                        thread::sleep(Duration::from_secs(eviction_interval_secs));

                        let locked_tree_clone = tree_clone.lock().unwrap();
                        // Run eviction
                        locked_tree_clone.evict_tenant_by_size(max_tree_size);

                        // Print the process queue
                        let locked_processed_queue = processed_queue_clone.lock().unwrap();
                        info!("Processed Queue: {:?}", locked_processed_queue);

                        // Print the running queue
                        let locked_running_queue = running_queue_clone.lock().unwrap();
                        info!("Running Queue: {:?}", locked_running_queue);
                    }
                });

                for worker in &workers {
                    tree.lock().unwrap().insert("", worker.url());
                }

                Router::CacheAware {
                    workers: Arc::new(RwLock::new(workers)),
                    tree,
                    running_queue,
                    processed_queue,
                    cache_threshold,
                    balance_abs_threshold,
                    balance_rel_threshold,
                    timeout_secs,
                    interval_secs,
                    _eviction_thread: Some(eviction_thread),
                }
            }
            PolicyConfig::PrefillDecodeConfig {
                selection_policy,
                prefill_urls,
                decode_urls,
                timeout_secs,
                interval_secs,
            } => {
                // Create PDRouter instance
                let pd_router = PDRouter::new(
                    prefill_urls,
                    decode_urls,
                    selection_policy,
                    timeout_secs,
                    interval_secs,
                )?;

                Router::PrefillDecode {
                    pd_router: Arc::new(pd_router),
                }
            }
        })
    }

    /// Get a reference to the workers shared across threads
    pub fn get_workers(&self) -> Arc<RwLock<Vec<Arc<dyn Worker>>>> {
        match self {
            Router::RoundRobin { workers, .. }
            | Router::Random { workers, .. }
            | Router::CacheAware { workers, .. } => Arc::clone(workers),
            Router::PrefillDecode { .. } => {
                // For PD mode, return empty list since we manage workers differently
                Arc::new(RwLock::new(Vec::new()))
            }
        }
    }

    pub fn wait_for_healthy_workers(
        worker_urls: &[String],
        timeout_secs: u64,
        interval_secs: u64,
    ) -> Result<(), String> {
        let start_time = std::time::Instant::now();
        let sync_client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        loop {
            if start_time.elapsed() > Duration::from_secs(timeout_secs) {
                error!(
                    "Timeout {}s waiting for workers {:?} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    timeout_secs, worker_urls
                );
                return Err(format!(
                    "Timeout {}s waiting for workers {:?} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    timeout_secs, worker_urls
                ));
            }

            let mut all_healthy = true;
            let mut unhealthy_workers = Vec::new();

            for url in worker_urls {
                match sync_client.get(&format!("{}/health", url)).send() {
                    Ok(res) => {
                        if !res.status().is_success() {
                            let msg = format!(
                                "Worker heatlh check is pending with status {}",
                                res.status()
                            );
                            info!("{}", msg);
                            all_healthy = false;
                            unhealthy_workers.push((url, msg));
                        }
                    }
                    Err(_) => {
                        let msg = format!("Worker is not ready yet");
                        info!("{}", msg);
                        all_healthy = false;
                        unhealthy_workers.push((url, msg));
                    }
                }
            }

            if all_healthy {
                info!("All workers are healthy");
                return Ok(());
            } else {
                info!("Initializing workers:");
                for (url, reason) in &unhealthy_workers {
                    info!("  {} - {}", url, reason);
                }
                thread::sleep(Duration::from_secs(interval_secs));
            }
        }
    }

    fn select_first_worker(&self) -> Result<Arc<dyn Worker>, String> {
        match self {
            Router::RoundRobin { workers, .. }
            | Router::Random { workers, .. }
            | Router::CacheAware { workers, .. } => {
                let workers_guard = workers.read().unwrap();
                if workers_guard.is_empty() {
                    Err("No workers are available".to_string())
                } else {
                    Ok(workers_guard[0].clone())
                }
            }
            Router::PrefillDecode { .. } => {
                Err("PrefillDecode mode doesn't use select_first_worker".to_string())
            }
        }
    }

    fn select_first_worker_url(&self) -> Result<String, String> {
        match self {
            Router::RoundRobin { workers, .. }
            | Router::Random { workers, .. }
            | Router::CacheAware { workers, .. } => {
                if workers.read().unwrap().is_empty() {
                    Err("No workers are available".to_string())
                } else {
                    Ok(workers.read().unwrap()[0].url().to_string())
                }
            }
            Router::PrefillDecode { .. } => {
                // For PD mode, we don't need this method as routing is handled by PDRouter
                Err("PrefillDecode mode doesn't use select_first_worker".to_string())
            }
        }
    }

    pub async fn send_request(
        &self,
        client: &reqwest::Client,
        worker: &Arc<dyn Worker>,
        route: &str,
        req: &HttpRequest,
    ) -> HttpResponse {
        let start = Instant::now();
        let mut request_builder = client.get(format!("{}{}", worker.url(), route));

        // Copy all headers from original request except for health endpoint because it does not need authorization
        if route != worker.worker_type().get_endpoints().health {
            for (name, value) in copy_request_headers(req) {
                // Skip Content-Type and Content-Length as .json() sets them
                if name.to_lowercase() != "content-type" && name.to_lowercase() != "content-length"
                {
                    request_builder = request_builder.header(name, value);
                }
            }
        }

        let response = match request_builder.send().await {
            Ok(res) => {
                let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

                match res.bytes().await {
                    Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                    Err(e) => HttpResponse::InternalServerError()
                        .body(format!("Failed to read response body: {}", e)),
                }
            }
            Err(e) => HttpResponse::InternalServerError().body(format!(
                "Failed to send request to worker {}: {}",
                worker, e
            )),
        };

        // Record request metrics
        if route != worker.worker_type().get_endpoints().health {
            let duration = start.elapsed();
            counter!("sgl_router_requests_total", "route" => route.to_string()).increment(1);
            histogram!("sgl_router_request_duration_seconds", "route" => route.to_string())
                .record(duration.as_secs_f64());

            if !response.status().is_success() {
                counter!("sgl_router_request_errors_total", "route" => route.to_string())
                    .increment(1);
            }
        }
        response
    }

    pub async fn route_to_first(
        &self,
        client: &reqwest::Client,
        route: &str,
        req: &HttpRequest,
    ) -> HttpResponse {
        const MAX_REQUEST_RETRIES: u32 = 3;
        const MAX_TOTAL_RETRIES: u32 = 6;
        let mut total_retries = 0;

        while total_retries < MAX_TOTAL_RETRIES {
            match self.select_first_worker() {
                Ok(worker) => {
                    let mut request_retries = 0;
                    // Try the same worker multiple times
                    while request_retries < MAX_REQUEST_RETRIES {
                        if total_retries >= 1 {
                            info!("Retrying request after {} failed attempts", total_retries);
                        }

                        let response = self.send_request(client, &worker, route, req).await;

                        if response.status().is_success() || worker.check_health().await.is_ok() {
                            // if the response is successful then we can return the response immediately.
                            // if the worker is healthy, it means the request is bad, so return the error response
                            return response;
                        }

                        warn!(
                            "Request to {} failed (attempt {}/{})",
                            &worker,
                            request_retries + 1,
                            MAX_REQUEST_RETRIES
                        );

                        request_retries += 1;
                        total_retries += 1;

                        if request_retries == MAX_REQUEST_RETRIES {
                            warn!("Removing failed worker: {}", &worker);
                            if let Err(e) = self.unified_remove_worker(worker) {
                                warn!("Failed to remove failed worker: {}", e.to_string());
                            }
                            break;
                        }
                    }
                }
                Err(e) => return HttpResponse::InternalServerError().body(e),
            }
        }

        HttpResponse::InternalServerError().body("All retry attempts failed")
    }

    pub async fn route_to_all(
        &self,
        client: &reqwest::Client,
        route: &str,
        req: &HttpRequest,
    ) -> HttpResponse {
        // Get all worker URLs based on router type
        let workers = match self {
            Router::PrefillDecode { .. } => {
                // For PD mode, route_to_all is not supported directly
                // It should be handled by PDRouter if needed
                return HttpResponse::NotImplemented()
                    .body("route_to_all not implemented for PrefillDecode mode");
            }
            _ => self.get_workers().read().unwrap().clone(),
        };

        // Send requests to all workers concurrently
        let mut tasks = Vec::new();
        for worker_url in workers.iter().map(|w| w.url()).collect::<Vec<_>>() {
            let mut request_builder = client.post(format!("{}{}", worker_url, route));

            // Copy headers from original request
            for (name, value) in copy_request_headers(req) {
                request_builder = request_builder.header(name, value);
            }

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
            HttpResponse::Ok().body("Operation completed on all servers")
        } else {
            HttpResponse::InternalServerError().body("Operation failed on one or more servers")
        }
    }

    pub async fn get_all_loads(
        &self,
        client: &reqwest::Client,
        _req: &HttpRequest,
    ) -> HttpResponse {
        // For PD mode, delegate to PDRouter
        match self {
            Router::PrefillDecode { pd_router } => {
                return pd_router.get_loads(client).await;
            }
            _ => {
                // For non-PD routers, handle normally
            }
        }

        let prefill_workers: Vec<Arc<dyn Worker>> = Vec::new();
        let decode_workers = self.get_workers().read().unwrap().clone();

        let (prefill_loads, decode_loads) =
            get_loads_helper(client, prefill_workers, decode_workers).await;

        HttpResponse::Ok().json(serde_json::json!({
            "prefill": prefill_loads,
            "decode": decode_loads
        }))
    }

    // New method to route typed requests directly
    pub async fn route_typed_request<
        T: crate::openai_api_types::GenerationRequest + serde::Serialize + Clone,
    >(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        typed_req: &T,
        route: &str,
    ) -> HttpResponse {
        match self {
            Router::PrefillDecode { .. } => HttpResponse::InternalServerError()
                .body("PD routing should use specialized typed handlers"),
            _ => {
                // Handle retries like the original implementation
                let start = Instant::now();
                const MAX_REQUEST_RETRIES: u32 = 3;
                const MAX_TOTAL_RETRIES: u32 = 6;
                let mut total_retries = 0;

                while total_retries < MAX_TOTAL_RETRIES {
                    // Extract routing text directly from typed request
                    let text = typed_req.extract_text_for_routing();
                    let is_stream = typed_req.is_stream();
                    // Select worker based on text
                    let worker_url = self.select_generate_worker_from_text(&text);
                    let worker = {
                        let workers_guard = self.get_workers();
                        let workers = workers_guard.read().unwrap();
                        match workers.iter().find(|w| w.url() == worker_url) {
                            Some(w) => w.clone(),
                            None => {
                                warn!("Worker for url {} not found, retrying", worker_url);
                                total_retries += 1;
                                continue;
                            }
                        }
                    };
                    let mut request_retries = 0;

                    // Try the same worker multiple times
                    while request_retries < MAX_REQUEST_RETRIES {
                        if total_retries >= 1 {
                            info!("Retrying request after {} failed attempts", total_retries);
                            counter!("sgl_router_retries_total", "route" => route.to_string())
                                .increment(1);
                        }

                        // Send typed request directly
                        let response = self
                            .send_typed_request(
                                client,
                                req,
                                typed_req,
                                route,
                                &worker_url,
                                is_stream,
                            )
                            .await;

                        if response.status().is_success() {
                            let duration = start.elapsed();
                            histogram!("sgl_router_generate_duration_seconds", "route" => route.to_string())
                                .record(duration.as_secs_f64());
                            return response;
                        } else if worker.check_health().await.is_ok() {
                            // if the worker is healthy, it means the request is bad, so return the error response
                            counter!("sgl_router_request_errors_total", "route" => route.to_string())
                                .increment(1);
                            return response;
                        }

                        warn!(
                            "Generate request to {} failed (attempt {}/{})",
                            worker_url,
                            request_retries + 1,
                            MAX_REQUEST_RETRIES
                        );

                        request_retries += 1;
                        total_retries += 1;

                        if request_retries == MAX_REQUEST_RETRIES {
                            warn!("Removing failed worker: {}", worker_url);
                            if let Err(e) = self.unified_remove_worker(worker) {
                                warn!("Failed to remove failed worker: {}", e.to_string());
                            }
                            break;
                        }
                    }
                }

                counter!("sgl_router_request_errors_total", "route" => route.to_string())
                    .increment(1);
                HttpResponse::InternalServerError().body("All retry attempts failed")
            }
        }
    }

    // Helper method to select worker from text
    fn select_generate_worker_from_text(&self, text: &str) -> String {
        match self {
            Router::RoundRobin {
                workers,
                current_index,
                ..
            } => {
                let idx = current_index
                    .fetch_update(
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                        |x| Some((x + 1) % workers.read().unwrap().len()),
                    )
                    .unwrap();
                workers.read().unwrap()[idx].url().to_string()
            }

            Router::Random { workers, .. } => workers.read().unwrap()
                [rand::random::<usize>() % workers.read().unwrap().len()]
            .url()
            .to_string(),

            Router::CacheAware {
                workers,
                tree,
                running_queue,
                processed_queue,
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                ..
            } => {
                let tree = tree.lock().unwrap();
                let mut running_queue = running_queue.lock().unwrap();

                // Get current load statistics
                let max_load = *running_queue.values().max().unwrap_or(&0);
                let min_load = *running_queue.values().min().unwrap_or(&0);

                // Load is considered imbalanced if:
                // 1. (max - min) > abs_threshold AND
                // 2. max > rel_threshold * min
                let is_imbalanced = max_load.saturating_sub(min_load) > *balance_abs_threshold
                    && (max_load as f32) > (min_load as f32 * balance_rel_threshold);

                let selected_url = if is_imbalanced {
                    // Log load balancing trigger and current queue state
                    info!(
                        "Load balancing triggered due to workload imbalance:\n\
                        Max load: {}, Min load: {}\n\
                        Current running queue: {:?}",
                        max_load, min_load, running_queue
                    );

                    counter!("sgl_router_load_balancing_events_total").increment(1);
                    gauge!("sgl_router_max_load").set(max_load as f64);
                    gauge!("sgl_router_min_load").set(min_load as f64);

                    // Use shortest queue routing when load is imbalanced
                    running_queue
                        .iter()
                        .min_by_key(|(_url, &count)| count)
                        .map(|(url, _)| url.clone())
                        .unwrap_or_else(|| workers.read().unwrap()[0].url().to_string())
                } else {
                    // Use cache-aware routing when load is balanced
                    let (matched_text, matched_worker) = tree.prefix_match(&text);
                    let matched_rate =
                        matched_text.chars().count() as f32 / text.chars().count() as f32;

                    if matched_rate > *cache_threshold {
                        counter!("sgl_router_cache_hits_total").increment(1);
                        matched_worker
                    } else {
                        counter!("sgl_router_cache_misses_total").increment(1);
                        tree.get_smallest_tenant()
                    }
                };

                // Update queues and tree
                *running_queue.get_mut(&selected_url).unwrap() += 1;

                *processed_queue
                    .lock()
                    .unwrap()
                    .get_mut(&selected_url)
                    .unwrap() += 1;

                gauge!("sgl_router_running_requests", "worker" => selected_url.to_string())
                    .set(*running_queue.get(&selected_url).unwrap() as f64);
                counter!("sgl_router_processed_requests_total", "worker" => selected_url.to_string()).increment(1);

                tree.insert(&text, &selected_url);

                selected_url
            }
            Router::PrefillDecode { .. } => {
                // For PD mode, we don't use this method
                return "PD_MODE_ERROR".to_string();
            }
        }
    }

    // Send typed request directly without conversion
    async fn send_typed_request<T: serde::Serialize>(
        &self,
        client: &reqwest::Client,
        req: &HttpRequest,
        typed_req: &T,
        route: &str,
        worker_url: &str,
        is_stream: bool,
    ) -> HttpResponse {
        let start = Instant::now();

        // Debug: Log what we're sending
        if let Ok(json_str) = serde_json::to_string_pretty(typed_req) {
            debug!("Sending request to {}: {}", route, json_str);
        }

        let mut request_builder = client
            .post(format!("{}{}", worker_url, route))
            .json(typed_req); // Use json() directly with typed request

        // Copy all headers from original request
        for (name, value) in copy_request_headers(req) {
            // Skip Content-Type and Content-Length as .json() sets them
            if name.to_lowercase() != "content-type" && name.to_lowercase() != "content-length" {
                request_builder = request_builder.header(&name, &value);
            }
        }

        let res = match request_builder.send().await {
            Ok(res) => res,
            Err(e) => {
                error!("Failed to send request to {}: {}", worker_url, e);
                return HttpResponse::InternalServerError().body(format!("Request failed: {}", e));
            }
        };

        let status = actix_web::http::StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // For non-streaming requests, get response first
            let response = match res.bytes().await {
                Ok(body) => HttpResponse::build(status).body(body.to_vec()),
                Err(e) => {
                    let error_msg = format!("Failed to get response body: {}", e);
                    HttpResponse::InternalServerError().body(error_msg)
                }
            };

            // Then decrement running queue counter if using CacheAware
            if let Router::CacheAware { running_queue, .. } = self {
                if let Ok(mut queue) = running_queue.lock() {
                    if let Some(count) = queue.get_mut(worker_url) {
                        *count = count.saturating_sub(1);
                    }
                }
            }

            // Record metrics
            let duration = start.elapsed();
            histogram!("sgl_router_generate_duration_seconds", "route" => route.to_string())
                .record(duration.as_secs_f64());
            counter!("sgl_router_requests_total", "route" => route.to_string()).increment(1);

            response
        } else if let Router::CacheAware { running_queue, .. } = self {
            let running_queue = Arc::clone(running_queue);
            let worker_url = worker_url.to_string();

            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(
                    res.bytes_stream()
                        .map_err(|_| {
                            actix_web::error::ErrorInternalServerError("Failed to read stream")
                        })
                        .inspect(move |bytes| {
                            let bytes = bytes.as_ref().unwrap();
                            if bytes
                                .as_ref()
                                .windows(12)
                                .any(|window| window == b"data: [DONE]")
                            {
                                let mut locked_queue = running_queue.lock().unwrap();
                                let count = locked_queue.get_mut(&worker_url).unwrap();
                                *count = count.saturating_sub(1);
                                debug!("Streaming is done!!")
                            }
                        }),
                )
        } else {
            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(res.bytes_stream().map_err(|_| {
                    actix_web::error::ErrorInternalServerError("Failed to read stream")
                }))
        }
    }

    pub async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        let worker = WorkerFactory::create_regular(worker_url.to_string());
        let (timeout_secs, interval_secs) = match self {
            Router::Random {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            Router::RoundRobin {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            Router::CacheAware {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            Router::PrefillDecode { .. } => {
                // For PD mode, we don't support adding workers via this method
                return Err("Adding workers to PrefillDecode router not supported via add_worker. Use dedicated PD management methods.".to_string());
            }
        };

        // Check for duplicates BEFORE health check to avoid unnecessary work
        match self {
            Router::RoundRobin { workers, .. }
            | Router::Random { workers, .. }
            | Router::CacheAware { workers, .. } => {
                let workers_vec = workers.read().unwrap();
                if workers_vec.iter().any(|w| w.url() == worker.url()) {
                    return Err(format!("Worker {} already exists", worker_url));
                }
            }
            Router::PrefillDecode { .. } => {
                return Err("Adding workers to PrefillDecode router not supported via add_worker. Use dedicated PD management methods.".to_string());
            }
        }

        // Only perform health check if worker is not a duplicate
        let health_check_result = crate::core::worker::utils::wait_for_healthy_workers(
            &[worker.clone()],
            interval_secs,
            timeout_secs,
        )
        .await;
        if health_check_result.is_err() {
            return Err(format!(
                "Failed to health check worker: {}",
                health_check_result.err().unwrap()
            ));
        }

        match self {
            Router::RoundRobin { workers, .. }
            | Router::Random { workers, .. }
            | Router::CacheAware { workers, .. } => {
                info!("Worker {} health check passed", worker);
                let mut workers_vec = workers.write().unwrap();
                // Double-check for duplicates after acquiring write lock (race condition protection)
                if workers_vec.iter().any(|w| w.url() == worker.url()) {
                    return Err(format!("Worker {} already exists", worker_url));
                }
                info!("Added worker: {}", worker_url);
                // Use the same worker instance that passed the health check
                workers_vec.push(worker.clone());
                gauge!("sgl_router_active_workers").set(workers_vec.len() as f64);
            }
            Router::PrefillDecode { .. } => {
                return Err("Adding workers to PrefillDecode router not supported via add_worker. Use dedicated PD management methods.".to_string());
            }
        }

        // If cache aware, initialize the queues for the new worker
        if let Router::CacheAware {
            running_queue,
            processed_queue,
            tree,
            ..
        } = self
        {
            // Add worker to running queue with initial count of 0
            running_queue
                .lock()
                .unwrap()
                .insert(worker.url().to_string(), 0);

            // Add worker to processed queue with initial count of 0
            processed_queue
                .lock()
                .unwrap()
                .insert(worker.url().to_string(), 0);

            // Add worker to tree
            tree.lock().unwrap().insert("", worker.url());
        }

        return Ok(format!("Successfully added worker: {}", worker));
    }

    pub async fn unified_add_worker(&self, worker: Arc<dyn Worker>) -> Result<String, String> {
        // Check for duplicates BEFORE health check to avoid unnecessary work
        let (timeout_secs, interval_secs) = match self {
            Router::PrefillDecode {pd_router} => {
                // For PD mode, delegate to PD router
                return pd_router.add_worker(worker).await.map_err(|e| e.to_string());
            }
            Router::Random {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            Router::RoundRobin {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
            Router::CacheAware {
                timeout_secs,
                interval_secs,
                ..
            } => (*timeout_secs, *interval_secs),
        };

        // Check for duplicates BEFORE health check to avoid unnecessary work
        match self {
            Router::RoundRobin { workers, .. }
            | Router::Random { workers, .. }
            | Router::CacheAware { workers, .. } => {
                let workers_vec = workers.read().unwrap();
                if workers_vec.iter().any(|w| w.url() == worker.url()) {
                    return Err(format!("Worker {} already exists", worker.url()));
                }
            }
            Router::PrefillDecode { .. } => unreachable!(),
        }

        // Only perform health check if worker is not a duplicate
        let health_check_result = crate::core::worker::utils::wait_for_healthy_workers(
            &[worker.clone()],
            interval_secs,
            timeout_secs,
        )
        .await;
        if health_check_result.is_err() {
            return Err(format!(
                "Failed to health check worker: {}",
                health_check_result.err().unwrap()
            ));
        }

        match self {
            Router::RoundRobin { workers, .. }
            | Router::Random { workers, .. }
            | Router::CacheAware { workers, .. } => {
                info!("Worker {} health check passed", worker);
                let mut workers_vec = workers.write().unwrap();
                // Double-check for duplicates after acquiring write lock (race condition protection)
                if workers_vec.iter().any(|w| w.url() == worker.url()) {
                    return Err(format!("Worker {} already exists", worker));
                }
                info!("Added worker: {}", worker);
                // Use the same worker instance that passed the health check
                workers_vec.push(worker.clone());
                gauge!("sgl_router_active_workers").set(workers_vec.len() as f64);
            }
            Router::PrefillDecode { .. } => unreachable!(),
        }

        // If cache aware, initialize the queues for the new worker
        if let Router::CacheAware {
            running_queue,
            processed_queue,
            tree,
            ..
        } = self
        {
            // Add worker to running queue with initial count of 0
            running_queue
                .lock()
                .unwrap()
                .insert(worker.url().to_string(), 0);

            // Add worker to processed queue with initial count of 0
            processed_queue
                .lock()
                .unwrap()
                .insert(worker.url().to_string(), 0);

            // Add worker to tree
            tree.lock().unwrap().insert("", worker.url());
        }

        return Ok(format!("Successfully added worker: {}", worker));
    }
 
    pub fn unified_remove_worker(&self, worker: Arc<dyn Worker>) -> Result<String, String> {
        if let Router::PrefillDecode {pd_router} = self {
            return pd_router.unified_remove_worker(worker).map_err(|e| e.to_string());
        }
        self.remove_worker_by_url(worker.url())
    }
    
    pub fn remove_worker_by_url(&self, worker_url: &str) -> Result<String, String> {
        match self {
            Router::RoundRobin { workers, .. }
            | Router::Random { workers, .. }
            | Router::CacheAware { workers, .. } => {
                let mut workers = workers.write().unwrap();
                // This is safe because we are holding the write lock
                if let Some(index) = workers.iter().position(|worker| worker.url() == worker_url) {
                    let worker = workers.remove(index);
                    info!("Removed worker: {}", &worker);
                    gauge!("sgl_router_active_workers").set(workers.len() as f64);
                } else {
                    warn!("Worker with url {} not found, skipping removal", worker_url);
                    return Err(format!("Worker with url {} not found, skipping removal", worker_url));
                }
            }
            Router::PrefillDecode { .. } => {
                warn!("Removing workers from PrefillDecode router not supported via remove_worker. Use dedicated PD management methods.");
                return Err("Removing workers from PrefillDecode router not supported via remove_worker.".to_string());
            }
        }

        // if cache aware, remove the worker from the tree
        if let Router::CacheAware {
            tree,
            running_queue,
            processed_queue,
            ..
        } = self
        {
            tree.lock().unwrap().remove_tenant(&worker_url);
            running_queue
                .lock()
                .unwrap()
                .remove(&worker_url.to_string());
            processed_queue
                .lock()
                .unwrap()
                .remove(&worker_url.to_string());
            info!(
                "Removed worker from tree and cleaned up queues: {}",
                worker_url
            );
        }
        return Ok(format!("Successfully removed worker: {}", worker_url));
    }

    /// Add a worker with PD mode support
    pub async fn add_pd_worker(
        &self,
        worker_url: &str,
        pod_type: crate::service_discovery::PodType,
        bootstrap_port: Option<u16>,
    ) -> Result<String, String> {
        match self {
            Router::PrefillDecode { pd_router } => match pod_type {
                crate::service_discovery::PodType::Prefill => pd_router
                    .add_prefill_server(worker_url.to_string(), bootstrap_port)
                    .await
                    .map_err(|e| e.to_string()),
                crate::service_discovery::PodType::Decode => pd_router
                    .add_decode_server(worker_url.to_string())
                    .await
                    .map_err(|e| e.to_string()),
                crate::service_discovery::PodType::Regular => {
                    Err("Regular pod type not supported in PD mode".to_string())
                }
            },
            _ => Err("add_pd_worker only supported in PD mode".to_string()),
        }
    }

    /// Remove a worker with PD mode support
    pub async fn remove_pd_worker(
        &self,
        worker_url: &str,
        pod_type: crate::service_discovery::PodType,
    ) -> Result<String, String> {
        match self {
            Router::PrefillDecode { pd_router } => match pod_type {
                crate::service_discovery::PodType::Prefill => pd_router
                    .remove_prefill_server(worker_url)
                    .await
                    .map_err(|e| e.to_string()),
                crate::service_discovery::PodType::Decode => pd_router
                    .remove_decode_server(worker_url)
                    .await
                    .map_err(|e| e.to_string()),
                crate::service_discovery::PodType::Regular => {
                    Err("Regular pod type not supported in PD mode".to_string())
                }
            },
            _ => Err("remove_pd_worker only supported in PD mode".to_string()),
        }
    }

    // PD-specific wrapper methods that delegate to PDRouter
    pub async fn route_pd_health_generate(
        &self,
        _client: &reqwest::Client,
        _req: &HttpRequest,
    ) -> HttpResponse {
        match self {
            Router::PrefillDecode { pd_router } => {
                pd_router.health_generate(&pd_router.http_client).await
            }
            _ => HttpResponse::InternalServerError().body("Not in PrefillDecode mode"),
        }
    }

    pub async fn route_pd_generate_typed(
        &self,
        _client: &reqwest::Client,
        req: &HttpRequest,
        typed_req: crate::pd_types::GenerateReqInput,
        route: &str,
    ) -> HttpResponse {
        match self {
            Router::PrefillDecode { pd_router } => {
                pd_router
                    .route_generate(&pd_router.http_client, req, typed_req, route)
                    .await
            }
            _ => HttpResponse::InternalServerError().body("Not in PrefillDecode mode"),
        }
    }

    pub async fn route_pd_chat_typed(
        &self,
        _client: &reqwest::Client,
        req: &HttpRequest,
        typed_req: crate::pd_types::ChatReqInput,
        route: &str,
    ) -> HttpResponse {
        match self {
            Router::PrefillDecode { pd_router } => {
                pd_router
                    .route_chat(&pd_router.http_client, req, typed_req, route)
                    .await
            }
            _ => HttpResponse::InternalServerError().body("Not in PrefillDecode mode"),
        }
    }

    pub async fn get_pd_server_info(
        &self,
        _client: &reqwest::Client,
        _req: &HttpRequest,
    ) -> HttpResponse {
        match self {
            Router::PrefillDecode { pd_router } => {
                pd_router.get_server_info(&pd_router.http_client).await
            }
            _ => HttpResponse::InternalServerError().body("Not in PrefillDecode mode"),
        }
    }

    pub async fn get_pd_models(
        &self,
        _client: &reqwest::Client,
        req: &HttpRequest,
    ) -> HttpResponse {
        match self {
            Router::PrefillDecode { pd_router } => {
                pd_router.get_models(&pd_router.http_client, req).await
            }
            _ => HttpResponse::InternalServerError().body("Not in PrefillDecode mode"),
        }
    }

    pub async fn route_pd_flush_cache(&self, _client: &reqwest::Client) -> HttpResponse {
        match self {
            Router::PrefillDecode { pd_router } => {
                pd_router.flush_cache(&pd_router.http_client).await
            }
            _ => HttpResponse::InternalServerError().body("Not in PrefillDecode mode"),
        }
    }

    pub async fn get_pd_model_info(
        &self,
        _client: &reqwest::Client,
        req: &HttpRequest,
    ) -> HttpResponse {
        match self {
            Router::PrefillDecode { pd_router } => {
                pd_router.get_model_info(&pd_router.http_client, req).await
            }
            _ => HttpResponse::InternalServerError().body("Not in PrefillDecode mode"),
        }
    }
}

pub(crate) async fn get_loads_helper(
    client: &reqwest::Client,
    prefill_workers: Vec<Arc<dyn Worker + 'static>>,
    decode_workers: Vec<Arc<dyn Worker + 'static>>,
) -> (Vec<serde_json::Value>, Vec<serde_json::Value>) {
    // Collect loads from all servers
    let prefill_loads =
        futures_util::future::join_all(prefill_workers.iter().map(|w| async move {
            let load = crate::core::worker::utils::get_worker_load(&client, w)
                .await
                .unwrap_or(-1);
            serde_json::json!({
                "engine": format!("(Prefill@{})", w.url()),
                "load": load as i64
            })
        }))
        .await;

    let decode_loads = futures_util::future::join_all(decode_workers.iter().map(|w| async move {
        let load = crate::core::worker::utils::get_worker_load(&client, w)
            .await
            .unwrap_or(-1);
        serde_json::json!({
            "engine": format!("(Decode@{})", w.url()),
            "load": load as i64
        })
    }))
    .await;
    (prefill_loads, decode_loads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service_discovery::PodType;
    use crate::test_utils::mock_servers::create_enhanced_mock_health_server;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    // ============================================================================
    // Test Helper Functions and Mock Servers
    // ============================================================================

    // Helper function to create workers with very short health check TTL for testing
    fn create_test_worker_with_short_ttl(url: String) -> Arc<dyn Worker> {
        WorkerFactory::create_regular_with_ttl(url, Duration::from_millis(1)) // 1ms TTL for testing
    }

    fn create_test_regular_router() -> Router {
        let workers = vec![
            WorkerFactory::create_regular("http://worker1:8080".to_string()),
            WorkerFactory::create_regular("http://worker2:8080".to_string()),
        ];
        Router::Random {
            workers: Arc::new(RwLock::new(workers)),
            timeout_secs: 5,
            interval_secs: 1,
        }
    }

    fn create_test_round_robin_router() -> Router {
        let workers = vec![
            WorkerFactory::create_regular("http://worker1:8080".to_string()),
            WorkerFactory::create_regular("http://worker2:8080".to_string()),
            WorkerFactory::create_regular("http://worker3:8080".to_string()),
        ];
        Router::RoundRobin {
            workers: Arc::new(RwLock::new(workers)),
            current_index: AtomicUsize::new(0),
            timeout_secs: 5,
            interval_secs: 1,
        }
    }

    fn create_test_cache_aware_router() -> Router {
        let workers = vec![
            WorkerFactory::create_regular("http://worker1:8080".to_string()),
            WorkerFactory::create_regular("http://worker2:8080".to_string()),
            WorkerFactory::create_regular("http://worker3:8080".to_string()),
        ];
        let mut running_queue = std::collections::HashMap::new();
        let mut processed_queue = std::collections::HashMap::new();
        
        for worker in &workers {
            running_queue.insert(worker.url().to_string(), 0);
            processed_queue.insert(worker.url().to_string(), 0);
        }

        let tree = Arc::new(Mutex::new(crate::tree::Tree::new()));
        for worker in &workers {
            tree.lock().unwrap().insert("", worker.url());
        }

        Router::CacheAware {
            workers: Arc::new(RwLock::new(workers)),
            tree,
            running_queue: Arc::new(Mutex::new(running_queue)),
            processed_queue: Arc::new(Mutex::new(processed_queue)),
            cache_threshold: 0.5,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            timeout_secs: 5,
            interval_secs: 1,
            _eviction_thread: None,
        }
    }



    // ============================================================================
    // Basic Router Tests
    // ============================================================================

    #[test]
    fn test_router_get_worker_urls_regular() {
        let router = create_test_regular_router();
        let workers = router.get_workers();
        let workers = workers.read().unwrap();
        let urls = workers
            .iter()
            .map(|w| w.url().to_string())
            .collect::<Vec<String>>();

        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));
    }

    #[test]
    fn test_select_first_worker_regular() {
        let router = create_test_regular_router();
        let result = router.select_first_worker_url();

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "http://worker1:8080");
    }

    #[test]
    fn test_select_first_worker_empty_workers() {
        let router = Router::Random {
            workers: Arc::new(RwLock::new(vec![])),
            timeout_secs: 5,
            interval_secs: 1,
        };
        let result = router.select_first_worker_url();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No workers are available"));
    }

    // ============================================================================
    // Round Robin Router Tests
    // ============================================================================

    #[test]
    fn test_round_robin_worker_selection() {
        let router = create_test_round_robin_router();
        
        // Test multiple selections to verify round-robin behavior
        let mut selected_workers = Vec::new();
        for _ in 0..6 {
            let url = router.select_generate_worker_from_text("test");
            selected_workers.push(url);
        }
        
        // Should cycle through workers in order
        assert_eq!(selected_workers[0], "http://worker1:8080");
        assert_eq!(selected_workers[1], "http://worker2:8080");
        assert_eq!(selected_workers[2], "http://worker3:8080");
        assert_eq!(selected_workers[3], "http://worker1:8080");
        assert_eq!(selected_workers[4], "http://worker2:8080");
        assert_eq!(selected_workers[5], "http://worker3:8080");
    }

    // ============================================================================
    // Random Router Tests
    // ============================================================================

    #[test]
    fn test_random_worker_selection() {
        let router = create_test_regular_router();
        
        // Test multiple selections - should select from available workers
        for _ in 0..10 {
            let url = router.select_generate_worker_from_text("test");
            assert!(url == "http://worker1:8080" || url == "http://worker2:8080");
        }
    }

    // ============================================================================
    // Cache Aware Router Tests
    // ============================================================================

    #[test]
    fn test_cache_aware_worker_selection_balanced() {
        let router = create_test_cache_aware_router();
        
        // When load is balanced, should use cache-aware routing
        let url = router.select_generate_worker_from_text("test input");
        
        // Should select one of the available workers
        let expected_urls = vec![
            "http://worker1:8080",
            "http://worker2:8080", 
            "http://worker3:8080"
        ];
        assert!(expected_urls.contains(&url.as_str()));
    }

    #[test]
    fn test_cache_aware_worker_selection_imbalanced() {
        let router = create_test_cache_aware_router();
        
        // Simulate imbalanced load
        if let Router::CacheAware { running_queue, .. } = &router {
            let mut queue = running_queue.lock().unwrap();
            *queue.get_mut("http://worker1:8080").unwrap() = 20; // High load
            *queue.get_mut("http://worker2:8080").unwrap() = 1;  // Low load
            *queue.get_mut("http://worker3:8080").unwrap() = 2;  // Medium load
        }
        
        // Should use load balancing (shortest queue)
        let url = router.select_generate_worker_from_text("test input");
        assert_eq!(url, "http://worker2:8080"); // Should select worker with lowest load
    }

    // ============================================================================
    // Policy Config Tests
    // ============================================================================

    #[test]
    fn test_policy_config_creation() {
        let random_config = PolicyConfig::RandomConfig {
            timeout_secs: 10,
            interval_secs: 2,
        };
        
        let round_robin_config = PolicyConfig::RoundRobinConfig {
            timeout_secs: 15,
            interval_secs: 3,
        };
        
        let cache_aware_config = PolicyConfig::CacheAwareConfig {
            cache_threshold: 0.8,
            balance_abs_threshold: 5,
            balance_rel_threshold: 2.0,
            eviction_interval_secs: 60,
            max_tree_size: 1000,
            timeout_secs: 20,
            interval_secs: 4,
        };
        
        // Test that configs can be created and cloned
        let _cloned_random = random_config.clone();
        let _cloned_round_robin = round_robin_config.clone();
        let _cloned_cache_aware = cache_aware_config.clone();
    }

    // ============================================================================
    // Worker Management Tests
    // ============================================================================

    #[test]
    fn test_remove_worker() {
        let router = create_test_regular_router();
        
        // Verify initial workers
        {
            let workers = router.get_workers();
            let workers = workers.read().unwrap();
            assert_eq!(workers.len(), 2);
        }
        
        // Remove a worker
        router.remove_worker_by_url("http://worker1:8080");
        
        // Verify worker was removed
        {
            let workers = router.get_workers();
            let workers = workers.read().unwrap();
            assert_eq!(workers.len(), 1);
            assert_eq!(workers[0].url(), "http://worker2:8080");
        }
    }

    #[test]
    fn test_remove_nonexistent_worker() {
        let router = create_test_regular_router();
        
        // Remove a worker that doesn't exist - should not panic
        router.remove_worker_by_url("http://nonexistent:8080");
        
        // Verify workers are unchanged
        {
            let workers = router.get_workers();
            let workers = workers.read().unwrap();
            assert_eq!(workers.len(), 2);
        }
    }

    #[test]
    fn test_cache_aware_worker_removal() {
        let router = create_test_cache_aware_router();
        
        // Remove a worker from cache-aware router
        router.remove_worker_by_url("http://worker1:8080");
        
        // Verify worker was removed from all data structures
        {
            let workers = router.get_workers();
            let workers = workers.read().unwrap();
            assert_eq!(workers.len(), 2);
        }
        
        if let Router::CacheAware { running_queue, processed_queue, .. } = &router {
            let running_queue = running_queue.lock().unwrap();
            let processed_queue = processed_queue.lock().unwrap();
            
            assert!(!running_queue.contains_key("http://worker1:8080"));
            assert!(!processed_queue.contains_key("http://worker1:8080"));
        }
    }

    // ============================================================================
    // PD Router Integration Tests
    // ============================================================================

    #[tokio::test]
    async fn test_add_pd_worker_with_regular_router() {
        let router = create_test_regular_router();

        let result = router
            .add_pd_worker("http://new-worker:8080", PodType::Prefill, Some(8081))
            .await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("add_pd_worker only supported in PD mode"));
    }

    #[tokio::test]
    async fn test_remove_pd_worker_with_regular_router() {
        let router = create_test_regular_router();

        let result = router
            .remove_pd_worker("http://worker:8080", PodType::Decode)
            .await;

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("remove_pd_worker only supported in PD mode"));
    }

    #[tokio::test]
    async fn test_pd_endpoints_with_regular_router() {
        let router = create_test_regular_router();
        let client = reqwest::Client::new();
        
        // Create a mock HTTP request
        let req = actix_web::test::TestRequest::get().to_http_request();
        
        // Test PD-specific endpoints return error for regular router
        let health_response = router.route_pd_health_generate(&client, &req).await;
        assert_eq!(health_response.status(), actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
        
        let models_response = router.get_pd_models(&client, &req).await;
        assert_eq!(models_response.status(), actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
        
        let server_info_response = router.get_pd_server_info(&client, &req).await;
        assert_eq!(server_info_response.status(), actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
        
        let model_info_response = router.get_pd_model_info(&client, &req).await;
        assert_eq!(model_info_response.status(), actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
        
        let flush_response = router.route_pd_flush_cache(&client).await;
        assert_eq!(flush_response.status(), actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
    }

    // ============================================================================
    // Request Header Tests
    // ============================================================================

    #[test]
    fn test_copy_request_headers() {
        let req = actix_web::test::TestRequest::get()
            .insert_header(("Authorization", "Bearer token"))
            .insert_header(("Content-Type", "application/json"))
            .insert_header(("Custom-Header", "custom-value"))
            .to_http_request();
        
        let headers = copy_request_headers(&req);
        
        assert!(headers.contains(&("authorization".to_string(), "Bearer token".to_string())));
        assert!(headers.contains(&("content-type".to_string(), "application/json".to_string())));
        assert!(headers.contains(&("custom-header".to_string(), "custom-value".to_string())));
    }

    #[test]
    fn test_copy_request_headers_empty() {
        let req = actix_web::test::TestRequest::get().to_http_request();
        let headers = copy_request_headers(&req);
        
        // TestRequest may add some default headers, so we just verify that it returns a Vec
        // and doesn't panic when there are no custom headers
        assert!(headers.is_empty() || !headers.is_empty()); // Always true, just testing no panic
    }

    // ============================================================================
    // Integration Tests with Mock Servers
    // ============================================================================

    #[tokio::test]
    async fn test_route_to_first_success() {
        let mock_url = crate::test_utils::mock_servers::create_mock_http_server(r#"{"success": true}"#, 200).await;
        
        let workers = vec![WorkerFactory::create_regular(mock_url)];
        let router = Router::Random {
            workers: Arc::new(RwLock::new(workers)),
            timeout_secs: 5,
            interval_secs: 1,
        };
        
        let client = reqwest::Client::new();
        let req = actix_web::test::TestRequest::get().to_http_request();
        
        let response = router.route_to_first(&client, "/test", &req).await;
        assert_eq!(response.status(), actix_web::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_all_loads_helper() {
        let mock_load_response = r#"{"load": 5}"#;
        let mock_url = crate::test_utils::mock_servers::create_mock_http_server(mock_load_response, 200).await;
        
        let client = reqwest::Client::new();
        let prefill_workers = vec![WorkerFactory::create_prefill(mock_url.clone(), Some(8081))];
        let decode_workers = vec![WorkerFactory::create_decode(mock_url)];
        
        let (prefill_loads, decode_loads) = get_loads_helper(&client, prefill_workers, decode_workers).await;
        
        assert_eq!(prefill_loads.len(), 1);
        assert_eq!(decode_loads.len(), 1);
        
        // Verify load structure
        assert!(prefill_loads[0]["engine"].as_str().unwrap().contains("Prefill"));
        assert!(decode_loads[0]["engine"].as_str().unwrap().contains("Decode"));
    }

    // ============================================================================
    // Timeout and Health Check Tests
    // ============================================================================

    #[tokio::test]
    async fn test_wait_for_healthy_workers_empty_list() {
        let workers: Vec<Arc<dyn Worker>> = vec![];
        let result = crate::core::worker::utils::wait_for_healthy_workers(
            &workers,
            1, // interval_secs
            1, // timeout_secs
        ).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_wait_for_healthy_workers_timeout() {
        let workers = vec![WorkerFactory::create_regular("http://nonexistent:8080".to_string())];
        let result = crate::core::worker::utils::wait_for_healthy_workers(
            &workers,
            1, // interval_secs 
            1, // timeout_secs
        ).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Timeout"));
    }

    // ============================================================================
    // Shared Test Infrastructure
    // ============================================================================



    #[tokio::test]
    async fn test_wait_for_healthy_workers_multiple_workers_mixed() {
        let (healthy_url, healthy_count) = create_enhanced_mock_health_server(
            vec![(200, r#"{"status": "healthy"}"#.to_string())],
            vec![Duration::from_millis(0)],
            Some(5)
        ).await;
        
        let (unhealthy_url, unhealthy_count) = create_enhanced_mock_health_server(
            vec![
                (503, r#"{"status": "unhealthy"}"#.to_string()),
                (503, r#"{"status": "unhealthy"}"#.to_string()),
                (200, r#"{"status": "healthy"}"#.to_string())
            ],
            vec![Duration::from_millis(0)],
            Some(10)
        ).await;
        
        let workers = vec![
            WorkerFactory::create_regular(healthy_url),
            WorkerFactory::create_regular(unhealthy_url),
        ];
        
        let result = crate::core::worker::utils::wait_for_healthy_workers(
            &workers,
            1, // interval_secs
            10, // timeout_secs
        ).await;
        assert!(result.is_ok());
        
        // Verify both workers are now healthy
        assert!(workers[0].is_healthy());
        assert!(workers[1].is_healthy());
        
        // Both workers should have been checked
        assert!(healthy_count.load(Ordering::SeqCst) >= 1);
        assert!(unhealthy_count.load(Ordering::SeqCst) >= 3);
    }



    #[tokio::test]
    async fn test_add_worker_with_health_check_success() {
        let (mock_url, call_count) = create_enhanced_mock_health_server(
            vec![(200, r#"{"status": "healthy"}"#.to_string())],
            vec![Duration::from_millis(0)],
            Some(5)
        ).await;
        
        let router = create_test_regular_router();
        let result = router.add_worker(&mock_url).await;
        
        assert!(result.is_ok());
        assert!(result.unwrap().contains("Successfully added worker"));
        assert!(call_count.load(Ordering::SeqCst) > 0);
        
        // Verify worker was actually added
        let workers = router.get_workers();
        let workers = workers.read().unwrap();
        assert!(workers.iter().any(|w| w.url() == mock_url));
    }

    #[tokio::test]
    async fn test_add_worker_with_health_check_failure() {
        let (mock_url, call_count) = create_enhanced_mock_health_server(
            vec![(503, r#"{"status": "unhealthy"}"#.to_string())],
            vec![Duration::from_millis(0)],
            Some(10)
        ).await;
        
        let router = create_test_regular_router();
        let result = router.add_worker(&mock_url).await;
        
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to health check worker"));
        assert!(call_count.load(Ordering::SeqCst) > 0);
        
        // Verify worker was not added
        let workers = router.get_workers();
        let workers = workers.read().unwrap();
        assert!(!workers.iter().any(|w| w.url() == mock_url));
    }

        #[tokio::test]
    async fn test_cache_aware_health_integration() {
        let (mock_url, call_count) = create_enhanced_mock_health_server(
            vec![(200, r#"{"status": "healthy"}"#.to_string())],
            vec![Duration::from_millis(0)],
            Some(10)
        ).await;
        
        let router = create_test_cache_aware_router();
        let result = router.add_worker(&mock_url).await;
        
        assert!(result.is_ok());
        assert!(call_count.load(Ordering::SeqCst) > 0);
        
        // Verify cache-aware structures were updated
        if let Router::CacheAware { running_queue, processed_queue, .. } = &router {
            let running_queue = running_queue.lock().unwrap();
            let processed_queue = processed_queue.lock().unwrap();
            
            assert!(running_queue.contains_key(&mock_url));
            assert!(processed_queue.contains_key(&mock_url));
            assert_eq!(*running_queue.get(&mock_url).unwrap(), 0);
            assert_eq!(*processed_queue.get(&mock_url).unwrap(), 0);
        }
    }

    // ============================================================================
    // Router Initialization with Batch Health Check Tests
    // ============================================================================

    #[tokio::test]
    async fn test_router_init_with_mixed_healthy_unhealthy_workers() {
        // Create healthy mock servers
        let (healthy_url1, healthy_count1) = create_enhanced_mock_health_server(
            vec![(200, r#"{"status": "healthy"}"#.to_string())],
            vec![Duration::from_millis(0)],
            Some(10)
        ).await;
        
        let (healthy_url2, healthy_count2) = create_enhanced_mock_health_server(
            vec![(200, r#"{"status": "healthy"}"#.to_string())],
            vec![Duration::from_millis(0)],
            Some(10)
        ).await;
        
        // Create unhealthy mock servers
        let (unhealthy_url1, unhealthy_count1) = create_enhanced_mock_health_server(
            vec![(503, r#"{"status": "unhealthy"}"#.to_string())],
            vec![Duration::from_millis(0)],
            Some(20)
        ).await;
        
        let (unhealthy_url2, unhealthy_count2) = create_enhanced_mock_health_server(
            vec![(500, r#"{"error": "internal error"}"#.to_string())],
            vec![Duration::from_millis(0)],
            Some(20)
        ).await;
        
        // Try to create router with mixed healthy/unhealthy workers
        let worker_urls = vec![
            healthy_url1.clone(),
            unhealthy_url1.clone(),
            healthy_url2.clone(),
            unhealthy_url2.clone(),
        ];
        
        let policy_config = PolicyConfig::RandomConfig {
            timeout_secs: 3, // Short timeout for test
            interval_secs: 1,
        };
        
        // Router initialization should fail because not all workers are healthy
        // Use spawn_blocking to handle the sync router creation from async context
        let result = tokio::task::spawn_blocking(move || {
            Router::new(worker_urls, policy_config)
        }).await.unwrap();
        
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Timeout"));
        
        // Verify health checks were attempted on all workers
        assert!(healthy_count1.load(Ordering::SeqCst) >= 1);
        assert!(healthy_count2.load(Ordering::SeqCst) >= 1);
        assert!(unhealthy_count1.load(Ordering::SeqCst) >= 1);
        assert!(unhealthy_count2.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn test_router_init_with_all_healthy_workers() {
        // Create multiple healthy mock servers
        let mut healthy_urls = Vec::new();
        let mut counters = Vec::new();
        
        for i in 0..4 {
            let (url, counter) = create_enhanced_mock_health_server(
                vec![(200, format!(r#"{{"status": "healthy", "worker_id": {}}}"#, i))],
                vec![Duration::from_millis(i * 50)], // Different delays
                Some(10)
            ).await;
            healthy_urls.push(url);
            counters.push(counter);
        }
        
        let policy_config = PolicyConfig::RoundRobinConfig {
            timeout_secs: 10,
            interval_secs: 1,
        };
        
        // Router initialization should succeed with all healthy workers
        let healthy_urls_clone = healthy_urls.clone();
        let result = tokio::task::spawn_blocking(move || {
            Router::new(healthy_urls_clone, policy_config)
        }).await.unwrap();
        
        assert!(result.is_ok());
        
        let router = result.unwrap();
        let workers = router.get_workers();
        let workers = workers.read().unwrap();
        
        // Verify all workers were added
        assert_eq!(workers.len(), 4);
        for (i, worker) in workers.iter().enumerate() {
            assert_eq!(worker.url(), healthy_urls[i]);
            assert!(worker.is_healthy());
        }
        
        // Verify health checks were performed
        for counter in &counters {
            assert!(counter.load(Ordering::SeqCst) >= 1);
        }
    }

    #[tokio::test]
    async fn test_router_init_with_eventually_healthy_workers() {
        // Create workers that become healthy after initial failures
        let mut worker_urls = Vec::new();
        let mut counters = Vec::new();
        
        for i in 0..3 {
            let (url, counter) = create_enhanced_mock_health_server(
                vec![
                    (503, r#"{"status": "starting"}"#.to_string()),
                    (503, r#"{"status": "loading"}"#.to_string()),
                    (200, format!(r#"{{"status": "healthy", "worker_id": {}}}"#, i)),
                ],
                vec![Duration::from_millis(100)], // Consistent delay
                Some(20)
            ).await;
            worker_urls.push(url);
            counters.push(counter);
        }
        
        let policy_config = PolicyConfig::CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 60,
            max_tree_size: 1000,
            timeout_secs: 15, // Longer timeout to allow workers to become healthy
            interval_secs: 2,
        };
        
        // Router initialization should eventually succeed
        let start = Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            Router::new(worker_urls, policy_config)
        }).await.unwrap();
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        assert!(duration >= Duration::from_secs(4)); // Should take some time
        
        let router = result.unwrap();
        let workers = router.get_workers();
        let workers = workers.read().unwrap();
        
        // Verify all workers are now healthy
        assert_eq!(workers.len(), 3);
        for worker in workers.iter() {
            assert!(worker.is_healthy());
        }
        
        // Verify multiple health checks were performed
        for counter in &counters {
            assert!(counter.load(Ordering::SeqCst) >= 3);
        }
    }

    // ============================================================================
    // Worker Addition with Health Check Tests
    // ============================================================================

    #[tokio::test]
    async fn test_add_multiple_workers_mixed_health_status() {
        let router = create_test_regular_router();
        
        // Create healthy and unhealthy workers
        let (healthy_url, healthy_count) = create_enhanced_mock_health_server(
            vec![(200, r#"{"status": "healthy"}"#.to_string())],
            vec![Duration::from_millis(0)],
            Some(5)
        ).await;
        
        let (unhealthy_url, unhealthy_count) = create_enhanced_mock_health_server(
            vec![(503, r#"{"status": "unhealthy"}"#.to_string())],
            vec![Duration::from_millis(0)],
            Some(10)
        ).await;
        
        // Add healthy worker - should succeed
        let result1 = router.add_worker(&healthy_url).await;
        assert!(result1.is_ok());
        assert!(result1.unwrap().contains("Successfully added worker"));
        
        // Add unhealthy worker - should fail
        let result2 = router.add_worker(&unhealthy_url).await;
        assert!(result2.is_err());
        assert!(result2.unwrap_err().contains("Failed to health check worker"));
        
        // Verify only healthy worker was added
        let workers = router.get_workers();
        let workers = workers.read().unwrap();
        let added_workers: Vec<_> = workers.iter()
            .filter(|w| w.url() == healthy_url || w.url() == unhealthy_url)
            .collect();
        
        assert_eq!(added_workers.len(), 1);
        assert_eq!(added_workers[0].url(), healthy_url);
        assert!(added_workers[0].is_healthy());
        
        // Verify health checks were performed
        assert!(healthy_count.load(Ordering::SeqCst) >= 1);
        assert!(unhealthy_count.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn test_add_worker_with_slow_health_check() {
        let router = create_test_regular_router();
        
        // Create worker with slow health check
        let (slow_url, slow_count) = create_enhanced_mock_health_server(
            vec![(200, r#"{"status": "healthy"}"#.to_string())],
            vec![Duration::from_millis(800)], // Slow response
            Some(5)
        ).await;
        
        let start = Instant::now();
        let result = router.add_worker(&slow_url).await;
        let duration = start.elapsed();
        
        // Should succeed despite slow response
        assert!(result.is_ok());
        assert!(duration >= Duration::from_millis(700)); // Should take at least 700ms
        
        // Verify worker was added and is healthy
        let workers = router.get_workers();
        let workers = workers.read().unwrap();
        let added_worker = workers.iter().find(|w| w.url() == slow_url);
        
        assert!(added_worker.is_some());
        assert!(added_worker.unwrap().is_healthy());
        assert_eq!(slow_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_add_duplicate_healthy_worker() {
        let router = create_test_regular_router();
        
        let (mock_url, call_count) = create_enhanced_mock_health_server(
            vec![(200, r#"{"status": "healthy"}"#.to_string())],
            vec![Duration::from_millis(0)],
            Some(10)
        ).await;
        
        // Add worker first time - should succeed
        let result1 = router.add_worker(&mock_url).await;
        assert!(result1.is_ok());
        
        // Add same worker again - should fail with duplicate error
        let result2 = router.add_worker(&mock_url).await;
        assert!(result2.is_err());
        assert!(result2.unwrap_err().contains("already exists"));
        
        // Verify worker appears only once
        let workers = router.get_workers();
        let workers = workers.read().unwrap();
        let matching_workers: Vec<_> = workers.iter()
            .filter(|w| w.url() == mock_url)
            .collect();
        
        assert_eq!(matching_workers.len(), 1);
        
        // Health check should have been called only once (for first addition)
        // because duplicate detection now happens before health check
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    // ============================================================================
    // Routing Failure and Worker Removal Tests
    // ============================================================================

    #[tokio::test]
    async fn test_routing_with_worker_failure_and_removal() {
        // Create a worker that fails during routing but passes initial health check
        let (failing_url, call_count) = create_enhanced_mock_health_server(
            vec![
                (200, r#"{"status": "healthy"}"#.to_string()), // Initial health check passes
                (500, r#"{"error": "internal error"}"#.to_string()), // Request 1 fails
                (500, r#"{"error": "internal error"}"#.to_string()), // Health check 1 fails
                (500, r#"{"error": "internal error"}"#.to_string()), // Request 2 fails
                (503, r#"{"status": "unhealthy"}"#.to_string()), // Health check 2 fails
                (500, r#"{"error": "internal error"}"#.to_string()), // Request 3 fails
                (503, r#"{"status": "unhealthy"}"#.to_string()), // Health check 3 fails
                (500, r#"{"error": "internal error"}"#.to_string()), // Additional requests fail
                (503, r#"{"status": "unhealthy"}"#.to_string()), // Additional health checks fail
            ],
            vec![Duration::from_millis(0)],
            Some(20)
        ).await;
        
        // Create router with the failing worker (using short TTL to avoid caching issues)
        let worker = create_test_worker_with_short_ttl(failing_url.clone());
        let router = Router::Random {
            workers: Arc::new(RwLock::new(vec![worker.clone()])),
            timeout_secs: 5,
            interval_secs: 1,
        };
        
        // Consume the first response (200 - health check) by calling check_health
        // This ensures routing requests get the subsequent 500 responses
        let _ = worker.check_health().await;
        // Wait for cache to expire
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        // Initial worker count
        {
            let workers = router.get_workers();
            let workers = workers.read().unwrap();
            assert_eq!(workers.len(), 1);
            assert_eq!(workers[0].url(), failing_url);
        }
        
        // Attempt to route request - should fail and remove worker
        let client = reqwest::Client::new();
        let req = actix_web::test::TestRequest::get().to_http_request();
        
        let response = router.route_to_first(&client, "/test", &req).await;
        
        // Should return error response since all workers failed
        assert_eq!(response.status(), actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
        
        // Worker should be removed after MAX_REQUEST_RETRIES failures
        {
            let workers = router.get_workers();
            let workers = workers.read().unwrap();
            assert_eq!(workers.len(), 0); // Worker should be removed
        }
        
        // Verify multiple calls were made (initial health check + failed requests + health checks)
        assert!(call_count.load(Ordering::SeqCst) >= 4);
    }

    #[tokio::test]
    async fn test_routing_with_multiple_workers_one_fails() {
        // Create one healthy worker and one that fails
        let (healthy_url, healthy_count) = create_enhanced_mock_health_server(
            vec![(200, r#"{"status": "healthy", "response": "success"}"#.to_string())],
            vec![Duration::from_millis(0)],
            Some(10)
        ).await;
        
        let (failing_url, failing_count) = create_enhanced_mock_health_server(
            vec![
                (200, r#"{"status": "healthy"}"#.to_string()), // Initial health check
                (500, r#"{"error": "internal error"}"#.to_string()), // Request 1 fails
                (500, r#"{"error": "internal error"}"#.to_string()), // Health check 1 fails
                (500, r#"{"error": "internal error"}"#.to_string()), // Request 2 fails
                (503, r#"{"status": "unhealthy"}"#.to_string()), // Health check 2 fails
                (500, r#"{"error": "internal error"}"#.to_string()), // Request 3 fails
                (503, r#"{"status": "unhealthy"}"#.to_string()), // Health check 3 fails
                (500, r#"{"error": "internal error"}"#.to_string()), // Additional requests fail
                (503, r#"{"status": "unhealthy"}"#.to_string()), // Additional health checks fail
            ],
            vec![Duration::from_millis(0)],
            Some(20)
        ).await;
        
        let healthy_worker = create_test_worker_with_short_ttl(healthy_url.clone());
        let failing_worker = create_test_worker_with_short_ttl(failing_url.clone());
        let router = Router::Random {
            workers: Arc::new(RwLock::new(vec![healthy_worker.clone(), failing_worker.clone()])),
            timeout_secs: 5,
            interval_secs: 1,
        };
        
        // Consume the first response for the failing worker (200 - health check)
        // This ensures routing requests get the subsequent 500 responses
        let _ = failing_worker.check_health().await;
        // Wait for cache to expire
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        // Initial worker count
        {
            let workers = router.get_workers();
            let workers = workers.read().unwrap();
            assert_eq!(workers.len(), 2);
        }
        
        let client = reqwest::Client::new();
        let req = actix_web::test::TestRequest::get().to_http_request();
        
        // Make multiple routing requests
        for _i in 0..5 {
            let response = router.route_to_first(&client, "/test", &req).await;
            
            // Some should succeed (from healthy worker), others might fail initially
            // but the healthy worker should eventually be selected
            if response.status().is_success() {
                // Successful response should be from healthy worker
                break;
            }
            
            // Allow some time between requests
            tokio::time::sleep(Duration::from_millis(100)).await;
            
            // Check if failing worker has been removed
            let workers = router.get_workers();
            let workers = workers.read().unwrap();
            if workers.len() == 1 {
                assert_eq!(workers[0].url(), healthy_url);
                break;
            }
        }
        
        // Eventually, the failing worker should be removed
        {
            let workers = router.get_workers();
            let workers = workers.read().unwrap();
            
            // Should have only the healthy worker remaining
            if workers.len() == 1 {
                assert_eq!(workers[0].url(), healthy_url);
                assert!(workers[0].is_healthy());
            }
        }
        
        // Verify both workers were accessed
        assert!(healthy_count.load(Ordering::SeqCst) >= 1);
        assert!(failing_count.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn test_routing_failure_with_health_check_recovery() {
        // Create worker that fails requests but stays healthy
        let (recovering_url, call_count) = create_enhanced_mock_health_server(
            vec![
                (200, r#"{"status": "healthy"}"#.to_string()), // Health checks pass
                (500, r#"{"error": "temporary error"}"#.to_string()), // Request fails
                (200, r#"{"status": "healthy"}"#.to_string()), // But health check passes
                (500, r#"{"error": "temporary error"}"#.to_string()),
                (200, r#"{"status": "healthy"}"#.to_string()),
                (500, r#"{"error": "temporary error"}"#.to_string()),
                (200, r#"{"status": "healthy"}"#.to_string()),
            ],
            vec![Duration::from_millis(0)],
            Some(20)
        ).await;
        
        let worker = create_test_worker_with_short_ttl(recovering_url.clone());
        let router = Router::Random {
            workers: Arc::new(RwLock::new(vec![worker.clone()])),
            timeout_secs: 5,
            interval_secs: 1,
        };
        
        // Consume the first response (200 - health check)
        let _ = worker.check_health().await;
        // Wait for cache to expire
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let client = reqwest::Client::new();
        let req = actix_web::test::TestRequest::get().to_http_request();
        
        // Attempt routing
        let response = router.route_to_first(&client, "/test", &req).await;
        
        // Should return the 500 error since worker is healthy (so request is considered bad)
        assert_eq!(response.status(), actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
        
        // Worker should NOT be removed since health checks pass
        {
            let workers = router.get_workers();
            let workers = workers.read().unwrap();
            assert_eq!(workers.len(), 1);
            assert_eq!(workers[0].url(), recovering_url);
            assert!(workers[0].is_healthy());
        }
        
        // Verify both request and health check calls were made
        assert!(call_count.load(Ordering::SeqCst) >= 2);
    }

    #[tokio::test]
    async fn test_routing_with_cache_aware_worker_removal() {
        // Create failing worker
        let (failing_url, call_count) = create_enhanced_mock_health_server(
            vec![
                (200, r#"{"status": "healthy"}"#.to_string()), // Initial health check
                (500, r#"{"error": "internal error"}"#.to_string()), // Request 1 fails
                (500, r#"{"error": "internal error"}"#.to_string()), // Health check 1 fails
                (500, r#"{"error": "internal error"}"#.to_string()), // Request 2 fails
                (503, r#"{"status": "unhealthy"}"#.to_string()), // Health check 2 fails
                (500, r#"{"error": "internal error"}"#.to_string()), // Request 3 fails
                (503, r#"{"status": "unhealthy"}"#.to_string()), // Health check 3 fails
                (500, r#"{"error": "internal error"}"#.to_string()), // Additional requests fail
                (503, r#"{"status": "unhealthy"}"#.to_string()), // Additional health checks fail
            ],
            vec![Duration::from_millis(0)],
            Some(20)
        ).await;
        
        // Create cache-aware router
        let worker = create_test_worker_with_short_ttl(failing_url.clone());
        let mut running_queue = std::collections::HashMap::new();
        let mut processed_queue = std::collections::HashMap::new();
        
        running_queue.insert(worker.url().to_string(), 0);
        processed_queue.insert(worker.url().to_string(), 0);

        let tree = Arc::new(Mutex::new(crate::tree::Tree::new()));
        tree.lock().unwrap().insert("", worker.url());

        let router = Router::CacheAware {
            workers: Arc::new(RwLock::new(vec![worker.clone()])),
            tree,
            running_queue: Arc::new(Mutex::new(running_queue)),
            processed_queue: Arc::new(Mutex::new(processed_queue)),
            cache_threshold: 0.5,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            timeout_secs: 5,
            interval_secs: 1,
            _eviction_thread: None,
        };
        
        // Consume the first response (200 - health check)
        let _ = worker.check_health().await;
        // Wait for cache to expire
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        // Verify initial cache-aware structures
        if let Router::CacheAware { running_queue, processed_queue, .. } = &router {
            let running_queue = running_queue.lock().unwrap();
            let processed_queue = processed_queue.lock().unwrap();
            
            assert!(running_queue.contains_key(&failing_url));
            assert!(processed_queue.contains_key(&failing_url));
        }
        
        let client = reqwest::Client::new();
        let req = actix_web::test::TestRequest::get().to_http_request();
        
        // Attempt routing - should fail and remove worker
        let response = router.route_to_first(&client, "/test", &req).await;
        assert_eq!(response.status(), actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
        
        // Worker should be removed from all cache-aware structures
        {
            let workers = router.get_workers();
            let workers = workers.read().unwrap();
            assert_eq!(workers.len(), 0);
        }
        
        if let Router::CacheAware { running_queue, processed_queue, .. } = &router {
            let running_queue = running_queue.lock().unwrap();
            let processed_queue = processed_queue.lock().unwrap();
            
            // Cache structures should be cleaned up
            assert!(!running_queue.contains_key(&failing_url));
            assert!(!processed_queue.contains_key(&failing_url));
        }
        
        assert!(call_count.load(Ordering::SeqCst) >= 4);
    }

    #[tokio::test]
    async fn test_routing_with_all_workers_failing() {
        // Create multiple workers that all fail
        let mut failing_urls = Vec::new();
        let mut counters = Vec::new();
        
        for i in 0..2 {
            let (url, counter) = create_enhanced_mock_health_server(
                vec![
                    (200, r#"{"status": "healthy"}"#.to_string()), // Initial health check
                    (500, format!(r#"{{"error": "worker {} failed"}}"#, i)), // Request 1 fails
                    (500, format!(r#"{{"error": "worker {} failed"}}"#, i)), // Health check 1 fails
                    (500, format!(r#"{{"error": "worker {} failed"}}"#, i)), // Request 2 fails
                    (503, r#"{"status": "unhealthy"}"#.to_string()), // Health check 2 fails
                    (500, format!(r#"{{"error": "worker {} failed"}}"#, i)), // Request 3 fails
                    (503, r#"{"status": "unhealthy"}"#.to_string()), // Health check 3 fails
                    (500, format!(r#"{{"error": "worker {} failed"}}"#, i)), // Additional requests fail
                    (503, r#"{"status": "unhealthy"}"#.to_string()), // Additional health checks fail
                ],
                vec![Duration::from_millis(0)],
                Some(20)
            ).await;
            failing_urls.push(url);
            counters.push(counter);
        }
        
        let workers: Vec<_> = failing_urls.iter()
            .map(|url| create_test_worker_with_short_ttl(url.clone()))
            .collect();
        
        // Consume the first response (200 - health check) for each worker
        for worker in &workers {
            let _ = worker.check_health().await;
        }
        // Wait for cache to expire
        tokio::time::sleep(Duration::from_millis(10)).await;
            
        let router = Router::RoundRobin {
            workers: Arc::new(RwLock::new(workers)),
            current_index: AtomicUsize::new(0),
            timeout_secs: 5,
            interval_secs: 1,
        };
        
        // Initial worker count
        {
            let workers = router.get_workers();
            let workers = workers.read().unwrap();
            assert_eq!(workers.len(), 2);
        }
        
        let client = reqwest::Client::new();
        let req = actix_web::test::TestRequest::get().to_http_request();
        
        // Attempt routing - should exhaust all workers and fail
        let response = router.route_to_first(&client, "/test", &req).await;
        assert_eq!(response.status(), actix_web::http::StatusCode::INTERNAL_SERVER_ERROR);
        
        // All workers should eventually be removed
        {
            let workers = router.get_workers();
            let workers = workers.read().unwrap();
            assert_eq!(workers.len(), 0);
        }
        
        // Verify all workers were attempted
        for counter in &counters {
            assert!(counter.load(Ordering::SeqCst) >= 1);
        }
    }

 
}
