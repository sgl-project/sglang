use crate::core::{HealthChecker, Worker, WorkerFactory};
use crate::pd_router::PDRouter;
use crate::pd_types::PDSelectionPolicy;
use crate::tree::Tree;
use ::metrics::{counter, gauge, histogram};
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use futures_util::{StreamExt, TryStreamExt};
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
        workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
        current_index: AtomicUsize,
        timeout_secs: u64,
        interval_secs: u64,
        _health_checker: Option<HealthChecker>,
    },
    Random {
        workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
        timeout_secs: u64,
        interval_secs: u64,
        _health_checker: Option<HealthChecker>,
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
        workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
        tree: Arc<Mutex<Tree>>,
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        timeout_secs: u64,
        interval_secs: u64,
        _eviction_thread: Option<thread::JoinHandle<()>>,
        _health_checker: Option<HealthChecker>,
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
        gauge!("sgl_router_active_workers").set(worker_urls.len() as f64);

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

        // For PrefillDecode, we need to handle workers differently
        match &policy_config {
            PolicyConfig::PrefillDecodeConfig { .. } => {
                // PD mode doesn't use the worker_urls parameter
                // We'll validate PD workers separately
            }
            _ => {
                // Wait until all workers are healthy for regular modes
                let worker_urls = worker_urls.clone();
                std::thread::spawn(move || {
                    Self::wait_for_healthy_workers(&worker_urls, timeout_secs, interval_secs)
                })
                .join()
                .map_err(|e| {
                    error!("Health-check thread panicked: {:?}", e);
                    format!("Health-check thread panicked: {e:?}")
                })??;
            }
        }

        // Create Worker trait objects from URLs
        let workers: Vec<Box<dyn Worker>> = worker_urls
            .iter()
            .map(|url| WorkerFactory::create_regular(url.clone()))
            .collect();

        // Create router based on policy...
        Ok(match policy_config {
            PolicyConfig::RandomConfig {
                timeout_secs,
                interval_secs,
            } => {
                let workers = Arc::new(RwLock::new(workers));
                let health_checker =
                    crate::core::start_health_checker(Arc::clone(&workers), interval_secs);
                Router::Random {
                    workers,
                    timeout_secs,
                    interval_secs,
                    _health_checker: Some(health_checker),
                }
            }
            PolicyConfig::RoundRobinConfig {
                timeout_secs,
                interval_secs,
            } => {
                let workers = Arc::new(RwLock::new(workers));
                let health_checker =
                    crate::core::start_health_checker(Arc::clone(&workers), interval_secs);
                Router::RoundRobin {
                    workers,
                    current_index: std::sync::atomic::AtomicUsize::new(0),
                    timeout_secs,
                    interval_secs,
                    _health_checker: Some(health_checker),
                }
            }
            PolicyConfig::CacheAwareConfig {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
                timeout_secs,
                interval_secs,
            } => {
                let tree = Arc::new(Mutex::new(Tree::new()));

                // Create background eviction thread
                let tree_clone = Arc::clone(&tree);
                let workers = Arc::new(RwLock::new(workers));
                let workers_clone = Arc::clone(&workers);
                let eviction_thread = thread::spawn(move || {
                    loop {
                        // Sleep for the specified interval
                        thread::sleep(Duration::from_secs(eviction_interval_secs));

                        let locked_tree_clone = tree_clone.lock().unwrap();
                        // Run eviction
                        locked_tree_clone.evict_tenant_by_size(max_tree_size);
                        drop(locked_tree_clone);

                        // Log worker loads and processed requests
                        let workers_guard = workers_clone.read().unwrap();
                        let loads: Vec<(String, usize)> = workers_guard
                            .iter()
                            .map(|w| (w.url().to_string(), w.load()))
                            .collect();
                        info!("Worker loads: {:?}", loads);

                        let processed: Vec<(String, usize)> = workers_guard
                            .iter()
                            .map(|w| (w.url().to_string(), w.processed_requests()))
                            .collect();
                        info!("Processed requests: {:?}", processed);
                    }
                });

                for worker in workers.read().unwrap().iter() {
                    tree.lock().unwrap().insert("", worker.url());
                }

                let health_checker =
                    crate::core::start_health_checker(Arc::clone(&workers), interval_secs);

                Router::CacheAware {
                    workers,
                    tree,
                    cache_threshold,
                    balance_abs_threshold,
                    balance_rel_threshold,
                    timeout_secs,
                    interval_secs,
                    _eviction_thread: Some(eviction_thread),
                    _health_checker: Some(health_checker),
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

    /// Get the current list of worker URLs
    pub fn get_worker_urls(&self) -> Vec<String> {
        match self {
            Router::RoundRobin { workers, .. }
            | Router::Random { workers, .. }
            | Router::CacheAware { workers, .. } => workers
                .read()
                .unwrap()
                .iter()
                .map(|w| w.url().to_string())
                .collect(),
            Router::PrefillDecode { .. } => Vec::new(),
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

    fn select_first_worker(&self) -> Result<String, String> {
        match self {
            Router::RoundRobin { workers, .. }
            | Router::Random { workers, .. }
            | Router::CacheAware { workers, .. } => {
                let workers_guard = workers.read().unwrap();
                if workers_guard.is_empty() {
                    Err("No workers are available".to_string())
                } else {
                    Ok(workers_guard[0].url().to_string())
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
        worker_url: &str,
        route: &str,
        req: &HttpRequest,
    ) -> HttpResponse {
        let start = Instant::now();
        let mut request_builder = client.get(format!("{}{}", worker_url, route));

        // Copy all headers from original request except for /health because it does not need authorization
        if route != "/health" {
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
                worker_url, e
            )),
        };

        // Record request metrics
        if route != "/health" {
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
                Ok(worker_url) => {
                    let mut request_retries = 0;

                    // Try the same worker multiple times
                    while request_retries < MAX_REQUEST_RETRIES {
                        if total_retries >= 1 {
                            info!("Retrying request after {} failed attempts", total_retries);
                        }

                        let response = self.send_request(client, &worker_url, route, req).await;

                        if response.status().is_success() {
                            return response;
                        } else {
                            // if the worker is healthy, it means the request is bad, so return the error response
                            let health_response =
                                self.send_request(client, &worker_url, "/health", req).await;
                            if health_response.status().is_success() {
                                return response;
                            }
                        }

                        warn!(
                            "Request to {} failed (attempt {}/{})",
                            worker_url,
                            request_retries + 1,
                            MAX_REQUEST_RETRIES
                        );

                        request_retries += 1;
                        total_retries += 1;

                        if request_retries == MAX_REQUEST_RETRIES {
                            warn!("Removing failed worker: {}", worker_url);
                            self.remove_worker(&worker_url);
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
        let worker_urls = match self {
            Router::PrefillDecode { .. } => {
                // For PD mode, route_to_all is not supported directly
                // It should be handled by PDRouter if needed
                return HttpResponse::NotImplemented()
                    .body("route_to_all not implemented for PrefillDecode mode");
            }
            _ => self.get_worker_urls(),
        };

        // Send requests to all workers concurrently
        let mut tasks = Vec::new();
        for worker_url in &worker_urls {
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

        let urls = self.get_worker_urls();
        let prefill_urls: Vec<String> = Vec::new();
        let decode_urls = urls;

        // Collect loads from all servers
        let mut prefill_loads = Vec::new();
        let mut decode_loads = Vec::new();

        // Get prefill loads
        for url in &prefill_urls {
            let load = self.get_worker_load(client, url).await.unwrap_or(-1);
            prefill_loads.push(serde_json::json!({
                "engine": format!("(Prefill@{})", url),
                "load": load as i64
            }));
        }

        // Get decode loads
        for url in &decode_urls {
            let load = self.get_worker_load(client, url).await.unwrap_or(-1);
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
                    let mut request_retries = 0;

                    // Try the same worker multiple times
                    while request_retries < MAX_REQUEST_RETRIES {
                        if total_retries >= 1 {
                            info!("Retrying request after {} failed attempts", total_retries);
                            counter!("sgl_router_retries_total", "route" => route.to_string())
                                .increment(1);
                        }

                        // For CacheAware router, increment load before request
                        let load_incremented = match self {
                            Router::CacheAware { workers, .. } => {
                                let workers_guard = workers.read().unwrap();
                                if let Some(worker) =
                                    workers_guard.iter().find(|w| w.url() == &worker_url)
                                {
                                    worker.increment_load();
                                    gauge!("sgl_router_running_requests", "worker" => worker_url.to_string())
                                        .set(worker.load() as f64);
                                    true
                                } else {
                                    false
                                }
                            }
                            _ => false,
                        };

                        // Send typed request directly
                        let response = self
                            .send_typed_request(
                                client,
                                req,
                                typed_req,
                                route,
                                &worker_url,
                                is_stream,
                                load_incremented,
                            )
                            .await;

                        if response.status().is_success() {
                            let duration = start.elapsed();
                            histogram!("sgl_router_generate_duration_seconds", "route" => route.to_string())
                                .record(duration.as_secs_f64());
                            return response;
                        } else {
                            // if the worker is healthy, it means the request is bad, so return the error response
                            let health_response =
                                self.send_request(client, &worker_url, "/health", req).await;
                            if health_response.status().is_success() {
                                counter!("sgl_router_request_errors_total", "route" => route.to_string())
                                    .increment(1);
                                return response;
                            }
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
                            self.remove_worker(&worker_url);
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

    // Helper method to select worker from text (returns index for RoundRobin/Random, URL for CacheAware)
    fn select_generate_worker_from_text(&self, text: &str) -> String {
        match self {
            Router::RoundRobin {
                workers,
                current_index,
                ..
            } => {
                let workers_guard = workers.read().unwrap();
                let idx = current_index
                    .fetch_update(
                        std::sync::atomic::Ordering::SeqCst,
                        std::sync::atomic::Ordering::SeqCst,
                        |x| Some((x + 1) % workers_guard.len()),
                    )
                    .unwrap();
                workers_guard[idx].url().to_string()
            }

            Router::Random { workers, .. } => {
                let workers_guard = workers.read().unwrap();
                workers_guard[rand::random::<usize>() % workers_guard.len()]
                    .url()
                    .to_string()
            }

            Router::CacheAware {
                workers,
                tree,
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                ..
            } => {
                let tree = tree.lock().unwrap();
                let workers_guard = workers.read().unwrap();

                // Get current load statistics from workers
                let loads: Vec<usize> = workers_guard.iter().map(|w| w.load()).collect();
                let max_load = *loads.iter().max().unwrap_or(&0);
                let min_load = *loads.iter().min().unwrap_or(&0);

                // Load is considered imbalanced if:
                // 1. (max - min) > abs_threshold AND
                // 2. max > rel_threshold * min
                let is_imbalanced = max_load.saturating_sub(min_load) > *balance_abs_threshold
                    && (max_load as f32) > (min_load as f32 * balance_rel_threshold);

                let selected_url = if is_imbalanced {
                    // Log load balancing trigger and current queue state
                    let worker_loads: Vec<(String, usize)> = workers_guard
                        .iter()
                        .map(|w| (w.url().to_string(), w.load()))
                        .collect();

                    info!(
                        "Load balancing triggered due to workload imbalance:\n\
                        Max load: {}, Min load: {}\n\
                        Current worker loads: {:?}",
                        max_load, min_load, worker_loads
                    );

                    counter!("sgl_router_load_balancing_events_total").increment(1);
                    gauge!("sgl_router_max_load").set(max_load as f64);
                    gauge!("sgl_router_min_load").set(min_load as f64);

                    // Use shortest queue routing when load is imbalanced
                    workers_guard
                        .iter()
                        .min_by_key(|w| w.load())
                        .map(|w| w.url().to_string())
                        .unwrap_or_else(|| workers_guard[0].url().to_string())
                } else {
                    // Use cache-aware routing when load is balanced
                    let (matched_text, matched_worker) = tree.prefix_match(&text);
                    let matched_rate =
                        matched_text.chars().count() as f32 / text.chars().count() as f32;

                    if matched_rate > *cache_threshold {
                        counter!("sgl_router_cache_hits_total").increment(1);
                        matched_worker.to_string()
                    } else {
                        counter!("sgl_router_cache_misses_total").increment(1);
                        tree.get_smallest_tenant()
                    }
                };

                // Find the selected worker and increment processed counter only
                if let Some(worker) = workers_guard.iter().find(|w| w.url() == &selected_url) {
                    worker.increment_processed();
                    counter!("sgl_router_processed_requests_total", "worker" => selected_url.to_string())
                        .increment(1);
                }

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
        load_incremented: bool, // Whether load was incremented for this request
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

                // Decrement load on error for CacheAware router
                if load_incremented {
                    if let Router::CacheAware { workers, .. } = self {
                        if let Ok(workers_guard) = workers.read() {
                            if let Some(worker) =
                                workers_guard.iter().find(|w| w.url() == worker_url)
                            {
                                worker.decrement_load();
                                gauge!("sgl_router_running_requests", "worker" => worker_url.to_string())
                                    .set(worker.load() as f64);
                            }
                        }
                    }
                }

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

            // Decrement load counter for non-streaming CacheAware requests
            if load_incremented && !is_stream {
                if let Router::CacheAware { workers, .. } = self {
                    if let Ok(workers_guard) = workers.read() {
                        if let Some(worker) = workers_guard.iter().find(|w| w.url() == worker_url) {
                            worker.decrement_load();
                            gauge!("sgl_router_running_requests", "worker" => worker_url.to_string())
                                .set(worker.load() as f64);
                        }
                    }
                }
            }

            // Record metrics
            let duration = start.elapsed();
            histogram!("sgl_router_generate_duration_seconds", "route" => route.to_string())
                .record(duration.as_secs_f64());
            counter!("sgl_router_requests_total", "route" => route.to_string()).increment(1);

            response
        } else if let Router::CacheAware { workers, .. } = self {
            // For streaming with CacheAware router, we need to manually decrement when done
            let workers = Arc::clone(workers);
            let worker_url = worker_url.to_string();

            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(
                    res.bytes_stream()
                        .map_err(|_| {
                            actix_web::error::ErrorInternalServerError("Failed to read stream")
                        })
                        .inspect(move |bytes| {
                            if let Ok(bytes) = bytes {
                                if bytes
                                    .as_ref()
                                    .windows(12)
                                    .any(|window| window == b"data: [DONE]")
                                {
                                    if let Ok(workers_guard) = workers.read() {
                                        if let Some(worker) =
                                            workers_guard.iter().find(|w| w.url() == &worker_url)
                                        {
                                            worker.decrement_load();
                                            gauge!("sgl_router_running_requests", "worker" => worker_url.to_string())
                                                .set(worker.load() as f64);
                                            debug!("Streaming is done!!")
                                        }
                                    }
                                }
                            }
                        }),
                )
        } else {
            // For non-CacheAware routers, just stream without load tracking
            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(res.bytes_stream().map_err(|_| {
                    actix_web::error::ErrorInternalServerError("Failed to read stream")
                }))
        }
    }

    pub async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
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

        let start_time = std::time::Instant::now();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        loop {
            if start_time.elapsed() > Duration::from_secs(timeout_secs) {
                error!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    timeout_secs, worker_url
                );
                return Err(format!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    timeout_secs, worker_url
                ));
            }

            match client.get(&format!("{}/health", worker_url)).send().await {
                Ok(res) => {
                    if res.status().is_success() {
                        match self {
                            Router::RoundRobin { workers, .. }
                            | Router::Random { workers, .. }
                            | Router::CacheAware { workers, .. } => {
                                info!("Worker {} health check passed", worker_url);
                                let mut workers_guard = workers.write().unwrap();
                                if workers_guard.iter().any(|w| w.url() == worker_url) {
                                    return Err(format!("Worker {} already exists", worker_url));
                                }
                                info!("Added worker: {}", worker_url);
                                let new_worker =
                                    WorkerFactory::create_regular(worker_url.to_string());
                                workers_guard.push(new_worker);
                                gauge!("sgl_router_active_workers").set(workers_guard.len() as f64);
                            }
                            Router::PrefillDecode { .. } => {
                                return Err("Adding workers to PrefillDecode router not supported via add_worker. Use dedicated PD management methods.".to_string());
                            }
                        }

                        // If cache aware, add worker to tree
                        if let Router::CacheAware { tree, .. } = self {
                            // Add worker to tree
                            tree.lock().unwrap().insert("", worker_url);
                        }

                        return Ok(format!("Successfully added worker: {}", worker_url));
                    } else {
                        info!(
                            "Worker {} health check is pending with status: {}.",
                            worker_url,
                            res.status()
                        );
                        // if the url does not have http or https prefix, warn users
                        if !worker_url.starts_with("http://") && !worker_url.starts_with("https://")
                        {
                            warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                        }

                        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                        continue;
                    }
                }
                Err(e) => {
                    info!(
                        "Worker {} health check is pending with error: {}",
                        worker_url, e
                    );

                    // if the url does not have http or https prefix, warn users
                    if !worker_url.starts_with("http://") && !worker_url.starts_with("https://") {
                        warn!("The worker url {} does not have http or https prefix. Please add the prefix to the url.", worker_url);
                    }

                    tokio::time::sleep(Duration::from_secs(interval_secs)).await;
                    continue;
                }
            }
        }
    }

    pub fn remove_worker(&self, worker_url: &str) {
        match self {
            Router::RoundRobin { workers, .. }
            | Router::Random { workers, .. }
            | Router::CacheAware { workers, .. } => {
                let mut workers_guard = workers.write().unwrap();
                if let Some(index) = workers_guard.iter().position(|w| w.url() == worker_url) {
                    workers_guard.remove(index);
                    info!("Removed worker: {}", worker_url);
                    gauge!("sgl_router_active_workers").set(workers_guard.len() as f64);
                } else {
                    warn!("Worker {} not found, skipping removal", worker_url);
                    return;
                }
            }
            Router::PrefillDecode { .. } => {
                warn!("Removing workers from PrefillDecode router not supported via remove_worker. Use dedicated PD management methods.");
                return;
            }
        }

        // if cache aware, remove the worker from the tree
        if let Router::CacheAware { tree, .. } = self {
            tree.lock().unwrap().remove_tenant(&worker_url);
            info!("Removed worker from tree: {}", worker_url);
        }
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

    async fn get_worker_load(&self, client: &reqwest::Client, worker_url: &str) -> Option<isize> {
        match client.get(&format!("{}/get_load", worker_url)).send().await {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service_discovery::PodType;

    fn create_test_regular_router() -> Router {
        let workers = vec![
            WorkerFactory::create_regular("http://worker1:8080".to_string()),
            WorkerFactory::create_regular("http://worker2:8080".to_string()),
        ];
        Router::Random {
            workers: Arc::new(RwLock::new(workers)),
            timeout_secs: 5,
            interval_secs: 1,
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

    // #[test]
    // fn test_router_get_worker_urls_pd_mode() {
    //     // For PD mode, get_worker_urls returns empty list
    //     // Note: PDRouter::new requires health checks which fail in tests
    //     // This test would need a mock server or different test setup
    // }

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

    // #[tokio::test]
    // async fn test_add_pd_worker_with_pd_router_regular_type() {
    //     // Note: PDRouter::new requires health checks which fail in tests
    //     // This test would need a mock server or different test setup
    // }

    // #[tokio::test]
    // async fn test_remove_pd_worker_with_pd_router_regular_type() {
    //     // Note: PDRouter::new requires health checks which fail in tests
    //     // This test would need a mock server or different test setup
    // }

    #[test]
    fn test_select_first_worker_regular() {
        let router = create_test_regular_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "http://worker1:8080");
    }

    // #[test]
    // fn test_select_first_worker_pd_mode() {
    //     // Note: PDRouter::new requires health checks which fail in tests
    //     // This test would need a mock server or different test setup
    // }

    #[test]
    fn test_wait_for_healthy_workers_empty_list() {
        let result = Router::wait_for_healthy_workers(&[], 1, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wait_for_healthy_workers_invalid_urls() {
        // This test will timeout quickly since the URLs are invalid
        let result =
            Router::wait_for_healthy_workers(&["http://nonexistent:8080".to_string()], 1, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Timeout"));
    }
}
