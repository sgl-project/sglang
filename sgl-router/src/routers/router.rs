use crate::core::{HealthChecker, Worker, WorkerFactory};
use crate::metrics::RouterMetrics;
use crate::policies::LoadBalancingPolicy;
use actix_web::http::header::{HeaderValue, CONTENT_TYPE};
use actix_web::{HttpRequest, HttpResponse};
use futures_util::{StreamExt, TryStreamExt};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};
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

/// Regular router that uses injected load balancing policies
#[derive(Debug)]
pub struct Router {
    workers: Arc<RwLock<Vec<Box<dyn Worker>>>>,
    policy: Arc<dyn LoadBalancingPolicy>,
    timeout_secs: u64,
    interval_secs: u64,
    _worker_loads: Arc<tokio::sync::watch::Receiver<HashMap<String, isize>>>,
    _load_monitor_handle: Option<Arc<tokio::task::JoinHandle<()>>>,
    _health_checker: Option<HealthChecker>,
}

impl Router {
    /// Create a new router with injected policy
    pub fn new(
        worker_urls: Vec<String>,
        policy: Arc<dyn LoadBalancingPolicy>,
        timeout_secs: u64,
        interval_secs: u64,
    ) -> Result<Self, String> {
        // Update active workers gauge
        RouterMetrics::set_active_workers(worker_urls.len());

        // Wait for workers to be healthy (skip if empty - for service discovery mode)
        if !worker_urls.is_empty() {
            Self::wait_for_healthy_workers(&worker_urls, timeout_secs, interval_secs)?;
        }

        // Create Worker trait objects from URLs
        let workers: Vec<Box<dyn Worker>> = worker_urls
            .iter()
            .map(|url| WorkerFactory::create_regular(url.clone()))
            .collect();

        // Initialize policy with workers if needed (e.g., for cache-aware)
        if let Some(cache_aware) = policy
            .as_any()
            .downcast_ref::<crate::policies::CacheAwarePolicy>()
        {
            cache_aware.init_workers(&workers);
        }

        let workers = Arc::new(RwLock::new(workers));
        let health_checker = crate::core::start_health_checker(Arc::clone(&workers), interval_secs);

        // Setup load monitoring for PowerOfTwo policy
        let (tx, rx) = tokio::sync::watch::channel(HashMap::new());
        let worker_loads = Arc::new(rx);

        let load_monitor_handle = if policy.name() == "power_of_two" {
            let monitor_urls = worker_urls.clone();
            let monitor_interval = interval_secs;
            let policy_clone = Arc::clone(&policy);

            Some(Arc::new(tokio::spawn(async move {
                Self::monitor_worker_loads(monitor_urls, tx, monitor_interval, policy_clone).await;
            })))
        } else {
            None
        };

        Ok(Router {
            workers,
            policy,
            timeout_secs,
            interval_secs,
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
        let workers_guard = self.workers.read().unwrap();
        if workers_guard.is_empty() {
            Err("No workers are available".to_string())
        } else {
            Ok(workers_guard[0].url().to_string())
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
            RouterMetrics::record_request(route);
            RouterMetrics::record_request_duration(route, duration);

            if !response.status().is_success() {
                RouterMetrics::record_request_error(route, "request_failed");
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
                    RouterMetrics::record_retry(route);
                }

                // Increment load before request if using RAII load tracking
                let load_incremented = if self.policy.name() == "cache_aware" {
                    let workers_guard = self.workers.read().unwrap();
                    if let Some(worker) = workers_guard.iter().find(|w| w.url() == &worker_url) {
                        worker.increment_load();
                        RouterMetrics::set_running_requests(&worker_url, worker.load());
                        true
                    } else {
                        false
                    }
                } else {
                    false
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
                    RouterMetrics::record_generate_duration(duration);
                    return response;
                } else {
                    // if the worker is healthy, it means the request is bad, so return the error response
                    let health_response =
                        self.send_request(client, &worker_url, "/health", req).await;
                    if health_response.status().is_success() {
                        RouterMetrics::record_request_error(route, "request_failed");
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

        RouterMetrics::record_request_error(route, "request_failed");
        HttpResponse::InternalServerError().body("All retry attempts failed")
    }

    // Helper method to select worker from text using the policy
    fn select_generate_worker_from_text(&self, text: &str) -> String {
        let workers = self.workers.read().unwrap();

        match self.policy.select_worker(&workers, Some(text)) {
            Some(idx) => workers[idx].url().to_string(),
            None => {
                warn!("No healthy workers available");
                String::new()
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

                // Decrement load on error if it was incremented
                if load_incremented {
                    if let Ok(workers_guard) = self.workers.read() {
                        if let Some(worker) = workers_guard.iter().find(|w| w.url() == worker_url) {
                            worker.decrement_load();
                            RouterMetrics::set_running_requests(&worker_url, worker.load());
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

            // Decrement load counter for non-streaming requests if it was incremented
            if load_incremented && !is_stream {
                if let Ok(workers_guard) = self.workers.read() {
                    if let Some(worker) = workers_guard.iter().find(|w| w.url() == worker_url) {
                        worker.decrement_load();
                        RouterMetrics::set_running_requests(&worker_url, worker.load());
                    }
                }
            }

            // Record metrics
            let duration = start.elapsed();
            RouterMetrics::record_generate_duration(duration);
            RouterMetrics::record_request(route);

            response
        } else if load_incremented {
            // For streaming with load tracking, we need to manually decrement when done
            let workers = Arc::clone(&self.workers);
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
                                            RouterMetrics::set_running_requests(
                                                &worker_url,
                                                worker.load(),
                                            );
                                            debug!("Streaming is done!!")
                                        }
                                    }
                                }
                            }
                        }),
                )
        } else {
            // For requests without load tracking, just stream
            HttpResponse::build(status)
                .insert_header((CONTENT_TYPE, HeaderValue::from_static("text/event-stream")))
                .streaming(res.bytes_stream().map_err(|_| {
                    actix_web::error::ErrorInternalServerError("Failed to read stream")
                }))
        }
    }

    pub async fn add_worker(&self, worker_url: &str) -> Result<String, String> {
        let start_time = std::time::Instant::now();
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(self.timeout_secs))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        loop {
            if start_time.elapsed() > Duration::from_secs(self.timeout_secs) {
                error!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    self.timeout_secs, worker_url
                );
                return Err(format!(
                    "Timeout {}s waiting for worker {} to become healthy. Please set --router-worker-startup-timeout-secs (sglang_router.launch_server) or --worker-startup-timeout-secs (sglang_worker.router) to a larger value",
                    self.timeout_secs, worker_url
                ));
            }

            match client.get(&format!("{}/health", worker_url)).send().await {
                Ok(res) => {
                    if res.status().is_success() {
                        info!("Worker {} health check passed", worker_url);
                        let mut workers_guard = self.workers.write().unwrap();
                        if workers_guard.iter().any(|w| w.url() == worker_url) {
                            return Err(format!("Worker {} already exists", worker_url));
                        }
                        info!("Added worker: {}", worker_url);
                        let new_worker = WorkerFactory::create_regular(worker_url.to_string());
                        workers_guard.push(new_worker);
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

                        tokio::time::sleep(Duration::from_secs(self.interval_secs)).await;
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

                    tokio::time::sleep(Duration::from_secs(self.interval_secs)).await;
                    continue;
                }
            }
        }
    }

    pub fn remove_worker(&self, worker_url: &str) {
        let mut workers_guard = self.workers.write().unwrap();
        if let Some(index) = workers_guard.iter().position(|w| w.url() == worker_url) {
            workers_guard.remove(index);
            info!("Removed worker: {}", worker_url);
            RouterMetrics::set_active_workers(workers_guard.len());
        } else {
            warn!("Worker {} not found, skipping removal", worker_url);
            return;
        }

        // If cache aware policy, remove the worker from the tree
        if let Some(cache_aware) = self
            .policy
            .as_any()
            .downcast_ref::<crate::policies::CacheAwarePolicy>()
        {
            cache_aware.remove_worker(worker_url);
            info!("Removed worker from tree: {}", worker_url);
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

    // Background task to monitor worker loads
    async fn monitor_worker_loads(
        worker_urls: Vec<String>,
        tx: tokio::sync::watch::Sender<HashMap<String, isize>>,
        interval_secs: u64,
        policy: Arc<dyn LoadBalancingPolicy>,
    ) {
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                error!("Failed to create HTTP client for load monitoring: {}", e);
                return;
            }
        };

        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));

        loop {
            interval.tick().await;

            let mut loads = HashMap::new();
            for url in &worker_urls {
                if let Some(load) = Self::get_worker_load_static(&client, url).await {
                    loads.insert(url.clone(), load);
                    debug!("Worker {} load: {}", url, load);
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
}

use crate::routers::{RouterTrait, WorkerManagement};
use async_trait::async_trait;
use reqwest::Client;

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

#[async_trait(?Send)]
impl RouterTrait for Router {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health(&self, _client: &Client, _req: &HttpRequest) -> HttpResponse {
        // Check local health state of all workers (consistent with PD router)
        // Note: This uses cached health status from background health checks, not live checks
        let mut all_healthy = true;
        let mut unhealthy_servers = Vec::new();

        for worker in self.workers.read().unwrap().iter() {
            if !worker.is_healthy() {
                all_healthy = false;
                unhealthy_servers.push(worker.url().to_string());
            }
        }

        if all_healthy {
            HttpResponse::Ok().body("All servers healthy")
        } else {
            HttpResponse::ServiceUnavailable()
                .body(format!("Unhealthy servers: {:?}", unhealthy_servers))
        }
    }

    async fn health_generate(&self, client: &Client, req: &HttpRequest) -> HttpResponse {
        // Test model generation capability by sending to first available worker
        // Note: This endpoint actually causes the model to generate a token, so we only test one worker
        self.route_to_first(client, "/health_generate", req).await
    }

    async fn get_server_info(&self, client: &Client, req: &HttpRequest) -> HttpResponse {
        self.route_to_first(client, "/get_server_info", req).await
    }

    async fn get_models(&self, client: &Client, req: &HttpRequest) -> HttpResponse {
        self.route_to_first(client, "/v1/models", req).await
    }

    async fn get_model_info(&self, client: &Client, req: &HttpRequest) -> HttpResponse {
        self.route_to_first(client, "/get_model_info", req).await
    }

    async fn route_generate(
        &self,
        client: &Client,
        req: &HttpRequest,
        body: serde_json::Value,
    ) -> HttpResponse {
        // Convert JSON to typed request
        match serde_json::from_value::<crate::openai_api_types::GenerateRequest>(body) {
            Ok(typed_req) => {
                self.route_typed_request(client, req, &typed_req, "/generate")
                    .await
            }
            Err(e) => HttpResponse::BadRequest().body(format!("Invalid request: {}", e)),
        }
    }

    async fn route_chat(
        &self,
        client: &Client,
        req: &HttpRequest,
        body: serde_json::Value,
    ) -> HttpResponse {
        // Convert JSON to typed request
        match serde_json::from_value::<crate::openai_api_types::ChatCompletionRequest>(body) {
            Ok(typed_req) => {
                self.route_typed_request(client, req, &typed_req, "/v1/chat/completions")
                    .await
            }
            Err(e) => HttpResponse::BadRequest().body(format!("Invalid request: {}", e)),
        }
    }

    async fn route_completion(
        &self,
        client: &Client,
        req: &HttpRequest,
        body: serde_json::Value,
    ) -> HttpResponse {
        // Convert JSON to typed request
        match serde_json::from_value::<crate::openai_api_types::CompletionRequest>(body) {
            Ok(typed_req) => {
                self.route_typed_request(client, req, &typed_req, "/v1/completions")
                    .await
            }
            Err(e) => HttpResponse::BadRequest().body(format!("Invalid request: {}", e)),
        }
    }

    async fn flush_cache(&self, client: &Client) -> HttpResponse {
        // Get all worker URLs
        let worker_urls = self.get_worker_urls();

        // Send requests to all workers concurrently without headers
        let mut tasks = Vec::new();
        for worker_url in &worker_urls {
            let request_builder = client.post(format!("{}/flush_cache", worker_url));
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
            HttpResponse::Ok().body("Cache flushed on all servers")
        } else {
            HttpResponse::InternalServerError().body("Cache flush failed on one or more servers")
        }
    }

    async fn get_worker_loads(&self, client: &Client) -> HttpResponse {
        let urls = self.get_worker_urls();
        let mut loads = Vec::new();

        // Get loads from all workers
        for url in &urls {
            let load = self.get_worker_load(client, url).await.unwrap_or(-1);
            loads.push(serde_json::json!({
                "worker": url,
                "load": load
            }));
        }

        HttpResponse::Ok().json(serde_json::json!({
            "workers": loads
        }))
    }

    fn router_type(&self) -> &'static str {
        "regular"
    }

    fn readiness(&self) -> HttpResponse {
        // Regular router is ready if it has at least one healthy worker
        let healthy_count = self
            .workers
            .read()
            .unwrap()
            .iter()
            .filter(|w| w.is_healthy())
            .count();

        if healthy_count > 0 {
            HttpResponse::Ok().json(serde_json::json!({
                "status": "ready",
                "healthy_workers": healthy_count,
                "total_workers": self.workers.read().unwrap().len()
            }))
        } else {
            HttpResponse::ServiceUnavailable().json(serde_json::json!({
                "status": "not_ready",
                "reason": "no healthy workers available",
                "total_workers": self.workers.read().unwrap().len()
            }))
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
            timeout_secs: 5,
            interval_secs: 1,
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
