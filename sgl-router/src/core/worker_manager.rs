//! Unified Worker Management Module
//!
//! Handles all aspects of worker lifecycle including discovery, initialization,
//! runtime management, and health monitoring.

use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::response::{IntoResponse, Response};
use futures::future;
use http::{Method, StatusCode};
use serde_json::Value;
use tokio::{
    sync::{watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, error, info, warn};

use crate::{
    core::{metrics_aggregator::MetricPack, ConnectionMode, WorkerRegistry, WorkerType},
    policies::PolicyRegistry,
    protocols::worker_spec::{FlushCacheResult, WorkerLoadInfo, WorkerLoadsResult},
};

/// Unified worker management
pub struct WorkerManager;

impl WorkerManager {
    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    /// Flush cache on all workers
    ///
    /// Sends a POST request to /flush_cache endpoint on all HTTP workers.
    /// Returns detailed results showing which workers succeeded and which failed.
    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> Result<FlushCacheResult, String> {
        warn!("Flushing cache for ALL workers - this may impact performance temporarily");

        let workers = worker_registry.get_all();

        let http_workers: Vec<_> = workers
            .iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return Ok(FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers: workers.len(),
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            });
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total workers)",
            http_workers.len(),
            workers.len()
        );

        let mut tasks = Vec::new();
        for worker in &http_workers {
            let url = worker.url().to_string();
            let flush_url = format!("{}/flush_cache", url);
            let mut request = client.post(&flush_url);

            if let Some(api_key) = worker.api_key() {
                request = request.header("Authorization", format!("Bearer {}", api_key));
            }

            let worker_url = url.clone();
            tasks.push(async move {
                let result = request.send().await;
                (worker_url, result)
            });
        }

        let results = future::join_all(tasks).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for (url, result) in results {
            match result {
                Ok(response) if response.status().is_success() => {
                    debug!("Successfully flushed cache on worker: {}", url);
                    successful.push(url);
                }
                Ok(response) => {
                    let error = format!("HTTP {}", response.status());
                    warn!("Failed to flush cache on worker {}: {}", url, error);
                    failed.push((url, error));
                }
                Err(e) => {
                    let error = e.to_string();
                    error!("Failed to connect to worker {}: {}", url, error);
                    failed.push((url, error));
                }
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush completed: {} succeeded, {} failed (out of {} HTTP workers)",
                successful.len(),
                failed.len(),
                http_workers.len()
            )
        };

        info!("{}", message);

        Ok(FlushCacheResult {
            successful,
            failed,
            total_workers: workers.len(),
            http_workers: http_workers.len(),
            message,
        })
    }
    pub async fn get_worker_load(
        url: &str,
        api_key: Option<&str>,
        client: &reqwest::Client,
    ) -> Option<HashMap<isize, isize>> {
        let load_url = format!("{}/get_load", url);
        let mut request = client.get(&load_url);

        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }

        match request.send().await {
            Ok(response) if response.status().is_success() => {
                match response.json::<Value>().await {
                    Ok(json) => {
                        // The /get_load endpoint returns an array of load info objects (one per DP rank)
                        // Each object has: {dp_rank, num_reqs, num_waiting_reqs, num_tokens}
                        if let Some(array) = json.as_array() {
                            let mut rank_tokens = HashMap::new();
                            for entry in array {
                                let dp_rank = entry
                                    .get("dp_rank")
                                    .and_then(|v| v.as_i64())
                                    .map(|rank| rank as isize);
                                let num_tokens = entry
                                    .get("num_tokens")
                                    .and_then(|v| v.as_i64())
                                    .map(|rank| rank as isize);
                                if let (Some(rank), Some(tokens)) = (dp_rank, num_tokens) {
                                    rank_tokens.insert(rank, tokens);
                                }
                            }
                            Some(rank_tokens)
                        } else {
                            warn!(
                                "Invalid load response from {}: expected array, got {:?}",
                                url, json
                            );
                            None
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse load response from {}: {}", url, e);
                        None
                    }
                }
            }
            Ok(response) => {
                warn!(
                    "Failed to get load from {}: HTTP {}",
                    url,
                    response.status()
                );
                None
            }
            Err(e) => {
                warn!("Failed to connect to {} for load check: {}", url, e);
                None
            }
        }
    }

    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        // Prepare tasks for parallel execution
        let mut tasks = Vec::new();
        for worker in &workers {
            let url = worker.url().to_string();
            let api_key = worker.api_key().clone();
            let worker_type = match worker.worker_type() {
                WorkerType::Regular => None,
                WorkerType::Prefill { .. } => Some("prefill".to_string()),
                WorkerType::Decode => Some("decode".to_string()),
            };
            let is_http = matches!(worker.connection_mode(), ConnectionMode::Http);
            let client = client.clone();

            tasks.push(async move {
                let dp_rank_loads = if is_http {
                    Self::get_worker_load(&url, api_key.as_deref(), &client)
                        .await
                        .unwrap_or(HashMap::new())
                } else {
                    HashMap::new()
                };
                let load = dp_rank_loads.values().sum();

                WorkerLoadInfo {
                    worker: url,
                    worker_type,
                    load,
                    dp_rank_loads,
                }
            });
        }

        let loads = future::join_all(tasks).await;

        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }

    pub async fn get_engine_metrics(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> Response {
        let engine_responses =
            match Self::fan_out_simple_request(worker_registry, client, "metrics", Method::GET)
                .await
            {
                Ok(x) => x,
                Err(e) => return e,
            };
        let engine_responses = engine_responses
            .into_iter()
            .map(|(worker_base_url, metrics_text)| MetricPack {
                labels: vec![("worker_addr".into(), worker_base_url)],
                metrics_text,
            })
            .collect();
        let text = match crate::core::metrics_aggregator::aggregate_metrics(engine_responses) {
            Ok(x) => x,
            Err(e) => {
                let error_msg = format!("Failed to aggregate metrics: {}", e);
                return (StatusCode::INTERNAL_SERVER_ERROR, error_msg).into_response();
            }
        };
        (StatusCode::OK, text).into_response()
    }

    async fn fan_out_simple_request(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
        endpoint: &str,
        method: Method,
    ) -> Result<Vec<(String, String)>, Response> {
        let workers = worker_registry.get_all();
        if workers.is_empty() {
            return Err((StatusCode::SERVICE_UNAVAILABLE, "No available workers").into_response());
        }

        let mut responses = vec![];
        // May do parallel requests later
        for worker in workers {
            let worker_url = worker.url().to_string();

            let url = format!("{}/{}", worker_url, endpoint);
            let mut request_builder = match method {
                Method::GET => client.get(url),
                Method::POST => client.post(url),
                _ => {
                    return Err((
                        StatusCode::METHOD_NOT_ALLOWED,
                        "Unsupported method for simple routing",
                    )
                        .into_response())
                }
            };

            if let Some(api_key) = worker.api_key() {
                request_builder =
                    request_builder.header("Authorization", format!("Bearer {}", api_key));
            }

            match request_builder.send().await {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    match res.text().await {
                        Ok(body_text) => {
                            if status.is_success() {
                                responses.push((worker_url, body_text));
                            }
                        }
                        Err(e) => {
                            warn!("fan_out_simple_request failed when reading text: {}", e)
                        }
                    }
                }
                Err(e) => warn!("fan_out_simple_request failed when sending: {}", e),
            }
        }

        Ok(responses)
    }
}

/// Load monitoring service that periodically fetches worker loads
pub struct LoadMonitor {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: reqwest::Client,
    interval: Duration,
    tx: watch::Sender<HashMap<String, isize>>,
    rx: watch::Receiver<HashMap<String, isize>>,
    monitor_handle: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl LoadMonitor {
    /// Create a new load monitor
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval_secs: u64,
    ) -> Self {
        let (tx, rx) = watch::channel(HashMap::new());

        Self {
            worker_registry,
            policy_registry,
            client,
            interval: Duration::from_secs(interval_secs),
            tx,
            rx,
            monitor_handle: Arc::new(Mutex::new(None)),
        }
    }

    /// Start monitoring worker loads
    pub async fn start(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if handle_guard.is_some() {
            debug!("Load monitoring already running");
            return;
        }

        info!(
            "Starting load monitoring with interval: {:?}",
            self.interval
        );

        let worker_registry = Arc::clone(&self.worker_registry);
        let policy_registry = Arc::clone(&self.policy_registry);
        let client = self.client.clone();
        let interval = self.interval;
        let tx = self.tx.clone();

        let handle = tokio::spawn(async move {
            Self::monitor_loop(worker_registry, policy_registry, client, interval, tx).await;
        });

        *handle_guard = Some(handle);
    }

    /// Stop monitoring worker loads
    pub async fn stop(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            info!("Stopping load monitoring");
            handle.abort();
            let _ = handle.await; // Wait for task to finish
        }
    }

    /// Get a receiver for load updates
    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, isize>> {
        self.rx.clone()
    }

    /// The main monitoring loop
    async fn monitor_loop(
        worker_registry: Arc<WorkerRegistry>,
        policy_registry: Arc<PolicyRegistry>,
        client: reqwest::Client,
        interval: Duration,
        tx: watch::Sender<HashMap<String, isize>>,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        loop {
            interval_timer.tick().await;
            let power_of_two_policies = policy_registry.get_all_power_of_two_policies();

            if power_of_two_policies.is_empty()
                && !policy_registry.is_dp_minimum_tokens_scheduler_enabled()
            {
                debug!("No PowerOfTwo policies found, skipping load fetch");
                continue;
            }

            let all_policies = policy_registry.get_all_policies();
            let result = WorkerManager::get_all_worker_loads(&worker_registry, &client).await;

            let mut loads = HashMap::new();
            let mut dp_rank_loads = HashMap::new();
            for load_info in result.loads {
                loads.insert(load_info.worker.clone(), load_info.load);
                dp_rank_loads.insert(load_info.worker, load_info.dp_rank_loads);
            }

            if !loads.is_empty() {
                debug!(
                    "Fetched loads from {} workers, updating {} PowerOfTwo policies",
                    loads.len(),
                    power_of_two_policies.len()
                );
                for policy in &power_of_two_policies {
                    policy.update_loads(&loads);
                }
                for policy in &all_policies {
                    policy.update_dp_loads(&dp_rank_loads)
                }
                let _ = tx.send(loads);
            } else {
                warn!("No loads fetched from workers");
            }
        }
    }

    /// Check if monitoring is currently active
    pub async fn is_running(&self) -> bool {
        let handle_guard = self.monitor_handle.lock().await;
        handle_guard.is_some()
    }
}

impl Drop for LoadMonitor {
    fn drop(&mut self) {
        if let Ok(mut handle_guard) = self.monitor_handle.try_lock() {
            if let Some(handle) = handle_guard.take() {
                handle.abort();
            }
        }
    }
}
