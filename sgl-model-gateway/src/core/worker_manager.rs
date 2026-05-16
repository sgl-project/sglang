//! Worker Management Module
//!
//! Provides worker lifecycle operations and fan-out request utilities.

use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::response::{IntoResponse, Response};
use futures::{
    future,
    stream::{self, StreamExt},
};
use http::StatusCode;
use serde_json::Value;
use tokio::{
    sync::{watch, Mutex},
    task::JoinHandle,
};
use tracing::{debug, info, warn};

use crate::{
    core::{metrics_aggregator::MetricPack, ConnectionMode, Worker, WorkerRegistry, WorkerType},
    policies::PolicyRegistry,
    protocols::worker_spec::{FlushCacheResult, WorkerLoadInfo, WorkerLoadsResult},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT: usize = 32;

/// Result of a fan-out request to a single worker
struct WorkerResponse {
    url: String,
    result: Result<reqwest::Response, reqwest::Error>,
}

/// Fan out requests to workers in parallel
async fn fan_out(
    workers: &[Arc<dyn Worker>],
    client: &reqwest::Client,
    endpoint: &str,
    method: reqwest::Method,
) -> Vec<WorkerResponse> {
    let futures: Vec<_> = workers
        .iter()
        .map(|worker| {
            let client = client.clone();
            let url = worker.url().to_string();
            let full_url = format!("{}/{}", url, endpoint);
            let api_key = worker.api_key().clone();
            let method = method.clone();

            async move {
                let mut req = client.request(method, &full_url).timeout(REQUEST_TIMEOUT);
                if let Some(key) = api_key {
                    req = req.bearer_auth(key);
                }
                WorkerResponse {
                    url,
                    result: req.send().await,
                }
            }
        })
        .collect();

    stream::iter(futures)
        .buffer_unordered(MAX_CONCURRENT)
        .collect()
        .await
}

/// Parsed `/v1/loads` response for a single engine process.
///
/// Holds both the aggregate (used for DP-blind worker views) and the per-DP-rank
/// num_total_tokens map (used to derive per-rank loads for DPAwareWorker entries).
#[derive(Debug, Clone)]
struct ParsedLoads {
    aggregate_total_tokens: Option<isize>,
    per_rank: HashMap<usize, isize>,
}

impl ParsedLoads {
    /// Resolve the load value for a specific worker.
    /// DP-aware workers get their per-rank num_total_tokens; everyone else
    /// gets the aggregate, falling back to -1 only if the JSON lacked the field.
    fn load_for_worker(&self, worker: &dyn Worker) -> isize {
        if worker.is_dp_aware() {
            if let Some(rank) = worker.dp_rank() {
                if let Some(v) = self.per_rank.get(&rank) {
                    return *v;
                }
            }
        }
        self.aggregate_total_tokens.unwrap_or(-1)
    }
}

pub enum EngineMetricsResult {
    Ok(String),
    Err(String),
}

impl IntoResponse for EngineMetricsResult {
    fn into_response(self) -> Response {
        match self {
            Self::Ok(text) => (StatusCode::OK, text).into_response(),
            Self::Err(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
        }
    }
}

pub struct WorkerManager;

impl WorkerManager {
    pub fn get_worker_urls(registry: &Arc<WorkerRegistry>) -> Vec<String> {
        registry
            .get_all()
            .iter()
            .map(|w| w.url().to_string())
            .collect()
    }

    pub async fn flush_cache_all(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> FlushCacheResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let http_workers: Vec<_> = workers
            .into_iter()
            .filter(|w| matches!(w.connection_mode(), ConnectionMode::Http))
            .collect();

        if http_workers.is_empty() {
            return FlushCacheResult {
                successful: vec![],
                failed: vec![],
                total_workers,
                http_workers: 0,
                message: "No HTTP workers available for cache flush".to_string(),
            };
        }

        info!(
            "Flushing cache on {} HTTP workers (out of {} total)",
            http_workers.len(),
            total_workers
        );

        let responses = fan_out(&http_workers, client, "flush_cache", reqwest::Method::POST).await;

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for resp in responses {
            match resp.result {
                Ok(r) if r.status().is_success() => successful.push(resp.url),
                Ok(r) => failed.push((resp.url, format!("HTTP {}", r.status()))),
                Err(e) => failed.push((resp.url, e.to_string())),
            }
        }

        let message = if failed.is_empty() {
            format!(
                "Successfully flushed cache on all {} HTTP workers",
                successful.len()
            )
        } else {
            format!(
                "Cache flush: {} succeeded, {} failed",
                successful.len(),
                failed.len()
            )
        };

        info!("{}", message);

        FlushCacheResult {
            successful,
            failed,
            total_workers,
            http_workers: http_workers.len(),
            message,
        }
    }

    pub async fn get_all_worker_loads(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        // Group workers by base URL so we issue at most one /v1/loads request per
        // engine process even when N DPAwareWorker entries share the same backend.
        type WorkerGroup = (Option<String>, Vec<Arc<dyn Worker>>);
        let mut groups: HashMap<String, WorkerGroup> = HashMap::new();
        for worker in &workers {
            if !matches!(worker.connection_mode(), ConnectionMode::Http) {
                continue;
            }
            let base = worker.base_url().to_string();
            groups
                .entry(base)
                .or_insert_with(|| (worker.api_key().clone(), Vec::new()))
                .1
                .push(Arc::clone(worker));
        }

        let futures: Vec<_> = groups
            .into_iter()
            .map(|(base_url, (api_key, group_workers))| {
                let client = client.clone();
                async move {
                    let parsed =
                        Self::parse_load_response(&client, &base_url, api_key.as_deref()).await;
                    (group_workers, parsed)
                }
            })
            .collect();

        let group_results = future::join_all(futures).await;

        let mut loads = Vec::with_capacity(total_workers);
        for worker in &workers {
            if !matches!(worker.connection_mode(), ConnectionMode::Http) {
                loads.push(WorkerLoadInfo {
                    worker: worker.url().to_string(),
                    worker_type: Self::worker_type_label(worker.worker_type()),
                    load: -1,
                });
            }
        }
        for (group_workers, parsed) in group_results {
            for worker in group_workers {
                let load = match &parsed {
                    Some(p) => p.load_for_worker(worker.as_ref()),
                    None => -1,
                };
                loads.push(WorkerLoadInfo {
                    worker: worker.url().to_string(),
                    worker_type: Self::worker_type_label(worker.worker_type()),
                    load,
                });
            }
        }

        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }

    fn worker_type_label(worker_type: &WorkerType) -> Option<String> {
        match worker_type {
            WorkerType::Regular => None,
            WorkerType::Prefill { .. } => Some("prefill".to_string()),
            WorkerType::Decode => Some("decode".to_string()),
        }
    }

    /// Fetch and parse /v1/loads for a single engine base URL.
    /// Returns `None` on transport/parse failure so callers can treat all
    /// workers in the group as `load = -1`.
    async fn parse_load_response(
        client: &reqwest::Client,
        base_url: &str,
        api_key: Option<&str>,
    ) -> Option<ParsedLoads> {
        let load_url = format!("{}/v1/loads?include=core", base_url);
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }

        let resp = req.send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let json: Value = resp.json().await.ok()?;

        let aggregate_total_tokens = json
            .get("aggregate")
            .and_then(|a| a.get("total_tokens"))
            .and_then(|v| v.as_i64())
            .map(|n| n as isize);

        let mut per_rank = HashMap::new();
        if let Some(arr) = json.get("loads").and_then(|v| v.as_array()) {
            for entry in arr {
                let rank = entry.get("dp_rank").and_then(|v| v.as_i64());
                let tokens = entry.get("num_total_tokens").and_then(|v| v.as_i64());
                if let (Some(r), Some(t)) = (rank, tokens) {
                    per_rank.insert(r as usize, t as isize);
                }
            }
        }

        Some(ParsedLoads {
            aggregate_total_tokens,
            per_rank,
        })
    }

    pub async fn get_engine_metrics(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> EngineMetricsResult {
        let workers = worker_registry.get_all();

        if workers.is_empty() {
            return EngineMetricsResult::Err("No available workers".to_string());
        }

        let responses = fan_out(&workers, client, "metrics", reqwest::Method::GET).await;

        let mut metric_packs = Vec::new();
        for resp in responses {
            if let Ok(r) = resp.result {
                if r.status().is_success() {
                    if let Ok(text) = r.text().await {
                        metric_packs.push(MetricPack {
                            labels: vec![("worker_addr".into(), resp.url)],
                            metrics_text: text,
                        });
                    }
                }
            }
        }

        if metric_packs.is_empty() {
            return EngineMetricsResult::Err("All backend requests failed".to_string());
        }

        match crate::core::metrics_aggregator::aggregate_metrics(metric_packs) {
            Ok(text) => EngineMetricsResult::Ok(text),
            Err(e) => EngineMetricsResult::Err(format!("Failed to aggregate metrics: {}", e)),
        }
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

    pub async fn stop(&self) {
        let mut handle_guard = self.monitor_handle.lock().await;
        if let Some(handle) = handle_guard.take() {
            info!("Stopping load monitoring");
            handle.abort();
            let _ = handle.await; // Wait for task to finish
        }
    }

    pub fn subscribe(&self) -> watch::Receiver<HashMap<String, isize>> {
        self.rx.clone()
    }

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
            let cache_aware_policies = policy_registry.get_all_cache_aware_policies();

            if power_of_two_policies.is_empty() && cache_aware_policies.is_empty() {
                debug!("No load-aware policies found, skipping load fetch");
                continue;
            }

            let result = WorkerManager::get_all_worker_loads(&worker_registry, &client).await;

            let mut loads = HashMap::new();
            for load_info in result.loads {
                loads.insert(load_info.worker, load_info.load);
            }

            if !loads.is_empty() {
                debug!(
                    "Fetched loads from {} workers, updating {} PoT + {} CacheAware policies",
                    loads.len(),
                    power_of_two_policies.len(),
                    cache_aware_policies.len()
                );
                for policy in &power_of_two_policies {
                    policy.update_loads(&loads);
                }
                for policy in &cache_aware_policies {
                    policy.update_loads(&loads);
                }
                let _ = tx.send(loads);
            } else {
                warn!("No loads fetched from workers");
            }
        }
    }

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
