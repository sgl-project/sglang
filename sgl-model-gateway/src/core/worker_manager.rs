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

#[derive(Debug, Default)]
struct ParsedLoadResponse {
    aggregate: Option<isize>,
    aggregate_from_ranks: Option<isize>,
    by_dp_rank: HashMap<usize, isize>,
}

type DpHttpGroupKey = (String, Option<String>);

#[derive(Debug)]
struct DpWorkerLoadTarget {
    url: String,
    worker_type: Option<String>,
    dp_rank: usize,
}

impl ParsedLoadResponse {
    fn from_json(json: &Value) -> Self {
        fn non_negative_isize(value: Option<&Value>) -> Option<isize> {
            value
                .and_then(Value::as_u64)
                .and_then(|value| isize::try_from(value).ok())
        }

        let aggregate = non_negative_isize(
            json.get("aggregate")
                .and_then(|aggregate| aggregate.get("total_tokens")),
        );
        let mut by_dp_rank = HashMap::new();
        let mut aggregate_from_ranks = None;

        if let Some(loads) = json.get("loads").and_then(Value::as_array) {
            let mut total = Some(0isize);
            let mut saw_valid_rank = false;

            for load in loads {
                let parsed = load
                    .get("dp_rank")
                    .and_then(Value::as_u64)
                    .and_then(|rank| usize::try_from(rank).ok())
                    .zip(non_negative_isize(load.get("num_total_tokens")));
                let Some((dp_rank, total_tokens)) = parsed else {
                    // A partial sum would make a malformed response look less
                    // loaded than it is. Keep valid rank entries for DP-aware
                    // routing, but do not expose an aggregate for this payload.
                    total = None;
                    continue;
                };

                saw_valid_rank = true;
                if by_dp_rank.insert(dp_rank, total_tokens).is_some() {
                    total = None;
                } else if let Some(current) = total {
                    total = current.checked_add(total_tokens);
                }
            }

            if saw_valid_rank {
                aggregate_from_ranks = total;
            }
        }

        Self {
            aggregate,
            aggregate_from_ranks,
            by_dp_rank,
        }
    }

    fn aggregate_load(&self) -> Option<isize> {
        self.aggregate.or(self.aggregate_from_ranks)
    }
}

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
        let mut loads = Vec::new();
        let mut direct_http_workers = Vec::new();
        let mut dp_http_groups: HashMap<DpHttpGroupKey, Vec<DpWorkerLoadTarget>> = HashMap::new();

        for worker in workers {
            let url = worker.url().to_string();
            let worker_type = match worker.worker_type() {
                WorkerType::Regular => None,
                WorkerType::Prefill { .. } => Some("prefill".to_string()),
                WorkerType::Decode => Some("decode".to_string()),
            };

            if !matches!(worker.connection_mode(), ConnectionMode::Http) {
                loads.push(WorkerLoadInfo {
                    worker: url,
                    worker_type,
                    load: -1,
                });
                continue;
            }

            let api_key = worker.api_key().clone();
            if let Some(dp_rank) = worker.dp_rank() {
                let base_url = worker.base_url().trim_end_matches('/').to_string();
                dp_http_groups
                    .entry((base_url, api_key))
                    .or_default()
                    .push(DpWorkerLoadTarget {
                        url,
                        worker_type,
                        dp_rank,
                    });
            } else {
                direct_http_workers.push((url, worker_type, api_key));
            }
        }

        let direct_futures = direct_http_workers
            .into_iter()
            .map(|(url, worker_type, api_key)| {
                let client = client.clone();
                async move {
                    let load = Self::fetch_load_response(&client, &url, api_key.as_deref())
                        .await
                        .and_then(|response| response.aggregate_load())
                        .unwrap_or(-1);
                    WorkerLoadInfo {
                        worker: url,
                        worker_type,
                        load,
                    }
                }
            });

        let dp_futures = dp_http_groups
            .into_iter()
            .map(|((base_url, api_key), workers)| {
                let client = client.clone();
                async move {
                    let by_dp_rank =
                        Self::fetch_load_response(&client, &base_url, api_key.as_deref())
                            .await
                            .map(|response| response.by_dp_rank)
                            .unwrap_or_default();
                    workers
                        .into_iter()
                        .map(|target| WorkerLoadInfo {
                            worker: target.url,
                            worker_type: target.worker_type,
                            load: by_dp_rank.get(&target.dp_rank).copied().unwrap_or(-1),
                        })
                        .collect::<Vec<_>>()
                }
            });

        let (direct_loads, dp_load_groups) = future::join(
            future::join_all(direct_futures),
            future::join_all(dp_futures),
        )
        .await;
        loads.extend(direct_loads);
        loads.extend(dp_load_groups.into_iter().flatten());
        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }

    async fn fetch_load_response(
        client: &reqwest::Client,
        url: &str,
        api_key: Option<&str>,
    ) -> Option<ParsedLoadResponse> {
        let load_url = format!("{}/v1/loads?include=core", url.trim_end_matches('/'));
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }

        match req.send().await {
            Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                Ok(json) => Some(ParsedLoadResponse::from_json(&json)),
                _ => None,
            },
            _ => None,
        }
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

            if power_of_two_policies.is_empty() {
                debug!("No PowerOfTwo policies found, skipping load fetch");
                continue;
            }

            let result = WorkerManager::get_all_worker_loads(&worker_registry, &client).await;

            let mut loads = HashMap::new();
            for load_info in result.loads {
                if load_info.load >= 0 {
                    loads.insert(load_info.worker, load_info.load);
                }
            }

            if !loads.is_empty() {
                debug!(
                    "Fetched loads from {} workers, updating {} PowerOfTwo policies",
                    loads.len(),
                    power_of_two_policies.len()
                );
            } else {
                warn!("No valid loads fetched from workers; clearing cached token loads");
            }
            for policy in &power_of_two_policies {
                policy.update_loads(&loads);
            }
            let _ = tx.send(loads);
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

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use axum::{extract::State, routing::get, Json, Router};
    use serde_json::json;
    use tokio::{net::TcpListener, task::JoinHandle};

    use super::*;
    use crate::core::{BasicWorkerBuilder, DPAwareWorkerBuilder};

    type MockLoadState = (Arc<AtomicUsize>, Arc<Value>);

    async fn mock_loads(State((calls, response)): State<MockLoadState>) -> Json<Value> {
        calls.fetch_add(1, Ordering::SeqCst);
        Json((*response).clone())
    }

    async fn start_load_server(response: Value) -> (String, Arc<AtomicUsize>, JoinHandle<()>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let app = Router::new()
            .route("/v1/loads", get(mock_loads))
            .with_state((Arc::clone(&calls), Arc::new(response)));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), calls, server)
    }

    #[test]
    fn test_parse_load_response_supports_rank_loads_and_legacy_aggregate() {
        let parsed = ParsedLoadResponse::from_json(&json!({
            "aggregate": {"total_tokens": 321},
            "loads": [
                {"dp_rank": 0, "num_total_tokens": 100},
                {"dp_rank": 1, "num_total_tokens": 221},
                {"dp_rank": 2, "num_total_tokens": -1},
                {"dp_rank": "bad", "num_total_tokens": 9}
            ]
        }));

        assert_eq!(parsed.aggregate_load(), Some(321));
        assert_eq!(parsed.by_dp_rank.get(&0), Some(&100));
        assert_eq!(parsed.by_dp_rank.get(&1), Some(&221));
        assert!(!parsed.by_dp_rank.contains_key(&2));
        assert_eq!(parsed.by_dp_rank.len(), 2);
    }

    #[test]
    fn test_rank_aggregate_requires_a_complete_valid_payload() {
        let complete = ParsedLoadResponse::from_json(&json!({
            "loads": [
                {"dp_rank": 0, "num_total_tokens": 100},
                {"dp_rank": 1, "num_total_tokens": 221}
            ]
        }));
        assert_eq!(complete.aggregate_load(), Some(321));

        let partial = ParsedLoadResponse::from_json(&json!({
            "loads": [
                {"dp_rank": 0, "num_total_tokens": 100},
                {"dp_rank": 1, "num_total_tokens": -1}
            ]
        }));
        assert_eq!(partial.aggregate_load(), None);
        assert_eq!(partial.by_dp_rank.get(&0), Some(&100));
        assert!(!partial.by_dp_rank.contains_key(&1));
    }

    #[tokio::test]
    async fn test_dp_workers_fetch_physical_loads_once_and_map_by_rank() {
        let (base_url, calls, server) = start_load_server(json!({
            "loads": [
                {"dp_rank": 0, "num_total_tokens": 900},
                {"dp_rank": 1, "num_total_tokens": 120}
            ]
        }))
        .await;
        let registry = WorkerRegistry::new();
        for rank in 0..3 {
            let worker = DPAwareWorkerBuilder::new(&base_url, rank, 3)
                .worker_type(WorkerType::Decode)
                .build();
            registry.register(Arc::new(worker));
        }

        let result = WorkerManager::get_all_worker_loads(&registry, &reqwest::Client::new()).await;
        let loads: HashMap<_, _> = result
            .loads
            .into_iter()
            .map(|load| (load.worker, load.load))
            .collect();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(loads.get(&format!("{base_url}@0")), Some(&900));
        assert_eq!(loads.get(&format!("{base_url}@1")), Some(&120));
        assert_eq!(loads.get(&format!("{base_url}@2")), Some(&-1));
        assert_eq!(result.total_workers, 3);
        assert_eq!(result.successful, 2);
        assert_eq!(result.failed, 1);
        server.abort();
    }

    #[tokio::test]
    async fn test_non_dp_worker_keeps_legacy_aggregate_load() {
        let (base_url, calls, server) =
            start_load_server(json!({"aggregate": {"total_tokens": 777}})).await;
        let registry = WorkerRegistry::new();
        registry.register(Arc::new(
            BasicWorkerBuilder::new(&base_url)
                .worker_type(WorkerType::Decode)
                .build(),
        ));

        let result = WorkerManager::get_all_worker_loads(&registry, &reqwest::Client::new()).await;

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.loads.len(), 1);
        assert_eq!(result.loads[0].worker, base_url);
        assert_eq!(result.loads[0].load, 777);
        assert_eq!(result.successful, 1);
        assert_eq!(result.failed, 0);
        server.abort();
    }

    #[tokio::test]
    async fn test_non_dp_worker_sums_current_per_rank_loads() {
        let (base_url, calls, server) = start_load_server(json!({
            "loads": [
                {"dp_rank": 0, "num_total_tokens": 300},
                {"dp_rank": 1, "num_total_tokens": 477}
            ]
        }))
        .await;
        let registry = WorkerRegistry::new();
        registry.register(Arc::new(
            BasicWorkerBuilder::new(&base_url)
                .worker_type(WorkerType::Decode)
                .build(),
        ));

        let result = WorkerManager::get_all_worker_loads(&registry, &reqwest::Client::new()).await;

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.loads.len(), 1);
        assert_eq!(result.loads[0].load, 777);
        assert_eq!(result.successful, 1);
        assert_eq!(result.failed, 0);
        server.abort();
    }
}
