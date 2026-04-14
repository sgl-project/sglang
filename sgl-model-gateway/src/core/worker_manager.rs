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

/// 带队列深度信息的worker负载数据
pub struct WorkerLoadWithQueue {
    pub worker: String,
    pub worker_type: Option<String>,
    pub load: isize,
    /// 等待队列中的请求数，-1表示获取失败
    pub waiting_reqs: isize,
}

/// 带队列深度信息的worker负载查询结果
pub struct WorkerLoadsWithQueueResult {
    pub loads: Vec<WorkerLoadWithQueue>,
    pub total_workers: usize,
    pub successful: usize,
    pub failed: usize,
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

        let futures: Vec<_> = workers
            .iter()
            .map(|worker| {
                let url = worker.url().to_string();
                let api_key = worker.api_key().clone();
                let worker_type = match worker.worker_type() {
                    WorkerType::Regular => None,
                    WorkerType::Prefill { .. } => Some("prefill".to_string()),
                    WorkerType::Decode => Some("decode".to_string()),
                };
                let is_http = matches!(worker.connection_mode(), ConnectionMode::Http);
                let client = client.clone();

                async move {
                    let load = if is_http {
                        Self::parse_load_response(&client, &url, api_key.as_deref()).await
                    } else {
                        -1
                    };
                    WorkerLoadInfo {
                        worker: url,
                        worker_type,
                        load,
                    }
                }
            })
            .collect();

        let loads = future::join_all(futures).await;
        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsResult {
            loads,
            total_workers,
            successful,
            failed,
        }
    }

    async fn parse_load_response(
        client: &reqwest::Client,
        url: &str,
        api_key: Option<&str>,
    ) -> isize {
        let load_url = format!("{}/get_load", url);
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }

        match req.send().await {
            Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                Ok(json) if json.is_array() => json
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(|e| e.get("num_tokens").and_then(|v| v.as_i64()))
                    .sum::<i64>() as isize,
                _ => -1,
            },
            _ => -1,
        }
    }

    /// 从 /get_load 端点同时解析token负载和等待队列请求数
    async fn parse_load_response_with_queue(
        client: &reqwest::Client,
        url: &str,
        api_key: Option<&str>,
    ) -> (isize, isize) {
        let load_url = format!("{}/get_load", url);
        let mut req = client.get(&load_url).timeout(REQUEST_TIMEOUT);
        if let Some(key) = api_key {
            req = req.bearer_auth(key);
        }

        match req.send().await {
            Ok(r) if r.status().is_success() => match r.json::<Value>().await {
                Ok(json) if json.is_array() => {
                    let arr = json.as_array().unwrap();
                    let load = arr
                        .iter()
                        .filter_map(|e| e.get("num_tokens").and_then(|v| v.as_i64()))
                        .sum::<i64>() as isize;
                    let waiting = arr
                        .iter()
                        .filter_map(|e| e.get("num_waiting_reqs").and_then(|v| v.as_i64()))
                        .sum::<i64>() as isize;
                    (load, waiting)
                }
                _ => (-1, -1),
            },
            _ => (-1, -1),
        }
    }

    /// 获取所有worker负载，包含队列深度信息
    pub async fn get_all_worker_loads_with_queue(
        worker_registry: &WorkerRegistry,
        client: &reqwest::Client,
    ) -> WorkerLoadsWithQueueResult {
        let workers = worker_registry.get_all();
        let total_workers = workers.len();

        let futures: Vec<_> = workers
            .iter()
            .map(|worker| {
                let url = worker.url().to_string();
                let api_key = worker.api_key().clone();
                let worker_type = match worker.worker_type() {
                    WorkerType::Regular => None,
                    WorkerType::Prefill { .. } => Some("prefill".to_string()),
                    WorkerType::Decode => Some("decode".to_string()),
                };
                let is_http = matches!(worker.connection_mode(), ConnectionMode::Http);
                let client = client.clone();

                async move {
                    let (load, waiting_reqs) = if is_http {
                        Self::parse_load_response_with_queue(&client, &url, api_key.as_deref())
                            .await
                    } else {
                        (-1, -1)
                    };
                    WorkerLoadWithQueue {
                        worker: url,
                        worker_type,
                        load,
                        waiting_reqs,
                    }
                }
            })
            .collect();

        let loads = future::join_all(futures).await;
        let successful = loads.iter().filter(|l| l.load >= 0).count();
        let failed = loads.iter().filter(|l| l.load < 0).count();

        WorkerLoadsWithQueueResult {
            loads,
            total_workers,
            successful,
            failed,
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
    /// 每个worker的等待队列请求数（用于队列感知路由）
    queue_depth_tx: watch::Sender<HashMap<String, usize>>,
    queue_depth_rx: watch::Receiver<HashMap<String, usize>>,
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
        let (queue_depth_tx, queue_depth_rx) = watch::channel(HashMap::new());

        Self {
            worker_registry,
            policy_registry,
            client,
            interval: Duration::from_secs(interval_secs),
            tx,
            rx,
            queue_depth_tx,
            queue_depth_rx,
            monitor_handle: Arc::new(Mutex::new(None)),
        }
    }

    /// 订阅worker等待队列深度数据
    pub fn subscribe_queue_depth(&self) -> watch::Receiver<HashMap<String, usize>> {
        self.queue_depth_rx.clone()
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
        let queue_depth_tx = self.queue_depth_tx.clone();

        let handle = tokio::spawn(async move {
            Self::monitor_loop(
                worker_registry,
                policy_registry,
                client,
                interval,
                tx,
                queue_depth_tx,
            )
            .await;
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
        queue_depth_tx: watch::Sender<HashMap<String, usize>>,
    ) {
        let mut interval_timer = tokio::time::interval(interval);

        loop {
            interval_timer.tick().await;

            let power_of_two_policies = policy_registry.get_all_power_of_two_policies();

            // 即使没有PowerOfTwo策略，也需要获取队列深度用于队列感知路由
            let result =
                WorkerManager::get_all_worker_loads_with_queue(&worker_registry, &client).await;

            let mut loads = HashMap::new();
            let mut queue_depths = HashMap::new();
            for load_info in &result.loads {
                loads.insert(load_info.worker.clone(), load_info.load);
                if load_info.waiting_reqs >= 0 {
                    queue_depths
                        .insert(load_info.worker.clone(), load_info.waiting_reqs as usize);
                }
            }

            if !loads.is_empty() {
                if !power_of_two_policies.is_empty() {
                    debug!(
                        "Fetched loads from {} workers, updating {} PowerOfTwo policies",
                        loads.len(),
                        power_of_two_policies.len()
                    );
                    for policy in &power_of_two_policies {
                        policy.update_loads(&loads);
                    }
                }
                let _ = tx.send(loads);
            } else {
                warn!("No loads fetched from workers");
            }

            if !queue_depths.is_empty() {
                debug!(
                    "Fetched queue depths from {} workers: {:?}",
                    queue_depths.len(),
                    queue_depths
                );
                let _ = queue_depth_tx.send(queue_depths);
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
