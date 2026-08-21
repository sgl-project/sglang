// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! KV Indexer fleet status reporting and Router-side freshness tracking.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{Semaphore, SemaphorePermit};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

pub const DEFAULT_STATUS_INTERVAL: Duration = Duration::from_millis(100);
pub const DEFAULT_STATUS_FRESHNESS: Duration = Duration::from_millis(500);

/// One report emitted by an Indexer and consumed by every Router.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexerStatusReport {
    pub indexer_id: String,
    /// Dialable gRPC endpoint advertised to Routers.
    pub endpoint: String,
    pub ready: bool,
    /// Query saturation in [0, 1].
    pub normalized_load: f64,
    pub ready_workers: u32,
    pub total_workers: u32,
}

/// Shared process state used by the gRPC query path and status reporter.
#[derive(Debug)]
pub struct IndexerStatusHandle {
    query_capacity: usize,
    query_semaphore: Semaphore,
    ready: AtomicBool,
    ready_workers: AtomicUsize,
    total_workers: AtomicUsize,
}

impl IndexerStatusHandle {
    pub fn new(query_capacity: usize) -> Self {
        assert!(query_capacity > 0, "query capacity must be positive");
        Self {
            query_capacity,
            query_semaphore: Semaphore::new(query_capacity),
            // Legacy live-only Indexers remain usable before a recovery-aware
            // Bridge configures expected workers.
            ready: AtomicBool::new(true),
            ready_workers: AtomicUsize::new(0),
            total_workers: AtomicUsize::new(0),
        }
    }

    pub(crate) fn try_acquire_query(&self) -> Result<SemaphorePermit<'_>, ()> {
        self.query_semaphore.try_acquire().map_err(|_| ())
    }

    pub fn set_coverage(&self, ready_workers: usize, total_workers: usize) {
        self.ready_workers.store(ready_workers, Ordering::Relaxed);
        self.total_workers.store(total_workers, Ordering::Relaxed);
        self.ready.store(
            total_workers > 0 && ready_workers == total_workers,
            Ordering::Release,
        );
    }

    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Release);
    }

    pub fn report(&self, indexer_id: String, endpoint: String) -> IndexerStatusReport {
        let available = self
            .query_semaphore
            .available_permits()
            .min(self.query_capacity);
        let in_flight = self.query_capacity - available;
        IndexerStatusReport {
            indexer_id,
            endpoint,
            ready: self.ready.load(Ordering::Acquire),
            normalized_load: in_flight as f64 / self.query_capacity as f64,
            ready_workers: self.ready_workers.load(Ordering::Relaxed) as u32,
            total_workers: self.total_workers.load(Ordering::Relaxed) as u32,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexerCandidate {
    pub indexer_id: String,
    pub endpoint: String,
    pub normalized_load: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatusReportError {
    EmptyId,
    EmptyEndpoint,
    InvalidLoad,
    InvalidCoverage,
}

impl std::fmt::Display for StatusReportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyId => f.write_str("indexer_id must not be empty"),
            Self::EmptyEndpoint => f.write_str("endpoint must not be empty"),
            Self::InvalidLoad => f.write_str("normalized_load must be finite and within [0, 1]"),
            Self::InvalidCoverage => f.write_str("ready_workers must not exceed total_workers"),
        }
    }
}

impl std::error::Error for StatusReportError {}

#[derive(Debug)]
struct StatusEntry {
    report: IndexerStatusReport,
    received_at: Instant,
}

/// Router-side registry. Static endpoints are used until the first dynamic
/// report arrives; afterwards only fresh READY reports are eligible.
#[derive(Debug)]
pub struct IndexerStatusRegistry {
    freshness: Duration,
    dynamic_seen: AtomicBool,
    static_candidates: Vec<IndexerCandidate>,
    entries: RwLock<HashMap<String, StatusEntry>>,
}

impl IndexerStatusRegistry {
    pub fn new(static_endpoints: Vec<String>, freshness: Duration) -> Self {
        let static_candidates = static_endpoints
            .into_iter()
            .enumerate()
            .map(|(i, endpoint)| IndexerCandidate {
                indexer_id: format!("static-{i}"),
                endpoint,
                normalized_load: 0.0,
            })
            .collect();
        Self {
            freshness,
            dynamic_seen: AtomicBool::new(false),
            static_candidates,
            entries: RwLock::new(HashMap::new()),
        }
    }

    pub fn record(&self, report: IndexerStatusReport) -> Result<(), StatusReportError> {
        validate_report(&report)?;
        let mut entries = self.entries.write().unwrap_or_else(|e| e.into_inner());
        entries.insert(
            report.indexer_id.clone(),
            StatusEntry {
                report,
                received_at: Instant::now(),
            },
        );
        self.dynamic_seen.store(true, Ordering::Release);
        Ok(())
    }

    pub fn candidates(&self) -> Vec<IndexerCandidate> {
        if !self.dynamic_seen.load(Ordering::Acquire) {
            return self.static_candidates.clone();
        }
        let now = Instant::now();
        let entries = self.entries.read().unwrap_or_else(|e| e.into_inner());
        let mut candidates: Vec<_> = entries
            .values()
            .filter(|entry| {
                entry.report.ready && now.duration_since(entry.received_at) <= self.freshness
            })
            .map(|entry| IndexerCandidate {
                indexer_id: entry.report.indexer_id.clone(),
                endpoint: entry.report.endpoint.clone(),
                normalized_load: entry.report.normalized_load,
            })
            .collect();
        candidates.sort_by(|a, b| {
            a.normalized_load
                .total_cmp(&b.normalized_load)
                .then_with(|| a.indexer_id.cmp(&b.indexer_id))
        });
        candidates
    }
}

fn validate_report(report: &IndexerStatusReport) -> Result<(), StatusReportError> {
    if report.indexer_id.trim().is_empty() {
        return Err(StatusReportError::EmptyId);
    }
    if report.endpoint.trim().is_empty() {
        return Err(StatusReportError::EmptyEndpoint);
    }
    if !report.normalized_load.is_finite() || !(0.0..=1.0).contains(&report.normalized_load) {
        return Err(StatusReportError::InvalidLoad);
    }
    if report.ready_workers > report.total_workers {
        return Err(StatusReportError::InvalidCoverage);
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct StatusReporterConfig {
    pub indexer_id: String,
    pub advertised_endpoint: String,
    pub router_urls: Vec<String>,
    pub interval: Duration,
}

impl StatusReporterConfig {
    pub fn from_env() -> Result<Option<Self>, String> {
        let routers = std::env::var("KV_INDEXER_ROUTER_URLS").unwrap_or_default();
        let router_urls: Vec<String> = routers
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        if router_urls.is_empty() {
            return Ok(None);
        }
        let indexer_id = std::env::var("KV_INDEXER_ID")
            .or_else(|_| std::env::var("HOSTNAME"))
            .unwrap_or_else(|_| "kv-indexer".to_string());
        let advertised_endpoint = std::env::var("KV_INDEXER_ADVERTISE_ENDPOINT").map_err(|_| {
            "KV_INDEXER_ADVERTISE_ENDPOINT is required when status reporting is enabled".to_string()
        })?;
        let interval = match std::env::var("KV_INDEXER_STATUS_INTERVAL_MS") {
            Ok(raw) => {
                let millis = raw.parse::<u64>().map_err(|_| {
                    "KV_INDEXER_STATUS_INTERVAL_MS must be a positive integer".to_string()
                })?;
                if millis == 0 {
                    return Err("KV_INDEXER_STATUS_INTERVAL_MS must be greater than zero".into());
                }
                Duration::from_millis(millis)
            }
            Err(_) => DEFAULT_STATUS_INTERVAL,
        };
        Ok(Some(Self {
            indexer_id,
            advertised_endpoint,
            router_urls,
            interval,
        }))
    }
}

pub fn spawn_status_reporter(
    status: Arc<IndexerStatusHandle>,
    config: StatusReporterConfig,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(config.interval)
            .build()
            .expect("status reporter HTTP client builds");
        let mut ticker = tokio::time::interval(config.interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            let report = status.report(
                config.indexer_id.clone(),
                config.advertised_endpoint.clone(),
            );
            let mut posts = tokio::task::JoinSet::new();
            for router in &config.router_urls {
                let client = client.clone();
                let report = report.clone();
                let router = router.clone();
                let url = format!("{}/v1/kv-indexer/status", router.trim_end_matches('/'));
                posts.spawn(async move {
                    match client.post(&url).json(&report).send().await {
                        Ok(response) if response.status().is_success() => {
                            debug!(router = %router, "reported KV Indexer status");
                        }
                        Ok(response) => {
                            warn!(router = %router, status = %response.status(), "Router rejected KV Indexer status");
                        }
                        Err(error) => {
                            warn!(router = %router, %error, "failed to report KV Indexer status");
                        }
                    }
                });
            }
            while let Some(result) = posts.join_next().await {
                if let Err(error) = result {
                    warn!(%error, "KV Indexer status report task failed");
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::{Json, State};
    use axum::http::StatusCode;
    use axum::routing::post;
    use axum::Router;
    use tokio::sync::mpsc;

    fn report(id: &str, ready: bool, load: f64) -> IndexerStatusReport {
        IndexerStatusReport {
            indexer_id: id.into(),
            endpoint: format!("http://{id}:50051"),
            ready,
            normalized_load: load,
            ready_workers: usize::from(ready) as u32,
            total_workers: 1,
        }
    }

    #[test]
    fn dynamic_reports_replace_static_fallback_and_sort_by_load() {
        let registry =
            IndexerStatusRegistry::new(vec!["http://static:50051".into()], Duration::from_secs(1));
        assert_eq!(registry.candidates()[0].endpoint, "http://static:50051");
        registry.record(report("busy", true, 0.8)).unwrap();
        registry.record(report("idle", true, 0.2)).unwrap();
        registry.record(report("syncing", false, 0.0)).unwrap();
        let ids: Vec<_> = registry
            .candidates()
            .into_iter()
            .map(|c| c.indexer_id)
            .collect();
        assert_eq!(ids, ["idle", "busy"]);
    }

    #[tokio::test]
    async fn stale_and_not_ready_replicas_are_excluded() {
        let registry = IndexerStatusRegistry::new(Vec::new(), Duration::from_millis(20));
        registry.record(report("ready", true, 0.1)).unwrap();
        registry.record(report("syncing", false, 0.0)).unwrap();
        assert_eq!(registry.candidates()[0].indexer_id, "ready");
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(registry.candidates().is_empty());
    }

    #[test]
    fn status_handle_reports_normalized_query_load_and_coverage() {
        let status = IndexerStatusHandle::new(2);
        status.set_coverage(1, 2);
        let permit = status.try_acquire_query().unwrap();
        let report = status.report("i".into(), "http://i:1".into());
        assert!(!report.ready);
        assert_eq!(report.normalized_load, 0.5);
        drop(permit);
    }

    async fn capture_status(
        State(tx): State<mpsc::Sender<IndexerStatusReport>>,
        Json(report): Json<IndexerStatusReport>,
    ) -> StatusCode {
        tx.send(report).await.unwrap();
        StatusCode::NO_CONTENT
    }

    #[tokio::test]
    async fn reporter_posts_at_configured_interval() {
        let (tx, mut rx) = mpsc::channel(4);
        let app = Router::new()
            .route("/v1/kv-indexer/status", post(capture_status))
            .with_state(tx);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let status = Arc::new(IndexerStatusHandle::new(4));
        status.set_coverage(1, 1);
        let reporter = spawn_status_reporter(
            status,
            StatusReporterConfig {
                indexer_id: "i1".into(),
                advertised_endpoint: "http://i1:50051".into(),
                router_urls: vec![format!("http://{addr}")],
                interval: Duration::from_millis(20),
            },
        );
        for _ in 0..2 {
            let report = tokio::time::timeout(Duration::from_millis(250), rx.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(report.indexer_id, "i1");
            assert!(report.ready);
        }
        reporter.abort();
        server.abort();
    }
}
