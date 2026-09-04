// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Load Monitor — the pull-mode feeder of [`EngineLoadTable`].
//!
//! Engine-reported load lives in ONE store, [`EngineLoadTable`], with two
//! feeders: the KV-events subscriber (engines that publish `LoadStat` on
//! their load socket) and this module, which polls `GET
//! /v1/loads?include=core` for engines that don't, or as a belt-and-braces
//! second source. Both write per-`(worker_url, dp_rank)` rows stamped with
//! the router-local receive time, so they mix freely (last write wins) and
//! age under the same freshness window, which this module aligns with
//! `--load-monitor-stale-after-ms`. Every load-aware policy then reads the
//! same [`crate::policies::engine_load::WorkerLoads`] view: engine depth
//! plus the slots this router acquired since that report.
//!
//! Pull mechanics (ported from AIgate's `internal/loadmonitor`):
//!
//! * **Trigger coalescing.** Every routed request calls [`LoadMonitor::trigger`]
//!   for the model's workers. The wake channel has capacity one, so under
//!   load a worker is refreshed as fast as one HTTP round-trip allows — at
//!   most one pull in flight plus one pending — never once per request.
//! * **Periodic fallback.** After each pull completes the task re-arms a
//!   `report_interval` timer, so an idle worker keeps a fresh snapshot and
//!   never gets stuck `stale` (which would make it permanently unroutable).
//! * **Freshness.** [`EngineLoadTable::worker_freshness`]: a worker is
//!   routable under the freshness gate only when every rank's latest gauge
//!   is younger than `stale_after`; a failed pull reads `unreachable`, an
//!   early-boot zero-capacity snapshot reads `stale`.

use crate::policies::engine_load::{
    EngineLoadTable, Freshness, LoadStat, PullStatus, WorkerAggregate,
};
use crate::server::metrics::{LoadPullOutcome, MetricsRegistry, ReportedLoadKind};
use dashmap::DashMap;
use serde::Deserialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// Cap on one `/v1/loads` body so a misbehaving engine cannot exhaust
/// router memory while polled.
const MAX_LOADS_RESPONSE_BYTES: usize = 4 << 20;

/// Tunables for the pull loop. Built from [`crate::config::LoadMonitorConfig`].
#[derive(Debug, Clone, Copy)]
pub struct LoadMonitorConfig {
    /// Periodic fallback pull interval, measured from the completion of the
    /// previous pull.
    pub report_interval: Duration,
    /// A report older than this is `stale`; also becomes the shared
    /// [`EngineLoadTable`] freshness window. Must exceed `report_interval`
    /// or idle workers flap between fresh and stale.
    pub stale_after: Duration,
    /// Per-pull HTTP timeout.
    pub request_timeout: Duration,
}

impl Default for LoadMonitorConfig {
    fn default() -> Self {
        Self {
            report_interval: Duration::from_millis(1000),
            stale_after: Duration::from_millis(3000),
            request_timeout: Duration::from_millis(1000),
        }
    }
}

/// Read-only diagnostic view of one worker's load as the table holds it.
#[derive(Debug, Clone)]
pub struct EndpointLoad {
    pub freshness: Freshness,
    /// Present once any rank has reported.
    pub aggregate: Option<WorkerAggregate>,
    /// Router-local pull sequence for this worker (pulls that actually ran).
    pub sequence: u64,
    /// Oldest rank timestamp of the latest report.
    pub received_at: Option<Instant>,
    pub last_error: Option<String>,
}

struct Endpoint {
    url: String,
    /// Capacity-one wake channel: at most one pending refresh request.
    wake_tx: mpsc::Sender<()>,
    sequence: AtomicU64,
    cancel: CancellationToken,
}

impl Endpoint {
    /// Retain one pending refresh without waiting. Further triggers while
    /// the slot is occupied coalesce.
    fn trigger(&self) {
        let _ = self.wake_tx.try_send(());
    }
}

pub struct LoadMonitor {
    cfg: LoadMonitorConfig,
    client: reqwest::Client,
    endpoints: DashMap<String, Arc<Endpoint>>,
    metrics: OnceLock<Arc<MetricsRegistry>>,
    drained: CancellationToken,
    table: Arc<EngineLoadTable>,
}

impl std::fmt::Debug for LoadMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadMonitor")
            .field("cfg", &self.cfg)
            .field("endpoints", &self.endpoints.len())
            .finish()
    }
}

impl LoadMonitor {
    /// Build a monitor that writes into `table`. The table's freshness
    /// window is set to `cfg.stale_after` so pushed and pulled reports age
    /// under one rule.
    pub fn new(cfg: LoadMonitorConfig, table: Arc<EngineLoadTable>) -> Arc<Self> {
        table.set_freshness(cfg.stale_after);
        // One connection per worker is plenty: pulls to the same worker are
        // serialised by the per-endpoint task.
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(1)
            .connect_timeout(cfg.request_timeout)
            .timeout(cfg.request_timeout)
            .build()
            .expect("load monitor http client builds");
        Arc::new(Self {
            cfg,
            client,
            endpoints: DashMap::new(),
            metrics: OnceLock::new(),
            drained: CancellationToken::new(),
            table,
        })
    }

    pub fn config(&self) -> &LoadMonitorConfig {
        &self.cfg
    }

    /// The shared store this monitor feeds.
    pub fn table(&self) -> &Arc<EngineLoadTable> {
        &self.table
    }

    /// Attach the process metrics registry (built after the monitor).
    pub fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        let _ = self.metrics.set(metrics);
    }

    /// Start polling `url` if it is not already tracked. Idempotent; two
    /// workers sharing one physical URL share one pull task.
    pub fn track(self: &Arc<Self>, url: &str) {
        if self.drained.is_cancelled() || self.endpoints.contains_key(url) {
            return;
        }
        let (wake_tx, wake_rx) = mpsc::channel(1);
        let ep = Arc::new(Endpoint {
            url: url.to_string(),
            wake_tx,
            sequence: AtomicU64::new(0),
            cancel: self.drained.child_token(),
        });
        if self
            .endpoints
            .insert(url.to_string(), Arc::clone(&ep))
            .is_some()
        {
            // Lost the race — the other insert already spawned a task.
            ep.cancel.cancel();
            return;
        }
        let this = Arc::clone(self);
        tokio::spawn(async move { this.run_endpoint(ep, wake_rx).await });
    }

    /// Stop polling `url` and drop its rows from the table.
    pub fn untrack(&self, url: &str) {
        if let Some((_, ep)) = self.endpoints.remove(url) {
            ep.cancel.cancel();
        }
        self.table.forget_worker(url);
    }

    /// Cancel every pull task. Called on router shutdown.
    pub fn drain(&self) {
        self.drained.cancel();
        self.endpoints.clear();
    }

    pub fn tracked_urls(&self) -> Vec<String> {
        self.endpoints.iter().map(|e| e.key().clone()).collect()
    }

    /// Ask for a refresh of every listed URL without waiting. Untracked
    /// URLs are ignored.
    pub fn trigger<'a>(&self, urls: impl IntoIterator<Item = &'a str>) {
        for url in urls {
            if let Some(ep) = self.endpoints.get(url) {
                ep.trigger();
            }
        }
    }

    /// Point-in-time diagnostic view of `url`.
    pub fn snapshot(&self, url: &str) -> EndpointLoad {
        let now = Instant::now();
        let sequence = self
            .endpoints
            .get(url)
            .map(|ep| ep.sequence.load(Ordering::Relaxed))
            .unwrap_or(0);
        let (aggregate, received_at) = match self.table.worker_aggregate(url) {
            Some((agg, at)) => (Some(agg), Some(at)),
            None => (None, None),
        };
        let last_error = match self.table.pull_status(url) {
            Some(PullStatus::Unreachable(e)) => Some(e),
            Some(PullStatus::ZeroCapacity) => {
                Some("engine reported zero token or request capacity".to_string())
            }
            None => None,
        };
        EndpointLoad {
            freshness: self.table.worker_freshness(url, now),
            aggregate,
            sequence,
            received_at,
            last_error,
        }
    }

    /// Whether `url` currently passes the freshness gate.
    pub fn is_fresh(&self, url: &str) -> bool {
        self.table.worker_freshness(url, Instant::now()) == Freshness::Fresh
    }

    /// One endpoint's pull loop — mirrors AIgate's `runPull`: sample
    /// immediately, then wait for either a trigger or the periodic
    /// fallback (re-armed from the completion of the previous pull), and
    /// merge a pending trigger into a fallback pull that is already due.
    async fn run_endpoint(self: Arc<Self>, ep: Arc<Endpoint>, mut wake_rx: mpsc::Receiver<()>) {
        // A trigger that raced ahead of the task start is satisfied by the
        // mandatory initial sample.
        let _ = wake_rx.try_recv();
        loop {
            if ep.cancel.is_cancelled() {
                return;
            }
            self.pull_and_store(&ep).await;
            let fallback = tokio::time::sleep(self.cfg.report_interval);
            tokio::select! {
                biased;
                _ = ep.cancel.cancelled() => return,
                _ = fallback => {
                    // A request signal already pending merges into this pull.
                    let _ = wake_rx.try_recv();
                }
                r = wake_rx.recv() => {
                    if r.is_none() {
                        return;
                    }
                }
            }
        }
    }

    async fn pull_and_store(&self, ep: &Endpoint) {
        ep.sequence.fetch_add(1, Ordering::Relaxed);
        let outcome = match self.pull_once(&ep.url).await {
            Ok(report) => {
                let now = Instant::now();
                for (rank, stat) in &report.ranks {
                    self.table.set(&ep.url, *rank, stat.clone(), now);
                }
                if report.positive_capacity {
                    self.table.set_pull_status(&ep.url, None);
                    LoadPullOutcome::Ok
                } else {
                    self.table
                        .set_pull_status(&ep.url, Some(PullStatus::ZeroCapacity));
                    LoadPullOutcome::ZeroCapacity
                }
            }
            Err(e) => {
                tracing::debug!(worker_url = %ep.url, error = %e, "load pull failed");
                self.table
                    .set_pull_status(&ep.url, Some(PullStatus::Unreachable(e)));
                LoadPullOutcome::Unreachable
            }
        };
        if let Some(m) = self.metrics.get() {
            m.record_load_pull(&ep.url, outcome);
            if let Some((agg, _)) = self.table.worker_aggregate(&ep.url) {
                m.set_reported_load(
                    &ep.url,
                    ReportedLoadKind::RunningRequests,
                    i64::try_from(agg.num_running_requests).unwrap_or(i64::MAX),
                );
                m.set_reported_load(
                    &ep.url,
                    ReportedLoadKind::WaitingRequests,
                    i64::try_from(agg.num_waiting_requests).unwrap_or(i64::MAX),
                );
                m.set_reported_load(
                    &ep.url,
                    ReportedLoadKind::FreeTokens,
                    i64::try_from(agg.free_tokens()).unwrap_or(i64::MAX),
                );
            }
        }
    }

    async fn pull_once(&self, worker_url: &str) -> Result<PulledReport, String> {
        let base = reqwest::Url::parse(worker_url).map_err(|e| format!("parse worker URL: {e}"))?;
        let mut url = base
            .join("/v1/loads")
            .map_err(|e| format!("join /v1/loads: {e}"))?;
        url.set_query(Some("include=core"));
        let resp = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("GET /v1/loads failed: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(format!("GET /v1/loads returned HTTP {status}"));
        }
        let body = resp
            .bytes()
            .await
            .map_err(|e| format!("read /v1/loads response: {e}"))?;
        if body.len() > MAX_LOADS_RESPONSE_BYTES {
            return Err("GET /v1/loads response exceeds 4 MiB".to_string());
        }
        parse_loads_response(&body)
    }
}

/// One validated `/v1/loads` body: per-rank gauges in [`LoadStat`] shape.
#[derive(Debug, Clone, PartialEq)]
pub struct PulledReport {
    pub ranks: Vec<(u32, LoadStat)>,
    /// False when every rank reports zero token or request capacity
    /// (SGLang's early-boot snapshot) — the report is stored but the worker
    /// reads as `stale`.
    pub positive_capacity: bool,
}

/// JSON envelope returned by the engine's `GET /v1/loads`.
#[derive(Debug, Deserialize)]
struct LoadsResponse {
    #[serde(default)]
    loads: Vec<LoadsRankJson>,
}

/// One `/v1/loads` DP-rank object (the `core` section). Numeric fields
/// default to zero when an older engine omits them. `num_used_tokens` maps
/// to `LoadStat::num_tokens` (KV tokens in use) — the same quantity the
/// push socket publishes.
#[derive(Debug, Deserialize)]
struct LoadsRankJson {
    #[serde(default)]
    dp_rank: i64,
    #[serde(default)]
    num_running_reqs: i64,
    #[serde(default)]
    num_waiting_reqs: i64,
    #[serde(default)]
    num_used_tokens: i64,
    #[serde(default)]
    max_total_num_tokens: i64,
    #[serde(default)]
    max_running_requests: i64,
}

/// Validate one `/v1/loads` body and convert its DP ranks to [`LoadStat`]s.
pub fn parse_loads_response(body: &[u8]) -> Result<PulledReport, String> {
    let parsed: LoadsResponse =
        serde_json::from_slice(body).map_err(|e| format!("parse /v1/loads response: {e}"))?;
    if parsed.loads.is_empty() {
        return Err("GET /v1/loads returned no DP ranks".to_string());
    }
    let mut seen = std::collections::HashSet::with_capacity(parsed.loads.len());
    let mut ranks = Vec::with_capacity(parsed.loads.len());
    let mut positive_capacity = true;
    for rank in &parsed.loads {
        for (name, v) in [
            ("dp_rank", rank.dp_rank),
            ("num_running_reqs", rank.num_running_reqs),
            ("num_waiting_reqs", rank.num_waiting_reqs),
            ("num_used_tokens", rank.num_used_tokens),
            ("max_total_num_tokens", rank.max_total_num_tokens),
            ("max_running_requests", rank.max_running_requests),
        ] {
            if v < 0 {
                return Err(format!("{name} must be non-negative, got {v}"));
            }
        }
        let dp_rank = u32::try_from(rank.dp_rank)
            .map_err(|_| format!("dp_rank {} out of range", rank.dp_rank))?;
        if !seen.insert(dp_rank) {
            return Err(format!("duplicate dp_rank {dp_rank}"));
        }
        if rank.max_total_num_tokens == 0 || rank.max_running_requests == 0 {
            positive_capacity = false;
        }
        ranks.push((
            dp_rank,
            LoadStat {
                num_running_reqs: rank.num_running_reqs as u64,
                num_waiting_reqs: rank.num_waiting_reqs as u64,
                num_tokens: rank.num_used_tokens as u64,
                max_total_num_tokens: rank.max_total_num_tokens as u64,
            },
        ));
    }
    Ok(PulledReport {
        ranks,
        positive_capacity,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use axum::{Json, Router};
    use serde_json::{json, Value};
    use std::sync::atomic::AtomicUsize;

    fn rank(dp: i32, running: i64, waiting: i64) -> Value {
        json!({
            "dp_rank": dp,
            "num_running_reqs": running,
            "num_waiting_reqs": waiting,
            "num_used_tokens": 100,
            "num_total_tokens": 150,
            "max_total_num_tokens": 1000,
            "max_running_requests": 16,
        })
    }

    #[test]
    fn parse_maps_ranks_onto_load_stat() {
        let body = json!({"timestamp": "x", "loads": [rank(0, 2, 1), rank(1, 3, 0)]});
        let r = parse_loads_response(body.to_string().as_bytes()).unwrap();
        assert!(r.positive_capacity);
        assert_eq!(r.ranks.len(), 2);
        assert_eq!(r.ranks[0].0, 0);
        assert_eq!(r.ranks[0].1.num_running_reqs, 2);
        assert_eq!(r.ranks[0].1.num_waiting_reqs, 1);
        assert_eq!(r.ranks[0].1.num_tokens, 100);
        assert_eq!(r.ranks[0].1.max_total_num_tokens, 1000);
        assert_eq!(r.ranks[1].1.num_running_reqs, 3);
    }

    #[test]
    fn parse_rejects_empty_duplicate_and_negative() {
        assert!(parse_loads_response(br#"{"loads": []}"#).is_err());
        let dup = json!({"loads": [rank(0, 0, 0), rank(0, 0, 0)]});
        assert!(parse_loads_response(dup.to_string().as_bytes())
            .unwrap_err()
            .contains("duplicate"));
        let neg = json!({"loads": [{"dp_rank": 0, "num_running_reqs": -1}]});
        assert!(parse_loads_response(neg.to_string().as_bytes()).is_err());
        assert!(parse_loads_response(b"not json").is_err());
    }

    #[test]
    fn missing_fields_default_to_zero_and_zero_capacity_is_flagged() {
        let r = parse_loads_response(br#"{"loads": [{"dp_rank": 0}]}"#).unwrap();
        assert_eq!(r.ranks[0].1.num_running_reqs, 0);
        assert!(!r.positive_capacity);
    }

    /// Mock engine: `/v1/loads` counts hits and serves a fixed rank set.
    async fn mock_engine(
        running: i64,
    ) -> (String, Arc<AtomicUsize>, tokio::sync::oneshot::Sender<()>) {
        let hits = Arc::new(AtomicUsize::new(0));
        let hits_c = Arc::clone(&hits);
        let app = Router::new().route(
            "/v1/loads",
            get(move || {
                let hits = Arc::clone(&hits_c);
                async move {
                    hits.fetch_add(1, Ordering::Relaxed);
                    Json(json!({"loads": [rank(0, running, 0), rank(1, running, 1)]}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = rx.await;
                })
                .await
                .unwrap();
        });
        (url, hits, tx)
    }

    async fn wait_until(mut pred: impl FnMut() -> bool, what: &str) {
        let start = Instant::now();
        while !pred() {
            assert!(
                start.elapsed() < Duration::from_secs(3),
                "timed out waiting for {what}"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    fn cfg(interval_ms: u64, stale_ms: u64) -> LoadMonitorConfig {
        LoadMonitorConfig {
            report_interval: Duration::from_millis(interval_ms),
            stale_after: Duration::from_millis(stale_ms),
            request_timeout: Duration::from_millis(500),
        }
    }

    #[tokio::test]
    async fn track_pulls_immediately_and_populates_the_shared_table() {
        let (url, hits, _srv) = mock_engine(4).await;
        let table = EngineLoadTable::new();
        let m = LoadMonitor::new(cfg(10_000, 30_000), Arc::clone(&table));
        // `new` aligns the shared window with stale_after.
        assert_eq!(table.freshness(), Duration::from_millis(30_000));
        assert_eq!(m.snapshot(&url).freshness, Freshness::Missing);
        m.track(&url);
        wait_until(|| m.is_fresh(&url), "initial pull").await;
        let snap = m.snapshot(&url);
        assert_eq!(snap.sequence, 1);
        let agg = snap.aggregate.unwrap();
        assert_eq!(agg.ranks, 2);
        assert_eq!(agg.num_total_requests(), 4 + 4 + 1);
        // Per-rank rows landed in the table the policies read.
        let fresh = table.snapshot_fresh(Instant::now());
        assert_eq!(fresh.get(&url).copied(), Some(9));
        assert_eq!(hits.load(Ordering::Relaxed), 1);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(hits.load(Ordering::Relaxed), 1);
        m.drain();
    }

    #[tokio::test]
    async fn trigger_coalesces_to_at_most_one_pending_pull() {
        let (url, hits, _srv) = mock_engine(1).await;
        let m = LoadMonitor::new(cfg(10_000, 30_000), EngineLoadTable::new());
        m.track(&url);
        wait_until(|| m.is_fresh(&url), "initial pull").await;
        for _ in 0..20 {
            m.trigger([url.as_str()]);
        }
        wait_until(|| hits.load(Ordering::Relaxed) >= 2, "triggered pull").await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            hits.load(Ordering::Relaxed) <= 3,
            "20 triggers must coalesce into at most one in-flight plus one pending pull, got {}",
            hits.load(Ordering::Relaxed)
        );
        m.drain();
    }

    #[tokio::test]
    async fn periodic_fallback_keeps_idle_worker_fresh() {
        let (url, hits, _srv) = mock_engine(0).await;
        let m = LoadMonitor::new(cfg(20, 200), EngineLoadTable::new());
        m.track(&url);
        wait_until(|| hits.load(Ordering::Relaxed) >= 4, "periodic pulls").await;
        assert!(m.is_fresh(&url));
        m.drain();
    }

    #[tokio::test]
    async fn stale_when_reports_stop_and_unreachable_when_engine_dies() {
        let (url, _hits, srv) = mock_engine(0).await;
        let m = LoadMonitor::new(cfg(10_000, 40), EngineLoadTable::new());
        m.track(&url);
        wait_until(|| m.is_fresh(&url), "initial pull").await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(m.snapshot(&url).freshness, Freshness::Stale);
        // The policy view drops a stale worker (falls back to the local counter).
        assert!(m.table().snapshot_fresh(Instant::now()).is_empty());
        drop(srv);
        tokio::time::sleep(Duration::from_millis(20)).await;
        m.trigger([url.as_str()]);
        wait_until(
            || m.snapshot(&url).freshness == Freshness::Unreachable,
            "unreachable",
        )
        .await;
        assert!(m.snapshot(&url).last_error.is_some());
        m.drain();
    }

    #[tokio::test]
    async fn zero_capacity_report_is_stale_not_fresh() {
        let app = Router::new().route(
            "/v1/loads",
            get(|| async { Json(json!({"loads": [{"dp_rank": 0, "num_running_reqs": 0}]})) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let m = LoadMonitor::new(cfg(10_000, 30_000), EngineLoadTable::new());
        m.track(&url);
        wait_until(|| m.snapshot(&url).sequence >= 1, "pull").await;
        assert_eq!(m.snapshot(&url).freshness, Freshness::Stale);
        assert!(!m.is_fresh(&url));
        m.drain();
    }

    #[tokio::test]
    async fn untrack_stops_polling_and_forgets_rows() {
        let (url, hits, _srv) = mock_engine(0).await;
        let table = EngineLoadTable::new();
        let m = LoadMonitor::new(cfg(10, 1000), Arc::clone(&table));
        m.track(&url);
        wait_until(|| hits.load(Ordering::Relaxed) >= 2, "pulls").await;
        m.untrack(&url);
        let at_untrack = hits.load(Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(hits.load(Ordering::Relaxed) <= at_untrack + 1);
        assert_eq!(m.snapshot(&url).freshness, Freshness::Missing);
        assert!(table.worker_aggregate(&url).is_none());
        m.drain();
    }
}
