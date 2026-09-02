// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Load Monitor — pull-mode engine load snapshots for routing.
//!
//! Ported from AIgate's `internal/loadmonitor` pull path. One background
//! task per physical worker URL polls `GET /v1/loads?include=core`,
//! aggregates the DP-rank rows into one endpoint-level [`LoadAggregate`],
//! and publishes it behind a freshness gate:
//!
//! * **Trigger coalescing.** Every routed request calls [`LoadMonitor::trigger`]
//!   for the model's workers. The wake channel has capacity one, so under
//!   load a worker is refreshed as fast as one HTTP round-trip allows — at
//!   most one pull in flight plus one pending — never once per request.
//! * **Periodic fallback.** After each pull completes the task re-arms a
//!   `report_interval` timer, so an idle worker keeps a fresh snapshot and
//!   never gets stuck `stale` (which would make it permanently unroutable).
//! * **Freshness.** A worker is [`Freshness::Fresh`] only when its latest
//!   successful pull is younger than `stale_after`. `missing` (never
//!   pulled), `stale` (too old, or the engine reported zero capacity while
//!   still booting) and `unreachable` (HTTP / transport / parse failure)
//!   workers are excluded from routing by the caller.
//! * **Local pre-deduction.** [`LoadMonitor::note_dispatch`] bumps a
//!   per-endpoint pending counter that [`LoadMonitor::load_score`] adds on
//!   top of the engine-reported request count; the counter resets on the
//!   next successful pull. This closes the blind window between a routing
//!   decision and the engine reflecting that request in `/v1/loads`, so a
//!   burst does not herd onto whichever worker looked lightest last pull.

use crate::server::metrics::{LoadPullOutcome, MetricsRegistry, ReportedLoadKind};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::Deserialize;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
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
    /// A successful report older than this is `stale`. Must exceed
    /// `report_interval` or idle workers flap between fresh and stale.
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

/// Endpoint-level aggregate over all DP ranks of one worker.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadAggregate {
    pub num_running_requests: i64,
    pub num_waiting_requests: i64,
    /// `running + waiting` — the request-count load signal.
    pub num_total_requests: i64,
    pub num_used_tokens: i64,
    /// Tokens held by running + queued requests — the prefill load signal.
    pub num_total_tokens: i64,
    pub max_total_tokens: i64,
    pub max_running_requests: i64,
    /// `max(0, max_total_tokens - num_used_tokens)`.
    pub free_tokens: i64,
    /// `max(0, max_running_requests - num_running_requests)`.
    pub available_request_slots: i64,
    pub dp_ranks: usize,
}

/// Routing-visible state of one worker's load report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// No pull has completed yet.
    Missing,
    /// Latest pull succeeded within `stale_after`.
    Fresh,
    /// Latest pull is too old, or the engine reported zero capacity
    /// (still initialising).
    Stale,
    /// Latest pull failed (HTTP status, transport, or payload error).
    Unreachable,
}

impl Freshness {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Fresh => "fresh",
            Self::Stale => "stale",
            Self::Unreachable => "unreachable",
        }
    }
}

/// Read-only view of one endpoint at a point in time.
#[derive(Debug, Clone)]
pub struct EndpointLoad {
    pub freshness: Freshness,
    /// Present for `Fresh` and `Stale`; `None` for `Missing` / `Unreachable`.
    pub aggregate: Option<LoadAggregate>,
    /// Router-local pull sequence (only pulls that actually ran count).
    pub sequence: u64,
    pub received_at: Option<Instant>,
    pub last_error: Option<String>,
    /// Requests routed here since the last successful pull.
    pub pending_dispatches: i64,
}

#[derive(Debug, Clone)]
enum ReportStatus {
    Healthy,
    /// Engine answered but every rank reported zero token or request
    /// capacity — SGLang's early-boot snapshot. Kept for diagnostics but
    /// demoted to `Stale` so the worker is not routable yet.
    ZeroCapacity,
    Unreachable(String),
}

#[derive(Debug, Clone)]
struct Report {
    status: ReportStatus,
    aggregate: Option<LoadAggregate>,
    received_at: Instant,
    sequence: u64,
}

struct Endpoint {
    url: String,
    /// Capacity-one wake channel: at most one pending refresh request.
    wake_tx: mpsc::Sender<()>,
    latest: RwLock<Option<Report>>,
    pending: AtomicI64,
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
    pub fn new(cfg: LoadMonitorConfig) -> Arc<Self> {
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
        })
    }

    pub fn config(&self) -> &LoadMonitorConfig {
        &self.cfg
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
            latest: RwLock::new(None),
            pending: AtomicI64::new(0),
            sequence: AtomicU64::new(0),
            cancel: self.drained.child_token(),
        });
        // Insert first so a concurrent `track` for the same URL is a no-op.
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

    /// Stop polling `url` and forget its snapshot.
    pub fn untrack(&self, url: &str) {
        if let Some((_, ep)) = self.endpoints.remove(url) {
            ep.cancel.cancel();
        }
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

    /// Record that one request was just routed to `url`. Adds to the
    /// pre-deduction pending count until the next successful pull.
    pub fn note_dispatch(&self, url: &str) {
        if let Some(ep) = self.endpoints.get(url) {
            ep.pending.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn freshness_of(&self, report: Option<&Report>, now: Instant) -> Freshness {
        match report {
            None => Freshness::Missing,
            Some(r) => match &r.status {
                ReportStatus::Unreachable(_) => Freshness::Unreachable,
                ReportStatus::ZeroCapacity => Freshness::Stale,
                ReportStatus::Healthy => {
                    if now.saturating_duration_since(r.received_at) >= self.cfg.stale_after {
                        Freshness::Stale
                    } else {
                        Freshness::Fresh
                    }
                }
            },
        }
    }

    /// Point-in-time view of `url`. Untracked URLs read as `Missing`.
    pub fn snapshot(&self, url: &str) -> EndpointLoad {
        let now = Instant::now();
        let Some(ep) = self.endpoints.get(url) else {
            return EndpointLoad {
                freshness: Freshness::Missing,
                aggregate: None,
                sequence: 0,
                received_at: None,
                last_error: None,
                pending_dispatches: 0,
            };
        };
        let latest = ep.latest.read().clone();
        let freshness = self.freshness_of(latest.as_ref(), now);
        let (aggregate, sequence, received_at, last_error) = match latest {
            None => (None, 0, None, None),
            Some(r) => {
                let err = match &r.status {
                    ReportStatus::Unreachable(e) => Some(e.clone()),
                    ReportStatus::ZeroCapacity => {
                        Some("engine reported zero token or request capacity".to_string())
                    }
                    ReportStatus::Healthy => None,
                };
                (r.aggregate, r.sequence, Some(r.received_at), err)
            }
        };
        EndpointLoad {
            freshness,
            aggregate,
            sequence,
            received_at,
            last_error,
            pending_dispatches: ep.pending.load(Ordering::Relaxed),
        }
    }

    pub fn is_fresh(&self, url: &str) -> bool {
        let now = Instant::now();
        match self.endpoints.get(url) {
            Some(ep) => self.freshness_of(ep.latest.read().as_ref(), now) == Freshness::Fresh,
            None => false,
        }
    }

    /// Request-count load score for a fresh worker: engine-reported
    /// `running + waiting` plus requests this router dispatched since that
    /// report. `None` when the worker is not fresh — callers fall back to
    /// their local counter.
    pub fn load_score(&self, url: &str) -> Option<i64> {
        let now = Instant::now();
        let ep = self.endpoints.get(url)?;
        let latest = ep.latest.read();
        if self.freshness_of(latest.as_ref(), now) != Freshness::Fresh {
            return None;
        }
        let reported = latest.as_ref()?.aggregate?.num_total_requests;
        Some(reported + ep.pending.load(Ordering::Relaxed).max(0))
    }

    /// Token load score for a fresh worker (prefill-side signal).
    pub fn token_score(&self, url: &str) -> Option<i64> {
        let now = Instant::now();
        let ep = self.endpoints.get(url)?;
        let latest = ep.latest.read();
        if self.freshness_of(latest.as_ref(), now) != Freshness::Fresh {
            return None;
        }
        Some(latest.as_ref()?.aggregate?.num_total_tokens)
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
        let sequence = ep.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        let (status, aggregate) = match self.pull_once(&ep.url).await {
            Ok(agg) if has_positive_capacity(&agg) => (ReportStatus::Healthy, Some(agg)),
            Ok(agg) => (ReportStatus::ZeroCapacity, Some(agg)),
            Err(e) => (ReportStatus::Unreachable(e), None),
        };
        let outcome = match &status {
            ReportStatus::Healthy => LoadPullOutcome::Ok,
            ReportStatus::ZeroCapacity => LoadPullOutcome::ZeroCapacity,
            ReportStatus::Unreachable(e) => {
                tracing::debug!(worker_url = %ep.url, error = %e, "load pull failed");
                LoadPullOutcome::Unreachable
            }
        };
        if matches!(status, ReportStatus::Healthy) {
            // Authoritative report supersedes the local pre-deduction.
            ep.pending.store(0, Ordering::Relaxed);
        }
        *ep.latest.write() = Some(Report {
            status,
            aggregate,
            received_at: Instant::now(),
            sequence,
        });
        if let Some(m) = self.metrics.get() {
            m.record_load_pull(&ep.url, outcome);
            if let Some(agg) = aggregate {
                m.set_reported_load(
                    &ep.url,
                    ReportedLoadKind::RunningRequests,
                    agg.num_running_requests,
                );
                m.set_reported_load(
                    &ep.url,
                    ReportedLoadKind::WaitingRequests,
                    agg.num_waiting_requests,
                );
                m.set_reported_load(&ep.url, ReportedLoadKind::FreeTokens, agg.free_tokens);
            }
        }
    }

    async fn pull_once(&self, worker_url: &str) -> Result<LoadAggregate, String> {
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

/// JSON envelope returned by the engine's `GET /v1/loads`.
#[derive(Debug, Deserialize)]
struct LoadsResponse {
    #[serde(default)]
    loads: Vec<LoadsRankJson>,
}

/// One `/v1/loads` DP-rank object (the `core` section). Numeric fields
/// default to zero when an older engine omits them.
#[derive(Debug, Deserialize)]
struct LoadsRankJson {
    #[serde(default)]
    dp_rank: i32,
    #[serde(default)]
    num_running_reqs: i64,
    #[serde(default)]
    num_waiting_reqs: i64,
    #[serde(default)]
    num_used_tokens: i64,
    #[serde(default)]
    num_total_tokens: i64,
    #[serde(default)]
    max_total_num_tokens: i64,
    #[serde(default)]
    max_running_requests: i64,
}

/// Validate one `/v1/loads` body and aggregate its DP ranks.
pub fn parse_loads_response(body: &[u8]) -> Result<LoadAggregate, String> {
    let parsed: LoadsResponse =
        serde_json::from_slice(body).map_err(|e| format!("parse /v1/loads response: {e}"))?;
    if parsed.loads.is_empty() {
        return Err("GET /v1/loads returned no DP ranks".to_string());
    }
    let mut seen = std::collections::HashSet::with_capacity(parsed.loads.len());
    let mut agg = LoadAggregate::default();
    for rank in &parsed.loads {
        if rank.dp_rank < 0 {
            return Err(format!(
                "dp_rank must be non-negative, got {}",
                rank.dp_rank
            ));
        }
        if !seen.insert(rank.dp_rank) {
            return Err(format!("duplicate dp_rank {}", rank.dp_rank));
        }
        for (name, v) in [
            ("num_running_reqs", rank.num_running_reqs),
            ("num_waiting_reqs", rank.num_waiting_reqs),
            ("num_used_tokens", rank.num_used_tokens),
            ("num_total_tokens", rank.num_total_tokens),
            ("max_total_num_tokens", rank.max_total_num_tokens),
            ("max_running_requests", rank.max_running_requests),
        ] {
            if v < 0 {
                return Err(format!("{name} must be non-negative, got {v}"));
            }
        }
        agg.num_running_requests += rank.num_running_reqs;
        agg.num_waiting_requests += rank.num_waiting_reqs;
        agg.num_used_tokens += rank.num_used_tokens;
        agg.num_total_tokens += rank.num_total_tokens;
        agg.max_total_tokens += rank.max_total_num_tokens;
        agg.max_running_requests += rank.max_running_requests;
        agg.dp_ranks += 1;
    }
    agg.num_total_requests = agg.num_running_requests + agg.num_waiting_requests;
    agg.free_tokens = (agg.max_total_tokens - agg.num_used_tokens).max(0);
    agg.available_request_slots = (agg.max_running_requests - agg.num_running_requests).max(0);
    Ok(agg)
}

fn has_positive_capacity(agg: &LoadAggregate) -> bool {
    agg.max_total_tokens > 0 && agg.max_running_requests > 0
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
    fn parse_aggregates_ranks_and_derives_fields() {
        let body = json!({"timestamp": "x", "loads": [rank(0, 2, 1), rank(1, 3, 0)]});
        let agg = parse_loads_response(body.to_string().as_bytes()).unwrap();
        assert_eq!(agg.dp_ranks, 2);
        assert_eq!(agg.num_running_requests, 5);
        assert_eq!(agg.num_waiting_requests, 1);
        assert_eq!(agg.num_total_requests, 6);
        assert_eq!(agg.num_total_tokens, 300);
        assert_eq!(agg.max_total_tokens, 2000);
        assert_eq!(agg.free_tokens, 1800);
        assert_eq!(agg.available_request_slots, 32 - 5);
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
    fn missing_fields_default_to_zero_and_zero_capacity_is_detected() {
        let agg = parse_loads_response(br#"{"loads": [{"dp_rank": 0}]}"#).unwrap();
        assert_eq!(agg.num_total_requests, 0);
        assert!(!has_positive_capacity(&agg));
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
                    Json(json!({"loads": [rank(0, running, 0)]}))
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
    async fn track_pulls_immediately_and_becomes_fresh() {
        let (url, hits, _srv) = mock_engine(4).await;
        let m = LoadMonitor::new(cfg(10_000, 30_000));
        assert_eq!(m.snapshot(&url).freshness, Freshness::Missing);
        m.track(&url);
        wait_until(|| m.is_fresh(&url), "initial pull").await;
        let snap = m.snapshot(&url);
        assert_eq!(snap.sequence, 1);
        assert_eq!(snap.aggregate.unwrap().num_total_requests, 4);
        assert_eq!(m.load_score(&url), Some(4));
        assert_eq!(hits.load(Ordering::Relaxed), 1);
        // Long interval: no periodic pull sneaks in.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(hits.load(Ordering::Relaxed), 1);
        m.drain();
    }

    #[tokio::test]
    async fn trigger_coalesces_and_resets_pending_dispatches() {
        let (url, hits, _srv) = mock_engine(1).await;
        let m = LoadMonitor::new(cfg(10_000, 30_000));
        m.track(&url);
        wait_until(|| m.is_fresh(&url), "initial pull").await;
        m.note_dispatch(&url);
        m.note_dispatch(&url);
        assert_eq!(m.load_score(&url), Some(3), "reported 1 + 2 pending");
        // Many triggers while idle → exactly one extra pull, which resets
        // the pre-deduction to the authoritative value.
        for _ in 0..20 {
            m.trigger([url.as_str()]);
        }
        wait_until(|| hits.load(Ordering::Relaxed) >= 2, "triggered pull").await;
        wait_until(|| m.snapshot(&url).pending_dispatches == 0, "pending reset").await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            hits.load(Ordering::Relaxed) <= 3,
            "20 triggers must coalesce into at most one in-flight plus one pending pull, got {}",
            hits.load(Ordering::Relaxed)
        );
        assert_eq!(m.load_score(&url), Some(1));
        m.drain();
    }

    #[tokio::test]
    async fn periodic_fallback_keeps_idle_worker_fresh() {
        let (url, hits, _srv) = mock_engine(0).await;
        let m = LoadMonitor::new(cfg(20, 200));
        m.track(&url);
        wait_until(|| hits.load(Ordering::Relaxed) >= 4, "periodic pulls").await;
        assert!(m.is_fresh(&url));
        m.drain();
    }

    #[tokio::test]
    async fn stale_when_reports_stop_and_unreachable_when_engine_dies() {
        let (url, _hits, srv) = mock_engine(0).await;
        // Interval far longer than stale_after so no refresh lands.
        let m = LoadMonitor::new(cfg(10_000, 40));
        m.track(&url);
        wait_until(|| m.is_fresh(&url), "initial pull").await;
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(m.snapshot(&url).freshness, Freshness::Stale);
        assert_eq!(m.load_score(&url), None);
        // Kill the engine, trigger a pull → unreachable with an error.
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
        let m = LoadMonitor::new(cfg(10_000, 30_000));
        m.track(&url);
        wait_until(|| m.snapshot(&url).sequence >= 1, "pull").await;
        assert_eq!(m.snapshot(&url).freshness, Freshness::Stale);
        assert!(!m.is_fresh(&url));
        m.drain();
    }

    #[tokio::test]
    async fn untrack_stops_polling_and_forgets_snapshot() {
        let (url, hits, _srv) = mock_engine(0).await;
        let m = LoadMonitor::new(cfg(10, 1000));
        m.track(&url);
        wait_until(|| hits.load(Ordering::Relaxed) >= 2, "pulls").await;
        m.untrack(&url);
        let at_untrack = hits.load(Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(hits.load(Ordering::Relaxed) <= at_untrack + 1);
        assert_eq!(m.snapshot(&url).freshness, Freshness::Missing);
        m.drain();
    }
}
