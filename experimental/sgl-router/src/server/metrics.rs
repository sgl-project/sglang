// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Lightweight in-process Prometheus exposition.
//!
//! We deliberately do NOT pull in the `metrics` + `metrics-exporter-prometheus`
//! crates: the observability surface is small enough that a hand-written
//! counter + histogram + gauge family is cheaper than a new dependency, and
//! it lets us label/serialise exactly the way the convergence and PD-affinity
//! tests want.
//!
//! All operations are concurrent — counters and gauges use
//! [`std::sync::atomic`], histograms use a [`Mutex<Vec<u64>>`] over a
//! fixed bucket set. Tests sub-second; production scrapes are 15s
//! cadence. Lock contention is not a concern at these rates.
//!
//! # Metrics surface
//!
//! | Metric | Type | Labels |
//! |---|---|---|
//! | `sgl_router_requests_total` | Counter | `route`, `method` |
//! | `sgl_router_worker_requests_total` | Counter | `worker_url`, `model_id`, `mode`, `outcome` |
//! | `sgl_router_request_duration_seconds` | Histogram | `model_id` |
//! | `sgl_router_ttft_seconds` | Histogram | `model_id` |
//! | `sgl_router_responses_total` | Counter | `route`, `method`, `status_code` |
//! | `sgl_router_overlap_blocks` | Histogram | `model_id` |
//! | `sgl_router_active_load` | Gauge | `worker_url`, `kind` |
//! | `sgl_router_workers` | Gauge | `mode` |
//! | `sgl_router_worker_health` | Gauge | `worker_url` |
//! | `sgl_router_worker_cb_state` | Gauge | `worker_url` |
//! | `sgl_router_worker_inflight_requests` | Gauge | `worker_url` |
//! | `sgl_router_stale_requests_total` | Counter | `outcome` |
//! | `sgl_router_decode_affinity_total` | Counter | `outcome` |
//! | `sgl_router_sticky_total` | Counter | `outcome` |
//! | `sgl_router_ingress_tokenize_errors_total` | Counter | `model_id` |
//! | `sgl_router_backpressure_rejected_total` | Counter | `model_id` |
//! | `sgl_router_queued_requests` | Gauge | (none) |
//! | `sgl_router_admission_wait_seconds` | Histogram | `model_id` |
//!
//! The four `sgl_router_worker*` gauges and `sgl_router_workers` are sampled
//! at scrape time from the live [`crate::workers::WorkerRegistry`] (passed to
//! [`MetricsRegistry::render_with_workers`]) rather than pushed — there is no
//! health-check loop to push from, and pull-on-scrape means a removed worker
//! stops emitting series immediately instead of leaving a stale gauge.
//!
//! The exposition is text/plain; version=0.0.4 per the Prometheus spec.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

/// Histogram bucket upper bounds for `sgl_router_overlap_blocks`. Blocks are
/// 32–64 tokens each, and the `MAX_CHAT_BODY_BYTES` cap bounds context length —
/// putting the practical ceiling for a maximum-length context in the low tens
/// of thousands of blocks. The ladder spans 0 → ~8k blocks at the resolution
/// worth charting; the `+Inf` bucket catches the longer-context tail beyond
/// 8000.
const OVERLAP_BLOCKS_BUCKETS: &[f64] = &[
    0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1000.0, 2000.0, 4000.0, 8000.0,
];

/// Histogram bucket upper bounds (seconds) for
/// `sgl_router_request_duration_seconds`. Standard latency ladder spanning
/// 5 ms → 30 s; the `+Inf` bucket catches anything slower (a request that
/// outlives the upstream's own timeouts).
const REQUEST_DURATION_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

/// Histogram bucket upper bounds (seconds) for `sgl_router_ttft_seconds`.
///
/// From 0.1 s up these edges are IDENTICAL to the SGLang engine's
/// `sglang:time_to_first_token_seconds` histogram (defined in
/// `python/sglang/srt/observability/metrics_collector.py`). Matching edges is
/// what makes a `histogram_quantile` comparison between the router and the
/// engine meaningful: the quantile interpolates within the same bucket on both
/// sides, so `quantile(router) - quantile(engine)` reflects real router
/// overhead rather than grid skew. With mismatched grids the two interpolations
/// run on different bucket widths and the difference can even go negative — the
/// router P50 reading *below* the engine P50 despite the router sitting in
/// front of it.
///
/// The four sub-100 ms edges have no engine counterpart (the engine's first
/// bucket is `[0, 0.1]`, so it cannot resolve a sub-100 ms TTFT at all). They
/// are router-only headroom: harmless for the comparison (they sit below the
/// engine's range) while letting the router resolve a genuinely fast TTFT.
const TTFT_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, // router-only sub-100 ms head
    0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0,
    400.0,
];

/// Histogram bucket upper bounds (seconds) for
/// `sgl_router_admission_wait_seconds` — time a request spends parked in the
/// admission wait queue. Extends the standard latency ladder out to 120 s: the
/// wait queue can park for a long time under sustained saturation, and the
/// request-duration ladder (capped at 30 s) would dump those into `+Inf`,
/// hiding them from `histogram_quantile`.
const ADMISSION_WAIT_BUCKETS: &[f64] = &[
    0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
];

/// Recordable outcome for a request — narrowed to a handful of variants so
/// the label cardinality stays bounded.
#[derive(Debug, Clone, Copy)]
pub enum RequestOutcome {
    Success,
    Error,
    Cancelled,
}

impl RequestOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Error => "error",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Worker dispatch mode label — narrowed to the three modes the policy
/// resolver distinguishes. The `Plain` variant covers the non-PD case.
#[derive(Debug, Clone, Copy)]
pub enum WorkerModeLabel {
    Prefill,
    Decode,
    Plain,
}

impl WorkerModeLabel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Plain => "plain",
        }
    }
}

/// Routing context a handler attaches to its `Response` (via response
/// extensions) so the outermost access-log/metrics middleware can record the
/// per-worker labels it otherwise cannot see. Present on every routed request
/// (the request reached a worker); absent on requests rejected before routing
/// (a body-validation 400, an admission 503 shed, model-not-found), which the
/// middleware records with an empty `worker_url`. Recording in one place — the
/// middleware — is what makes `worker_requests_total` cover EVERY request, so
/// `sum by (outcome)` reflects all ingress, not just dispatched traffic.
#[derive(Debug, Clone)]
pub struct RequestLogContext {
    pub worker_url: String,
    pub model_id: String,
    pub mode: WorkerModeLabel,
}

/// Derive the bounded [`RequestOutcome`] label from the client-visible HTTP
/// status: 2xx is success, 504 is the stale-request cancellation, everything
/// else (incl. forwarded 4xx/5xx and proxy-side 5xx) is an error. Shared by the
/// middleware so every request — routed or rejected pre-routing — is labelled
/// the same way.
///
/// The 504→`Cancelled` mapping conflates the router's own stale-request cancel
/// with a genuine upstream 504 forwarded unchanged; the unambiguous signal for a
/// real stale-cancel is `stale_requests_total{outcome="expired"}`, which only
/// the janitor increments.
pub fn outcome_from_status(status: u16) -> RequestOutcome {
    match status {
        200..=299 => RequestOutcome::Success,
        504 => RequestOutcome::Cancelled,
        _ => RequestOutcome::Error,
    }
}

/// Decode-affinity outcome — see `select_decode_with_affinity` for the
/// three reasons the affinity may not be honored.
#[derive(Debug, Clone, Copy)]
pub enum DecodeAffinityOutcome {
    SameHostPicked,
    FallbackBreaker,
    FallbackLoadImbalance,
}

impl DecodeAffinityOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::SameHostPicked => "same_host_picked",
            Self::FallbackBreaker => "fallback_breaker",
            Self::FallbackLoadImbalance => "fallback_load_imbalance",
        }
    }
}

/// Sticky-policy selection outcome — see `StickyPolicy::select` for the
/// four branches.
#[derive(Debug, Clone, Copy)]
pub enum StickyOutcome {
    /// Routing key found and its assigned worker is still healthy.
    Hit,
    /// Routing key seen for the first time — a worker was assigned.
    Assigned,
    /// Routing key's assigned worker left the healthy set — remapped.
    Remap,
    /// Request carried no routing key — delegated to the fallback policy.
    NoRoutingKey,
}

impl StickyOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Assigned => "assigned",
            Self::Remap => "remap",
            Self::NoRoutingKey => "no_routing_key",
        }
    }
}

/// Stale-request outcome label.
#[derive(Debug, Clone, Copy)]
pub enum StaleRequestOutcome {
    Expired,
}

impl StaleRequestOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Expired => "expired",
        }
    }
}

/// Active-load kind label — separates the two axes of per-worker load.
#[derive(Debug, Clone, Copy)]
pub enum ActiveLoadKind {
    PrefillTokens,
    DecodeBlocks,
}

impl ActiveLoadKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::PrefillTokens => "prefill_tokens",
            Self::DecodeBlocks => "decode_blocks",
        }
    }
}

/// The shared metrics registry, held on `AppContext`. Cheap to clone — all
/// internal state is `Arc`/`Atomic`/`Mutex`-protected.
#[derive(Debug, Default)]
pub struct MetricsRegistry {
    // Edge intake, counted at the `access_log_and_record` middleware BEFORE the
    // handler runs, so it includes requests parked/shed/cancelled/dropped before
    // a response is produced. `requests_total - responses_total` (per route,method)
    // = received but never answered, which the post-dispatch `worker_requests_total`
    // cannot see.
    requests_total: Mutex<HashMap<EdgeKey, Arc<AtomicU64>>>,
    // Per-worker dispatch outcomes. Recorded after a worker is picked, so blind
    // to pre-dispatch drops (see `requests_total`).
    worker_requests_total: Mutex<HashMap<RequestKey, Arc<AtomicU64>>>,
    // Keyed by `model_id` only: a model's pool is either all-plain or all-PD
    // (the registry rejects mixed pools), so the worker `mode` would be a pure
    // function of `model_id` here — a redundant label. Per-worker `mode` lives
    // on `worker_requests_total` / the worker gauges instead.
    request_duration: Mutex<HashMap<String, Histogram>>,
    ttft_seconds: Mutex<HashMap<String, Histogram>>,
    // Edge responses by route/method/status (incl. early-exit 400/413/503 and the
    // CatchPanicLayer-synthesized 500).
    responses_total: Mutex<HashMap<EdgeResponseKey, Arc<AtomicU64>>>,
    overlap_blocks: Mutex<HashMap<String, Histogram>>,
    active_load: Mutex<HashMap<ActiveLoadKey, Arc<AtomicI64>>>,
    stale_requests_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
    decode_affinity_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
    sticky_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
    ingress_tokenize_errors_total: Mutex<HashMap<String, Arc<AtomicU64>>>,
    backpressure_rejected_total: Mutex<HashMap<String, Arc<AtomicU64>>>,
    /// Current admission wait-queue depth (parked requests). A single
    /// router-wide gauge — the queue is shared across the router, so this is a
    /// global count, not per-model.
    queued_requests: AtomicI64,
    /// Time a parked request waited before admission (seconds), per model.
    /// Only parked-then-admitted requests are recorded.
    admission_wait_seconds: Mutex<HashMap<String, Histogram>>,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct RequestKey {
    worker_url: String,
    model_id: String,
    mode: &'static str,
    outcome: &'static str,
}

/// Labels for the edge `requests_total` counter. `route` is the matched route
/// template (a small fixed set), not the raw URI, so cardinality stays bounded.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct EdgeKey {
    route: String,
    method: String,
}

/// Labels for the edge `responses_total` counter: an [`EdgeKey`] plus the final
/// client-visible HTTP status.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct EdgeResponseKey {
    route: String,
    method: String,
    status_code: u16,
}

/// Per-worker state sampled from the [`crate::workers::WorkerRegistry`] at
/// scrape time and rendered as the `sgl_router_workers` /
/// `sgl_router_worker_*` gauge families. Built by the `/metrics` route from
/// the live registry on every scrape — see [`MetricsRegistry::render_with_workers`].
#[derive(Debug, Clone)]
pub struct WorkerSnapshot {
    pub worker_url: String,
    /// `"plain"`, `"prefill"`, or `"decode"`.
    pub mode: &'static str,
    /// Circuit breaker would currently admit a request (`would_allow`).
    pub healthy: bool,
    /// Circuit breaker state code: 0=closed, 1=open, 2=half_open.
    pub cb_state: u8,
    /// In-flight request count for this worker (`Worker::active_load`).
    pub inflight: i64,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct ActiveLoadKey {
    worker_url: String,
    kind: &'static str,
}

#[derive(Debug)]
struct Histogram {
    /// Bucket upper bounds this histogram observes against (e.g.
    /// [`OVERLAP_BLOCKS_BUCKETS`] or [`REQUEST_DURATION_BUCKETS`]). Held
    /// per-instance so a single `Histogram` type backs metrics with
    /// different bucket ladders.
    bounds: &'static [f64],
    /// One counter per boundary in `bounds`, plus one for `+Inf`. Buckets
    /// are cumulative on render but stored as non-cumulative counts here.
    buckets: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        debug_assert!(
            bounds.windows(2).all(|w| w[0] <= w[1]),
            "histogram bounds must be ascending; `observe` relies on first-match placement",
        );
        Self {
            bounds,
            buckets: vec![0; bounds.len() + 1],
            sum: 0.0,
            count: 0,
        }
    }

    fn observe(&mut self, value: f64) {
        let mut placed = false;
        for (i, &bound) in self.bounds.iter().enumerate() {
            if value <= bound {
                self.buckets[i] += 1;
                placed = true;
                break;
            }
        }
        if !placed {
            // +Inf bucket
            let last = self.buckets.len() - 1;
            self.buckets[last] += 1;
        }
        self.sum += value;
        self.count += 1;
    }
}

impl MetricsRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Bump `sgl_router_worker_requests_total` for the given worker / model / mode / outcome.
    pub fn record_worker_request(
        &self,
        worker_url: &str,
        model_id: &str,
        mode: WorkerModeLabel,
        outcome: RequestOutcome,
    ) {
        let key = RequestKey {
            worker_url: worker_url.to_owned(),
            model_id: model_id.to_owned(),
            mode: mode.as_str(),
            outcome: outcome.as_str(),
        };
        let mut guard = self.worker_requests_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Observe an overlap-blocks count for `sgl_router_overlap_blocks`.
    pub fn observe_overlap_blocks(&self, model_id: &str, blocks: u64) {
        let mut guard = self.overlap_blocks.lock();
        let hist = guard
            .entry(model_id.to_owned())
            .or_insert_with(|| Histogram::new(OVERLAP_BLOCKS_BUCKETS));
        hist.observe(blocks as f64);
    }

    /// Observe end-to-end request latency (seconds) for
    /// `sgl_router_request_duration_seconds`. Recorded once the upstream
    /// outcome is known, regardless of success or error — a slow error is
    /// still latency the operator cares about.
    pub fn observe_request_duration(&self, model_id: &str, seconds: f64) {
        // Drop non-finite observations before touching the map: a NaN would
        // poison the series `sum` permanently (NaN propagates through every
        // later add). Guarding here (not in `Histogram::observe`) also avoids
        // materializing an empty series for a dropped observation. Current
        // callers feed `Instant::elapsed`, so this is defense-in-depth.
        if !seconds.is_finite() {
            return;
        }
        let mut guard = self.request_duration.lock();
        let hist = guard
            .entry(model_id.to_owned())
            .or_insert_with(|| Histogram::new(REQUEST_DURATION_BUCKETS));
        hist.observe(seconds);
    }

    /// Observe time-to-first-token (seconds) for `sgl_router_ttft_seconds` —
    /// the interval from request receipt to the first response chunk arriving
    /// from the upstream worker. Recorded only for successful *streaming*
    /// responses; non-streaming "first token" equals total latency, which
    /// `sgl_router_request_duration_seconds` already captures. Uses
    /// [`TTFT_BUCKETS`], whose edges align with the engine's TTFT histogram so
    /// the two are directly comparable in `histogram_quantile`.
    pub fn observe_ttft(&self, model_id: &str, seconds: f64) {
        // See `observe_request_duration` — drop non-finite before the map.
        if !seconds.is_finite() {
            return;
        }
        let mut guard = self.ttft_seconds.lock();
        let hist = guard
            .entry(model_id.to_owned())
            .or_insert_with(|| Histogram::new(TTFT_BUCKETS));
        hist.observe(seconds);
    }

    /// Bump `sgl_router_requests_total{route,method}` — true edge intake, counted
    /// by the `access_log_and_record` middleware BEFORE the handler runs, so a
    /// request that is parked/shed/cancelled/dropped before producing a response
    /// is still counted. `requests_total - responses_total` is the
    /// received-but-not-answered gap.
    pub fn record_ingress(&self, route: &str, method: &str) {
        let key = EdgeKey {
            route: route.to_owned(),
            method: method.to_owned(),
        };
        let mut guard = self.requests_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump `sgl_router_responses_total{route,method,status_code}` for the HTTP
    /// status the client ultimately saw. Driven by the `access_log_and_record`
    /// middleware, so it counts EVERY response across all routes — including
    /// error short-circuits that never reach a handler's own bookkeeping (an
    /// admission 503, a body-limit 413). For streaming responses the status is
    /// the one sent with the headers, so a stream that fails mid-body after a
    /// 200 header is counted as 200 (the breaker / duration metrics capture
    /// mid-stream failures). `route` is the matched template, so cardinality
    /// stays bounded.
    pub fn record_response(&self, route: &str, method: &str, status_code: u16) {
        let key = EdgeResponseKey {
            route: route.to_owned(),
            method: method.to_owned(),
            status_code,
        };
        let mut guard = self.responses_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Set `sgl_router_active_load` for the given worker + kind. Replaces the
    /// previous value (gauge semantics).
    pub fn set_active_load(&self, worker_url: &str, kind: ActiveLoadKind, value: i64) {
        let key = ActiveLoadKey {
            worker_url: worker_url.to_owned(),
            kind: kind.as_str(),
        };
        let mut guard = self.active_load.lock();
        let gauge = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicI64::new(0)))
            .clone();
        drop(guard);
        gauge.store(value, Ordering::Relaxed);
    }

    /// Bump `sgl_router_stale_requests_total{outcome}`.
    pub fn record_stale_request(&self, outcome: StaleRequestOutcome) {
        let mut guard = self.stale_requests_total.lock();
        let counter = guard
            .entry(outcome.as_str())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump `sgl_router_decode_affinity_total{outcome}`.
    pub fn record_decode_affinity(&self, outcome: DecodeAffinityOutcome) {
        let mut guard = self.decode_affinity_total.lock();
        let counter = guard
            .entry(outcome.as_str())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump `sgl_router_sticky_total{outcome}`.
    pub fn record_sticky(&self, outcome: StickyOutcome) {
        let mut guard = self.sticky_total.lock();
        let counter = guard
            .entry(outcome.as_str())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump `sgl_router_ingress_tokenize_errors_total{model_id}`.
    ///
    /// Recorded ONLY when the tokenization offload SHOULD have fired but the
    /// router's chat encoder failed: a chat request (`messages`) on a model with
    /// a chat encoder that did not yield engine-equivalent ids. That request
    /// silently fell back to engine-side tokenization, defeating the offload —
    /// the actionable "offload broken" signal. It stays at ~0 in healthy
    /// operation and climbs only on a real tokenizer problem; successful
    /// forwards and expected omissions (tools / multimodal / thinking, whose
    /// ids are engine-equivalent but withheld by the safe-predicate) are NOT
    /// counted. Pairs with the per-occurrence WARN log in `tokenize_text`.
    pub fn record_ingress_tokenize_error(&self, model_id: &str) {
        let mut guard = self.ingress_tokenize_errors_total.lock();
        let counter = guard
            .entry(model_id.to_owned())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump `sgl_router_backpressure_rejected_total{model_id}` — a request the
    /// admission gate shed with 503 because every worker was at its in-flight
    /// cap and the wait queue was full.
    pub fn record_backpressure_rejected(&self, model_id: &str) {
        let mut guard = self.backpressure_rejected_total.lock();
        let counter = guard
            .entry(model_id.to_owned())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Adjust `sgl_router_queued_requests` by `delta` (+1 when a request parks,
    /// -1 when it leaves the wait queue). The gauge atomic is the source of
    /// truth: applying balanced deltas keeps it exact under concurrency and
    /// converges to 0 at idle, unlike storing a separately-computed depth
    /// (which can publish a stale value when two parks/unparks interleave).
    pub fn add_queued_requests(&self, delta: i64) {
        self.queued_requests.fetch_add(delta, Ordering::Relaxed);
    }

    /// Observe the time (seconds) a request spent parked in the admission wait
    /// queue before being admitted, for `sgl_router_admission_wait_seconds`.
    /// Recorded only for parked-then-admitted requests, so the histogram's
    /// `_count` is the number of requests that had to wait at all.
    pub fn observe_admission_wait(&self, model_id: &str, seconds: f64) {
        if !seconds.is_finite() {
            return;
        }
        let mut guard = self.admission_wait_seconds.lock();
        let hist = guard
            .entry(model_id.to_owned())
            .or_insert_with(|| Histogram::new(ADMISSION_WAIT_BUCKETS));
        hist.observe(seconds);
    }

    /// Render the registry as a Prometheus 0.0.4 exposition-format string
    /// with no live worker snapshot. The per-worker gauges emit only their
    /// HELP/TYPE headers and a zeroed pool-size series. Production scrapes
    /// go through [`Self::render_with_workers`]; this exists for callers
    /// (and tests) that have no [`crate::workers::WorkerRegistry`] handy.
    pub fn render(&self) -> String {
        self.render_with_workers(&[])
    }

    /// Render the full exposition, sampling the supplied per-worker
    /// [`WorkerSnapshot`]s into the `sgl_router_workers` /
    /// `sgl_router_worker_*` gauge families.
    pub fn render_with_workers(&self, workers: &[WorkerSnapshot]) -> String {
        let mut out = String::new();

        // requests_total — true edge intake (every request, counted before the handler)
        out.push_str(
            "# HELP sgl_router_requests_total Requests received at the router HTTP edge, counted before the handler runs (true intake; includes requests dropped/shed/cancelled before a response).\n",
        );
        out.push_str("# TYPE sgl_router_requests_total counter\n");
        let guard = self.requests_total.lock();
        let mut entries: Vec<(&EdgeKey, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.route, &a.0.method).cmp(&(&b.0.route, &b.0.method)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_requests_total{{route=\"{}\",method=\"{}\"}} {}\n",
                escape_label(&key.route),
                escape_label(&key.method),
                value,
            ));
        }
        drop(guard);

        // worker_requests_total
        out.push_str(
            "# HELP sgl_router_worker_requests_total Total chat-completions requests dispatched to a worker.\n",
        );
        out.push_str("# TYPE sgl_router_worker_requests_total counter\n");
        let guard = self.worker_requests_total.lock();
        // Sort for stable output — easier for tests.
        let mut entries: Vec<(&RequestKey, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| {
            (&a.0.worker_url, &a.0.model_id, a.0.mode, a.0.outcome).cmp(&(
                &b.0.worker_url,
                &b.0.model_id,
                b.0.mode,
                b.0.outcome,
            ))
        });
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_worker_requests_total{{worker_url=\"{}\",model_id=\"{}\",mode=\"{}\",outcome=\"{}\"}} {}\n",
                escape_label(&key.worker_url),
                escape_label(&key.model_id),
                key.mode,
                key.outcome,
                value,
            ));
        }
        drop(guard);

        // request_duration histogram
        out.push_str(
            "# HELP sgl_router_request_duration_seconds End-to-end latency of chat-completions requests dispatched to a worker, in seconds (streaming requests are measured to stream completion).\n",
        );
        out.push_str("# TYPE sgl_router_request_duration_seconds histogram\n");
        let guard = self.request_duration.lock();
        let mut models: Vec<&String> = guard.keys().collect();
        models.sort();
        for model_id in models {
            let hist = guard.get(model_id).unwrap();
            let label_body = format!("model_id=\"{}\"", escape_label(model_id));
            render_histogram(
                &mut out,
                "sgl_router_request_duration_seconds",
                &label_body,
                hist,
            );
        }
        drop(guard);

        // ttft histogram
        out.push_str(
            "# HELP sgl_router_ttft_seconds Time to first token (first upstream response chunk) for streaming requests, in seconds.\n",
        );
        out.push_str("# TYPE sgl_router_ttft_seconds histogram\n");
        let guard = self.ttft_seconds.lock();
        let mut models: Vec<&String> = guard.keys().collect();
        models.sort();
        for model_id in models {
            let hist = guard.get(model_id).unwrap();
            let label_body = format!("model_id=\"{}\"", escape_label(model_id));
            render_histogram(&mut out, "sgl_router_ttft_seconds", &label_body, hist);
        }
        drop(guard);

        // responses_total — edge, by route/method/status (incl. early-exit 400/413/503)
        out.push_str(
            "# HELP sgl_router_responses_total HTTP responses returned at the router edge, by route, method and status code (recorded at response-headers time, so a mid-stream failure after a 200 header counts as 200).\n",
        );
        out.push_str("# TYPE sgl_router_responses_total counter\n");
        let guard = self.responses_total.lock();
        let mut entries: Vec<(&EdgeResponseKey, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| {
            (&a.0.route, &a.0.method, a.0.status_code).cmp(&(
                &b.0.route,
                &b.0.method,
                b.0.status_code,
            ))
        });
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_responses_total{{route=\"{}\",method=\"{}\",status_code=\"{}\"}} {}\n",
                escape_label(&key.route),
                escape_label(&key.method),
                key.status_code,
                value,
            ));
        }
        drop(guard);

        // overlap_blocks histogram
        out.push_str(
            "# HELP sgl_router_overlap_blocks Overlap-block count observed at cache-aware-zmq policy selection.\n",
        );
        out.push_str("# TYPE sgl_router_overlap_blocks histogram\n");
        let guard = self.overlap_blocks.lock();
        let mut models: Vec<&String> = guard.keys().collect();
        models.sort();
        for model_id in models {
            let hist = guard.get(model_id).unwrap();
            let label_body = format!("model_id=\"{}\"", escape_label(model_id));
            render_histogram(&mut out, "sgl_router_overlap_blocks", &label_body, hist);
        }
        drop(guard);

        // active_load gauge
        out.push_str(
            "# HELP sgl_router_active_load Per-worker active load (prefill_tokens or decode_blocks).\n",
        );
        out.push_str("# TYPE sgl_router_active_load gauge\n");
        let guard = self.active_load.lock();
        let mut entries: Vec<(&ActiveLoadKey, i64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.worker_url, a.0.kind).cmp(&(&b.0.worker_url, b.0.kind)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_active_load{{worker_url=\"{}\",kind=\"{}\"}} {}\n",
                escape_label(&key.worker_url),
                key.kind,
                value,
            ));
        }
        drop(guard);

        // Worker gauges — sampled from the live registry snapshot passed in,
        // not stored. Rendering from the snapshot (rather than a pushed map)
        // means a removed worker stops emitting series on the very next
        // scrape instead of leaving a stale gauge pinned at its last value.

        // workers (pool size by mode). Emit all three modes so the series
        // exist (at 0) even before any worker of that mode is discovered.
        out.push_str("# HELP sgl_router_workers Registered workers by mode.\n");
        out.push_str("# TYPE sgl_router_workers gauge\n");
        for mode in ["plain", "prefill", "decode"] {
            let count = workers.iter().filter(|w| w.mode == mode).count();
            out.push_str(&format!(
                "sgl_router_workers{{mode=\"{}\"}} {}\n",
                mode, count,
            ));
        }

        // Sort the per-worker series by URL for stable output (tests + diffs).
        let mut sorted: Vec<&WorkerSnapshot> = workers.iter().collect();
        sorted.sort_by(|a, b| a.worker_url.cmp(&b.worker_url));

        // worker_health (1=breaker would admit a request, 0=breaker open)
        out.push_str(
            "# HELP sgl_router_worker_health Worker health: 1 = circuit breaker admits requests, 0 = rejecting (open within cooldown, or half-open with a probe in flight). May read 1 while sgl_router_worker_cb_state=1 (open but cooldown elapsed).\n",
        );
        out.push_str("# TYPE sgl_router_worker_health gauge\n");
        for w in &sorted {
            out.push_str(&format!(
                "sgl_router_worker_health{{worker_url=\"{}\"}} {}\n",
                escape_label(&w.worker_url),
                u8::from(w.healthy),
            ));
        }

        // worker_cb_state (0=closed, 1=open, 2=half_open)
        out.push_str(
            "# HELP sgl_router_worker_cb_state Circuit breaker state per worker (0=closed, 1=open, 2=half_open).\n",
        );
        out.push_str("# TYPE sgl_router_worker_cb_state gauge\n");
        for w in &sorted {
            out.push_str(&format!(
                "sgl_router_worker_cb_state{{worker_url=\"{}\"}} {}\n",
                escape_label(&w.worker_url),
                w.cb_state,
            ));
        }

        // worker_inflight_requests (in-flight request count per worker)
        out.push_str(
            "# HELP sgl_router_worker_inflight_requests In-flight requests currently dispatched to each worker.\n",
        );
        out.push_str("# TYPE sgl_router_worker_inflight_requests gauge\n");
        for w in &sorted {
            out.push_str(&format!(
                "sgl_router_worker_inflight_requests{{worker_url=\"{}\"}} {}\n",
                escape_label(&w.worker_url),
                w.inflight,
            ));
        }

        // stale_requests_total
        out.push_str(
            "# HELP sgl_router_stale_requests_total Total stale-request cancellations fired by the janitor.\n",
        );
        out.push_str("# TYPE sgl_router_stale_requests_total counter\n");
        let guard = self.stale_requests_total.lock();
        let mut entries: Vec<(&&str, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by_key(|e| *e.0);
        for (outcome, value) in entries {
            out.push_str(&format!(
                "sgl_router_stale_requests_total{{outcome=\"{}\"}} {}\n",
                outcome, value,
            ));
        }
        drop(guard);

        // decode_affinity_total
        out.push_str(
            "# HELP sgl_router_decode_affinity_total Decode-affinity outcomes from select_decode_with_affinity.\n",
        );
        out.push_str("# TYPE sgl_router_decode_affinity_total counter\n");
        let guard = self.decode_affinity_total.lock();
        let mut entries: Vec<(&&str, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by_key(|e| *e.0);
        for (outcome, value) in entries {
            out.push_str(&format!(
                "sgl_router_decode_affinity_total{{outcome=\"{}\"}} {}\n",
                outcome, value,
            ));
        }
        drop(guard);

        // sticky_total
        out.push_str(
            "# HELP sgl_router_sticky_total Sticky-session selection outcomes from StickyPolicy.\n",
        );
        out.push_str("# TYPE sgl_router_sticky_total counter\n");
        let guard = self.sticky_total.lock();
        let mut entries: Vec<(&&str, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by_key(|e| *e.0);
        for (outcome, value) in entries {
            out.push_str(&format!(
                "sgl_router_sticky_total{{outcome=\"{}\"}} {}\n",
                outcome, value,
            ));
        }
        drop(guard);

        // ingress_tokenize_errors_total
        out.push_str(
            "# HELP sgl_router_ingress_tokenize_errors_total Chat requests on a chat-encoder model whose ingress tokenization failed, silently falling back to engine-side tokenization (the input_ids offload was defeated).\n",
        );
        out.push_str("# TYPE sgl_router_ingress_tokenize_errors_total counter\n");
        let guard = self.ingress_tokenize_errors_total.lock();
        let mut entries: Vec<(&String, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (model_id, value) in entries {
            out.push_str(&format!(
                "sgl_router_ingress_tokenize_errors_total{{model_id=\"{}\"}} {}\n",
                escape_label(model_id),
                value,
            ));
        }
        drop(guard);

        // backpressure_rejected_total
        out.push_str(
            "# HELP sgl_router_backpressure_rejected_total Requests rejected with 503 because every worker was at its in-flight cap and the admission wait queue was full.\n",
        );
        out.push_str("# TYPE sgl_router_backpressure_rejected_total counter\n");
        let guard = self.backpressure_rejected_total.lock();
        let mut entries: Vec<(&String, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));
        for (model_id, value) in entries {
            out.push_str(&format!(
                "sgl_router_backpressure_rejected_total{{model_id=\"{}\"}} {}\n",
                escape_label(model_id),
                value,
            ));
        }
        drop(guard);

        // queued_requests (router-wide gauge, no labels)
        out.push_str(
            "# HELP sgl_router_queued_requests Requests currently parked in the admission wait queue (router-wide).\n",
        );
        out.push_str("# TYPE sgl_router_queued_requests gauge\n");
        out.push_str(&format!(
            "sgl_router_queued_requests {}\n",
            self.queued_requests.load(Ordering::Relaxed),
        ));

        // admission_wait_seconds histogram
        out.push_str(
            "# HELP sgl_router_admission_wait_seconds Time a request spent parked in the admission wait queue before admission, in seconds.\n",
        );
        out.push_str("# TYPE sgl_router_admission_wait_seconds histogram\n");
        let guard = self.admission_wait_seconds.lock();
        let mut models: Vec<&String> = guard.keys().collect();
        models.sort();
        for model_id in models {
            let hist = guard.get(model_id).unwrap();
            let label_body = format!("model_id=\"{}\"", escape_label(model_id));
            render_histogram(
                &mut out,
                "sgl_router_admission_wait_seconds",
                &label_body,
                hist,
            );
        }
        drop(guard);

        out
    }
}

/// Render one labelled histogram family (`<name>_bucket` / `_sum` /
/// `_count`) into `out`. `label_body` is the inside-of-braces label set
/// WITHOUT the trailing `le` (e.g. `model_id="tiny"`) and is
/// emitted verbatim — callers escape their own label values. Buckets are
/// rendered cumulatively per the Prometheus histogram contract, with a
/// final `+Inf` bucket.
fn render_histogram(out: &mut String, name: &str, label_body: &str, hist: &Histogram) {
    let mut cumulative: u64 = 0;
    for (i, &bound) in hist.bounds.iter().enumerate() {
        cumulative += hist.buckets[i];
        out.push_str(&format!(
            "{name}_bucket{{{label_body},le=\"{bound}\"}} {cumulative}\n"
        ));
    }
    cumulative += hist.buckets[hist.bounds.len()];
    out.push_str(&format!(
        "{name}_bucket{{{label_body},le=\"+Inf\"}} {cumulative}\n"
    ));
    out.push_str(&format!("{name}_sum{{{label_body}}} {}\n", hist.sum));
    out.push_str(&format!("{name}_count{{{label_body}}} {}\n", hist.count));
}

/// Prometheus label-value escape rule per
/// https://prometheus.io/docs/instrumenting/exposition_formats/.
/// We only escape `\`, `"`, and newline — the three characters the
/// reference parser rejects unescaped.
fn escape_label(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str(r"\\"),
            '"' => out.push_str(r#"\""#),
            '\n' => out.push_str(r"\n"),
            other => out.push(other),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_renders_only_help_lines() {
        let reg = MetricsRegistry::new();
        let out = reg.render();
        // Should at least carry HELP / TYPE for every metric family.
        assert!(out.contains("# TYPE sgl_router_requests_total counter"));
        assert!(out.contains("# TYPE sgl_router_worker_requests_total counter"));
        assert!(out.contains("# TYPE sgl_router_request_duration_seconds histogram"));
        assert!(out.contains("# TYPE sgl_router_ttft_seconds histogram"));
        assert!(out.contains("# TYPE sgl_router_responses_total counter"));
        assert!(out.contains("# TYPE sgl_router_overlap_blocks histogram"));
        assert!(out.contains("# TYPE sgl_router_active_load gauge"));
        assert!(out.contains("# TYPE sgl_router_workers gauge"));
        assert!(out.contains("# TYPE sgl_router_worker_health gauge"));
        assert!(out.contains("# TYPE sgl_router_worker_cb_state gauge"));
        assert!(out.contains("# TYPE sgl_router_worker_inflight_requests gauge"));
        assert!(out.contains("# TYPE sgl_router_stale_requests_total counter"));
        assert!(out.contains("# TYPE sgl_router_decode_affinity_total counter"));
        assert!(out.contains("# TYPE sgl_router_sticky_total counter"));
        assert!(out.contains("# TYPE sgl_router_ingress_tokenize_errors_total counter"));
        // Pool-size series exist (at 0) for all three modes even with no
        // workers, so dashboards have a stable series to graph.
        assert!(out.contains(r#"sgl_router_workers{mode="plain"} 0"#));
        assert!(out.contains(r#"sgl_router_workers{mode="prefill"} 0"#));
        assert!(out.contains(r#"sgl_router_workers{mode="decode"} 0"#));
    }

    #[test]
    fn observe_request_duration_writes_buckets_sum_and_count() {
        let reg = MetricsRegistry::new();
        // 25 ms, 120 ms, 600 ms for model "tiny".
        reg.observe_request_duration("tiny", 0.025);
        reg.observe_request_duration("tiny", 0.12);
        reg.observe_request_duration("tiny", 0.6);
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_request_duration_seconds_count{model_id="tiny"} 3"#),
            "expected count=3; got:\n{out}",
        );
        // 0.025 <= 0.025, so the le=0.025 bucket is 1 (cumulative).
        assert!(
            out.contains(
                r#"sgl_router_request_duration_seconds_bucket{model_id="tiny",le="0.025"} 1"#
            ),
            "expected le=0.025 bucket = 1; got:\n{out}",
        );
        // le=1 is cumulative over all three observations.
        assert!(
            out.contains(r#"sgl_router_request_duration_seconds_bucket{model_id="tiny",le="1"} 3"#),
            "expected le=1 bucket = 3; got:\n{out}",
        );
        assert!(out.contains(
            r#"sgl_router_request_duration_seconds_bucket{model_id="tiny",le="+Inf"} 3"#
        ));
    }

    #[test]
    fn request_duration_separates_by_model() {
        let reg = MetricsRegistry::new();
        reg.observe_request_duration("a", 0.01);
        reg.observe_request_duration("b", 0.01);
        let out = reg.render();
        assert!(out.contains(r#"sgl_router_request_duration_seconds_count{model_id="a"} 1"#));
        assert!(out.contains(r#"sgl_router_request_duration_seconds_count{model_id="b"} 1"#));
    }

    #[test]
    fn request_duration_overflow_lands_in_plus_inf_bucket_only() {
        let reg = MetricsRegistry::new();
        // 45s is beyond the top finite bound (30s) — the operationally
        // critical "outlived the upstream timeout" case the +Inf bucket exists
        // for. It must NOT appear in le="30" but must be in le="+Inf"/_count,
        // and _sum must reflect the full value.
        reg.observe_request_duration("m", 45.0);
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_request_duration_seconds_bucket{model_id="m",le="30"} 0"#),
            "45s must not fall in the le=30 bucket; got:\n{out}",
        );
        assert!(
            out.contains(r#"sgl_router_request_duration_seconds_bucket{model_id="m",le="+Inf"} 1"#)
        );
        assert!(out.contains(r#"sgl_router_request_duration_seconds_count{model_id="m"} 1"#));
        assert!(out.contains(r#"sgl_router_request_duration_seconds_sum{model_id="m"} 45"#));
    }

    #[test]
    fn observe_request_duration_ignores_non_finite() {
        let reg = MetricsRegistry::new();
        reg.observe_request_duration("m", f64::NAN);
        reg.observe_request_duration("m", f64::INFINITY);
        let out = reg.render();
        // Nothing recorded — no count series for the model (sum stays uncorrupted).
        assert!(
            !out.contains(r#"sgl_router_request_duration_seconds_count{model_id="m"}"#),
            "non-finite observations must be dropped, not bucketed; got:\n{out}",
        );
    }

    #[test]
    fn observe_ttft_writes_buckets_sum_and_count() {
        let reg = MetricsRegistry::new();
        reg.observe_ttft("tiny", 0.04);
        reg.observe_ttft("tiny", 0.2);
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_ttft_seconds_count{model_id="tiny"} 2"#),
            "expected ttft count=2; got:\n{out}",
        );
        // 0.04 <= 0.05, so the le=0.05 bucket is 1 (cumulative).
        assert!(
            out.contains(r#"sgl_router_ttft_seconds_bucket{model_id="tiny",le="0.05"} 1"#),
            "expected le=0.05 bucket = 1; got:\n{out}",
        );
        // le=0.2 (an engine-aligned edge) is cumulative over both observations.
        assert!(out.contains(r#"sgl_router_ttft_seconds_bucket{model_id="tiny",le="0.2"} 2"#));
    }

    #[test]
    fn ttft_buckets_align_with_engine_grid() {
        // The engine's `sglang:time_to_first_token_seconds` edges from 0.1 s up.
        // These MUST all appear verbatim in the router's TTFT histogram, else a
        // `histogram_quantile` comparison silently interpolates on mismatched
        // grids. The sub-100 ms head (0.005..0.05) is router-only and not
        // asserted here.
        let reg = MetricsRegistry::new();
        reg.observe_ttft("m", 0.5);
        let out = reg.render();
        for le in [
            "0.1", "0.2", "0.4", "0.6", "0.8", "1", "2", "4", "6", "8", "10", "20", "40", "60",
            "80", "100", "200", "400",
        ] {
            assert!(
                out.contains(&format!(
                    r#"sgl_router_ttft_seconds_bucket{{model_id="m",le="{le}"}}"#
                )),
                "missing engine-aligned TTFT bucket le={le}; got:\n{out}",
            );
        }
    }

    #[test]
    fn record_response_counts_by_route_method_status_code() {
        let reg = MetricsRegistry::new();
        reg.record_response("/v1/chat/completions", "POST", 200);
        reg.record_response("/v1/chat/completions", "POST", 200);
        reg.record_response("/v1/chat/completions", "POST", 502);
        reg.record_response("/v1/models", "GET", 200);
        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="200"} 2"#
        ));
        assert!(out.contains(
            r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="502"} 1"#
        ));
        assert!(out.contains(
            r#"sgl_router_responses_total{route="/v1/models",method="GET",status_code="200"} 1"#
        ));
    }

    #[test]
    fn record_ingress_counts_by_route_method() {
        let reg = MetricsRegistry::new();
        reg.record_ingress("/v1/chat/completions", "POST");
        reg.record_ingress("/v1/chat/completions", "POST");
        reg.record_ingress("/v1/models", "GET");
        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_requests_total{route="/v1/chat/completions",method="POST"} 2"#
        ));
        assert!(out.contains(r#"sgl_router_requests_total{route="/v1/models",method="GET"} 1"#));
    }

    #[test]
    fn render_with_workers_emits_per_worker_gauges_and_pool_size() {
        let reg = MetricsRegistry::new();
        let workers = vec![
            WorkerSnapshot {
                worker_url: "http://p0:30000".into(),
                mode: "prefill",
                healthy: true,
                cb_state: 0,
                inflight: 5,
            },
            WorkerSnapshot {
                worker_url: "http://d0:30000".into(),
                mode: "decode",
                healthy: false,
                cb_state: 1,
                inflight: 0,
            },
        ];
        let out = reg.render_with_workers(&workers);
        // Pool size by mode.
        assert!(out.contains(r#"sgl_router_workers{mode="prefill"} 1"#));
        assert!(out.contains(r#"sgl_router_workers{mode="decode"} 1"#));
        assert!(out.contains(r#"sgl_router_workers{mode="plain"} 0"#));
        // Health: healthy prefill = 1, unhealthy decode = 0.
        assert!(out.contains(r#"sgl_router_worker_health{worker_url="http://p0:30000"} 1"#));
        assert!(out.contains(r#"sgl_router_worker_health{worker_url="http://d0:30000"} 0"#));
        // Circuit breaker state codes.
        assert!(out.contains(r#"sgl_router_worker_cb_state{worker_url="http://p0:30000"} 0"#));
        assert!(out.contains(r#"sgl_router_worker_cb_state{worker_url="http://d0:30000"} 1"#));
        // In-flight request counts.
        assert!(
            out.contains(r#"sgl_router_worker_inflight_requests{worker_url="http://p0:30000"} 5"#)
        );
        assert!(
            out.contains(r#"sgl_router_worker_inflight_requests{worker_url="http://d0:30000"} 0"#)
        );
    }

    #[test]
    fn render_without_workers_emits_no_per_worker_series() {
        let reg = MetricsRegistry::new();
        let out = reg.render();
        // Headers present, but no per-worker series lines.
        assert!(out.contains("# TYPE sgl_router_worker_health gauge"));
        assert!(!out.contains("sgl_router_worker_health{"));
        assert!(!out.contains("sgl_router_worker_cb_state{"));
        assert!(!out.contains("sgl_router_worker_inflight_requests{"));
    }

    #[test]
    fn record_worker_request_emits_labelled_counter_line() {
        let reg = MetricsRegistry::new();
        reg.record_worker_request(
            "http://worker-a:30000",
            "tiny",
            WorkerModeLabel::Prefill,
            RequestOutcome::Success,
        );
        reg.record_worker_request(
            "http://worker-a:30000",
            "tiny",
            WorkerModeLabel::Prefill,
            RequestOutcome::Success,
        );
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_worker_requests_total{worker_url="http://worker-a:30000",model_id="tiny",mode="prefill",outcome="success"} 2"#),
            "render did not include the expected counter line; got:\n{out}",
        );
    }

    #[test]
    fn observe_overlap_blocks_writes_buckets_and_count() {
        let reg = MetricsRegistry::new();
        reg.observe_overlap_blocks("tiny", 3);
        reg.observe_overlap_blocks("tiny", 9);
        reg.observe_overlap_blocks("tiny", 50);
        let out = reg.render();
        // 3 observations -> count=3, sum=62
        assert!(out.contains(r#"sgl_router_overlap_blocks_count{model_id="tiny"} 3"#));
        assert!(out.contains(r#"sgl_router_overlap_blocks_sum{model_id="tiny"} 62"#));
        // The le=64 bucket is cumulative: 3 is <=4, 9 is <=16, 50 is <=64.
        assert!(
            out.contains(r#"sgl_router_overlap_blocks_bucket{model_id="tiny",le="64"} 3"#),
            "bucket le=64 should be 3 (cumulative); got:\n{out}",
        );
        // The le=4 bucket should include only the 3.
        assert!(
            out.contains(r#"sgl_router_overlap_blocks_bucket{model_id="tiny",le="4"} 1"#),
            "bucket le=4 should be 1; got:\n{out}",
        );
    }

    #[test]
    fn set_active_load_gauge_overwrites() {
        let reg = MetricsRegistry::new();
        reg.set_active_load("http://w:30000", ActiveLoadKind::PrefillTokens, 100);
        reg.set_active_load("http://w:30000", ActiveLoadKind::PrefillTokens, 250);
        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_active_load{worker_url="http://w:30000",kind="prefill_tokens"} 250"#,
        ));
        // First write must NOT appear.
        assert!(!out.contains(
            r#"sgl_router_active_load{worker_url="http://w:30000",kind="prefill_tokens"} 100"#,
        ));
    }

    #[test]
    fn stale_request_counter_increments() {
        let reg = MetricsRegistry::new();
        reg.record_stale_request(StaleRequestOutcome::Expired);
        reg.record_stale_request(StaleRequestOutcome::Expired);
        reg.record_stale_request(StaleRequestOutcome::Expired);
        let out = reg.render();
        assert!(out.contains(r#"sgl_router_stale_requests_total{outcome="expired"} 3"#));
    }

    #[test]
    fn decode_affinity_counter_emits_three_outcomes() {
        let reg = MetricsRegistry::new();
        reg.record_decode_affinity(DecodeAffinityOutcome::SameHostPicked);
        reg.record_decode_affinity(DecodeAffinityOutcome::SameHostPicked);
        reg.record_decode_affinity(DecodeAffinityOutcome::FallbackBreaker);
        reg.record_decode_affinity(DecodeAffinityOutcome::FallbackLoadImbalance);
        let out = reg.render();
        assert!(out.contains(r#"sgl_router_decode_affinity_total{outcome="same_host_picked"} 2"#));
        assert!(out.contains(r#"sgl_router_decode_affinity_total{outcome="fallback_breaker"} 1"#));
        assert!(out
            .contains(r#"sgl_router_decode_affinity_total{outcome="fallback_load_imbalance"} 1"#,));
    }

    #[test]
    fn sticky_counter_emits_all_outcomes() {
        let reg = MetricsRegistry::new();
        reg.record_sticky(StickyOutcome::Hit);
        reg.record_sticky(StickyOutcome::Hit);
        reg.record_sticky(StickyOutcome::Assigned);
        reg.record_sticky(StickyOutcome::Remap);
        reg.record_sticky(StickyOutcome::NoRoutingKey);
        let out = reg.render();
        assert!(out.contains(r#"sgl_router_sticky_total{outcome="hit"} 2"#));
        assert!(out.contains(r#"sgl_router_sticky_total{outcome="assigned"} 1"#));
        assert!(out.contains(r#"sgl_router_sticky_total{outcome="remap"} 1"#));
        assert!(out.contains(r#"sgl_router_sticky_total{outcome="no_routing_key"} 1"#));
    }

    #[test]
    fn backpressure_rejected_counter_increments_per_model() {
        let reg = MetricsRegistry::new();
        reg.record_backpressure_rejected("tiny");
        reg.record_backpressure_rejected("tiny");
        reg.record_backpressure_rejected("other");
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_backpressure_rejected_total{model_id="tiny"} 2"#),
            "{out}",
        );
        assert!(
            out.contains(r#"sgl_router_backpressure_rejected_total{model_id="other"} 1"#),
            "{out}",
        );
    }

    #[test]
    fn queued_requests_gauge_tracks_balanced_deltas() {
        let reg = MetricsRegistry::new();
        reg.add_queued_requests(1);
        reg.add_queued_requests(1);
        reg.add_queued_requests(-1);
        let out = reg.render();
        assert!(
            out.contains("# TYPE sgl_router_queued_requests gauge"),
            "{out}"
        );
        assert!(out.contains("sgl_router_queued_requests 1\n"), "{out}");
    }

    #[test]
    fn admission_wait_histogram_writes_buckets_sum_and_count() {
        let reg = MetricsRegistry::new();
        reg.observe_admission_wait("tiny", 0.3);
        reg.observe_admission_wait("tiny", 1.5);
        let out = reg.render();
        assert!(
            out.contains("# TYPE sgl_router_admission_wait_seconds histogram"),
            "{out}"
        );
        assert!(
            out.contains(r#"sgl_router_admission_wait_seconds_count{model_id="tiny"} 2"#),
            "{out}",
        );
    }

    #[test]
    fn ingress_tokenize_error_counter_increments_per_model() {
        let reg = MetricsRegistry::new();
        reg.record_ingress_tokenize_error("tiny");
        reg.record_ingress_tokenize_error("tiny");
        reg.record_ingress_tokenize_error("other");
        let out = reg.render();
        assert!(
            out.contains(r#"sgl_router_ingress_tokenize_errors_total{model_id="tiny"} 2"#),
            "expected tiny=2; got:\n{out}",
        );
        assert!(
            out.contains(r#"sgl_router_ingress_tokenize_errors_total{model_id="other"} 1"#),
            "expected other=1; got:\n{out}",
        );
    }

    #[test]
    fn ingress_tokenize_error_absent_until_recorded() {
        // Healthy operation never calls the recorder, so no per-model series
        // should exist — only the HELP/TYPE headers.
        let reg = MetricsRegistry::new();
        let out = reg.render();
        assert!(out.contains("# TYPE sgl_router_ingress_tokenize_errors_total counter"));
        assert!(
            !out.contains("sgl_router_ingress_tokenize_errors_total{"),
            "no per-model series until an error is recorded; got:\n{out}",
        );
    }

    #[test]
    fn label_values_escape_quotes_and_backslashes() {
        let reg = MetricsRegistry::new();
        reg.record_worker_request(
            r#"http://"weird":30000"#,
            r"back\slash",
            WorkerModeLabel::Plain,
            RequestOutcome::Error,
        );
        let out = reg.render();
        assert!(
            out.contains(r#"worker_url="http://\"weird\":30000""#),
            "render did not escape double-quote; got:\n{out}",
        );
        assert!(
            out.contains(r#"model_id="back\\slash""#),
            "render did not escape backslash; got:\n{out}",
        );
    }

    #[test]
    fn histogram_plus_inf_bucket_catches_overflow() {
        let reg = MetricsRegistry::new();
        // 8001 is just above the last finite bucket (8000); it should land
        // in +Inf only.
        reg.observe_overlap_blocks("m", 8001);
        let out = reg.render();
        assert!(out.contains(r#"sgl_router_overlap_blocks_bucket{model_id="m",le="8000"} 0"#));
        assert!(out.contains(r#"sgl_router_overlap_blocks_bucket{model_id="m",le="+Inf"} 1"#));
    }
}
