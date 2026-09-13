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
//! | `sgl_router_responses_total` | Counter | `route`, `method`, `status_code` |
//! | `sgl_router_worker_requests_total` | Counter | `worker_url`, `model_id`, `mode`, `outcome` |
//! | `sgl_router_request_duration_seconds` | Histogram | `model_id` |
//! | `sgl_router_ttft_seconds` | Histogram | `model_id` |
//! | `sgl_router_active_load` | Gauge | `worker_url`, `kind` |
//! | `sgl_router_workers` | Gauge | `mode` |
//! | `sgl_router_worker_health` | Gauge | `worker_url` |
//! | `sgl_router_worker_cb_state` | Gauge | `worker_url` |
//! | `sgl_router_worker_inflight_requests` | Gauge | `worker_url` |
//! | `sgl_router_stale_requests_total` | Counter | `outcome` |
//! | `sgl_router_decode_affinity_total` | Counter | `outcome` |
//! | `sgl_router_sticky_total` | Counter | `outcome` |
//! | `sgl_router_policy_decisions_total` | Counter | `policy`, `reason` |
//! | `sgl_router_policy_selection_failures_total` | Counter | `policy`, `reason` |
//! | `sgl_router_cache_admission_evaluated_total` | Counter | — |
//! | `sgl_router_cache_admission_rejected_total` | Counter | — |
//! | `sgl_router_cache_pressure_guard_compared_total` | Counter | — |
//! | `sgl_router_cache_pressure_guard_override_total` | Counter | — |
//! | `sgl_router_cache_monitor_decisions_total` | Counter | `source` |
//! | `sgl_router_cache_aware_decisions_total` | Counter | `model_id`, `decision` |
//! | `sgl_router_cache_aware_query_blocks_total` | Counter | `model_id`, `decision` |
//! | `sgl_router_matched_overlap_blocks_total` | Counter | `model_id`, `decision` |
//! | `sgl_router_selected_overlap_blocks_total` | Counter | `model_id`, `decision` |
//! | `sgl_router_overlap_blocks` | Histogram | `model_id` |
//! | `sgl_router_selected_owner_tier_total` | Counter | `model_id`, `tier` |
//! | `sgl_router_zero_match_block0_total` | Counter | `model_id`, `presence` |
//! | `sgl_router_ingress_tokenize_errors_total` | Counter | `model_id` |
//!
//! # Reading cache locality against the engine
//!
//! The `query`/`matched`/`selected` block counters decompose the router's view
//! of cache locality into terms that can be subtracted. The ratio to compare
//! against the engine's
//! `sglang:cached_tokens_total / sglang:prompt_tokens_total` is
//! `selected / query`, NOT `matched / query` — the latter is the fleet-wide
//! best and reads structurally high, because the router meters the deepest
//! prefix ANYONE holds even on the selections where it then routed somewhere
//! else. `matched - selected` is locality the routing decision gave up, and it
//! is attributable to a decision bucket because all three counters share the
//! `(model_id, decision)` key.
//!
//! The residual between `selected / query` and the engine's number is NOT
//! one-directional. It runs high when cache state changed between selection
//! and prefill (the engine evicted what the tree still lists), and low when a
//! worker serves traffic while publishing no KV events — a `/server_info`
//! probe that failed at registration, or a page-size disagreement, both of
//! which leave a worker fully routable and permanently absent from the tree.
//! Such a worker reads `selected = 0` on every selection that lands on it,
//! indistinguishable from one that genuinely holds nothing, so a
//! `selected / query` sitting *below* the engine's hit rate points at the
//! subscriber fleet rather than at eviction.
//!
//! Block counts convert to the engine's token units by multiplying by
//! `sgl_router_kv_block_size`. The query-block denominator rounds a partial
//! trailing block up to a whole one, so per request it can overstate the
//! engine's token count by up to one block less a token — averaging half a
//! block on uniformly distributed lengths, which is where the two denominators
//! agree in aggregate.
//!
//! The four `sgl_router_worker*` gauges and `sgl_router_workers` are sampled
//! at scrape time from the live [`crate::workers::WorkerRegistry`] (passed to
//! [`MetricsRegistry::render_with_workers`]) rather than pushed — there is no
//! health-check loop to push from, and pull-on-scrape means a removed worker
//! stops emitting series immediately instead of leaving a stale gauge.
//!
//! The exposition is text/plain; version=0.0.4 per the Prometheus spec.

use crate::config::PolicyKind;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

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

/// Histogram bucket upper bounds for `sgl_router_overlap_blocks`. Blocks are
/// 16-64 tokens each and `MAX_CHAT_BODY_BYTES` bounds context length, putting
/// the practical ceiling for a maximum-length context in the low tens of
/// thousands of blocks. The ladder spans 0 -> ~8k blocks at the resolution
/// worth charting; `+Inf` catches the longer-context tail. The leading `0.0`
/// edge is load-bearing: a zero-overlap selection is the most common outcome
/// on an unhealthy fleet, and without its own bucket it would be
/// indistinguishable from a one-block match.
const OVERLAP_BLOCKS_BUCKETS: &[f64] = &[
    0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1000.0, 2000.0, 4000.0, 8000.0,
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

#[derive(Debug, Clone, Copy)]
pub(crate) enum PolicySelectionFailureReason {
    PrefillAdmissionExhausted,
    CacheCandidatesExhausted,
    ProposalEmpty,
}

impl PolicySelectionFailureReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::PrefillAdmissionExhausted => "prefill_admission_exhausted",
            Self::CacheCandidatesExhausted => "cache_candidates_exhausted",
            Self::ProposalEmpty => "proposal_empty",
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

/// Terminal outcome of one cache-aware prefill selection. Exactly one is
/// recorded per request routed by the cache-aware policy that resolved a
/// worker, so the labels sum to the cache-aware request rate less the
/// selections that ended in a 503.
///
/// Most of those 503s are in `sgl_router_policy_selection_failures_total`,
/// but not all: an out-of-process indexer that rejects the query contract
/// raises the same error without going through the recorder, so reconciling
/// the two counters can leave a small unexplained gap on indexer deployments.
/// Do not treat their sum as a closed accounting of cache-aware requests.
///
/// The vocabulary is deliberately the set of outcomes the routing path can
/// tell apart TODAY. Where the policy collapses a distinction — a tree miss,
/// a match that failed `cache_affinity_min_matched_tokens`, and an owner that
/// is not a live worker all return the same `None` from
/// `CacheAwarePolicy::propose_prefill` — this collapses it too rather than
/// reaching into the policy for a finer split.
/// `sgl_router_zero_match_block0_total` is the finer split that does not need
/// one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheAwareDecision {
    /// A prefix owner won the selection.
    CacheHit,
    /// The lookup ran and produced no usable candidate: nothing matched,
    /// nothing cleared the cache gate, or no owner was a live worker. The
    /// request fell back to the ordinary min-load path.
    NoCandidates,
    /// The lookup produced candidates and hard admission or the pressure
    /// guard rejected every one of them, so locality was given up to
    /// capacity. Distinct from [`Self::NoCandidates`] because the fix is
    /// different: this one is a sizing problem, that one a cache problem.
    CandidatesExhausted,
    /// No prefix lookup happened at all — no routing tokens, no block size
    /// established from the fleet yet, or no prefix provider configured. A
    /// router that has just started reads this until the first worker reports
    /// its page size, which is the honest answer to "why is my hit rate 0".
    LookupUnavailable,
}

impl CacheAwareDecision {
    fn as_str(self) -> &'static str {
        match self {
            Self::CacheHit => "cache_hit",
            Self::NoCandidates => "no_candidates",
            Self::CandidatesExhausted => "candidates_exhausted",
            Self::LookupUnavailable => "lookup_unavailable",
        }
    }
}

/// One selection's locality triple as the call site computes it, before it is
/// booked. A struct rather than three `u64` arguments because the three are
/// mutually constrained and swapping two of them at a call site would be
/// silent — and would render as a hit rate above 100%.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalityBlocks {
    /// Blocks the request hashed to — the denominator.
    pub query: u64,
    /// Blocks the deepest holder in the fleet has.
    pub matched: u64,
    /// Blocks the worker this request was actually sent to has.
    pub selected: u64,
}

/// The block counts one cache-aware selection contributes, held per
/// `(model_id, decision)`.
///
/// The three answer three different questions about the same request, and the
/// differences between them are the point:
///
/// * `query` — how much prefix the request has. The denominator.
/// * `matched` — how much of it the *best* worker in the fleet holds. The
///   ceiling: what a router with no load constraints could have reused.
/// * `selected` — how much of it the worker actually picked holds. What the
///   engine should be able to reuse, modulo eviction between now and prefill.
///
/// They nest: `selected <= matched <= query`, per booking and therefore
/// (addition preserves it) for the totals. [`Self::record`] asserts it in
/// debug builds — every ratio charted off these counters assumes it, and a
/// violation renders as a hit rate above 100% or a negative loss bar, which is
/// worth catching at the booking site rather than in Grafana.
#[derive(Debug, Default)]
struct CacheAwareBlocks {
    query: AtomicU64,
    matched: AtomicU64,
    selected: AtomicU64,
}

impl CacheAwareBlocks {
    /// Add one selection's contribution to all three counters.
    ///
    /// Not atomic as a group: a scrape landing between the adds sees a partial
    /// booking. That is fine for counters read through `rate()` over any
    /// multi-scrape window — the increments are never lost, only briefly
    /// displaced. What the shared key DOES buy is that a numerator can never
    /// be attributed to a different decision than its denominator.
    fn record(&self, query: u64, matched: u64, selected: u64) {
        debug_assert!(
            selected <= matched && matched <= query,
            "locality decomposition must nest: selected {selected} <= matched {matched} <= query {query}",
        );
        self.query.fetch_add(query, Ordering::Relaxed);
        self.matched.fetch_add(matched, Ordering::Relaxed);
        self.selected.fetch_add(selected, Ordering::Relaxed);
    }
}

/// The shared metrics registry, held on `AppContext`. Cheap to clone — all
/// internal state is `Arc`/`Atomic`/`Mutex`-protected.
#[derive(Debug, Default)]
pub struct MetricsRegistry {
    // Edge counters (recorded at the app.rs middleware): intake at entry,
    // responses at exit. `requests_total - responses_total` = received but
    // never answered, which `worker_requests_total` (post-dispatch) can't see.
    requests_total: Mutex<HashMap<EdgeKey, Arc<AtomicU64>>>,
    responses_total: Mutex<HashMap<EdgeResponseKey, Arc<AtomicU64>>>,
    // Per-worker dispatch outcomes (formerly `requests_total`). Recorded after
    // dispatch, so blind to pre-dispatch drops; kept per-worker for the
    // routing-convergence tests.
    worker_requests_total: Mutex<HashMap<RequestKey, Arc<AtomicU64>>>,
    // Keyed by `model_id` only: a model's pool is either all-plain or all-PD
    // (the registry rejects mixed pools), so the worker `mode` would be a pure
    // function of `model_id` here — a redundant label. Per-worker `mode` lives
    // on `worker_requests_total` / the worker gauges instead.
    request_duration: Mutex<HashMap<String, Histogram>>,
    ttft_seconds: Mutex<HashMap<String, Histogram>>,
    active_load: Mutex<HashMap<ActiveLoadKey, Arc<AtomicI64>>>,
    stale_requests_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
    decode_affinity_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
    sticky_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
    policy_decisions_total: Mutex<HashMap<PolicyDecisionKey, Arc<AtomicU64>>>,
    policy_selection_failures_total: Mutex<HashMap<PolicyDecisionKey, Arc<AtomicU64>>>,
    cache_admission_evaluated_total: AtomicU64,
    cache_admission_rejected_total: AtomicU64,
    cache_pressure_guard_compared_total: AtomicU64,
    cache_pressure_guard_override_total: AtomicU64,
    cache_monitor_decisions_total: Mutex<HashMap<&'static str, Arc<AtomicU64>>>,
    cache_aware_decisions_total: Mutex<HashMap<CacheAwareDecisionKey, Arc<AtomicU64>>>,
    /// The three block counters that decompose cache-aware locality, kept in
    /// one map value so a selection books all of them against one key and they
    /// can never disagree about which decision they belong to. Rendered as
    /// three metric families sharing `(model_id, decision)`.
    cache_aware_blocks: Mutex<HashMap<CacheAwareDecisionKey, Arc<CacheAwareBlocks>>>,
    overlap_blocks: Mutex<HashMap<String, Histogram>>,
    selected_owner_tier_total: Mutex<HashMap<SelectedTierKey, Arc<AtomicU64>>>,
    zero_match_block0_total: Mutex<HashMap<ZeroMatchKey, Arc<AtomicU64>>>,
    ingress_tokenize_errors_total: Mutex<HashMap<String, Arc<AtomicU64>>>,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct RequestKey {
    worker_url: String,
    model_id: String,
    mode: &'static str,
    outcome: &'static str,
}

/// Labels for the edge `requests_total` (intake) counter. `route` is the matched
/// template (small fixed set), so cardinality is bounded.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct EdgeKey {
    route: String,
    method: String,
}

/// Labels for the edge `responses_total` counter: `EdgeKey` + final HTTP status.
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

/// Key shared by `cache_aware_decisions_total` and the three block counters,
/// so a numerator and its denominator always carry the same labels.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct CacheAwareDecisionKey {
    model_id: String,
    decision: &'static str,
}

/// Key for [`MetricsRegistry::record_selected_owner_tier`]. `tier` is a
/// `Tiers::SLOTS` label, `none`, or `unknown` — see that method.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct SelectedTierKey {
    model_id: String,
    tier: &'static str,
}

/// Key for [`MetricsRegistry::record_zero_match_block0`]. `presence` is
/// `in_tree` or `absent` — see that method.
#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct ZeroMatchKey {
    model_id: String,
    presence: &'static str,
}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
struct PolicyDecisionKey {
    policy: String,
    reason: String,
}

#[derive(Debug)]
struct Histogram {
    /// Bucket upper bounds this histogram observes against. Held per-instance
    /// so a single `Histogram` type backs metrics with different bucket ladders.
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

    /// Bump the edge intake counter `requests_total{route,method}`. Called at the
    /// middleware before worker pick, so it sees pre-dispatch drops.
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

    /// Bump `worker_requests_total`. Recorded after dispatch — see `record_ingress`
    /// for true intake.
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

    /// Bump the edge counter `responses_total{route,method,status_code}`. Called
    /// at the middleware, so it captures every outcome — incl. early-exit
    /// 400/413/503 that the old per-handler site skipped.
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

    /// Record the final Prefill policy decision.
    pub fn record_policy_decision(&self, policy: &str, reason: &str) {
        let key = PolicyDecisionKey {
            policy: policy.to_owned(),
            reason: reason.to_owned(),
        };
        let mut guard = self.policy_decisions_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_policy_selection_failure(
        &self,
        policy: PolicyKind,
        reason: PolicySelectionFailureReason,
    ) {
        let key = PolicyDecisionKey {
            policy: policy.to_string(),
            reason: reason.as_str().to_owned(),
        };
        let mut guard = self.policy_selection_failures_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Cache-Aware candidates evaluated by hard admission.
    pub fn record_cache_admission_evaluations(&self, count: u64) {
        self.cache_admission_evaluated_total
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Cache-Aware candidates rejected by hard admission.
    pub fn record_cache_admission_rejections(&self, count: u64) {
        self.cache_admission_rejected_total
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Pressure-guard pairs compared and overridden with complete monitor data.
    pub fn record_cache_pressure_guard(&self, compared: u64, overrides: u64) {
        self.cache_pressure_guard_compared_total
            .fetch_add(compared, Ordering::Relaxed);
        self.cache_pressure_guard_override_total
            .fetch_add(overrides, Ordering::Relaxed);
    }

    /// Load source used for a Cache-Aware decision. Benchmarks reject
    /// `router_local` results to verify that monitor data affected selection.
    pub fn record_cache_monitor_decision(&self, source: &'static str) {
        let mut guard = self.cache_monitor_decisions_total.lock();
        let counter = guard
            .entry(source)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Book one cache-aware selection: its terminal decision, and — when the
    /// lookup ran — the query / fleet-best / selected block triple under the
    /// same `(model_id, decision)` key.
    ///
    /// `blocks` is `None` for a selection with no lookup to attribute
    /// (`lookup_unavailable`), which is why
    /// `sgl_router_cache_aware_decisions_total` is the larger of the two
    /// totals and the block counters must never be used as a request count.
    ///
    /// Panics in debug builds if the triple does not nest — see
    /// [`CacheAwareBlocks`].
    pub fn record_cache_aware_selection(
        &self,
        model_id: &str,
        decision: CacheAwareDecision,
        blocks: Option<LocalityBlocks>,
    ) {
        let key = CacheAwareDecisionKey {
            model_id: model_id.to_owned(),
            decision: decision.as_str(),
        };
        let mut guard = self.cache_aware_decisions_total.lock();
        let counter = guard
            .entry(key.clone())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);

        let Some(blocks) = blocks else {
            return;
        };
        let mut guard = self.cache_aware_blocks.lock();
        let entry = guard.entry(key).or_default().clone();
        drop(guard);
        entry.record(blocks.query, blocks.matched, blocks.selected);

        let mut guard = self.overlap_blocks.lock();
        let hist = guard
            .entry(model_id.to_owned())
            .or_insert_with(|| Histogram::new(OVERLAP_BLOCKS_BUCKETS));
        hist.observe(blocks.matched as f64);
    }

    /// Book the cheapest storage tier on which the SELECTED worker holds this
    /// request's matched prefix, in `Tiers::SLOTS` vocabulary (`device`,
    /// `host`, `disk`, `external`) so it joins with
    /// `sgl_router_kv_tree_blocks`.
    ///
    /// Two labels are not tiers. `none` means the chosen worker holds no part
    /// of this prefix — every `no_candidates` selection, and any hit whose
    /// winner is not the owner. `unknown` means the tier is not observable on
    /// this deployment: the out-of-process indexer's wire contract carries no
    /// medium, so an indexer fleet reads `unknown` on every selection rather
    /// than a fabricated `device`.
    ///
    /// `host / (device + host)` is the fraction of affinity routes that count
    /// on a load-back rather than an in-place hit — the number that says
    /// whether the host tier is carrying the fleet.
    pub fn record_selected_owner_tier(&self, model_id: &str, tier: &'static str) {
        let key = SelectedTierKey {
            model_id: model_id.to_owned(),
            tier,
        };
        let mut guard = self.selected_owner_tier_total.lock();
        let counter = guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone();
        drop(guard);
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Split a ZERO-overlap cache-aware selection by whether the request's
    /// FIRST queried block hash exists anywhere in the KV tree.
    ///
    /// A zero match is the most common outcome on an unhealthy fleet and is
    /// also completely ambiguous. `absent` means no node carries that hash:
    /// the engines never published the block, or published and then removed it
    /// — an engine-side gap. `in_tree` means the hash IS carried but the
    /// root-anchored walk could not reach it — a linkage problem on the router
    /// side. The two call for work in different repositories, and a zero
    /// `matched` alone cannot tell them apart.
    ///
    /// Recorded only for a zero-overlap selection whose provider can answer
    /// (the in-process tree), so it is a strict subset of
    /// `sgl_router_cache_aware_decisions_total` and never exceeds the zero
    /// bucket of `sgl_router_overlap_blocks`.
    pub fn record_zero_match_block0(&self, model_id: &str, present_in_tree: bool) {
        let key = ZeroMatchKey {
            model_id: model_id.to_owned(),
            presence: if present_in_tree { "in_tree" } else { "absent" },
        };
        let mut guard = self.zero_match_block0_total.lock();
        let counter = guard
            .entry(key)
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

        // requests_total — edge intake (every request, counted before dispatch)
        out.push_str(
            "# HELP sgl_router_requests_total Total requests received at the router HTTP edge, counted before worker dispatch (true intake).\n",
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

        // worker_requests_total — per-worker dispatch outcomes (formerly requests_total)
        out.push_str(
            "# HELP sgl_router_worker_requests_total Chat-completions requests dispatched to a worker, by dispatch outcome.\n",
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
            "# HELP sgl_router_responses_total Responses returned at the router HTTP edge, by route, method and HTTP status code.\n",
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

        // policy_decisions_total
        out.push_str(
            "# HELP sgl_router_policy_decisions_total Final Prefill policy decisions by policy and bounded reason.\n",
        );
        out.push_str("# TYPE sgl_router_policy_decisions_total counter\n");
        let guard = self.policy_decisions_total.lock();
        let mut entries: Vec<(&PolicyDecisionKey, u64)> = guard
            .iter()
            .map(|(key, value)| (key, value.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.policy, &a.0.reason).cmp(&(&b.0.policy, &b.0.reason)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_policy_decisions_total{{policy=\"{}\",reason=\"{}\"}} {}\n",
                escape_label(&key.policy),
                escape_label(&key.reason),
                value,
            ));
        }
        drop(guard);

        // policy_selection_failures_total
        out.push_str(
            "# HELP sgl_router_policy_selection_failures_total Failed Prefill policy selections by policy and bounded reason.\n",
        );
        out.push_str("# TYPE sgl_router_policy_selection_failures_total counter\n");
        let guard = self.policy_selection_failures_total.lock();
        let mut entries: Vec<(&PolicyDecisionKey, u64)> = guard
            .iter()
            .map(|(key, value)| (key, value.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.policy, &a.0.reason).cmp(&(&b.0.policy, &b.0.reason)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_policy_selection_failures_total{{policy=\"{}\",reason=\"{}\"}} {}\n",
                escape_label(&key.policy),
                escape_label(&key.reason),
                value,
            ));
        }
        drop(guard);

        out.push_str(
            "# HELP sgl_router_cache_admission_evaluated_total Cache-Aware candidates evaluated by hard admission.\n",
        );
        out.push_str("# TYPE sgl_router_cache_admission_evaluated_total counter\n");
        out.push_str(&format!(
            "sgl_router_cache_admission_evaluated_total {}\n",
            self.cache_admission_evaluated_total.load(Ordering::Relaxed),
        ));
        out.push_str(
            "# HELP sgl_router_cache_admission_rejected_total Cache-Aware candidates rejected by hard admission.\n",
        );
        out.push_str("# TYPE sgl_router_cache_admission_rejected_total counter\n");
        out.push_str(&format!(
            "sgl_router_cache_admission_rejected_total {}\n",
            self.cache_admission_rejected_total.load(Ordering::Relaxed),
        ));
        out.push_str(
            "# HELP sgl_router_cache_pressure_guard_compared_total Complete fresh Cache-Aware candidate pairs evaluated by the pressure guard.\n",
        );
        out.push_str("# TYPE sgl_router_cache_pressure_guard_compared_total counter\n");
        out.push_str(&format!(
            "sgl_router_cache_pressure_guard_compared_total {}\n",
            self.cache_pressure_guard_compared_total
                .load(Ordering::Relaxed),
        ));
        out.push_str(
            "# HELP sgl_router_cache_pressure_guard_override_total Pressure-guard comparisons whose outcome differs from cache/work ordering without the guard.\n",
        );
        out.push_str("# TYPE sgl_router_cache_pressure_guard_override_total counter\n");
        out.push_str(&format!(
            "sgl_router_cache_pressure_guard_override_total {}\n",
            self.cache_pressure_guard_override_total
                .load(Ordering::Relaxed),
        ));
        out.push_str(
            "# HELP sgl_router_cache_monitor_decisions_total Cache-Aware candidate resolutions by actual load source.\n",
        );
        out.push_str("# TYPE sgl_router_cache_monitor_decisions_total counter\n");
        let guard = self.cache_monitor_decisions_total.lock();
        let mut entries: Vec<(&&str, u64)> = guard
            .iter()
            .map(|(source, value)| (source, value.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by_key(|entry| *entry.0);
        for (source, value) in entries {
            out.push_str(&format!(
                "sgl_router_cache_monitor_decisions_total{{source=\"{}\"}} {}\n",
                source, value,
            ));
        }
        drop(guard);

        // cache_aware_decisions_total — one terminal outcome per selection
        out.push_str(
            "# HELP sgl_router_cache_aware_decisions_total Cache-aware prefill selections by terminal decision: cache_hit = a prefix owner won; no_candidates = the lookup ran and produced no usable owner; candidates_exhausted = owners existed and admission or the pressure guard rejected every one; lookup_unavailable = no prefix lookup happened (no routing tokens, no fleet block size yet, or no provider configured). Selections that ended in a 503 are in sgl_router_policy_selection_failures_total instead.\n",
        );
        out.push_str("# TYPE sgl_router_cache_aware_decisions_total counter\n");
        let guard = self.cache_aware_decisions_total.lock();
        let mut entries: Vec<(&CacheAwareDecisionKey, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.model_id, a.0.decision).cmp(&(&b.0.model_id, b.0.decision)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_cache_aware_decisions_total{{model_id=\"{}\",decision=\"{}\"}} {}\n",
                escape_label(&key.model_id),
                key.decision,
                value,
            ));
        }
        drop(guard);

        // The locality triple — three families off one keyed value, rendered
        // together so a reader meets the denominator before the numerators.
        let guard = self.cache_aware_blocks.lock();
        let mut entries: Vec<(&CacheAwareDecisionKey, &Arc<CacheAwareBlocks>)> =
            guard.iter().collect();
        entries.sort_by(|a, b| (&a.0.model_id, a.0.decision).cmp(&(&b.0.model_id, b.0.decision)));
        for (name, help, pick) in [
            (
                "sgl_router_cache_aware_query_blocks_total",
                "Block hashes the request was looked up on — the denominator of every locality ratio. Emitted only for selections that reached the lookup, so it does NOT join with sgl_router_cache_aware_decisions_total on decision=\"lookup_unavailable\".",
                (|b: &CacheAwareBlocks| b.query.load(Ordering::Relaxed))
                    as fn(&CacheAwareBlocks) -> u64,
            ),
            (
                "sgl_router_matched_overlap_blocks_total",
                "Blocks held by the BEST-matching worker in the fleet, by terminal decision. Counter form of sgl_router_overlap_blocks_sum; an upper bound on reusable prefix, not what was routed to.",
                (|b: &CacheAwareBlocks| b.matched.load(Ordering::Relaxed))
                    as fn(&CacheAwareBlocks) -> u64,
            ),
            (
                "sgl_router_selected_overlap_blocks_total",
                "Blocks held by the worker actually SELECTED, by terminal decision. Divided by query blocks this is the router's prediction of the engine's cache hit rate; its shortfall against matched blocks is locality the routing decision gave up. Reads 0 for a worker that publishes no KV events, which is indistinguishable here from one holding nothing.",
                (|b: &CacheAwareBlocks| b.selected.load(Ordering::Relaxed))
                    as fn(&CacheAwareBlocks) -> u64,
            ),
        ] {
            out.push_str(&format!("# HELP {name} {help}\n"));
            out.push_str(&format!("# TYPE {name} counter\n"));
            for (key, blocks) in &entries {
                out.push_str(&format!(
                    "{name}{{model_id=\"{}\",decision=\"{}\"}} {}\n",
                    escape_label(&key.model_id),
                    key.decision,
                    pick(blocks),
                ));
            }
        }
        drop(guard);

        // overlap_blocks histogram — distribution behind matched_overlap_blocks_total
        out.push_str(
            "# HELP sgl_router_overlap_blocks Fleet-best overlap-block count observed at cache-aware selection. The distribution behind sgl_router_matched_overlap_blocks_total: the le=\"0\" bucket is the zero-match share, which sgl_router_zero_match_block0_total then attributes to the engine or the router.\n",
        );
        out.push_str("# TYPE sgl_router_overlap_blocks histogram\n");
        let guard = self.overlap_blocks.lock();
        let mut models: Vec<&String> = guard.keys().collect();
        models.sort();
        for model_id in models {
            let hist = &guard[model_id];
            let label_body = format!("model_id=\"{}\"", escape_label(model_id));
            render_histogram(&mut out, "sgl_router_overlap_blocks", &label_body, hist);
        }
        drop(guard);

        // selected_owner_tier_total — which tier the chosen worker holds it on
        out.push_str(
            "# HELP sgl_router_selected_owner_tier_total Cache-aware selections by the cheapest storage tier on which the CHOSEN worker holds the matched prefix, in the same vocabulary as sgl_router_kv_tree_blocks: device = served in place, host = served by load-back from host memory, disk / external = from a storage backend. Two labels are not tiers: none = the chosen worker holds no part of this prefix, unknown = the deployment's prefix provider does not report tiers (the out-of-process indexer).\n",
        );
        out.push_str("# TYPE sgl_router_selected_owner_tier_total counter\n");
        let guard = self.selected_owner_tier_total.lock();
        let mut entries: Vec<(&SelectedTierKey, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.model_id, a.0.tier).cmp(&(&b.0.model_id, b.0.tier)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_selected_owner_tier_total{{model_id=\"{}\",tier=\"{}\"}} {}\n",
                escape_label(&key.model_id),
                key.tier,
                value,
            ));
        }
        drop(guard);

        // zero_match_block0_total — whose fault a zero match is
        out.push_str(
            "# HELP sgl_router_zero_match_block0_total Zero-overlap cache-aware selections split by whether the request's FIRST queried block hash exists anywhere in the KV tree. absent = the fleet never published that block or removed it (engine-side gap); in_tree = the hash is carried but the root-anchored walk could not reach it (router-side linkage). Strict subset of sgl_router_cache_aware_decisions_total, and never more than the le=\"0\" bucket of sgl_router_overlap_blocks.\n",
        );
        out.push_str("# TYPE sgl_router_zero_match_block0_total counter\n");
        let guard = self.zero_match_block0_total.lock();
        let mut entries: Vec<(&ZeroMatchKey, u64)> = guard
            .iter()
            .map(|(k, v)| (k, v.load(Ordering::Relaxed)))
            .collect();
        entries.sort_by(|a, b| (&a.0.model_id, a.0.presence).cmp(&(&b.0.model_id, b.0.presence)));
        for (key, value) in entries {
            out.push_str(&format!(
                "sgl_router_zero_match_block0_total{{model_id=\"{}\",presence=\"{}\"}} {}\n",
                escape_label(&key.model_id),
                key.presence,
                value,
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
pub(crate) fn escape_label(s: &str) -> String {
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
        assert!(out.contains("# TYPE sgl_router_request_duration_seconds histogram"));
        assert!(out.contains("# TYPE sgl_router_ttft_seconds histogram"));
        assert!(out.contains("# TYPE sgl_router_responses_total counter"));
        assert!(out.contains("# TYPE sgl_router_active_load gauge"));
        assert!(out.contains("# TYPE sgl_router_workers gauge"));
        assert!(out.contains("# TYPE sgl_router_worker_health gauge"));
        assert!(out.contains("# TYPE sgl_router_worker_cb_state gauge"));
        assert!(out.contains("# TYPE sgl_router_worker_inflight_requests gauge"));
        assert!(out.contains("# TYPE sgl_router_stale_requests_total counter"));
        assert!(out.contains("# TYPE sgl_router_decode_affinity_total counter"));
        assert!(out.contains("# TYPE sgl_router_sticky_total counter"));
        assert!(out.contains("# TYPE sgl_router_policy_decisions_total counter"));
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
        reg.record_response("/v1/chat/completions", "POST", 504);
        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="200"} 2"#
        ));
        assert!(out.contains(
            r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="502"} 1"#
        ));
        assert!(out.contains(
            r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="504"} 1"#
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
    fn policy_decisions_are_keyed_by_policy_and_reason() {
        let reg = MetricsRegistry::new();
        reg.record_policy_decision("session_aware", "session_primary");
        reg.record_policy_decision("session_aware", "session_primary");
        reg.record_policy_decision("cache_aware", "cache_candidate");

        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_policy_decisions_total{policy="cache_aware",reason="cache_candidate"} 1"#
        ));
        assert!(out.contains(
            r#"sgl_router_policy_decisions_total{policy="session_aware",reason="session_primary"} 2"#
        ));
    }

    #[test]
    fn policy_selection_failures_are_keyed_by_policy_and_reason() {
        let reg = MetricsRegistry::new();
        reg.record_policy_selection_failure(
            PolicyKind::SessionAware,
            PolicySelectionFailureReason::PrefillAdmissionExhausted,
        );
        reg.record_policy_selection_failure(
            PolicyKind::CacheAware,
            PolicySelectionFailureReason::CacheCandidatesExhausted,
        );
        reg.record_policy_selection_failure(
            PolicyKind::RoundRobin,
            PolicySelectionFailureReason::ProposalEmpty,
        );

        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_policy_selection_failures_total{policy="session_aware",reason="prefill_admission_exhausted"} 1"#
        ));
        assert!(out.contains(
            r#"sgl_router_policy_selection_failures_total{policy="cache_aware",reason="cache_candidates_exhausted"} 1"#
        ));
        assert!(out.contains(
            r#"sgl_router_policy_selection_failures_total{policy="round_robin",reason="proposal_empty"} 1"#
        ));
    }

    #[test]
    fn cache_monitor_and_guard_counters_are_exposed() {
        let reg = MetricsRegistry::new();
        reg.record_cache_monitor_decision("estimated_prefill_queue_ms");
        reg.record_cache_admission_evaluations(3);
        reg.record_cache_admission_rejections(2);
        reg.record_cache_pressure_guard(3, 1);

        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_cache_monitor_decisions_total{source="estimated_prefill_queue_ms"} 1"#
        ));
        assert!(out.contains("sgl_router_cache_admission_evaluated_total 3"));
        assert!(out.contains("sgl_router_cache_admission_rejected_total 2"));
        assert!(out.contains("sgl_router_cache_pressure_guard_compared_total 3"));
        assert!(out.contains("sgl_router_cache_pressure_guard_override_total 1"));
    }

    /// Every new family renders `model_id` through `escape_label`. A model id
    /// with a quote or a backslash in it would otherwise emit a line the
    /// Prometheus text parser rejects, and a rejected line takes the WHOLE
    /// scrape with it — not just that series.
    #[test]
    fn locality_families_escape_the_model_label() {
        let reg = MetricsRegistry::new();
        let model = r#"mod"el\x"#;
        reg.record_cache_aware_selection(
            model,
            CacheAwareDecision::CacheHit,
            Some(LocalityBlocks {
                query: 10,
                matched: 4,
                selected: 4,
            }),
        );
        reg.record_selected_owner_tier(model, "device");
        reg.record_zero_match_block0(model, false);

        let out = reg.render();
        let escaped = r#"mod\"el\\x"#;
        for family in [
            "sgl_router_cache_aware_decisions_total",
            "sgl_router_cache_aware_query_blocks_total",
            "sgl_router_matched_overlap_blocks_total",
            "sgl_router_selected_overlap_blocks_total",
            "sgl_router_selected_owner_tier_total",
            "sgl_router_zero_match_block0_total",
        ] {
            let want = format!(r#"{family}{{model_id="{escaped}""#);
            assert!(out.contains(&want), "missing {want} in:\n{out}");
        }
        assert!(out.contains(&format!(
            r#"sgl_router_overlap_blocks_count{{model_id="{escaped}"}} 1"#
        )));
        assert!(
            !out.contains(&format!(r#"model_id="{model}""#)),
            "a raw unescaped model id must never reach the exposition",
        );
    }

    #[test]
    fn locality_triple_shares_one_decision_key() {
        let reg = MetricsRegistry::new();
        reg.record_cache_aware_selection(
            "tiny",
            CacheAwareDecision::CacheHit,
            Some(LocalityBlocks {
                query: 100,
                matched: 80,
                selected: 80,
            }),
        );
        reg.record_cache_aware_selection(
            "tiny",
            CacheAwareDecision::CandidatesExhausted,
            Some(LocalityBlocks {
                query: 100,
                matched: 90,
                selected: 0,
            }),
        );

        let out = reg.render();
        for (name, decision, value) in [
            (
                "sgl_router_cache_aware_query_blocks_total",
                "cache_hit",
                100,
            ),
            ("sgl_router_matched_overlap_blocks_total", "cache_hit", 80),
            ("sgl_router_selected_overlap_blocks_total", "cache_hit", 80),
            (
                "sgl_router_cache_aware_query_blocks_total",
                "candidates_exhausted",
                100,
            ),
            (
                "sgl_router_matched_overlap_blocks_total",
                "candidates_exhausted",
                90,
            ),
            // The whole point of the split: locality the CAPACITY decision
            // gave up must not be averaged into the hit bucket.
            (
                "sgl_router_selected_overlap_blocks_total",
                "candidates_exhausted",
                0,
            ),
        ] {
            let want = format!(r#"{name}{{model_id="tiny",decision="{decision}"}} {value}"#);
            assert!(out.contains(&want), "missing {want} in:\n{out}");
        }
        assert!(out.contains(
            r#"sgl_router_cache_aware_decisions_total{model_id="tiny",decision="cache_hit"} 1"#
        ));
    }

    /// A selection with no lookup books a decision and nothing else — a zero
    /// triple would drag every ratio down for a reason unrelated to the cache.
    #[test]
    fn a_selection_without_a_lookup_contributes_no_blocks() {
        let reg = MetricsRegistry::new();
        reg.record_cache_aware_selection("tiny", CacheAwareDecision::LookupUnavailable, None);

        let out = reg.render();
        assert!(out.contains(
            r#"sgl_router_cache_aware_decisions_total{model_id="tiny",decision="lookup_unavailable"} 1"#
        ));
        assert!(
            !out.contains(r#"sgl_router_cache_aware_query_blocks_total{model_id="tiny""#),
            "no lookup means no denominator; got:\n{out}"
        );
        assert!(
            !out.contains(r#"sgl_router_overlap_blocks_count{model_id="tiny"}"#),
            "and no histogram observation; got:\n{out}"
        );
    }

    /// The histogram tracks the FLEET-BEST overlap, so its `le="0"` bucket is
    /// the zero-match share that `zero_match_block0_total` then attributes.
    #[test]
    fn overlap_histogram_observes_matched_and_keeps_a_zero_bucket() {
        let reg = MetricsRegistry::new();
        for matched in [0, 0, 40] {
            reg.record_cache_aware_selection(
                "tiny",
                CacheAwareDecision::CacheHit,
                Some(LocalityBlocks {
                    query: 100,
                    matched,
                    selected: matched,
                }),
            );
        }

        let out = reg.render();
        assert!(out.contains(r#"sgl_router_overlap_blocks_count{model_id="tiny"} 3"#));
        assert!(out.contains(r#"sgl_router_overlap_blocks_bucket{model_id="tiny",le="0"} 2"#));
        // 40 lands in le=64, not le=32.
        assert!(out.contains(r#"sgl_router_overlap_blocks_bucket{model_id="tiny",le="32"} 2"#));
        assert!(out.contains(r#"sgl_router_overlap_blocks_bucket{model_id="tiny",le="64"} 3"#));
    }

    #[test]
    fn tier_and_zero_match_counters_render_their_labels() {
        let reg = MetricsRegistry::new();
        reg.record_selected_owner_tier("tiny", "device");
        reg.record_selected_owner_tier("tiny", "host");
        reg.record_selected_owner_tier("tiny", "host");
        reg.record_selected_owner_tier("tiny", "unknown");
        reg.record_zero_match_block0("tiny", true);
        reg.record_zero_match_block0("tiny", false);
        reg.record_zero_match_block0("tiny", false);

        let out = reg.render();
        for want in [
            r#"sgl_router_selected_owner_tier_total{model_id="tiny",tier="device"} 1"#,
            r#"sgl_router_selected_owner_tier_total{model_id="tiny",tier="host"} 2"#,
            r#"sgl_router_selected_owner_tier_total{model_id="tiny",tier="unknown"} 1"#,
            r#"sgl_router_zero_match_block0_total{model_id="tiny",presence="in_tree"} 1"#,
            r#"sgl_router_zero_match_block0_total{model_id="tiny",presence="absent"} 2"#,
        ] {
            assert!(out.contains(want), "missing {want} in:\n{out}");
        }
    }

    /// The nesting the dashboards assume is asserted at the booking site, so a
    /// call site that swaps two counts fails here rather than rendering a hit
    /// rate above 100%.
    #[test]
    #[should_panic(expected = "locality decomposition must nest")]
    #[cfg(debug_assertions)]
    fn a_selection_that_beats_the_fleet_best_is_rejected() {
        let reg = MetricsRegistry::new();
        reg.record_cache_aware_selection(
            "tiny",
            CacheAwareDecision::CacheHit,
            Some(LocalityBlocks {
                query: 100,
                matched: 10,
                selected: 20,
            }),
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
}
