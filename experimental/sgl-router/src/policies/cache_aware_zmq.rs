// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-aware-ZMQ selection policy.
//!
//! Combines the KV-event-fed [`HashTree`] with active-load scoring and
//! tokenizer-driven block-hash lookup to pick the worker most likely to
//! already hold the request's prefix in its KV cache.
//!
//! # Selection algorithm
//!
//! Given `workers` (already filtered to healthy + matching pool by the
//! caller) and a `SelectionContext` carrying the JSON request body and the
//! ingress-precomputed routing tokens:
//!
//! Every load *ranking* below uses [`WorkerLoads::load_of`] (the queue gate
//! reads [`WorkerLoads::waiting_of`] instead): where a fresh
//! [`super::engine_load::EngineLoadTable`] snapshot exists, the
//! engine-reported queue depth (`num_running + num_waiting`) PLUS this
//! worker's dispatches since that snapshot was taken (the engine hasn't had
//! a chance to report them yet); otherwise the router-side in-flight
//! counter `Worker::active_load()`. The "since that snapshot" bound matters:
//! the engine only refreshes its gauge every few seconds
//! ([`super::engine_load::EngineLoadTable`]'s freshness window), which at
//! sustained request rates is several selections' worth of staleness, and
//! using it unadjusted lets a burst of back-to-back decisions all read the
//! same "worker looks idle" number and all pile onto it before the gauge
//! catches up. Adding back only the
//! not-yet-reported dispatches (not the worker's full `active_load`, which
//! can include long-held slots from slow-draining streaming responses)
//! self-corrects within the same burst without overcorrecting away from
//! workers that are idle on the engine side but still draining a finished
//! stream to a slow client.
//!
//! 1. **Load gate**, per [`crate::config::LoadGate`]. Two mutually exclusive
//!    strategies; see that type for why each measures what it does.
//!
//!    `PerWorkerQueue` defers to step 3, because a per-worker question cannot
//!    be answered until the cache has named a worker. The cache lookup is
//!    unchanged, but every min-load fallback in this file additionally prefers
//!    a non-queueing worker — including the below-threshold tree-miss path,
//!    which is the highest-volume one.
//!
//!    `FleetSpread` — the default — decides here: when the spread is wide it
//!    skips the cache lookup entirely and picks the lowest-load of a
//!    `min_load_choices` sample,
//!    abandoning locality for every request on the strength of one busy one.
//! 2. **Routing tokens.** Prefer the ingress-precomputed ids
//!    (`ctx.request_tokens()`); fall back to tokenizing the body here
//!    (chat-encoder-aware for chat traffic, raw `prompt`/`text` otherwise)
//!    for callers that didn't pre-tokenize. On any failure (no tokens, no
//!    tokenizer, encode error, empty), fall through to step 4 (min-load).
//! 3. **Hash + match.** Compute block hashes via
//!    [`super::kv_events::compute_block_hashes`], query the shared hash tree
//!    for the longest matching prefix. If `match_rate > cache_threshold`,
//!    pick the lowest-load worker whose `url` appears in the match result and
//!    whose engine queue is under `PerWorkerQueue`'s limit, if one is
//!    configured. Otherwise, fall through.
//! 4. **Min-load fallback.** Least-loaded among `min_load_choices` uniformly
//!    sampled workers that are not known to be queueing; if every worker is
//!    queueing, least-loaded among a sample of the whole fleet. Sampling is
//!    the power-of-X defence against cross-replica herds: an exact fleet
//!    minimum makes every replica with the same load snapshots pick the same
//!    worker. The second tier is what makes this unable to fail — the gate is
//!    never allowed to turn a busy fleet into a selection error. Reaching it
//!    means no worker is unqueued, and with prefix owners present that
//!    saturation is recorded as `cache_hit_all_queued` — the fleet-saturation
//!    signal. The label keys on saturation, not on where the sampled draw
//!    lands, which usually is NOT a prefix owner; treat it as locality only
//!    on fleets pinned to unsampled picks (`min_load_choices >= fleet`).
//!
//! The implementation never returns `None` for a non-empty `workers` slice;
//! a misconfigured tree or tokenizer degrades to round-robin-with-load
//! tiebreak, not a routing failure.

use crate::config::{CacheAwareConfig, LoadGate};

use crate::policies::engine_load::{EngineLoadTable, WorkerDepth};
use crate::policies::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree,
};
use crate::policies::{request_tokens_for, Policy, SelectionContext};
use crate::server::metrics::{CacheAwareDecision, MetricsRegistry};
use crate::tokenizer::TokenizerRegistry;
use crate::workers::Worker;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

/// Two of the min-load fallbacks are reachable in steady state rather than
/// being invariant violations, and `select` runs once per *evaluation* — which
/// admission repeats on a claim race. Logging them per evaluation lets a client
/// (empty-content messages) or a single misconfigured model flood the log and
/// bury the genuinely-never-fires `error!` diagnostics beside them.
///
/// `sgl_router_cache_aware_decisions_total` already carries the exact volume,
/// so these logs only need to name the cause: emit the first occurrence, then
/// roughly 1-in-`FALLBACK_LOG_SAMPLE`. Separate counters so a flood of one
/// cause cannot starve the other's visibility.
const FALLBACK_LOG_SAMPLE: u64 = 64;
static TOKENIZATION_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);
static HASH_CONFIG_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);
static MATCHED_FALLBACK_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);
static QUEUE_GATE_BLIND_LOG_COUNTER: AtomicU64 = AtomicU64::new(0);

fn should_log(counter: &AtomicU64) -> bool {
    counter
        .fetch_add(1, Ordering::Relaxed)
        .is_multiple_of(FALLBACK_LOG_SAMPLE)
}

/// Selection policy that scores candidates by tree-overlap with the
/// request's prefix and falls back to load-based picking when the tree
/// doesn't have useful signal.
pub struct CacheAwareZmqPolicy {
    config: CacheAwareConfig,
    /// Per-process KV-event hash tree, fed by the indexer. Cheap to
    /// clone an `Arc`; we never write to the tree from here.
    tree: Arc<HashTree>,
    /// Tokenizer registry — selection reads `model_id` from the context
    /// and looks up the per-model tokenizer.
    tokenizers: Arc<TokenizerRegistry>,
    /// Worker-sourced block size, shared with the `KvEventIndex` that
    /// seeds it on worker registration. Read once per request; if
    /// `None` (no worker has reported a `page_size` yet) the policy
    /// degrades to min-load — the router cannot hash a prompt without
    /// a block size that matches what the worker publishes.
    block_size_oracle: Arc<BlockSizeOracle>,
    /// Engine-reported per-worker load (running + waiting), shared with the
    /// `KvEventIndex` load subscriber. Read once per selection; a worker with
    /// a fresh snapshot uses it in place of the router-side in-flight counter
    /// (`Worker::active_load`), falling back to that counter when the snapshot
    /// is stale or absent (cold start / worker predates load publishing).
    engine_load: Arc<EngineLoadTable>,
    /// Optional metrics sink. Set via [`Self::with_metrics`] by the policy
    /// factory for the production policy; `None` in unit tests and
    /// non-cache-aware call sites. When set, each cache-aware selection
    /// records the prefix-overlap block count into
    /// `sgl_router_overlap_blocks`. Set once via [`Self::with_metrics`]
    /// (tests) or the `Policy::attach_metrics` hook (production, called by
    /// `PolicyRegistry::attach_metrics` after the registry is built).
    metrics: OnceLock<Arc<MetricsRegistry>>,
}

impl std::fmt::Debug for CacheAwareZmqPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheAwareZmqPolicy")
            .field("config", &self.config)
            .field("tree_nodes", &self.tree.node_count())
            .finish()
    }
}

/// Snapshot of the load-imbalance check, carried out of
/// [`CacheAwareZmqPolicy::balance_check`] so the caller can log the
/// numbers behind a rebalance decision.
struct BalanceCheck {
    min_load: usize,
    max_load: usize,
    abs_diff: usize,
    imbalanced: bool,
}

/// Per-selection load lookup. Built once per `select` from a single
/// [`EngineLoadTable::fresh_worker_state`] pass: a worker with a fresh
/// engine-reported snapshot uses its queue depth (`num_running +
/// num_waiting`) plus its own dispatches acquired since that snapshot's
/// timestamp (see [`Self::load_of`]); otherwise it falls back to the
/// router-side in-flight counter (`Worker::active_load`). Holding the
/// snapshot keeps every per-worker `load_of` an O(1) map lookup.
struct WorkerLoads {
    /// url -> (engine-reported depth + queue, that snapshot's oldest-rank
    /// timestamp).
    fresh: HashMap<String, (WorkerDepth, Instant)>,
}

impl WorkerLoads {
    /// Build the per-selection snapshot from one `fresh_worker_state` pass.
    /// The single construction chokepoint guarantees every comparison in a
    /// given `select` sees one consistent view of load.
    fn from_engine(table: &EngineLoadTable, now: Instant) -> Self {
        Self {
            fresh: table.fresh_worker_state(now),
        }
    }

    /// A worker's current load: the engine-reported queue depth as of the
    /// last fresh snapshot, plus this worker's own dispatches made *since*
    /// that snapshot's timestamp — i.e. exactly the requests the engine
    /// hasn't had a chance to report back on yet. This is deliberately not
    /// the worker's full `active_load()`: that counter also includes
    /// long-held slots from slow-draining streaming responses (see
    /// `crate::proxy::Proxy::forward_streaming_to`'s `stream_guards` doc)
    /// that the engine's own last report has likely already accounted for —
    /// adding the full counter on top would bias selection away from workers
    /// that are idle on the engine side but still slowly draining a finished
    /// stream to a client.
    ///
    /// This correction is per-router-process: it only sees dispatches THIS
    /// router pod made. It closes the single-pod stale-gauge herd, but does
    /// not coordinate with other router replicas — two pods can still both
    /// read the same stale engine number and independently pile onto the
    /// same worker within one gauge-refresh window. Closing that would need
    /// cross-replica state sharing, which this fix does not attempt.
    fn load_of(&self, w: &Worker) -> usize {
        match self.fresh.get(w.url.as_str()) {
            // `saturating_add`, not an assertable invariant: both operands
            // are bounded by real concurrency limits (a worker's in-flight
            // count is bounded well below `usize::MAX` by connection and
            // request-rate limits upstream of the router), so overflow here
            // is unreachable from real traffic — reaching it would mean a
            // problem (memory exhaustion, a corrupt engine payload) that is
            // already symptomatic elsewhere, not something worth a panic on
            // this per-request hot path.
            Some(&(d, at)) => d.depth().saturating_add(w.slots_acquired_since(at)),
            None => w.active_load(),
        }
    }

    /// How many requests are queued on this worker's engine, or `None` when no
    /// fresh snapshot says.
    ///
    /// Deliberately not defaulted to 0 or to any router-side value. The
    /// router-side counter has no queue component at all — it cannot tell a
    /// dispatched-and-running request from a dispatched-and-waiting one — so
    /// there is no honest substitute, and inventing one would silently turn a
    /// queue gate into a comparison against a different quantity. `None` means
    /// "unknown", and callers gate open on it.
    ///
    /// Unlike [`Self::load_of`] this carries no since-snapshot correction: a
    /// dispatch this router just made is not known to be *queued* — the engine
    /// may well be running it — so adding it here would manufacture queue
    /// depth that may not exist.
    fn waiting_of(&self, w: &Worker) -> Option<usize> {
        self.fresh.get(w.url.as_str()).map(|(d, _)| d.waiting())
    }

    /// Number of workers whose load came from the engine (vs the router-side
    /// fallback). Used only to annotate the rebalance log.
    fn engine_worker_count(&self) -> usize {
        self.fresh.len()
    }
}

/// Outcome of [`CacheAwareZmqPolicy::match_request`] — the matched cache
/// owners plus the best-mode stats the caller logs and meters. In a uniform
/// fleet these describe the single established hashing mode; in a bimodal
/// fleet they are the union of owners and the best-matching mode's numbers.
struct MatchOutcome {
    /// URLs of workers owning a matched prefix, drawn only from hashing modes
    /// whose per-mode overlap cleared `cache_threshold`. Empty ⇒ no affinity.
    owner_urls: std::collections::HashSet<String>,
    /// Best (max-rate) mode's matched-block count — logging / metrics only.
    matched_blocks: usize,
    /// Best mode's query-block count — the match_rate denominator and the
    /// query-blocks metric sample.
    query_blocks: usize,
    /// Best mode's match_rate.
    match_rate: f32,
    /// True once some mode produced query blocks (else too few tokens to hash).
    had_blocks: bool,
    /// True when some mode cleared the threshold even if its matched node had
    /// no live owner — lets the caller separate "unowned" from "below threshold".
    any_above_threshold: bool,
    /// "bigram" | "unigram" | "dual", for the debug log.
    label: &'static str,
}

impl CacheAwareZmqPolicy {
    pub fn new(
        config: CacheAwareConfig,
        tree: Arc<HashTree>,
        tokenizers: Arc<TokenizerRegistry>,
        block_size_oracle: Arc<BlockSizeOracle>,
        engine_load: Arc<EngineLoadTable>,
    ) -> Self {
        Self {
            config,
            tree,
            tokenizers,
            block_size_oracle,
            engine_load,
            metrics: OnceLock::new(),
        }
    }

    /// Attach a metrics sink so each cache-aware selection records the
    /// prefix-overlap block count into `sgl_router_overlap_blocks`. Builder
    /// form used by tests; production wiring goes through the
    /// `Policy::attach_metrics` hook.
    pub fn with_metrics(self, metrics: Arc<MetricsRegistry>) -> Self {
        self.attach_metrics_once(metrics);
        self
    }

    fn attach_metrics_once(&self, metrics: Arc<MetricsRegistry>) {
        if self.metrics.set(metrics).is_ok() {
            tracing::info!("cache-aware-zmq metrics attached");
        } else {
            tracing::warn!(
                "cache-aware-zmq metrics attached more than once; second registry ignored"
            );
        }
    }

    fn record_decision(&self, model_id: &str, decision: CacheAwareDecision) {
        if let Some(metrics) = self.metrics.get() {
            metrics.record_cache_aware_decision(model_id, decision);
        }
    }

    /// Lowest-load worker among `choices` uniformly sampled candidates, by
    /// the per-selection load lookup — power-of-X choices, not a fleet-wide
    /// minimum.
    ///
    /// Ranking the whole fleet and taking the single minimum converges
    /// across router replicas: every replica reads the same engine load
    /// snapshots, none sees the others' dispatches made since the snapshot
    /// ([`WorkerLoads::load_of`] corrects only its own), so they all hand
    /// the request to the same current minimum and overshoot it. Sampling
    /// `choices` workers per pick keeps the selection near-minimal while
    /// making concurrent replicas diverge to different near-minima.
    /// `choices == 1` is uniform-random within the tier; `choices >=
    /// eligible pool` degenerates to the deterministic exact minimum
    /// (unsampled — equal-load ties resolve in pool order, so pinning
    /// `usize::MAX` keeps tests deterministic). A programmatic
    /// `choices == 0` (impossible from the CLI, which is `NonZeroUsize`)
    /// is treated as 1.
    ///
    /// Under [`LoadGate::PerWorkerQueue`] this stays a two-tier pick, and
    /// the sampling happens WITHIN each tier: the sample is drawn from the
    /// workers that are not already queueing, and only when none exists
    /// from the queueing ones — a sample never lands on a queueing worker
    /// while an unqueued one exists.
    ///
    /// The two tiers are not optional. Load is ranked by total depth while the
    /// gate reads queue length, and those disagree exactly where this gate
    /// earns its keep: a shallow worker with a backlog is the fleet minimum by
    /// depth, so a single-tier min-load fallback hands the request straight
    /// back to the cache home the gate just rejected. Preferring unqueued
    /// workers first is what makes the diversion actually happen; falling back
    /// to the whole fleet is what keeps an all-queueing fleet routable
    /// instead of returning `None` and failing every request.
    fn pick_min_load(
        workers: &[Arc<Worker>],
        loads: &WorkerLoads,
        gate: &LoadGate,
        choices: usize,
    ) -> Option<Arc<Worker>> {
        // Tier order is decided by the gate, never by sampling: rank the
        // unqueued workers if any exist, else the whole fleet.
        let mut pool: Vec<usize> = workers
            .iter()
            .enumerate()
            .filter(|(_, w)| gate.admits_affinity(loads.waiting_of(w)))
            .map(|(i, _)| i)
            .collect();
        if pool.is_empty() {
            pool.extend(0..workers.len());
        }
        // Guard against a programmatically constructed 0 (the CLI is
        // NonZeroUsize; the field is a bare usize for full-fleet pinning).
        let k = choices.max(1).min(pool.len());
        if k == 0 {
            // workers is empty.
            return None;
        }
        let min_of = |candidates: &[usize]| {
            candidates
                .iter()
                .min_by_key(|&&i| loads.load_of(&workers[i]))
                .map(|&i| Arc::clone(&workers[i]))
        };
        if k == pool.len() {
            // The unsampled exact minimum: no shuffle, so ties resolve in
            // pool order exactly as the pre-sampling implementation did.
            return min_of(&pool);
        }
        // partial_shuffle leaves `amount` uniformly sampled DISTINCT
        // elements in the TAIL of the slice (rand 0.8: it randomizes
        // positions len-1 down to len-amount) and returns them as the
        // first tuple element. Reading the front is the classic misuse:
        // the remainder is only disturbed by swaps and is biased toward
        // low indices.
        use rand::seq::SliceRandom;
        let (sample, _) = pool.partial_shuffle(&mut rand::thread_rng(), k);
        min_of(sample)
    }

    /// Detect load imbalance. Returns the min/max load snapshot together
    /// with the `imbalanced` verdict — `true` when the spread between max
    /// and min load is large enough that cache-aware routing would dump
    /// even more on the hot worker. The caller logs these numbers so every
    /// rebalance decision is visible in the logs.
    ///
    /// `min_load`/`max_load` are [`WorkerLoads::load_of`] values, i.e. for a
    /// worker with a fresh engine snapshot this is the engine-reported depth
    /// PLUS this router's own not-yet-reported dispatches — not the raw
    /// engine number alone. An on-call reader comparing this log's
    /// `max_load` against the engine's own `/metrics` queue depth during an
    /// incident should expect them to differ by that correction.
    fn balance_check(
        &self,
        workers: &[Arc<Worker>],
        loads: &WorkerLoads,
        abs_threshold: usize,
        rel_factor: f32,
    ) -> BalanceCheck {
        let (min_load, max_load) = workers.iter().fold((usize::MAX, 0usize), |(mn, mx), w| {
            let l = loads.load_of(w);
            (mn.min(l), mx.max(l))
        });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };
        let abs_diff = max_load.saturating_sub(min_load);
        let rel_threshold = (min_load as f32 * rel_factor) as usize;
        let imbalanced = abs_diff > abs_threshold && max_load > rel_threshold;
        BalanceCheck {
            min_load,
            max_load,
            abs_diff,
            imbalanced,
        }
    }

    /// Match the request against the tree across one or both hashing modes.
    ///
    /// Uniform fleet (oracle not bimodal): hash once with the established mode
    /// — the historical fast path, one hash + one lookup. Bimodal fleet
    /// (EAGLE-family + non-EAGLE behind one base model): hash the request BOTH
    /// ways, look up each, and union the owners of every mode whose per-mode
    /// overlap cleared `cache_threshold`. The two hash spaces are disjoint
    /// (SHA256 over different preimages), so each mode's query only matches its
    /// own family's tree entries and the union cannot cross-contaminate.
    fn match_request(
        &self,
        tokens: &[u32],
        block_size: usize,
        primary_bigram: bool,
    ) -> MatchOutcome {
        let bimodal = self.block_size_oracle.is_bimodal();
        // Primary mode first; append the opposite only when the fleet is mixed.
        let modes: &[bool] = match (bimodal, primary_bigram) {
            (false, true) => &[true],
            (false, false) => &[false],
            (true, true) => &[true, false],
            (true, false) => &[false, true],
        };

        let mut owner_urls: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut had_blocks = false;
        let mut any_above_threshold = false;
        // (rate, matched_blocks, query_blocks) of the best-matching mode, for
        // logging + metrics. Seeded by the first hashed mode so a zero-match
        // request still reports a real query-block denominator.
        let mut best: Option<(f32, usize, usize)> = None;

        for &mode_bigram in modes {
            let hashes = if mode_bigram {
                compute_block_hashes_bigram(tokens, block_size)
            } else {
                compute_block_hashes(tokens, block_size)
            };
            if hashes.is_empty() {
                continue;
            }
            had_blocks = true;
            let matched = self.tree.match_prefix(None, &hashes);
            debug_assert!(matched.matched_blocks <= hashes.len());
            let rate = matched.matched_blocks as f32 / hashes.len() as f32;
            if best.is_none_or(|(r, _, _)| rate > r) {
                best = Some((rate, matched.matched_blocks, hashes.len()));
            }
            // Per-mode threshold BEFORE unioning: a mode contributes owners only
            // if its own overlap cleared the bar, so a strong match in one family
            // never drags in a weak match's owners from the other.
            if rate > self.config.cache_threshold {
                any_above_threshold = true;
                for w in matched.workers {
                    owner_urls.insert(w.url);
                }
            }
        }

        let (match_rate, matched_blocks, query_blocks) = best.unwrap_or((0.0, 0, 0));
        let label = match (bimodal, primary_bigram) {
            (true, _) => "dual",
            (false, true) => "bigram",
            (false, false) => "unigram",
        };
        MatchOutcome {
            owner_urls,
            matched_blocks,
            query_blocks,
            match_rate,
            had_blocks,
            any_above_threshold,
            label,
        }
    }
}

impl Policy for CacheAwareZmqPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let model_id = ctx.model().0.as_str();
        if workers.is_empty() {
            self.record_decision(model_id, CacheAwareDecision::NoWorkers);
            return None;
        }

        // Per-selection load lookup: engine-reported queue depth where fresh,
        // else the router-side in-flight counter. One snapshot pass serves
        // every comparison below (imbalance check, min-load fallback,
        // matched-set tiebreak).
        let loads = WorkerLoads::from_engine(&self.engine_load, Instant::now());
        let queue_limit = self.config.load_gate.queue_limit();

        // 1. Load gate. With a queue limit configured the gate is per-worker and
        //    cannot be decided yet — it needs to know which worker the cache
        //    would pick — so it moves to step 3 and only the audit log runs here.
        //    Without one, the fleet-spread check can short-circuit the whole
        //    selection before any tokenization happens.
        //
        //    Both paths log `engine_load_workers` against `engine_load_expected`
        //    on every selection: that pair is the only signal that `load_of` has
        //    fallen back to the per-process in-flight counter, which silently
        //    rescales every load comparison below it.
        match self.config.load_gate {
            LoadGate::PerWorkerQueue(limit) => {
                // `waiting_of` is `None` for every worker here, so the gate
                // admits everything and the fleet-spread strategy it replaced
                // is not running either — the router has no load override at
                // all. Silent otherwise: `cache_worker_queued` sitting at 0 is
                // indistinguishable from a healthy fleet, and the
                // workers/expected pair reads 0/0 when no worker ever
                // advertised a load port, which looks correctly configured.
                if loads.engine_worker_count() == 0 && should_log(&QUEUE_GATE_BLIND_LOG_COUNTER) {
                    tracing::warn!(
                        model = %ctx.model(),
                        worker_queue_limit = limit.get(),
                        engine_load_expected = self.engine_load.expected_count(),
                        "cache-aware-zmq: --worker-queue-limit is set but no worker has a fresh engine load snapshot, so the queue gate is inert and the fleet-spread gate it replaced is not running. Check that engines advertise a load port and publish LoadStat",
                    );
                }
                tracing::debug!(
                    model = %ctx.model(),
                    worker_queue_limit = limit.get(),
                    engine_load_workers = loads.engine_worker_count(),
                    engine_load_expected = self.engine_load.expected_count(),
                    "cache-aware-zmq: per-worker queue gate active; deferring the load gate to the matched set",
                );
            }
            LoadGate::FleetSpread {
                abs_threshold,
                rel_threshold,
            } => {
                let balance = self.balance_check(workers, &loads, abs_threshold, rel_threshold);
                tracing::debug!(
                    model = %ctx.model(),
                    min_load = balance.min_load,
                    max_load = balance.max_load,
                    abs_diff = balance.abs_diff,
                    balance_abs_threshold = abs_threshold,
                    balance_rel_threshold = rel_threshold,
                    imbalanced = balance.imbalanced,
                    engine_load_workers = loads.engine_worker_count(),
                    engine_load_expected = self.engine_load.expected_count(),
                    "cache-aware-zmq: load-balance check considered",
                );
                if balance.imbalanced {
                    self.record_decision(model_id, CacheAwareDecision::LoadImbalance);
                    let chosen = Self::pick_min_load(
                        workers,
                        &loads,
                        &self.config.load_gate,
                        self.config.min_load_choices,
                    );
                    if let Some(w) = &chosen {
                        tracing::info!(
                            model = %ctx.model(),
                            worker = %w.url,
                            worker_load = loads.load_of(w),
                            min_load = balance.min_load,
                            max_load = balance.max_load,
                            abs_diff = balance.abs_diff,
                            balance_abs_threshold = abs_threshold,
                            balance_rel_threshold = rel_threshold,
                            engine_load_workers = loads.engine_worker_count(),
                            engine_load_expected = self.engine_load.expected_count(),
                            "cache-aware-zmq: load imbalance detected — bypassing cache, routing to sampled min-load worker",
                        );
                    }
                    return chosen;
                }
            }
        }

        // 2. Routing tokens. Prefer the ids computed once at ingress; fall
        //    back to tokenizing the body here so the policy stays usable for
        //    callers that don't pre-tokenize (e.g. unit tests). In production
        //    the ingress always pre-tokenizes, so this is a single tokenize.
        let fallback_ids;
        let tokens: &[u32] = match ctx.request_tokens() {
            Some(t) if !t.is_empty() => t,
            _ => {
                let body = match ctx.request_body() {
                    Some(b) if !b.is_empty() => b,
                    _ => {
                        self.record_decision(model_id, CacheAwareDecision::RequestBodyUnavailable);
                        tracing::error!(
                            model = %ctx.model(),
                            "cache-aware-zmq: policy reached without request tokens or a request body; ingress validation invariant broken",
                        );
                        return Self::pick_min_load(
                            workers,
                            &loads,
                            &self.config.load_gate,
                            self.config.min_load_choices,
                        );
                    }
                };
                let value = match serde_json::from_slice::<serde_json::Value>(body) {
                    Ok(value) => value,
                    Err(error) => {
                        self.record_decision(model_id, CacheAwareDecision::RequestJsonInvalid);
                        tracing::error!(
                            model = %ctx.model(),
                            error = %error,
                            "cache-aware-zmq: policy received invalid JSON after ingress validation; invariant broken",
                        );
                        return Self::pick_min_load(
                            workers,
                            &loads,
                            &self.config.load_gate,
                            self.config.min_load_choices,
                        );
                    }
                };
                let Some(rt) = request_tokens_for(&self.tokenizers, ctx.model(), &value) else {
                    self.record_decision(model_id, CacheAwareDecision::TokenizationUnavailable);
                    if should_log(&TOKENIZATION_LOG_COUNTER) {
                        tracing::warn!(
                            model = %ctx.model(),
                            "cache-aware-zmq: routing tokens unavailable, routing by sampled min-load; check tokenizer and chat encoder configuration. Also expected for requests whose messages carry no text",
                        );
                    }
                    return Self::pick_min_load(
                        workers,
                        &loads,
                        &self.config.load_gate,
                        self.config.min_load_choices,
                    );
                };
                fallback_ids = rt.ids;
                &fallback_ids
            }
        };

        // 3. Hash + match.
        // Source both hashing properties from the worker. Reading them as one
        // published configuration avoids a first-registration window where
        // block_size is visible but an EAGLE bigram flag is not.
        let Some((block_size, is_bigram)) = self.block_size_oracle.hash_config() else {
            self.record_decision(model_id, CacheAwareDecision::HashConfigUnknown);
            // Not necessarily the startup window: `KvEventIndex::add_worker`
            // returns before it reaches the oracle when a worker publishes no
            // KV events, so a fleet where no engine publishes leaves this
            // permanently unset and the configured policy silently degrades to
            // min-load for the process lifetime. Warn (sampled) so that state
            // cannot hide at debug level.
            if should_log(&HASH_CONFIG_LOG_COUNTER) {
                tracing::warn!(
                    model = %ctx.model(),
                    "cache-aware-zmq: no worker has published a complete block-hashing config, routing by sampled min-load. Transient while workers register; if it persists, no engine is publishing KV events",
                );
            }
            return Self::pick_min_load(
                workers,
                &loads,
                &self.config.load_gate,
                self.config.min_load_choices,
            );
        };
        // Uniform fleet: hash once with the established mode. Bimodal fleet
        // (EAGLE-family + non-EAGLE behind one base model): hash both ways and
        // union the eligible owners, so a prefix warm on either family is a hit.
        let outcome = self.match_request(tokens, block_size as usize, is_bigram);
        if !outcome.had_blocks {
            self.record_decision(model_id, CacheAwareDecision::NoHashBlocks);
            return Self::pick_min_load(
                workers,
                &loads,
                &self.config.load_gate,
                self.config.min_load_choices,
            );
        }
        tracing::debug!(
            model = %ctx.model(),
            hashing = outcome.label,
            n_blocks = outcome.query_blocks,
            matched_blocks = outcome.matched_blocks,
            matched_workers = outcome.owner_urls.len(),
            match_rate = outcome.match_rate,
            cache_threshold = self.config.cache_threshold,
            "cache-aware-zmq match_prefix",
        );
        // Record overlap + query-block count before the affinity branch so the
        // histogram captures the full distribution, including below-threshold
        // selections. In a bimodal fleet these are the best-matching mode's
        // numbers, keeping one sample per request.
        if let Some(m) = self.metrics.get() {
            m.observe_overlap_blocks(model_id, outcome.matched_blocks as u64);
            m.add_cache_aware_query_blocks(model_id, outcome.query_blocks as u64);
        }
        if outcome.owner_urls.is_empty() {
            // Same taxonomy as the single-mode path: a cleared-threshold match
            // whose node has no live owner is distinct from no match at all.
            let decision = if outcome.any_above_threshold {
                CacheAwareDecision::MatchedNodeUnowned
            } else {
                CacheAwareDecision::BelowThreshold
            };
            self.record_decision(model_id, decision);
            tracing::debug!(
                model = %ctx.model(),
                n_blocks = outcome.query_blocks,
                matched_blocks = outcome.matched_blocks,
                match_rate = outcome.match_rate,
                cache_threshold = self.config.cache_threshold,
                "cache-aware-zmq: no eligible cache owner, falling back to sampled min-load",
            );
            return Self::pick_min_load(
                workers,
                &loads,
                &self.config.load_gate,
                self.config.min_load_choices,
            );
        }
        // Among the matched owners (either family, in a bimodal fleet), pick the
        // lowest-load one that is not already queueing. The gate and the tiebreak
        // read different quantities — queue length vs total depth — so filtering
        // the whole set is not the same as vetoing its minimum: the least-loaded
        // prefix owner can be the one with a backlog, while a slightly busier
        // owner has none.
        let matched_urls = &outcome.owner_urls;
        let best_matched: Option<Arc<Worker>> = workers
            .iter()
            .filter(|w| matched_urls.contains(w.url.as_str()))
            .filter(|w| self.config.load_gate.admits_affinity(loads.waiting_of(w)))
            .min_by_key(|w| loads.load_of(w))
            .map(Arc::clone);
        let Some(chosen) = best_matched else {
            let owners_present = workers
                .iter()
                .any(|w| matched_urls.contains(w.url.as_str()));
            // The fallback prefers an unqueued worker but is never queue-
            // *bounded*: with no unqueued worker it samples the whole fleet
            // and takes the least-loaded of the sample regardless, so a fleet
            // where everything is queueing still routes
            // instead of failing every request.
            let fallback = Self::pick_min_load(
                workers,
                &loads,
                &self.config.load_gate,
                self.config.min_load_choices,
            );
            // That second tier is the only way the fallback can land back on a
            // prefix owner: every owner failed the gate, so the preference tier
            // excluded them too, so reaching one means NO worker in the fleet is
            // unqueued. Routing is then identical to affinity, which is why it
            // is booked as a hit — but under a distinct label, because "every
            // worker is queueing" is the single most important thing an operator
            // sizing the limit needs to see, and burying it in `cache_hit` would
            // make a fully saturated fleet read as a healthy one.
            //
            // Classify the SATURATION, not the draw. Landing on an owner
            // implies the all-queued tier (every owner failed the gate, so no
            // tier-1 pool contains one), but under sampling the converse is
            // random: an all-queued pick usually lands on a non-owner, and
            // booking that as `CacheWorkerQueued` would both under-report the
            // saturation signal this label exists for and poison the diverted-
            // overlap histogram with requests where NO unqueued alternative —
            // hence no diversion — existed.
            let all_queued = !workers
                .iter()
                .any(|w| self.config.load_gate.admits_affinity(loads.waiting_of(w)));
            let landed_on_owner = fallback
                .as_ref()
                .is_some_and(|w| matched_urls.contains(w.url.as_str()));
            // `landed_on_owner` is intentionally not a disjunct here: it is
            // implied by `all_queued` (see above), so it carries no
            // classification information. It stays for the log line below.
            let decision = if owners_present && all_queued {
                CacheAwareDecision::CacheHitAllQueued
            } else if owners_present {
                CacheAwareDecision::CacheWorkerQueued
            } else {
                // Separate from the queue case: the prefix owners were not
                // candidates at all (e.g. they drained from discovery between
                // the tree match and this selection).
                CacheAwareDecision::MatchedWorkersIneligible
            };
            self.record_decision(model_id, decision);
            // Record what the diversion cost, but only when one happened: the
            // gate tests queue length alone and is blind to how much prefix is
            // at stake, so the distribution of what it throws away is the
            // evidence for whether that simplification holds. Compared against
            // `sgl_router_overlap_blocks` over all selections — a diverted curve
            // that skews high means the gate is trading large cached prefixes
            // for small waits.
            if matches!(decision, CacheAwareDecision::CacheWorkerQueued) {
                if let Some(m) = self.metrics.get() {
                    m.observe_diverted_overlap_blocks(model_id, outcome.matched_blocks as u64);
                }
            }
            if should_log(&MATCHED_FALLBACK_LOG_COUNTER) {
                tracing::info!(
                    model = %ctx.model(),
                    worker = fallback.as_ref().map(|w| w.url.as_str()),
                    worker_load = fallback.as_ref().map(|w| loads.load_of(w)),
                    worker_waiting = fallback.as_ref().and_then(|w| loads.waiting_of(w)),
                    matched_min_waiting = workers
                        .iter()
                        .filter(|w| matched_urls.contains(w.url.as_str()))
                        .filter_map(|w| loads.waiting_of(w))
                        .min(),
                    n_blocks = outcome.query_blocks,
                    matched_blocks = outcome.matched_blocks,
                    matched_workers = matched_urls.len(),
                    worker_queue_limit = queue_limit,
                    owners_present,
                    landed_on_owner,
                    engine_load_workers = loads.engine_worker_count(),
                    engine_load_expected = self.engine_load.expected_count(),
                    "cache-aware-zmq: cache affinity given up, falling back to sampled min-load",
                );
            }
            return fallback;
        };
        self.record_decision(model_id, CacheAwareDecision::CacheHit);
        tracing::debug!(
            model = %ctx.model(),
            worker = %chosen.url,
            matched_blocks = outcome.matched_blocks,
            "cache-aware-zmq: selected worker by cache overlap",
        );
        Some(chosen)
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        self.attach_metrics_once(metrics);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CacheAwareConfig;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::engine_load::LoadStat;
    use crate::policies::kv_events::tree::KvWorkerId;
    use crate::policies::kv_events::HashTree;
    use crate::tokenizer::adapter;
    use std::num::NonZeroUsize;
    use std::time::Duration;

    /// Helpers pin `min_load_choices: usize::MAX` so every fallback is the
    /// deterministic fleet-wide minimum — these tests assert *which* worker
    /// other policy mechanics route to, and per-pick sampling would make
    /// that nondeterministic. The sampling behaviour itself has dedicated
    /// tests below (`min_load_sampled_*`). Inline `..Default::default()`
    /// literals that skip the helpers stay deterministic only because their
    /// 2-worker fleets make k == pool; do NOT copy such a literal onto a 3+
    /// worker test — route through a helper or pin the field explicitly.
    fn cfg_default() -> CacheAwareConfig {
        CacheAwareConfig {
            cache_threshold: 0.5,
            min_load_choices: usize::MAX,
            ..Default::default()
        }
    }

    /// Helper: build a `BlockSizeOracle` already primed to the test's
    /// canonical block size (4). Mirrors what `KvEventIndex::add_worker`
    /// would do when the first real worker registers.
    fn oracle_for_tests(block_size: u32) -> Arc<BlockSizeOracle> {
        let o = BlockSizeOracle::new();
        o.try_set(block_size)
            .expect("fresh oracle accepts first set");
        o.set_bigram(false);
        o
    }

    fn worker(url: &str, model_id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId(model_id.into())],
            bootstrap_port: None,
        }))
    }

    /// Build a policy with a fresh (empty) engine-load table, so selection
    /// reads the router-side `active_load` counter — matching the
    /// pre-load-aware behaviour these tests assert.
    fn new_policy(
        config: CacheAwareConfig,
        tree: Arc<HashTree>,
        tokenizers: Arc<TokenizerRegistry>,
        oracle: Arc<BlockSizeOracle>,
    ) -> CacheAwareZmqPolicy {
        CacheAwareZmqPolicy::new(config, tree, tokenizers, oracle, EngineLoadTable::new())
    }

    /// Build a policy with an explicit engine-load table, for tests that
    /// exercise engine-reported load overriding the router-side counter.
    fn new_policy_with_load(
        config: CacheAwareConfig,
        tree: Arc<HashTree>,
        tokenizers: Arc<TokenizerRegistry>,
        oracle: Arc<BlockSizeOracle>,
        engine_load: Arc<EngineLoadTable>,
    ) -> CacheAwareZmqPolicy {
        CacheAwareZmqPolicy::new(config, tree, tokenizers, oracle, engine_load)
    }

    fn load_stat(running: u64, waiting: u64) -> LoadStat {
        LoadStat {
            num_running_reqs: running,
            num_waiting_reqs: waiting,
            num_tokens: 0,
            max_total_num_tokens: 0,
        }
    }

    fn tokenizer_registry_with_tiny() -> Arc<TokenizerRegistry> {
        let cfg = crate::config::Config {
            server: crate::config::ServerConfig {
                host: "0".into(),
                port: 0,
            },
            observability: Default::default(),
            model: crate::config::ModelConfig {
                id: "tiny".into(),
                tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
                policy: crate::config::PolicyKind::RoundRobin,
                circuit_breaker: None,
                cache_aware: None,
                sticky: None,
            },
            discovery: crate::config::DiscoveryBackend::StaticUrls(
                crate::config::StaticUrlsDiscoveryConfig {
                    urls: vec!["http://placeholder:0".into()],
                },
            ),
            proxy: crate::config::ProxyConfig::default(),
            active_load: crate::config::ActiveLoadConfig::default(),
        };
        Arc::new(TokenizerRegistry::load_from_config(&cfg).expect("load tiny tokenizer"))
    }

    /// Empty workers list returns None (parity with other policies).
    #[test]
    fn empty_workers_returns_none() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(b"{\"prompt\":\"hi\"}"));
        assert!(policy.select(&[], &ctx).is_none());
    }

    /// Empty tree: no overlap signal anywhere, fall through to min-load.
    #[test]
    fn empty_tree_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Bump w0's load so min-load picks w1 deterministically.
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"prompt":"hello world"}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Tree contains w0's prefix; cache-aware selection picks w0 even
    /// though w1 has lower load (the load skew is below the imbalance
    /// threshold, so cache wins).
    #[test]
    fn non_empty_tree_highest_overlap_wins() {
        let tree = Arc::new(HashTree::new());
        // Insert w0's tokens into the tree. The tiny tokenizer's hash
        // chain for our input is whatever `compute_block_hashes` returns;
        // we mimic the policy's hashing path so the test stays
        // deterministic against tokenizer changes.
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world"; // longer → more blocks
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        assert!(
            !hashes.is_empty(),
            "tiny tokenizer must produce at least one full block",
        );
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0, // any match counts
                ..Default::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w0:30000");
    }

    /// Helper: an oracle for a **bimodal** fleet — `primary_bigram` established
    /// first, then the opposite mode registers, latching `is_bimodal()`.
    /// Mirrors `add_worker` seeing an EAGLE-family worker and a non-EAGLE worker
    /// behind the same base model.
    fn bimodal_oracle(block_size: u32, primary_bigram: bool) -> Arc<BlockSizeOracle> {
        let o = BlockSizeOracle::new();
        o.try_set(block_size)
            .expect("fresh oracle accepts first set");
        o.set_bigram(primary_bigram); // primary mode
        o.set_bigram(!primary_bigram); // opposite mode -> bimodal
        assert!(o.is_bimodal(), "two opposite modes must latch bimodal");
        o
    }

    /// Bimodal fleet, primary = bigram (EAGLE). A prefix cached ONLY on the
    /// unigram-hashing worker is still matched: the query is dual-hashed, so
    /// the unigram entry is found even though the established primary is bigram.
    /// Without dual-hashing a bigram-only query never touches the unigram tree
    /// entry and the unigram worker is starved to min-load — the exact
    /// mixed-fleet artifact this change fixes.
    #[test]
    fn bimodal_matches_unigram_owner_under_bigram_primary() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        // non-EAGLE worker publishes UNIGRAM block hashes.
        let hashes = compute_block_hashes(&ids, block_size as usize);
        assert!(!hashes.is_empty());
        tree.insert(
            &KvWorkerId::new("http://unigram:30000".into(), 0),
            None,
            &hashes,
        );

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0, // any match counts
                ..Default::default()
            },
            tree,
            registry,
            bimodal_oracle(4, /* primary_bigram= */ true),
        );
        let w_eagle = worker("http://eagle:30000", "tiny");
        let w_unigram = worker("http://unigram:30000", "tiny");
        // Skew load so min-load would pick the EAGLE worker; only cache affinity
        // (via the unigram/secondary hash) can select the unigram owner.
        let _g1 = w_unigram.load_guard();
        let _g2 = w_unigram.load_guard();
        let workers = vec![Arc::clone(&w_eagle), Arc::clone(&w_unigram)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": text })).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://unigram:30000",
            "dual-hash must find the unigram owner despite a bigram primary",
        );
    }

    /// Gate guard: the SAME unigram-only prefix under a UNIMODAL bigram oracle
    /// is NOT matched — the query hashes bigram only, misses the unigram entry,
    /// and falls back to min-load (the EAGLE worker). Pins that dual-hashing
    /// engages strictly when the fleet is bimodal, not on every request.
    #[test]
    fn unimodal_bigram_does_not_match_unigram_owner() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        tree.insert(
            &KvWorkerId::new("http://unigram:30000".into(), 0),
            None,
            &hashes,
        );

        // Unimodal bigram oracle: only set_bigram(true), never the opposite.
        let oracle = BlockSizeOracle::new();
        oracle.try_set(4).unwrap();
        oracle.set_bigram(true);
        assert!(!oracle.is_bimodal());

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
                ..Default::default()
            },
            tree,
            registry,
            oracle,
        );
        let w_eagle = worker("http://eagle:30000", "tiny");
        let w_unigram = worker("http://unigram:30000", "tiny");
        // the unigram worker holds the prefix but has higher load; a bigram-only query can't
        // see its unigram entry, so min-load (eagle) wins.
        let _g1 = w_unigram.load_guard();
        let _g2 = w_unigram.load_guard();
        let workers = vec![Arc::clone(&w_eagle), Arc::clone(&w_unigram)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": text })).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://eagle:30000",
            "unimodal bigram must not match a unigram entry (no dual-hash)",
        );
    }

    /// Symmetric direction: bimodal fleet, primary = unigram. A prefix
    /// cached only on the bigram (EAGLE) worker is matched via the secondary
    /// bigram hash — proving dual-hash works regardless of which mode is primary.
    #[test]
    fn bimodal_matches_bigram_owner_under_unigram_primary() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        // EAGLE worker publishes BIGRAM block hashes.
        let hashes = compute_block_hashes_bigram(&ids, block_size as usize);
        assert!(!hashes.is_empty());
        tree.insert(
            &KvWorkerId::new("http://eagle:30000".into(), 0),
            None,
            &hashes,
        );

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
                ..Default::default()
            },
            tree,
            registry,
            bimodal_oracle(4, /* primary_bigram= */ false),
        );
        let w_eagle = worker("http://eagle:30000", "tiny");
        let w_unigram = worker("http://unigram:30000", "tiny");
        let _g1 = w_eagle.load_guard();
        let _g2 = w_eagle.load_guard();
        let workers = vec![Arc::clone(&w_eagle), Arc::clone(&w_unigram)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": text })).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://eagle:30000",
            "dual-hash must find the bigram owner despite a unigram primary",
        );
    }

    /// The cache-aware path records the matched prefix-overlap block count
    /// into `sgl_router_overlap_blocks`. Regression: the metric was defined
    /// but never observed in production, so the histogram stayed empty and
    /// gave no signal that cache-aware routing was matching anything.
    #[test]
    fn records_overlap_blocks_metric() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        assert!(!hashes.is_empty());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let metrics = MetricsRegistry::new();
        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
                ..Default::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
        )
        .with_metrics(Arc::clone(&metrics));

        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let _ = policy.select(&workers, &ctx).expect("must pick");

        let rendered = metrics.render();
        assert!(
            rendered.contains("sgl_router_overlap_blocks_count{model_id=\"tiny\"}"),
            "overlap_blocks histogram must be observed on a cache-aware selection; got:\n{rendered}"
        );
        assert!(
            rendered.contains(&format!(
                "sgl_router_cache_aware_query_blocks_total{{model_id=\"tiny\"}} {}",
                hashes.len()
            )),
            "query-block denominator must be recorded; got:\n{rendered}"
        );
        assert!(
            rendered.contains(&format!(
                "sgl_router_overlap_blocks_sum{{model_id=\"tiny\"}} {}",
                hashes.len()
            )),
            "matched-block numerator must use overlap_blocks_sum; got:\n{rendered}"
        );
        assert!(
            rendered.contains(
                "sgl_router_cache_aware_decisions_total{model_id=\"tiny\",decision=\"cache_hit\"} 1"
            ),
            "cache-overlap selection must record its terminal decision; got:\n{rendered}"
        );
    }

    /// Production wiring path: the policy is stored as `Arc<dyn Policy>` in a
    /// `PolicyRegistry`, then `PolicyRegistry::attach_metrics` injects the
    /// registry — exactly what `AppContext::with_active_load` does at startup.
    /// Exercises trait dispatch (the default no-op vs the `CacheAwareZmqPolicy`
    /// override) and the registry fan-out, neither of which the `with_metrics`
    /// builder test covers.
    #[test]
    fn attach_metrics_via_registry_records_overlap() {
        let tree = Arc::new(HashTree::new());
        let toks = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = toks.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        assert!(!hashes.is_empty());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
                ..Default::default()
            },
            tree,
            toks,
            oracle_for_tests(4),
        );
        let model = ModelId("tiny".into());
        let registry = crate::policies::PolicyRegistry::default();
        registry.insert(model.clone(), Arc::new(policy));

        // The production injection point — not the `with_metrics` builder.
        let metrics = MetricsRegistry::new();
        registry.attach_metrics(Arc::clone(&metrics));

        let chosen_policy = registry.get(&model).unwrap();
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let _ = chosen_policy.select(&workers, &ctx).expect("must pick");

        let rendered = metrics.render();
        assert!(
            rendered.contains("sgl_router_overlap_blocks_count{model_id=\"tiny\"}"),
            "PolicyRegistry::attach_metrics must wire overlap recording through the trait; got:\n{rendered}"
        );
        assert!(
            rendered.contains("sgl_router_cache_aware_query_blocks_total{model_id=\"tiny\"}"),
            "production metrics injection must wire the lookup denominator; got:\n{rendered}"
        );
        assert!(
            rendered.contains(
                "sgl_router_cache_aware_decisions_total{model_id=\"tiny\",decision=\"cache_hit\"} 1"
            ),
            "production metrics injection must wire policy decisions; got:\n{rendered}"
        );
    }

    /// The overlap observation is recorded *before* the cache-threshold branch,
    /// so low-overlap selections that fall back to min-load are still counted.
    /// `cache_threshold: 1.0` forces the fallback (match_rate is always <= 1.0)
    /// even on a full prefix match; assert the histogram is still observed AND
    /// the pick came from min-load (w1), not the cache-overlap worker (w0).
    #[test]
    fn overlap_recorded_even_when_selection_falls_back() {
        let tree = Arc::new(HashTree::new());
        let toks = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = toks.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        assert!(!hashes.is_empty());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let metrics = MetricsRegistry::new();
        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 1.0, // match_rate <= 1.0 always -> always fall back
                ..Default::default()
            },
            tree,
            toks,
            oracle_for_tests(4),
        )
        .with_metrics(Arc::clone(&metrics));

        // Bump w0's load so min-load picks w1 — distinguishing a min-load
        // fallback from the cache-overlap pick (which would be w0). Two guards
        // mirror `empty_tree_falls_back_to_min_load` (below the imbalance
        // threshold, so the cache-aware path is still reached).
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");

        assert_eq!(
            chosen.url, "http://w1:30000",
            "cache_threshold 1.0 must force a min-load fallback (w1), not the overlap worker (w0)"
        );
        let rendered = metrics.render();
        assert!(
            rendered.contains("sgl_router_overlap_blocks_count{model_id=\"tiny\"}"),
            "overlap must be recorded even on the below-threshold fallback; got:\n{rendered}"
        );
        assert!(
            rendered.contains(
                "sgl_router_cache_aware_decisions_total{model_id=\"tiny\",decision=\"below_threshold\"} 1"
            ),
            "below-threshold fallback reason must be visible; got:\n{rendered}"
        );
    }

    /// A full hash match can land on an intermediate tree node whose worker
    /// owners were removed while a descendant keeps the node alive. This is
    /// not a renderer/hash miss and must not be counted as below-threshold.
    #[test]
    fn full_match_without_owner_records_unowned_node() {
        let tree = Arc::new(HashTree::new());
        let tokens: Vec<u32> = (1..=12).collect();
        let hashes = compute_block_hashes(&tokens, 4);
        assert_eq!(hashes.len(), 3);
        let owner = KvWorkerId::new("http://w0:30000".into(), 0);
        tree.insert(&owner, None, &hashes);
        tree.remove(&owner, &[hashes[1]]);

        let metrics = MetricsRegistry::new();
        let policy = new_policy(
            cfg_default(),
            Arc::clone(&tree),
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        )
        .with_metrics(Arc::clone(&metrics));
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let query_tokens = &tokens[..8];
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(query_tokens));
        let _ = policy.select(&workers, &ctx).expect("min-load fallback");

        let rendered = metrics.render();
        assert!(rendered.contains(
            "sgl_router_cache_aware_decisions_total{model_id=\"tiny\",decision=\"matched_node_unowned\"} 1"
        ));
        assert!(
            !rendered.contains(
                "sgl_router_cache_aware_decisions_total{model_id=\"tiny\",decision=\"below_threshold\"}"
            ),
            "a full hash match with no owner is not a below-threshold parity miss; got:\n{rendered}",
        );
    }

    /// A matched tree owner can be absent from the eligible worker slice
    /// because admission filters workers at their in-flight cap before policy
    /// selection. Keep that designed load behavior distinct from an unowned
    /// tree node, which points to tree lifecycle state.
    #[test]
    fn matched_owner_outside_candidates_records_ineligible_worker() {
        let tree = Arc::new(HashTree::new());
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_block_hashes(&tokens, 4);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let metrics = MetricsRegistry::new();
        let policy = new_policy(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        )
        .with_metrics(Arc::clone(&metrics));
        let eligible_workers = vec![worker("http://w1:30000", "tiny")];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&tokens));
        let chosen = policy
            .select(&eligible_workers, &ctx)
            .expect("min-load fallback");

        assert_eq!(chosen.url, "http://w1:30000");
        assert!(metrics.render().contains(
            "sgl_router_cache_aware_decisions_total{model_id=\"tiny\",decision=\"matched_workers_ineligible\"} 1"
        ));
    }

    /// Each policy evaluation emits exactly one terminal decision even though
    /// different admission attempts for one request may evaluate the policy
    /// more than once.
    #[test]
    fn each_policy_evaluation_records_exactly_one_decision() {
        fn decision_total(rendered: &str) -> u64 {
            rendered
                .lines()
                .filter(|line| {
                    line.starts_with(
                        "sgl_router_cache_aware_decisions_total{model_id=\"tiny\",decision=",
                    )
                })
                .filter_map(|line| line.split_whitespace().last()?.parse::<u64>().ok())
                .sum()
        }

        let tree = Arc::new(HashTree::new());
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes = compute_block_hashes(&tokens, 4);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        let metrics = MetricsRegistry::new();
        let policy = new_policy(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        )
        .with_metrics(Arc::clone(&metrics));
        let workers = vec![worker("http://w0:30000", "tiny")];
        let model = ModelId("tiny".into());

        let no_workers = SelectionContext::new(&model, None);
        assert!(policy.select(&[], &no_workers).is_none());

        let no_body = SelectionContext::new(&model, None);
        let _ = policy
            .select(&workers, &no_body)
            .expect("min-load fallback");

        let cache_hit = SelectionContext::new(&model, None).with_request_tokens(Some(&tokens));
        let _ = policy.select(&workers, &cache_hit).expect("cache hit");

        let rendered = metrics.render();
        for decision in ["no_workers", "request_body_unavailable", "cache_hit"] {
            assert!(
                rendered.contains(&format!(
                    "sgl_router_cache_aware_decisions_total{{model_id=\"tiny\",decision=\"{decision}\"}} 1"
                )),
                "expected exactly one {decision} decision; got:\n{rendered}",
            );
        }
        assert_eq!(
            decision_total(&rendered),
            3,
            "three policy evaluations must emit three total decisions; got:\n{rendered}",
        );
    }

    /// End-to-end bigram wiring (the fix that takes `overlap_blocks_sum` from
    /// 0 to non-zero for EAGLE models): an EAGLE worker publishes its blocks
    /// under BIGRAM hashes. Only a router whose oracle reports `is_bigram` —
    /// and thus hashes its query with the bigram hasher — matches them, so
    /// overlap is non-zero and it picks the cached worker. A unigram-hashing
    /// router against the SAME tree matches nothing (overlap recorded as 0).
    #[test]
    fn bigram_routing_matches_only_with_bigram_hashing() {
        fn overlap_sum(rendered: &str) -> f64 {
            rendered
                .lines()
                .find(|l| l.starts_with("sgl_router_overlap_blocks_sum{model_id=\"tiny\"}"))
                .and_then(|l| l.split_whitespace().last())
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(-1.0)
        }

        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        // The EAGLE worker publishes BIGRAM block hashes.
        let bigram_hashes = compute_block_hashes_bigram(&ids, block_size as usize);
        assert!(!bigram_hashes.is_empty());
        assert_ne!(
            bigram_hashes,
            compute_block_hashes(&ids, block_size as usize),
            "bigram and unigram hashes must differ for this prefix"
        );
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": text })).unwrap();

        // Bigram-aware router (oracle.is_bigram == true): query hashes match
        // the bigram tree -> overlap > 0 and it picks the matched worker w0.
        {
            let tree = Arc::new(HashTree::new());
            tree.insert(
                &KvWorkerId::new("http://w0:30000".into(), 0),
                None,
                &bigram_hashes,
            );
            let oracle = BlockSizeOracle::new();
            oracle.try_set(block_size).unwrap();
            oracle.set_bigram(true);
            let metrics = MetricsRegistry::new();
            let policy = new_policy(
                CacheAwareConfig {
                    cache_threshold: 0.0,
                    ..Default::default()
                },
                tree,
                Arc::clone(&registry),
                oracle,
            )
            .with_metrics(Arc::clone(&metrics));
            let workers = vec![
                worker("http://w0:30000", "tiny"),
                worker("http://w1:30000", "tiny"),
            ];
            let ctx = SelectionContext::new(&model, Some(&body));
            let chosen = policy.select(&workers, &ctx).expect("must pick");
            assert_eq!(
                chosen.url, "http://w0:30000",
                "bigram-aware router must match w0's bigram-hashed prefix"
            );
            assert!(
                overlap_sum(&metrics.render()) > 0.0,
                "overlap_blocks_sum must be > 0 once the router hashes with bigram"
            );
        }

        // Unigram router vs the SAME bigram tree:
        // query hashes never match -> overlap recorded as 0.
        {
            let tree = Arc::new(HashTree::new());
            tree.insert(
                &KvWorkerId::new("http://w0:30000".into(), 0),
                None,
                &bigram_hashes,
            );
            let oracle = BlockSizeOracle::new();
            oracle.try_set(block_size).unwrap();
            oracle.set_bigram(false);
            let metrics = MetricsRegistry::new();
            let policy = new_policy(
                CacheAwareConfig {
                    cache_threshold: 0.0,
                    ..Default::default()
                },
                tree,
                Arc::clone(&registry),
                oracle,
            )
            .with_metrics(Arc::clone(&metrics));
            let workers = vec![
                worker("http://w0:30000", "tiny"),
                worker("http://w1:30000", "tiny"),
            ];
            let ctx = SelectionContext::new(&model, Some(&body));
            let _ = policy.select(&workers, &ctx).expect("must pick");
            assert_eq!(
                overlap_sum(&metrics.render()),
                0.0,
                "unigram hashing matches nothing in a bigram tree -> overlap_sum == 0"
            );
        }
    }

    /// A chat-completions request on a model with a chat template must route by
    /// the **chat-templated** tokens (BOS + role markers + content) — the tokens
    /// the engine actually cached — not by the raw joined content. Worker w0
    /// published its blocks under the templated tokens; only a router that
    /// renders the same template hashes a matching query. Hashing the raw
    /// content instead would match nothing, leaving live `overlap_blocks_sum`
    /// at 0 for chat traffic.
    #[test]
    fn chat_request_routes_by_templated_tokens() {
        let registry = tokenizer_registry_with_tiny();
        let template = serde_json::json!({
            "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}<|assistant|>",
            "bos_token": "<s>",
        });
        registry.attach_chat_template_for_test("tiny", &template);

        let messages = serde_json::json!([{"role":"user","content":"hello world hello world"}]);
        // Engine-side blocks are keyed on tokenize(render(messages)).
        let templated_tokens = registry.encode_chat("tiny", &messages).unwrap();
        let block_size = 4u32;
        let templated_hashes = compute_block_hashes(&templated_tokens, block_size as usize);
        assert!(
            !templated_hashes.is_empty(),
            "templated prompt must produce at least one block"
        );

        let tree = Arc::new(HashTree::new());
        tree.insert(
            &KvWorkerId::new("http://w0:30000".into(), 0),
            None,
            &templated_hashes,
        );

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
                ..Default::default()
            },
            tree,
            registry,
            oracle_for_tests(block_size),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": messages,
        }))
        .unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "chat request must route by chat-templated tokens to the worker holding that prefix"
        );
    }

    /// Templated and raw-content hashings must genuinely differ, confirming
    /// the chat-template path does real work (a no-op template would make this
    /// assertion fail, and raw-content hashes would miss the engine's
    /// templated blocks).
    #[test]
    fn chat_templated_hashes_differ_from_raw_content_hashes() {
        let registry = tokenizer_registry_with_tiny();
        let template = serde_json::json!({
            "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}<|assistant|>",
            "bos_token": "<s>",
        });
        registry.attach_chat_template_for_test("tiny", &template);
        let content = "hello world hello world";
        let messages = serde_json::json!([{"role":"user","content":content}]);

        let templated = registry.encode_chat("tiny", &messages).unwrap();
        let raw = adapter::encode(&registry.get("tiny").unwrap(), content).unwrap();
        assert_ne!(
            compute_block_hashes(&templated, 4),
            compute_block_hashes(&raw, 4),
            "templated and raw-content block hashes must differ"
        );
    }

    /// The DeepSeek-V4 built-in encoder is dispatched for chat requests when a
    /// model has it (no Jinja template). The query tokens come from the V4
    /// encoder, so a worker holding that encoded prefix is matched. (The V4
    /// markers aren't special tokens in the tiny fixture, but the dispatch +
    /// routing wiring is what's under test; byte-exact V4 token parity is pinned
    /// by `dsv4`'s string goldens and validated live.)
    #[test]
    fn chat_request_routes_via_dsv4_encoder() {
        let registry = tokenizer_registry_with_tiny();
        registry.attach_chat_encoder_for_test("tiny", crate::tokenizer::ChatEncoder::DeepSeekV4);
        assert!(registry.has_chat_encoder("tiny"));

        let messages =
            serde_json::json!([{"role":"user","content":"hello world hello world hello world"}]);
        let encoded = registry.encode_chat("tiny", &messages).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&encoded, block_size as usize);
        assert!(!hashes.is_empty());

        let tree = Arc::new(HashTree::new());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
                ..Default::default()
            },
            tree,
            registry,
            oracle_for_tests(block_size),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({ "messages": messages })).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "dsv4 chat request must route by the V4-encoded prefix"
        );
    }

    /// Helper: a tree holding `content`'s RAW-tokenized block hashes on w0, the
    /// two workers, and a policy — the fixture the raw-fallback routing tests
    /// share. Returns (policy, workers, model).
    fn raw_prefix_fixture(
        registry: Arc<TokenizerRegistry>,
        content: &str,
    ) -> (CacheAwareZmqPolicy, Vec<Arc<Worker>>, ModelId) {
        let raw_tokens = adapter::encode(&registry.get("tiny").unwrap(), content).unwrap();
        let hashes = compute_block_hashes(&raw_tokens, 4);
        assert!(
            !hashes.is_empty(),
            "raw content must produce at least one block"
        );
        let tree = Arc::new(HashTree::new());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
                ..Default::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        (policy, workers, ModelId("tiny".into()))
    }

    /// Graceful degradation: a model that HAS a chat template whose render fails
    /// (here it always raises) must fall back to hashing the RAW content and
    /// still route by prefix — not error, not blindly min-load. Exercises the
    /// `request_tokens_for` fall-through that the leaf `encode_chat`-returns-None
    /// tests don't reach at the routing level.
    #[test]
    fn chat_render_failure_falls_back_to_raw_routing() {
        let registry = tokenizer_registry_with_tiny();
        registry.attach_chat_template_for_test(
            "tiny",
            &serde_json::json!({
                "chat_template": "{{ raise_exception('boom') }}",
                "bos_token": "<s>",
            }),
        );
        let content = "hello world hello world hello world";
        let (policy, workers, model) = raw_prefix_fixture(registry, content);
        let body = serde_json::to_vec(&serde_json::json!({
            "messages": [{"role": "user", "content": content}],
        }))
        .unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "a failed template render must degrade to raw-content routing"
        );
    }

    /// A chat request on a model WITHOUT a chat template routes by the raw
    /// joined `messages[*].content` — the common config where the model ships
    /// no `chat_template`. Covers the `request_tokens_for` path that skips the
    /// template block entirely for a `messages` body.
    #[test]
    fn chat_on_template_less_model_routes_by_raw_content() {
        let registry = tokenizer_registry_with_tiny(); // no template attached
        assert!(!registry.has_chat_encoder("tiny"));
        let content = "hello world hello world hello world";
        let (policy, workers, model) = raw_prefix_fixture(registry, content);
        let body = serde_json::to_vec(&serde_json::json!({
            "messages": [{"role": "user", "content": content}],
        }))
        .unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w0:30000");
    }

    /// A `/v1/completions` (`prompt`) request on a model that DOES have a chat
    /// template must still use the raw path — the template applies only to
    /// `messages` traffic. Guards the `messages`-presence gate in
    /// `request_tokens_for`.
    #[test]
    fn completions_prompt_on_templated_model_uses_raw_path() {
        let registry = tokenizer_registry_with_tiny();
        registry.attach_chat_template_for_test(
            "tiny",
            &serde_json::json!({
                "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}",
                "bos_token": "<s>",
            }),
        );
        let content = "hello world hello world hello world";
        let (policy, workers, model) = raw_prefix_fixture(registry, content);
        // `prompt` body (no `messages`) -> raw path, so it matches the raw tree.
        let body = serde_json::to_vec(&serde_json::json!({ "prompt": content })).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w0:30000");
    }

    /// Two workers both hold the prefix; the lower-load one wins.
    #[test]
    fn tie_break_by_lowest_active_load() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        assert!(!hashes.is_empty());
        // Both workers hold the prefix.
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        tree.insert(&KvWorkerId::new("http://w1:30000".into(), 0), None, &hashes);

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
                ..Default::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Bump w0 to load=1; w1 is at 0 — tiebreak picks w1.
        let _g = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// w0 holds the prefix but is heavily overloaded → imbalance branch
    /// skips cache-aware and picks w1.
    #[test]
    fn imbalanced_pool_skips_cache_check() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0, // would normally always match
                load_gate: LoadGate::FleetSpread {
                    abs_threshold: 5,
                    rel_threshold: 2.0,
                },
                ..Default::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Bump w0 well above the imbalance threshold.
        let mut guards = Vec::new();
        for _ in 0..20 {
            guards.push(w0.load_guard());
        }
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000", "imbalance must dominate");
    }

    /// Fresh engine-reported load drives the imbalance + min-load decision
    /// instead of the router-side in-flight counter. Both workers hold the
    /// prefix and have zero router-side load, so without engine load the
    /// tiebreak would pick w0 (stable order). Engine load says w0 is hot
    /// (50) and w1 is light (1) → the imbalance branch routes to w1.
    #[test]
    fn engine_load_overrides_active_load() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        tree.insert(&KvWorkerId::new("http://w1:30000".into(), 0), None, &hashes);

        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(50, 0), now);
        engine_load.set("http://w1:30000", 0, load_stat(1, 0), now);

        let policy = new_policy_with_load(
            CacheAwareConfig {
                cache_threshold: 0.0,
                load_gate: LoadGate::FleetSpread {
                    abs_threshold: 5,
                    rel_threshold: 2.0,
                },
                ..Default::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        // Router-side counters are both 0 — only engine load is skewed.
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w1:30000",
            "engine-reported load must drive selection",
        );
    }

    /// When load is balanced enough that the imbalance branch does NOT fire,
    /// the matched-set tiebreak still uses engine load: both workers hold the
    /// prefix, engine load says w1 is lighter → w1 wins. (Guards against a
    /// regression that reverted the tiebreak to `active_load()`.)
    #[test]
    fn matched_set_tiebreak_uses_engine_load() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        tree.insert(&KvWorkerId::new("http://w1:30000".into(), 0), None, &hashes);

        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(10, 0), now);
        engine_load.set("http://w1:30000", 0, load_stat(2, 0), now);

        let policy = new_policy_with_load(
            CacheAwareConfig {
                cache_threshold: 0.0,
                // High thresholds so the imbalance fast-path never fires (10 vs
                // 2) and selection reaches the matched-set tiebreak.
                load_gate: LoadGate::FleetSpread {
                    abs_threshold: 100,
                    rel_threshold: 100.0,
                },
                ..Default::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w1:30000",
            "matched-set tiebreak must use engine load",
        );
    }

    /// Recent dispatches made AFTER the engine's last snapshot are added on
    /// top of the reported load. Without this, repeated `select` calls in
    /// the same burst would all read the same "worker looks idle" engine
    /// number and all pile onto it before the gauge catches up. w0 looks
    /// lighter by the raw engine numbers alone (1 vs 3), but three slots
    /// claimed on w0 after the snapshot flip the effective load in w1's
    /// favor (1+3=4 > 3+0=3).
    #[test]
    fn recent_dispatches_are_added_on_top_of_engine_load() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        tree.insert(&KvWorkerId::new("http://w1:30000".into(), 0), None, &hashes);

        let engine_load = EngineLoadTable::new();
        let snapshot_at = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(1, 0), snapshot_at);
        engine_load.set("http://w1:30000", 0, load_stat(3, 0), snapshot_at);

        let policy = new_policy_with_load(
            CacheAwareConfig {
                cache_threshold: 0.0,
                // High thresholds so the imbalance fast-path never fires on
                // the raw engine numbers (1 vs 3) and selection reaches the
                // matched-set tiebreak, which also uses `load_of`.
                load_gate: LoadGate::FleetSpread {
                    abs_threshold: 100,
                    rel_threshold: 100.0,
                },
                ..Default::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Three requests dispatched to w0 AFTER the engine's snapshot —
        // exactly the "burst the engine hasn't reported back on yet" shape.
        let _g1 = w0.load_guard();
        let _g2 = w0.load_guard();
        let _g3 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w1:30000",
            "w0's effective load (1 engine + 3 recent = 4) must exceed w1's \
             (3 engine + 0 recent = 3), even though the raw engine numbers \
             alone favor w0",
        );
    }

    /// `load_of` must use the OLDEST rank's timestamp as the "since" cutoff
    /// for a multi-rank worker, not the newest — this pins the end-to-end
    /// wiring of the choice `EngineLoadTable::fresh_worker_state` makes (see
    /// its doc comment). A regression to "newest" would silently treat the
    /// dispatch below as already covered by rank1's later snapshot, even
    /// though rank0's older snapshot doesn't reflect it.
    #[test]
    fn load_of_uses_oldest_rank_timestamp_for_multi_rank_worker() {
        let engine_load = EngineLoadTable::new();
        let earlier = Instant::now();
        let w = worker("http://w:30000", "tiny");
        // Real sleeps, not synthetic `Instant` offsets: the dispatch's
        // timestamp is captured internally by `load_guard()` and isn't
        // injectable (see `worker.rs`'s `slots_acquired_since` tests for the
        // same reasoning).
        std::thread::sleep(Duration::from_millis(5));
        let _g = w.load_guard(); // dispatched strictly between earlier/later
        std::thread::sleep(Duration::from_millis(5));
        let later = Instant::now();
        engine_load.set("http://w:30000", 0, load_stat(1, 0), earlier);
        engine_load.set("http://w:30000", 1, load_stat(1, 0), later);

        let loads = WorkerLoads::from_engine(&engine_load, later);
        assert_eq!(
            loads.load_of(&w),
            3,
            "depth (1+1=2) plus the one dispatch made after the OLDEST \
             rank's timestamp = 3; using the newest rank's timestamp \
             instead would exclude that dispatch and wrongly give 2",
        );
    }

    /// A stale engine snapshot falls back to PURE `active_load()` — the
    /// recent-dispatch correction only applies alongside a fresh snapshot
    /// (see `load_of`'s `Some` branch). A regression that added
    /// `slots_acquired_since` to the fallback branch too would double-count
    /// this worker's own in-flight guards.
    #[test]
    fn load_of_fallback_does_not_add_recent_dispatches_on_top_of_active_load() {
        let engine_load = EngineLoadTable::new();
        let stale = Instant::now() - Duration::from_secs(3600);
        engine_load.set("http://w:30000", 0, load_stat(50, 0), stale);
        let w = worker("http://w:30000", "tiny");
        let _g1 = w.load_guard();
        let _g2 = w.load_guard();

        let loads = WorkerLoads::from_engine(&engine_load, Instant::now());
        assert_eq!(
            loads.load_of(&w),
            2,
            "must equal active_load() exactly (2) — not the stale depth \
             (50) plus anything, and not active_load() plus a second \
             correction",
        );
    }

    /// A stale engine snapshot is ignored: selection falls back to the
    /// router-side `active_load` counter. w0's (stale) engine load is high,
    /// but w1 carries a router-side guard, so fallback picks w0.
    #[test]
    fn stale_engine_load_falls_back_to_active_load() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);
        tree.insert(&KvWorkerId::new("http://w1:30000".into(), 0), None, &hashes);

        // Past the default freshness window; an hour ago is comfortably stale.
        let engine_load = EngineLoadTable::new();
        let stale = Instant::now() - Duration::from_secs(3600);
        engine_load.set("http://w0:30000", 0, load_stat(50, 0), stale);

        let policy = new_policy_with_load(
            CacheAwareConfig {
                cache_threshold: 0.0,
                ..Default::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Router-side: w1 has one in-flight request, w0 has none. With the
        // stale engine load ignored, the tiebreak picks w0 (load 0 < 1).
        let _g = w1.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "stale engine load must be ignored in favour of active_load",
        );
    }

    /// Tokenizer is missing for the requested model → fall back to
    /// min-load (no panic, no error).
    #[test]
    fn missing_tokenizer_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let empty_registry = Arc::new(TokenizerRegistry::default());
        let policy = new_policy(cfg_default(), tree, empty_registry, oracle_for_tests(4));
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"prompt":"hello"}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Missing body → fall back to min-load.
    #[test]
    fn missing_request_body_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Body present but no recognizable prompt field → fall back.
    #[test]
    fn body_without_prompt_field_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"frobnicate":42}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Body has a non-text shape that yields zero tokens → fall back.
    /// (Tokenizer always returns ≥0 ids; an empty string yields the
    /// empty vec, then `compute_block_hashes` returns empty too.)
    #[test]
    fn empty_text_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"prompt":""}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// Match rate below the threshold → fall back. Threshold = 0.99
    /// means the tree must match every single block; we insert an
    /// UNRELATED chain so the rate is 0.
    #[test]
    fn low_match_rate_falls_back_to_min_load() {
        let tree = Arc::new(HashTree::new());
        // Tree contains a chain unrelated to the test's request.
        tree.insert(
            &KvWorkerId::new("http://w0:30000".into(), 0),
            None,
            &[999, 998, 997],
        );

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.99,
                ..Default::default()
            },
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = br#"{"prompt":"hello world hello world hello world"}"#;
        let ctx = SelectionContext::new(&model, Some(body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w1:30000");
    }

    /// The whole point of sampling: with `min_load_choices = 1` every pick is
    /// one uniform draw, so two replicas (or one over time) stop converging
    /// on the same current minimum and the load diffuses. Over 200 selects a
    /// deterministic pick would land on one URL exclusively; sampling must
    /// visit both — the failing alternative is fixed, not statistical
    /// (single-draw visits either side with p ≥ 1/2 each select).
    #[test]
    fn min_load_sampled_pick_diffuses_across_near_min_workers() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            CacheAwareConfig {
                min_load_choices: 1,
                ..cfg_default()
            },
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            seen.insert(
                policy
                    .select(&workers, &ctx)
                    .expect("must pick")
                    .url
                    .clone(),
            );
        }
        assert_eq!(
            seen.len(),
            2,
            "single-draw picks must diffuse; got {seen:?}"
        );
    }

    /// Sampling relaxes the exact MINIMUM, never the tier: with the queue
    /// gate active and one worker queueing, a single-draw pick still cannot
    /// land on it while an unqueued worker exists.
    #[test]
    fn min_load_sampled_pick_respects_the_queue_tier() {
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(4, 4), now); // queueing
        engine_load.set("http://w1:30000", 0, load_stat(15, 0), now); // deep, unqueued
        let policy = new_policy_with_load(
            CacheAwareConfig {
                min_load_choices: 1,
                ..queue_cfg(2)
            },
            Arc::new(HashTree::new()),
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
            engine_load,
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        for _ in 0..50 {
            assert_eq!(
                policy.select(&workers, &ctx).expect("must pick").url,
                "http://w1:30000",
                "a queueing worker must never win while an unqueued one exists",
            );
        }
    }

    /// `choices >= pool` is the old exact minimum, kept as the escape hatch
    /// the deployment can set without a rebuild.
    #[test]
    fn min_load_full_sample_is_the_exact_minimum() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            CacheAwareConfig {
                min_load_choices: 2,
                ..cfg_default()
            },
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let _g = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        for _ in 0..50 {
            assert_eq!(
                policy.select(&workers, &ctx).expect("must pick").url,
                "http://w1:30000",
                "a full sample must rank the whole pool and take the minimum",
            );
        }
    }

    /// Reach, not shape: with five workers every one must be a possible
    /// single-draw pick (P(any worker missed) ≤ 5·(4/5)⁵⁰⁰ ≈ 10⁻⁴⁸). This is
    /// the test the 2-worker diffusion test CANNOT be: on n=2 a biased
    /// sampler that only ever reaches two of the candidates still passes
    /// "both workers visited", so only a ≥4-worker fleet detects a sampler
    /// reading the wrong end of a partial shuffle.
    #[test]
    fn min_load_sampled_pick_reaches_every_worker_on_five_workers() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            CacheAwareConfig {
                min_load_choices: 1,
                ..cfg_default()
            },
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let workers: Vec<_> = (0..5)
            .map(|i| worker(&format!("http://w{i}:30000"), "tiny"))
            .collect();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..500 {
            seen.insert(
                policy
                    .select(&workers, &ctx)
                    .expect("must pick")
                    .url
                    .clone(),
            );
        }
        assert_eq!(
            seen.len(),
            5,
            "every worker must be reachable by the sampler; got {seen:?}"
        );
    }

    /// The shipped default path: 1 < k < pool. Deterministic to assert
    /// against because the deepest worker can only win a sampled pick when
    /// it shares the sample with no one lighter — impossible with three
    /// lighter workers and two draws, so "deepest never picked" has no
    /// statistical tail at all.
    #[test]
    fn min_load_power_of_two_never_picks_the_deepest_of_four() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            CacheAwareConfig {
                min_load_choices: 2,
                ..cfg_default()
            },
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let workers: Vec<_> = (0..4)
            .map(|i| worker(&format!("http://w{i}:30000"), "tiny"))
            .collect();
        let _heavy = workers[3].load_guard();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        for _ in 0..100 {
            assert_ne!(
                policy.select(&workers, &ctx).expect("must pick").url,
                "http://w3:30000",
                "a 2-sample can always dodge the single deepest worker",
            );
        }
    }

    /// A full-tie fleet (all loads equal) is where "choices >= pool
    /// recovers the deterministic exact minimum" pays the helpers back:
    /// the pick is stable across selects under the usize::MAX pin, which a
    /// shuffled full-sample implementation would break without failing any
    /// assertion that pins load skew.
    #[test]
    fn min_load_full_sample_is_deterministic_under_ties() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            cfg_default(),
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        // Pool order, not just "some fixed worker": the no-shuffle full
        // sample preserves the pre-sampling tiebreak, and pinning the URL
        // (not just cross-select stability) guards that contract.
        let first = policy
            .select(&workers, &ctx)
            .expect("must pick")
            .url
            .clone();
        assert_eq!(first, "http://w0:30000");
        for _ in 0..50 {
            assert_eq!(policy.select(&workers, &ctx).expect("must pick").url, first);
        }
    }

    /// Where `min_load_sampled_pick_respects_the_queue_tier` degenerates
    /// (its tier-1 pool is a singleton, so no sampler runs), this one
    /// exercises the actual draw: two unqueued workers + one queueing, a
    /// single-draw pick must stay inside the tier over 100 selects.
    /// P(queueing worker surviving under a whole-fleet-sampling regression)
    /// = (2/3)¹⁰⁰ ≈ 10⁻¹⁸.
    #[test]
    fn min_load_sampled_draw_stays_inside_the_unqueued_tier() {
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(4, 4), now); // queueing
        engine_load.set("http://w1:30000", 0, load_stat(9, 0), now); // unqueued
        engine_load.set("http://w2:30000", 0, load_stat(15, 0), now); // unqueued
        let policy = new_policy_with_load(
            CacheAwareConfig {
                min_load_choices: 1,
                ..queue_cfg(2)
            },
            Arc::new(HashTree::new()),
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
            engine_load,
        );
        let workers: Vec<_> = (0..3)
            .map(|i| worker(&format!("http://w{i}:30000"), "tiny"))
            .collect();
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            let picked = policy.select(&workers, &ctx).expect("must pick");
            assert_ne!(
                picked.url, "http://w0:30000",
                "a queueing worker must never win while an unqueued one exists",
            );
            seen.insert(picked.url.clone());
        }
        assert_eq!(
            seen.len(),
            2,
            "both unqueued workers are reachable draws; got {seen:?}"
        );
    }

    /// The CLI is NonZeroUsize, but a programmatically constructed 0 is
    /// clamped to a single draw rather than passed to `partial_shuffle` as
    /// an empty sample. Pins the documented clamp.
    #[test]
    fn min_load_zero_choices_clamps_to_a_single_draw() {
        let tree = Arc::new(HashTree::new());
        let policy = new_policy(
            CacheAwareConfig {
                min_load_choices: 0,
                ..cfg_default()
            },
            tree,
            tokenizer_registry_with_tiny(),
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let workers = vec![Arc::clone(&w0)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, None);
        assert_eq!(
            policy.select(&workers, &ctx).expect("must pick").url,
            "http://w0:30000",
        );
    }

    /// Sampling relaxes the exact minimum, never the saturation signal: at
    /// choices=1 with every worker queueing, a pick landing on a NON-owner
    /// is now the common case — but the routing could not have used locality
    /// anyway (no unqueued worker existed), so the decision must still be
    /// `cache_hit_all_queued`, not `cache_worker_queued`, or the fleet-
    /// saturation label goes silent exactly when it should climb.
    #[test]
    fn queue_gate_all_queued_classification_keys_on_saturation_not_the_draw() {
        let registry = tokenizer_registry_with_tiny();
        let (tree, body) = queue_fixture(&registry, &["http://w0:30000"]);
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(8, 5), now); // owner, queueing
        engine_load.set("http://w1:30000", 0, load_stat(1, 5), now); // non-owner, queueing, shallower
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(&body));

        let metrics = MetricsRegistry::new();
        let policy = new_policy_with_load(
            CacheAwareConfig {
                min_load_choices: 1,
                ..queue_cfg(4)
            },
            Arc::clone(&tree),
            Arc::clone(&registry),
            oracle_for_tests(4),
            Arc::clone(&engine_load),
        )
        .with_metrics(Arc::clone(&metrics));
        // Tier 2 must stay routable: every pick is Some, and it may land on
        // either worker — that draw is the saturation this label captures.
        for _ in 0..50 {
            policy
                .select(&workers, &ctx)
                .expect("tier 2 must keep routing");
        }
        let rendered = metrics.render();
        assert!(
            rendered.contains(
                "sgl_router_cache_aware_decisions_total{model_id=\"tiny\",decision=\"cache_hit_all_queued\"} 50"
            ),
            "every all-queued pick is saturation regardless of where the draw lands; got:\n{rendered}"
        );
        assert!(
            !rendered.contains("decision=\"cache_worker_queued\""),
            "nothing was diverted — no unqueued worker existed; got:\n{rendered}"
        );
    }

    /// Byte-slice helper over the shared `extract_prompt_text_from_value` free
    /// function, so the extraction-shape tests below stay terse.
    fn extract_prompt_text(body: &[u8]) -> Option<String> {
        let v: serde_json::Value = serde_json::from_slice(body).ok()?;
        crate::policies::extract_prompt_text_from_value(&v)
    }

    /// Chat completions shape with `messages[*].content` string.
    #[test]
    fn extract_prompt_chat_string_content() {
        let body = br#"{"model":"x","messages":[{"role":"user","content":"hello"}]}"#;
        let s = extract_prompt_text(body).unwrap();
        assert_eq!(s, "hello");
    }

    /// Chat completions shape with multimodal content blocks (text parts).
    #[test]
    fn extract_prompt_chat_block_content() {
        let body = br#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":"x"}]}]}"#;
        let s = extract_prompt_text(body).unwrap();
        assert_eq!(s, "hi");
    }

    /// `/v1/completions` array form is joined with newlines.
    #[test]
    fn extract_prompt_completions_array() {
        let body = br#"{"prompt":["a","b","c"]}"#;
        let s = extract_prompt_text(body).unwrap();
        assert_eq!(s, "a\nb\nc");
    }

    /// SGLang native `text` field.
    #[test]
    fn extract_prompt_sglang_text_field() {
        let body = br#"{"text":"abc"}"#;
        let s = extract_prompt_text(body).unwrap();
        assert_eq!(s, "abc");
    }

    /// Unknown shape → None.
    #[test]
    fn extract_prompt_unknown_shape_returns_none() {
        let body = br#"{"frobnicate":42}"#;
        assert!(extract_prompt_text(body).is_none());
    }

    /// Lifecycle: removing a worker from the tree via `clear_worker`
    /// makes subsequent matches miss; the policy then falls back to
    /// min-load.
    #[test]
    fn lifecycle_clear_worker_removes_overlap() {
        let tree = Arc::new(HashTree::new());
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, text).unwrap();
        let block_size = 4u32;
        let hashes = compute_block_hashes(&ids, block_size as usize);
        let kw0 = KvWorkerId::new("http://w0:30000".into(), 0);
        tree.insert(&kw0, None, &hashes);

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
                ..Default::default()
            },
            tree.clone(),
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let body = serde_json::to_vec(&serde_json::json!({"prompt": text})).unwrap();

        // Before clear: w0 wins.
        let ctx = SelectionContext::new(&model, Some(&body));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen.url, "http://w0:30000");

        // After clear: tree no longer attributes the prefix to w0.
        tree.clear_worker(&kw0);
        // Bump w0's load so min-load fallback distinguishes from w1.
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let chosen2 = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(chosen2.url, "http://w1:30000");
    }

    /// `request_tokens_for` flags chat-encoder output as engine-equivalent (safe
    /// to forward to the engine as `input_ids`): the ids match what the engine
    /// tokenizes from its own chat template.
    #[test]
    fn request_tokens_chat_encoder_is_engine_equivalent() {
        let registry = tokenizer_registry_with_tiny();
        registry.attach_chat_template_for_test(
            "tiny",
            &serde_json::json!({
                "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}",
                "bos_token": "<s>",
            }),
        );
        let messages = serde_json::json!([{"role":"user","content":"hello world"}]);
        let expected = registry.encode_chat("tiny", &messages).unwrap();

        let model = ModelId("tiny".into());
        let value = serde_json::json!({ "model": "tiny", "messages": messages });
        let rt = request_tokens_for(&registry, &model, &value).expect("tokens");
        assert!(
            rt.engine_equivalent,
            "chat-encoder ids must be engine-equivalent"
        );
        assert_eq!(rt.ids, expected);
    }

    /// `request_tokens_for` on the raw-prompt path (no chat encoder) is NOT
    /// engine-equivalent — the engine would still apply its template, so the
    /// router's raw ids must not be forwarded as `input_ids`.
    #[test]
    fn request_tokens_raw_prompt_not_engine_equivalent() {
        let registry = tokenizer_registry_with_tiny(); // no template attached
        assert!(!registry.has_chat_encoder("tiny"));
        let model = ModelId("tiny".into());
        let value = serde_json::json!({ "prompt": "hello world" });
        let rt = request_tokens_for(&registry, &model, &value).expect("tokens");
        assert!(!rt.engine_equivalent);
        assert!(!rt.ids.is_empty());
    }

    /// `request_tokens_for` returns `None` when there is no routable prompt
    /// field — the handler then forwards nothing and the engine tokenizes as
    /// usual.
    #[test]
    fn request_tokens_none_for_unroutable_body() {
        let registry = tokenizer_registry_with_tiny();
        let model = ModelId("tiny".into());
        let value = serde_json::json!({ "frobnicate": 42 });
        assert!(request_tokens_for(&registry, &model, &value).is_none());
    }

    /// `select` consumes the ingress-precomputed tokens and does NOT
    /// re-tokenize the body: the body here tokenizes to an unrelated prefix
    /// (which the tree does not hold), but the ctx tokens point at w0's cached
    /// prefix, so w0 wins. If `select` re-tokenized the body it would miss and
    /// fall back to min-load (w1).
    #[test]
    fn select_prefers_ingress_tokens_over_body() {
        let registry = tokenizer_registry_with_tiny();
        let text = "hello world hello world hello world";
        let tok = registry.get("tiny").unwrap();
        let tree_ids = adapter::encode(&tok, text).unwrap();
        let hashes = compute_block_hashes(&tree_ids, 4);
        assert!(!hashes.is_empty());
        let tree = Arc::new(HashTree::new());
        tree.insert(&KvWorkerId::new("http://w0:30000".into(), 0), None, &hashes);

        let policy = new_policy(
            CacheAwareConfig {
                cache_threshold: 0.0,
                ..Default::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        // Load w0 so a min-load fallback would pick w1 — distinguishes "used
        // ctx tokens (w0)" from "re-tokenized the body and missed (w1)".
        let _g = w0.load_guard();
        let _g2 = w0.load_guard();
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        // Body tokenizes to an unrelated prefix the tree does NOT hold.
        let body = serde_json::to_vec(&serde_json::json!({"prompt":"zzz unrelated"})).unwrap();
        let ctx = SelectionContext::new(&model, Some(&body)).with_request_tokens(Some(&tree_ids));
        let chosen = policy.select(&workers, &ctx).expect("must pick");
        assert_eq!(
            chosen.url, "http://w0:30000",
            "select must use ctx tokens (w0's prefix), not re-tokenize the body"
        );
    }

    // ---- per-worker queue gate ----

    const QUEUE_TEXT: &str = "hello world hello world hello world";

    fn queue_cfg(limit: usize) -> CacheAwareConfig {
        CacheAwareConfig {
            cache_threshold: 0.0, // any match counts; the gate is what's under test
            load_gate: LoadGate::PerWorkerQueue(NonZeroUsize::new(limit).expect("limit > 0")),
            min_load_choices: usize::MAX,
        }
    }

    /// Fleet-spread config with the shipped defaults, for the side-by-side arms
    /// that characterise the strategy the queue gate replaces.
    fn spread_cfg() -> CacheAwareConfig {
        CacheAwareConfig {
            cache_threshold: 0.0,
            min_load_choices: usize::MAX,
            ..Default::default()
        }
    }

    /// Seed `owners` as prefix holders for [`QUEUE_TEXT`] and return the tree
    /// plus the encoded request body.
    fn queue_fixture(
        registry: &Arc<TokenizerRegistry>,
        owners: &[&str],
    ) -> (Arc<HashTree>, Vec<u8>) {
        let tree = Arc::new(HashTree::new());
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, QUEUE_TEXT).unwrap();
        let hashes = compute_block_hashes(&ids, 4);
        assert!(
            !hashes.is_empty(),
            "fixture must produce at least one block"
        );
        for owner in owners {
            tree.insert(&KvWorkerId::new((*owner).into(), 0), None, &hashes);
        }
        let body = serde_json::to_vec(&serde_json::json!({"prompt": QUEUE_TEXT})).unwrap();
        (tree, body)
    }

    /// The gate reads the queue, not total depth. A cache home that is busy but
    /// draining — high `running`, empty queue — keeps its traffic; sending it
    /// elsewhere would trade a warm prefix for nothing.
    #[test]
    fn queue_gate_keeps_affinity_on_a_busy_but_unqueued_cache_home() {
        let registry = tokenizer_registry_with_tiny();
        let (tree, body) = queue_fixture(&registry, &["http://w0:30000"]);
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        // w0 is far deeper than w1, but nothing is waiting on it.
        engine_load.set("http://w0:30000", 0, load_stat(30, 0), now);
        engine_load.set("http://w1:30000", 0, load_stat(1, 0), now);

        let policy = new_policy_with_load(
            queue_cfg(4),
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(&body));
        assert_eq!(
            policy.select(&workers, &ctx).expect("must pick").url,
            "http://w0:30000",
            "depth 30 with an empty queue must not cost the request its cache home",
        );
    }

    /// The case a depth threshold cannot express, and the reason this gate is
    /// keyed on the queue: engines have been observed queueing while running
    /// well under their concurrency cap. Here the cache home is SHALLOWER than
    /// the alternative — depth 13 vs 20 — yet it is the one that would make the
    /// request wait. Any depth-based gate keeps feeding it; the queue gate does
    /// not, and the fleet-spread arm does not either.
    #[test]
    fn queue_gate_diverts_a_shallow_but_queueing_cache_home() {
        let registry = tokenizer_registry_with_tiny();
        let (tree, body) = queue_fixture(&registry, &["http://w0:30000"]);
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(8, 5), now); // depth 13, queue 5
        engine_load.set("http://w1:30000", 0, load_stat(20, 0), now); // depth 20, queue 0

        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(&body));

        let metrics = MetricsRegistry::new();
        let policy = new_policy_with_load(
            queue_cfg(4),
            Arc::clone(&tree),
            Arc::clone(&registry),
            oracle_for_tests(4),
            Arc::clone(&engine_load),
        )
        .with_metrics(Arc::clone(&metrics));
        assert_eq!(
            policy.select(&workers, &ctx).expect("must pick").url,
            "http://w1:30000",
            "a queue of 5 disqualifies the cache home even though it is the shallower worker",
        );
        let rendered = metrics.render();
        assert!(
            rendered.contains(
                "sgl_router_cache_aware_decisions_total{model_id=\"tiny\",decision=\"cache_worker_queued\"} 1"
            ),
            "the diversion must be attributable to the queue gate; got:\n{rendered}"
        );

        let spread = new_policy_with_load(
            spread_cfg(),
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        assert_eq!(
            spread.select(&workers, &ctx).expect("must pick").url,
            "http://w0:30000",
            "the fleet-spread strategy keeps feeding the queueing worker: spread is only 7",
        );
    }

    /// Boundary: the limit is a ceiling on the queue, not a preference. A cache
    /// home one under it keeps the request.
    #[test]
    fn queue_gate_boundary_is_at_the_limit_not_below_it() {
        let registry = tokenizer_registry_with_tiny();
        let model = ModelId("tiny".into());
        for (waiting, expected) in [(3usize, "http://w0:30000"), (4, "http://w1:30000")] {
            let (tree, body) = queue_fixture(&registry, &["http://w0:30000"]);
            let engine_load = EngineLoadTable::new();
            let now = Instant::now();
            engine_load.set("http://w0:30000", 0, load_stat(5, waiting as u64), now);
            engine_load.set("http://w1:30000", 0, load_stat(5, 0), now);
            let policy = new_policy_with_load(
                queue_cfg(4),
                tree,
                Arc::clone(&registry),
                oracle_for_tests(4),
                engine_load,
            );
            let workers = vec![
                worker("http://w0:30000", "tiny"),
                worker("http://w1:30000", "tiny"),
            ];
            let ctx = SelectionContext::new(&model, Some(&body));
            assert_eq!(
                policy.select(&workers, &ctx).expect("must pick").url,
                expected,
                "queue={waiting} against limit=4",
            );
        }
    }

    /// Locality survives through a second prefix owner: only the queueing one is
    /// skipped, and the request still lands on a worker holding the prefix
    /// rather than on the idle non-owner. Recorded as a hit, because it is one.
    #[test]
    fn queue_gate_prefers_an_unqueued_prefix_owner_over_an_idle_non_owner() {
        let registry = tokenizer_registry_with_tiny();
        let (tree, body) = queue_fixture(&registry, &["http://w0:30000", "http://w1:30000"]);
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(6, 9), now); // owner, queueing
        engine_load.set("http://w1:30000", 0, load_stat(18, 0), now); // owner, no queue
        engine_load.set("http://w2:30000", 0, load_stat(0, 0), now); // idle non-owner

        let metrics = MetricsRegistry::new();
        let policy = new_policy_with_load(
            queue_cfg(4),
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        )
        .with_metrics(Arc::clone(&metrics));
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
            worker("http://w2:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(&body));
        assert_eq!(
            policy.select(&workers, &ctx).expect("must pick").url,
            "http://w1:30000",
            "the unqueued prefix owner must win over an idle worker holding nothing",
        );
        let rendered = metrics.render();
        assert!(
            rendered.contains(
                "sgl_router_cache_aware_decisions_total{model_id=\"tiny\",decision=\"cache_hit\"} 1"
            ),
            "serving from a second prefix owner is still a cache hit; got:\n{rendered}"
        );
    }

    /// Fail-open: when every worker is queueing the policy must still return
    /// one. Filtering the min-load fallback too would return `None`, which the
    /// admission path turns into a hard selection failure on every request.
    #[test]
    fn queue_gate_never_returns_none_when_the_whole_fleet_is_queueing() {
        let registry = tokenizer_registry_with_tiny();
        let (tree, body) = queue_fixture(&registry, &["http://w0:30000"]);
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(20, 9), now);
        engine_load.set("http://w1:30000", 0, load_stat(9, 6), now);

        let metrics = MetricsRegistry::new();
        let policy = new_policy_with_load(
            queue_cfg(4),
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        )
        .with_metrics(Arc::clone(&metrics));
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(&body));
        assert_eq!(
            policy.select(&workers, &ctx).expect("must still route").url,
            "http://w1:30000",
            "an all-queueing fleet still routes, to the least loaded worker",
        );
        assert!(
            metrics.render().contains(
                "sgl_router_cache_aware_decisions_total{model_id=\"tiny\",decision=\"cache_hit_all_queued\"} 1"
            ),
            "every worker queueing is saturation, whatever the draw picks: with NO unqueued \
             alternative the gate vetoed nothing, so this is the saturation label, not a \
             diversion — and under sampled picks it lands off-owner almost always, which is \
             why the classification keys on saturation rather than where the draw landed",
        );
    }

    /// The fallback's second tier can land back on the queueing cache home when
    /// it is also the least loaded. Routing is then identical to affinity, so it
    /// is not a diversion — but it is reachable ONLY when no worker in the fleet
    /// is unqueued, so it gets its own label rather than being folded into
    /// `cache_hit`, where a fully saturated fleet would read as a healthy one.
    #[test]
    fn queue_gate_records_all_queued_when_the_fallback_lands_on_the_cache_home() {
        let registry = tokenizer_registry_with_tiny();
        let (tree, body) = queue_fixture(&registry, &["http://w0:30000"]);
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        // Both queueing; the cache home is also the fleet minimum by depth.
        engine_load.set("http://w0:30000", 0, load_stat(2, 5), now); // depth 7
        engine_load.set("http://w1:30000", 0, load_stat(30, 8), now); // depth 38

        let metrics = MetricsRegistry::new();
        let policy = new_policy_with_load(
            queue_cfg(4),
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        )
        .with_metrics(Arc::clone(&metrics));
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(&body));
        assert_eq!(
            policy.select(&workers, &ctx).expect("must pick").url,
            "http://w0:30000",
        );
        let rendered = metrics.render();
        assert!(
            rendered.contains(
                "sgl_router_cache_aware_decisions_total{model_id=\"tiny\",decision=\"cache_hit_all_queued\"} 1"
            ),
            "landing back on the cache home is a hit, but a saturated one; got:\n{rendered}"
        );
        assert!(
            !rendered.contains("decision=\"cache_worker_queued\""),
            "no locality was given up; got:\n{rendered}"
        );
        assert!(
            !rendered.contains("decision=\"cache_hit\"}"),
            "must NOT be indistinguishable from an unqueued hit — that is the \
             whole point of the separate label; got:\n{rendered}"
        );
    }

    /// Over-firing, the other half of what this replaces: a wide fleet spread
    /// makes `FleetSpread` bypass the cache for every request. The queue gate
    /// never consults the spread, so a cache home with no backlog keeps its
    /// traffic no matter how uneven the fleet is.
    #[test]
    fn queue_gate_ignores_fleet_spread_when_the_cache_home_is_not_queueing() {
        let registry = tokenizer_registry_with_tiny();
        let (tree, body) = queue_fixture(&registry, &["http://w0:30000"]);
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(10, 0), now);
        engine_load.set("http://w1:30000", 0, load_stat(2, 0), now);
        engine_load.set("http://w2:30000", 0, load_stat(45, 0), now); // spread 43

        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
            worker("http://w2:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(&body));

        let policy = new_policy_with_load(
            queue_cfg(4),
            Arc::clone(&tree),
            Arc::clone(&registry),
            oracle_for_tests(4),
            Arc::clone(&engine_load),
        );
        assert_eq!(
            policy.select(&workers, &ctx).expect("must pick").url,
            "http://w0:30000",
            "a wide spread must not divert a request whose cache home has no queue",
        );

        let spread = new_policy_with_load(
            spread_cfg(),
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        assert_eq!(
            spread.select(&workers, &ctx).expect("must pick").url,
            "http://w1:30000",
            "documents the behaviour the queue gate replaces",
        );
    }

    /// No fresh engine snapshot means no queue signal, and the router-side
    /// in-flight counter cannot supply one — it cannot tell a running request
    /// from a waiting one. The gate fails open rather than comparing the limit
    /// against a different quantity.
    #[test]
    fn queue_gate_fails_open_without_an_engine_snapshot() {
        let registry = tokenizer_registry_with_tiny();
        let (tree, body) = queue_fixture(&registry, &["http://w0:30000"]);
        // Empty engine-load table -> `waiting_of` is None for every worker.
        let policy = new_policy(queue_cfg(1), tree, registry, oracle_for_tests(4));

        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(&body));

        // Even a limit of 1 with router-side load piled on w0 cannot gate it:
        // there is no queue reading to compare against.
        let _g: Vec<_> = (0..10).map(|_| w0.load_guard()).collect();
        assert_eq!(
            policy.select(&workers, &ctx).expect("must pick").url,
            "http://w0:30000",
            "an unknown queue leaves the cache home eligible",
        );
    }

    /// The queue preference applies to EVERY min-load fallback, not just the
    /// matched-set diversion — including the below-threshold tree-miss path,
    /// which is the highest-volume one. Without this, replacing the gate with
    /// `None` at the seven early-fallback call sites leaves the suite green
    /// while every cache miss during a tokenizer or oracle outage piles onto
    /// whichever worker is shallowest, backlog and all.
    #[test]
    fn queue_gate_applies_to_the_below_threshold_fallback() {
        let registry = tokenizer_registry_with_tiny();
        let (tree, body) = queue_fixture(&registry, &["http://w0:30000"]);
        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        // w0 is the fleet minimum by depth but is queueing; w1 is deeper and idle.
        engine_load.set("http://w0:30000", 0, load_stat(2, 7), now); // depth 9
        engine_load.set("http://w1:30000", 0, load_stat(25, 0), now); // depth 25

        // cache_threshold 1.0 makes every match fall below the bar, so the
        // request takes the below-threshold path rather than the matched-set one.
        let metrics = MetricsRegistry::new();
        let policy = new_policy_with_load(
            CacheAwareConfig {
                cache_threshold: 1.0,
                load_gate: LoadGate::PerWorkerQueue(NonZeroUsize::new(4).unwrap()),
                ..Default::default()
            },
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        )
        .with_metrics(Arc::clone(&metrics));
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(&body));
        assert_eq!(
            policy.select(&workers, &ctx).expect("must pick").url,
            "http://w1:30000",
            "a below-threshold fallback must still avoid the queueing worker",
        );
        assert!(
            metrics.render().contains(
                "sgl_router_cache_aware_decisions_total{model_id=\"tiny\",decision=\"below_threshold\"} 1"
            ),
            "this must be exercising the below-threshold path, not the matched-set one",
        );
    }

    /// `waiting_of` deliberately carries no `slots_acquired_since` correction:
    /// a request this router just dispatched is not known to be QUEUED — the
    /// engine may well be running it — so adding it would manufacture queue
    /// depth that does not exist and turn the gate back into an over-firing
    /// depth-ish check under burst.
    #[test]
    fn queue_gate_does_not_count_this_routers_own_recent_dispatches_as_queued() {
        let registry = tokenizer_registry_with_tiny();
        let (tree, body) = queue_fixture(&registry, &["http://w0:30000"]);
        let engine_load = EngineLoadTable::new();
        // Snapshot taken in the past, reporting an empty queue.
        let earlier = Instant::now() - Duration::from_millis(500);
        engine_load.set("http://w0:30000", 0, load_stat(5, 0), earlier);
        engine_load.set("http://w1:30000", 0, load_stat(5, 0), earlier);

        let policy = new_policy_with_load(
            queue_cfg(4),
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        );
        let w0 = worker("http://w0:30000", "tiny");
        let w1 = worker("http://w1:30000", "tiny");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(&body));

        // Eight dispatches since that snapshot — well past the limit of 4 if
        // they were (wrongly) counted as queue depth.
        let _g: Vec<_> = (0..8).map(|_| w0.load_guard()).collect();
        assert_eq!(
            policy.select(&workers, &ctx).expect("must pick").url,
            "http://w0:30000",
            "recent dispatches raise depth, not queue — the gate must not fire on them",
        );
    }

    /// A real diversion records what it gave up, so the forgone-prefix
    /// distribution can be compared against the all-selections one. Without
    /// this there is no way to tell a gate that diverts a representative sample
    /// from one that is trading away large cached prefixes to dodge short
    /// queues.
    #[test]
    fn queue_gate_records_the_overlap_a_diversion_gave_up() {
        let registry = tokenizer_registry_with_tiny();
        let (tree, body) = queue_fixture(&registry, &["http://w0:30000"]);
        let tok = registry.get("tiny").unwrap();
        let ids = adapter::encode(&tok, QUEUE_TEXT).unwrap();
        let expected_blocks = compute_block_hashes(&ids, 4).len();

        let engine_load = EngineLoadTable::new();
        let now = Instant::now();
        engine_load.set("http://w0:30000", 0, load_stat(5, 9), now); // owner, queueing
        engine_load.set("http://w1:30000", 0, load_stat(5, 0), now);

        let metrics = MetricsRegistry::new();
        let policy = new_policy_with_load(
            queue_cfg(4),
            tree,
            registry,
            oracle_for_tests(4),
            engine_load,
        )
        .with_metrics(Arc::clone(&metrics));
        let workers = vec![
            worker("http://w0:30000", "tiny"),
            worker("http://w1:30000", "tiny"),
        ];
        let model = ModelId("tiny".into());
        let ctx = SelectionContext::new(&model, Some(&body));
        assert_eq!(
            policy.select(&workers, &ctx).expect("must pick").url,
            "http://w1:30000",
        );
        let rendered = metrics.render();
        assert!(
            rendered.contains(&format!(
                "sgl_router_diverted_overlap_blocks_sum{{model_id=\"tiny\"}} {expected_blocks}"
            )),
            "a diversion must record the prefix it forwent; got:\n{rendered}"
        );
        assert!(
            rendered.contains("sgl_router_diverted_overlap_blocks_count{model_id=\"tiny\"} 1"),
            "exactly one diversion observed; got:\n{rendered}"
        );
    }

    /// Selections that kept their cache home must NOT appear in the diverted
    /// histogram — otherwise it measures traffic rather than sacrifice, and the
    /// comparison against the all-selections curve is meaningless.
    #[test]
    fn queue_gate_records_no_diverted_overlap_when_affinity_is_kept() {
        let registry = tokenizer_registry_with_tiny();
        let model = ModelId("tiny".into());

        // (a) cache home has room -> plain hit.
        // (b) every worker is queueing and the fallback lands back on the home
        //     -> cache_hit_all_queued, which gave up nothing.
        for (home, other) in [((5u64, 0u64), (5u64, 0u64)), ((2, 5), (30, 8))] {
            let (tree, body) = queue_fixture(&registry, &["http://w0:30000"]);
            let engine_load = EngineLoadTable::new();
            let now = Instant::now();
            engine_load.set("http://w0:30000", 0, load_stat(home.0, home.1), now);
            engine_load.set("http://w1:30000", 0, load_stat(other.0, other.1), now);

            let metrics = MetricsRegistry::new();
            let policy = new_policy_with_load(
                queue_cfg(4),
                tree,
                Arc::clone(&registry),
                oracle_for_tests(4),
                engine_load,
            )
            .with_metrics(Arc::clone(&metrics));
            let workers = vec![
                worker("http://w0:30000", "tiny"),
                worker("http://w1:30000", "tiny"),
            ];
            let ctx = SelectionContext::new(&model, Some(&body));
            assert_eq!(
                policy.select(&workers, &ctx).expect("must pick").url,
                "http://w0:30000",
                "home={home:?} other={other:?} should keep the cache home",
            );
            let rendered = metrics.render();
            assert!(
                !rendered.contains("sgl_router_diverted_overlap_blocks_count{model_id=\"tiny\"} 1"),
                "no prefix was given up for home={home:?}; got:\n{rendered}"
            );
        }
    }
}
