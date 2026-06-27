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
//! Every load comparison below uses [`WorkerLoads::load_of`]: where a fresh
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
//! 1. **Load-imbalance fast-path.** If `max_load - min_load >
//!    balance_abs_threshold` AND `max_load > min_load *
//!    balance_rel_threshold`, skip the cache lookup and pick the
//!    lowest-load worker. This prevents one hot worker from dominating
//!    cache-aware selection while every other worker idles.
//! 2. **Routing tokens.** Prefer the ingress-precomputed ids
//!    (`ctx.request_tokens()`); fall back to tokenizing the body here
//!    (chat-encoder-aware for chat traffic, raw `prompt`/`text` otherwise)
//!    for callers that didn't pre-tokenize. On any failure (no tokens, no
//!    tokenizer, encode error, empty), fall through to step 4 (min-load).
//! 3. **Hash + match.** Compute block hashes via
//!    [`super::kv_events::compute_block_hashes`], query the shared hash tree
//!    for the longest matching prefix. If `match_rate > cache_threshold`,
//!    pick the lowest-load worker whose `url` appears in the match result.
//!    Otherwise, fall through.
//! 4. **Min-load fallback.** Pick the lowest-load worker.
//!
//! The implementation never returns `None` for a non-empty `workers` slice;
//! a misconfigured tree or tokenizer degrades to round-robin-with-load
//! tiebreak, not a routing failure.

use crate::config::CacheAwareConfig;

use crate::policies::engine_load::EngineLoadTable;
use crate::policies::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree,
};
use crate::policies::{request_tokens_for, Policy, SelectionContext};
use crate::server::metrics::MetricsRegistry;
use crate::tokenizer::TokenizerRegistry;
use crate::workers::Worker;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

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
    /// url -> (engine-reported depth, that snapshot's oldest-rank timestamp).
    fresh: HashMap<String, (usize, Instant)>,
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
            // are bounded by real admission/concurrency limits (a worker's
            // in-flight count is capped well below `usize::MAX` by
            // `SlotRegistry::try_claim`'s admission cap), so overflow here
            // is unreachable from real traffic — reaching it would mean a
            // problem (memory exhaustion, a corrupt engine payload) that is
            // already symptomatic elsewhere, not something worth a panic on
            // this per-request hot path.
            Some(&(engine_load, at)) => engine_load.saturating_add(w.slots_acquired_since(at)),
            None => w.active_load(),
        }
    }

    /// Number of workers whose load came from the engine (vs the router-side
    /// fallback). Used only to annotate the rebalance log.
    fn engine_worker_count(&self) -> usize {
        self.fresh.len()
    }
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
        let _ = self.metrics.set(metrics);
        self
    }

    /// Lowest-load worker by the per-selection load lookup — ties broken by
    /// stable iteration order (the order the registry returned, i.e.
    /// dashmap-undefined). For production traffic the ties are rare; tests
    /// pin the load skew.
    fn pick_min_load(workers: &[Arc<Worker>], loads: &WorkerLoads) -> Option<Arc<Worker>> {
        workers
            .iter()
            .min_by_key(|w| loads.load_of(w))
            .map(Arc::clone)
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
    fn balance_check(&self, workers: &[Arc<Worker>], loads: &WorkerLoads) -> BalanceCheck {
        let (min_load, max_load) = workers.iter().fold((usize::MAX, 0usize), |(mn, mx), w| {
            let l = loads.load_of(w);
            (mn.min(l), mx.max(l))
        });
        let min_load = if min_load == usize::MAX { 0 } else { min_load };
        let abs_diff = max_load.saturating_sub(min_load);
        let rel_threshold = (min_load as f32 * self.config.balance_rel_threshold) as usize;
        let imbalanced = abs_diff > self.config.balance_abs_threshold && max_load > rel_threshold;
        BalanceCheck {
            min_load,
            max_load,
            abs_diff,
            imbalanced,
        }
    }
}

impl Policy for CacheAwareZmqPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        if workers.is_empty() {
            return None;
        }

        // Per-selection load lookup: engine-reported queue depth where fresh,
        // else the router-side in-flight counter. One snapshot pass serves
        // every comparison below (imbalance check, min-load fallback,
        // matched-set tiebreak).
        let loads = WorkerLoads::from_engine(&self.engine_load, Instant::now());

        // 1. Load-imbalance fast-path: even the best cache hit gets
        //    dropped in favour of evening out load. Logged on every
        //    request (debug) so the input to the decision is auditable;
        //    the actual rebalance is logged at info when it fires.
        let balance = self.balance_check(workers, &loads);
        tracing::debug!(
            model = %ctx.model(),
            min_load = balance.min_load,
            max_load = balance.max_load,
            abs_diff = balance.abs_diff,
            balance_abs_threshold = self.config.balance_abs_threshold,
            balance_rel_threshold = self.config.balance_rel_threshold,
            imbalanced = balance.imbalanced,
            engine_load_workers = loads.engine_worker_count(),
            engine_load_expected = self.engine_load.expected_count(),
            "cache-aware-zmq: load-balance check considered",
        );
        if balance.imbalanced {
            let chosen = Self::pick_min_load(workers, &loads);
            if let Some(w) = &chosen {
                tracing::info!(
                    model = %ctx.model(),
                    worker = %w.url,
                    worker_load = loads.load_of(w),
                    min_load = balance.min_load,
                    max_load = balance.max_load,
                    abs_diff = balance.abs_diff,
                    balance_abs_threshold = self.config.balance_abs_threshold,
                    balance_rel_threshold = self.config.balance_rel_threshold,
                    engine_load_workers = loads.engine_worker_count(),
                    engine_load_expected = self.engine_load.expected_count(),
                    "cache-aware-zmq: load imbalance detected — bypassing cache, routing to min-load worker",
                );
            }
            return chosen;
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
                    _ => return Self::pick_min_load(workers, &loads),
                };
                let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else {
                    return Self::pick_min_load(workers, &loads);
                };
                let Some(rt) = request_tokens_for(&self.tokenizers, ctx.model(), &value) else {
                    return Self::pick_min_load(workers, &loads);
                };
                fallback_ids = rt.ids;
                &fallback_ids
            }
        };

        // 3. Hash + match.
        // Source block_size from the worker — the router can only hash
        // prompts at the block size the workers publish at. If no worker
        // has registered yet (oracle empty), cache-aware routing has no
        // ground truth to score against; fall back to min-load.
        let Some(block_size) = self.block_size_oracle.get() else {
            tracing::debug!(
                model = %ctx.model(),
                "cache-aware-zmq: block size unknown (no worker page_size yet), falling back to min-load",
            );
            return Self::pick_min_load(workers, &loads);
        };
        // EAGLE-family workers hash KV blocks over token bigrams; the query
        // hashes must match the worker's stored hashes or the tree lookup
        // always misses (overlap stays 0). The oracle carries the worker-
        // reported flag.
        let is_bigram = self.block_size_oracle.is_bigram();
        let block_hashes = if is_bigram {
            compute_block_hashes_bigram(tokens, block_size as usize)
        } else {
            compute_block_hashes(tokens, block_size as usize)
        };
        if block_hashes.is_empty() {
            return Self::pick_min_load(workers, &loads);
        }
        let matched = self.tree.match_prefix(None, &block_hashes);
        let match_rate = matched.matched_blocks as f32 / block_hashes.len() as f32;
        tracing::debug!(
            model = %ctx.model(),
            hashing = if is_bigram { "bigram" } else { "unigram" },
            n_blocks = block_hashes.len(),
            matched_blocks = matched.matched_blocks,
            match_rate,
            cache_threshold = self.config.cache_threshold,
            "cache-aware-zmq match_prefix",
        );
        // Record the matched overlap into `sgl_router_overlap_blocks` before
        // the threshold branch, so the histogram captures the full
        // distribution — including low-overlap selections that fall back to
        // min-load. This is the quantitative signal that cache-aware routing
        // is matching prefixes at all.
        if let Some(m) = self.metrics.get() {
            m.observe_overlap_blocks(ctx.model().0.as_str(), matched.matched_blocks as u64);
        }
        if match_rate <= self.config.cache_threshold || matched.workers.is_empty() {
            tracing::debug!(
                model = %ctx.model(),
                match_rate,
                cache_threshold = self.config.cache_threshold,
                "cache-aware-zmq: overlap below threshold, falling back to min-load",
            );
            return Self::pick_min_load(workers, &loads);
        }
        // Among workers in the matched set, pick the lowest-load one.
        let matched_urls: std::collections::HashSet<&str> =
            matched.workers.iter().map(|kw| kw.url.as_str()).collect();
        let best_matched: Option<Arc<Worker>> = workers
            .iter()
            .filter(|w| matched_urls.contains(w.url.as_str()))
            .min_by_key(|w| loads.load_of(w))
            .map(Arc::clone);
        let chosen = best_matched.or_else(|| Self::pick_min_load(workers, &loads));
        if let Some(w) = &chosen {
            tracing::debug!(
                model = %ctx.model(),
                worker = %w.url,
                matched_blocks = matched.matched_blocks,
                "cache-aware-zmq: selected worker by cache overlap",
            );
        }
        chosen
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        let _ = self.metrics.set(metrics);
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
    use std::time::Duration;

    fn cfg_default() -> CacheAwareConfig {
        CacheAwareConfig {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
        }
    }

    /// Helper: build a `BlockSizeOracle` already primed to the test's
    /// canonical block size (4). Mirrors what `KvEventIndex::add_worker`
    /// would do when the first real worker registers.
    fn oracle_for_tests(block_size: u32) -> Arc<BlockSizeOracle> {
        let o = BlockSizeOracle::new();
        o.try_set(block_size)
            .expect("fresh oracle accepts first set");
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
                ..Default::default()
            },
            observability: Default::default(),
            model: crate::config::ModelConfig {
                id: "tiny".into(),
                tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
                tokenizer_shards: 1,
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
            admission: crate::config::AdmissionConfig::default(),
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
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
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
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
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
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
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
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
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
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
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

        // Unigram router (default is_bigram == false) vs the SAME bigram tree:
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
            let metrics = MetricsRegistry::new();
            let policy = new_policy(
                CacheAwareConfig {
                    cache_threshold: 0.0,
                    balance_abs_threshold: 32,
                    balance_rel_threshold: 1.1,
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
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
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
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
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
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
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
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
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
                balance_abs_threshold: 5,
                balance_rel_threshold: 2.0,
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
                balance_abs_threshold: 5,
                balance_rel_threshold: 2.0,
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
                balance_abs_threshold: 100,
                balance_rel_threshold: 100.0,
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
                balance_abs_threshold: 100,
                balance_rel_threshold: 100.0,
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
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
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
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
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
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
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

        let policy = CacheAwareZmqPolicy::new(
            CacheAwareConfig {
                cache_threshold: 0.0,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
            },
            tree,
            registry,
            oracle_for_tests(4),
            EngineLoadTable::new(),
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
}
