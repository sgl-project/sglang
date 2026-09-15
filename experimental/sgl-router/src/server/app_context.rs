// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::Config;

use crate::policies::active_load::ActiveLoadRegistry;
use crate::policies::buckets::BucketSelector;
use crate::policies::engine_load::EngineLoadTable;
use crate::policies::kv_events::BlockSizeOracle;
use crate::policies::prefix_provider::RadixTreePrefixProvider;
use crate::policies::PolicyRegistry;
use crate::proxy::Proxy;
use crate::server::inflight::InflightHttp;
use crate::server::metrics::MetricsRegistry;
use crate::tokenizer::TokenizerRegistry;
use crate::workers::WorkerRegistry;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

/// `/readyz` readiness as a one-way door: `NOT_READY -> READY -> DRAINING`,
/// and never backwards. One atomic rather than a pair of bools so the latch is
/// the transition itself — `mark_ready`'s compare-exchange simply cannot
/// succeed from `DRAINING` — instead of an invariant stated in a doc comment
/// and enforced by nobody.
const READINESS_NOT_READY: u8 = 0;
const READINESS_READY: u8 = 1;
const READINESS_DRAINING: u8 = 2;

pub struct AppContext {
    pub config: Config,
    pub tokenizers: Arc<TokenizerRegistry>,
    pub proxy: Arc<Proxy>,
    pub registry: Arc<WorkerRegistry>,
    pub policies: Arc<PolicyRegistry>,
    /// Converts static Bucket configuration into request candidate domains.
    pub bucket_selector: Arc<BucketSelector>,
    /// Per-worker active-load bookkeeping shared by the proxy, policies,
    /// timeout janitor, and metrics.
    pub active_load: Arc<ActiveLoadRegistry>,
    /// Lightweight Prometheus-format metrics registry served via
    /// `/metrics`. Shared with the chat handler (requests_total),
    /// active-load registry, policy-specific counters, and PD dispatch.
    pub metrics: Arc<MetricsRegistry>,
    /// Shared Engine LoadStat table; ingress captures one immutable snapshot per request.
    pub engine_load: Arc<EngineLoadTable>,
    pub prefix_index: Option<Arc<dyn sgl_kv_indexer::PrefixIndex>>,
    pub radix_tree_prefix_provider: Option<RadixTreePrefixProvider>,
    pub block_size_oracle: Arc<BlockSizeOracle>,
    /// Open HTTP exchanges, on every route. What axum's graceful shutdown
    /// is actually waiting on during the drain — `active_load` sees only the
    /// proxied subset.
    pub inflight_http: Arc<InflightHttp>,
    readiness: AtomicU8,
}

impl AppContext {
    pub fn new(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
    ) -> Self {
        Self::with_active_load(
            config,
            tokenizers,
            proxy,
            registry,
            policies,
            ActiveLoadRegistry::with_defaults(),
        )
    }

    /// Construct an [`AppContext`] with an explicit [`ActiveLoadRegistry`].
    /// Production wires the default (5-minute timeout, SystemTimeClock)
    /// via [`Self::new`]; tests that exercise the janitor pass a registry
    /// built with a `MockClock`.
    pub fn with_active_load(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
        active_load: Arc<ActiveLoadRegistry>,
    ) -> Self {
        let metrics = MetricsRegistry::new();
        // Wire the per-worker active-load gauge so `sgl_router_active_load`
        // mirrors the live counter on every register / drop / sweep.
        // Without this, the metric is permanently 0 in production even
        // though the chat handler is faithfully calling `register`.
        active_load.attach_metrics(Arc::clone(&metrics));
        // The metrics registry is built after the policy registry, so attach
        // it here for policies that emit their own counters.
        policies.attach_metrics(Arc::clone(&metrics));
        let bucket_selector = Arc::new(BucketSelector::new(config.model.bucket_config.clone()));
        Self {
            config,
            tokenizers,
            proxy,
            registry,
            policies,
            bucket_selector,
            active_load,
            metrics,
            prefix_index: None,
            radix_tree_prefix_provider: None,
            block_size_oracle: BlockSizeOracle::new(),
            engine_load: EngineLoadTable::new(),
            inflight_http: InflightHttp::new(),
            readiness: AtomicU8::new(READINESS_NOT_READY),
        }
    }

    /// Report bootstrap as finished, unless the pod has already begun draining.
    /// The `DRAINING` state is a one-way door (see [`Self::mark_not_ready`]),
    /// and a compare-exchange is what enforces it: a plain store would let any
    /// later caller — a re-initialization path, a discovery-recovery hook —
    /// flip `/readyz` back to 200 seconds before the listener closes, re-arming
    /// the rolling-update race the drain exists to close.
    pub fn mark_ready(&self) {
        // Relaxed: this flag does not synchronize other state; readers only
        // care about eventual visibility, not happens-before with surrounding
        // ops. Failure means the state was already READY or is DRAINING —
        // correct in both cases, so the result is deliberately discarded.
        let _ = self.readiness.compare_exchange(
            READINESS_NOT_READY,
            READINESS_READY,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    /// Flip `/readyz` to 503, permanently. Called at the start of the SIGTERM
    /// drain so probes and any probe-driven load balancer see this pod as
    /// not-ready while the endpoint removal (triggered by the pod's
    /// `deletionTimestamp`, not by this flip) propagates. See
    /// [`crate::server::shutdown::drain_for_termination`] for which mechanism
    /// the pause is sized for.
    ///
    /// Not the inverse of [`mark_ready`](Self::mark_ready): this transition
    /// cannot be undone, because the process it announces cannot be either.
    pub fn mark_not_ready(&self) {
        self.readiness.store(READINESS_DRAINING, Ordering::Relaxed);
    }

    /// Whether bootstrap finished — only ONE term of the `/readyz` predicate,
    /// which also requires a non-empty worker registry (see
    /// `server::routes::health::readyz`).
    pub fn is_ready(&self) -> bool {
        self.readiness.load(Ordering::Relaxed) == READINESS_READY
    }

    #[cfg(test)]
    pub fn stub() -> Self {
        Self {
            config: Config {
                server: crate::config::ServerConfig {
                    host: "x".into(),
                    port: 0,
                    ..Default::default()
                },
                observability: Default::default(),
                model: crate::config::ModelConfig {
                    id: "stub-model".into(),
                    tokenizer_path: "stub".into(),
                    policy: crate::config::PolicyKind::RoundRobin,
                    decode_policy: Default::default(),
                    bucket_config: None,
                    circuit_breaker: None,
                    cache_aware: None,
                    sticky: None,
                    affinity: None,
                    fused: None,
                    eligibility: None,
                },
                discovery: crate::config::DiscoveryBackend::StaticUrls(
                    crate::config::StaticUrlsDiscoveryConfig {
                        urls: vec!["http://placeholder:0".into()],
                    },
                ),
                proxy: crate::config::ProxyConfig::default(),
                active_load: crate::config::ActiveLoadConfig::default(),
            },
            tokenizers: Arc::new(TokenizerRegistry::default()),
            proxy: Arc::new(Proxy::new(std::time::Duration::from_secs(60)).expect("stub proxy")),
            registry: Arc::new(WorkerRegistry::default()),
            policies: Arc::new(PolicyRegistry::default()),
            bucket_selector: Arc::new(BucketSelector::new(None)),
            active_load: ActiveLoadRegistry::with_defaults(),
            metrics: MetricsRegistry::new(),
            prefix_index: None,
            radix_tree_prefix_provider: None,
            block_size_oracle: BlockSizeOracle::new(),
            engine_load: EngineLoadTable::new(),
            inflight_http: InflightHttp::new(),
            readiness: AtomicU8::new(READINESS_NOT_READY),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_not_ready_flips_readiness_back_off() {
        let ctx = AppContext::stub();
        // stub starts not-ready; mark_ready is the readiness on-switch.
        ctx.mark_ready();
        assert!(ctx.is_ready(), "mark_ready must report ready");
        // The SIGTERM drain path needs the inverse so /readyz can flip to 503
        // before the server stops accepting.
        ctx.mark_not_ready();
        assert!(
            !ctx.is_ready(),
            "mark_not_ready must flip readiness back off",
        );
    }

    /// The safety-critical direction. A `mark_ready` reaching a draining pod
    /// would put `/readyz` back to 200 with the listener seconds from closing,
    /// silently re-arming the rolling-update race — so the latch is tested,
    /// not just documented.
    #[test]
    fn readiness_does_not_come_back_once_draining() {
        let ctx = AppContext::stub();
        ctx.mark_ready();
        ctx.mark_not_ready();

        ctx.mark_ready();
        assert!(
            !ctx.is_ready(),
            "mark_ready must not re-ready a pod that has begun draining",
        );
    }

    /// `mark_ready` is idempotent: the compare-exchange failing because the
    /// state is already READY must not be mistaken for the draining case.
    #[test]
    fn mark_ready_is_idempotent() {
        let ctx = AppContext::stub();
        ctx.mark_ready();
        ctx.mark_ready();
        assert!(
            ctx.is_ready(),
            "a second mark_ready must keep the pod ready"
        );
    }
}
