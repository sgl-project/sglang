// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::{Config, PolicyKind};
use crate::policies::pools::PdPoolResolver;
use crate::policies::state::engine_load::{ActiveLoadRegistry, EngineLoadTable, JanitorHandle};
use crate::policies::state::kv_events::{BlockSizeOracle, KvEventIndex, KvIndexMetrics};
use crate::policies::state::AffinityStore;
use crate::policy_reorg::buckets::BucketResolver;
use crate::policy_reorg::{BuildError, PolicyDependencies};
use crate::proxy::Proxy;
use crate::server::inflight::InflightHttp;
use crate::server::metrics::MetricsRegistry;
use crate::tokenizer::TokenizerRegistry;
use crate::workers::WorkerRegistry;
use sgl_kv_indexer::PrefixIndex;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;

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
    /// Orders buckets and runs their attached policies for every request.
    pub buckets: Arc<BucketResolver>,
    /// Per-worker active-load bookkeeping shared by the proxy, policies,
    /// timeout janitor, and metrics.
    pub active_load: Arc<ActiveLoadRegistry>,
    /// Prometheus-format metrics served via `/metrics`.
    pub metrics: Arc<MetricsRegistry>,
    /// Shared engine LoadStat table; each selection pass captures one snapshot.
    pub engine_load: Arc<EngineLoadTable>,
    pub block_size_oracle: Arc<BlockSizeOracle>,
    /// `/metrics` handles for the KV storage-tier series; `None` without a
    /// local tree, where every series would be a structural zero.
    pub kv_metrics: Option<KvIndexMetrics>,
    /// Open HTTP exchanges, on every route. What axum's graceful shutdown
    /// is actually waiting on during the drain — `active_load` sees only the
    /// proxied subset.
    pub inflight_http: Arc<InflightHttp>,
    readiness: AtomicU8,
    _affinity_sweeper: Option<JanitorHandle>,
}

impl AppContext {
    /// A context without engine state: no KV index and no remote indexer.
    pub fn new(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        registry: Arc<WorkerRegistry>,
    ) -> Result<Self, BuildError> {
        Self::with_engine_state(
            config,
            tokenizers,
            proxy,
            registry,
            ActiveLoadRegistry::with_defaults(),
            None,
            None,
        )
    }

    /// Wires the bucket engine over the shared engine state: the KV-event
    /// index (load table, block size, local prefix lookup) and an optional
    /// remote prefix indexer.
    pub fn with_engine_state(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        registry: Arc<WorkerRegistry>,
        active_load: Arc<ActiveLoadRegistry>,
        kv_index: Option<Arc<KvEventIndex>>,
        prefix_index: Option<Arc<dyn PrefixIndex>>,
    ) -> Result<Self, BuildError> {
        let metrics = MetricsRegistry::new();
        active_load.attach_metrics(Arc::clone(&metrics));
        let (engine_load, block_size_oracle, kv_metrics) = match &kv_index {
            Some(index) => (
                index.engine_load(),
                index.block_size_oracle(),
                index.metrics_source(),
            ),
            None => (EngineLoadTable::new(), BlockSizeOracle::new(), None),
        };
        let model = &config.model;
        let (idle, eviction) = match model.policy {
            PolicyKind::Sticky => {
                let sticky = model.sticky.clone().unwrap_or_default();
                (sticky.idle_secs, sticky.eviction_interval_secs)
            }
            _ => {
                let affinity = model.affinity.clone().unwrap_or_default();
                (
                    affinity.session_idle_secs,
                    affinity.session_eviction_interval_secs,
                )
            }
        };
        let affinity = AffinityStore::new(Duration::from_secs(idle));
        let affinity_sweeper = affinity.spawn_sweeper(Duration::from_secs(eviction));
        let deps = PolicyDependencies {
            metrics: Arc::clone(&metrics),
            affinity,
            local_cache: kv_index,
            remote_cache: prefix_index,
            block_size: Arc::clone(&block_size_oracle),
        };
        let buckets = BucketResolver::from_config(
            model,
            PdPoolResolver::new(Arc::clone(&registry)),
            Arc::clone(&engine_load),
            &deps,
        )?;
        Ok(Self {
            config,
            tokenizers,
            proxy,
            registry,
            buckets: Arc::new(buckets),
            active_load,
            metrics,
            engine_load,
            block_size_oracle,
            kv_metrics,
            inflight_http: InflightHttp::new(),
            readiness: AtomicU8::new(READINESS_NOT_READY),
            _affinity_sweeper: affinity_sweeper,
        })
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
        let config = Config {
            server: crate::config::ServerConfig {
                host: "x".into(),
                port: 0,
                ..Default::default()
            },
            observability: Default::default(),
            model: crate::config::ModelConfig {
                id: "stub-model".into(),
                tokenizer_path: "stub".into(),
                policy: PolicyKind::RoundRobin,
                decode_policy: Default::default(),
                bucket_config: None,
                circuit_breaker: None,
                cache_aware: None,
                sticky: None,
                affinity: None,
                eligibility: None,
                sampling_overrides: Default::default(),
            },
            discovery: crate::config::DiscoveryBackend::StaticUrls(
                crate::config::StaticUrlsDiscoveryConfig {
                    urls: vec!["http://placeholder:0".into()],
                },
            ),
            proxy: crate::config::ProxyConfig::default(),
            active_load: crate::config::ActiveLoadConfig::default(),
        };
        Self::new(
            config,
            Arc::new(TokenizerRegistry::default()),
            Arc::new(Proxy::new(Duration::from_secs(60)).expect("stub proxy")),
            Arc::new(WorkerRegistry::default()),
        )
        .expect("stub context")
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
