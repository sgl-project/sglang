// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::buckets_reorg::BucketResolver;
use crate::config::{Config, PolicyKind};
use crate::policies_reorg::factory::{build_pools, Dependencies};
use crate::proxy::Proxy;
use crate::server::inflight::InflightHttp;
use crate::server::metrics::MetricsRegistry;
use crate::state::active_load::{ActiveLoadRegistry, JanitorHandle};
use crate::state::engine_load::EngineLoadTable;
use crate::state::kv_events::{BlockSizeOracle, KvEventIndex, KvIndexMetrics};
use crate::state::{AffinityStore, PrefixSource};
use crate::tokenizer::TokenizerRegistry;
use crate::workers::WorkerRegistry;
use sgl_kv_indexer::PrefixIndex;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// `/readyz` readiness as a one-way door: `NOT_READY -> READY -> DRAINING`,
/// and never backwards.
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
    /// is actually waiting on during the drain; `active_load` sees only the
    /// proxied subset.
    pub inflight_http: Arc<InflightHttp>,
    readiness: AtomicU8,
    _affinity_sweeper: Option<JanitorHandle>,
}

impl AppContext {
    /// A context without engine state: no KV index and no remote indexer.
    /// The configuration was validated by the CLI, so policy construction
    /// cannot fail here.
    pub fn new(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        registry: Arc<WorkerRegistry>,
    ) -> Self {
        Self::with_engine_state(
            config,
            tokenizers,
            proxy,
            registry,
            ActiveLoadRegistry::with_defaults(),
            None,
            None,
        )
        .expect("validated configuration builds its policies")
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
    ) -> anyhow::Result<Self> {
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
        let prefix = match (prefix_index, &kv_index) {
            (Some(index), _) => Some(PrefixSource::Remote {
                index,
                block_size: Arc::clone(&block_size_oracle),
            }),
            (None, Some(index)) => Some(PrefixSource::Local(Arc::clone(index))),
            (None, None) => None,
        };
        let deps = Dependencies {
            affinity,
            prefix: prefix.map(Arc::new),
            metrics: Arc::clone(&metrics),
        };
        let pools = build_pools(model, &deps)?;
        Ok(Self {
            buckets: Arc::new(BucketResolver::new(Arc::clone(&registry), pools)),
            config,
            tokenizers,
            proxy,
            registry,
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

    pub fn mark_ready(&self) {
        let _ = self.readiness.compare_exchange(
            READINESS_NOT_READY,
            READINESS_READY,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    pub fn mark_not_ready(&self) {
        self.readiness.store(READINESS_DRAINING, Ordering::Relaxed);
    }

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
                fused: None,
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
