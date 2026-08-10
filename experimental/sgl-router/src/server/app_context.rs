// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::Config;

use crate::policies::active_load::ActiveLoadRegistry;
use crate::policies::itl::ItlTable;
use crate::policies::kv_events::KvEventIndex;
use crate::policies::PolicyRegistry;
use crate::proxy::Proxy;
use crate::server::admission::AdmissionQueue;
use crate::server::cache_sim_tee::CacheSimTee;
use crate::server::metrics::MetricsRegistry;
use crate::server::s3_export::S3ExportSink;
use crate::tokenizer::TokenizerRegistry;
use crate::workers::WorkerRegistry;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

#[derive(Debug)]
pub struct AppContext {
    pub config: Config,
    pub tokenizers: Arc<TokenizerRegistry>,
    pub proxy: Arc<Proxy>,
    pub registry: Arc<WorkerRegistry>,
    pub policies: Arc<PolicyRegistry>,
    /// Per-worker active-load bookkeeping. Shared between the proxy
    /// (which mints guards on the request hot path), the cache-aware
    /// policy (which reads per-worker load when scoring candidates), and
    /// the stale-request janitor (which sweeps expired entries).
    pub active_load: Arc<ActiveLoadRegistry>,
    /// Lightweight Prometheus-format metrics registry served via
    /// `/metrics`. Shared with the chat handler (worker_requests_total),
    /// cache-aware-zmq policy (overlap_blocks), active-load registry
    /// (active_load gauge + stale_requests_total), and PD resolver
    /// (decode_affinity_total).
    pub metrics: Arc<MetricsRegistry>,
    /// Router-side admission control. Gates the chat hot path: caps in-flight
    /// requests per worker and parks (or sheds with 503) excess. A pass-through
    /// when no per-worker cap is configured.
    pub admission: Arc<AdmissionQueue>,
    /// Per-worker router-observed inter-token latency (ITL), fed from the SSE
    /// pump's `on_inter_chunk` hook and read by the retry path to avoid
    /// re-dispatching onto a decode-congested worker.
    pub itl: Arc<ItlTable>,
    /// Best-effort tee of each request's ingress-computed `input_ids` to the
    /// theoretical cache-sim (`--cache-sim-url`). `None` when the flag is
    /// unset. The chat/completions handler offers to it after tokenizing;
    /// see [`crate::server::cache_sim_tee`].
    pub cache_sim_tee: Option<Arc<CacheSimTee>>,
    /// Best-effort S3 token-export sink. `None` when
    /// `observability.token_export_s3_uri` is unset or AWS credentials are
    /// missing. Zero overhead on the hot path when `None`.
    pub s3_export_sink: Option<Arc<S3ExportSink>>,
    /// KV-event index, when cache-aware-zmq routing is active.
    ///
    /// Injected rather than constructed here for the same reason as the
    /// `attach_metrics` wire-ins above: `main` builds the index before the
    /// worker manager is spawned, which happens before this context exists.
    /// `/readyz` reads it to decide whether peer bootstrap has settled, and
    /// `/internal/kv_snapshot` reads it to serve sibling replicas.
    kv_index: OnceLock<Arc<KvEventIndex>>,
    ready: AtomicBool,
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
        Self::with_active_load_and_itl(
            config,
            tokenizers,
            proxy,
            registry,
            policies,
            active_load,
            ItlTable::new(),
        )
    }

    /// As [`Self::with_active_load`], but with an explicit [`ItlTable`].
    /// Production ([`crate::main`]) builds the table BEFORE the worker
    /// manager is spawned and passes the same handle to both the manager
    /// (so it prunes the table on `Removed`) and here (so the chat handler
    /// and `/metrics` read it) — the manager is spawned before the
    /// `AppContext` exists, so the table can't be owned solely by the
    /// context. [`Self::with_active_load`] / [`Self::new`] mint a private
    /// table for tests that don't exercise discovery-driven pruning.
    pub fn with_active_load_and_itl(
        config: Config,
        tokenizers: Arc<TokenizerRegistry>,
        proxy: Arc<Proxy>,
        registry: Arc<WorkerRegistry>,
        policies: Arc<PolicyRegistry>,
        active_load: Arc<ActiveLoadRegistry>,
        itl: Arc<ItlTable>,
    ) -> Self {
        let metrics = MetricsRegistry::new();
        // Wire the per-worker active-load gauge so `sgl_router_active_load`
        // mirrors the live counter on every register / drop / sweep.
        // Without this, the metric is permanently 0 in production even
        // though the chat handler is faithfully calling `register`.
        active_load.attach_metrics(Arc::clone(&metrics));
        // Same rationale for the cache-aware-zmq policy's
        // `sgl_router_overlap_blocks`: the metrics registry is built here,
        // after the policy registry, so inject it now. No-op for policies
        // that don't emit metrics.
        policies.attach_metrics(Arc::clone(&metrics));
        // Same rationale for the proxy's `sgl_router_engine_aborts_total`
        // counter: the drop-side of `AbortOnDrop` reads the metrics handle
        // stashed on `Proxy`, and this is the one place where both objects
        // are alive together. Without this wire-in the abort counter is
        // permanently zero even though the WARN log still fires per abort.
        proxy.attach_metrics(Arc::clone(&metrics));
        let admission = Arc::new(AdmissionQueue::new(config.admission, Arc::clone(&metrics)));
        // Spawn the cache-sim tee when configured. Read before `config` is
        // moved into `Self`; an empty/whitespace URL is treated as unset.
        let max_captures = config.observability.cache_sim_max_concurrent_captures;
        let cache_sim_tee = config
            .observability
            .cache_sim_url
            .as_ref()
            .map(|u| u.trim())
            .filter(|u| !u.is_empty())
            .map(|url| CacheSimTee::spawn(url.to_owned(), Arc::clone(&metrics), max_captures));
        // 读取 pod 名用于对象 key 去重；未注入时退化为 "unknown-pod"。
        let pod = std::env::var("POD_NAME").unwrap_or_else(|_| "unknown-pod".to_string());
        let s3_export_sink = config
            .observability
            .token_export_s3_uri
            .as_ref()
            .map(|u| u.trim())
            .filter(|u| !u.is_empty())
            .and_then(|uri| S3ExportSink::spawn(uri, pod, Arc::clone(&metrics)));
        Self {
            config,
            tokenizers,
            proxy,
            registry,
            policies,
            active_load,
            metrics,
            admission,
            itl,
            cache_sim_tee,
            s3_export_sink,
            kv_index: OnceLock::new(),
            ready: AtomicBool::new(false),
        }
    }

    /// Attach the KV-event index. Called once by `main`; later calls are
    /// ignored so a mis-wire cannot swap the index out from under `/readyz`.
    pub fn attach_kv_index(&self, index: Arc<KvEventIndex>) {
        if self.kv_index.set(index).is_err() {
            tracing::warn!("kv index already attached to AppContext; ignoring");
        }
    }

    pub fn kv_index(&self) -> Option<&Arc<KvEventIndex>> {
        self.kv_index.get()
    }

    /// Whether initial cache-aware bootstrap has settled.
    ///
    /// `true` when there is no KV index at all (cache-aware-zmq disabled), so
    /// routers that never bootstrap are unaffected by the readiness gate.
    pub fn kv_bootstrap_settled(&self) -> bool {
        self.kv_index
            .get()
            .is_none_or(|idx| idx.bootstrap().settled())
    }

    pub fn mark_ready(&self) {
        // Relaxed: this flag does not synchronize other state; readers only
        // care about eventual visibility, not happens-before with surrounding ops.
        self.ready.store(true, Ordering::Relaxed);
    }

    /// Inverse of [`mark_ready`](Self::mark_ready): flip `/readyz` back to 503.
    /// Called on SIGTERM so the EndpointSlice controller deregisters this pod
    /// before the server stops accepting, closing the rolling-update race where
    /// new requests land on a socket that is about to close.
    pub fn mark_not_ready(&self) {
        self.ready.store(false, Ordering::Relaxed);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn stub() -> Self {
        // Share one registry between `metrics` and `admission`, mirroring
        // production, so admission metrics are visible via `ctx.metrics`.
        let metrics = MetricsRegistry::new();
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
                    tokenizer_shards: 1,
                    tokenizer_backend: Default::default(),
                    tokenizer_l1_cache_mb: 0,
                    policy: crate::config::PolicyKind::RoundRobin,
                    circuit_breaker: None,
                    cache_aware: None,
                    sticky: None,
                    max_output_tokens: None,
                    forward_input_ids: true,
                },
                discovery: crate::config::DiscoveryBackend::StaticUrls(
                    crate::config::StaticUrlsDiscoveryConfig {
                        urls: vec!["http://placeholder:0".into()],
                    },
                ),
                proxy: crate::config::ProxyConfig::default(),
                active_load: crate::config::ActiveLoadConfig::default(),
                admission: crate::config::AdmissionConfig::default(),
                retry: crate::config::RetryConfig::default(),
            },
            tokenizers: Arc::new(TokenizerRegistry::default()),
            proxy: Arc::new(Proxy::new(std::time::Duration::from_secs(60)).expect("stub proxy")),
            registry: Arc::new(WorkerRegistry::default()),
            policies: Arc::new(PolicyRegistry::default()),
            active_load: ActiveLoadRegistry::with_defaults(),
            admission: Arc::new(AdmissionQueue::new(
                crate::config::AdmissionConfig::Disabled,
                Arc::clone(&metrics),
            )),
            metrics,
            itl: ItlTable::new(),
            cache_sim_tee: None,
            s3_export_sink: None,
            // Unset: a stub has no KV index, so `kv_bootstrap_settled()`
            // reports true and the readiness gate is inert unless a test
            // attaches one.
            kv_index: OnceLock::new(),
            ready: AtomicBool::new(false),
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
}
