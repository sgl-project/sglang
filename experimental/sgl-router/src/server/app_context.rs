// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::Config;

use crate::policies::active_load::ActiveLoadRegistry;
use crate::policies::PolicyRegistry;
use crate::proxy::Proxy;
use crate::server::metrics::MetricsRegistry;
use crate::tokenizer::TokenizerRegistry;
use crate::workers::WorkerRegistry;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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
    /// `/metrics`. Shared with the chat handler (requests_total),
    /// cache-aware-zmq policy (overlap_blocks), active-load registry
    /// (active_load gauge + stale_requests_total), and PD resolver
    /// (decode_affinity_total).
    pub metrics: Arc<MetricsRegistry>,
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
        Self {
            config,
            tokenizers,
            proxy,
            registry,
            policies,
            active_load,
            metrics,
            ready: AtomicBool::new(false),
        }
    }

    pub fn mark_ready(&self) {
        // Relaxed: this flag does not synchronize other state; readers only
        // care about eventual visibility, not happens-before with surrounding ops.
        self.ready.store(true, Ordering::Relaxed);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub fn stub() -> Self {
        Self {
            config: Config {
                server: crate::config::ServerConfig {
                    host: "x".into(),
                    port: 0,
                },
                observability: Default::default(),
                model: crate::config::ModelConfig {
                    id: "stub-model".into(),
                    tokenizer_path: "stub".into(),
                    policy: crate::config::PolicyKind::RoundRobin,
                    circuit_breaker: None,
                    cache_aware: None,
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
            active_load: ActiveLoadRegistry::with_defaults(),
            metrics: MetricsRegistry::new(),
            ready: AtomicBool::new(false),
        }
    }
}
