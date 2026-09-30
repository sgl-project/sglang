// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! `--dp-aware`: the router pins each plain-mode dispatch to one attention-DP
//! rank via `X-Data-Parallel-Rank`, and a client can never set that header
//! itself.

use axum::body::Body;
use axum::http::Request;
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, DpAwareConfig, ModelConfig, ObservabilityConfig,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults as build_policy_registry;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

fn cfg(dp_aware: bool) -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            tokenizer_shards: 1,
            tokenizer_backend: Default::default(),
            tokenizer_l1_cache_mb: 0,
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            decode_policy: None,
            dp_aware: DpAwareConfig {
                enabled: dp_aware,
                rank_queue_limit: None,
            },
            sticky: None,
            max_output_tokens: None,
            sampling_overrides: Default::default(),
            forward_input_ids: true,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        load_monitor: sgl_router::config::LoadMonitorConfig::default(),
        active_load: ActiveLoadConfig::default(),
        admission: sgl_router::config::AdmissionConfig::default(),
        retry: sgl_router::config::RetryConfig::default(),
    }
}

/// Router over one mock worker whose `/server_info` would report `dp_size`
/// (stamped directly here, as the worker manager does after introspection).
fn router(worker: &MockWorker, dp_aware: bool, dp_size: u32) -> axum::Router {
    let cfg = cfg(dp_aware);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let id = WorkerId("w1".into());
    let _ = registry.add(WorkerSpec {
        id: id.clone(),
        url: worker.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
        transfer_group: None,
    });
    registry.get(&id).unwrap().set_dp_size(dp_size);
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    build_router(Arc::new(AppContext::new(
        cfg, tokenizers, proxy, registry, policies,
    )))
}

fn chat(spoofed_rank: Option<&str>) -> Request<Body> {
    let body = serde_json::to_vec(&serde_json::json!({
        "model": "tiny", "messages": [{"role": "user", "content": "hi"}]
    }))
    .unwrap();
    let mut req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    if let Some(r) = spoofed_rank {
        req = req.header("x-data-parallel-rank", r);
    }
    req.body(Body::from(body)).unwrap()
}

fn seen_rank(worker: &MockWorker) -> Option<String> {
    worker
        .captured
        .lock()
        .unwrap()
        .headers
        .get("x-data-parallel-rank")
        .cloned()
}

/// With no engine load reports, the router's own dispatch counts decide, so
/// four back-to-back requests land on four distinct ranks — and a client's
/// spoofed rank never reaches the engine.
#[tokio::test]
async fn dp_aware_pins_each_dispatch_to_a_rank_and_spreads_them() {
    let worker = MockWorker::start(vec![]).await;
    let app = router(&worker, true, 4);
    let mut ranks = HashSet::new();
    for _ in 0..4 {
        app.clone().oneshot(chat(Some("7"))).await.unwrap();
        let rank = seen_rank(&worker).expect("--dp-aware must send a rank");
        assert_ne!(
            rank, "7",
            "the client's rank must be replaced, not forwarded"
        );
        let r: u32 = rank.parse().unwrap();
        assert!(r < 4, "rank {r} out of range for dp_size 4");
        ranks.insert(r);
    }
    assert_eq!(
        ranks.len(),
        4,
        "min-load over dispatch counts spreads: {ranks:?}"
    );
}

/// Without the flag the engine's DP controller picks, exactly as before — and
/// the client still cannot pick for it.
#[tokio::test]
async fn without_dp_aware_no_rank_is_sent_and_client_rank_is_stripped() {
    let worker = MockWorker::start(vec![]).await;
    let app = router(&worker, false, 4);
    app.oneshot(chat(Some("3"))).await.unwrap();
    assert_eq!(seen_rank(&worker), None);
}

/// A single-rank worker (or one not introspected yet) gets no rank.
#[tokio::test]
async fn dp_aware_leaves_single_rank_workers_alone() {
    let worker = MockWorker::start(vec![]).await;
    let app = router(&worker, true, 1);
    app.oneshot(chat(None)).await.unwrap();
    assert_eq!(seen_rank(&worker), None);
}
