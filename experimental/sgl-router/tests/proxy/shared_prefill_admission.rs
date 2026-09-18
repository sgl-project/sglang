// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Admission at the HTTP boundary: capacity rejection is final within a
//! bucket, and a rejected session binding is kept while the request falls back.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use sgl_router::config::{
    ActiveLoadConfig, AffinityConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::state::engine_load::{LoadStat, NativeCacheRankLoad};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

fn config(policy: PolicyKind) -> Config {
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
            policy,
            decode_policy: Default::default(),
            bucket_config: None,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity: Some(AffinityConfig::default()),
            eligibility: None,
            sampling_overrides: Default::default(),
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

async fn fixture(policy: PolicyKind) -> (Arc<AppContext>, Vec<MockWorker>) {
    let backends = vec![
        MockWorker::start(vec![]).await,
        MockWorker::start(vec![]).await,
    ];
    let cfg = config(policy);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for (index, backend) in backends.iter().enumerate() {
        registry
            .add(WorkerSpec {
                id: WorkerId(format!("w{index}")),
                url: backend.url.clone(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            })
            .unwrap();
    }
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let ctx = AppContext::new(cfg, tokenizers, proxy, registry).unwrap();
    (Arc::new(ctx), backends)
}

/// Fresh native load with `used` of `capacity` KV tokens in use.
fn set_kv_usage(ctx: &AppContext, url: &str, used: u64, capacity: u64) {
    ctx.engine_load.set(
        url,
        0,
        LoadStat {
            num_running_reqs: 0,
            num_waiting_reqs: 0,
            num_tokens: used,
            max_total_num_tokens: capacity,
            native_cache: Some(NativeCacheRankLoad {
                num_waiting_uncached_tokens: 0,
                num_total_tokens: used,
                max_running_requests: 16,
                total_prefill_uncached_tokens: 1,
                total_prefill_busy_us: 1,
            }),
        },
        Instant::now(),
    );
}

async fn send_chat(ctx: &Arc<AppContext>, session: Option<&str>) -> StatusCode {
    let mut request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    if let Some(session) = session {
        request = request.header("x-session-id", session);
    }
    build_router(Arc::clone(ctx))
        .oneshot(
            request
                .body(Body::from(
                    serde_json::to_vec(&serde_json::json!({
                        "model": "tiny",
                        "messages": [{"role": "user", "content": "hi"}],
                    }))
                    .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap()
        .status()
}

fn dispatched(backends: &[MockWorker]) -> Vec<usize> {
    (0..backends.len())
        .filter(|&i| backends[i].captured.lock().unwrap().last_body.is_some())
        .collect()
}

#[tokio::test]
async fn capacity_exhaustion_returns_503_when_every_engine_is_rejected() {
    let (ctx, backends) = fixture(PolicyKind::PowerOfTwo).await;
    for backend in &backends {
        set_kv_usage(&ctx, &backend.url, 100, 100);
    }

    assert_eq!(send_chat(&ctx, None).await, StatusCode::SERVICE_UNAVAILABLE);
    assert!(dispatched(&backends).is_empty());
    assert!(ctx.metrics.render().contains(
        r#"sgl_router_policy_selection_failures_total{policy="power_of_two",reason="prefill_admission_exhausted"} 1"#
    ));
}

#[tokio::test]
async fn a_rejected_session_binding_is_kept_while_the_request_falls_back() {
    let (ctx, backends) = fixture(PolicyKind::SessionAware).await;

    assert_eq!(send_chat(&ctx, Some("s1")).await, StatusCode::OK);
    let [bound] = dispatched(&backends)[..] else {
        panic!("one engine dispatched");
    };
    set_kv_usage(&ctx, &backends[bound].url, 100, 100);

    assert_eq!(send_chat(&ctx, Some("s1")).await, StatusCode::OK);
    assert_eq!(
        dispatched(&backends).len(),
        2,
        "the other engine took the request"
    );
    let metrics = ctx.metrics.render();
    assert!(metrics.contains(r#"reason="assigned"} 1"#), "{metrics}");
    assert!(
        metrics.contains(r#"reason="session_admission_fallback"} 1"#),
        "{metrics}"
    );
}
