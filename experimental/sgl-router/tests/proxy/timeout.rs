// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Tests that the router does not wedge indefinitely when an upstream
//! worker accepts the TCP connection but never sends response headers.
//!
//! Without a configured `.timeout(...)` on the reqwest client, a stalled
//! backend hangs the axum handler future forever and the test harness
//! would just timeout. We assert here that the router returns a fast,
//! clean 504 (`upstream_timeout`) instead — a timeout is a gateway timeout,
//! the same status class as the stale-deadline cancel.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use sgl_router::config::{
    Config, DiscoveryBackend, InflightLoadConfig, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults as build_policy_registry;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

fn config(_worker_url: &str) -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: Some("tests/fixtures/tiny_tokenizer.json".into()),
            disable_input_ids_forwarding: false,
            tokenizer: Default::default(),
            policy: PolicyKind::RoundRobin,
            decode_policy: Default::default(),
            dp_aware: false,
            bucket_config: None,
            reorg_buckets: None,
            reorg_admission: Default::default(),
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity: None,
            fused: None,
            eligibility: None,
            sampling_overrides: Default::default(),
            profile: Default::default(),
            default_chat_template_kwargs: Default::default(),
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        router_inflight_load: InflightLoadConfig::default(),
    }
}

#[tokio::test]
async fn non_streaming_request_times_out_when_worker_hangs() {
    // Worker accepts and then sleeps for 5s; router timeout is 200ms.
    let worker =
        crate::common::mock_worker::MockWorker::start_hanging(Duration::from_secs(5)).await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId("w1".into()),
        url: worker.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        ..Default::default()
    });
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_millis(200)).unwrap());
    let ctx = Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies));
    let app = build_router(ctx.clone());

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false
            }))
            .unwrap(),
        ))
        .unwrap();

    let started = std::time::Instant::now();
    // Outer guard so a regression doesn't wedge CI forever.
    let res = tokio::time::timeout(Duration::from_secs(2), app.oneshot(req))
        .await
        .expect("router must return within 2s when proxy timeout is 200ms")
        .unwrap();
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(1),
        "router must short-circuit on upstream timeout; elapsed {elapsed:?}"
    );
    assert_eq!(res.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(
        res.headers().get("x-router-error-code").unwrap(),
        "upstream_timeout"
    );
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&bytes);
    // A hung worker is the most common hard worker failure there is, so it must
    // land in `outcome="error"` — the series a per-worker error-ratio alert
    // watches. Deriving the outcome from the 504 status instead would silently
    // reclassify it as `cancelled` and blind that alert.
    assert!(
        ctx.metrics
            .render()
            .lines()
            .any(|l| l.starts_with("sgl_router_worker_requests_total{")
                && l.contains(r#"outcome="error""#)),
        "an upstream timeout must be counted outcome=error, not cancelled:\n{}",
        ctx.metrics.render(),
    );
    assert!(
        body_str.contains("\"code\":\"upstream_timeout\""),
        "body: {body_str}"
    );
    // No leak of worker URL or reqwest source chain to the client.
    assert!(
        !body_str.contains(&worker.url),
        "worker URL must not leak in client-visible body: {body_str}"
    );
}

/// Plain workers at `urls` behind a router whose upstream timeout is 200 ms.
fn hanging_ctx(urls: &[&str], max_attempts: u32) -> Arc<AppContext> {
    let mut cfg = config("");
    cfg.proxy.max_attempts = std::num::NonZeroU32::new(max_attempts).unwrap();
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for (i, url) in urls.iter().enumerate() {
        registry
            .add(WorkerSpec {
                id: WorkerId(format!("w{i}")),
                url: (*url).into(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("tiny".into())],
                ..Default::default()
            })
            .unwrap();
    }
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_millis(200)).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn streaming_chat() -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            })
            .to_string(),
        ))
        .unwrap()
}

/// A streaming worker that never sends headers times out like a non-streaming one,
/// counts against its breaker, and with retries on, the request moves to another worker.
#[tokio::test]
async fn streaming_request_times_out_waiting_for_headers() {
    let hanging =
        crate::common::mock_worker::MockWorker::start_hanging(Duration::from_secs(5)).await;
    let ctx = hanging_ctx(&[&hanging.url], 1);
    let app = build_router(ctx.clone());
    // The default breaker opens after three consecutive failures.
    for _ in 0..3 {
        let res = tokio::time::timeout(
            Duration::from_secs(2),
            app.clone().oneshot(streaming_chat()),
        )
        .await
        .expect("router must not wait past its upstream timeout for stream headers")
        .unwrap();
        assert_eq!(res.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(
            res.headers().get("x-router-error-code").unwrap(),
            "upstream_timeout"
        );
    }
    let worker = ctx.registry.get(&WorkerId("w0".into())).unwrap();
    assert!(
        !worker.breaker.would_allow(),
        "timeouts must open the breaker"
    );

    let live = crate::common::mock_worker::MockWorker::start(vec!["data: [DONE]\n\n"]).await;
    let ctx = hanging_ctx(&[&hanging.url, &live.url], 2);
    let app = build_router(ctx);
    for _ in 0..2 {
        let res = tokio::time::timeout(
            Duration::from_secs(2),
            app.clone().oneshot(streaming_chat()),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }
}
