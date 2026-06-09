// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for `policy = "sticky"`: a routing key read from the
//! operator-configured header pins a session to one worker, and the
//! `sgl_router_sticky_total` outcomes are recorded. Runs against two
//! `MockWorker` backends (CPU-only, no GPU).

use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig, StickyConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults as build_policy_registry;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

/// Build an `AppContext` running the sticky policy over the given worker
/// URLs, reading the routing key from `header_name`. Eviction is pushed far
/// out so the background sweeper never fires mid-test.
fn build_sticky_ctx(header_name: &str, worker_urls: &[String]) -> Arc<AppContext> {
    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::Sticky,
            circuit_breaker: None,
            cache_aware: None,
            sticky: Some(StickyConfig {
                header_name: header_name.to_string(),
                fallback_policy: PolicyKind::RoundRobin,
                idle_secs: 3600,
                eviction_interval_secs: 3600,
            }),
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    for (i, url) in worker_urls.iter().enumerate() {
        let _ = registry.add(WorkerSpec {
            id: WorkerId(format!("w{i}")),
            url: url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            bootstrap_port: None,
        });
    }
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(TEST_TIMEOUT).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

fn chat_request(header: Option<(&str, &str)>) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    if let Some((name, value)) = header {
        builder = builder.header(name, value);
    }
    builder
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false
            }))
            .unwrap(),
        ))
        .unwrap()
}

/// Parse `sgl_router_requests_total{...,outcome="success"} N` lines into a
/// map of worker_url -> success count.
fn success_counts(metrics: &str) -> std::collections::HashMap<String, u64> {
    let mut counts = std::collections::HashMap::new();
    for line in metrics.lines() {
        let Some(rest) = line.strip_prefix("sgl_router_requests_total{") else {
            continue;
        };
        if !rest.contains(r#"outcome="success""#) {
            continue;
        }
        let Some(url_start) = rest.find(r#"worker_url=""#) else {
            continue;
        };
        let after = &rest[url_start + r#"worker_url=""#.len()..];
        let Some(url_end) = after.find('"') else {
            continue;
        };
        let url = after[..url_end].to_string();
        let value: u64 = line
            .rsplit(' ')
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        *counts.entry(url).or_insert(0) += value;
    }
    counts
}

/// Read `sgl_router_sticky_total{outcome="<outcome>"} N`.
fn sticky_count(metrics: &str, outcome: &str) -> u64 {
    let needle = format!(r#"sgl_router_sticky_total{{outcome="{outcome}"}} "#);
    metrics
        .lines()
        .find_map(|l| l.strip_prefix(&needle))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

#[tokio::test]
async fn same_routing_key_pins_to_one_worker() {
    let w0 = MockWorker::start(vec![]).await;
    let w1 = MockWorker::start(vec![]).await;
    let ctx = build_sticky_ctx("x-sgl-routing-key", &[w0.url.clone(), w1.url.clone()]);
    let app = build_router(ctx.clone());

    const N: usize = 5;
    for _ in 0..N {
        let res = app
            .clone()
            .oneshot(chat_request(Some(("x-sgl-routing-key", "alice"))))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }

    let metrics = ctx.metrics.render();
    let counts = success_counts(&metrics);
    let total: u64 = counts.values().sum();
    assert_eq!(total, N as u64, "all requests should succeed: {counts:?}");
    let pinned: Vec<_> = counts.iter().filter(|(_, &c)| c > 0).collect();
    assert_eq!(
        pinned.len(),
        1,
        "all same-key requests must hit exactly one worker: {counts:?}"
    );
    assert_eq!(*pinned[0].1, N as u64);

    // One assignment, the rest hits.
    assert_eq!(sticky_count(&metrics, "assigned"), 1, "metrics:\n{metrics}");
    assert_eq!(sticky_count(&metrics, "hit"), (N - 1) as u64);
}

#[tokio::test]
async fn remaps_to_survivor_when_pinned_worker_is_removed() {
    let w0 = MockWorker::start(vec![]).await;
    let w1 = MockWorker::start(vec![]).await;
    let worker_urls = vec![w0.url.clone(), w1.url.clone()];
    let ctx = build_sticky_ctx("x-sgl-routing-key", &worker_urls);
    let app = build_router(ctx.clone());

    // Pin the key, then discover which worker it landed on.
    let res = app
        .clone()
        .oneshot(chat_request(Some(("x-sgl-routing-key", "alice"))))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let pinned_url = success_counts(&ctx.metrics.render())
        .into_iter()
        .find(|(_, c)| *c > 0)
        .map(|(url, _)| url)
        .expect("a worker should have served the first request");
    let pinned_idx = worker_urls.iter().position(|u| *u == pinned_url).unwrap();
    let survivor_url = worker_urls[1 - pinned_idx].clone();

    // Remove the pinned worker from the registry; the next same-key request
    // must remap to the survivor (not fail).
    ctx.registry.remove(&WorkerId(format!("w{pinned_idx}")));
    let res = app
        .clone()
        .oneshot(chat_request(Some(("x-sgl-routing-key", "alice"))))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let metrics = ctx.metrics.render();
    assert_eq!(sticky_count(&metrics, "remap"), 1, "{metrics}");
    // The survivor served the second request.
    let counts = success_counts(&metrics);
    assert_eq!(counts.get(&survivor_url).copied().unwrap_or(0), 1);
}

#[tokio::test]
async fn keyless_request_is_served_via_fallback() {
    let w0 = MockWorker::start(vec![]).await;
    let w1 = MockWorker::start(vec![]).await;
    let ctx = build_sticky_ctx("x-sgl-routing-key", &[w0.url.clone(), w1.url.clone()]);
    let app = build_router(ctx.clone());

    // No routing-key header at all.
    let res = app.clone().oneshot(chat_request(None)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let metrics = ctx.metrics.render();
    assert_eq!(sticky_count(&metrics, "no_routing_key"), 1, "{metrics}");
    assert_eq!(sticky_count(&metrics, "assigned"), 0);
}

#[tokio::test]
async fn only_the_configured_header_name_is_honored() {
    // Router configured to read the key from `x-session-id`.
    let w0 = MockWorker::start(vec![]).await;
    let w1 = MockWorker::start(vec![]).await;
    let ctx = build_sticky_ctx("x-session-id", &[w0.url.clone(), w1.url.clone()]);
    let app = build_router(ctx.clone());

    // A request using the configured header pins (assigned), and a repeat hits.
    for _ in 0..2 {
        let res = app
            .clone()
            .oneshot(chat_request(Some(("x-session-id", "s-1"))))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }
    // A request carrying the *default* header (not the configured one) must
    // be treated as keyless — proving the header name is dynamic, not baked in.
    let res = app
        .clone()
        .oneshot(chat_request(Some(("x-sgl-routing-key", "s-1"))))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let metrics = ctx.metrics.render();
    assert_eq!(sticky_count(&metrics, "assigned"), 1, "{metrics}");
    assert_eq!(sticky_count(&metrics, "hit"), 1);
    assert_eq!(sticky_count(&metrics, "no_routing_key"), 1);
}
