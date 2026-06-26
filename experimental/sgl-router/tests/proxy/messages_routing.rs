// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the `/v1/messages` Anthropic passthrough route.
//!
//! The mock worker echoes the same `chat` handler onto `/v1/messages`, so a
//! request hitting the router's `/v1/messages` route is forwarded to the
//! worker's `/v1/messages` and the body comes back. This asserts the
//! passthrough contract: no translation, no `input_ids` injection.

use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
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
use http_body_util::BodyExt;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

fn build_ctx_with_worker(url: &str) -> Arc<AppContext> {
    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId("w1".into()),
        url: url.to_string(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    });
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(TEST_TIMEOUT).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

/// Same as `build_ctx_with_worker` but registers the worker in PD prefill mode,
/// so the `/v1/messages` route hits its PD-reject guard instead of forwarding.
fn build_ctx_with_prefill_worker(url: &str) -> Arc<AppContext> {
    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId("w1".into()),
        url: url.to_string(),
        mode: WorkerMode::Prefill,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    });
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(TEST_TIMEOUT).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

#[tokio::test]
async fn messages_non_streaming_passthrough_200() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

    let body = serde_json::to_vec(&serde_json::json!({
        "model": "tiny",
        "max_tokens": 16,
        "stream": false,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .unwrap();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn messages_streaming_passthrough_sse() {
    let chunks: Vec<&'static str> = vec![
        "data: {\"type\":\"content_block_delta\"}\n\n",
        "data: {\"type\":\"message_stop\"}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker = crate::common::mock_worker::MockWorker::start(chunks.clone()).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

    let body = serde_json::to_vec(&serde_json::json!({
        "model": "tiny",
        "stream": true,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .unwrap();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(
        res.headers().get("content-type").unwrap().to_str().unwrap(),
        "text/event-stream"
    );
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let data = crate::common::streaming::parse_sse_data(&bytes);
    assert_eq!(data.len(), 3);
    assert_eq!(data[2], "[DONE]");
}

#[tokio::test]
async fn messages_missing_model_returns_400() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

    let body = serde_json::to_vec(&serde_json::json!({
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .unwrap();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    // Router-originated errors on this endpoint are Anthropic-shaped.
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["type"], "error", "envelope type must be 'error'");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("model"));
}

#[tokio::test]
async fn messages_body_forwarded_unchanged_no_input_ids() {
    // Core passthrough invariant: the Anthropic body reaches the worker
    // byte-identical, with no `input_ids` injected (the worker tokenizes).
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

    let sent = serde_json::to_vec(&serde_json::json!({
        "model": "tiny",
        "max_tokens": 16,
        "stream": false,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .unwrap();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(sent.clone()))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    let forwarded = worker.captured.lock().unwrap().last_body.clone();
    let fwd: serde_json::Value = serde_json::from_slice(&forwarded.unwrap()).unwrap();
    assert!(fwd.get("input_ids").is_none(), "must not inject input_ids");
    assert_eq!(
        fwd,
        serde_json::from_slice::<serde_json::Value>(&sent).unwrap()
    );
}

#[tokio::test]
async fn messages_rejects_pd_disaggregated_mode() {
    // The passthrough does not do chat.rs's decode-peer + bootstrap injection,
    // so it must reject PD mode explicitly rather than silently forwarding to a
    // single prefill worker and hanging.
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_prefill_worker(&worker.url);
    let app = build_router(ctx);

    let body = serde_json::to_vec(&serde_json::json!({
        "model": "tiny",
        "stream": false,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .unwrap();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn messages_upstream_error_is_anthropic_shaped_and_does_not_leak_url() {
    // Register an unreachable worker so the proxy returns UpstreamUnreachable.
    // The Anthropic error envelope must be sanitized — no worker URL in the body.
    let ctx = build_ctx_with_worker("http://127.0.0.1:1"); // port 1: connection refused fast
    let app = build_router(ctx);
    let body = serde_json::to_vec(&serde_json::json!({
        "model": "tiny",
        "stream": false,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .unwrap();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    // Upstream-unreachable maps to 502 (BAD_GATEWAY) via ApiError.
    assert!(res.status().is_server_error(), "got {}", res.status());
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()["type"],
        "error",
        "envelope must be Anthropic-shaped: {text}"
    );
    // The worker URL must NOT leak into the client-facing message.
    assert!(
        !text.contains("127.0.0.1"),
        "worker URL leaked into client error: {text}"
    );
}
