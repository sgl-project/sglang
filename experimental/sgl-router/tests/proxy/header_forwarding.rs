// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::body::Body;
use axum::http::Request;
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
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

#[tokio::test]
async fn forwards_whitelisted_headers_strips_others() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = Config {
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
        admission: sgl_router::config::AdmissionConfig::default(),
        retry: sgl_router::config::RetryConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId("w1".into()),
        url: worker.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    });
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(Duration::from_secs(5)).unwrap());
    let app = build_router(Arc::new(AppContext::new(
        cfg, tokenizers, proxy, registry, policies,
    )));

    let body = serde_json::to_vec(&serde_json::json!({
        "model":"tiny","messages":[{"role":"user","content":"hi"}]
    }))
    .unwrap();

    // Use a spoofed content-length that differs from the real body length so we
    // can distinguish "inbound value forwarded" from "reqwest auto-computed it".
    let spoofed_content_length = "99999";
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer test")
        .header("x-request-id", "abc-123")
        .header("x-sgl-route-key", "k1")
        .header("cookie", "should-not-forward=true")
        .header("host", "example.com")
        .header("content-length", spoofed_content_length)
        .header("transfer-encoding", "chunked")
        .body(Body::from(body))
        .unwrap();
    app.oneshot(req).await.unwrap();

    let seen = worker.captured.lock().unwrap();
    // Whitelisted headers are forwarded with their inbound VALUES intact —
    // a regression that mangles, uppercases, or drops the value (e.g.,
    // forwarding the name but not the value) must fail this assertion.
    assert_eq!(
        seen.headers.get("authorization").map(String::as_str),
        Some("Bearer test"),
        "authorization must be forwarded with its inbound value verbatim",
    );
    assert_eq!(
        seen.headers.get("x-request-id").map(String::as_str),
        Some("abc-123"),
        "x-request-id must be forwarded with its inbound value verbatim",
    );
    assert_eq!(
        seen.headers.get("x-sgl-route-key").map(String::as_str),
        Some("k1"),
        "x-sgl-route-key must be forwarded with its inbound value verbatim",
    );
    // Cookie must be stripped.
    assert!(!seen.seen.contains("cookie"));
    // transfer-encoding is hop-by-hop and must not be forwarded (reqwest does not
    // re-add it for a regular body, so absence check is reliable here).
    assert!(
        !seen.seen.contains("transfer-encoding"),
        "transfer-encoding is hop-by-hop and must be stripped"
    );
    // content-length: the inbound spoofed value must not reach the upstream.
    // reqwest may auto-compute its own content-length for the outbound body,
    // so we assert value-inequality rather than absence.
    assert_ne!(
        seen.headers.get("content-length").map(|s| s.as_str()),
        Some(spoofed_content_length),
        "router must not forward the inbound content-length value to upstream"
    );
    // Host: the inbound value must not reach the upstream.
    let captured_host: Option<&String> = seen.headers.get("host");
    assert_ne!(
        captured_host,
        Some(&"example.com".to_string()),
        "router must not forward the inbound Host header to upstream"
    );
}
