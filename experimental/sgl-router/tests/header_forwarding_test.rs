// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod common;

use axum::body::Body;
use axum::http::Request;
use sgl_router::config::{Config, ModelConfig, ObservabilityConfig, ServerConfig, WorkerConfig};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use std::sync::Arc;
use tower::ServiceExt;

#[tokio::test]
async fn forwards_whitelisted_headers_strips_others() {
    let worker = common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        models: vec![ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
        }],
        workers: vec![WorkerConfig {
            url: worker.url.clone(),
        }],
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.clone()));
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));

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
    // Whitelisted headers are forwarded.
    assert!(seen.seen.contains("authorization"));
    assert!(seen.seen.contains("x-request-id"));
    assert!(seen.seen.contains("x-sgl-route-key"));
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
