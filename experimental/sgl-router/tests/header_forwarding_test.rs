// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod common;

use sgl_router::config::{Config, ModelConfig, ObservabilityConfig, ServerConfig, WorkerConfig};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use axum::body::Body;
use axum::http::Request;
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
        worker: WorkerConfig {
            url: worker.url.clone(),
        },
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.clone()));
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));

    let body = serde_json::to_vec(&serde_json::json!({
        "model":"tiny","messages":[{"role":"user","content":"hi"}]
    }))
    .unwrap();

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("authorization", "Bearer test")
        .header("x-request-id", "abc-123")
        .header("x-sgl-route-key", "k1")
        .header("cookie", "should-not-forward=true")
        .header("host", "example.com")
        .body(Body::from(body))
        .unwrap();
    app.oneshot(req).await.unwrap();

    let seen = worker.captured.lock().unwrap();
    assert!(seen.seen.contains("authorization"));
    assert!(seen.seen.contains("x-request-id"));
    assert!(seen.seen.contains("x-sgl-route-key"));
    assert!(!seen.seen.contains("cookie"));
    // axum/reqwest reset host automatically; we just check we didn't propagate
    // the inbound Host: example.com value (reqwest will set host to the worker's
    // bound address, not the client-supplied "example.com")
    assert!(!seen.seen.contains("host") || !seen.seen.contains("example.com"));
}
