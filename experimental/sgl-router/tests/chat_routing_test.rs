// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod common;

use sgl_router::config::{Config, ModelConfig, ObservabilityConfig, ServerConfig, WorkerConfig};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use std::sync::Arc;
use tower::ServiceExt;

fn config(worker_url: &str) -> Config {
    Config {
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
            url: worker_url.into(),
        },
    }
}

#[tokio::test]
async fn non_streaming_returns_200() {
    let worker = common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.clone()));
    let ctx = Arc::new(AppContext::new(cfg, tokenizers, proxy));
    let app = build_router(ctx);

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
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "ok");
}

#[tokio::test]
async fn upstream_unreachable_502() {
    let cfg = config("http://127.0.0.1:1"); // no listener
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(cfg.worker.url.clone()));
    let ctx = Arc::new(AppContext::new(cfg, tokenizers, proxy));
    let app = build_router(ctx);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .unwrap(),
        ))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(
        res.headers().get("x-router-error-code").unwrap(),
        "upstream_error"
    );
}
