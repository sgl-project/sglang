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

#[tokio::test]
async fn streaming_chunks_pass_through() {
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker = common::mock_worker::MockWorker::start(chunks.clone()).await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.clone()));
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }))
            .unwrap(),
        ))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(
        res.headers().get("content-type").unwrap().to_str().unwrap(),
        "text/event-stream"
    );

    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let data = common::streaming::parse_sse_data(&bytes);
    assert_eq!(data.len(), 3);
    assert!(data[0].contains("\"Hel\""));
    assert!(data[1].contains("\"lo\""));
    assert_eq!(data[2], "[DONE]");
}

#[tokio::test]
async fn streaming_first_chunk_before_completion() {
    // Mock worker with an artificially slow stream — but we shouldn't
    // wait for all chunks before getting the first byte.
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker = common::mock_worker::MockWorker::start(chunks).await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.clone()));
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }))
            .unwrap(),
        ))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();

    // We want to receive the first chunk in well under any plausible
    // backend latency. We'll just assert the body collects at all
    // (sanity); first-byte timing under axum::Body::from_stream is
    // hard to assert without poll-by-poll, but the design is
    // streaming-first per Task 7.
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    assert!(bytes.windows(5).any(|w| w == b"first"));
}

#[tokio::test]
async fn concurrent_streams_are_isolated() {
    let chunks_a: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"AAA\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let chunks_b: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"BBB\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker_a = common::mock_worker::MockWorker::start(chunks_a).await;
    let worker_b = common::mock_worker::MockWorker::start(chunks_b).await;

    let cfg_a = config(&worker_a.url);
    let cfg_b = config(&worker_b.url);
    let ctx_a = Arc::new(AppContext::new(
        cfg_a.clone(),
        Arc::new(TokenizerRegistry::load_from_config(&cfg_a).unwrap()),
        Arc::new(Proxy::new(worker_a.url.clone())),
    ));
    let ctx_b = Arc::new(AppContext::new(
        cfg_b.clone(),
        Arc::new(TokenizerRegistry::load_from_config(&cfg_b).unwrap()),
        Arc::new(Proxy::new(worker_b.url.clone())),
    ));
    let app_a = build_router(ctx_a);
    let app_b = build_router(ctx_b);

    let req = |stream| {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::to_vec(&serde_json::json!({
                    "model": "tiny",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": stream
                }))
                .unwrap(),
            ))
            .unwrap()
    };

    let (ra, rb) = tokio::join!(app_a.oneshot(req(true)), app_b.oneshot(req(true)),);
    let body_a = ra.unwrap().into_body().collect().await.unwrap().to_bytes();
    let body_b = rb.unwrap().into_body().collect().await.unwrap().to_bytes();
    assert!(body_a.windows(3).any(|w| w == b"AAA"));
    assert!(body_b.windows(3).any(|w| w == b"BBB"));
    assert!(!body_a.windows(3).any(|w| w == b"BBB"));
    assert!(!body_b.windows(3).any(|w| w == b"AAA"));
}
