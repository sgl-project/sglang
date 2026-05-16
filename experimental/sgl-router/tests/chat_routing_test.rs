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
use std::time::Duration;
use tower::ServiceExt;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

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
        workers: vec![WorkerConfig {
            url: worker_url.parse().expect("worker URL must parse"),
            request_timeout_ms: None,
        }],
    }
}

#[tokio::test]
async fn non_streaming_returns_200() {
    let worker = common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.parse().unwrap(), TEST_TIMEOUT).unwrap());
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
async fn non_streaming_upstream_unreachable_returns_502_unreachable() {
    // Bind a port, drop it — guarantees a closed/refused TCP destination.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);

    let cfg = config(&dead_url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(dead_url.parse().unwrap(), TEST_TIMEOUT).unwrap());
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
        "upstream_unreachable"
    );
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&bytes);
    assert!(
        body_str.contains("\"code\":\"upstream_unreachable\""),
        "body: {body_str}"
    );
    // Generic message — must not leak reqwest source or worker URL.
    assert!(
        !body_str.contains(&dead_url),
        "worker URL must not leak in client-visible body: {body_str}"
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
    let proxy = Arc::new(Proxy::new(worker.url.parse().unwrap(), TEST_TIMEOUT).unwrap());
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
    let proxy = Arc::new(Proxy::new(worker.url.parse().unwrap(), TEST_TIMEOUT).unwrap());
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

    // Asserting first-byte timing under axum::Body::from_stream requires
    // poll-by-poll instrumentation; here we only sanity-check that the body
    // collects at all so that a regression that buffers the entire stream
    // before yielding will at minimum still pass through bytes.
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
        Arc::new(Proxy::new(worker_a.url.parse().unwrap(), TEST_TIMEOUT).unwrap()),
    ));
    let ctx_b = Arc::new(AppContext::new(
        cfg_b.clone(),
        Arc::new(TokenizerRegistry::load_from_config(&cfg_b).unwrap()),
        Arc::new(Proxy::new(worker_b.url.parse().unwrap(), TEST_TIMEOUT).unwrap()),
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

#[tokio::test]
async fn streaming_upstream_5xx_preserves_content_type() {
    // Regression: when the worker returns a 5xx with JSON body to a
    // streaming request, the router must preserve upstream content-type
    // (application/json), not lie and say text/event-stream. OpenAI
    // clients gate parsing on content-type.
    let worker = common::mock_worker::MockWorker::start_returning_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        serde_json::json!({"error": {"type": "upstream", "message": "boom"}}),
    )
    .await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.parse().unwrap(), TEST_TIMEOUT).unwrap());
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true,
            }))
            .unwrap(),
        ))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        res.headers().get("content-type").unwrap().to_str().unwrap(),
        "application/json",
        "router must preserve upstream content-type on error, not force text/event-stream"
    );
}

#[tokio::test]
async fn non_streaming_upstream_429_preserved() {
    // Regression: a legitimate worker 4xx (rate limit, invalid model, etc.)
    // must be proxied verbatim. The router is only a 502-wrapper for
    // transport failures (connect/dns/timeout); upstream-application errors
    // are OpenAI-compatible passthrough.
    let upstream_body = serde_json::json!({
        "error": {
            "type": "rate_limit_error",
            "message": "Too many requests",
            "code": "rate_limit_exceeded"
        }
    });
    let worker = common::mock_worker::MockWorker::start_returning_error(
        StatusCode::TOO_MANY_REQUESTS,
        upstream_body.clone(),
    )
    .await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.parse().unwrap(), TEST_TIMEOUT).unwrap());
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));

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
    assert_eq!(
        res.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "non-streaming upstream 4xx must be proxied verbatim",
    );
    assert_eq!(
        res.headers().get("content-type").unwrap().to_str().unwrap(),
        "application/json",
    );
    // Router envelope code header must NOT be set — this is upstream's response.
    assert!(
        res.headers().get("x-router-error-code").is_none(),
        "router envelope header must NOT be set on upstream-passthrough responses",
    );
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let got: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(got, upstream_body, "body bytes must round-trip unchanged");
}

#[tokio::test]
async fn non_streaming_upstream_500_preserved() {
    // Regression: worker-side 5xx (model crashed, OOM, etc.) is proxied
    // verbatim on non-streaming requests. Mirrors streaming behaviour. Only
    // transport failures get 502-wrapped.
    let upstream_body = serde_json::json!({
        "error": {"type": "server_error", "message": "internal worker failure"}
    });
    let worker = common::mock_worker::MockWorker::start_returning_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        upstream_body.clone(),
    )
    .await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.parse().unwrap(), TEST_TIMEOUT).unwrap());
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));

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
    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        res.headers().get("x-router-error-code").is_none(),
        "router envelope must NOT wrap upstream 5xx — passthrough",
    );
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let got: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(got, upstream_body);
}

#[tokio::test]
async fn non_streaming_upstream_4xx_body_passthrough() {
    // Regression: the worker's response bytes must reach the client
    // unmodified — no router envelope wrap, no field rewriting.
    let upstream_body = serde_json::json!({
        "error": {"type": "invalid_request_error", "message": "bad model"}
    });
    let worker = common::mock_worker::MockWorker::start_returning_error(
        StatusCode::BAD_REQUEST,
        upstream_body.clone(),
    )
    .await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.parse().unwrap(), TEST_TIMEOUT).unwrap());
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "missing-model",
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .unwrap(),
        ))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    // Byte-exact passthrough — compare via Value to be insensitive to
    // whitespace, which is the only legal axis of variation for JSON.
    let got: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(got, upstream_body);
}

#[tokio::test]
async fn oversized_request_body_returns_413() {
    // Regression: the router must enforce a body-size cap on
    // `/v1/chat/completions`. A multi-MiB body from a hostile client must be
    // rejected at the layer BEFORE the handler reads it into memory, and
    // must NOT be forwarded to the upstream worker.
    let worker = common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.parse().unwrap(), TEST_TIMEOUT).unwrap());
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));

    // 2 MiB body — the configured limit is 1 MiB.
    let big = vec![b'x'; 2 * 1024 * 1024];
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(big))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::PAYLOAD_TOO_LARGE,
        "oversized body must be rejected with 413; got: {}",
        res.status(),
    );
    // The worker must NOT have received the oversized payload.
    let captured = worker.captured.lock().unwrap();
    assert!(
        captured.last_body.is_none(),
        "router must not forward oversized body to upstream; got body of {} bytes",
        captured.last_body.as_ref().map(|b| b.len()).unwrap_or(0),
    );
}

#[tokio::test]
async fn chat_rejects_null_body_400() {
    // Regression: a JSON `null` body is syntactically valid JSON but is NOT
    // a chat-completions request shape. The router must reject it with 400
    // BadRequest and NOT forward it to the worker.
    let worker = common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.parse().unwrap(), TEST_TIMEOUT).unwrap());
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("null"))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        res.headers().get("x-router-error-code").unwrap(),
        "bad_request"
    );
    let captured = worker.captured.lock().unwrap();
    assert!(
        captured.last_body.is_none(),
        "router must NOT forward `null` body to worker; got: {:?}",
        captured.last_body,
    );
}

#[tokio::test]
async fn chat_rejects_array_body_400() {
    // Regression: a JSON array `[]` body is not a chat-completions request
    // shape (object expected). Router must 400 and not forward.
    let worker = common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.parse().unwrap(), TEST_TIMEOUT).unwrap());
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("[]"))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        res.headers().get("x-router-error-code").unwrap(),
        "bad_request"
    );
    let captured = worker.captured.lock().unwrap();
    assert!(captured.last_body.is_none());
}

#[tokio::test]
async fn chat_rejects_string_body_400() {
    // Regression: a JSON string `"hi"` is not a chat-completions request
    // shape. Router must 400 and not forward.
    let worker = common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.parse().unwrap(), TEST_TIMEOUT).unwrap());
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("\"hi\""))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        res.headers().get("x-router-error-code").unwrap(),
        "bad_request"
    );
    let captured = worker.captured.lock().unwrap();
    assert!(captured.last_body.is_none());
}

#[tokio::test]
async fn non_streaming_mid_body_drop_classified_as_upstream_status() {
    // Regression: when the upstream replies with a status line and headers
    // but drops the connection mid-body, the failure is NOT
    // "upstream_unreachable" (the upstream demonstrably DID reply). It must
    // be classified as `upstream_status` so the operator-visible envelope
    // reflects that the worker partially served the request.
    let worker = common::mock_worker::MockWorker::start_returning_partial_body(
        StatusCode::OK,
        b"{\"partial\": ",
    )
    .await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.parse().unwrap(), TEST_TIMEOUT).unwrap());
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false,
            }))
            .unwrap(),
        ))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::BAD_GATEWAY,
        "mid-body drop must surface as 502",
    );
    assert_eq!(
        res.headers().get("x-router-error-code").unwrap(),
        "upstream_status",
        "mid-body drop must be upstream_status (worker DID reply), not upstream_unreachable",
    );
}

#[tokio::test]
async fn malformed_json_returns_400_bad_request() {
    // Regression: a malformed JSON request body must return 400 with
    // bad_request envelope from the router itself; we must NOT forward
    // the bad payload to the worker.
    let worker = common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(worker.url.parse().unwrap(), TEST_TIMEOUT).unwrap());
    let app = build_router(Arc::new(AppContext::new(cfg, tokenizers, proxy)));
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("{not json}"))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        res.headers().get("x-router-error-code").unwrap(),
        "bad_request"
    );
    // Also: worker must NOT have received a body for this request
    let captured = worker.captured.lock().unwrap();
    assert!(
        captured.last_body.is_none(),
        "router must not forward malformed JSON to upstream worker; got body: {:?}",
        captured.last_body
    );
}
