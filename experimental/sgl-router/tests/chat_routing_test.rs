// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

mod common;

use sgl_router::config::{
    Config, DiscoveryBackend, DiscoveryConfig, ModelConfig, ObservabilityConfig, ServerConfig,
    StaticFileDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry as build_policy_registry;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{Worker, WorkerRegistry};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

fn config_for(_worker_url: &str) -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        models: vec![ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: "round_robin".into(),
            circuit_breaker: None,
        }],
        discovery: DiscoveryConfig {
            backend: DiscoveryBackend::StaticFile(StaticFileDiscoveryConfig {
                path: "/tmp/x.toml".into(),
                poll_interval_ms: 200,
            }),
        },
    }
}

fn build_ctx_with_worker(url: &str) -> Arc<AppContext> {
    let cfg = config_for(url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    registry.add(WorkerSpec {
        id: WorkerId("w1".into()),
        url: url.to_string(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
    });
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    // Proxy still requires a worker_url for the legacy constructor; the
    // breaker-gated `forward_*_to` methods supply per-request URLs from the
    // registry, so this placeholder is never used for routing.
    let placeholder_url = reqwest::Url::parse("http://placeholder.invalid").unwrap();
    let proxy = Arc::new(Proxy::new(placeholder_url, TEST_TIMEOUT).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

#[tokio::test]
async fn non_streaming_returns_200() {
    let worker = common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_worker(&worker.url);
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

    let ctx = build_ctx_with_worker(&dead_url);
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
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

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
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker = common::mock_worker::MockWorker::start(chunks).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

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

    let ctx_a = build_ctx_with_worker(&worker_a.url);
    let ctx_b = build_ctx_with_worker(&worker_b.url);
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
    let worker = common::mock_worker::MockWorker::start_returning_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        serde_json::json!({"error": {"type": "upstream", "message": "boom"}}),
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

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
    let ctx = build_ctx_with_worker(&worker.url);
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
    let ctx = build_ctx_with_worker(&worker.url);
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
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

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
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

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
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

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
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

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
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

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
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

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
    let worker = common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);
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
    // Worker must NOT have received a body for this request.
    let captured = worker.captured.lock().unwrap();
    assert!(
        captured.last_body.is_none(),
        "router must not forward malformed JSON to upstream worker; got body: {:?}",
        captured.last_body
    );
}

#[tokio::test]
async fn no_healthy_workers_returns_503() {
    // Build a context with an empty registry for model "tiny" — no workers.
    let cfg = config_for("http://unused");
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default()); // empty — no workers added
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let placeholder_url = reqwest::Url::parse("http://placeholder.invalid").unwrap();
    let proxy = Arc::new(Proxy::new(placeholder_url, TEST_TIMEOUT).unwrap());
    let ctx = Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies));
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
    assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        res.headers().get("x-router-error-code").unwrap(),
        "service_unavailable"
    );
}

#[tokio::test]
async fn forward_json_to_records_failure_on_5xx() {
    use sgl_router::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    use sgl_router::server::error::ApiError;
    use std::sync::Arc;
    use std::time::Duration;

    let worker = common::mock_worker::MockWorker::start_returning_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        serde_json::json!({"error": {"type": "x"}}),
    )
    .await;

    let proxy =
        Proxy::new(worker.url.parse().unwrap(), Duration::from_secs(5)).unwrap();
    let breaker = Arc::new(CircuitBreaker::with_config(CircuitBreakerConfig {
        threshold: 1,
        cool_down: Duration::from_secs(30),
    }));

    let headers = axum::http::HeaderMap::new();
    let body = bytes::Bytes::from(b"{}".to_vec());
    let _: Result<_, ApiError> = proxy
        .forward_json_to(
            &worker.url,
            &breaker,
            "/v1/chat/completions",
            &headers,
            body,
        )
        .await;

    assert!(
        !breaker.allow(),
        "one 5xx with threshold=1 should open the breaker"
    );
}

#[tokio::test]
async fn forward_json_to_rejects_when_breaker_open() {
    use sgl_router::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    use sgl_router::server::error::ApiError;
    use std::sync::Arc;
    use std::time::Duration;

    let worker = common::mock_worker::MockWorker::start(vec![]).await;
    let proxy =
        Proxy::new(worker.url.parse().unwrap(), Duration::from_secs(5)).unwrap();
    let breaker = Arc::new(CircuitBreaker::with_config(CircuitBreakerConfig {
        threshold: 1,
        cool_down: Duration::from_secs(30),
    }));
    breaker.record_failure(); // open immediately

    let headers = axum::http::HeaderMap::new();
    let body = bytes::Bytes::from(b"{}".to_vec());
    let res = proxy
        .forward_json_to(
            &worker.url,
            &breaker,
            "/v1/chat/completions",
            &headers,
            body,
        )
        .await;

    let err = res.expect_err("breaker open → ApiError");
    match err {
        ApiError::ServiceUnavailable(_) => {}
        other => panic!("expected ServiceUnavailable, got {other:?}"),
    }
}

/// Regression test: LoadGuard must be held for the *body* lifetime of a
/// streaming response, not just for the handler lifetime.
///
/// Before the fix, the handler dropped `_guard` as soon as it returned
/// (which happens when headers arrive), so `active_load()` was 0 while
/// the SSE pump was still relaying bytes. This test catches that bug.
#[tokio::test]
async fn streaming_load_guard_persists_for_body_lifetime() {
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"c\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    // Each chunk is delayed by 50ms, total ~200ms of streaming.
    let worker = common::mock_worker::MockWorker::start_slow_stream(
        chunks,
        std::time::Duration::from_millis(50),
    )
    .await;

    let cfg = config_for(&worker.url);
    let registry = Arc::new(WorkerRegistry::default());
    registry.add(WorkerSpec {
        id: WorkerId("w1".into()),
        url: worker.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
    });
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let placeholder_url = reqwest::Url::parse("http://placeholder.invalid").unwrap();
    let proxy = Arc::new(Proxy::new(placeholder_url, TEST_TIMEOUT).unwrap());
    let ctx = Arc::new(AppContext::new(
        cfg,
        tokenizers,
        proxy,
        registry.clone(),
        policies,
    ));
    let app = build_router(ctx);

    // Grab the Worker handle so we can assert active_load().
    let w_handle: Arc<Worker> = registry
        .workers_for(&ModelId("tiny".into()))
        .into_iter()
        .next()
        .expect("worker registered");

    let body = serde_json::to_vec(&serde_json::json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
    }))
    .unwrap();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();

    let res = app.oneshot(req).await.unwrap();

    // The handler has returned (headers arrived).  Wait a moment for the
    // first chunk's delay to pass, then assert load is still held.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert!(
        w_handle.active_load() >= 1,
        "load should be >= 1 mid-stream, got {}",
        w_handle.active_load()
    );

    // Drain the entire body — this drives the SSE pump to completion.
    let _bytes = BodyExt::collect(res.into_body()).await.unwrap().to_bytes();

    // After the body is fully consumed and dropped, the guard must be
    // released.  Give the spawned task a brief moment to clean up.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert_eq!(
        w_handle.active_load(),
        0,
        "load should be 0 after stream completes"
    );
}
