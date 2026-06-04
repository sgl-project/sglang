// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, DiscoveryConfig, ModelConfig, ObservabilityConfig,
    PolicyKind, ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults as build_policy_registry;
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
            policy: PolicyKind::RoundRobin,
            circuit_breaker: None,
            cache_aware: None,
        }],
        discovery: DiscoveryConfig {
            backend: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                urls: vec!["http://placeholder:0".into()],
            }),
        },
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    }
}

fn build_ctx_with_worker(url: &str) -> Arc<AppContext> {
    let cfg = config_for(url);
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
    // Per-request worker URLs flow from the registry through
    // `forward_*_to(&worker.url, ...)`; the proxy itself is URL-less.
    let proxy = Arc::new(Proxy::new(TEST_TIMEOUT).unwrap());
    Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies))
}

#[tokio::test]
async fn non_streaming_returns_200() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
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
    let worker = crate::common::mock_worker::MockWorker::start(chunks.clone()).await;
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
    let data = crate::common::streaming::parse_sse_data(&bytes);
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
    let worker = crate::common::mock_worker::MockWorker::start(chunks).await;
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
    let worker_a = crate::common::mock_worker::MockWorker::start(chunks_a).await;
    let worker_b = crate::common::mock_worker::MockWorker::start(chunks_b).await;

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
    let worker = crate::common::mock_worker::MockWorker::start_returning_error(
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
    let worker = crate::common::mock_worker::MockWorker::start_returning_error(
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
    let worker = crate::common::mock_worker::MockWorker::start_returning_error(
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
    //
    // We register `tiny` as the model so the handler resolves it against
    // the registry, then have the worker simulate a 4xx — this test is
    // about *upstream-returned* errors passing through, not about a
    // router-side model-not-found error.
    let upstream_body = serde_json::json!({
        "error": {"type": "invalid_request_error", "message": "bad input"}
    });
    let worker = crate::common::mock_worker::MockWorker::start_returning_error(
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
                "model": "tiny",
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
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
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
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
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
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
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
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
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
    let worker = crate::common::mock_worker::MockWorker::start_returning_partial_body(
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
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
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
    let proxy = Arc::new(Proxy::new(TEST_TIMEOUT).unwrap());
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
        "no_healthy_workers"
    );
}

/// A worker is registered for a model that is NOT in `cfg.models` (so the
/// policy registry has no entry for it).  The handler returns 404
/// `model_not_found` rather than 500 — clients can recover by sending a
/// different model name; an internal_error would mask the misconfiguration.
#[tokio::test]
async fn unknown_model_with_no_policy_returns_404_model_not_found() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let cfg = config_for(&worker.url);
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    // Register a worker that claims to serve "ghost-7b" — a model the
    // policy registry knows nothing about.
    let _ = registry.add(WorkerSpec {
        id: WorkerId("w-ghost".into()),
        url: worker.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("ghost-7b".into())],
        bootstrap_port: None,
    });
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(TEST_TIMEOUT).unwrap());
    let ctx = Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies));
    let app = build_router(ctx);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "ghost-7b",
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .unwrap(),
        ))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    assert_eq!(
        res.headers().get("x-router-error-code").unwrap(),
        "model_not_found",
    );
}

#[tokio::test]
async fn forward_json_to_records_failure_on_body_drop() {
    // Regression: previously `forward_json_to` recorded breaker
    // success/failure right after headers — so a worker that returned
    // 200 OK and then dropped the body got credited as healthy. A worker
    // that does this repeatedly stays eligible. The fix moves the
    // breaker record to after the body completes, treating a body-drop
    // as failure.
    use sgl_router::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    use sgl_router::server::error::ApiError;
    use std::sync::Arc;
    use std::time::Duration;

    let worker = crate::common::mock_worker::MockWorker::start_returning_partial_body(
        StatusCode::OK,
        b"{\"par",
    )
    .await;

    let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
    let breaker = Arc::new(CircuitBreaker::with_config(CircuitBreakerConfig {
        threshold: std::num::NonZeroU32::new(1).unwrap(),
        cool_down: Duration::from_secs(30),
    }));

    let headers = axum::http::HeaderMap::new();
    let body = bytes::Bytes::from(b"{}".to_vec());
    let res: Result<_, ApiError> = proxy
        .forward_json_to(
            &worker.url,
            &breaker,
            "/v1/chat/completions",
            &headers,
            body,
        )
        .await;
    assert!(res.is_err(), "body drop should surface as ApiError");
    assert!(
        !breaker.would_allow(),
        "body drop must trip the breaker (threshold=1)"
    );
}

#[tokio::test]
async fn forward_json_to_records_success_only_after_body_completes() {
    // Counterpart of the body-drop regression: clean 2xx + clean body
    // MUST call `record_success` on the breaker, even if there were
    // prior failures. Without this, the breaker can never recover from
    // a transient failure spike — it would open on the threshold-th
    // failure and stay open until cool_down, ignoring any successful
    // traffic in between.
    //
    // An earlier version of this test only asserted `breaker.would_allow()`
    // after a single clean call against a fresh breaker, which is true
    // by default — the test never actually observed the success path
    // affecting breaker state. We instead seed one prior failure (one
    // short of threshold), make a clean call, then induce one more
    // failure. If `record_success` fired on the clean call, the failure
    // count is back to 1 and the breaker stays closed. If it didn't,
    // the count is now 2 and the breaker opens.
    use sgl_router::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    use sgl_router::server::error::ApiError;
    use std::sync::Arc;
    use std::time::Duration;

    let ok_worker = crate::common::mock_worker::MockWorker::start_returning_error(
        StatusCode::OK,
        serde_json::json!({}),
    )
    .await;
    let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
    let breaker = Arc::new(CircuitBreaker::with_config(CircuitBreakerConfig {
        threshold: std::num::NonZeroU32::new(2).unwrap(),
        cool_down: Duration::from_secs(30),
    }));
    // Seed one prior failure (threshold-1) — breaker still admits.
    breaker.record_failure();
    assert!(
        breaker.would_allow(),
        "one failure under threshold=2 keeps the breaker closed (sanity)",
    );

    let headers = axum::http::HeaderMap::new();
    let res: Result<_, ApiError> = proxy
        .forward_json_to(
            &ok_worker.url,
            &breaker,
            "/v1/chat/completions",
            &headers,
            bytes::Bytes::from_static(b"{}"),
        )
        .await;
    assert!(res.is_ok(), "clean OK call must succeed: {res:?}");

    // The observable side-effect of `record_success` on the OK body
    // path: failure count is reset to 0. One more failure now must
    // leave us at 1 (not 2), so the breaker stays closed.
    breaker.record_failure();
    assert!(
        breaker.would_allow(),
        "clean success on the OK body path must reset the failure count — \
         if `record_success` was never called, the seed failure would still \
         be live and this single new failure would trip threshold=2",
    );
}

#[tokio::test]
async fn forward_streaming_to_records_failure_on_mid_stream_drop() {
    // Streaming counterpart of the body-drop regression. Headers say 200
    // OK, then the worker drops mid-body. The breaker must observe this
    // as a failure — `bytes_stream_to_body` reads the rest of the
    // stream on a spawned pump, so the recording has to flow through
    // that pump's completion path.
    use http_body_util::BodyExt;
    use sgl_router::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    use sgl_router::server::error::ApiError;
    use std::sync::Arc;
    use std::time::Duration;

    let worker = crate::common::mock_worker::MockWorker::start_returning_partial_body(
        StatusCode::OK,
        b"data: hi\n\n",
    )
    .await;

    let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
    let breaker = Arc::new(CircuitBreaker::with_config(CircuitBreakerConfig {
        threshold: std::num::NonZeroU32::new(1).unwrap(),
        cool_down: Duration::from_secs(30),
    }));

    let headers = axum::http::HeaderMap::new();
    let body = bytes::Bytes::from(b"{}".to_vec());
    let res: Result<_, ApiError> = proxy
        .forward_streaming_to(
            &worker.url,
            &breaker,
            "/v1/chat/completions",
            &headers,
            body,
            None,
        )
        .await;

    let resp = res.expect("headers are 200 OK; transport-level Ok");
    // Drain the body — the pump will see the mid-flight drop and
    // surface an error chunk, then close.
    let _ = resp.into_body().collect().await;
    // After the stream drains, the breaker MUST have recorded failure.
    // Poll briefly because the pump runs on a spawned task.
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while breaker.would_allow() && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        !breaker.would_allow(),
        "stream drop must trip the breaker (threshold=1)"
    );
}

#[tokio::test]
async fn forward_json_to_records_failure_on_5xx() {
    use sgl_router::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    use sgl_router::server::error::ApiError;
    use std::sync::Arc;
    use std::time::Duration;

    let worker = crate::common::mock_worker::MockWorker::start_returning_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        serde_json::json!({"error": {"type": "x"}}),
    )
    .await;

    let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
    let breaker = Arc::new(CircuitBreaker::with_config(CircuitBreakerConfig {
        threshold: std::num::NonZeroU32::new(1).unwrap(),
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

    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
    let breaker = Arc::new(CircuitBreaker::with_config(CircuitBreakerConfig {
        threshold: std::num::NonZeroU32::new(1).unwrap(),
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
        ApiError::BreakerOpen { .. } => {}
        other => panic!("expected BreakerOpen, got {other:?}"),
    }
}

/// A malformed worker URL (operator typo in `discovery.static_urls`, broken k8s
/// annotation) must surface as 503 `worker_misconfigured` (not 500
/// `internal_error`) AND trip the worker's circuit breaker so the malformed
/// worker drops out of `healthy_workers_for` and subsequent requests skip
/// it.
#[tokio::test]
async fn forward_json_to_malformed_url_returns_worker_misconfigured_and_trips_breaker() {
    use sgl_router::health::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig};
    use sgl_router::server::error::ApiError;
    use std::sync::Arc;
    use std::time::Duration;

    let proxy = Proxy::new(Duration::from_secs(5)).unwrap();
    let breaker = Arc::new(CircuitBreaker::with_config(CircuitBreakerConfig {
        threshold: std::num::NonZeroU32::new(1).unwrap(),
        cool_down: Duration::from_secs(30),
    }));

    let headers = axum::http::HeaderMap::new();
    let body = bytes::Bytes::from(b"{}".to_vec());
    let res = proxy
        .forward_json_to(
            "not-a-url",
            &breaker,
            "/v1/chat/completions",
            &headers,
            body,
        )
        .await;

    let err = res.expect_err("malformed URL → ApiError");
    match &err {
        ApiError::WorkerMisconfigured { worker, .. } => {
            assert_eq!(worker, "not-a-url", "{err:?}");
        }
        other => panic!("expected WorkerMisconfigured, got {other:?}"),
    }
    assert!(
        !breaker.allow(),
        "WorkerMisconfigured must trip the breaker so the worker drops out of selection",
    );
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
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        chunks,
        Duration::from_millis(50),
    )
    .await;

    let cfg = config_for(&worker.url);
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId("w1".into()),
        url: worker.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    });
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(TEST_TIMEOUT).unwrap());
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
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(
        w_handle.active_load() >= 1,
        "load should be >= 1 mid-stream, got {}",
        w_handle.active_load()
    );

    // Drain the entire body — this drives the SSE pump to completion.
    let _bytes = BodyExt::collect(res.into_body()).await.unwrap().to_bytes();

    // After the body is fully consumed and dropped, the guard must be
    // released.  Give the spawned task a brief moment to clean up.
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(
        w_handle.active_load(),
        0,
        "load should be 0 after stream completes"
    );
}

/// Task A: the chat handler mints an `ActiveLoadGuard` from the shared
/// `ActiveLoadRegistry` and drops it when the request completes. The
/// non-streaming path drops the guard on handler exit; this test
/// asserts the round-trip increment → 0 across a single request.
#[tokio::test]
async fn non_streaming_active_load_increments_then_returns_to_zero() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let active_load = Arc::clone(&ctx.active_load);
    let app = build_router(ctx);

    assert_eq!(
        active_load.inflight_count(),
        0,
        "registry must start with no in-flight requests",
    );

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
    assert_eq!(res.status(), StatusCode::OK);
    // Drain the body so any pending background work runs to completion.
    let _ = res.into_body().collect().await.unwrap().to_bytes();

    // The handler has returned, so the active-load guard must have
    // dropped — counters are back to zero.
    assert_eq!(
        active_load.inflight_count(),
        0,
        "active-load registry must be empty after non-streaming handler returns",
    );
    let w_id = WorkerId("w1".into());
    assert_eq!(
        active_load.prefill_load(&w_id),
        0,
        "prefill_load must decrement on response end",
    );
}

/// Task A: the streaming path holds the `ActiveLoadGuard` until the
/// SSE pump finishes. Mid-stream the registry shows `inflight_count >= 1`;
/// after the body drains it returns to 0. Counterpart to
/// `streaming_load_guard_persists_for_body_lifetime` — both guards must
/// live for the FULL response lifetime.
#[tokio::test]
async fn streaming_active_load_persists_for_body_lifetime() {
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"c\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        chunks,
        Duration::from_millis(50),
    )
    .await;

    let cfg = config_for(&worker.url);
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId("w1".into()),
        url: worker.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    });
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(TEST_TIMEOUT).unwrap());
    let ctx = Arc::new(AppContext::new(cfg, tokenizers, proxy, registry, policies));
    let active_load = Arc::clone(&ctx.active_load);
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

    // The handler has returned (headers arrived). The streaming pump is
    // still running, so the registry's per-request entry must remain.
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(
        active_load.inflight_count() >= 1,
        "registry inflight must be >= 1 mid-stream, got {}",
        active_load.inflight_count(),
    );
    let w_id = WorkerId("w1".into());
    assert!(
        active_load.prefill_load(&w_id) >= 1,
        "prefill_load must be > 0 mid-stream, got {}",
        active_load.prefill_load(&w_id),
    );

    // Drain the body — drives the SSE pump to completion.
    let _ = res.into_body().collect().await.unwrap().to_bytes();
    tokio::time::sleep(Duration::from_millis(20)).await;

    assert_eq!(
        active_load.inflight_count(),
        0,
        "registry must be empty after stream drains",
    );
    assert_eq!(
        active_load.prefill_load(&w_id),
        0,
        "prefill_load must be 0 after stream drains",
    );
}

/// Task A: a streaming client that disconnects mid-stream still drops
/// both guards. The SSE pump's `tx.send().await.is_err()` branch is what
/// triggers the drop — when the axum Body is dropped on the client side,
/// the channel receiver closes and the pump exits.
#[tokio::test]
async fn streaming_active_load_drops_on_client_disconnect() {
    // Slow stream: 4 chunks × 100 ms each. The test only reads the
    // first chunk then drops the body, simulating a client disconnect.
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"c\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        chunks,
        Duration::from_millis(100),
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let active_load = Arc::clone(&ctx.active_load);
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

    // Read one chunk to confirm the stream is live, then drop the body.
    use futures::StreamExt;
    let mut data_stream = res.into_body().into_data_stream();
    let _first = data_stream.next().await;
    drop(data_stream);

    // Wait long enough for the SSE pump to notice the receiver-drop and
    // exit (per `bytes_stream_to_body_breaks_on_client_disconnect` test
    // in sse.rs, that takes well under 200 ms).
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert_eq!(
        active_load.inflight_count(),
        0,
        "client disconnect must drop the streaming pump's guards within one tick",
    );
}

/// Task D: stale-request janitor expiry surfaces as HTTP 504 with
/// `x-router-error-code: stale_request_expired`. The chat handler
/// races the upstream fetch against the janitor's per-request
/// cancellation token; when the token wins, the handler returns
/// `ApiError::StaleRequestExpired`.
///
/// Wiring: build an `AppContext` with a short
/// `stale_request_timeout` `ActiveLoadRegistry` + spawn a janitor
/// with sub-second cadence + dispatch to a slow upstream that takes
/// longer than the timeout. The janitor sweeps before the upstream
/// returns; cancellation fires; handler returns 504.
#[tokio::test]
async fn janitor_expiry_returns_504_stale_request_expired() {
    use sgl_router::policies::active_load::{spawn_janitor, ActiveLoadRegistry};
    // Upstream that takes 2s to respond — longer than our 50ms
    // stale_request_timeout.
    let worker =
        crate::common::mock_worker::MockWorker::start_hanging(Duration::from_secs(2)).await;

    let cfg = config_for(&worker.url);
    let registry = Arc::new(WorkerRegistry::default());
    let _ = registry.add(WorkerSpec {
        id: WorkerId("w1".into()),
        url: worker.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    });
    let policies = Arc::new(build_policy_registry(&cfg).unwrap());
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let proxy = Arc::new(Proxy::new(TEST_TIMEOUT).unwrap());
    // Aggressive 50ms timeout: the janitor will sweep on the next
    // tick (every 20ms) and fire the cancellation token before the
    // upstream returns.
    let active_load = ActiveLoadRegistry::new(
        Arc::new(sgl_router::policies::active_load::SystemTimeClock),
        Duration::from_millis(50),
    );
    let _janitor = spawn_janitor(Arc::clone(&active_load), Duration::from_millis(20));
    let ctx = Arc::new(AppContext::with_active_load(
        cfg,
        tokenizers,
        proxy,
        registry,
        policies,
        active_load,
    ));
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
        StatusCode::GATEWAY_TIMEOUT,
        "stale-request expiry must surface as 504",
    );
    assert_eq!(
        res.headers()
            .get("x-router-error-code")
            .and_then(|v| v.to_str().ok()),
        Some("stale_request_expired"),
        "504 response must carry x-router-error-code: stale_request_expired",
    );
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("\"code\":\"stale_request_expired\""),
        "504 body must encode the same code in the JSON envelope: {body_str}",
    );
}

/// Task A: a non-streaming request that errors out (upstream
/// unreachable) still drops the active-load guard. The handler's normal
/// return path is the only drop point — confirming the guard is on the
/// stack (not inside a long-lived future) is what this test pins.
#[tokio::test]
async fn non_streaming_error_path_drops_active_load_guard() {
    // Dead upstream — first connect attempt fails fast.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let dead_url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);

    let ctx = build_ctx_with_worker(&dead_url);
    let active_load = Arc::clone(&ctx.active_load);
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

    // Drain so any drop-on-body-end work runs.
    let _ = res.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        active_load.inflight_count(),
        0,
        "error path must drop the active-load guard",
    );
}
