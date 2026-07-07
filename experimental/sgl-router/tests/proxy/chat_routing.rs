// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults as build_policy_registry;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::server::routes::chat::MAX_CHAT_BODY_BYTES;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::{WireProtocol, Worker, WorkerRegistry};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

fn config_for(_worker_url: &str) -> Config {
    Config {
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

/// A standard `stream: true` chat request against the "tiny" model.
fn streaming_chat_request() -> Request<Body> {
    Request::builder()
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
        .unwrap()
}

/// Poll the metrics rendering until it contains `needle` or a 2 s deadline
/// passes, returning the final rendering either way (the caller asserts, so
/// a miss produces a useful diff). Streaming metrics are recorded from the
/// SSE pump's spawned task, which may not have completed when the client
/// finishes draining the body — a fixed sleep is flaky under CI load.
async fn await_metrics_containing(ctx: &Arc<AppContext>, needle: &str) -> String {
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    loop {
        let m = ctx.metrics.render();
        if m.contains(needle) || std::time::Instant::now() > deadline {
            return m;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
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
    let v: Value = serde_json::from_slice(&bytes).unwrap();
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

/// A successful (2xx) streaming request records both TTFT (fired by the SSE
/// pump on the first chunk) and end-to-end request_duration (recorded by the
/// drop-guard when the stream completes). End-to-end coverage of the chat
/// handler installing the hooks — the sse-level unit tests only cover the
/// pump primitive in isolation.
#[tokio::test]
async fn streaming_2xx_request_records_ttft_and_duration() {
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker = crate::common::mock_worker::MockWorker::start(chunks).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx.clone());

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
    // Draining drives the pump to completion: fires the TTFT hook on the
    // first chunk and drops the duration guard at stream end.
    let _ = res.into_body().collect().await.unwrap().to_bytes();
    // The duration guard records from the pump task; give it a beat to drop,
    // matching the active-load streaming tests' synchronization.
    tokio::time::sleep(Duration::from_millis(20)).await;

    let m = ctx.metrics.render();
    assert!(
        m.contains(r#"sgl_router_ttft_seconds_count{model_id="tiny"} 1"#),
        "TTFT must be recorded once for a 2xx streaming request; got:\n{m}",
    );
    assert!(
        m.contains(r#"sgl_router_request_duration_seconds_count{model_id="tiny"} 1"#),
        "request_duration must be recorded at stream completion; got:\n{m}",
    );
}

/// A 2xx streaming response with N chunks records exactly N-1 inter-token
/// gaps in `sgl_router_itl_seconds` — the first chunk is TTFT, every later
/// chunk contributes one gap. End-to-end coverage of the chat handler
/// installing the ITL hook; the sse-level unit tests cover the pump
/// primitive in isolation.
#[tokio::test]
async fn streaming_2xx_request_records_itl_per_chunk_gap() {
    // start_slow_stream paces the chunks so they arrive as distinct reads —
    // back-to-back writes from the plain mock can coalesce into one TCP
    // segment and undercount the gaps.
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        chunks,
        Duration::from_millis(20),
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx.clone());

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
    let _ = res.into_body().collect().await.unwrap().to_bytes();

    // The ITL hook records from the SSE pump's spawned task — poll briefly.
    let expected = r#"sgl_router_itl_seconds_count{model_id="tiny"} 2"#;
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    let m = loop {
        let m = ctx.metrics.render();
        if m.contains(expected) || std::time::Instant::now() > deadline {
            break m;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    };
    assert!(
        m.contains(expected),
        "3 upstream chunks must record exactly 2 inter-token gaps; got:\n{m}",
    );
}

/// End-of-stream truth: an engine that commits `200 OK`, streams a
/// well-formed in-band `data: {"error"...}` event, and closes cleanly is
/// byte-level indistinguishable from success to every headers-time metric.
/// `sgl_router_stream_outcome_total` must classify it as `inband_error` —
/// and the circuit breaker must NOT trip (transport was healthy; the
/// verdict is application-level, deliberately kept out of routing).
#[tokio::test]
async fn streaming_inband_error_records_stream_outcome_without_tripping_breaker() {
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
        "data: {\"error\": {\"message\": \"The request queue is full.\", \"code\": 503}}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker = crate::common::mock_worker::MockWorker::start(chunks).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx.clone());

    // Run THREE in-band-error streams — the default breaker threshold. If a
    // regression ever counted an in-band error as a breaker failure, three
    // consecutive ones would trip it; a single request could never (one
    // failure < threshold), making the breaker assertion below vacuous.
    for _ in 0..3 {
        let res = app.clone().oneshot(streaming_chat_request()).await.unwrap();
        // The 200 was committed before the in-band error existed — wire truth.
        assert_eq!(res.status(), StatusCode::OK);
        let _ = res.into_body().collect().await.unwrap().to_bytes();
    }

    let expected = format!(
        r#"sgl_router_stream_outcome_total{{worker_url="{}",model_id="tiny",outcome="inband_error"}} 3"#,
        worker.url,
    );
    let m = await_metrics_containing(&ctx, &expected).await;
    assert!(
        m.contains(&expected),
        "in-band errors must be classified at stream end; got:\n{m}",
    );
    // Headers-time counters still (correctly) say 200/success — the new
    // metric is the only place the in-band failure is visible.
    assert!(m.contains(
        r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="200"} 3"#
    ));
    // Transport was clean: threshold-many in-band errors must NOT trip the
    // breaker (they record success, an application-level verdict is kept out
    // of routing).
    for w in ctx.registry.all() {
        assert!(
            w.breaker.would_allow(),
            "in-band errors must not trip the circuit breaker",
        );
    }
}

/// The `ok` leg of the stream-outcome classification: a clean streaming
/// completion records exactly one `outcome="ok"` sample and nothing else.
#[tokio::test]
async fn streaming_clean_completion_records_stream_outcome_ok() {
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker = crate::common::mock_worker::MockWorker::start(chunks).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx.clone());

    let res = app.oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let _ = res.into_body().collect().await.unwrap().to_bytes();

    let expected = format!(
        r#"sgl_router_stream_outcome_total{{worker_url="{}",model_id="tiny",outcome="ok"}} 1"#,
        worker.url,
    );
    let m = await_metrics_containing(&ctx, &expected).await;
    assert!(
        m.contains(&expected),
        "clean stream must record outcome=ok; got:\n{m}",
    );
    for absent in [
        r#"outcome="inband_error""#,
        r#"outcome="upstream_error""#,
        r#"outcome="client_disconnect""#,
    ] {
        assert!(
            !m.contains(absent),
            "clean stream must record ONLY outcome=ok, found {absent}; got:\n{m}",
        );
    }
}

/// A worker that commits 200 and then breaks the byte stream mid-flight must
/// classify as `upstream_error` — exercising the COMPOSED completion hook
/// (breaker record_failure AND the stream-end metric firing from the same
/// closure), which the proxy-level mid-stream-drop test can't see (it passes
/// no on_stream_end hook).
#[tokio::test]
async fn streaming_mid_stream_drop_records_stream_outcome_upstream_error() {
    let worker = crate::common::mock_worker::MockWorker::start_returning_partial_body(
        StatusCode::OK,
        b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx.clone());

    let res = app.oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    // Drain — the pump sees the mid-flight drop and surfaces an error chunk.
    let _ = res.into_body().collect().await;

    let expected = format!(
        r#"sgl_router_stream_outcome_total{{worker_url="{}",model_id="tiny",outcome="upstream_error"}} 1"#,
        worker.url,
    );
    let m = await_metrics_containing(&ctx, &expected).await;
    assert!(
        m.contains(&expected),
        "mid-stream transport drop must classify as upstream_error; got:\n{m}",
    );
}

/// Precedence: an in-band error event followed by a transport break must
/// classify as `inband_error`, not `upstream_error` — the in-band event is
/// the specific signal (the engine's own verdict); the subsequent drop is
/// its side effect. Pins the branch ORDER in chat.rs's outcome mapping.
#[tokio::test]
async fn streaming_inband_error_wins_over_subsequent_transport_drop() {
    let worker = crate::common::mock_worker::MockWorker::start_returning_partial_body(
        StatusCode::OK,
        b"data: {\"error\": {\"message\": \"aborted\", \"code\": 503}}\n\n",
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx.clone());

    let res = app.oneshot(streaming_chat_request()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let _ = res.into_body().collect().await;

    let expected = format!(
        r#"sgl_router_stream_outcome_total{{worker_url="{}",model_id="tiny",outcome="inband_error"}} 1"#,
        worker.url,
    );
    let m = await_metrics_containing(&ctx, &expected).await;
    assert!(
        m.contains(&expected),
        "in-band error must win over the transport drop that follows it; got:\n{m}",
    );
    assert!(
        !m.contains(r#"outcome="upstream_error""#),
        "the same stream must not double-classify; got:\n{m}",
    );
}

/// A non-2xx streaming response must NOT record TTFT (the error body is not a
/// generated token — the gate lives in `Proxy::forward_streaming_to`), but it
/// MUST still record request_duration (latency of a failed request matters)
/// and the response status. Guards the 2xx-gating decision end-to-end.
#[tokio::test]
async fn streaming_5xx_request_records_duration_and_status_but_not_ttft() {
    let worker = crate::common::mock_worker::MockWorker::start_returning_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        serde_json::json!({"error": "boom"}),
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx.clone());

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
    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let _ = res.into_body().collect().await;
    tokio::time::sleep(Duration::from_millis(20)).await;

    let m = ctx.metrics.render();
    assert!(
        !m.contains("sgl_router_ttft_seconds_count{"),
        "TTFT must NOT be recorded for a non-2xx streaming response; got:\n{m}",
    );
    assert!(
        !m.contains("sgl_router_itl_seconds_count{"),
        "ITL must NOT be recorded for a non-2xx streaming response (error-body \
         chunk pacing is not a token cadence); got:\n{m}",
    );
    assert!(
        m.contains(
            r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="500"} 1"#
        ),
        "the 500 status must be counted; got:\n{m}",
    );
    assert!(
        m.contains(r#"sgl_router_request_duration_seconds_count{model_id="tiny"} 1"#),
        "request_duration must be recorded even for a failed streaming request; got:\n{m}",
    );
    // The stream-outcome metric is gated to 2xx-committed streams — a 5xx
    // error body draining is not a stream outcome (it would pollute the
    // metric's ok-ratio with error-body drains classified as `ok`).
    assert!(
        !m.contains("sgl_router_stream_outcome_total{"),
        "stream outcome must NOT be recorded for a non-2xx streaming response; got:\n{m}",
    );
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
    let got: Value = serde_json::from_slice(&bytes).unwrap();
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
    let got: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(got, upstream_body);
}

/// A response the router FORWARDS from a worker with a non-2xx status is an
/// `Ok(Response)` at the router layer (only transport failures become `Err`).
/// The per-worker `sgl_router_worker_requests_total` outcome must be derived from the
/// client-visible HTTP status, not from `Result::Ok`/`Err` — so a forwarded 5xx
/// is counted `outcome="error"`, NOT credited as a success.
#[tokio::test]
async fn forwarded_5xx_records_outcome_error_not_success() {
    let worker = crate::common::mock_worker::MockWorker::start_returning_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        serde_json::json!({"error": {"type": "server_error", "message": "boom"}}),
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx.clone());

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
    assert_eq!(res.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let _ = res.into_body().collect().await.unwrap().to_bytes();

    let m = ctx.metrics.render();
    let error_line = format!(
        r#"sgl_router_worker_requests_total{{worker_url="{}",model_id="tiny",mode="plain",outcome="error"}} 1"#,
        worker.url,
    );
    assert!(
        m.contains(&error_line),
        "a forwarded 5xx must be counted as outcome=\"error\"; got:\n{m}",
    );
    let success_line = format!(
        r#"sgl_router_worker_requests_total{{worker_url="{}",model_id="tiny",mode="plain",outcome="success"}}"#,
        worker.url,
    );
    assert!(
        !m.contains(&success_line),
        "a forwarded 5xx must NOT be credited as a success; got:\n{m}",
    );
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
    let got: Value = serde_json::from_slice(&bytes).unwrap();
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
    let app = build_router(ctx.clone());

    // One byte over the configured cap, so the test tracks the cap
    // (`MAX_CHAT_BODY_BYTES`) instead of a hardcoded size.
    let big = vec![b'x'; MAX_CHAT_BODY_BYTES + 1];
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
    {
        let captured = worker.captured.lock().unwrap();
        assert!(
            captured.last_body.is_none(),
            "router must not forward oversized body to upstream; got body of {} bytes",
            captured.last_body.as_ref().map(|b| b.len()).unwrap_or(0),
        );
    }
    // The 413 is produced by the body-limit layer BEFORE the handler runs; the
    // outermost `access_log_and_record` middleware must still count it.
    assert!(
        ctx.metrics
            .render()
            .contains(r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="413"} 1"#),
        "body-limit 413 must be counted in responses_total: {}",
        ctx.metrics.render(),
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
async fn non_streaming_mid_body_drop_classified_as_upstream_body_incomplete() {
    // Regression: when the upstream replies with a status line and headers
    // but drops the connection mid-body, the failure is NOT
    // "upstream_unreachable" (the upstream demonstrably DID reply). It is
    // classified as `upstream_body_incomplete`, and the worker's real status
    // (200 here) is preserved in `x-router-upstream-status` rather than lost
    // behind the synthesized 502.
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
        "upstream_body_incomplete",
        "mid-body drop must be upstream_body_incomplete (worker DID reply), not upstream_unreachable",
    );
    assert_eq!(
        res.headers().get("x-router-upstream-status").unwrap(),
        "200",
        "the worker's real status must be preserved end-to-end, not discarded",
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

/// A worker is registered for a model that is NOT the configured `cfg.model` (so the
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
            WireProtocol::Http1,
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
            WireProtocol::Http1,
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
            WireProtocol::Http1,
            &breaker,
            "/v1/chat/completions",
            &headers,
            body,
            None,
            None,
            None,
            None,
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

/// Client-visible contract of a streaming mid-body drop, through `build_router`.
/// This is the asymmetric half of the non-streaming case: headers were already
/// sent as 200, so the client keeps a 200 (NOT the synthesized 502 of the
/// non-streaming path), there is NO `x-router-error-code` / `x-router-upstream-status`,
/// and `responses_total` counts it as a 200 (the breaker / duration metrics
/// capture the mid-stream failure — see the breaker test above).
#[tokio::test]
async fn streaming_mid_body_drop_stays_200_with_no_router_headers() {
    let worker = crate::common::mock_worker::MockWorker::start_returning_partial_body(
        StatusCode::OK,
        b"data: hi\n\n",
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx.clone());

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
    assert_eq!(
        res.status(),
        StatusCode::OK,
        "streaming mid-drop: headers were already sent as 200, so the client keeps 200",
    );
    assert!(
        res.headers().get("x-router-error-code").is_none(),
        "a 200-then-drop stream is not router-originated — no x-router-error-code",
    );
    assert!(
        res.headers().get("x-router-upstream-status").is_none(),
        "no status was synthesized over the worker, so no x-router-upstream-status",
    );
    let _ = res.into_body().collect().await;

    let m = ctx.metrics.render();
    assert!(
        m.contains(
            r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="200"} 1"#
        ),
        "a streaming mid-drop counts as a 200 at the edge: {m}",
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
            WireProtocol::Http1,
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
            WireProtocol::Http1,
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
            WireProtocol::Http1,
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
    let app = build_router(ctx.clone());

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
    // The stale-cancel is a routed request (a worker was selected), so it lands
    // in requests_total with the worker's labels and outcome="cancelled" — the
    // 504 → Cancelled mapping in `outcome_from_status`, distinct from "error".
    let m = ctx.metrics.render();
    assert!(
        m.contains(&format!(
            r#"sgl_router_worker_requests_total{{worker_url="{}",model_id="tiny",mode="plain",outcome="cancelled"}}"#,
            worker.url
        )),
        "stale 504 must be counted in worker_requests_total as outcome=cancelled: {m}",
    );

    // The stale-timeout cancel arm leaves the unary abort guard armed (the
    // engine is still generating a response no one will read) — distinct
    // from the client-disconnect trigger covered elsewhere, this is the
    // janitor-driven arm of the same `AbortOnDrop`.
    wait_for_abort(&worker.abort_log, Duration::from_secs(2)).await;
    let log = worker.abort_log.lock().unwrap();
    assert_eq!(
        log.len(),
        1,
        "a stale-request-janitor timeout must trigger exactly one abort"
    );
    let rid = log[0]["rid"].as_str().expect("rid must be a string");
    assert!(rid.starts_with("router-"));
}

/// Streaming counterpart of `janitor_expiry_returns_504_stale_request_expired`.
/// The streaming arm's `AbortOnDrop` guard is constructed INSIDE
/// `forward_streaming_to`, only after a response is received — so a
/// stale-timeout that fires while still waiting on headers (the engine
/// hasn't responded at all yet) drops `fetch` before that guard ever exists.
/// Without an eager pre-headers guard, that would leak the in-flight engine
/// request with no abort sent — exactly the gap the non-streaming arm avoids
/// by constructing its guard before dispatch. This pins the fix.
#[tokio::test]
async fn janitor_expiry_on_streaming_request_before_headers_still_aborts() {
    use sgl_router::policies::active_load::{spawn_janitor, ActiveLoadRegistry};
    // Upstream that takes 2s to even send headers — longer than our 50ms
    // stale_request_timeout, so the janitor fires while `fetch` is still
    // awaiting the response (no headers, no internal guard yet).
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
                "stream": true,
            }))
            .unwrap(),
        ))
        .unwrap();
    let res = app.oneshot(req).await.unwrap();
    assert_eq!(
        res.status(),
        StatusCode::GATEWAY_TIMEOUT,
        "stale-request expiry must surface as 504 for streaming too",
    );

    wait_for_abort(&worker.abort_log, Duration::from_secs(2)).await;
    let log = worker.abort_log.lock().unwrap();
    assert_eq!(
        log.len(),
        1,
        "a stale-timeout firing BEFORE headers arrive on a streaming request \
         must still trigger an abort — the engine may already be working on \
         a request no client will ever see"
    );
    let rid = log[0]["rid"].as_str().expect("rid must be a string");
    assert!(rid.starts_with("router-"));
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

/// End-to-end admission control: with a per-worker cap of 1, a second request
/// parks at the router (does NOT 503) while the first holds the only slot
/// mid-stream, and is admitted once the first stream completes and frees the
/// slot. Exercises the real handler wiring (`acquire` + the `AdmissionGuard`
/// firing on stream end), which the `AdmissionQueue` unit tests cannot reach.
#[tokio::test]
async fn admission_parks_second_request_until_first_stream_frees_the_slot() {
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        chunks,
        Duration::from_millis(80),
    )
    .await;

    // cap=1 per worker, bounded wait queue large enough that a parked request
    // is not shed.
    let mut cfg = config_for(&worker.url);
    cfg.admission = sgl_router::config::AdmissionConfig::Enabled {
        max_concurrent_per_worker: std::num::NonZeroUsize::new(1).unwrap(),
        max_queued_requests: Some(4),
    };
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

    let make_req = || {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        }))
        .unwrap();
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap()
    };

    // Request A claims the only slot. Hold its response (don't drain) so the
    // slot stays occupied for the stream's lifetime.
    let res_a = build_router(ctx.clone()).oneshot(make_req()).await.unwrap();
    assert_eq!(res_a.status(), StatusCode::OK);

    // Request B must park (cap=1, A holds the slot). Spawn it and confirm it
    // neither completes nor sheds while A is in flight.
    let ctx_b = ctx.clone();
    let b = tokio::spawn(async move { build_router(ctx_b).oneshot(make_req()).await.unwrap() });
    tokio::time::sleep(Duration::from_millis(40)).await;
    assert!(
        !b.is_finished(),
        "B should park while A holds the only slot, not return"
    );
    assert!(
        ctx.metrics
            .render()
            .contains("sgl_router_queued_requests 1\n"),
        "B should be counted as parked: {}",
        ctx.metrics.render(),
    );

    // Drain A: frees the slot and wakes B.
    let _ = BodyExt::collect(res_a.into_body()).await.unwrap();

    // B is now admitted and completes successfully.
    let res_b = tokio::time::timeout(Duration::from_secs(2), b)
        .await
        .expect("B must be admitted once A frees the slot")
        .expect("B task panicked");
    assert_eq!(res_b.status(), StatusCode::OK);
    let _ = BodyExt::collect(res_b.into_body()).await.unwrap();

    // Wait queue fully drained.
    assert!(
        ctx.metrics
            .render()
            .contains("sgl_router_queued_requests 0\n"),
        "queue should drain to 0: {}",
        ctx.metrics.render(),
    );
}

/// A request shed by the admission gate (503 `service_overloaded`) must show up
/// in the dedicated `sgl_router_backpressure_rejected_total` counter, the general
/// `sgl_router_responses_total{status_code="503"}` series, AND
/// `sgl_router_worker_requests_total{outcome="error"}` with an empty `worker_url`. The
/// shed returns via `?` before reaching a worker, so only the outermost
/// `access_log_and_record` middleware can count it — which is what makes
/// `sum by (outcome)` include sheds instead of undercounting them.
#[tokio::test]
async fn admission_shed_503_is_counted_in_responses_total() {
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        chunks,
        Duration::from_millis(80),
    )
    .await;

    // cap=1, depth cap 0: once the single slot is taken, the next request is
    // shed immediately (never parks).
    let mut cfg = config_for(&worker.url);
    cfg.admission = sgl_router::config::AdmissionConfig::Enabled {
        max_concurrent_per_worker: std::num::NonZeroUsize::new(1).unwrap(),
        max_queued_requests: Some(0),
    };
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

    let make_req = || {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "tiny",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        }))
        .unwrap();
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap()
    };

    // A claims the only slot and holds it mid-stream (don't drain the body).
    let res_a = build_router(ctx.clone()).oneshot(make_req()).await.unwrap();
    assert_eq!(res_a.status(), StatusCode::OK);

    // B is shed: worker at cap, wait queue depth 0 -> 503 service_overloaded.
    let res_b = build_router(ctx.clone()).oneshot(make_req()).await.unwrap();
    assert_eq!(res_b.status(), StatusCode::SERVICE_UNAVAILABLE);

    let m = ctx.metrics.render();
    // The dedicated reject counter (was already wired).
    assert!(
        m.contains(r#"sgl_router_backpressure_rejected_total{model_id="tiny"} 1"#),
        "shed must increment the dedicated reject counter: {m}",
    );
    // The general response-code series now sees the shed too (the fix).
    assert!(
        m.contains(
            r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="503"} 1"#
        ),
        "shed 503 must be counted in responses_total: {m}",
    );
    // And the by-outcome request counter: the shed returns before routing, so the
    // middleware records it with an empty worker_url and outcome="error". This is
    // the line that makes `sum by (outcome)` include sheds.
    assert!(
        m.lines()
            .any(|l| l.starts_with("sgl_router_worker_requests_total{")
                && l.contains(r#"worker_url="""#)
                && l.contains(r#"outcome="error""#)),
        "shed must be counted in worker_requests_total with empty worker_url + outcome=error: {m}",
    );

    // Drain A to release resources cleanly.
    let _ = BodyExt::collect(res_a.into_body()).await.unwrap();
}

/// Guards the refactor that moved status counting into the global middleware:
/// a successful (200) response must be counted in `sgl_router_responses_total`
/// exactly once per request — not double-counted (a leftover handler call) nor
/// dropped (the middleware skipping the success path). Two requests => exactly 2.
#[tokio::test]
async fn success_200_counted_once_per_request_in_responses_total() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_worker(&worker.url);

    let make_req = || {
        Request::builder()
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
            .unwrap()
    };

    for _ in 0..2 {
        let res = build_router(ctx.clone()).oneshot(make_req()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let _ = res.into_body().collect().await.unwrap();
    }

    let m = ctx.metrics.render();
    assert!(
        m.contains(
            r#"sgl_router_responses_total{route="/v1/chat/completions",method="POST",status_code="200"} 2"#
        ),
        "two successes must count to exactly 2 (no double- or under-count): {m}",
    );
    // Both successes are also true intake (counted at entry, through build_router).
    assert!(
        m.contains(r#"sgl_router_requests_total{route="/v1/chat/completions",method="POST"} 2"#),
        "two successes must each be counted as intake: {m}",
    );
    // A routed success must carry its per-worker labels in requests_total — the
    // handler attaches them via RequestLogContext and the middleware records
    // them. Guards the label migration against regressing routed traffic to an
    // empty worker_url (which would blank the per-worker convergence panels).
    assert!(
        ctx.metrics.render().contains(&format!(
            r#"sgl_router_worker_requests_total{{worker_url="{}",model_id="tiny",mode="plain",outcome="success"}} 2"#,
            worker.url
        )),
        "two routed successes must count to 2 with full per-worker labels: {}",
        ctx.metrics.render(),
    );
}

/// Infra paths (`/healthz`, `/readyz`, `/metrics`) are health/scrape probes. The
/// middleware counts them in the edge counters `requests_total{route,method}` and
/// `responses_total{route,method,status_code}` (so a failing probe stays
/// observable), but deliberately skips the per-worker `worker_requests_total` —
/// otherwise probe successes would swamp the by-outcome view. A `/healthz` 200
/// must therefore appear in the edge counters yet leave `worker_requests_total`
/// empty.
#[tokio::test]
async fn infra_path_counted_in_edge_counters_but_not_worker_requests_total() {
    let ctx = build_ctx_with_worker("http://placeholder:0");

    let req = Request::builder()
        .method("GET")
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();
    let res = build_router(ctx.clone()).oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let _ = res.into_body().collect().await.unwrap();

    let m = ctx.metrics.render();
    // Counted in the edge counters — a probe must stay observable there.
    assert!(
        m.contains(r#"sgl_router_requests_total{route="/healthz",method="GET"} 1"#),
        "infra intake must be counted in requests_total: {m}",
    );
    assert!(
        m.contains(
            r#"sgl_router_responses_total{route="/healthz",method="GET",status_code="200"} 1"#
        ),
        "infra 200 must be counted in responses_total: {m}",
    );
    // ...but NOT in the per-worker counter: probe successes must not swamp the
    // by-outcome view (the `# TYPE` line starts with `#`, not the metric name, so
    // it is not matched here).
    assert!(
        !m.lines()
            .any(|l| l.starts_with("sgl_router_worker_requests_total{")),
        "infra path must NOT be counted in worker_requests_total: {m}",
    );
}

/// A request to a path the router does not serve (a true 404) collapses to the
/// `route="unmatched"` label — NOT the raw URI — so scanner / fuzzer traffic
/// can't explode the metric label cardinality. It is still counted at the edge.
#[tokio::test]
async fn unrouted_path_collapses_to_unmatched_label() {
    let ctx = build_ctx_with_worker("http://placeholder:0");

    let req = Request::builder()
        .method("GET")
        .uri("/no/such/route")
        .body(Body::empty())
        .unwrap();
    let res = build_router(ctx.clone()).oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let _ = res.into_body().collect().await.unwrap();

    let m = ctx.metrics.render();
    assert!(
        m.contains(r#"sgl_router_requests_total{route="unmatched",method="GET"} 1"#),
        "an unrouted path must collapse to route=\"unmatched\": {m}",
    );
    // The raw URI must never become a metric label (cardinality safety).
    assert!(
        !m.contains("/no/such/route"),
        "raw unrouted URI must not appear as a label: {m}",
    );
}

/// Regression for the production false-shedding leak: with admission ENABLED, a
/// streaming request whose client disconnects mid-stream must release its
/// per-worker admission slot (`Worker.active_requests`), not just the
/// active-load registry. A leaked slot pins the worker at its cap, so the gate
/// then sheds later requests with 503 while the engine is actually idle.
///
/// `streaming_active_load_drops_on_client_disconnect` covers the active-load
/// *registry* on this path; this covers the *admission* counter the gate reads.
#[tokio::test]
async fn admission_slot_released_on_streaming_client_disconnect() {
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

    // cap=1, depth 0: if the slot leaks on disconnect, the next request is shed.
    let mut cfg = config_for(&worker.url);
    cfg.admission = sgl_router::config::AdmissionConfig::Enabled {
        max_concurrent_per_worker: std::num::NonZeroUsize::new(1).unwrap(),
        max_queued_requests: Some(0),
    };
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
        Arc::clone(&registry),
        policies,
    ));
    let w = registry.get(&WorkerId("w1".into())).unwrap();

    let make_req = || {
        Request::builder()
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
            .unwrap()
    };

    // Request A claims the only slot; read one chunk so the stream is live.
    let res_a = build_router(ctx.clone()).oneshot(make_req()).await.unwrap();
    assert_eq!(res_a.status(), StatusCode::OK);
    use futures::StreamExt;
    let mut data_stream = res_a.into_body().into_data_stream();
    let _first = data_stream.next().await;
    assert_eq!(
        w.active_load(),
        1,
        "slot must be held while the stream is live"
    );

    // Client disconnects mid-stream.
    drop(data_stream);

    // The SSE pump should notice the receiver-drop and release the admission slot.
    for _ in 0..100 {
        if w.active_load() == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        w.active_load(),
        0,
        "admission slot must be released when the streaming client disconnects",
    );

    // Behavioral check: a fresh request must be admitted, not falsely shed.
    let res_b = build_router(ctx.clone()).oneshot(make_req()).await.unwrap();
    assert_eq!(
        res_b.status(),
        StatusCode::OK,
        "next request must be admitted after the slot is released, not 503",
    );
    let _ = res_b.into_body().collect().await.unwrap();
}

/// Closer reproduction of the production false-shedding symptom: sustained
/// CONCURRENT streaming load through the admission queue's hand-off path, with
/// clients disconnecting mid-stream. Every slot claimed (directly or via
/// hand-off to a parked waiter) must be released; if any leaks, the worker pins
/// at its cap, later waiters never admit, and the gate sheds while the engine is
/// idle. Asserts the worker drains to zero and all requests are admitted.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn admission_slots_not_leaked_under_concurrent_streaming_disconnects() {
    let chunks: Vec<&'static str> = vec![
        "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"c\"}}]}\n\n",
        "data: [DONE]\n\n",
    ];
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        chunks,
        Duration::from_millis(40),
    )
    .await;

    // cap=2 + a queue deep enough that all arrivals PARK (none shed) -> every
    // admission after the first two goes through the FIFO hand-off path.
    let mut cfg = config_for(&worker.url);
    cfg.admission = sgl_router::config::AdmissionConfig::Enabled {
        max_concurrent_per_worker: std::num::NonZeroUsize::new(2).unwrap(),
        max_queued_requests: Some(64),
    };
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
        Arc::clone(&registry),
        policies,
    ));
    let w = registry.get(&WorkerId("w1".into())).unwrap();

    let mut handles = Vec::new();
    for _ in 0..24u32 {
        let ctx2 = ctx.clone();
        handles.push(tokio::spawn(async move {
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
            // Each request must be admitted (parked then handed a slot); a leak
            // would make a later waiter hang here forever.
            let res =
                tokio::time::timeout(Duration::from_secs(10), build_router(ctx2).oneshot(req))
                    .await
                    .expect("request must admit within timeout (a leaked slot would hang it)")
                    .unwrap();
            let status = res.status();
            // Read one chunk, then disconnect mid-stream.
            use futures::StreamExt;
            let mut ds = res.into_body().into_data_stream();
            let _ = ds.next().await;
            drop(ds);
            status
        }));
    }
    for h in handles {
        let status = h.await.expect("task panicked");
        assert_eq!(
            status,
            StatusCode::OK,
            "every request must be admitted (no false shed)"
        );
    }

    // All disconnected; every claimed/handed-off slot must have been released.
    for _ in 0..200 {
        if w.active_load() == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        w.active_load(),
        0,
        "no admission slot may leak after concurrent streaming disconnects",
    );
    assert_eq!(
        ctx.active_load.inflight_count(),
        0,
        "active-load registry must drain too"
    );

    // Every one of the 24 concurrent disconnects must independently trigger
    // its own abort — exercising the shared `AbortOnDrop` machinery (the
    // `reached_end` AtomicBool, the fire-and-forget spawn) under contention,
    // not just a single request at a time.
    wait_for_abort_count(&worker.abort_log, 24, Duration::from_secs(5)).await;
    let log = worker.abort_log.lock().unwrap();
    assert_eq!(
        log.len(),
        24,
        "all 24 concurrent disconnects must each trigger exactly one abort"
    );
    let rids: std::collections::HashSet<&str> =
        log.iter().filter_map(|v| v["rid"].as_str()).collect();
    assert_eq!(
        rids.len(),
        24,
        "all 24 aborted rids must be unique — a collision would mean two requests \
         shared (or one overwrote) the other's router-minted rid"
    );
}

/// Test-only `tracing` writer appending to a shared buffer.
#[derive(Clone)]
struct LogCapture(Arc<std::sync::Mutex<Vec<u8>>>);
impl std::io::Write for LogCapture {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Process-wide capturing subscriber over a shared buffer. `set_global_default`
/// can be installed only once, so a `OnceLock` guards it and the buffer
/// accumulates every test's access log. Capture tests isolate their own lines by
/// a unique `x-request-id` (see [`access_log_count`]), so concurrent tests
/// writing to the shared buffer don't perturb the count. A *global* subscriber
/// (vs a thread-local default) is required because under the parallel suite the
/// handler's access-log events are not reliably emitted on the test's own thread.
fn global_log_buf() -> Arc<std::sync::Mutex<Vec<u8>>> {
    use tracing_subscriber::util::SubscriberInitExt;
    static LOG_BUF: std::sync::OnceLock<Arc<std::sync::Mutex<Vec<u8>>>> =
        std::sync::OnceLock::new();
    LOG_BUF
        .get_or_init(|| {
            let buf = Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
            let b = buf.clone();
            let _ = tracing_subscriber::fmt()
                .with_ansi(false)
                .with_writer(move || LogCapture(b.clone()))
                .finish()
                .try_init();
            buf
        })
        .clone()
}

fn make_chat_req(streaming: bool, request_id: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .header("x-request-id", request_id)
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": streaming,
            }))
            .unwrap(),
        ))
        .unwrap()
}

/// Count access-log lines for one specific request id. Tagging the request with
/// a unique id and filtering on it isolates this test's log lines from any other
/// test's `http_request` events that share the captured `tracing` default
/// under parallel execution.
fn access_log_count(logs: &str, request_id: &str) -> usize {
    logs.lines()
        .filter(|l| l.contains("http_request") && l.contains(request_id))
        .count()
}

/// A post-dispatch error (unreachable upstream → 502) must be logged EXACTLY
/// once in the access log. Logging happens once, centrally, in the
/// `access_log_and_record` middleware (the handler no longer logs), so a routed
/// error produces a single `http_request` line — not two.
#[tokio::test]
async fn post_dispatch_error_logged_once_in_access_log() {
    let buf = global_log_buf();
    let rid = "rid-post-dispatch-once";

    // Worker is registered (healthy on add) but unreachable → forward fails with
    // UpstreamUnreachable (502), a post-dispatch error.
    let ctx = build_ctx_with_worker("http://127.0.0.1:1");
    let res = build_router(ctx)
        .oneshot(make_chat_req(false, rid))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::BAD_GATEWAY);

    let logs = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
    let n = access_log_count(&logs, rid);
    assert_eq!(
        n, 1,
        "post-dispatch error must be logged exactly once (no wrapper double-log); got {n}:\n{logs}"
    );
}

/// A successful (200) request must be logged exactly once — by the
/// `access_log_and_record` middleware. Guards against a regression that
/// re-introduces handler-side logging on top of the middleware.
#[tokio::test]
async fn success_logged_once_in_access_log() {
    let buf = global_log_buf();
    let rid = "rid-success-once";

    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let res = build_router(ctx)
        .oneshot(make_chat_req(false, rid))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let _ = res.into_body().collect().await.unwrap();

    let logs = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
    let n = access_log_count(&logs, rid);
    assert_eq!(
        n, 1,
        "a success must be logged exactly once; got {n}:\n{logs}"
    );
}

// ---- Abort-on-disconnect (full handler, via build_router) -----------------
//
// Plain-mode (non-PD) coverage for the `/abort_request` wiring added in
// `chat_completions_inner`: a client-supplied body `rid` is reused verbatim,
// an absent one is minted as `router-<uuid>`, and only an early
// disconnect — never a normal completion — sends an abort. PD-mode exclusion
// is covered separately in `pd_bootstrap_injection.rs`, which already owns
// the PD worker-registry setup.

/// Build a chat-completion request body, optionally carrying a client
/// `"rid"` field (the body-level identifier the engine adopts and the router
/// later aborts by — distinct from the `x-request-id` HTTP header `
/// make_chat_req` sets for access-log correlation).
fn chat_req_with_body_rid(streaming: bool, body_rid: Option<&str>) -> Request<Body> {
    chat_req_with_rids(streaming, body_rid, None)
}

/// Like [`chat_req_with_body_rid`], plus an optional `x-request-id` HEADER —
/// the gateway/access-log correlation id, distinct from the body-level `rid`
/// the engine adopts. Used to test that a missing body `rid` falls back to
/// deriving the engine-facing rid from this header (not an unrelated random
/// UUID), so router/gateway logs (keyed on `x-request-id`) and engine logs
/// (keyed on `rid`) can be cross-referenced by the same token.
fn chat_req_with_rids(
    streaming: bool,
    body_rid: Option<&str>,
    x_request_id: Option<&str>,
) -> Request<Body> {
    let mut body = serde_json::json!({
        "model": "tiny",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": streaming,
    });
    if let Some(rid) = body_rid {
        body["rid"] = Value::String(rid.to_string());
    }
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    if let Some(xrid) = x_request_id {
        builder = builder.header("x-request-id", xrid);
    }
    builder
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

/// Poll `log` until it has at least one entry or `timeout` elapses.
async fn wait_for_abort(log: &Arc<std::sync::Mutex<Vec<Value>>>, timeout: Duration) {
    let deadline = std::time::Instant::now() + timeout;
    while log.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Poll `log` until it has at least `count` entries or `timeout` elapses —
/// the multi-request counterpart of [`wait_for_abort`], for tests asserting
/// on N independent aborts landing rather than just "at least one."
async fn wait_for_abort_count(
    log: &Arc<std::sync::Mutex<Vec<Value>>>,
    count: usize,
    timeout: Duration,
) {
    let deadline = std::time::Instant::now() + timeout;
    while log.lock().unwrap().len() < count && std::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// A client that disconnects mid-SSE-stream (drops the response body before
/// it's drained) must trigger exactly one `/abort_request` to the engine,
/// carrying a `router-`-minted rid (no client `rid` was supplied).
#[tokio::test]
async fn streaming_disconnect_triggers_engine_abort() {
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        vec!["data: a\n\n", "data: b\n\n", "data: c\n\n"],
        Duration::from_millis(50),
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

    let res = app
        .oneshot(chat_req_with_body_rid(true, None))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // Read exactly one chunk, then drop the body — simulating the client
    // going away before the engine finishes streaming.
    use futures::StreamExt;
    let mut data_stream = res.into_body().into_data_stream();
    assert!(
        data_stream.next().await.is_some(),
        "expected at least one chunk before drop"
    );
    drop(data_stream);

    wait_for_abort(&worker.abort_log, Duration::from_secs(2)).await;
    let log = worker.abort_log.lock().unwrap();
    assert_eq!(
        log.len(),
        1,
        "client disconnect mid-stream must trigger exactly one abort"
    );
    let rid = log[0]["rid"].as_str().expect("rid must be a string");
    assert!(
        rid.starts_with("router-"),
        "no client rid was supplied — must mint a `router-`-prefixed rid, got {rid}"
    );
    assert_eq!(log[0]["abort_all"], false);
}

/// A stream drained to its normal completion must never trigger an abort.
#[tokio::test]
async fn streaming_normal_completion_does_not_abort() {
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        vec!["data: a\n\n", "data: b\n\n"],
        Duration::from_millis(10),
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

    let res = app
        .oneshot(chat_req_with_body_rid(true, None))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let _ = res.into_body().collect().await.unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        worker.abort_log.lock().unwrap().is_empty(),
        "a stream drained to completion must never trigger an abort"
    );
}

/// A client-supplied body `rid` must be reused verbatim for the abort — not
/// overridden by a router-minted one — so an external abort-by-rid keeps
/// working.
#[tokio::test]
async fn streaming_disconnect_reuses_client_supplied_rid() {
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        vec!["data: a\n\n", "data: b\n\n", "data: c\n\n"],
        Duration::from_millis(50),
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

    let res = app
        .oneshot(chat_req_with_body_rid(true, Some("client-rid-123")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    use futures::StreamExt;
    let mut data_stream = res.into_body().into_data_stream();
    assert!(data_stream.next().await.is_some());
    drop(data_stream);

    wait_for_abort(&worker.abort_log, Duration::from_secs(2)).await;
    let log = worker.abort_log.lock().unwrap();
    assert_eq!(log.len(), 1);
    assert_eq!(
        log[0]["rid"], "client-rid-123",
        "a client-supplied rid must be reused verbatim, not overridden"
    );
}

/// With no client-supplied body `rid`, the minted engine-facing rid must be
/// derived from the gateway's `x-request-id` header (already logged by the
/// router's own access-log middleware, server/app.rs) — not an unrelated
/// random UUID. This is what lets an operator take a `rid` out of an engine
/// log line and find the matching router/gateway log line via the same
/// `x-request-id` value, and vice versa.
#[tokio::test]
async fn streaming_disconnect_derives_rid_from_x_request_id_header() {
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        vec!["data: a\n\n", "data: b\n\n", "data: c\n\n"],
        Duration::from_millis(50),
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

    let res = app
        .oneshot(chat_req_with_rids(true, None, Some("gw-correlate-456")))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    use futures::StreamExt;
    let mut data_stream = res.into_body().into_data_stream();
    assert!(data_stream.next().await.is_some());
    drop(data_stream);

    wait_for_abort(&worker.abort_log, Duration::from_secs(2)).await;
    let log = worker.abort_log.lock().unwrap();
    assert_eq!(log.len(), 1);
    assert_eq!(
        log[0]["rid"], "router-gw-correlate-456",
        "with no body rid, the minted rid must incorporate x-request-id verbatim, \
         so it's findable from the access-log's own x-request-id field"
    );
}

/// A client-supplied body `rid` still wins over `x-request-id` when both are
/// present — the body-level `rid` is the more specific, explicit signal (an
/// external abort-by-rid integration would set it deliberately), so it must
/// not be silently overridden by the more general gateway-correlation header.
#[tokio::test]
async fn streaming_disconnect_prefers_client_body_rid_over_x_request_id_header() {
    let worker = crate::common::mock_worker::MockWorker::start_slow_stream(
        vec!["data: a\n\n", "data: b\n\n", "data: c\n\n"],
        Duration::from_millis(50),
    )
    .await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

    let res = app
        .oneshot(chat_req_with_rids(
            true,
            Some("client-rid-wins"),
            Some("gw-should-be-ignored"),
        ))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    use futures::StreamExt;
    let mut data_stream = res.into_body().into_data_stream();
    assert!(data_stream.next().await.is_some());
    drop(data_stream);

    wait_for_abort(&worker.abort_log, Duration::from_secs(2)).await;
    let log = worker.abort_log.lock().unwrap();
    assert_eq!(log.len(), 1);
    assert_eq!(
        log[0]["rid"], "client-rid-wins",
        "a client-supplied body rid must take priority over x-request-id"
    );
}

/// Non-streaming: the handler future being dropped before the engine
/// responds (the real shape of a client disconnect under axum — the runtime
/// drops the handler future) must trigger exactly one abort.
#[tokio::test]
async fn non_streaming_handler_drop_triggers_engine_abort() {
    let worker =
        crate::common::mock_worker::MockWorker::start_hanging(Duration::from_secs(10)).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);
    let req = chat_req_with_body_rid(false, None);

    // Spawn the handler call so it can be cancelled mid-flight, exactly as
    // the axum runtime cancels a handler future on a real client disconnect.
    let handle = tokio::spawn(async move { app.oneshot(req).await });
    tokio::time::sleep(Duration::from_millis(150)).await;
    handle.abort();
    let _ = handle.await;

    wait_for_abort(&worker.abort_log, Duration::from_secs(2)).await;
    let log = worker.abort_log.lock().unwrap();
    assert_eq!(
        log.len(),
        1,
        "a dropped handler future (client disconnect) must trigger exactly one abort"
    );
    let rid = log[0]["rid"].as_str().expect("rid must be a string");
    assert!(rid.starts_with("router-"));
    assert_eq!(log[0]["abort_all"], false);
}

/// Non-streaming: a normal completion (the client waits for the full
/// response) must never trigger an abort — the guard must be disarmed before
/// drop.
#[tokio::test]
async fn non_streaming_normal_completion_does_not_abort() {
    let worker = crate::common::mock_worker::MockWorker::start(vec![]).await;
    let ctx = build_ctx_with_worker(&worker.url);
    let app = build_router(ctx);

    let res = app
        .oneshot(chat_req_with_body_rid(false, None))
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let _ = res.into_body().collect().await.unwrap();

    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(
        worker.abort_log.lock().unwrap().is_empty(),
        "a normal completion must never trigger an abort"
    );
}
