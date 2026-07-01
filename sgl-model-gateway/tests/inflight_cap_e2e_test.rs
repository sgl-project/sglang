//! E2E tests for the in-flight request cap.
//!
//! Mock worker slow-stream (`set_slow_stream_chunks`, `response_delay_ms`):
//! the handler sleeps `response_delay_ms` before returning the head, then
//! emits one chunk per `response_delay_ms`. `app.oneshot(req).await`
//! resolves at head-return while the body streams lazily — so firing a
//! second request right after the first head returns hits the window where
//! the first body is still streaming.

mod common;

use std::time::Duration;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use common::{
    mock_worker::{set_slow_stream_chunks, HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext,
};
use serde_json::json;
use smg::config::RouterConfig;
use tower::ServiceExt;

/// In-flight cap on, req/s limiter disabled, no queue.
async fn ctx_with_cap(cap: i32, port: u16) -> AppTestContext {
    let config = RouterConfig::builder()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(30200)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(2)
        .worker_startup_check_interval_secs(1)
        .disable_rate_limiting()
        .queue_size(0)
        .queue_timeout_secs(60)
        .max_inflight_requests(cap)
        .build_unchecked();

    AppTestContext::new_with_config(
        config,
        vec![MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 200,
            fail_rate: 0.0,
        }],
    )
    .await
}

/// Both limiters on. `reqs_cap` req/s capacity, `inflight_cap` in-flight
/// capacity, `queue` queue depth.
async fn ctx_with_both(
    reqs_cap: i32,
    inflight_cap: i32,
    queue: usize,
    port: u16,
) -> AppTestContext {
    let config = RouterConfig::builder()
        .regular_mode(vec![])
        .random_policy()
        .host("127.0.0.1")
        .port(30200)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(2)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(reqs_cap)
        .rate_limit_tokens_per_second(1) // negligible refill so the leak test is timing-proof
        .queue_size(queue)
        .queue_timeout_secs(60)
        .max_inflight_requests(inflight_cap)
        .build_unchecked();

    AppTestContext::new_with_config(
        config,
        vec![MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 300,
            fail_rate: 0.0,
        }],
    )
    .await
}

/// Drain a response body so its TokenGuardBody drops and the permit returns.
async fn drain(resp: axum::response::Response) {
    let _ = axum::body::to_bytes(resp.into_body(), 1024 * 1024).await;
}

fn stream_request() -> Request<Body> {
    let payload = json!({ "text": "hello", "stream": true });
    Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap()
}

/// Permit is held during the active body stream and returned only on body-end.
#[tokio::test]
async fn permit_held_during_stream_and_returned_on_body_end() {
    let port = 19210;
    set_slow_stream_chunks(port, 4);
    let ctx = ctx_with_cap(1, port).await;

    // req1: head returns ~200ms; body streams to ~1000ms. Permit held by body.
    let app = ctx.create_app().await;
    let resp1 = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);

    // req2 fires while req1 body streams: the only permit is held => 429.
    // (Would wrongly be 200 if the permit returned at head-return, not body-end.)
    let app = ctx.create_app().await;
    let resp2 = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::TOO_MANY_REQUESTS);
    drain(resp2).await;

    // Permit returns only when req1's body fully drains.
    drain(resp1).await;
    let app = ctx.create_app().await;
    let resp3 = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(resp3.status(), StatusCode::OK);
    drain(resp3).await;

    ctx.shutdown().await;
}

/// While req1's body streams, every follow-up is 429 until req1 drains.
#[tokio::test]
async fn all_followups_429_until_body_drained() {
    let port = 19212;
    set_slow_stream_chunks(port, 3);
    let ctx = ctx_with_cap(1, port).await;

    let app = ctx.create_app().await;
    let resp1 = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);

    for i in 0..3 {
        let app = ctx.create_app().await;
        let r = app.oneshot(stream_request()).await.unwrap();
        assert_eq!(r.status(), StatusCode::TOO_MANY_REQUESTS, "follow-up #{i}");
        drain(r).await;
    }

    drain(resp1).await;

    let app = ctx.create_app().await;
    let r = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    drain(r).await;

    ctx.shutdown().await;
}

/// The in-flight cap is binding even when the req/s gate has free capacity:
/// req2 passes req/s then 429s on inflight; after req1 drains, req3 succeeds.
#[tokio::test]
async fn inflight_cap_binds_after_reqs_gate() {
    let port = 19220;
    set_slow_stream_chunks(port, 4);
    let ctx = ctx_with_both(8, 1, 0, port).await;

    let app = ctx.create_app().await;
    let resp1 = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);

    // req2 has req/s capacity (8) but the single inflight permit is held.
    let app = ctx.create_app().await;
    let resp2 = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::TOO_MANY_REQUESTS);
    drain(resp2).await;

    drain(resp1).await;

    let app = ctx.create_app().await;
    let resp3 = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(resp3.status(), StatusCode::OK);
    drain(resp3).await;

    ctx.shutdown().await;
}

/// An in-flight overflow must release the req/s token it already acquired —
/// otherwise repeated overflow 429s drain the req/s bucket permanently.
///
/// Proven by inspecting `rate_limiter.available_tokens()`, NOT by status
/// (req/s-429 and inflight-429 are indistinguishable). req/s capacity is 8;
/// req1 holds one token while streaming, leaving seven free. Each overflow
/// acquires one of those, 429s on inflight, and must return it. After the
/// burst ~7 must remain free; a missing release leaves ~0 (refill is 1/s,
/// negligible over a sub-second burst, so 4.0 cleanly separates them).
#[tokio::test]
async fn inflight_overflow_releases_reqs_token_no_leak() {
    let port = 19224;
    set_slow_stream_chunks(port, 6); // req1 streams for ~1.8s
    let ctx = ctx_with_both(8, 1, 0, port).await;

    // req1: holds 1 req/s token + the single inflight permit. 7 req/s free.
    let app = ctx.create_app().await;
    let resp1 = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);

    // Burst: each overflow passes req/s then 429s on inflight. With the
    // release the tokens return; without it they leak (refill is 1/s,
    // too slow to mask the leak over a sub-second burst).
    for _ in 0..10 {
        let app = ctx.create_app().await;
        let r = app.oneshot(stream_request()).await.unwrap();
        assert_eq!(r.status(), StatusCode::TOO_MANY_REQUESTS);
        drain(r).await;
    }

    let rate_limiter = ctx
        .app_context
        .rate_limiter
        .clone()
        .expect("req/s limiter enabled");
    let avail = rate_limiter.available_tokens().await;
    assert!(
        avail > 4.0,
        "inflight overflow must release the req/s token (no leak); \
         available_tokens={avail} (expected ~7, capacity 8 minus req1)"
    );

    drain(resp1).await;
    ctx.shutdown().await;
}

/// Queued requests must NOT hold in-flight permits (queue-fill DoS fix).
///
/// Bug 2: if the in-flight permit were acquired before the req/s gate, a
/// request queued waiting for a req/s token would hold an in-flight slot —
/// exhausting the cap with requests that aren't even running. The fix
/// acquires the in-flight permit inside `run_and_guard`, only after a req/s
/// token is granted.
///
/// req/s cap 1, inflight cap 2, queue 10. req1 streams (holds 1 inflight,
/// 1 free). req2 can't get a req/s token → queues. The free permit must
/// stay available (req2 didn't take it). Proven by inspecting
/// `inflight_limiter.available_tokens()` while req2 sits in the queue.
#[tokio::test]
async fn queued_requests_do_not_hold_inflight_permit() {
    let port = 19230;
    set_slow_stream_chunks(port, 6); // req1 streams for ~1.8s
    let ctx = ctx_with_both(1, 2, 10, port).await;

    // req1: gets the only req/s token + 1 inflight permit.
    let app = ctx.create_app().await;
    let resp1 = app.oneshot(stream_request()).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);

    // req2: req/s exhausted (cap 1) → queues, blocks.
    let app2 = ctx.create_app().await;
    let resp2 = tokio::spawn(async move { app2.oneshot(stream_request()).await.unwrap() });

    tokio::time::sleep(Duration::from_millis(300)).await;

    // req2 is queued (not running) so it must NOT hold the free inflight
    // permit. Fix: avail=1. Bug (inflight before req/s): avail=0.
    let inflight = ctx
        .app_context
        .inflight_limiter
        .as_ref()
        .expect("inflight enabled")
        .clone();
    let avail = inflight.available_tokens().await;
    assert!(
        avail >= 1.0,
        "queued requests must not hold in-flight permits; available={avail}"
    );

    drain(resp1).await;
    // 408 = queued but timed out (pre-existing req/s acquire inner-timeout);
    // 200 = queued and succeeded. A 429 means it never entered the queue.
    let resp2 = resp2.await.unwrap();
    assert_ne!(
        resp2.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "req2 must enter the queue, not 429 at the req/s gate"
    );
    drain(resp2).await;
    ctx.shutdown().await;
}
