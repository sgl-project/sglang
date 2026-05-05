//! Upstream request cancellation tests
//!
//! Verifies that when a client disconnects mid-stream, the gateway
//! terminates the upstream request to the backend worker promptly
//! (via the `tokio::select!` / `tx.closed()` mechanism in the router).

use std::sync::Arc;
use std::time::Duration;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::json;
use smg::config::RouterConfig;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{
        clear_slow_stream_chunks, clear_stream_error_after_chunks, get_stream_tracking_state,
        reset_stream_tracker, set_slow_stream_chunks, set_stream_error_after_chunks,
        wait_for_chunks_plateau, StreamTrackingState,
    },
    AppTestContext, TestRouterConfig, TestWorkerConfig,
};

/// Read up to `max_chunks` data frames from a streaming response body.
async fn read_n_chunks(body: &mut Body, max_chunks: usize) -> usize {
    let mut chunks_read = 0;
    while chunks_read < max_chunks {
        match body.frame().await {
            Some(Ok(frame)) if frame.is_data() => {
                chunks_read += 1;
            }
            Some(Ok(_)) => continue,
            _ => break,
        }
    }
    chunks_read
}

/// Default plateau parameters: stop polling once chunks_sent has been
/// stable for 250ms, with a 3s overall timeout. Both bounds are generous
/// enough to absorb CI noise while still being faster than any natural
/// stream completion in these tests.
const PLATEAU_QUIET_WINDOW: Duration = Duration::from_millis(250);
const PLATEAU_TIMEOUT: Duration = Duration::from_secs(3);

async fn assert_cancelled_before_completion(port: u16) -> StreamTrackingState {
    let state = wait_for_chunks_plateau(port, PLATEAU_QUIET_WINDOW, PLATEAU_TIMEOUT)
        .await
        .unwrap_or_else(|| {
            panic!(
                "Stream tracking state should exist for worker port {}",
                port
            )
        });

    assert!(
        !state.completed,
        "Stream should NOT have completed - gateway should have cancelled it. \
         Chunks sent: {}, total: {}",
        state.chunks_sent, state.total_chunks
    );
    assert!(
        state.chunks_sent < state.total_chunks,
        "Worker should have sent fewer chunks than total ({} < {}). \
         Stream was not cancelled in time.",
        state.chunks_sent,
        state.total_chunks
    );
    state
}

#[cfg(test)]
mod upstream_cancel_tests {
    use super::*;

    /// Test that the gateway cancels the upstream stream when the client
    /// disconnects before consuming all chunks.
    ///
    /// Setup:
    ///   - Mock worker sends 20 chunks with 50ms delay between each (~1s total).
    ///   - Client reads a few chunks then drops the response body.
    ///
    /// Expectation:
    ///   - The mock worker stops producing once the gateway closes its
    ///     upstream connection. We assert that by polling for a plateau
    ///     in `chunks_sent` and snapshotting before/after — proving the
    ///     worker actually halted instead of just being slower than our
    ///     fixed sleep.
    #[tokio::test]
    async fn test_streaming_cancel_on_client_disconnect() {
        let worker_port = 20250;
        let total_chunks: usize = 20;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = TestRouterConfig::round_robin(4250);
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 50)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Tell me a long story"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        let chunks_read = read_n_chunks(&mut body, 3).await;
        assert!(
            chunks_read > 0,
            "Should have read at least one chunk before disconnecting"
        );

        // Snapshot the worker counter the moment we drop, then poll for
        // plateau. If cancel propagation is broken the counter will keep
        // climbing past total_chunks within PLATEAU_TIMEOUT.
        let snapshot = get_stream_tracking_state(worker_port)
            .map(|s| s.chunks_sent)
            .unwrap_or(0);
        drop(body);

        let final_state = assert_cancelled_before_completion(worker_port).await;
        assert!(
            final_state.chunks_sent <= snapshot + 4,
            "Chunks_sent grew by more than the channel buffer ({} -> {}); \
             gateway likely did not propagate cancel.",
            snapshot,
            final_state.chunks_sent
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// Test that a fully consumed stream is NOT cancelled prematurely —
    /// the worker sends all chunks and completes normally.
    #[tokio::test]
    async fn test_streaming_completes_when_client_consumes_all() {
        let worker_port = 20251;
        let total_chunks: usize = 5;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = TestRouterConfig::round_robin(4251);
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 10)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Short response"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let _full_body = resp.into_body().collect().await.unwrap().to_bytes();

        let state = wait_for_chunks_plateau(worker_port, PLATEAU_QUIET_WINDOW, PLATEAU_TIMEOUT)
            .await
            .expect("Stream tracking state should exist");
        assert!(
            state.completed,
            "Stream should have completed when client consumed all chunks. \
             Chunks sent: {}, total: {}",
            state.chunks_sent, state.total_chunks
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// Test that a non-streaming request is not affected by cancel logic.
    #[tokio::test]
    async fn test_non_streaming_request_unaffected() {
        let config = TestRouterConfig::round_robin(4252);
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20252)]).await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(
            body.get("object").and_then(|v| v.as_str()),
            Some("chat.completion"),
            "Non-streaming response should be a complete chat.completion object"
        );

        ctx.shutdown().await;
    }

    /// Cancel before the worker emits any chunk. Catches a select! that
    /// only wakes `tx.closed()` after the first `stream.next()` resolves.
    #[tokio::test]
    async fn test_streaming_cancel_before_first_chunk() {
        let worker_port = 20253;
        let total_chunks: usize = 10;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = TestRouterConfig::round_robin(4253);
        let ctx = AppTestContext::new_with_config(
            // 200ms per-chunk delay; the very first chunk takes the
            // full 200ms because the worker sleeps before emitting.
            config,
            vec![TestWorkerConfig::slow(worker_port, 200)],
        )
        .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Drop immediately, before pulling any frame.
        drop(resp.into_body());

        let state = assert_cancelled_before_completion(worker_port).await;
        assert!(
            state.chunks_sent <= 4,
            "Worker should have sent very few chunks (≤ buffer capacity), \
             saw {} of {}",
            state.chunks_sent,
            state.total_chunks
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// Disconnecting *after* the stream completes naturally must be a
    /// no-op — no panic, no spurious "cancel" log, completed=true stays.
    #[tokio::test]
    async fn test_streaming_cancel_after_done_is_noop() {
        let worker_port = 20254;
        let total_chunks: usize = 3;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = TestRouterConfig::round_robin(4254);
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 5)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain entire body, then drop after a short pause.
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        assert!(!body.is_empty());
        tokio::time::sleep(Duration::from_millis(50)).await;
        drop(body);

        let state = wait_for_chunks_plateau(worker_port, PLATEAU_QUIET_WINDOW, PLATEAU_TIMEOUT)
            .await
            .expect("tracking state");
        assert!(state.completed, "Stream should have completed cleanly");
        assert_eq!(state.chunks_sent, state.total_chunks);

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// Cancelling one client request must not affect a concurrent request
    /// that's hitting a *different* upstream replica.
    #[tokio::test]
    async fn test_cancel_one_request_does_not_affect_concurrent() {
        let worker_a = 20255;
        let worker_b = 20256;
        let total_a: usize = 20;
        let total_b: usize = 5;

        reset_stream_tracker(worker_a);
        reset_stream_tracker(worker_b);
        set_slow_stream_chunks(worker_a, total_a);
        set_slow_stream_chunks(worker_b, total_b);

        let config = TestRouterConfig::round_robin(4255);
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::slow(worker_a, 50),
                TestWorkerConfig::slow(worker_b, 10),
            ],
        )
        .await;

        // Two parallel requests. Round-robin should send them to different
        // workers. We don't strictly need to know which got which, but we
        // assume the FIRST request lands on worker_a — that's the one we
        // cancel — and we await the SECOND to completion.
        let app = ctx.create_app().await;
        let app2 = app.clone();

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Long"}],
            "stream": true
        });
        let body_str = Arc::new(serde_json::to_string(&payload).unwrap());

        let body_str_a = Arc::clone(&body_str);
        let h_cancel = tokio::spawn(async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from((*body_str_a).clone()))
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            let mut body = resp.into_body();
            let _ = read_n_chunks(&mut body, 2).await;
            drop(body);
        });

        // Small stagger so round-robin index advances.
        tokio::time::sleep(Duration::from_millis(20)).await;

        let body_str_b = Arc::clone(&body_str);
        let h_consume = tokio::spawn(async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from((*body_str_b).clone()))
                .unwrap();
            let resp = app2.oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            let _ = resp.into_body().collect().await.unwrap();
        });

        h_cancel.await.unwrap();
        h_consume.await.unwrap();

        // Worker that should have completed normally.
        let state_b = wait_for_chunks_plateau(worker_b, PLATEAU_QUIET_WINDOW, PLATEAU_TIMEOUT)
            .await
            .expect("worker_b tracking state");
        // Worker that should have been cancelled.
        let state_a = wait_for_chunks_plateau(worker_a, PLATEAU_QUIET_WINDOW, PLATEAU_TIMEOUT)
            .await
            .expect("worker_a tracking state");

        // We don't know which worker got which request because round-robin
        // is shared across the run, so accept either ordering.
        let (cancelled, completed) = if state_a.completed {
            (state_b, state_a)
        } else {
            (state_a, state_b)
        };
        assert!(
            !cancelled.completed,
            "Cancelled stream should not have completed (sent {}/{})",
            cancelled.chunks_sent, cancelled.total_chunks
        );
        assert!(
            completed.completed,
            "Concurrent stream should have completed (sent {}/{})",
            completed.chunks_sent, completed.total_chunks
        );

        clear_slow_stream_chunks(worker_a);
        clear_slow_stream_chunks(worker_b);
        ctx.shutdown().await;
    }

    /// Mid-stream worker error must surface as `Stream error: …` to the
    /// client, must NOT be silently swallowed as cancel, and must trigger
    /// the gateway's error log path. We assert (a) the client sees the
    /// error frame and (b) `chunks_sent` reflects the partial output the
    /// worker sent before erroring.
    #[tokio::test]
    async fn test_streaming_worker_error_propagates_not_cancel() {
        let worker_port = 20257;
        let total_chunks: usize = 10;
        let error_after: usize = 3;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);
        set_stream_error_after_chunks(worker_port, error_after);

        let config = TestRouterConfig::round_robin(4256);
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 10)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Collecting the body should yield a transport-level error AFTER
        // the first few chunks. axum surfaces the upstream error as a
        // failed body.collect(), so we just stitch frames manually.
        let mut body = resp.into_body();
        let mut combined = Vec::<u8>::new();
        let mut saw_transport_err = false;
        loop {
            match body.frame().await {
                Some(Ok(frame)) => {
                    if let Ok(data) = frame.into_data() {
                        combined.extend_from_slice(&data);
                    }
                }
                Some(Err(_)) => {
                    saw_transport_err = true;
                    break;
                }
                None => break,
            }
        }
        let combined_str = String::from_utf8_lossy(&combined);
        assert!(
            saw_transport_err
                || combined_str.contains("Stream error")
                || combined_str.contains("simulated upstream worker crash"),
            "Client should observe a stream error event or transport error; got: {}",
            combined_str
        );

        // Worker should have sent some chunks but not all, and the stream
        // should NOT be marked completed (we crashed before [DONE]).
        let state = wait_for_chunks_plateau(worker_port, PLATEAU_QUIET_WINDOW, PLATEAU_TIMEOUT)
            .await
            .expect("tracking state");
        assert!(!state.completed, "Errored stream should not be completed");
        assert!(
            state.chunks_sent <= error_after,
            "Worker reported {} chunks_sent, expected ≤ {}",
            state.chunks_sent,
            error_after
        );

        clear_stream_error_after_chunks(worker_port);
        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// PD-disagg streaming cancel: client disconnects mid-decode-stream.
    /// The decode worker's slow-stream tracker is what proves cancel
    /// actually reached the upstream — prefill is fully drained
    /// synchronously by `process_prefill_response`, so it's expected
    /// to complete regardless.
    #[tokio::test]
    async fn test_pd_streaming_cancel_on_client_disconnect() {
        let prefill_port = 20258;
        let decode_port = 20259;
        let total_chunks: usize = 20;

        reset_stream_tracker(decode_port);
        set_slow_stream_chunks(decode_port, total_chunks);

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![(format!("http://127.0.0.1:{}", prefill_port), None)],
                vec![format!("http://127.0.0.1:{}", decode_port)],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4257)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::prefill(prefill_port), {
                // Use a slow decode worker (50ms per chunk).
                let mut w = TestWorkerConfig::decode(decode_port);
                w.response_delay_ms = 50;
                w
            }],
        )
        .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "text": "PD streaming test",
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        let chunks_read = read_n_chunks(&mut body, 3).await;
        assert!(
            chunks_read > 0,
            "Should have read at least one chunk before disconnecting"
        );

        let snapshot = get_stream_tracking_state(decode_port)
            .map(|s| s.chunks_sent)
            .unwrap_or(0);
        drop(body);

        let final_state = assert_cancelled_before_completion(decode_port).await;
        assert!(
            final_state.chunks_sent <= snapshot + 4,
            "Decode chunks_sent grew by more than the buffer ({} -> {}); \
             gateway likely did not propagate cancel through PD path.",
            snapshot,
            final_state.chunks_sent
        );

        clear_slow_stream_chunks(decode_port);
        ctx.shutdown().await;
    }

    /// /v1/responses with no persistence (`store: false`, no conversation):
    /// client disconnect must propagate to the upstream worker.
    #[tokio::test]
    async fn test_responses_streaming_cancel_no_persistence() {
        let worker_port = 20260;
        let total_chunks: usize = 20;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4258)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 50)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "input": "Tell me a story",
            "stream": true,
            "store": false
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        // First two events are response.created/response.in_progress, then
        // chunks start. Read a few to ensure we're past the bootstrap.
        let chunks_read = read_n_chunks(&mut body, 3).await;
        assert!(chunks_read > 0);

        let snapshot = get_stream_tracking_state(worker_port)
            .map(|s| s.chunks_sent)
            .unwrap_or(0);
        drop(body);

        let final_state = assert_cancelled_before_completion(worker_port).await;
        assert!(
            final_state.chunks_sent <= snapshot + 4,
            "/responses chunks_sent grew by more than the buffer ({} -> {}); \
             gateway likely did not propagate cancel through /responses path.",
            snapshot,
            final_state.chunks_sent
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// /v1/responses with persistence (`store: true`): the upstream is
    /// intentionally NOT cancelled on client disconnect — the gateway
    /// keeps consuming so the response can be persisted. We assert the
    /// worker reaches `completed = true` after the client disconnects.
    #[tokio::test]
    async fn test_responses_streaming_persistence_drains_after_disconnect() {
        let worker_port = 20261;
        let total_chunks: usize = 6;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4259)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 20)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "input": "Tell me a story",
            "stream": true,
            "store": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        let _ = read_n_chunks(&mut body, 2).await;
        drop(body);

        // With persistence the gateway keeps reading; worker should
        // eventually mark the stream completed despite client gone.
        let state = wait_for_chunks_plateau(
            worker_port,
            Duration::from_millis(500),
            Duration::from_secs(5),
        )
        .await
        .expect("tracking state");
        assert!(
            state.completed,
            "Persistence path should drain upstream to completion despite \
             client disconnect (sent {}/{})",
            state.chunks_sent, state.total_chunks
        );
        assert_eq!(state.chunks_sent, state.total_chunks);

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// OpenAI-mode (non-/responses) chat-completions cancel: this exercises
    /// the `OpenAIRouter` impl, which is a separate codepath from
    /// `http::Router`, so the same cancel semantics need their own test.
    #[tokio::test]
    async fn test_openai_router_streaming_cancel() {
        let worker_port = 20262;
        let total_chunks: usize = 20;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = RouterConfig::builder()
            .openai_mode(vec![format!("http://127.0.0.1:{}", worker_port)])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4260)
            .max_payload_size(8 * 1024 * 1024)
            .request_timeout_secs(60)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(32)
            .queue_timeout_secs(5)
            .build_unchecked();

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(worker_port, 50)])
                .await;
        let app = ctx.create_app().await;

        let payload = json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.into_body();
        let chunks_read = read_n_chunks(&mut body, 3).await;
        assert!(chunks_read > 0);

        let snapshot = get_stream_tracking_state(worker_port)
            .map(|s| s.chunks_sent)
            .unwrap_or(0);
        drop(body);

        let final_state = assert_cancelled_before_completion(worker_port).await;
        assert!(
            final_state.chunks_sent <= snapshot + 4,
            "OpenAI-mode chunks_sent grew by more than the buffer ({} -> {})",
            snapshot,
            final_state.chunks_sent
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }
}
