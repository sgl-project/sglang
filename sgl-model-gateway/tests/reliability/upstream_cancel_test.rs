//! Upstream request cancellation tests
//!
//! Verifies that when a client disconnects mid-stream, the gateway
//! terminates the upstream request to the backend worker promptly
//! (via the `tokio::select!` / `tx.closed()` mechanism in the router).

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{
        clear_slow_stream_chunks, get_stream_tracking_state, reset_stream_tracker,
        set_slow_stream_chunks,
    },
    AppTestContext, TestRouterConfig, TestWorkerConfig,
};

#[cfg(test)]
mod upstream_cancel_tests {
    use super::*;

    /// Test that the gateway cancels the upstream stream when the client
    /// disconnects before consuming all chunks.
    ///
    /// Setup:
    ///   - Mock worker sends 20 chunks with 50ms delay between each (total ~1s)
    ///   - Client reads a few chunks then drops the response
    ///
    /// Expectation:
    ///   - The mock worker should NOT have sent all 20 chunks
    ///   - The stream should NOT be marked as completed
    #[tokio::test]
    async fn test_streaming_cancel_on_client_disconnect() {
        let worker_port = 20250;
        let total_chunks: usize = 20;

        // Configure the mock worker (by port) to send slow chunks,
        // and reset tracking state before the test.
        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = TestRouterConfig::round_robin(4250);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::slow(worker_port, 50)], // 50ms per chunk
        )
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

        // Read a few chunks from the streaming response then drop it
        let mut body = resp.into_body();
        let mut chunks_read = 0;
        let max_chunks_to_read = 3;

        while chunks_read < max_chunks_to_read {
            match body.frame().await {
                Some(Ok(frame)) => {
                    if frame.is_data() {
                        chunks_read += 1;
                    }
                }
                _ => break,
            }
        }

        assert!(
            chunks_read > 0,
            "Should have read at least one chunk before disconnecting"
        );

        // Drop the body to simulate client disconnect
        drop(body);

        // Give the gateway time to propagate the cancellation to the upstream
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Verify the mock worker did NOT send all chunks
        let state = get_stream_tracking_state(worker_port);
        assert!(
            state.is_some(),
            "Stream tracking state should exist for worker"
        );

        let state = state.unwrap();
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

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// Test that a fully consumed stream is NOT cancelled prematurely -
    /// the worker sends all chunks and completes normally.
    #[tokio::test]
    async fn test_streaming_completes_when_client_consumes_all() {
        let worker_port = 20251;
        let total_chunks: usize = 5;

        reset_stream_tracker(worker_port);
        set_slow_stream_chunks(worker_port, total_chunks);

        let config = TestRouterConfig::round_robin(4251);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::slow(worker_port, 10)], // 10ms per chunk
        )
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

        // Consume the entire body
        let _full_body = resp.into_body().collect().await.unwrap().to_bytes();

        // Give a moment for the worker to finalize
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        let state = get_stream_tracking_state(worker_port);
        assert!(state.is_some(), "Stream tracking state should exist");

        let state = state.unwrap();
        assert!(
            state.completed,
            "Stream should have completed when client consumed all chunks. \
             Chunks sent: {}, total: {}",
            state.chunks_sent, state.total_chunks
        );
        assert_eq!(
            state.chunks_sent, state.total_chunks,
            "All chunks should have been sent"
        );

        clear_slow_stream_chunks(worker_port);
        ctx.shutdown().await;
    }

    /// Test that a non-streaming request is not affected by cancel logic.
    #[tokio::test]
    async fn test_non_streaming_request_unaffected() {
        let config = TestRouterConfig::round_robin(4252);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::healthy(20252)],
        )
        .await;

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
}
