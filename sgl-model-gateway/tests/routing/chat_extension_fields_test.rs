//! End-to-end regression test: the gateway must preserve top-level extension
//! fields (e.g. `cache_salt`) on chat completions instead of silently
//! dropping them before forwarding to the worker.
//!
//! See https://github.com/sgl-project/sglang/issues/48793

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{take_last_chat_payload, HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext,
};

#[cfg(test)]
mod chat_extension_fields_tests {
    use super::*;

    /// `cache_salt` (and other engine extension fields) sent to
    /// /v1/chat/completions must reach the worker unchanged.
    #[tokio::test]
    async fn test_chat_completions_preserves_cache_salt() {
        let port = 19450;
        let ctx = AppTestContext::new(vec![MockWorkerConfig {
            port,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 1,
            "cache_salt": "tenant-a",
            "extra_key": "k",
            "stream": false,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let forwarded = take_last_chat_payload(port)
            .expect("mock worker should have received the chat request");

        assert_eq!(
            forwarded.get("cache_salt").and_then(|v| v.as_str()),
            Some("tenant-a"),
            "cache_salt must be forwarded to the worker, got: {forwarded}"
        );
        assert_eq!(
            forwarded.get("extra_key").and_then(|v| v.as_str()),
            Some("k"),
            "other engine extension fields must also be preserved, got: {forwarded}"
        );
        assert_eq!(
            forwarded
                .get("max_completion_tokens")
                .and_then(|v| v.as_u64()),
            Some(1),
            "deprecated max_tokens must still migrate to max_completion_tokens, got: {forwarded}"
        );
        assert!(
            !forwarded.as_object().unwrap().contains_key("max_tokens"),
            "deprecated max_tokens must not be re-added after normalize() clears it, got: {forwarded}"
        );

        ctx.shutdown().await;
    }
}
