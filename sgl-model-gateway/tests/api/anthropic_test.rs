//! Integration tests for the Anthropic Messages API endpoint (`POST /v1/messages`).
//!
//! The gateway forwards Anthropic-format requests to the backend worker's
//! `/v1/messages` endpoint via the HTTP router.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{PolicyConfig, RouterConfig};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType as MockWorkerType},
    AppTestContext,
};

/// Build an [`AppTestContext`] whose router is in regular HTTP mode so
/// that requests are routed through the HTTP router.
async fn http_mode_ctx(worker_configs: Vec<MockWorkerConfig>) -> AppTestContext {
    let config = RouterConfig::builder()
        .regular_mode(vec![])
        .policy(PolicyConfig::Random)
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked();

    AppTestContext::new_with_config(config, worker_configs).await
}

fn make_worker_cfg() -> MockWorkerConfig {
    MockWorkerConfig {
        port: 0,
        worker_type: MockWorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }
}

async fn assert_status_ok(resp: axum::response::Response) -> axum::response::Response {
    if resp.status() != StatusCode::OK {
        let status = resp.status();
        let headers = resp.headers().clone();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap_or_default();
        let body_str = String::from_utf8_lossy(&body);
        panic!(
            "unexpected status {} headers={:?} body={}",
            status, headers, body_str
        );
    }
    resp
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Non-streaming tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_anthropic_messages_basic() {
        let ctx = http_mode_ctx(vec![make_worker_cfg()]).await;
        let app = ctx.create_app().await;

        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello!"}]
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let resp = assert_status_ok(resp).await;

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        // Response must conform to the Anthropic Messages API schema.
        assert_eq!(json["type"], "message");
        assert_eq!(json["role"], "assistant");
        assert!(json["id"].as_str().unwrap().starts_with("msg_"));
        assert!(json["content"].is_array());
        assert!(!json["content"].as_array().unwrap().is_empty());
        assert_eq!(json["content"][0]["type"], "text");
        assert!(json["stop_reason"].as_str().is_some());
        assert!(json["usage"]["input_tokens"].is_number());
        assert!(json["usage"]["output_tokens"].is_number());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_anthropic_messages_with_system_prompt() {
        let ctx = http_mode_ctx(vec![make_worker_cfg()]).await;
        let app = ctx.create_app().await;

        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 512,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "What is 2+2?"}]
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let resp = assert_status_ok(resp).await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["type"], "message");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_anthropic_messages_multi_turn() {
        let ctx = http_mode_ctx(vec![make_worker_cfg()]).await;
        let app = ctx.create_app().await;

        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [
                {"role": "user",      "content": "Hello!"},
                {"role": "assistant", "content": "Hi there! How can I help?"},
                {"role": "user",      "content": "Tell me a joke."}
            ]
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let _resp = assert_status_ok(resp).await;
        ctx.shutdown().await;
    }

    // -------------------------------------------------------------------------
    // Streaming tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_anthropic_messages_streaming() {
        let ctx = http_mode_ctx(vec![make_worker_cfg()]).await;
        let app = ctx.create_app().await;

        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "stream": true,
            "messages": [{"role": "user", "content": "Count to 3."}]
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let resp = assert_status_ok(resp).await;

        // Streaming response must use text/event-stream content type.
        let ct = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.contains("text/event-stream"),
            "expected text/event-stream, got: {}",
            ct
        );

        // Collect and verify SSE events.
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let raw = String::from_utf8_lossy(&bytes);

        // Each data line must carry valid JSON.
        let data_lines: Vec<&str> = raw
            .lines()
            .filter(|l| l.starts_with("data:"))
            .collect();
        assert!(
            !data_lines.is_empty(),
            "expected SSE data lines, got:\n{}",
            raw
        );
        for line in &data_lines {
            let data = line.trim_start_matches("data:").trim();
            let parsed: Result<serde_json::Value, _> = serde_json::from_str(data);
            assert!(parsed.is_ok(), "invalid JSON in SSE data line: {}", line);
        }

        // The stream must contain the mandatory Anthropic lifecycle events.
        assert!(raw.contains("message_start"), "missing message_start");
        assert!(raw.contains("message_stop"),  "missing message_stop");

        ctx.shutdown().await;
    }

    // -------------------------------------------------------------------------
    // Error-handling tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_anthropic_messages_invalid_json() {
        let ctx = http_mode_ctx(vec![make_worker_cfg()]).await;
        let app = ctx.create_app().await;

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from("not valid json"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "invalid_request_error");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_anthropic_messages_missing_model_field() {
        let ctx = http_mode_ctx(vec![make_worker_cfg()]).await;
        let app = ctx.create_app().await;

        // `model` is required for worker selection; omitting it must return 400.
        let body = json!({
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}]
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        ctx.shutdown().await;
    }

    // -------------------------------------------------------------------------
    // count_tokens tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_anthropic_count_tokens_basic() {
        let ctx = http_mode_ctx(vec![make_worker_cfg()]).await;
        let app = ctx.create_app().await;

        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello!"}]
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages/count_tokens")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let resp = assert_status_ok(resp).await;

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

        assert!(
            json["input_tokens"].is_number(),
            "expected input_tokens to be a number, got: {}",
            json
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_anthropic_count_tokens_with_system_and_tools() {
        let ctx = http_mode_ctx(vec![make_worker_cfg()]).await;
        let app = ctx.create_app().await;

        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "What time is it?"}],
            "tools": [{
                "name": "get_time",
                "description": "Get current time",
                "input_schema": {"type": "object", "properties": {}}
            }]
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages/count_tokens")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let resp = assert_status_ok(resp).await;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["input_tokens"].is_number());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_anthropic_count_tokens_invalid_json() {
        let ctx = http_mode_ctx(vec![make_worker_cfg()]).await;
        let app = ctx.create_app().await;

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages/count_tokens")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from("not valid json"))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "invalid_request_error");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_anthropic_count_tokens_no_workers() {
        let ctx = http_mode_ctx(vec![]).await;
        let app = ctx.create_app().await;

        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}]
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages/count_tokens")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert!(
            resp.status().is_client_error() || resp.status().is_server_error(),
            "expected 4xx/5xx without workers, got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_anthropic_messages_no_workers() {
        // No workers → the router returns a 4xx/5xx.
        let ctx = http_mode_ctx(vec![]).await;
        let app = ctx.create_app().await;

        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}]
        });

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert!(
            resp.status().is_client_error() || resp.status().is_server_error(),
            "expected 4xx/5xx without workers, got {}",
            resp.status()
        );
        ctx.shutdown().await;
    }
}
