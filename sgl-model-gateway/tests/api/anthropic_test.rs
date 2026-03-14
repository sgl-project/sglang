//! Integration tests for the Anthropic Messages API endpoint (`POST /v1/messages`).
//!
//! The gateway proxies Anthropic-format requests directly to the backend worker's
//! `/v1/messages` endpoint without any protocol conversion, because the SGLang
//! inference engine natively supports the Anthropic Messages API.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{PolicyConfig, RouterConfig, RoutingMode};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext,
};

/// Build an [`AppTestContext`] whose router is in OpenAI (HTTP-proxy) mode so
/// that requests are routed through [`OpenAIRouter`], which implements
/// `route_anthropic_messages`.
async fn openai_mode_ctx(worker_configs: Vec<MockWorkerConfig>) -> AppTestContext {
    // Start mock workers first to obtain their URLs.
    let mut workers = Vec::new();
    let mut worker_urls = Vec::new();
    for cfg in worker_configs {
        let mut w = crate::common::mock_worker::MockWorker::new(cfg);
        let url = w.start().await.expect("mock worker failed to start");
        worker_urls.push(url);
        workers.push(w);
    }
    if !workers.is_empty() {
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }

    let config = RouterConfig::builder()
        .openai_mode(worker_urls)
        .with_policy(PolicyConfig::Random)
        .host("127.0.0.1")
        .port(0)
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(600)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked();

    AppTestContext::new_with_config(config, vec![]).await
}

fn make_worker_cfg() -> MockWorkerConfig {
    MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Non-streaming tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_anthropic_messages_basic() {
        let ctx = openai_mode_ctx(vec![make_worker_cfg()]).await;
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

        assert_eq!(resp.status(), StatusCode::OK);

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
        let ctx = openai_mode_ctx(vec![make_worker_cfg()]).await;
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

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["type"], "message");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_anthropic_messages_multi_turn() {
        let ctx = openai_mode_ctx(vec![make_worker_cfg()]).await;
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

        assert_eq!(resp.status(), StatusCode::OK);
        ctx.shutdown().await;
    }

    // -------------------------------------------------------------------------
    // Streaming tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_anthropic_messages_streaming() {
        let ctx = openai_mode_ctx(vec![make_worker_cfg()]).await;
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

        assert_eq!(resp.status(), StatusCode::OK);

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
        let ctx = openai_mode_ctx(vec![make_worker_cfg()]).await;
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
        let ctx = openai_mode_ctx(vec![make_worker_cfg()]).await;
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
        let ctx = openai_mode_ctx(vec![make_worker_cfg()]).await;
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

        assert_eq!(resp.status(), StatusCode::OK);

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
        let ctx = openai_mode_ctx(vec![make_worker_cfg()]).await;
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

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["input_tokens"].is_number());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_anthropic_count_tokens_invalid_json() {
        let ctx = openai_mode_ctx(vec![make_worker_cfg()]).await;
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
        let ctx = openai_mode_ctx(vec![]).await;
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
        let ctx = openai_mode_ctx(vec![]).await;
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
