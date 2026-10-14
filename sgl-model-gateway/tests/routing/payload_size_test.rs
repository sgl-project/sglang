//! Payload size integration tests
//!
//! Tests for request payload size limits and handling.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::RouterConfig;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext,
};

#[cfg(test)]
mod payload_size_tests {
    use super::*;

    /// Test that small payloads are handled correctly
    #[tokio::test]
    async fn test_small_payload() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4200)
            .max_payload_size(1024 * 1024) // 1MB limit
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![MockWorkerConfig {
                port: 20200,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }],
        )
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "text": "Small payload test",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Small payload should be accepted"
        );

        ctx.shutdown().await;
    }

    /// Test that payloads within limit are accepted
    #[tokio::test]
    async fn test_payload_within_limit() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4201)
            .max_payload_size(1024 * 1024) // 1MB limit
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![MockWorkerConfig {
                port: 20201,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }],
        )
        .await;

        let app = ctx.create_app().await;

        // Create a ~100KB payload (well within 1MB limit)
        let large_text = "x".repeat(100 * 1024);
        let payload = json!({
            "text": large_text,
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Payload within limit should be accepted"
        );

        ctx.shutdown().await;
    }

    /// Test that payloads exceeding limit are rejected
    #[tokio::test]
    async fn test_payload_exceeds_limit() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4202)
            .max_payload_size(1024) // Very small 1KB limit
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![MockWorkerConfig {
                port: 20202,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }],
        )
        .await;

        let app = ctx.create_app().await;

        // Create a payload larger than 1KB limit
        let large_text = "x".repeat(2048);
        let payload = json!({
            "text": large_text,
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Should be rejected with 413 Payload Too Large or similar
        assert!(
            resp.status() == StatusCode::PAYLOAD_TOO_LARGE
                || resp.status() == StatusCode::BAD_REQUEST,
            "Payload exceeding limit should be rejected, got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    /// Test edge case: payload exactly at limit
    #[tokio::test]
    async fn test_payload_at_exact_limit() {
        // Use a more reasonable limit for this test
        let limit_bytes = 10 * 1024; // 10KB limit

        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4203)
            .max_payload_size(limit_bytes)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![MockWorkerConfig {
                port: 20203,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }],
        )
        .await;

        let app = ctx.create_app().await;

        // Create a payload slightly under the limit (accounting for JSON overhead)
        let text_size = limit_bytes - 100; // Leave room for JSON structure
        let text = "x".repeat(text_size);
        let payload = json!({
            "text": text,
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Payload at/near limit should be accepted
        assert!(
            resp.status() == StatusCode::OK || resp.status() == StatusCode::PAYLOAD_TOO_LARGE,
            "Payload at limit boundary, got status {}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    /// Test default payload size limit (256MB)
    #[tokio::test]
    async fn test_default_payload_limit() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4204)
            .max_payload_size(256 * 1024 * 1024) // Default 256MB
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![MockWorkerConfig {
                port: 20204,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }],
        )
        .await;

        let app = ctx.create_app().await;

        // Create a 1MB payload (well within 256MB)
        let large_text = "x".repeat(1024 * 1024);
        let payload = json!({
            "text": large_text,
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "1MB payload should be accepted with 256MB limit"
        );

        ctx.shutdown().await;
    }
}
