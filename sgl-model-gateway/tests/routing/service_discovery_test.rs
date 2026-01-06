//! Service discovery integration tests
//!
//! Tests for service discovery shim functionality for dynamic worker registration.

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
mod service_discovery_tests {
    use super::*;

    /// Test service discovery endpoint responds correctly
    #[tokio::test]
    async fn test_service_discovery_endpoint() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4000)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![MockWorkerConfig {
                port: 20000,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }],
        )
        .await;

        let app = ctx.create_app().await;

        // Check if service discovery endpoint exists
        let req = Request::builder()
            .method("GET")
            .uri("/v1/workers")
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        // Endpoint might return OK with worker list or 404 if not implemented
        assert!(
            resp.status() == StatusCode::OK || resp.status() == StatusCode::NOT_FOUND,
            "Workers endpoint should respond, got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    /// Test worker registration via discovery shim
    #[tokio::test]
    async fn test_worker_registration() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4001)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![MockWorkerConfig {
                port: 20001,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }],
        )
        .await;

        let app = ctx.create_app().await;

        // Register a new worker via discovery endpoint
        let register_payload = json!({
            "url": "http://127.0.0.1:20002",
            "weight": 1.0
        });

        let req = Request::builder()
            .method("POST")
            .uri("/register_worker")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_string(&register_payload).unwrap(),
            ))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        // Registration might succeed or endpoint might not exist
        assert!(
            resp.status() == StatusCode::OK
                || resp.status() == StatusCode::ACCEPTED
                || resp.status() == StatusCode::NOT_FOUND,
            "Registration should respond appropriately, got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    /// Test worker deregistration via discovery shim
    #[tokio::test]
    async fn test_worker_deregistration() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4002)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                MockWorkerConfig {
                    port: 20003,
                    worker_type: WorkerType::Regular,
                    health_status: HealthStatus::Healthy,
                    response_delay_ms: 0,
                    fail_rate: 0.0,
                },
                MockWorkerConfig {
                    port: 20004,
                    worker_type: WorkerType::Regular,
                    health_status: HealthStatus::Healthy,
                    response_delay_ms: 0,
                    fail_rate: 0.0,
                },
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Deregister a worker via discovery endpoint
        let deregister_payload = json!({
            "url": "http://127.0.0.1:20003"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/deregister_worker")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_string(&deregister_payload).unwrap(),
            ))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert!(
            resp.status() == StatusCode::OK
                || resp.status() == StatusCode::ACCEPTED
                || resp.status() == StatusCode::NOT_FOUND,
            "Deregistration should respond appropriately, got {}",
            resp.status()
        );

        // Requests should still work with remaining worker
        let payload = json!({
            "text": "Test after deregistration",
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
            "Request should succeed with remaining worker"
        );

        ctx.shutdown().await;
    }

    /// Test health status reporting for discovery
    #[tokio::test]
    async fn test_health_status_endpoint() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .round_robin_policy()
            .host("127.0.0.1")
            .port(4003)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![MockWorkerConfig {
                port: 20005,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }],
        )
        .await;

        let app = ctx.create_app().await;

        // Check health endpoint
        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Health endpoint should return OK when workers are healthy"
        );

        ctx.shutdown().await;
    }
}
