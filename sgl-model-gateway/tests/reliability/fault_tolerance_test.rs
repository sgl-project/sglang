//! Fault tolerance integration tests
//!
//! Tests for system resilience: worker failures, network issues, and recovery.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::config::{CircuitBreakerConfig, RetryConfig};
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

#[cfg(test)]
mod fault_tolerance_tests {
    use super::*;

    /// Test that requests are rerouted when a worker fails
    #[tokio::test]
    async fn test_worker_failure_reroute() {
        let config = TestRouterConfig::round_robin_with_reliability(
            4100,
            RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 100,
                ..Default::default()
            },
            CircuitBreakerConfig {
                failure_threshold: 2,
                success_threshold: 1,
                timeout_duration_secs: 2,
                window_duration_secs: 10,
            },
        );

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(20100, 1.0), // Always fails
                TestWorkerConfig::healthy(20101),    // Always succeeds
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Requests should succeed via retry to healthy worker
        for i in 0..10 {
            let payload = json!({
                "text": format!("Fault tolerance test {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "Request should succeed via reroute to healthy worker"
            );
        }

        ctx.shutdown().await;
    }

    /// Test behavior when all workers are temporarily unavailable
    #[tokio::test]
    async fn test_all_workers_temporarily_failing() {
        let config = TestRouterConfig::round_robin_with_retry(
            4101,
            RetryConfig {
                max_retries: 2,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            },
        );

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(20102, 1.0), // Always fails
                TestWorkerConfig::flaky(20103, 1.0), // Always fails
            ],
        )
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "text": "Test with all failing workers",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Should fail when all workers are failing
        assert!(
            resp.status() == StatusCode::INTERNAL_SERVER_ERROR
                || resp.status() == StatusCode::SERVICE_UNAVAILABLE,
            "Request should fail when all workers are failing, got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    /// Test graceful handling of slow workers
    #[tokio::test]
    async fn test_slow_worker_handling() {
        let config = TestRouterConfig::round_robin(4102);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::slow(20104, 500), // Slow worker
                TestWorkerConfig::healthy(20105),   // Fast worker
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Send concurrent requests
        let mut handles = Vec::new();
        let success_count = Arc::new(AtomicUsize::new(0));

        for i in 0..10 {
            let app_clone = app.clone();
            let success_clone = Arc::clone(&success_count);

            let handle = tokio::spawn(async move {
                let payload = json!({
                    "text": format!("Slow worker test {}", i),
                    "stream": false
                });

                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_string(&payload).unwrap()))
                    .unwrap();

                let resp = app_clone.oneshot(req).await.unwrap();
                if resp.status() == StatusCode::OK {
                    success_clone.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // All requests should eventually succeed
        assert_eq!(
            success_count.load(Ordering::SeqCst),
            10,
            "All requests should succeed despite slow worker"
        );

        ctx.shutdown().await;
    }

    /// Test circuit breaker prevents cascading failures
    #[tokio::test]
    async fn test_circuit_breaker_prevents_cascade() {
        let config = TestRouterConfig::round_robin_with_reliability(
            4103,
            RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            },
            CircuitBreakerConfig {
                failure_threshold: 2,
                success_threshold: 1,
                timeout_duration_secs: 5,
                window_duration_secs: 10,
            },
        );

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(20106, 1.0), // Failing worker
                TestWorkerConfig::healthy(20107),    // Healthy worker
            ],
        )
        .await;

        let app = ctx.create_app().await;
        let mut success_count = 0;

        // Send many requests - after CB opens, all should route to healthy worker
        for i in 0..20 {
            let payload = json!({
                "text": format!("Circuit breaker cascade test {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        // Most requests should succeed after CB opens on failing worker
        assert!(
            success_count >= 15,
            "Most requests should succeed after circuit breaker opens, got {} successes",
            success_count
        );

        ctx.shutdown().await;
    }

    /// Test recovery after worker comes back online (simulated via healthy worker)
    #[tokio::test]
    async fn test_system_stability_under_partial_failure() {
        let config = TestRouterConfig::round_robin_with_reliability(
            4104,
            RetryConfig {
                max_retries: 2,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            },
            CircuitBreakerConfig {
                failure_threshold: 3,
                success_threshold: 1,
                timeout_duration_secs: 2,
                window_duration_secs: 10,
            },
        );

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(20108, 0.5), // 50% failure rate
                TestWorkerConfig::healthy(20109),    // Always succeeds
            ],
        )
        .await;

        let app = ctx.create_app().await;
        let mut success_count = 0;

        // System should maintain stability with partial failures
        for i in 0..30 {
            let payload = json!({
                "text": format!("Stability test {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        // With retries and one healthy worker, most should succeed
        assert!(
            success_count >= 25,
            "System should maintain stability under partial failure, got {} successes",
            success_count
        );

        ctx.shutdown().await;
    }
}
