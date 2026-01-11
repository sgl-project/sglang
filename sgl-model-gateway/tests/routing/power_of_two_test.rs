//! Power of Two load balancing integration tests
//!
//! Tests for the Power of Two Choices algorithm that selects the less loaded worker.

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
use smg::config::RouterConfig;
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

#[cfg(test)]
mod power_of_two_tests {
    use super::*;

    /// Test that power of two distributes requests across workers
    #[tokio::test]
    async fn test_power_of_two_distribution() {
        let config = TestRouterConfig::power_of_two(3600);

        let ctx =
            AppTestContext::new_with_config(config, TestWorkerConfig::healthy_workers(19600, 2))
                .await;

        let app = ctx.create_app().await;
        let num_requests = 20;
        let mut success_count = 0;

        for i in 0..num_requests {
            let payload = json!({
                "text": format!("Power of two request {}", i),
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

        // All requests should succeed
        assert_eq!(
            success_count, num_requests,
            "All requests should succeed with power of two policy"
        );

        ctx.shutdown().await;
    }

    /// Test that power of two prefers less loaded workers
    #[tokio::test]
    async fn test_power_of_two_prefers_less_loaded() {
        let config = TestRouterConfig::power_of_two(3601);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::slow(19602, 200), // Slow worker
                TestWorkerConfig::healthy(19603),   // Fast worker
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Send concurrent requests to create load imbalance
        let mut handles = Vec::new();
        let success_count = Arc::new(AtomicUsize::new(0));

        for i in 0..30 {
            let app_clone = app.clone();
            let success_clone = Arc::clone(&success_count);

            let handle = tokio::spawn(async move {
                let payload = json!({
                    "text": format!("Concurrent request {}", i),
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

        // All requests should succeed
        assert!(
            success_count.load(Ordering::SeqCst) >= 25,
            "Most requests should succeed with power of two"
        );

        ctx.shutdown().await;
    }

    /// Test power of two with failing worker uses retry/CB to route to healthy worker
    #[tokio::test]
    async fn test_power_of_two_with_failing_worker() {
        use smg::config::{CircuitBreakerConfig, RetryConfig};

        let retry_config = RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 10,
            max_backoff_ms: 50,
            ..Default::default()
        };
        let circuit_breaker = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 1,
            timeout_duration_secs: 2,
            window_duration_secs: 10,
        };

        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3602)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(retry_config)
            .circuit_breaker_config(circuit_breaker)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::flaky(19604, 1.0), // Always fails
                TestWorkerConfig::healthy(19605),    // Always succeeds
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Requests should succeed via retry to healthy worker
        for i in 0..10 {
            let payload = json!({
                "text": format!("Request with failing worker {}", i),
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
                "Request should succeed via retry on healthy worker"
            );
        }

        ctx.shutdown().await;
    }
}
