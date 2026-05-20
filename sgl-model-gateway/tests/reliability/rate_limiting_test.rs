//! Rate limiting integration tests
//!
//! Tests for rate limiting and concurrency control.

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
mod rate_limiting_tests {
    use super::*;

    /// Test that concurrent requests are handled within limits
    #[tokio::test]
    async fn test_concurrent_requests_within_limit() {
        let config = TestRouterConfig::with_concurrency(3400, 10);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(19300, 50)]).await;

        let app = ctx.create_app().await;

        // Send 5 concurrent requests (within limit of 10)
        let mut handles = Vec::new();
        let success_count = Arc::new(AtomicUsize::new(0));

        for i in 0..5 {
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

        // Wait for all requests to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // All requests should succeed within the limit
        assert_eq!(
            success_count.load(Ordering::SeqCst),
            5,
            "All concurrent requests should succeed within limit"
        );

        ctx.shutdown().await;
    }

    /// Test rate limit tokens per second
    #[tokio::test]
    async fn test_rate_limit_tokens() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host("127.0.0.1")
            .port(3401)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(100)
            .rate_limit_tokens_per_second(50) // 50 tokens/sec
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(19301)]).await;

        let app = ctx.create_app().await;

        // Send requests and verify they succeed
        let mut success_count = 0;
        for i in 0..10 {
            let payload = json!({
                "text": format!("Rate limited request {}", i),
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

        // Most requests should succeed (some might be rate limited)
        assert!(
            success_count >= 5,
            "At least half of requests should succeed, got {}",
            success_count
        );

        ctx.shutdown().await;
    }

    /// Test unlimited concurrent requests when set to 0
    #[tokio::test]
    async fn test_unlimited_concurrent_requests() {
        let config = TestRouterConfig::with_concurrency(3402, 0); // Unlimited

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(19302, 10)]).await;

        let app = ctx.create_app().await;

        // Send many concurrent requests
        let mut handles = Vec::new();
        let success_count = Arc::new(AtomicUsize::new(0));

        for i in 0..20 {
            let app_clone = app.clone();
            let success_clone = Arc::clone(&success_count);

            let handle = tokio::spawn(async move {
                let payload = json!({
                    "text": format!("Unlimited request {}", i),
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

        // All requests should succeed with unlimited concurrency
        assert_eq!(
            success_count.load(Ordering::SeqCst),
            20,
            "All requests should succeed with unlimited concurrency"
        );

        ctx.shutdown().await;
    }

    /// Test queue behavior when requests exceed capacity
    #[tokio::test]
    async fn test_queue_behavior() {
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host("127.0.0.1")
            .port(3403)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(2) // Very low limit
            .queue_size(5) // Small queue
            .queue_timeout_secs(1) // Short timeout
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::slow(19303, 200)], // Longer delay to test queuing
        )
        .await;

        let app = ctx.create_app().await;

        // Send requests that will exceed capacity
        let mut handles = Vec::new();
        let success_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));

        for i in 0..10 {
            let app_clone = app.clone();
            let success_clone = Arc::clone(&success_count);
            let error_clone = Arc::clone(&error_count);

            let handle = tokio::spawn(async move {
                let payload = json!({
                    "text": format!("Queue test {}", i),
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
                } else {
                    error_clone.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // Some requests should succeed, some might timeout or be rejected
        let successes = success_count.load(Ordering::SeqCst);
        let errors = error_count.load(Ordering::SeqCst);

        assert!(
            successes > 0,
            "At least some requests should succeed, got {} successes and {} errors",
            successes,
            errors
        );

        ctx.shutdown().await;
    }
}
