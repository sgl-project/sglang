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
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(9300, 50)]).await;

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
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(9301)]).await;

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
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::slow(9302, 10)]).await;

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
            vec![TestWorkerConfig::slow(9303, 200)], // Longer delay to test queuing
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

    /// Test that concurrency is strictly enforced when rate_limit_tokens_per_second is not set.
    ///
    /// With the fix, refill_rate=0 (pure concurrency mode) means tokens are only returned
    /// when responses complete. Without queuing enabled in the test framework, excess
    /// requests are immediately rejected with 429, proving that the token bucket does NOT
    /// auto-refill tokens over time.
    ///
    /// Before the fix, refill_rate defaulted to max_concurrent_requests, so tokens would
    /// refill over time and more requests would slip through.
    #[tokio::test]
    async fn test_concurrent_requests_actually_limited() {
        // Use with_concurrency which does NOT set rate_limit_tokens_per_second.
        // After the fix, this means refill_rate=0 (pure concurrency / semaphore mode).
        let config = TestRouterConfig::with_concurrency(3404, 2); // max 2 concurrent

        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::slow(9304, 300)], // 300ms delay per request
        )
        .await;

        let app = ctx.create_app().await;

        let mut handles = Vec::new();
        let success_count = Arc::new(AtomicUsize::new(0));
        let rejected_count = Arc::new(AtomicUsize::new(0));

        // Send 6 concurrent requests (3x the limit of 2)
        for i in 0..6 {
            let app_clone = app.clone();
            let success_clone = Arc::clone(&success_count);
            let rejected_clone = Arc::clone(&rejected_count);

            let handle = tokio::spawn(async move {
                let payload = json!({
                    "text": format!("Concurrency test {}", i),
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
                } else if resp.status() == StatusCode::TOO_MANY_REQUESTS {
                    rejected_clone.fetch_add(1, Ordering::SeqCst);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.await.unwrap();
        }

        let successes = success_count.load(Ordering::SeqCst);
        let rejected = rejected_count.load(Ordering::SeqCst);

        // With pure concurrency mode (refill_rate=0) and no queue in test framework:
        // - Exactly 2 requests should succeed (acquiring the 2 available tokens)
        // - The remaining 4 should be rejected with 429 (no tokens, no queue)
        assert_eq!(
            successes, 2,
            "Only max_concurrent_requests (2) should succeed, got {} successes and {} rejected",
            successes, rejected
        );
        assert_eq!(
            rejected, 4,
            "Excess requests should be rejected with 429, got {} rejected",
            rejected
        );

        ctx.shutdown().await;
    }
}
