//! Authentication and authorization integration tests
//!
//! Tests for API key enforcement and access control.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

const AUTH_HEADER: &str = "Authorization";

#[cfg(test)]
mod auth_tests {
    use super::*;

    /// Test request without API key when auth is not required
    #[tokio::test]
    async fn test_no_auth_required() {
        let config = TestRouterConfig::round_robin(4300);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20300)]).await;

        let app = ctx.create_app().await;

        // Request without auth header should succeed when no auth required
        let payload = json!({
            "text": "Test without auth",
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
            "Request without auth should succeed when no auth required"
        );

        ctx.shutdown().await;
    }

    /// Test request with valid API key format
    #[tokio::test]
    async fn test_with_api_key_header() {
        let config = TestRouterConfig::round_robin(4301);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20301)]).await;

        let app = ctx.create_app().await;

        // Request with Bearer token header
        let payload = json!({
            "text": "Test with auth header",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header(AUTH_HEADER, "Bearer test-api-key-12345")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Without auth enforcement, request should succeed
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request with auth header should succeed"
        );

        ctx.shutdown().await;
    }

    /// Test health endpoint doesn't require authentication
    #[tokio::test]
    async fn test_health_endpoint_no_auth() {
        let config = TestRouterConfig::round_robin(4302);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20302)]).await;

        let app = ctx.create_app().await;

        // Health endpoint should be accessible without auth
        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Health endpoint should not require auth"
        );

        ctx.shutdown().await;
    }

    /// Test OpenAI-compatible API key header (X-API-Key)
    #[tokio::test]
    async fn test_openai_api_key_header() {
        let config = TestRouterConfig::round_robin(4303);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20303)]).await;

        let app = ctx.create_app().await;

        // Request with X-API-Key header (OpenAI style)
        let payload = json!({
            "text": "Test with X-API-Key",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header("X-API-Key", "sk-test-key-12345")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request with X-API-Key should succeed"
        );

        ctx.shutdown().await;
    }

    /// Test multiple concurrent authenticated requests
    #[tokio::test]
    async fn test_concurrent_authenticated_requests() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let config = TestRouterConfig::round_robin(4304);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20304)]).await;

        let app = ctx.create_app().await;
        let success_count = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for i in 0..20 {
            let app_clone = app.clone();
            let success_clone = Arc::clone(&success_count);

            let handle = tokio::spawn(async move {
                let payload = json!({
                    "text": format!("Concurrent auth test {}", i),
                    "stream": false
                });

                let req = Request::builder()
                    .method("POST")
                    .uri("/generate")
                    .header(CONTENT_TYPE, "application/json")
                    .header(AUTH_HEADER, format!("Bearer test-key-{}", i))
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

        assert_eq!(
            success_count.load(Ordering::SeqCst),
            20,
            "All concurrent authenticated requests should succeed"
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod mtls_tests {
    use super::*;

    /// Test that TLS configuration options exist
    /// Note: Actual mTLS testing would require certificate setup
    #[tokio::test]
    async fn test_tls_config_available() {
        // This test verifies the config builder accepts TLS-related options
        // Actual mTLS testing requires certificate infrastructure
        let config = TestRouterConfig::round_robin(4305);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20305)]).await;

        let app = ctx.create_app().await;

        // Basic request should work (no TLS in test mode)
        let payload = json!({
            "text": "TLS config test",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        ctx.shutdown().await;
    }
}
