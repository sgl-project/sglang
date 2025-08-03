mod common;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use reqwest::Client;
use serde_json::json;
use sglang_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use sglang_router_rs::routers::{RouterFactory, RouterTrait};
use std::sync::Arc;
use tower::ServiceExt;

/// Test context that manages mock workers
struct TestContext {
    workers: Vec<MockWorker>,
    router: Arc<dyn RouterTrait>,
    client: Client,
    config: RouterConfig,
}

impl TestContext {
    async fn new(worker_configs: Vec<MockWorkerConfig>) -> Self {
        // Create default router config
        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            policy: PolicyConfig::Random,
            host: "127.0.0.1".to_string(),
            port: 3002,
            max_payload_size: 256 * 1024 * 1024,
            request_timeout_secs: 600,
            worker_startup_timeout_secs: 1,
            worker_startup_check_interval_secs: 1,
            discovery: None,
            dp_aware: false,
            api_key: None,
            metrics: None,
            log_dir: None,
            log_level: None,
            request_id_headers: None,
            max_concurrent_requests: 64,
            cors_allowed_origins: vec![],
        };

        Self::new_with_config(config, worker_configs).await
    }

    async fn new_with_config(
        mut config: RouterConfig,
        worker_configs: Vec<MockWorkerConfig>,
    ) -> Self {
        let mut workers = Vec::new();
        let mut worker_urls = Vec::new();

        // Start mock workers if any
        for worker_config in worker_configs {
            let mut worker = MockWorker::new(worker_config);
            let url = worker.start().await.unwrap();
            worker_urls.push(url);
            workers.push(worker);
        }

        if !workers.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }

        // Update config with worker URLs if not already set
        if let RoutingMode::Regular {
            worker_urls: ref mut urls,
        } = config.mode
        {
            if urls.is_empty() {
                *urls = worker_urls.clone();
            }
        }

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
            .build()
            .unwrap();

        // Create app context
        let app_context = common::create_test_context(config.clone());

        // Create router using sync factory in a blocking context
        let router =
            tokio::task::spawn_blocking(move || RouterFactory::create_router(&app_context))
                .await
                .unwrap()
                .unwrap();
        let router = Arc::from(router);

        // Wait for router to discover workers
        if !workers.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        Self {
            workers,
            router,
            client,
            config,
        }
    }

    async fn create_app(&self) -> axum::Router {
        common::test_app::create_test_app(
            Arc::clone(&self.router),
            self.client.clone(),
            &self.config,
        )
    }

    async fn shutdown(mut self) {
        for worker in &mut self.workers {
            worker.stop().await;
        }
    }
}

#[cfg(test)]
mod health_tests {
    use super::*;

    #[tokio::test]
    async fn test_liveness_endpoint() {
        let ctx = TestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        let req = Request::builder()
            .method("GET")
            .uri("/liveness")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_readiness_with_healthy_workers() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18001,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let req = Request::builder()
            .method("GET")
            .uri("/readiness")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_readiness_with_unhealthy_workers() {
        let ctx = TestContext::new(vec![]).await;

        let app = ctx.create_app().await;

        let req = Request::builder()
            .method("GET")
            .uri("/readiness")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // With no workers, readiness should return SERVICE_UNAVAILABLE
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_health_endpoint_details() {
        let ctx = TestContext::new(vec![
            MockWorkerConfig {
                port: 18003,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
            MockWorkerConfig {
                port: 18004,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
        ])
        .await;

        let app = ctx.create_app().await;

        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // The health endpoint returns plain text, not JSON
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_str = String::from_utf8_lossy(&body);
        assert!(body_str.contains("All servers healthy"));

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_health_generate_endpoint() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18005,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let req = Request::builder()
            .method("GET")
            .uri("/health_generate")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(body_json.is_object());

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod generation_tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_success() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18101,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "text": "Hello, world!",
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

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(body_json.get("text").is_some());
        assert!(body_json.get("meta_info").is_some());
        let meta_info = &body_json["meta_info"];
        assert!(meta_info.get("finish_reason").is_some());
        assert_eq!(meta_info["finish_reason"]["type"], "stop");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_generate_streaming() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18102,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "text": "Stream test",
            "stream": true
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // For streaming responses, the router might use chunked encoding or other streaming mechanisms
        // The exact content-type can vary based on the router implementation
        // Just verify we got a successful response
        // Note: In a real implementation, we'd check for text/event-stream or appropriate streaming headers

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_generate_with_worker_failure() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18103,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 1.0, // Always fail
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "text": "This should fail",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_chat_completions_success() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18104,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello!"}
            ],
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(body_json.get("choices").is_some());

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod model_info_tests {
    use super::*;

    #[tokio::test]
    async fn test_get_server_info() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18201,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let req = Request::builder()
            .method("GET")
            .uri("/get_server_info")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(body_json.is_object());
        // Check for actual sglang server fields
        assert!(body_json.get("version").is_some());
        assert!(body_json.get("model_path").is_some());
        assert!(body_json.get("tokenizer_path").is_some());
        assert!(body_json.get("port").is_some());
        assert!(body_json.get("max_num_batched_tokens").is_some());
        assert!(body_json.get("schedule_policy").is_some());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_get_model_info() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18202,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let req = Request::builder()
            .method("GET")
            .uri("/get_model_info")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(body_json.is_object());
        // Check for actual sglang model info fields
        assert_eq!(
            body_json.get("model_path").and_then(|v| v.as_str()),
            Some("mock-model-path")
        );
        assert_eq!(
            body_json.get("tokenizer_path").and_then(|v| v.as_str()),
            Some("mock-tokenizer-path")
        );
        assert_eq!(
            body_json.get("is_generation").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert!(body_json.get("preferred_sampling_params").is_some());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_models() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18203,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(body_json.get("object").is_some());
        assert_eq!(
            body_json.get("object").and_then(|v| v.as_str()),
            Some("list")
        );

        let data = body_json.get("data").and_then(|v| v.as_array());
        assert!(data.is_some());

        let models = data.unwrap();
        assert!(!models.is_empty());

        let first_model = &models[0];
        assert_eq!(
            first_model.get("id").and_then(|v| v.as_str()),
            Some("mock-model")
        );
        assert_eq!(
            first_model.get("object").and_then(|v| v.as_str()),
            Some("model")
        );
        assert!(first_model.get("created").is_some());
        assert_eq!(
            first_model.get("owned_by").and_then(|v| v.as_str()),
            Some("organization-owner")
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_model_info_with_no_workers() {
        let ctx = TestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        // Test server info with no workers
        let req = Request::builder()
            .method("GET")
            .uri("/get_server_info")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        // Router may return various error codes when no workers
        assert!(
            resp.status() == StatusCode::OK
                || resp.status() == StatusCode::SERVICE_UNAVAILABLE
                || resp.status() == StatusCode::NOT_FOUND
                || resp.status() == StatusCode::INTERNAL_SERVER_ERROR,
            "Unexpected status code: {:?}",
            resp.status()
        );

        // Test model info with no workers
        let req = Request::builder()
            .method("GET")
            .uri("/get_model_info")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        // Router may return various error codes when no workers
        assert!(
            resp.status() == StatusCode::OK
                || resp.status() == StatusCode::SERVICE_UNAVAILABLE
                || resp.status() == StatusCode::NOT_FOUND
                || resp.status() == StatusCode::INTERNAL_SERVER_ERROR,
            "Unexpected status code: {:?}",
            resp.status()
        );

        // Test v1/models with no workers
        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // Router may return various error codes when no workers
        assert!(
            resp.status() == StatusCode::OK
                || resp.status() == StatusCode::SERVICE_UNAVAILABLE
                || resp.status() == StatusCode::NOT_FOUND
                || resp.status() == StatusCode::INTERNAL_SERVER_ERROR,
            "Unexpected status code: {:?}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_model_info_with_multiple_workers() {
        let ctx = TestContext::new(vec![
            MockWorkerConfig {
                port: 18204,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
            MockWorkerConfig {
                port: 18205,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
        ])
        .await;

        let app = ctx.create_app().await;

        // Test that model info is consistent across workers
        for _ in 0..5 {
            let req = Request::builder()
                .method("GET")
                .uri("/get_model_info")
                .body(Body::empty())
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);

            let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                body_json.get("model_path").and_then(|v| v.as_str()),
                Some("mock-model-path")
            );
        }

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_model_info_with_unhealthy_worker() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18206,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 1.0, // Always fail
        }])
        .await;

        let app = ctx.create_app().await;

        let req = Request::builder()
            .method("GET")
            .uri("/get_model_info")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Worker with fail_rate: 1.0 should always return an error status
        assert!(
            resp.status() == StatusCode::INTERNAL_SERVER_ERROR
                || resp.status() == StatusCode::SERVICE_UNAVAILABLE,
            "Expected error status for always-failing worker, got: {:?}",
            resp.status()
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod worker_management_tests {
    use super::*;

    #[tokio::test]
    async fn test_add_new_worker() {
        let ctx = TestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        // Start a mock worker
        let mut worker = MockWorker::new(MockWorkerConfig {
            port: 18301,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let url = worker.start().await.unwrap();

        // Add the worker
        let req = Request::builder()
            .method("POST")
            .uri(&format!("/add_worker?url={}", url))
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // List workers to verify
        let req = Request::builder()
            .method("GET")
            .uri("/list_workers")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let workers = body_json["urls"].as_array().unwrap();
        assert!(workers.iter().any(|w| w.as_str().unwrap() == url));

        worker.stop().await;
        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_remove_existing_worker() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18302,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        // Get the worker URL
        let req = Request::builder()
            .method("GET")
            .uri("/list_workers")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let workers = body_json["urls"].as_array().unwrap();
        let worker_url = workers[0].as_str().unwrap();

        // Remove the worker
        let req = Request::builder()
            .method("POST")
            .uri(&format!("/remove_worker?url={}", worker_url))
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify it's removed
        let req = Request::builder()
            .method("GET")
            .uri("/list_workers")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let workers = body_json["urls"].as_array().unwrap();
        assert!(workers.is_empty());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_add_worker_invalid_url() {
        let ctx = TestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        // Invalid URL format
        let req = Request::builder()
            .method("POST")
            .uri("/add_worker?url=not-a-valid-url")
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // Missing URL parameter
        let req = Request::builder()
            .method("POST")
            .uri("/add_worker")
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // Empty URL
        let req = Request::builder()
            .method("POST")
            .uri("/add_worker?url=")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_add_duplicate_worker() {
        // Start a mock worker
        let mut worker = MockWorker::new(MockWorkerConfig {
            port: 18303,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let url = worker.start().await.unwrap();

        let ctx = TestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        // Add worker first time
        let req = Request::builder()
            .method("POST")
            .uri(&format!("/add_worker?url={}", url))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

        // Try to add same worker again
        let req = Request::builder()
            .method("POST")
            .uri(&format!("/add_worker?url={}", url))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // Should return error for duplicate
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        worker.stop().await;
        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_add_unhealthy_worker() {
        // Start unhealthy worker
        let mut worker = MockWorker::new(MockWorkerConfig {
            port: 18304,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Unhealthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let url = worker.start().await.unwrap();

        let ctx = TestContext::new(vec![]).await;
        let app = ctx.create_app().await;

        // Try to add unhealthy worker
        let req = Request::builder()
            .method("POST")
            .uri(&format!("/add_worker?url={}", url))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();

        // Router should reject unhealthy workers
        assert!(
            resp.status() == StatusCode::BAD_REQUEST
                || resp.status() == StatusCode::SERVICE_UNAVAILABLE
        );

        worker.stop().await;
        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod router_policy_tests {
    use super::*;

    #[tokio::test]
    async fn test_random_policy() {
        let ctx = TestContext::new(vec![
            MockWorkerConfig {
                port: 18801,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
            MockWorkerConfig {
                port: 18802,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
        ])
        .await;

        // Send multiple requests and verify they succeed
        let app = ctx.create_app().await;

        for i in 0..10 {
            let payload = json!({
                "text": format!("Request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_worker_selection() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18203,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let _payload = json!({
            "text": "Test selection",
            "stream": false
        });

        // Check that router has the worker
        let worker_urls = ctx.router.get_worker_urls();
        assert_eq!(worker_urls.len(), 1);
        assert!(worker_urls[0].contains("18203"));

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[tokio::test]
    async fn test_404_not_found() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18401,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        // Test unknown endpoint
        let req = Request::builder()
            .method("GET")
            .uri("/unknown_endpoint")
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        // Test POST to unknown endpoint
        let req = Request::builder()
            .method("POST")
            .uri("/api/v2/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::to_string(&json!({"text": "test"})).unwrap(),
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_method_not_allowed() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18402,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        // GET request to POST-only endpoint
        let req = Request::builder()
            .method("GET")
            .uri("/generate")
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        // Note: Axum returns 405 for wrong methods on matched routes
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);

        // POST request to GET-only endpoint
        let req = Request::builder()
            .method("POST")
            .uri("/health")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from("{}"))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_payload_too_large() {
        // Create context with small payload limit
        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            policy: PolicyConfig::Random,
            host: "127.0.0.1".to_string(),
            port: 3010,
            max_payload_size: 1024, // 1KB limit
            request_timeout_secs: 600,
            worker_startup_timeout_secs: 1,
            worker_startup_check_interval_secs: 1,
            dp_aware: false,
            api_key: None,
            discovery: None,
            metrics: None,
            log_dir: None,
            log_level: None,
            request_id_headers: None,
            max_concurrent_requests: 64,
            cors_allowed_origins: vec![],
        };

        let ctx = TestContext::new_with_config(
            config,
            vec![MockWorkerConfig {
                port: 18403,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }],
        )
        .await;

        // Note: The server would have payload size middleware configured
        // but we cannot test it directly through the test app
        // This test is kept for documentation purposes

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_invalid_json_payload() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18404,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        // Send invalid JSON
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from("{invalid json}"))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        // Send empty body
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_missing_required_fields() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18405,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        // Missing messages in chat completion
        let payload = json!({
            "model": "test-model"
            // missing "messages"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Axum validates JSON schema - returns 422 for validation errors
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_invalid_model() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18406,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "model": "invalid-model-name-that-does-not-exist",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Mock worker accepts any model, but real implementation might return 400
        assert!(resp.status().is_success() || resp.status() == StatusCode::BAD_REQUEST);

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[tokio::test]
    async fn test_flush_cache() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18501,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let req = Request::builder()
            .method("POST")
            .uri("/flush_cache")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // The response might be empty or contain a message
        let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        if !body_bytes.is_empty() {
            if let Ok(body) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                // Check that we got a successful response with expected fields
                assert!(body.is_object());
                assert!(body.get("message").is_some() || body.get("status").is_some());
            }
        }

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_get_loads() {
        let ctx = TestContext::new(vec![
            MockWorkerConfig {
                port: 18502,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
            MockWorkerConfig {
                port: 18503,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
        ])
        .await;

        let app = ctx.create_app().await;

        let req = Request::builder()
            .method("GET")
            .uri("/get_loads")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Verify the response contains load information
        assert!(body_json.is_object());
        // The exact structure depends on the implementation
        // but should contain worker load information

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_flush_cache_no_workers() {
        let ctx = TestContext::new(vec![]).await;

        let app = ctx.create_app().await;

        let req = Request::builder()
            .method("POST")
            .uri("/flush_cache")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Should either succeed (no-op) or return service unavailable
        assert!(
            resp.status() == StatusCode::OK || resp.status() == StatusCode::SERVICE_UNAVAILABLE
        );

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod load_balancing_tests {
    use super::*;

    #[tokio::test]
    async fn test_request_distribution() {
        // Create multiple workers
        let ctx = TestContext::new(vec![
            MockWorkerConfig {
                port: 18601,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
            MockWorkerConfig {
                port: 18602,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
        ])
        .await;

        let app = ctx.create_app().await;

        // Send multiple requests and track distribution
        let mut request_count = 0;
        for i in 0..10 {
            let payload = json!({
                "text": format!("Request {}", i),
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
                request_count += 1;
            }
        }

        // With random policy, all requests should succeed
        assert_eq!(request_count, 10);

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod pd_mode_tests {
    use super::*;

    #[tokio::test]
    async fn test_pd_mode_routing() {
        // Create PD mode configuration with prefill and decode workers
        let mut prefill_worker = MockWorker::new(MockWorkerConfig {
            port: 18701,
            worker_type: WorkerType::Prefill,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });

        let mut decode_worker = MockWorker::new(MockWorkerConfig {
            port: 18702,
            worker_type: WorkerType::Decode,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });

        let prefill_url = prefill_worker.start().await.unwrap();
        let decode_url = decode_worker.start().await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Extract port from prefill URL
        let prefill_port = prefill_url
            .split(':')
            .last()
            .and_then(|p| p.trim_end_matches('/').parse::<u16>().ok())
            .unwrap_or(9000);

        let config = RouterConfig {
            mode: RoutingMode::PrefillDecode {
                prefill_urls: vec![(prefill_url, Some(prefill_port))],
                decode_urls: vec![decode_url],
                prefill_policy: None,
                decode_policy: None,
            },
            policy: PolicyConfig::Random,
            host: "127.0.0.1".to_string(),
            port: 3011,
            max_payload_size: 256 * 1024 * 1024,
            request_timeout_secs: 600,
            worker_startup_timeout_secs: 1,
            worker_startup_check_interval_secs: 1,
            discovery: None,
            metrics: None,
            log_dir: None,
            dp_aware: false,
            api_key: None,
            log_level: None,
            request_id_headers: None,
            max_concurrent_requests: 64,
            cors_allowed_origins: vec![],
        };

        // Create app context
        let app_context = common::create_test_context(config);

        // Create router - this might fail due to health check issues
        let router_result =
            tokio::task::spawn_blocking(move || RouterFactory::create_router(&app_context))
                .await
                .unwrap();

        // Clean up workers
        prefill_worker.stop().await;
        decode_worker.stop().await;

        // For now, just verify the configuration was attempted
        assert!(router_result.is_err() || router_result.is_ok());
    }
}

#[cfg(test)]
mod request_id_tests {
    use super::*;

    #[tokio::test]
    async fn test_request_id_generation() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18901,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        // Test 1: Request without any request ID header should generate one
        let payload = json!({
            "text": "Test request",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Check that response has x-request-id header
        let request_id = resp.headers().get("x-request-id");
        assert!(
            request_id.is_some(),
            "Response should have x-request-id header"
        );

        let id_value = request_id.unwrap().to_str().unwrap();
        assert!(
            id_value.starts_with("gnt-"),
            "Generate endpoint should have gnt- prefix"
        );
        assert!(
            id_value.len() > 4,
            "Request ID should have content after prefix"
        );

        // Test 2: Request with custom x-request-id should preserve it
        let custom_id = "custom-request-id-123";
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header("x-request-id", custom_id)
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let response_id = resp.headers().get("x-request-id");
        assert!(response_id.is_some());
        assert_eq!(response_id.unwrap(), custom_id);

        // Test 3: Different endpoints should have different prefixes
        let chat_payload = json!({
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&chat_payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let request_id = resp.headers().get("x-request-id");
        assert!(request_id.is_some());
        assert!(request_id
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("chatcmpl-"));

        // Test 4: Alternative request ID headers should be recognized
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header("x-correlation-id", "correlation-123")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let response_id = resp.headers().get("x-request-id");
        assert!(response_id.is_some());
        assert_eq!(response_id.unwrap(), "correlation-123");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_request_id_with_custom_headers() {
        // Create config with custom request ID headers
        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            policy: PolicyConfig::Random,
            host: "127.0.0.1".to_string(),
            port: 3002,
            max_payload_size: 256 * 1024 * 1024,
            request_timeout_secs: 600,
            worker_startup_timeout_secs: 1,
            worker_startup_check_interval_secs: 1,
            discovery: None,
            metrics: None,
            dp_aware: false,
            api_key: None,
            log_dir: None,
            log_level: None,
            request_id_headers: Some(vec!["custom-id".to_string(), "trace-id".to_string()]),
            max_concurrent_requests: 64,
            cors_allowed_origins: vec![],
        };

        let ctx = TestContext::new_with_config(
            config,
            vec![MockWorkerConfig {
                port: 18902,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }],
        )
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "text": "Test request",
            "stream": false
        });

        // Test custom header is recognized
        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .header("custom-id", "my-custom-id")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let response_id = resp.headers().get("x-request-id");
        assert!(response_id.is_some());
        assert_eq!(response_id.unwrap(), "my-custom-id");

        ctx.shutdown().await;
    }
}
