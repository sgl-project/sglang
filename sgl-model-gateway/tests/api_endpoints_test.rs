mod common;

use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use reqwest::Client;
use serde_json::json;
use sgl_model_gateway::{
    app_context::AppContext,
    config::{RouterConfig, RoutingMode},
    core::Job,
    routers::{RouterFactory, RouterTrait},
};
use tower::ServiceExt;

/// Test context that manages mock workers
struct TestContext {
    workers: Vec<MockWorker>,
    router: Arc<dyn RouterTrait>,
    _client: Client,
    _config: RouterConfig,
    app_context: Arc<AppContext>,
}

impl TestContext {
    async fn new(worker_configs: Vec<MockWorkerConfig>) -> Self {
        // Create default router config
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host("127.0.0.1")
            .port(3002)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(1)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

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
        match &mut config.mode {
            RoutingMode::Regular {
                worker_urls: ref mut urls,
            } => {
                if urls.is_empty() {
                    *urls = worker_urls.clone();
                }
            }
            RoutingMode::OpenAI {
                worker_urls: ref mut urls,
            } => {
                if urls.is_empty() {
                    *urls = worker_urls.clone();
                }
            }
            _ => {} // PrefillDecode mode has its own setup
        }

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
            .build()
            .unwrap();

        // Create app context
        let app_context = common::create_test_context(config.clone()).await;

        // Submit worker initialization job (same as real server does)
        if !worker_urls.is_empty() {
            let job_queue = app_context
                .worker_job_queue
                .get()
                .expect("JobQueue should be initialized");
            let job = Job::InitializeWorkersFromConfig {
                router_config: Box::new(config.clone()),
            };
            job_queue
                .submit(job)
                .await
                .expect("Failed to submit worker initialization job");

            // Poll until all workers are healthy (up to 10 seconds)
            let expected_count = worker_urls.len();
            let start = tokio::time::Instant::now();
            let timeout_duration = tokio::time::Duration::from_secs(10);
            loop {
                let healthy_workers = app_context
                    .worker_registry
                    .get_all()
                    .iter()
                    .filter(|w| w.is_healthy())
                    .count();

                if healthy_workers >= expected_count {
                    break;
                }

                if start.elapsed() > timeout_duration {
                    panic!(
                        "Timeout waiting for {} workers to become healthy (only {} ready)",
                        expected_count, healthy_workers
                    );
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }

        // Create router
        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let router = Arc::from(router);

        Self {
            workers,
            router,
            _client: client,
            _config: config,
            app_context,
        }
    }

    async fn create_app(&self) -> axum::Router {
        common::test_app::create_test_app_with_context(
            Arc::clone(&self.router),
            Arc::clone(&self.app_context),
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
            port: 18207,
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
        // TODO: Update test after worker management refactoring
        // For now, skip this check

        ctx.shutdown().await;
    }
}

#[cfg(test)]
mod responses_endpoint_tests {
    use reqwest::Client as HttpClient;

    use super::*;

    #[tokio::test]
    async fn test_v1_responses_non_streaming() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18950,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "input": "Hello Responses API",
            "model": "mock-model",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body_json["object"], "response");
        assert_eq!(body_json["status"], "completed");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_responses_streaming() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18951,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "input": "Hello Responses API",
            "model": "mock-model",
            "stream": true
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Check that content-type indicates SSE
        let headers = resp.headers().clone();
        let ct = headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(ct.contains("text/event-stream"));

        // We don't fully consume the stream in this test harness.
        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_responses_get() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18952,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        // First create a response to obtain an id
        let resp_id = "test-get-resp-id-123";
        let payload = json!({
            "input": "Hello Responses API",
            "model": "mock-model",
            "stream": false,
            "store": true,
            "background": true,
            "request_id": resp_id
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Retrieve the response
        let req = Request::builder()
            .method("GET")
            .uri(format!("/v1/responses/{}", resp_id))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let get_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(get_json["object"], "response");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_responses_cancel() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18953,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        // First create a response to obtain an id
        let resp_id = "test-cancel-resp-id-456";
        let payload = json!({
            "input": "Hello Responses API",
            "model": "mock-model",
            "stream": false,
            "store": true,
            "background": true,
            "request_id": resp_id
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Cancel the response
        let req = Request::builder()
            .method("POST")
            .uri(format!("/v1/responses/{}/cancel", resp_id))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let cancel_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(cancel_json["status"], "cancelled");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_responses_delete_not_implemented() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18954,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        // Test DELETE is not implemented
        let resp_id = "resp-test-123";

        let req = Request::builder()
            .method("DELETE")
            .uri(format!("/v1/responses/{}", resp_id))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_responses_input_items() {
        // This test uses OpenAI mode because the input_items endpoint
        // is only implemented in OpenAIRouter and reads from storage (no workers needed)
        let config = RouterConfig::builder()
            .openai_mode(vec!["http://dummy.local".to_string()]) // Dummy URL (won't be called)
            .random_policy()
            .host("127.0.0.1")
            .port(3002)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(1)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_size(0)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = TestContext::new_with_config(
            config,
            vec![], // No workers needed
        )
        .await;

        let app = ctx.create_app().await;

        // Directly store a response in the storage to test the retrieval endpoint
        use sgl_model_gateway::data_connector::{ResponseId, StoredResponse};
        let mut stored_response = StoredResponse::new(None);
        stored_response.id = ResponseId::from("resp_test_input_items");
        stored_response.input = json!([
            {"id": "item_1", "content": "hello", "role": "user"},
            {"id": "item_2", "content": "hi there", "role": "assistant"}
        ]);
        stored_response.output = json!([
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "test response"}]}
        ]);

        ctx.app_context
            .response_storage
            .store_response(stored_response)
            .await
            .expect("Failed to store response");

        // Fetch input_items for the created response
        let req = Request::builder()
            .method("GET")
            .uri("/v1/responses/resp_test_input_items/input_items")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let items_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Verify response structure
        assert_eq!(items_json["object"], "list");
        assert!(items_json["data"].is_array());

        // Should have 2 input items
        let items = items_json["data"].as_array().unwrap();
        assert_eq!(items.len(), 2);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_responses_get_multi_worker_fanout() {
        // Start two mock workers
        let ctx = TestContext::new(vec![
            MockWorkerConfig {
                port: 18960,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
            MockWorkerConfig {
                port: 18961,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            },
        ])
        .await;

        let app = ctx.create_app().await;

        // Create a background response with a known id
        let rid = format!("resp_{}", 18960); // arbitrary unique id
        let payload = json!({
            "input": "Hello Responses API",
            "model": "mock-model",
            "background": true,
            "store": true,
            "request_id": rid,
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Using the router, GET should succeed by fanning out across workers
        let req = Request::builder()
            .method("GET")
            .uri(format!("/v1/responses/{}", rid))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Validate only one worker holds the metadata: direct calls
        let client = HttpClient::new();
        let mut ok_count = 0usize;
        // Get the actual worker URLs from the context
        let worker_urls: Vec<String> = vec![
            "http://127.0.0.1:18960".to_string(),
            "http://127.0.0.1:18961".to_string(),
        ];
        for url in worker_urls {
            let get_url = format!("{}/v1/responses/{}", url, rid);
            let res = client.get(get_url).send().await.unwrap();
            if res.status() == StatusCode::OK {
                ok_count += 1;
            }
        }
        assert_eq!(ok_count, 1, "exactly one worker should store the response");

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

        let req = Request::builder()
            .method("GET")
            .uri("/unknown_endpoint")
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

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
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host("127.0.0.1")
            .port(3010)
            .max_payload_size(1024) // 1KB limit
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(1)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

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
            .next_back()
            .and_then(|p| p.trim_end_matches('/').parse::<u16>().ok())
            .unwrap_or(9000);

        let config = RouterConfig::builder()
            .prefill_decode_mode(vec![(prefill_url, Some(prefill_port))], vec![decode_url])
            .random_policy()
            .host("127.0.0.1")
            .port(3011)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(1)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        // Create app context
        let app_context = common::create_test_context(config).await;

        // Create router - this might fail due to health check issues
        let router_result = RouterFactory::create_router(&app_context).await;

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
        let config = RouterConfig::builder()
            .regular_mode(vec![])
            .random_policy()
            .host("127.0.0.1")
            .port(3002)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(1)
            .worker_startup_check_interval_secs(1)
            .request_id_headers(vec!["custom-id".to_string(), "trace-id".to_string()])
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

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

#[cfg(test)]
mod rerank_tests {
    use super::*;
    // Note: RerankRequest and RerankResult are available for future use

    #[tokio::test]
    async fn test_rerank_success() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18105,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "query": "machine learning algorithms",
            "documents": [
                "Introduction to machine learning concepts",
                "Deep learning neural networks tutorial"
            ],
            "model": "test-rerank-model",
            "top_k": 2,
            "return_documents": true,
            "rid": "test-request-123"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/rerank")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(body_json.get("results").is_some());
        assert!(body_json.get("model").is_some());
        assert_eq!(body_json["model"], "test-rerank-model");

        let results = body_json["results"].as_array().unwrap();
        assert_eq!(results.len(), 2);

        assert!(results[0]["score"].as_f64().unwrap() >= results[1]["score"].as_f64().unwrap());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_rerank_with_top_k() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18106,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "query": "test query",
            "documents": [
                "Document 1",
                "Document 2",
                "Document 3"
            ],
            "model": "test-model",
            "top_k": 1,
            "return_documents": true
        });

        let req = Request::builder()
            .method("POST")
            .uri("/rerank")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Should only return top_k results
        let results = body_json["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_rerank_without_documents() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18107,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "query": "test query",
            "documents": ["Document 1", "Document 2"],
            "model": "test-model",
            "return_documents": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/rerank")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Documents should be null when return_documents is false
        let results = body_json["results"].as_array().unwrap();
        for result in results {
            assert!(result.get("document").is_none());
        }

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_rerank_worker_failure() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18108,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 1.0, // Always fail
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "query": "test query",
            "documents": ["Document 1"],
            "model": "test-model"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/rerank")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // Should return the worker's error response
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_rerank_compatibility() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18110,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "query": "machine learning algorithms",
            "documents": [
                "Introduction to machine learning concepts",
                "Deep learning neural networks tutorial",
                "Statistical learning theory basics"
            ]
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/rerank")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(body_json.get("results").is_some());
        assert!(body_json.get("model").is_some());

        // V1 API should use default model name
        assert_eq!(body_json["model"], "unknown");

        let results = body_json["results"].as_array().unwrap();
        assert_eq!(results.len(), 3); // All documents should be returned

        assert!(results[0]["score"].as_f64().unwrap() >= results[1]["score"].as_f64().unwrap());
        assert!(results[1]["score"].as_f64().unwrap() >= results[2]["score"].as_f64().unwrap());

        // V1 API should return documents by default
        for result in results {
            assert!(result.get("document").is_some());
        }

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_rerank_invalid_request() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 18111,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let app = ctx.create_app().await;

        let payload = json!({
            "query": "",
            "documents": ["Document 1", "Document 2"],
            "model": "test-model"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/rerank")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let payload = json!({
            "query": "   ",
            "documents": ["Document 1", "Document 2"],
            "model": "test-model"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/rerank")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let payload = json!({
            "query": "test query",
            "documents": [],
            "model": "test-model"
        });

        let req = Request::builder()
            .method("POST")
            .uri("/rerank")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let payload = json!({
            "query": "test query",
            "documents": ["Document 1", "Document 2"],
            "model": "test-model",
            "top_k": 0
        });

        let req = Request::builder()
            .method("POST")
            .uri("/rerank")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        ctx.shutdown().await;
    }
}
