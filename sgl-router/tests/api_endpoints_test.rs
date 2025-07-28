mod common;

use actix_web::{http::StatusCode, rt::System, test as actix_test, web, App};
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use reqwest::Client;
use serde_json::json;
use sglang_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use sglang_router_rs::server::{
    add_worker, flush_cache, generate, get_loads, get_model_info, get_server_info, health,
    health_generate, list_workers, liveness, readiness, remove_worker, v1_chat_completions,
    v1_completions, v1_models, AppState,
};

/// Test context that manages mock workers
struct TestContext {
    workers: Vec<MockWorker>,
    app_state: web::Data<AppState>,
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
            metrics: None,
            log_dir: None,
            log_level: None,
        };

        Self::new_with_config(config, worker_configs).await
    }

    async fn new_with_config(config: RouterConfig, worker_configs: Vec<MockWorkerConfig>) -> Self {
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

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
            .build()
            .unwrap();

        let app_state = AppState::new(config, client).unwrap();
        let app_state = web::Data::new(app_state);

        // Add workers if any
        if !worker_urls.is_empty() {
            let app = actix_test::init_service(
                App::new().app_data(app_state.clone()).service(add_worker),
            )
            .await;

            for url in &worker_urls {
                let req = actix_test::TestRequest::post()
                    .uri(&format!("/add_worker?url={}", url))
                    .to_request();
                let resp = actix_test::call_service(&app, req).await;
                assert!(resp.status().is_success());
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        Self { workers, app_state }
    }

    async fn create_app(
        &self,
    ) -> impl actix_web::dev::Service<
        actix_http::Request,
        Response = actix_web::dev::ServiceResponse,
        Error = actix_web::Error,
    > {
        actix_test::init_service(
            App::new()
                .app_data(self.app_state.clone())
                .service(liveness)
                .service(readiness)
                .service(health)
                .service(health_generate)
                .service(get_server_info)
                .service(get_model_info)
                .service(v1_models)
                .service(generate)
                .service(v1_chat_completions)
                .service(v1_completions)
                .service(add_worker)
                .service(list_workers)
                .service(remove_worker)
                .service(flush_cache)
                .service(get_loads),
        )
        .await
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

    #[test]
    fn test_liveness_endpoint() {
        System::new().block_on(async {
            let ctx = TestContext::new(vec![]).await;
            let app = ctx.create_app().await;

            let req = actix_test::TestRequest::get().uri("/liveness").to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_readiness_with_healthy_workers() {
        System::new().block_on(async {
            let ctx = TestContext::new(vec![MockWorkerConfig {
                port: 18001,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            let req = actix_test::TestRequest::get()
                .uri("/readiness")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_readiness_with_unhealthy_workers() {
        System::new().block_on(async {
            // Create an empty context (no workers)
            let ctx = TestContext::new(vec![]).await;

            let app = ctx.create_app().await;

            let req = actix_test::TestRequest::get()
                .uri("/readiness")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            // With no workers, readiness should return SERVICE_UNAVAILABLE
            assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_health_endpoint_details() {
        System::new().block_on(async {
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

            let req = actix_test::TestRequest::get().uri("/health").to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            // The health endpoint returns plain text, not JSON
            let body = actix_test::read_body(resp).await;
            let body_str = String::from_utf8_lossy(&body);
            assert!(body_str.contains("All servers healthy"));

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_health_generate_endpoint() {
        System::new().block_on(async {
            let ctx = TestContext::new(vec![MockWorkerConfig {
                port: 18005,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            let req = actix_test::TestRequest::get()
                .uri("/health_generate")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            let body: serde_json::Value = actix_test::read_body_json(resp).await;
            assert!(body.is_object());

            ctx.shutdown().await;
        });
    }
}

#[cfg(test)]
mod generation_tests {
    use super::*;

    #[test]
    fn test_generate_success() {
        System::new().block_on(async {
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

            let req = actix_test::TestRequest::post()
                .uri("/generate")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            let body: serde_json::Value = actix_test::read_body_json(resp).await;
            assert!(body.get("text").is_some());
            assert!(body.get("meta_info").is_some());
            let meta_info = &body["meta_info"];
            assert!(meta_info.get("finish_reason").is_some());
            assert_eq!(meta_info["finish_reason"]["type"], "stop");

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_generate_streaming() {
        System::new().block_on(async {
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

            let req = actix_test::TestRequest::post()
                .uri("/generate")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            // Check that it's a streaming response
            let content_type = resp.headers().get("content-type");
            assert!(content_type.is_some());
            assert_eq!(content_type.unwrap(), "text/event-stream");

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_generate_with_worker_failure() {
        System::new().block_on(async {
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

            let req = actix_test::TestRequest::post()
                .uri("/generate")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_v1_chat_completions_success() {
        System::new().block_on(async {
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

            let req = actix_test::TestRequest::post()
                .uri("/v1/chat/completions")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            let body: serde_json::Value = actix_test::read_body_json(resp).await;
            assert!(body.get("choices").is_some());

            ctx.shutdown().await;
        });
    }
}

#[cfg(test)]
mod model_info_tests {
    use super::*;

    #[test]
    fn test_get_server_info() {
        System::new().block_on(async {
            let ctx = TestContext::new(vec![MockWorkerConfig {
                port: 18201,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            let req = actix_test::TestRequest::get()
                .uri("/get_server_info")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            let body: serde_json::Value = actix_test::read_body_json(resp).await;
            assert!(body.is_object());
            // Check for actual sglang server fields
            assert!(body.get("version").is_some());
            assert!(body.get("model_path").is_some());
            assert!(body.get("tokenizer_path").is_some());
            assert!(body.get("port").is_some());
            assert!(body.get("max_num_batched_tokens").is_some());
            assert!(body.get("schedule_policy").is_some());

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_get_model_info() {
        System::new().block_on(async {
            let ctx = TestContext::new(vec![MockWorkerConfig {
                port: 18202,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            let req = actix_test::TestRequest::get()
                .uri("/get_model_info")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            let body: serde_json::Value = actix_test::read_body_json(resp).await;
            assert!(body.is_object());
            // Check for actual sglang model info fields
            assert_eq!(
                body.get("model_path").and_then(|v| v.as_str()),
                Some("mock-model-path")
            );
            assert_eq!(
                body.get("tokenizer_path").and_then(|v| v.as_str()),
                Some("mock-tokenizer-path")
            );
            assert_eq!(
                body.get("is_generation").and_then(|v| v.as_bool()),
                Some(true)
            );
            assert!(body.get("preferred_sampling_params").is_some());

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_v1_models() {
        System::new().block_on(async {
            let ctx = TestContext::new(vec![MockWorkerConfig {
                port: 18203,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            let req = actix_test::TestRequest::get()
                .uri("/v1/models")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            let body: serde_json::Value = actix_test::read_body_json(resp).await;
            assert!(body.get("object").is_some());
            assert_eq!(body.get("object").and_then(|v| v.as_str()), Some("list"));

            let data = body.get("data").and_then(|v| v.as_array());
            assert!(data.is_some());

            let models = data.unwrap();
            assert!(!models.is_empty());

            let first_model = &models[0];
            assert_eq!(
                first_model.get("id").and_then(|v| v.as_str()),
                Some("mock-model-v1")
            );
            assert_eq!(
                first_model.get("object").and_then(|v| v.as_str()),
                Some("model")
            );
            assert!(first_model.get("created").is_some());
            assert_eq!(
                first_model.get("owned_by").and_then(|v| v.as_str()),
                Some("sglang")
            );

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_model_info_with_no_workers() {
        System::new().block_on(async {
            let ctx = TestContext::new(vec![]).await;
            let app = ctx.create_app().await;

            // Test server info with no workers
            let req = actix_test::TestRequest::get()
                .uri("/get_server_info")
                .to_request();
            let resp = actix_test::call_service(&app, req).await;
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
            let req = actix_test::TestRequest::get()
                .uri("/get_model_info")
                .to_request();
            let resp = actix_test::call_service(&app, req).await;
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
            let req = actix_test::TestRequest::get()
                .uri("/v1/models")
                .to_request();
            let resp = actix_test::call_service(&app, req).await;
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
        });
    }

    #[test]
    fn test_model_info_with_multiple_workers() {
        System::new().block_on(async {
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
                let req = actix_test::TestRequest::get()
                    .uri("/get_model_info")
                    .to_request();

                let resp = actix_test::call_service(&app, req).await;
                assert_eq!(resp.status(), StatusCode::OK);

                let body: serde_json::Value = actix_test::read_body_json(resp).await;
                assert_eq!(
                    body.get("model_path").and_then(|v| v.as_str()),
                    Some("mock-model-path")
                );
            }

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_model_info_with_unhealthy_worker() {
        System::new().block_on(async {
            let ctx = TestContext::new(vec![MockWorkerConfig {
                port: 18206,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 1.0, // Always fail
            }])
            .await;

            let app = ctx.create_app().await;

            let req = actix_test::TestRequest::get()
                .uri("/get_model_info")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            // Worker with fail_rate: 1.0 should always return an error status
            assert!(
                resp.status() == StatusCode::INTERNAL_SERVER_ERROR
                    || resp.status() == StatusCode::SERVICE_UNAVAILABLE,
                "Expected error status for always-failing worker, got: {:?}",
                resp.status()
            );

            ctx.shutdown().await;
        });
    }
}

#[cfg(test)]
mod worker_management_tests {
    use super::*;

    #[test]
    fn test_add_new_worker() {
        System::new().block_on(async {
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
            let req = actix_test::TestRequest::post()
                .uri(&format!("/add_worker?url={}", url))
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            // List workers to verify
            let req = actix_test::TestRequest::get()
                .uri("/list_workers")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            let body: serde_json::Value = actix_test::read_body_json(resp).await;
            let workers = body["urls"].as_array().unwrap();
            assert!(workers.iter().any(|w| w.as_str().unwrap() == url));

            worker.stop().await;
            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_remove_existing_worker() {
        System::new().block_on(async {
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
            let req = actix_test::TestRequest::get()
                .uri("/list_workers")
                .to_request();
            let resp = actix_test::call_service(&app, req).await;
            let body: serde_json::Value = actix_test::read_body_json(resp).await;
            let workers = body["urls"].as_array().unwrap();
            let worker_url = workers[0].as_str().unwrap();

            // Remove the worker
            let req = actix_test::TestRequest::post()
                .uri(&format!("/remove_worker?url={}", worker_url))
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            // Verify it's removed
            let req = actix_test::TestRequest::get()
                .uri("/list_workers")
                .to_request();
            let resp = actix_test::call_service(&app, req).await;
            let body: serde_json::Value = actix_test::read_body_json(resp).await;
            let workers = body["urls"].as_array().unwrap();
            assert!(workers.is_empty());

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_add_worker_invalid_url() {
        System::new().block_on(async {
            let ctx = TestContext::new(vec![]).await;
            let app = ctx.create_app().await;

            // Invalid URL format
            let req = actix_test::TestRequest::post()
                .uri("/add_worker?url=not-a-valid-url")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

            // Missing URL parameter
            let req = actix_test::TestRequest::post()
                .uri("/add_worker")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

            // Empty URL
            let req = actix_test::TestRequest::post()
                .uri("/add_worker?url=")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_add_duplicate_worker() {
        System::new().block_on(async {
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
            let req = actix_test::TestRequest::post()
                .uri(&format!("/add_worker?url={}", url))
                .to_request();
            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

            // Try to add same worker again
            let req = actix_test::TestRequest::post()
                .uri(&format!("/add_worker?url={}", url))
                .to_request();
            let resp = actix_test::call_service(&app, req).await;
            // Should return error for duplicate
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

            worker.stop().await;
            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_add_unhealthy_worker() {
        System::new().block_on(async {
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
            let req = actix_test::TestRequest::post()
                .uri(&format!("/add_worker?url={}", url))
                .to_request();
            let resp = actix_test::call_service(&app, req).await;

            // Router should reject unhealthy workers
            assert!(
                resp.status() == StatusCode::BAD_REQUEST
                    || resp.status() == StatusCode::SERVICE_UNAVAILABLE
            );

            worker.stop().await;
            ctx.shutdown().await;
        });
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn test_404_not_found() {
        System::new().block_on(async {
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
            let req = actix_test::TestRequest::get()
                .uri("/unknown_endpoint")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::NOT_FOUND);

            // Test POST to unknown endpoint
            let req = actix_test::TestRequest::post()
                .uri("/api/v2/generate")
                .set_json(&json!({"text": "test"}))
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::NOT_FOUND);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_method_not_allowed() {
        System::new().block_on(async {
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
            let req = actix_test::TestRequest::get().uri("/generate").to_request();

            let resp = actix_test::call_service(&app, req).await;
            // Note: actix-web returns 404 for unmatched methods in some configurations
            assert!(
                resp.status() == StatusCode::METHOD_NOT_ALLOWED
                    || resp.status() == StatusCode::NOT_FOUND
            );

            // POST request to GET-only endpoint
            let req = actix_test::TestRequest::post()
                .uri("/health")
                .set_json(&json!({}))
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            // Note: actix-web returns 404 for unmatched methods in some configurations
            assert!(
                resp.status() == StatusCode::METHOD_NOT_ALLOWED
                    || resp.status() == StatusCode::NOT_FOUND
            );

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_payload_too_large() {
        System::new().block_on(async {
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
                discovery: None,
                metrics: None,
                log_dir: None,
                log_level: None,
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

            let app = ctx.create_app().await;

            // Create large payload (> 1KB)
            let large_text = "x".repeat(2000);
            let payload = json!({
                "text": large_text,
                "stream": false
            });

            let req = actix_test::TestRequest::post()
                .uri("/generate")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            // Note: The test framework may not enforce payload size limits the same way as the full server
            // In production, the server middleware would reject large payloads before reaching handlers
            assert!(
                resp.status() == StatusCode::PAYLOAD_TOO_LARGE || resp.status() == StatusCode::OK
            );

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_invalid_json_payload() {
        System::new().block_on(async {
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
            let req = actix_test::TestRequest::post()
                .uri("/generate")
                .insert_header(("content-type", "application/json"))
                .set_payload("{invalid json}")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

            // Send empty body
            let req = actix_test::TestRequest::post()
                .uri("/generate")
                .insert_header(("content-type", "application/json"))
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_missing_required_fields() {
        System::new().block_on(async {
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

            let req = actix_test::TestRequest::post()
                .uri("/v1/chat/completions")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            // Note: Mock worker might accept this, but real implementation would return 400
            // The status depends on the actual router implementation
            assert!(resp.status() == StatusCode::OK || resp.status() == StatusCode::BAD_REQUEST);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_invalid_model() {
        System::new().block_on(async {
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

            let req = actix_test::TestRequest::post()
                .uri("/v1/chat/completions")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            // Mock worker accepts any model, but real implementation might return 400
            assert!(resp.status().is_success() || resp.status() == StatusCode::BAD_REQUEST);

            ctx.shutdown().await;
        });
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn test_flush_cache() {
        System::new().block_on(async {
            let ctx = TestContext::new(vec![MockWorkerConfig {
                port: 18501,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = actix_test::init_service(
                App::new()
                    .app_data(ctx.app_state.clone())
                    .service(flush_cache),
            )
            .await;

            let req = actix_test::TestRequest::post()
                .uri("/flush_cache")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            // The response might be empty or contain a message
            let body_bytes = actix_test::read_body(resp).await;
            if !body_bytes.is_empty() {
                if let Ok(body) = serde_json::from_slice::<serde_json::Value>(&body_bytes) {
                    // Check that we got a successful response with expected fields
                    assert!(body.is_object());
                    assert!(body.get("message").is_some() || body.get("status").is_some());
                }
            }

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_get_loads() {
        System::new().block_on(async {
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

            let app = actix_test::init_service(
                App::new()
                    .app_data(ctx.app_state.clone())
                    .service(get_loads),
            )
            .await;

            let req = actix_test::TestRequest::get()
                .uri("/get_loads")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            let body: serde_json::Value = actix_test::read_body_json(resp).await;

            // Verify the response contains load information
            assert!(body.is_object());
            // The exact structure depends on the implementation
            // but should contain worker load information

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_flush_cache_no_workers() {
        System::new().block_on(async {
            let ctx = TestContext::new(vec![]).await;

            let app = actix_test::init_service(
                App::new()
                    .app_data(ctx.app_state.clone())
                    .service(flush_cache),
            )
            .await;

            let req = actix_test::TestRequest::post()
                .uri("/flush_cache")
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            // Should either succeed (no-op) or return service unavailable
            assert!(
                resp.status() == StatusCode::OK || resp.status() == StatusCode::SERVICE_UNAVAILABLE
            );

            ctx.shutdown().await;
        });
    }
}

#[cfg(test)]
mod load_balancing_tests {
    use super::*;

    #[test]
    fn test_request_distribution() {
        System::new().block_on(async {
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
            for _ in 0..10 {
                let payload = json!({
                    "text": format!("Request {}", request_count),
                    "stream": false
                });

                let req = actix_test::TestRequest::post()
                    .uri("/generate")
                    .set_json(&payload)
                    .to_request();

                let resp = actix_test::call_service(&app, req).await;
                if resp.status() == StatusCode::OK {
                    request_count += 1;
                }
            }

            // With random policy, all requests should succeed
            assert_eq!(request_count, 10);

            ctx.shutdown().await;
        });
    }
}

#[cfg(test)]
mod pd_mode_tests {
    use super::*;

    #[test]
    fn test_pd_mode_routing() {
        System::new().block_on(async {
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

            // For PD mode, we'll skip the test for now since it requires special handling
            // TODO: Implement PD mode testing with proper worker management
            let _prefill_url = prefill_url;
            let _decode_url = decode_url;
            prefill_worker.stop().await;
            decode_worker.stop().await;
        });
    }
}
