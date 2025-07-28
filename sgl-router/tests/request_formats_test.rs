mod common;

use actix_web::{http::StatusCode, rt::System, test as actix_test, web, App};
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use reqwest::Client;
use serde_json::json;
use sglang_router_rs::config::{PolicyConfig, RouterConfig, RoutingMode};
use sglang_router_rs::server::{
    add_worker, generate, v1_chat_completions, v1_completions, AppState,
};

/// Test context for request type testing
struct RequestTestContext {
    workers: Vec<MockWorker>,
    app_state: web::Data<AppState>,
}

impl RequestTestContext {
    async fn new(worker_configs: Vec<MockWorkerConfig>) -> Self {
        let mut workers = Vec::new();
        let mut worker_urls = Vec::new();

        // Start mock workers
        for config in worker_configs {
            let mut worker = MockWorker::new(config);
            let url = worker.start().await.unwrap();
            worker_urls.push(url);
            workers.push(worker);
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // Create router config
        let config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec![],
            },
            policy: PolicyConfig::Random,
            host: "127.0.0.1".to_string(),
            port: 3006,
            max_payload_size: 256 * 1024 * 1024,
            request_timeout_secs: 600,
            worker_startup_timeout_secs: 1,
            worker_startup_check_interval_secs: 1,
            discovery: None,
            metrics: None,
            log_dir: None,
            log_level: None,
        };

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.request_timeout_secs))
            .build()
            .unwrap();

        let app_state = AppState::new(config, client).unwrap();
        let app_state = web::Data::new(app_state);

        // Add workers via HTTP API
        let app =
            actix_test::init_service(App::new().app_data(app_state.clone()).service(add_worker))
                .await;

        for url in &worker_urls {
            let req = actix_test::TestRequest::post()
                .uri(&format!("/add_worker?url={}", url))
                .to_request();
            let resp = actix_test::call_service(&app, req).await;
            assert!(resp.status().is_success());
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

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
                .service(generate)
                .service(v1_chat_completions)
                .service(v1_completions),
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
mod generate_input_format_tests {
    use super::*;

    #[test]
    fn test_generate_with_text_input() {
        System::new().block_on(async {
            let ctx = RequestTestContext::new(vec![MockWorkerConfig {
                port: 21001,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            // Standard text input
            let payload = json!({
                "text": "Hello world",
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

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_generate_with_prompt_input() {
        System::new().block_on(async {
            let ctx = RequestTestContext::new(vec![MockWorkerConfig {
                port: 21002,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            // Prompt input (alternative to text)
            let payload = json!({
                "prompt": "Once upon a time",
                "stream": false
            });

            let req = actix_test::TestRequest::post()
                .uri("/generate")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_generate_with_input_ids() {
        System::new().block_on(async {
            let ctx = RequestTestContext::new(vec![MockWorkerConfig {
                port: 21003,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            // Input IDs (tokenized input)
            let payload = json!({
                "input_ids": [1, 2, 3, 4, 5],
                "stream": false
            });

            let req = actix_test::TestRequest::post()
                .uri("/generate")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_generate_with_all_parameters() {
        System::new().block_on(async {
            let ctx = RequestTestContext::new(vec![MockWorkerConfig {
                port: 21004,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            // All generation parameters
            let payload = json!({
                "text": "Complete this",
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "max_new_tokens": 100,
                "min_new_tokens": 10,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
                "repetition_penalty": 1.1,
                "stop": [".", "!", "?"],
                "stream": false
            });

            let req = actix_test::TestRequest::post()
                .uri("/generate")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            ctx.shutdown().await;
        });
    }
}

#[cfg(test)]
mod chat_completion_format_tests {
    use super::*;

    #[test]
    fn test_chat_with_system_message() {
        System::new().block_on(async {
            let ctx = RequestTestContext::new(vec![MockWorkerConfig {
                port: 21010,
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
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ]
            });

            let req = actix_test::TestRequest::post()
                .uri("/v1/chat/completions")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            ctx.shutdown().await;
        });
    }

    // Note: Function calling and tools tests are commented out because
    // they require special handling in the mock worker that's not implemented yet.
    // In production, these would be forwarded to the actual model.

    // #[test]
    // fn test_chat_with_function_calling() {
    //     // Test would go here when mock worker supports function calling
    // }

    // #[test]
    // fn test_chat_with_tools() {
    //     // Test would go here when mock worker supports tools
    // }

    #[test]
    fn test_chat_with_response_format() {
        System::new().block_on(async {
            let ctx = RequestTestContext::new(vec![MockWorkerConfig {
                port: 21013,
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
                    {"role": "user", "content": "Return JSON"}
                ],
                "response_format": {
                    "type": "json_object"
                }
            });

            let req = actix_test::TestRequest::post()
                .uri("/v1/chat/completions")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            ctx.shutdown().await;
        });
    }
}

#[cfg(test)]
mod completion_format_tests {
    use super::*;

    #[test]
    fn test_completion_with_single_prompt() {
        System::new().block_on(async {
            let ctx = RequestTestContext::new(vec![MockWorkerConfig {
                port: 21020,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            let payload = json!({
                "model": "test-model",
                "prompt": "Once upon a time",
                "max_tokens": 50
            });

            let req = actix_test::TestRequest::post()
                .uri("/v1/completions")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            let body: serde_json::Value = actix_test::read_body_json(resp).await;
            assert!(body.get("choices").is_some());

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_completion_with_batch_prompts() {
        System::new().block_on(async {
            let ctx = RequestTestContext::new(vec![MockWorkerConfig {
                port: 21021,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            let payload = json!({
                "model": "test-model",
                "prompt": ["First prompt", "Second prompt", "Third prompt"],
                "max_tokens": 30
            });

            let req = actix_test::TestRequest::post()
                .uri("/v1/completions")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_completion_with_echo() {
        System::new().block_on(async {
            let ctx = RequestTestContext::new(vec![MockWorkerConfig {
                port: 21022,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            let payload = json!({
                "model": "test-model",
                "prompt": "Echo this prompt",
                "echo": true,
                "max_tokens": 20
            });

            let req = actix_test::TestRequest::post()
                .uri("/v1/completions")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_completion_with_logprobs() {
        System::new().block_on(async {
            let ctx = RequestTestContext::new(vec![MockWorkerConfig {
                port: 21023,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            let payload = json!({
                "model": "test-model",
                "prompt": "Calculate probability",
                "logprobs": 5,
                "max_tokens": 10
            });

            let req = actix_test::TestRequest::post()
                .uri("/v1/completions")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_completion_with_suffix() {
        System::new().block_on(async {
            let ctx = RequestTestContext::new(vec![MockWorkerConfig {
                port: 21024,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            let payload = json!({
                "model": "test-model",
                "prompt": "Insert text here: ",
                "suffix": " and continue from here.",
                "max_tokens": 20
            });

            let req = actix_test::TestRequest::post()
                .uri("/v1/completions")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            ctx.shutdown().await;
        });
    }
}

#[cfg(test)]
mod stop_sequence_tests {
    use super::*;

    #[test]
    fn test_stop_sequences_array() {
        System::new().block_on(async {
            let ctx = RequestTestContext::new(vec![MockWorkerConfig {
                port: 21030,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            let payload = json!({
                "text": "Generate until stop",
                "stop": [".", "!", "?", "\n"],
                "stream": false
            });

            let req = actix_test::TestRequest::post()
                .uri("/generate")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            ctx.shutdown().await;
        });
    }

    #[test]
    fn test_stop_sequences_string() {
        System::new().block_on(async {
            let ctx = RequestTestContext::new(vec![MockWorkerConfig {
                port: 21031,
                worker_type: WorkerType::Regular,
                health_status: HealthStatus::Healthy,
                response_delay_ms: 0,
                fail_rate: 0.0,
            }])
            .await;

            let app = ctx.create_app().await;

            let payload = json!({
                "text": "Generate until stop",
                "stop": "\n\n",
                "stream": false
            });

            let req = actix_test::TestRequest::post()
                .uri("/generate")
                .set_json(&payload)
                .to_request();

            let resp = actix_test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);

            ctx.shutdown().await;
        });
    }
}
