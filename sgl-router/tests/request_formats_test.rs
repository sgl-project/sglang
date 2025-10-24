mod common;

use std::sync::Arc;

use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use reqwest::Client;
use serde_json::json;
use sglang_router_rs::{
    config::{RouterConfig, RoutingMode},
    routers::{RouterFactory, RouterTrait},
};

/// Test context that manages mock workers
struct TestContext {
    workers: Vec<MockWorker>,
    _router: Arc<dyn RouterTrait>,
    worker_urls: Vec<String>,
}

impl TestContext {
    async fn new(worker_configs: Vec<MockWorkerConfig>) -> Self {
        let mut config = RouterConfig::builder()
            .regular_mode(vec![])
            .port(3003)
            .worker_startup_timeout_secs(1)
            .worker_startup_check_interval_secs(1)
            .build_unchecked();

        let mut workers = Vec::new();
        let mut worker_urls = Vec::new();

        for worker_config in worker_configs {
            let mut worker = MockWorker::new(worker_config);
            let url = worker.start().await.unwrap();
            worker_urls.push(url);
            workers.push(worker);
        }

        if !workers.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        }

        config.mode = RoutingMode::Regular {
            worker_urls: worker_urls.clone(),
        };

        let app_context = common::create_test_context(config.clone());

        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let router = Arc::from(router);

        if !workers.is_empty() {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }

        Self {
            workers,
            _router: router,
            worker_urls: worker_urls.clone(),
        }
    }

    async fn shutdown(mut self) {
        // Small delay to ensure any pending operations complete
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        for worker in &mut self.workers {
            worker.stop().await;
        }

        // Another small delay to ensure cleanup completes
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    async fn make_request(
        &self,
        endpoint: &str,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let client = Client::new();

        // Use the first worker URL from the context
        let worker_url = self
            .worker_urls
            .first()
            .ok_or_else(|| "No workers available".to_string())?;

        let response = client
            .post(format!("{}{}", worker_url, endpoint))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("Request failed with status: {}", response.status()));
        }

        response
            .json::<serde_json::Value>()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))
    }
}

#[cfg(test)]
mod request_format_tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_request_formats() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 19001,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": "Hello, world!",
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "text": "Tell me a story",
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 100,
                "top_p": 0.9
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "input_ids": [1, 2, 3, 4, 5],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 50
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_chat_completions_formats() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 19002,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            "stream": false
        });

        let result = ctx.make_request("/v1/chat/completions", payload).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.get("choices").is_some());
        assert!(response.get("id").is_some());
        assert_eq!(
            response.get("object").and_then(|v| v.as_str()),
            Some("chat.completion")
        );

        let payload = json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Tell me a joke"}
            ],
            "temperature": 0.8,
            "max_tokens": 150,
            "top_p": 0.95,
            "stream": false
        });

        let result = ctx.make_request("/v1/chat/completions", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_completions_formats() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 19003,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "model": "test-model",
            "prompt": "Once upon a time",
            "max_tokens": 50,
            "stream": false
        });

        let result = ctx.make_request("/v1/completions", payload).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(response.get("choices").is_some());
        assert_eq!(
            response.get("object").and_then(|v| v.as_str()),
            Some("text_completion")
        );

        let payload = json!({
            "model": "test-model",
            "prompt": ["First prompt", "Second prompt"],
            "temperature": 0.5,
            "stream": false
        });

        let result = ctx.make_request("/v1/completions", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "model": "test-model",
            "prompt": "The capital of France is",
            "max_tokens": 10,
            "logprobs": 5,
            "stream": false
        });

        let result = ctx.make_request("/v1/completions", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_batch_requests() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 19004,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": ["First text", "Second text", "Third text"],
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 50
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "input_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_special_parameters() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 19005,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": "Test",
            "return_logprob": true,
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "text": "Generate JSON",
            "sampling_params": {
                "temperature": 0.0,
                "json_schema": "$$ANY$$"
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        let payload = json!({
            "text": "Continue forever",
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 100,
                "ignore_eos": true
            },
            "stream": false
        });

        let result = ctx.make_request("/generate", payload).await;
        assert!(result.is_ok());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_error_handling() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 19006,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({});

        let result = ctx.make_request("/generate", payload).await;
        // Mock worker accepts empty body
        assert!(result.is_ok());

        ctx.shutdown().await;
    }
}
