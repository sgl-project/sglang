mod common;

use std::sync::Arc;

use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use futures_util::StreamExt;
use reqwest::Client;
use serde_json::json;
use sgl_model_gateway::{
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
            .port(3004)
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

        let app_context = common::create_test_context(config.clone()).await;

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

    async fn make_streaming_request(
        &self,
        endpoint: &str,
        body: serde_json::Value,
    ) -> Result<Vec<String>, String> {
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

        // Check if it's a streaming response
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if !content_type.contains("text/event-stream") {
            return Err("Response is not a stream".to_string());
        }

        let mut stream = response.bytes_stream();
        let mut events = Vec::new();

        while let Some(chunk) = stream.next().await {
            if let Ok(bytes) = chunk {
                let text = String::from_utf8_lossy(&bytes);
                for line in text.lines() {
                    if let Some(stripped) = line.strip_prefix("data: ") {
                        events.push(stripped.to_string());
                    }
                }
            }
        }

        Ok(events)
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    #[tokio::test]
    async fn test_generate_streaming() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 20001,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 10,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": "Stream test",
            "stream": true,
            "sampling_params": {
                "temperature": 0.7,
                "max_new_tokens": 10
            }
        });

        let result = ctx.make_streaming_request("/generate", payload).await;
        assert!(result.is_ok());

        let events = result.unwrap();
        // Should have at least one data chunk and [DONE]
        assert!(events.len() >= 2);
        assert_eq!(events.last().unwrap(), "[DONE]");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_chat_completions_streaming() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 20002,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 10,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Count to 3"}
            ],
            "stream": true,
            "max_tokens": 20
        });

        let result = ctx
            .make_streaming_request("/v1/chat/completions", payload)
            .await;
        assert!(result.is_ok());

        let events = result.unwrap();
        assert!(events.len() >= 2); // At least one chunk + [DONE]

        for event in &events {
            if event != "[DONE]" {
                let parsed: Result<serde_json::Value, _> = serde_json::from_str(event);
                assert!(parsed.is_ok(), "Invalid JSON in SSE event: {}", event);

                let json = parsed.unwrap();
                assert_eq!(
                    json.get("object").and_then(|v| v.as_str()),
                    Some("chat.completion.chunk")
                );
            }
        }

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_v1_completions_streaming() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 20003,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 10,
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "model": "test-model",
            "prompt": "Once upon a time",
            "stream": true,
            "max_tokens": 15
        });

        let result = ctx.make_streaming_request("/v1/completions", payload).await;
        assert!(result.is_ok());

        let events = result.unwrap();
        assert!(events.len() >= 2); // At least one chunk + [DONE]

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_streaming_with_error() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 20004,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 1.0, // Always fail
        }])
        .await;

        let payload = json!({
            "text": "This should fail",
            "stream": true
        });

        let result = ctx.make_streaming_request("/generate", payload).await;
        // With fail_rate: 1.0, the request should fail
        assert!(result.is_err());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_streaming_timeouts() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 20005,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 100, // Slow response
            fail_rate: 0.0,
        }])
        .await;

        let payload = json!({
            "text": "Slow stream",
            "stream": true,
            "sampling_params": {
                "max_new_tokens": 5
            }
        });

        let start = std::time::Instant::now();
        let result = ctx.make_streaming_request("/generate", payload).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        let events = result.unwrap();

        // Should have received multiple chunks over time
        assert!(!events.is_empty());
        assert!(elapsed.as_millis() >= 100); // At least one delay

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_batch_streaming() {
        let ctx = TestContext::new(vec![MockWorkerConfig {
            port: 20006,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 10,
            fail_rate: 0.0,
        }])
        .await;

        // Batch request with streaming
        let payload = json!({
            "text": ["First", "Second", "Third"],
            "stream": true,
            "sampling_params": {
                "max_new_tokens": 5
            }
        });

        let result = ctx.make_streaming_request("/generate", payload).await;
        assert!(result.is_ok());

        let events = result.unwrap();
        // Should have multiple events for batch
        assert!(events.len() >= 4); // At least 3 responses + [DONE]

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_sse_format_parsing() {
        let parse_sse_chunk = |chunk: &[u8]| -> Vec<String> {
            let text = String::from_utf8_lossy(chunk);
            text.lines()
                .filter(|line| line.starts_with("data: "))
                .map(|line| line[6..].to_string())
                .collect()
        };

        let sse_data =
            b"data: {\"text\":\"Hello\"}\n\ndata: {\"text\":\" world\"}\n\ndata: [DONE]\n\n";
        let events = parse_sse_chunk(sse_data);

        assert_eq!(events.len(), 3);
        assert_eq!(events[0], "{\"text\":\"Hello\"}");
        assert_eq!(events[1], "{\"text\":\" world\"}");
        assert_eq!(events[2], "[DONE]");

        let mixed = b"event: message\ndata: {\"test\":true}\n\n: comment\ndata: [DONE]\n\n";
        let events = parse_sse_chunk(mixed);

        assert_eq!(events.len(), 2);
        assert_eq!(events[0], "{\"test\":true}");
        assert_eq!(events[1], "[DONE]");
    }
}
