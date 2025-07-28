use actix_web::{middleware, web, App, HttpRequest, HttpResponse, HttpServer};
use futures_util::StreamExt;
use serde_json::json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use uuid;

/// Configuration for mock worker behavior
#[derive(Clone)]
pub struct MockWorkerConfig {
    pub port: u16,
    pub worker_type: WorkerType,
    pub health_status: HealthStatus,
    pub response_delay_ms: u64,
    pub fail_rate: f32,
}

#[derive(Clone, Debug)]
pub enum WorkerType {
    Regular,
    Prefill,
    Decode,
}

#[derive(Clone, Debug)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    Degraded,
}

/// Mock worker server for testing
pub struct MockWorker {
    config: Arc<RwLock<MockWorkerConfig>>,
    server_handle: Option<actix_web::dev::ServerHandle>,
}

impl MockWorker {
    pub fn new(config: MockWorkerConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            server_handle: None,
        }
    }

    /// Start the mock worker server
    pub async fn start(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let config = self.config.clone();
        let port = config.read().await.port;

        let server = HttpServer::new(move || {
            App::new()
                .app_data(web::Data::new(config.clone()))
                .wrap(middleware::Logger::default())
                .route("/health", web::get().to(health_handler))
                .route("/health_generate", web::get().to(health_generate_handler))
                .route("/get_server_info", web::get().to(server_info_handler))
                .route("/get_model_info", web::get().to(model_info_handler))
                .route("/generate", web::post().to(generate_handler))
                .route(
                    "/v1/chat/completions",
                    web::post().to(chat_completions_handler),
                )
                .route("/v1/completions", web::post().to(completions_handler))
                .route("/flush_cache", web::post().to(flush_cache_handler))
                .route("/v1/models", web::get().to(v1_models_handler))
        })
        .bind(("127.0.0.1", port))?
        .run();

        let handle = server.handle();
        self.server_handle = Some(handle);

        tokio::spawn(server);

        Ok(format!("http://127.0.0.1:{}", port))
    }

    /// Stop the mock worker server
    pub async fn stop(&mut self) {
        if let Some(handle) = self.server_handle.take() {
            // First try graceful stop with short timeout
            handle.stop(false);
            // Give it a moment to stop gracefully
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
    }

    /// Update the mock worker configuration
    pub async fn update_config<F>(&self, updater: F)
    where
        F: FnOnce(&mut MockWorkerConfig),
    {
        let mut config = self.config.write().await;
        updater(&mut *config);
    }
}

// Handler implementations

/// Check if request should fail based on configured fail_rate
async fn should_fail(config: &MockWorkerConfig) -> bool {
    rand::random::<f32>() < config.fail_rate
}

async fn health_handler(config: web::Data<Arc<RwLock<MockWorkerConfig>>>) -> HttpResponse {
    let config = config.read().await;

    // Note: We don't apply fail_rate to health endpoint to allow workers to be added successfully
    // fail_rate is only applied to actual request endpoints

    match config.health_status {
        HealthStatus::Healthy => HttpResponse::Ok().json(json!({
            "status": "healthy",
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "worker_type": format!("{:?}", config.worker_type),
        })),
        HealthStatus::Unhealthy => HttpResponse::ServiceUnavailable().json(json!({
            "status": "unhealthy",
            "error": "Worker is not responding"
        })),
        HealthStatus::Degraded => HttpResponse::Ok().json(json!({
            "status": "degraded",
            "warning": "High load detected"
        })),
    }
}

async fn health_generate_handler(config: web::Data<Arc<RwLock<MockWorkerConfig>>>) -> HttpResponse {
    let config = config.read().await;

    // Simulate failure based on fail_rate
    if should_fail(&config).await {
        return HttpResponse::InternalServerError().json(json!({
            "error": "Random failure for testing"
        }));
    }

    if matches!(config.health_status, HealthStatus::Healthy) {
        HttpResponse::Ok().json(json!({
            "status": "ok",
            "queue_length": 0,
            "processing_time_ms": config.response_delay_ms
        }))
    } else {
        HttpResponse::ServiceUnavailable().json(json!({
            "error": "Generation service unavailable"
        }))
    }
}

async fn server_info_handler(config: web::Data<Arc<RwLock<MockWorkerConfig>>>) -> HttpResponse {
    let config = config.read().await;

    // Simulate failure based on fail_rate
    if should_fail(&config).await {
        return HttpResponse::InternalServerError().json(json!({
            "error": "Random failure for testing"
        }));
    }

    // Return response matching actual sglang server implementation
    HttpResponse::Ok().json(json!({
        // Server args fields
        "model_path": "mock-model-path",
        "tokenizer_path": "mock-tokenizer-path",
        "port": config.port,
        "host": "127.0.0.1",
        "max_num_batched_tokens": 32768,
        "max_prefill_tokens": 16384,
        "mem_fraction_static": 0.88,
        "tp_size": 1,
        "dp_size": 1,
        "stream_interval": 8,
        "dtype": "float16",
        "device": "cuda",
        "enable_flashinfer": true,
        "enable_p2p_check": true,
        "context_length": 32768,
        "chat_template": null,
        "disable_radix_cache": false,
        "enable_torch_compile": false,
        "trust_remote_code": false,
        "show_time_cost": false,

        // Scheduler info fields
        "waiting_queue_size": 0,
        "running_queue_size": 0,
        "req_to_token_ratio": 1.2,
        "min_running_requests": 0,
        "max_running_requests": 2048,
        "max_req_num": 8192,
        "max_batch_tokens": 32768,
        "schedule_policy": "lpm",
        "schedule_conservativeness": 1.0,

        // Additional fields
        "version": "0.3.0",
        "internal_states": [{
            "waiting_queue_size": 0,
            "running_queue_size": 0
        }]
    }))
}

async fn model_info_handler(config: web::Data<Arc<RwLock<MockWorkerConfig>>>) -> HttpResponse {
    let config = config.read().await;

    // Simulate failure based on fail_rate
    if should_fail(&config).await {
        return HttpResponse::InternalServerError().json(json!({
            "error": "Random failure for testing"
        }));
    }

    // Return response matching actual sglang server implementation
    HttpResponse::Ok().json(json!({
        "model_path": "mock-model-path",
        "tokenizer_path": "mock-tokenizer-path",
        "is_generation": true,
        "preferred_sampling_params": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 2048
        }
    }))
}

async fn generate_handler(
    config: web::Data<Arc<RwLock<MockWorkerConfig>>>,
    _req: HttpRequest,
    payload: web::Json<serde_json::Value>,
) -> HttpResponse {
    let config = config.read().await;

    // Simulate failure based on fail_rate
    if should_fail(&config).await {
        return HttpResponse::InternalServerError().json(json!({
            "error": "Random failure for testing"
        }));
    }

    // Simulate processing delay
    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if is_stream {
        // Return streaming response matching sglang format
        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let stream_delay = config.response_delay_ms;
        let request_id = format!("mock-req-{}", rand::random::<u32>());

        tokio::spawn(async move {
            let tokens = vec!["This ", "is ", "a ", "mock ", "response."];
            let timestamp_start = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();

            for (i, token) in tokens.iter().enumerate() {
                let chunk = json!({
                    "text": token,
                    "meta_info": {
                        "id": &request_id,
                        "finish_reason": if i == tokens.len() - 1 {
                            json!({"type": "stop", "matched_stop": null})
                        } else {
                            json!(null)
                        },
                        "prompt_tokens": 10,
                        "completion_tokens": i + 1,
                        "cached_tokens": 0,
                        "e2e_latency": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64() - timestamp_start
                    }
                });

                if tx
                    .send(format!(
                        "data: {}\n\n",
                        serde_json::to_string(&chunk).unwrap()
                    ))
                    .await
                    .is_err()
                {
                    break;
                }

                if stream_delay > 0 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(stream_delay)).await;
                }
            }

            let _ = tx.send("data: [DONE]\n\n".to_string()).await;
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);

        HttpResponse::Ok()
            .content_type("text/event-stream")
            .insert_header(("Cache-Control", "no-cache"))
            .streaming(stream.map(|chunk| Ok::<_, actix_web::Error>(bytes::Bytes::from(chunk))))
    } else {
        // Return non-streaming response matching sglang format
        let request_id = format!("mock-req-{}", rand::random::<u32>());

        HttpResponse::Ok().json(json!({
            "text": "Mock generated response for the input",
            "meta_info": {
                "id": request_id,
                "finish_reason": {
                    "type": "stop",
                    "matched_stop": null
                },
                "prompt_tokens": 10,
                "completion_tokens": 7,
                "cached_tokens": 0,
                "e2e_latency": 0.042
            }
        }))
    }
}

async fn chat_completions_handler(
    config: web::Data<Arc<RwLock<MockWorkerConfig>>>,
    payload: web::Json<serde_json::Value>,
) -> HttpResponse {
    let config = config.read().await;

    // Simulate failure
    if rand::random::<f32>() < config.fail_rate {
        return HttpResponse::InternalServerError().json(json!({
            "error": "Chat completion failed"
        }));
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if is_stream {
        // Return proper streaming response for chat completions
        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let stream_delay = config.response_delay_ms;
        let model = payload
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or("mock-model")
            .to_string();

        tokio::spawn(async move {
            let chat_id = format!("chatcmpl-mock{}", rand::random::<u32>());
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // Send initial chunk with role
            let initial_chunk = json!({
                "id": &chat_id,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": &model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant"
                    },
                    "finish_reason": null
                }]
            });

            let _ = tx
                .send(format!(
                    "data: {}\n\n",
                    serde_json::to_string(&initial_chunk).unwrap()
                ))
                .await;

            // Send content chunks
            let content_chunks = [
                "This ",
                "is ",
                "a ",
                "mock ",
                "streaming ",
                "chat ",
                "response.",
            ];
            for chunk in content_chunks.iter() {
                let data = json!({
                    "id": &chat_id,
                    "object": "chat.completion.chunk",
                    "created": timestamp,
                    "model": &model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": chunk
                        },
                        "finish_reason": null
                    }]
                });

                if tx
                    .send(format!(
                        "data: {}\n\n",
                        serde_json::to_string(&data).unwrap()
                    ))
                    .await
                    .is_err()
                {
                    break;
                }

                if stream_delay > 0 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(stream_delay)).await;
                }
            }

            // Send final chunk with finish_reason
            let final_chunk = json!({
                "id": &chat_id,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": &model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            });

            let _ = tx
                .send(format!(
                    "data: {}\n\n",
                    serde_json::to_string(&final_chunk).unwrap()
                ))
                .await;
            let _ = tx.send("data: [DONE]\n\n".to_string()).await;
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);

        HttpResponse::Ok()
            .content_type("text/event-stream")
            .insert_header(("Cache-Control", "no-cache"))
            .streaming(stream.map(|chunk| Ok::<_, actix_web::Error>(bytes::Bytes::from(chunk))))
    } else {
        // Non-streaming response matching OpenAI format
        let model = payload
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or("mock-model")
            .to_string();

        HttpResponse::Ok().json(json!({
            "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            "object": "chat.completion",
            "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock chat completion response."
                },
                "logprobs": null,
                "finish_reason": "stop",
                "matched_stop": null
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
                "prompt_tokens_details": {
                    "cached_tokens": 0
                }
            }
        }))
    }
}

async fn completions_handler(
    config: web::Data<Arc<RwLock<MockWorkerConfig>>>,
    payload: web::Json<serde_json::Value>,
) -> HttpResponse {
    let config = config.read().await;

    if rand::random::<f32>() < config.fail_rate {
        return HttpResponse::InternalServerError().json(json!({
            "error": "Completion failed"
        }));
    }

    // Check if streaming is requested
    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let prompts = payload
        .get("prompt")
        .map(|p| {
            if p.is_array() {
                p.as_array().unwrap().len()
            } else {
                1
            }
        })
        .unwrap_or(1);

    if is_stream {
        // Return streaming response for completions
        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let stream_delay = config.response_delay_ms;
        let model = payload
            .get("model")
            .and_then(|m| m.as_str())
            .unwrap_or("mock-model")
            .to_string();

        tokio::spawn(async move {
            let completion_id = format!("cmpl-mock{}", rand::random::<u32>());
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            // Stream completions for each prompt
            for prompt_idx in 0..prompts {
                let prompt_suffix = format!("{} ", prompt_idx);
                let tokens = vec!["This ", "is ", "mock ", "completion ", &prompt_suffix];

                for (token_idx, token) in tokens.iter().enumerate() {
                    let data = json!({
                        "id": &completion_id,
                        "object": "text_completion",
                        "created": timestamp,
                        "model": &model,
                        "choices": [{
                            "text": token,
                            "index": prompt_idx,
                            "logprobs": null,
                            "finish_reason": if token_idx == tokens.len() - 1 { Some("stop") } else { None }
                        }]
                    });

                    if tx
                        .send(format!(
                            "data: {}\n\n",
                            serde_json::to_string(&data).unwrap()
                        ))
                        .await
                        .is_err()
                    {
                        return;
                    }

                    if stream_delay > 0 {
                        tokio::time::sleep(tokio::time::Duration::from_millis(stream_delay)).await;
                    }
                }
            }

            let _ = tx.send("data: [DONE]\n\n".to_string()).await;
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);

        HttpResponse::Ok()
            .content_type("text/event-stream")
            .insert_header(("Cache-Control", "no-cache"))
            .streaming(stream.map(|chunk| Ok::<_, actix_web::Error>(bytes::Bytes::from(chunk))))
    } else {
        // Return non-streaming response
        let mut choices = vec![];
        for i in 0..prompts {
            choices.push(json!({
                "text": format!("Mock completion {}", i),
                "index": i,
                "logprobs": null,
                "finish_reason": "stop"
            }));
        }

        HttpResponse::Ok().json(json!({
            "id": format!("cmpl-mock{}", rand::random::<u32>()),
            "object": "text_completion",
            "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "model": payload.get("model").and_then(|m| m.as_str()).unwrap_or("mock-model"),
            "choices": choices,
            "usage": {
                "prompt_tokens": 5 * prompts,
                "completion_tokens": 10 * prompts,
                "total_tokens": 15 * prompts
            }
        }))
    }
}

async fn flush_cache_handler(config: web::Data<Arc<RwLock<MockWorkerConfig>>>) -> HttpResponse {
    let config = config.read().await;

    // Simulate failure based on fail_rate
    if should_fail(&config).await {
        return HttpResponse::InternalServerError().json(json!({
            "error": "Random failure for testing"
        }));
    }

    HttpResponse::Ok().json(json!({
        "status": "success",
        "message": "Cache flushed",
        "freed_entries": 42
    }))
}

async fn v1_models_handler(config: web::Data<Arc<RwLock<MockWorkerConfig>>>) -> HttpResponse {
    let config = config.read().await;

    // Simulate failure based on fail_rate
    if should_fail(&config).await {
        return HttpResponse::InternalServerError().json(json!({
            "error": "Random failure for testing"
        }));
    }

    HttpResponse::Ok().json(json!({
        "object": "list",
        "data": [{
            "id": "mock-model-v1",
            "object": "model",
            "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "owned_by": "sglang",
            "permission": [{
                "id": "modelperm-mock",
                "object": "model_permission",
                "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                "allow_create_engine": false,
                "allow_sampling": true,
                "allow_logprobs": true,
                "allow_search_indices": false,
                "allow_view": true,
                "allow_fine_tuning": false,
                "organization": "*",
                "group": null,
                "is_blocking": false
            }],
            "root": "mock-model-v1",
            "parent": null
        }]
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_worker_lifecycle() {
        let config = MockWorkerConfig {
            port: 18080,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        };

        let mut worker = MockWorker::new(config);

        // Start the worker
        let url = worker.start().await.unwrap();
        assert_eq!(url, "http://127.0.0.1:18080");

        // Give server time to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Test health endpoint
        let client = reqwest::Client::new();
        let resp = client.get(&format!("{}/health", url)).send().await.unwrap();

        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "healthy");

        // Update config to unhealthy
        worker
            .update_config(|c| c.health_status = HealthStatus::Unhealthy)
            .await;

        // Test health again
        let resp = client.get(&format!("{}/health", url)).send().await.unwrap();

        assert_eq!(resp.status(), 503);

        // Stop the worker
        worker.stop().await;
    }
}
