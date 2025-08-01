use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive},
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
    Router,
};
use futures_util::stream::{self, StreamExt};
use serde_json::json;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use uuid::Uuid;

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
#[allow(dead_code)]
pub enum WorkerType {
    Regular,
    Prefill,
    Decode,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum HealthStatus {
    Healthy,
    Unhealthy,
    Degraded,
}

/// Mock worker server for testing
pub struct MockWorker {
    config: Arc<RwLock<MockWorkerConfig>>,
    shutdown_handle: Option<tokio::task::JoinHandle<()>>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl MockWorker {
    pub fn new(config: MockWorkerConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            shutdown_handle: None,
            shutdown_tx: None,
        }
    }

    /// Start the mock worker server
    pub async fn start(&mut self) -> Result<String, Box<dyn std::error::Error>> {
        let config = self.config.clone();
        let port = config.read().await.port;

        // If port is 0, find an available port
        let port = if port == 0 {
            let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
            let port = listener.local_addr()?.port();
            drop(listener);
            config.write().await.port = port;
            port
        } else {
            port
        };

        let app = Router::new()
            .route("/health", get(health_handler))
            .route("/health_generate", get(health_generate_handler))
            .route("/get_server_info", get(server_info_handler))
            .route("/get_model_info", get(model_info_handler))
            .route("/generate", post(generate_handler))
            .route("/v1/chat/completions", post(chat_completions_handler))
            .route("/v1/completions", post(completions_handler))
            .route("/flush_cache", post(flush_cache_handler))
            .route("/v1/models", get(v1_models_handler))
            .with_state(config);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        self.shutdown_tx = Some(shutdown_tx);

        // Spawn the server in a separate task
        let handle = tokio::spawn(async move {
            let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Failed to bind to port {}: {}", port, e);
                    return;
                }
            };

            let server = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            });

            if let Err(e) = server.await {
                eprintln!("Server error: {}", e);
            }
        });

        self.shutdown_handle = Some(handle);

        // Wait for the server to start
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let url = format!("http://127.0.0.1:{}", port);
        Ok(url)
    }

    /// Stop the mock worker server
    pub async fn stop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }

        if let Some(handle) = self.shutdown_handle.take() {
            // Wait for the server to shut down
            let _ = tokio::time::timeout(tokio::time::Duration::from_secs(5), handle).await;
        }
    }
}

impl Drop for MockWorker {
    fn drop(&mut self) {
        // Clean shutdown when dropped
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
    }
}

// Handler implementations

/// Check if request should fail based on configured fail_rate
async fn should_fail(config: &MockWorkerConfig) -> bool {
    rand::random::<f32>() < config.fail_rate
}

async fn health_handler(State(config): State<Arc<RwLock<MockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    match config.health_status {
        HealthStatus::Healthy => Json(json!({
            "status": "healthy",
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "worker_type": format!("{:?}", config.worker_type),
        }))
        .into_response(),
        HealthStatus::Unhealthy => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "unhealthy",
                "error": "Worker is not responding"
            })),
        )
            .into_response(),
        HealthStatus::Degraded => Json(json!({
            "status": "degraded",
            "warning": "High load detected"
        }))
        .into_response(),
    }
}

async fn health_generate_handler(State(config): State<Arc<RwLock<MockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    if should_fail(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    if matches!(config.health_status, HealthStatus::Healthy) {
        Json(json!({
            "status": "ok",
            "queue_length": 0,
            "processing_time_ms": config.response_delay_ms
        }))
        .into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "Generation service unavailable"
            })),
        )
            .into_response()
    }
}

async fn server_info_handler(State(config): State<Arc<RwLock<MockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    if should_fail(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    Json(json!({
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
        "waiting_queue_size": 0,
        "running_queue_size": 0,
        "req_to_token_ratio": 1.2,
        "min_running_requests": 0,
        "max_running_requests": 2048,
        "max_req_num": 8192,
        "max_batch_tokens": 32768,
        "schedule_policy": "lpm",
        "schedule_conservativeness": 1.0,
        "version": "0.3.0",
        "internal_states": [{
            "waiting_queue_size": 0,
            "running_queue_size": 0
        }]
    }))
    .into_response()
}

async fn model_info_handler(State(config): State<Arc<RwLock<MockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    if should_fail(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    Json(json!({
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
    .into_response()
}

async fn generate_handler(
    State(config): State<Arc<RwLock<MockWorkerConfig>>>,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let config = config.read().await;

    if should_fail(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if is_stream {
        let stream_delay = config.response_delay_ms;

        // Check if it's a batch request
        let is_batch = payload.get("text").and_then(|t| t.as_array()).is_some();

        let batch_size = if is_batch {
            payload
                .get("text")
                .and_then(|t| t.as_array())
                .map(|arr| arr.len())
                .unwrap_or(1)
        } else {
            1
        };

        let mut events = Vec::new();

        // Generate events for each item in batch
        for i in 0..batch_size {
            let timestamp_start = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64();

            let data = json!({
                "text": format!("Mock response {}", i + 1),
                "meta_info": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "completion_tokens_wo_jump_forward": 5,
                    "input_token_logprobs": null,
                    "output_token_logprobs": null,
                    "first_token_latency": stream_delay as f64 / 1000.0,
                    "time_to_first_token": stream_delay as f64 / 1000.0,
                    "time_per_output_token": 0.01,
                    "end_time": timestamp_start + (stream_delay as f64 / 1000.0),
                    "start_time": timestamp_start,
                    "finish_reason": {
                        "type": "stop",
                        "reason": "length"
                    }
                },
                "stage": "mid"
            });

            events.push(Ok::<_, Infallible>(Event::default().data(data.to_string())));
        }

        // Add [DONE] event
        events.push(Ok(Event::default().data("[DONE]")));

        let stream = stream::iter(events);

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(json!({
            "text": "This is a mock response.",
            "meta_info": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "completion_tokens_wo_jump_forward": 5,
                "input_token_logprobs": null,
                "output_token_logprobs": null,
                "first_token_latency": config.response_delay_ms as f64 / 1000.0,
                "time_to_first_token": config.response_delay_ms as f64 / 1000.0,
                "time_per_output_token": 0.01,
                "finish_reason": {
                    "type": "stop",
                    "reason": "length"
                }
            }
        }))
        .into_response()
    }
}

async fn chat_completions_handler(
    State(config): State<Arc<RwLock<MockWorkerConfig>>>,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let config = config.read().await;

    if should_fail(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Random failure for testing",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            })),
        )
            .into_response();
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if is_stream {
        let request_id = format!("chatcmpl-{}", Uuid::new_v4());

        let stream = stream::once(async move {
            let chunk = json!({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": timestamp,
                "model": "mock-model",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "This is a mock chat response."
                    },
                    "finish_reason": null
                }]
            });

            Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
        })
        .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }));

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(json!({
            "id": format!("chatcmpl-{}", Uuid::new_v4()),
            "object": "chat.completion",
            "created": timestamp,
            "model": "mock-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock chat response."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
        .into_response()
    }
}

async fn completions_handler(
    State(config): State<Arc<RwLock<MockWorkerConfig>>>,
    Json(payload): Json<serde_json::Value>,
) -> Response {
    let config = config.read().await;

    if should_fail(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Random failure for testing",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            })),
        )
            .into_response();
    }

    if config.response_delay_ms > 0 {
        tokio::time::sleep(tokio::time::Duration::from_millis(config.response_delay_ms)).await;
    }

    let is_stream = payload
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if is_stream {
        let request_id = format!("cmpl-{}", Uuid::new_v4());

        let stream = stream::once(async move {
            let chunk = json!({
                "id": request_id,
                "object": "text_completion",
                "created": timestamp,
                "model": "mock-model",
                "choices": [{
                    "text": "This is a mock completion.",
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": null
                }]
            });

            Ok::<_, Infallible>(Event::default().data(chunk.to_string()))
        })
        .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }));

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(json!({
            "id": format!("cmpl-{}", Uuid::new_v4()),
            "object": "text_completion",
            "created": timestamp,
            "model": "mock-model",
            "choices": [{
                "text": "This is a mock completion.",
                "index": 0,
                "logprobs": null,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }))
        .into_response()
    }
}

async fn flush_cache_handler(State(config): State<Arc<RwLock<MockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    if should_fail(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "Random failure for testing"
            })),
        )
            .into_response();
    }

    Json(json!({
        "message": "Cache flushed successfully"
    }))
    .into_response()
}

async fn v1_models_handler(State(config): State<Arc<RwLock<MockWorkerConfig>>>) -> Response {
    let config = config.read().await;

    if should_fail(&config).await {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": {
                    "message": "Random failure for testing",
                    "type": "internal_error",
                    "code": "internal_error"
                }
            })),
        )
            .into_response();
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Json(json!({
        "object": "list",
        "data": [{
            "id": "mock-model",
            "object": "model",
            "created": timestamp,
            "owned_by": "organization-owner"
        }]
    }))
    .into_response()
}

impl Default for MockWorkerConfig {
    fn default() -> Self {
        Self {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        }
    }
}
