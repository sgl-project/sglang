//! Mock servers for testing

#![allow(dead_code)]

use std::{net::SocketAddr, sync::Arc};

use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Response, Sse,
    },
    routing::post,
    Json, Router,
};
use futures_util::stream::{self, StreamExt};
use serde_json::json;
use tokio::net::TcpListener;

/// Mock OpenAI API server for testing
pub struct MockOpenAIServer {
    addr: SocketAddr,
    _handle: tokio::task::JoinHandle<()>,
}

#[derive(Clone)]
struct MockServerState {
    require_auth: bool,
    expected_auth: Option<String>,
}

impl MockOpenAIServer {
    /// Create and start a new mock OpenAI server
    pub async fn new() -> Self {
        Self::new_with_auth(None).await
    }

    /// Create and start a new mock OpenAI server with optional auth requirement
    pub async fn new_with_auth(expected_auth: Option<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let state = Arc::new(MockServerState {
            require_auth: expected_auth.is_some(),
            expected_auth,
        });

        let app = Router::new()
            .route("/v1/chat/completions", post(mock_chat_completions))
            .route("/v1/completions", post(mock_completions))
            .route("/v1/models", post(mock_models).get(mock_models))
            .with_state(state);

        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        // Give the server a moment to start
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        Self {
            addr,
            _handle: handle,
        }
    }

    /// Get the base URL for this mock server
    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }
}

/// Mock chat completions endpoint
async fn mock_chat_completions(req: Request<Body>) -> Response {
    let (_, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    let request: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(req) => req,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    // Extract model from request or use default (owned String to satisfy 'static in stream)
    let model: String = request
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt-3.5-turbo")
        .to_string();

    // If stream requested, return SSE
    let is_stream = request
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if is_stream {
        let created = 1677652288u64;
        // Single chunk then [DONE]
        let model_chunk = model.clone();
        let event_stream = stream::once(async move {
            let chunk = json!({
                "id": "chatcmpl-123456789",
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_chunk,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "Hello!"
                    },
                    "finish_reason": null
                }]
            });
            Ok::<_, std::convert::Infallible>(Event::default().data(chunk.to_string()))
        })
        .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }));

        Sse::new(event_stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        // Create a mock non-streaming response
        let response = json!({
            "id": "chatcmpl-123456789",
            "object": "chat.completion",
            "created": 1677652288,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm a mock OpenAI assistant. How can I help you today?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21
            }
        });

        Json(response).into_response()
    }
}

/// Mock completions endpoint (legacy)
async fn mock_completions(req: Request<Body>) -> Response {
    let (_, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    let request: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(req) => req,
        Err(_) => return StatusCode::BAD_REQUEST.into_response(),
    };

    let model = request["model"].as_str().unwrap_or("text-davinci-003");

    let response = json!({
        "id": "cmpl-123456789",
        "object": "text_completion",
        "created": 1677652288,
        "model": model,
        "choices": [{
            "text": " This is a mock completion response.",
            "index": 0,
            "logprobs": null,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12
        }
    });

    Json(response).into_response()
}

/// Mock models endpoint
async fn mock_models(State(state): State<Arc<MockServerState>>, req: Request<Body>) -> Response {
    // Optionally enforce Authorization header
    if state.require_auth {
        let auth = req
            .headers()
            .get("authorization")
            .or_else(|| req.headers().get("Authorization"))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        let auth_ok = match (&state.expected_auth, auth) {
            (Some(expected), Some(got)) => &got == expected,
            (None, Some(_)) => true,
            _ => false,
        };
        if !auth_ok {
            let mut response = Response::new(Body::from(
                json!({
                    "error": {
                        "message": "Unauthorized",
                        "type": "invalid_request_error"
                    }
                })
                .to_string(),
            ));
            *response.status_mut() = StatusCode::UNAUTHORIZED;
            response
                .headers_mut()
                .insert("WWW-Authenticate", HeaderValue::from_static("Bearer"));
            return response;
        }
    }

    let response = json!({
        "object": "list",
        "data": [
            {
                "id": "gpt-4",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai"
            },
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai"
            }
        ]
    });

    Json(response).into_response()
}
