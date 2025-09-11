//! Comprehensive integration tests for OpenAI backend functionality

use axum::{
    body::Body,
    extract::Request,
    http::{Method, StatusCode},
    routing::post,
    Router,
};
use serde_json::json;
use sglang_router_rs::{
    config::{RouterConfig, RoutingMode},
    protocols::spec::{
        ChatCompletionRequest, ChatMessage, CompletionRequest, GenerateRequest, UserMessageContent,
    },
    routers::{openai_router::OpenAIRouter, RouterTrait},
};
use std::sync::Arc;
use tower::ServiceExt;

mod common;
use common::mock_openai_server::MockOpenAIServer;

/// Helper function to create a minimal chat completion request for testing
fn create_minimal_chat_request() -> ChatCompletionRequest {
    let val = json!({
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 100
    });
    serde_json::from_value(val).unwrap()
}

/// Helper function to create a minimal completion request for testing
fn create_minimal_completion_request() -> CompletionRequest {
    CompletionRequest {
        model: "gpt-3.5-turbo".to_string(),
        prompt: sglang_router_rs::protocols::spec::StringOrArray::String("Hello".to_string()),
        suffix: None,
        max_tokens: Some(100),
        temperature: None,
        top_p: None,
        n: None,
        stream: false,
        stream_options: None,
        logprobs: None,
        echo: false,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        best_of: None,
        logit_bias: None,
        user: None,
        seed: None,
        top_k: None,
        min_p: None,
        min_tokens: None,
        repetition_penalty: None,
        regex: None,
        ebnf: None,
        json_schema: None,
        stop_token_ids: None,
        no_stop_trim: false,
        ignore_eos: false,
        skip_special_tokens: true,
        lora_path: None,
        session_params: None,
        return_hidden_states: false,
        other: serde_json::Map::new(),
    }
}

// ============= Basic Unit Tests =============

/// Test basic OpenAI router creation and configuration
#[tokio::test]
async fn test_openai_router_creation() {
    let router = OpenAIRouter::new("https://api.openai.com".to_string(), None).await;

    assert!(router.is_ok(), "Router creation should succeed");

    let router = router.unwrap();
    assert_eq!(router.router_type(), "openai");
    assert!(!router.is_pd_mode());
}

/// Test health endpoints
#[tokio::test]
async fn test_openai_router_health() {
    let router = OpenAIRouter::new("https://api.openai.com".to_string(), None)
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::GET)
        .uri("/health")
        .body(Body::empty())
        .unwrap();

    let response = router.health(req).await;
    assert_eq!(response.status(), StatusCode::OK);
}

/// Test server info endpoint
#[tokio::test]
async fn test_openai_router_server_info() {
    let router = OpenAIRouter::new("https://api.openai.com".to_string(), None)
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::GET)
        .uri("/info")
        .body(Body::empty())
        .unwrap();

    let response = router.get_server_info(req).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_, body) = response.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();

    assert!(body_str.contains("openai"));
}

/// Test models endpoint
#[tokio::test]
async fn test_openai_router_models() {
    // Use mock server for deterministic models response
    let mock_server = MockOpenAIServer::new().await;
    let router = OpenAIRouter::new(mock_server.base_url(), None)
        .await
        .unwrap();

    let req = Request::builder()
        .method(Method::GET)
        .uri("/models")
        .body(Body::empty())
        .unwrap();

    let response = router.get_models(req).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_, body) = response.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
    let models: serde_json::Value = serde_json::from_str(&body_str).unwrap();

    assert_eq!(models["object"], "list");
    assert!(models["data"].is_array());
}

/// Test router factory with OpenAI routing mode
#[tokio::test]
async fn test_router_factory_openai_mode() {
    let routing_mode = RoutingMode::OpenAI {
        worker_urls: vec!["https://api.openai.com".to_string()],
    };

    let router_config =
        RouterConfig::new(routing_mode, sglang_router_rs::config::PolicyConfig::Random);

    let app_context = common::create_test_context(router_config);

    let router = sglang_router_rs::routers::RouterFactory::create_router(&app_context).await;
    assert!(
        router.is_ok(),
        "Router factory should create OpenAI router successfully"
    );

    let router = router.unwrap();
    assert_eq!(router.router_type(), "openai");
}

/// Test that unsupported endpoints return proper error codes
#[tokio::test]
async fn test_unsupported_endpoints() {
    let router = OpenAIRouter::new("https://api.openai.com".to_string(), None)
        .await
        .unwrap();

    // Test generate endpoint (SGLang-specific, should not be supported)
    let generate_request = GenerateRequest {
        prompt: None,
        text: Some("Hello world".to_string()),
        input_ids: None,
        parameters: None,
        sampling_params: None,
        stream: false,
        return_logprob: false,
        lora_path: None,
        session_params: None,
        return_hidden_states: false,
        rid: None,
    };

    let response = router.route_generate(None, &generate_request).await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);

    // Test completion endpoint (should also not be supported)
    let completion_request = create_minimal_completion_request();
    let response = router.route_completion(None, &completion_request).await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
}

// ============= Mock Server E2E Tests =============

/// Test chat completion with mock OpenAI server
#[tokio::test]
async fn test_openai_router_chat_completion_with_mock() {
    // Start a mock OpenAI server
    let mock_server = MockOpenAIServer::new().await;
    let base_url = mock_server.base_url();

    // Create router pointing to mock server
    let router = OpenAIRouter::new(base_url, None).await.unwrap();

    // Create a minimal chat completion request
    let mut chat_request = create_minimal_chat_request();
    chat_request.messages = vec![ChatMessage::User {
        role: "user".to_string(),
        content: UserMessageContent::Text("Hello, how are you?".to_string()),
        name: None,
    }];
    chat_request.temperature = Some(0.7);

    // Route the request
    let response = router.route_chat(None, &chat_request).await;

    // Should get a successful response from mock server
    assert_eq!(response.status(), StatusCode::OK);

    let (_, body) = response.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
    let chat_response: serde_json::Value = serde_json::from_str(&body_str).unwrap();

    // Verify it's a valid chat completion response
    assert_eq!(chat_response["object"], "chat.completion");
    assert_eq!(chat_response["model"], "gpt-3.5-turbo");
    assert!(!chat_response["choices"].as_array().unwrap().is_empty());
}

/// Test full E2E flow with Axum server
#[tokio::test]
async fn test_openai_e2e_with_server() {
    // Start mock OpenAI server
    let mock_server = MockOpenAIServer::new().await;
    let base_url = mock_server.base_url();

    // Create router
    let router = OpenAIRouter::new(base_url, None).await.unwrap();

    // Create Axum app with chat completions endpoint
    let app = Router::new().route(
        "/v1/chat/completions",
        post({
            let router = Arc::new(router);
            move |req: Request<Body>| {
                let router = router.clone();
                async move {
                    let (parts, body) = req.into_parts();
                    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
                    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();

                    let chat_request: ChatCompletionRequest =
                        serde_json::from_str(&body_str).unwrap();

                    router.route_chat(Some(&parts.headers), &chat_request).await
                }
            }
        }),
    );

    // Make a request to the server
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "gpt-3.5-turbo",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, world!"
                    }
                ],
                "max_tokens": 100
            })
            .to_string(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let response_json: serde_json::Value = serde_json::from_slice(&body).unwrap();

    // Verify the response structure
    assert_eq!(response_json["object"], "chat.completion");
    assert_eq!(response_json["model"], "gpt-3.5-turbo");
    assert!(!response_json["choices"].as_array().unwrap().is_empty());
}

/// Test streaming chat completions pass-through with mock server
#[tokio::test]
async fn test_openai_router_chat_streaming_with_mock() {
    let mock_server = MockOpenAIServer::new().await;
    let base_url = mock_server.base_url();
    let router = OpenAIRouter::new(base_url, None).await.unwrap();

    // Build a streaming chat request
    let val = json!({
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 10,
        "stream": true
    });
    let chat_request: ChatCompletionRequest = serde_json::from_value(val).unwrap();

    let response = router.route_chat(None, &chat_request).await;
    assert_eq!(response.status(), StatusCode::OK);

    // Should be SSE
    let headers = response.headers();
    let ct = headers
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_ascii_lowercase();
    assert!(ct.contains("text/event-stream"));

    // Read entire stream body and assert chunks + DONE
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("chat.completion.chunk"));
    assert!(text.contains("[DONE]"));
}

/// Test circuit breaker functionality
#[tokio::test]
async fn test_openai_router_circuit_breaker() {
    // Create router with circuit breaker config
    let cb_config = sglang_router_rs::config::CircuitBreakerConfig {
        failure_threshold: 2,
        success_threshold: 1,
        timeout_duration_secs: 1,
        window_duration_secs: 10,
    };

    let router = OpenAIRouter::new(
        "http://invalid-url-that-will-fail".to_string(),
        Some(cb_config),
    )
    .await
    .unwrap();

    let chat_request = create_minimal_chat_request();

    // First few requests should fail and record failures
    for _ in 0..3 {
        let response = router.route_chat(None, &chat_request).await;
        // Should get either an error or circuit breaker response
        assert!(
            response.status() == StatusCode::INTERNAL_SERVER_ERROR
                || response.status() == StatusCode::SERVICE_UNAVAILABLE
        );
    }
}

/// Test that Authorization header is forwarded in /v1/models
#[tokio::test]
async fn test_openai_router_models_auth_forwarding() {
    // Start a mock server that requires Authorization
    let expected_auth = "Bearer test-token".to_string();
    let mock_server = MockOpenAIServer::new_with_auth(Some(expected_auth.clone())).await;
    let router = OpenAIRouter::new(mock_server.base_url(), None)
        .await
        .unwrap();

    // 1) Without auth header -> expect 401
    let req = Request::builder()
        .method(Method::GET)
        .uri("/models")
        .body(Body::empty())
        .unwrap();

    let response = router.get_models(req).await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);

    // 2) With auth header -> expect 200
    let req = Request::builder()
        .method(Method::GET)
        .uri("/models")
        .header("Authorization", expected_auth)
        .body(Body::empty())
        .unwrap();

    let response = router.get_models(req).await;
    assert_eq!(response.status(), StatusCode::OK);

    let (_, body) = response.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
    let models: serde_json::Value = serde_json::from_str(&body_str).unwrap();
    assert_eq!(models["object"], "list");
}
