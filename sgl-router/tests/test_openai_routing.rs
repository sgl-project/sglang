//! Comprehensive integration tests for OpenAI backend functionality

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use axum::{
    body::Body,
    extract::Request,
    http::{Method, StatusCode},
    response::Response,
    routing::post,
    Json, Router,
};
use serde_json::json;
use sglang_router_rs::{
    config::{
        ConfigError, ConfigValidator, HistoryBackend, OracleConfig, RouterConfig, RoutingMode,
    },
    data_connector::{
        MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
        ResponseId, ResponseStorage, StoredResponse,
    },
    protocols::{
        chat::{ChatCompletionRequest, ChatMessage, UserMessageContent},
        common::StringOrArray,
        completion::CompletionRequest,
        generate::GenerateRequest,
        responses::{ResponseInput, ResponsesGetParams, ResponsesRequest},
    },
    routers::{openai::OpenAIRouter, RouterTrait},
};
use tokio::{
    net::TcpListener,
    time::{sleep, Duration},
};
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
        prompt: StringOrArray::String("Hello".to_string()),
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
        sampling_seed: None,
        other: serde_json::Map::new(),
    }
}

/// Test basic OpenAI router creation and configuration
#[tokio::test]
async fn test_openai_router_creation() {
    let router = OpenAIRouter::new(
        vec!["https://api.openai.com".to_string()],
        None,
        Arc::new(MemoryResponseStorage::new()),
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
    )
    .await;

    assert!(router.is_ok(), "Router creation should succeed");

    let router = router.unwrap();
    assert_eq!(router.router_type(), "openai");
    assert!(!router.is_pd_mode());
}

/// Test server info endpoint
#[tokio::test]
async fn test_openai_router_server_info() {
    let router = OpenAIRouter::new(
        vec!["https://api.openai.com".to_string()],
        None,
        Arc::new(MemoryResponseStorage::new()),
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
    )
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
    let router = OpenAIRouter::new(
        vec![mock_server.base_url()],
        None,
        Arc::new(MemoryResponseStorage::new()),
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
    )
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

#[tokio::test]
async fn test_openai_router_responses_with_mock() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    let app = Router::new().route(
        "/v1/responses",
        post({
            move |Json(request): Json<serde_json::Value>| {
                let counter = counter_clone.clone();
                async move {
                    let idx = counter.fetch_add(1, Ordering::SeqCst) + 1;
                    let model = request
                        .get("model")
                        .and_then(|v| v.as_str())
                        .unwrap_or("gpt-4o-mini")
                        .to_string();
                    let id = format!("resp_mock_{idx}");
                    let response = json!({
                        "id": id,
                        "object": "response",
                        "created_at": 1_700_000_000 + idx as i64,
                        "status": "completed",
                        "model": model,
                        "output": [{
                            "type": "message",
                            "id": format!("msg_{idx}"),
                            "role": "assistant",
                            "status": "completed",
                            "content": [{
                                "type": "output_text",
                                "text": format!("mock_output_{idx}"),
                                "annotations": []
                            }]
                        }],
                        "metadata": {}
                    });
                    Json(response)
                }
            }
        }),
    );

    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let base_url = format!("http://{}", addr);
    let storage = Arc::new(MemoryResponseStorage::new());

    let router = OpenAIRouter::new(
        vec![base_url],
        None,
        storage.clone(),
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
    )
    .await
    .unwrap();

    let request1 = ResponsesRequest {
        model: "gpt-4o-mini".to_string(),
        input: ResponseInput::Text("Say hi".to_string()),
        store: Some(true),
        ..Default::default()
    };

    let response1 = router.route_responses(None, &request1, None).await;
    assert_eq!(response1.status(), StatusCode::OK);
    let body1_bytes = axum::body::to_bytes(response1.into_body(), usize::MAX)
        .await
        .unwrap();
    let body1: serde_json::Value = serde_json::from_slice(&body1_bytes).unwrap();
    let resp1_id = body1["id"].as_str().expect("id missing").to_string();
    assert_eq!(body1["previous_response_id"], serde_json::Value::Null);

    let request2 = ResponsesRequest {
        model: "gpt-4o-mini".to_string(),
        input: ResponseInput::Text("Thanks".to_string()),
        store: Some(true),
        previous_response_id: Some(resp1_id.clone()),
        ..Default::default()
    };

    let response2 = router.route_responses(None, &request2, None).await;
    assert_eq!(response2.status(), StatusCode::OK);
    let body2_bytes = axum::body::to_bytes(response2.into_body(), usize::MAX)
        .await
        .unwrap();
    let body2: serde_json::Value = serde_json::from_slice(&body2_bytes).unwrap();
    let resp2_id = body2["id"].as_str().expect("second id missing");
    assert_eq!(
        body2["previous_response_id"].as_str(),
        Some(resp1_id.as_str())
    );

    let stored1 = storage
        .get_response(&ResponseId::from(resp1_id.clone()))
        .await
        .unwrap()
        .expect("first response missing");
    assert_eq!(stored1.input, "Say hi");
    assert_eq!(stored1.output, "mock_output_1");
    assert!(stored1.previous_response_id.is_none());

    let stored2 = storage
        .get_response(&ResponseId::from(resp2_id))
        .await
        .unwrap()
        .expect("second response missing");
    assert_eq!(stored2.previous_response_id.unwrap().0, resp1_id);
    assert_eq!(stored2.output, "mock_output_2");

    let get1 = router
        .get_response(None, &stored1.id.0, &ResponsesGetParams::default())
        .await;
    assert_eq!(get1.status(), StatusCode::OK);
    let get1_body_bytes = axum::body::to_bytes(get1.into_body(), usize::MAX)
        .await
        .unwrap();
    let get1_json: serde_json::Value = serde_json::from_slice(&get1_body_bytes).unwrap();
    assert_eq!(get1_json, body1);

    let get2 = router
        .get_response(None, &stored2.id.0, &ResponsesGetParams::default())
        .await;
    assert_eq!(get2.status(), StatusCode::OK);
    let get2_body_bytes = axum::body::to_bytes(get2.into_body(), usize::MAX)
        .await
        .unwrap();
    let get2_json: serde_json::Value = serde_json::from_slice(&get2_body_bytes).unwrap();
    assert_eq!(get2_json, body2);

    server.abort();
}

#[tokio::test]
async fn test_openai_router_responses_streaming_with_mock() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let sse_handler = post(|Json(_request): Json<serde_json::Value>| async move {
        let response_id = "resp_stream_123";
        let message_id = "msg_stream_123";
        let final_text = "Once upon a streamed unicorn adventure.";

        let events = vec![
            (
                "response.created",
                json!({
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": 1_700_000_500,
                        "status": "in_progress",
                        "model": "",
                        "output": [],
                        "parallel_tool_calls": true,
                        "previous_response_id": null,
                        "reasoning": null,
                        "store": false,
                        "temperature": 1.0,
                        "text": {"format": {"type": "text"}},
                        "tool_choice": "auto",
                        "tools": [],
                        "top_p": 1.0,
                        "truncation": "disabled",
                        "usage": null,
                        "metadata": null
                    }
                }),
            ),
            (
                "response.output_item.added",
                json!({
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": []
                    }
                }),
            ),
            (
                "response.output_text.delta",
                json!({
                    "type": "response.output_text.delta",
                    "sequence_number": 2,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "Once upon a streamed unicorn adventure.",
                    "logprobs": []
                }),
            ),
            (
                "response.output_text.done",
                json!({
                    "type": "response.output_text.done",
                    "sequence_number": 3,
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": final_text,
                    "logprobs": []
                }),
            ),
            (
                "response.output_item.done",
                json!({
                    "type": "response.output_item.done",
                    "sequence_number": 4,
                    "output_index": 0,
                    "item": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": final_text,
                            "annotations": [],
                            "logprobs": []
                        }]
                    }
                }),
            ),
            (
                "response.completed",
                json!({
                    "type": "response.completed",
                    "sequence_number": 5,
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": 1_700_000_500,
                        "status": "completed",
                        "model": "",
                        "output": [{
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{
                                "type": "output_text",
                                "text": final_text,
                                "annotations": [],
                                "logprobs": []
                            }]
                        }],
                        "parallel_tool_calls": true,
                        "previous_response_id": null,
                        "reasoning": null,
                        "store": false,
                        "temperature": 1.0,
                        "text": {"format": {"type": "text"}},
                        "tool_choice": "auto",
                        "tools": [],
                        "top_p": 1.0,
                        "truncation": "disabled",
                        "usage": {
                            "input_tokens": 10,
                            "input_tokens_details": {"cached_tokens": 0},
                            "output_tokens": 20,
                            "output_tokens_details": {"reasoning_tokens": 5},
                            "total_tokens": 30
                        },
                        "metadata": null,
                        "instructions": null,
                        "user": null
                    }
                }),
            ),
        ];

        let sse_payload = events
            .into_iter()
            .map(|(event, data)| format!("event: {}\ndata: {}\n\n", event, data))
            .collect::<String>();

        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(Body::from(sse_payload))
            .unwrap()
    });

    let app = Router::new().route("/v1/responses", sse_handler);

    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let base_url = format!("http://{}", addr);
    let storage = Arc::new(MemoryResponseStorage::new());

    // Seed a previous response so previous_response_id logic has data to pull from.
    let mut previous = StoredResponse::new(
        "Earlier bedtime question".to_string(),
        "Earlier answer".to_string(),
        None,
    );
    previous.id = ResponseId::from("resp_prev_chain");
    storage.store_response(previous).await.unwrap();

    let router = OpenAIRouter::new(
        vec![base_url],
        None,
        storage.clone(),
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
    )
    .await
    .unwrap();

    let mut metadata = HashMap::new();
    metadata.insert("topic".to_string(), json!("unicorns"));

    let request = ResponsesRequest {
        model: "gpt-5-nano".to_string(),
        input: ResponseInput::Text("Tell me a bedtime story.".to_string()),
        instructions: Some("Be kind".to_string()),
        metadata: Some(metadata),
        previous_response_id: Some("resp_prev_chain".to_string()),
        store: Some(true),
        stream: Some(true),
        ..Default::default()
    };

    let response = router.route_responses(None, &request, None).await;
    assert_eq!(response.status(), StatusCode::OK);

    let headers = response.headers();
    let ct = headers
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_ascii_lowercase();
    assert!(ct.contains("text/event-stream"));

    let response_body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_text = String::from_utf8(response_body.to_vec()).unwrap();
    assert!(body_text.contains("response.completed"));
    assert!(body_text.contains("Once upon a streamed unicorn adventure."));

    // Wait for the storage task to persist the streaming response.
    let target_id = ResponseId::from("resp_stream_123");
    let stored = loop {
        if let Some(resp) = storage.get_response(&target_id).await.unwrap() {
            break resp;
        }
        sleep(Duration::from_millis(10)).await;
    };

    assert_eq!(stored.input, "Tell me a bedtime story.");
    assert_eq!(stored.output, "Once upon a streamed unicorn adventure.");
    assert_eq!(
        stored
            .previous_response_id
            .as_ref()
            .expect("previous_response_id missing")
            .0,
        "resp_prev_chain"
    );
    assert_eq!(stored.metadata.get("topic"), Some(&json!("unicorns")));
    assert_eq!(stored.instructions.as_deref(), Some("Be kind"));
    assert_eq!(stored.model.as_deref(), Some("gpt-5-nano"));
    assert_eq!(stored.user, None);
    assert_eq!(stored.raw_response["store"], json!(true));
    assert_eq!(
        stored.raw_response["previous_response_id"].as_str(),
        Some("resp_prev_chain")
    );
    assert_eq!(stored.raw_response["metadata"]["topic"], json!("unicorns"));
    assert_eq!(
        stored.raw_response["instructions"].as_str(),
        Some("Be kind")
    );

    server.abort();
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
    let router = OpenAIRouter::new(
        vec!["https://api.openai.com".to_string()],
        None,
        Arc::new(MemoryResponseStorage::new()),
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
    )
    .await
    .unwrap();

    let generate_request = GenerateRequest {
        text: Some("Hello world".to_string()),
        input_ids: None,
        input_embeds: None,
        image_data: None,
        video_data: None,
        audio_data: None,
        sampling_params: None,
        return_logprob: Some(false),
        logprob_start_len: None,
        top_logprobs_num: None,
        token_ids_logprob: None,
        return_text_in_logprobs: false,
        stream: false,
        log_metrics: true,
        return_hidden_states: false,
        modalities: None,
        session_params: None,
        lora_path: None,
        lora_id: None,
        custom_logit_processor: None,
        bootstrap_host: None,
        bootstrap_port: None,
        bootstrap_room: None,
        bootstrap_pair_key: None,
        data_parallel_rank: None,
        background: false,
        conversation_id: None,
        priority: None,
        extra_key: None,
        no_logs: false,
        custom_labels: None,
        return_bytes: false,
        return_entropy: false,
        rid: None,
    };

    let response = router.route_generate(None, &generate_request, None).await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);

    let completion_request = create_minimal_completion_request();
    let response = router
        .route_completion(None, &completion_request, None)
        .await;
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
}

/// Test chat completion with mock OpenAI server
#[tokio::test]
async fn test_openai_router_chat_completion_with_mock() {
    // Start a mock OpenAI server
    let mock_server = MockOpenAIServer::new().await;
    let base_url = mock_server.base_url();

    // Create router pointing to mock server
    let router = OpenAIRouter::new(
        vec![base_url],
        None,
        Arc::new(MemoryResponseStorage::new()),
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
    )
    .await
    .unwrap();

    // Create a minimal chat completion request
    let mut chat_request = create_minimal_chat_request();
    chat_request.messages = vec![ChatMessage::User {
        content: UserMessageContent::Text("Hello, how are you?".to_string()),
        name: None,
    }];
    chat_request.temperature = Some(0.7);

    // Route the request
    let response = router.route_chat(None, &chat_request, None).await;

    // Should get a successful response from mock server
    assert_eq!(response.status(), StatusCode::OK);

    let (_, body) = response.into_parts();
    let body_bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
    let body_str = String::from_utf8(body_bytes.to_vec()).unwrap();
    let chat_response: serde_json::Value = serde_json::from_str(&body_str).unwrap();

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
    let router = OpenAIRouter::new(
        vec![base_url],
        None,
        Arc::new(MemoryResponseStorage::new()),
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
    )
    .await
    .unwrap();

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

                    router
                        .route_chat(Some(&parts.headers), &chat_request, None)
                        .await
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

    assert_eq!(response_json["object"], "chat.completion");
    assert_eq!(response_json["model"], "gpt-3.5-turbo");
    assert!(!response_json["choices"].as_array().unwrap().is_empty());
}

/// Test streaming chat completions pass-through with mock server
#[tokio::test]
async fn test_openai_router_chat_streaming_with_mock() {
    let mock_server = MockOpenAIServer::new().await;
    let base_url = mock_server.base_url();
    let router = OpenAIRouter::new(
        vec![base_url],
        None,
        Arc::new(MemoryResponseStorage::new()),
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
    )
    .await
    .unwrap();

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

    let response = router.route_chat(None, &chat_request, None).await;
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
        vec!["http://invalid-url-that-will-fail".to_string()],
        Some(cb_config),
        Arc::new(MemoryResponseStorage::new()),
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
    )
    .await
    .unwrap();

    let chat_request = create_minimal_chat_request();

    // First few requests should fail and record failures
    for _ in 0..3 {
        let response = router.route_chat(None, &chat_request, None).await;
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
    let router = OpenAIRouter::new(
        vec![mock_server.base_url()],
        None,
        Arc::new(MemoryResponseStorage::new()),
        Arc::new(MemoryConversationStorage::new()),
        Arc::new(MemoryConversationItemStorage::new()),
    )
    .await
    .unwrap();

    // 1) Without auth header -> expect 200 with empty model list
    // (multi-endpoint aggregation silently skips failed endpoints)
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
    assert_eq!(models["data"].as_array().unwrap().len(), 0); // Empty when auth fails

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

#[test]
fn oracle_config_validation_requires_config_when_enabled() {
    let config = RouterConfig::builder()
        .openai_mode(vec!["https://api.openai.com".to_string()])
        .history_backend(HistoryBackend::Oracle)
        .build_unchecked();

    let err =
        ConfigValidator::validate(&config).expect_err("config should fail without oracle details");

    match err {
        ConfigError::MissingRequired { field } => {
            assert_eq!(field, "oracle");
        }
        other => panic!("unexpected error: {:?}", other),
    }
}

#[test]
fn oracle_config_validation_accepts_dsn_only() {
    let config = RouterConfig::builder()
        .openai_mode(vec!["https://api.openai.com".to_string()])
        .oracle_history(OracleConfig {
            wallet_path: None,
            connect_descriptor: "tcps://db.example.com:1522/service".to_string(),
            username: "scott".to_string(),
            password: "tiger".to_string(),
            pool_min: 1,
            pool_max: 4,
            pool_timeout_secs: 30,
        })
        .build_unchecked();

    ConfigValidator::validate(&config).expect("dsn-based config should validate");
}

#[test]
fn oracle_config_validation_accepts_wallet_alias() {
    let config = RouterConfig::builder()
        .openai_mode(vec!["https://api.openai.com".to_string()])
        .oracle_history(OracleConfig {
            wallet_path: Some("/etc/sglang/oracle-wallet".to_string()),
            connect_descriptor: "db_low".to_string(),
            username: "app_user".to_string(),
            password: "secret".to_string(),
            pool_min: 1,
            pool_max: 8,
            pool_timeout_secs: 45,
        })
        .build_unchecked();

    ConfigValidator::validate(&config).expect("wallet-based config should validate");
}
