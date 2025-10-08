// Integration test for Responses API

use sglang_router_rs::protocols::spec::{
    GenerationRequest, ReasoningEffort, ResponseInput, ResponseReasoningParam, ResponseStatus,
    ResponseTool, ResponseToolType, ResponsesRequest, ResponsesResponse, ServiceTier, ToolChoice,
    ToolChoiceValue, Truncation, UsageInfo,
};

mod common;
use common::mock_mcp_server::MockMCPServer;
use common::mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType};
use sglang_router_rs::config::{
    CircuitBreakerConfig, ConnectionMode, HealthCheckConfig, PolicyConfig, RetryConfig,
    RouterConfig, RoutingMode,
};
use sglang_router_rs::routers::RouterFactory;
use sglang_router_rs::server::AppContext;
use std::sync::Arc;

#[tokio::test]
async fn test_non_streaming_mcp_minimal_e2e_with_persistence() {
    // Start mock MCP server
    let mut mcp = MockMCPServer::start().await.expect("start mcp");

    // Write a temp MCP config file
    let mcp_yaml = format!(
        "servers:\n  - name: mock\n    protocol: streamable\n    url: {}\n",
        mcp.url()
    );
    let dir = tempfile::tempdir().expect("tmpdir");
    let cfg_path = dir.path().join("mcp.yaml");
    std::fs::write(&cfg_path, mcp_yaml).expect("write mcp cfg");

    // Start mock OpenAI worker
    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let worker_url = worker.start().await.expect("start worker");

    // Build router config (HTTP OpenAI mode)
    let router_cfg = RouterConfig {
        mode: RoutingMode::OpenAI {
            worker_urls: vec![worker_url],
        },
        connection_mode: ConnectionMode::Http,
        policy: PolicyConfig::Random,
        host: "127.0.0.1".to_string(),
        port: 0,
        max_payload_size: 8 * 1024 * 1024,
        request_timeout_secs: 60,
        worker_startup_timeout_secs: 5,
        worker_startup_check_interval_secs: 1,
        dp_aware: false,
        api_key: None,
        discovery: None,
        metrics: None,
        log_dir: None,
        log_level: Some("warn".to_string()),
        request_id_headers: None,
        max_concurrent_requests: 32,
        queue_size: 0,
        queue_timeout_secs: 5,
        rate_limit_tokens_per_second: None,
        cors_allowed_origins: vec![],
        retry: RetryConfig::default(),
        circuit_breaker: CircuitBreakerConfig::default(),
        disable_retries: false,
        disable_circuit_breaker: false,
        health_check: HealthCheckConfig::default(),
        enable_igw: false,
        model_path: None,
        tokenizer_path: None,
        history_backend: sglang_router_rs::config::HistoryBackend::Memory,
        oracle: None,
        reasoning_parser: None,
        tool_call_parser: None,
    };

    // Create router and context
    let ctx = AppContext::new(router_cfg, reqwest::Client::new(), 64, None).expect("ctx");
    let router = RouterFactory::create_router(&Arc::new(ctx))
        .await
        .expect("router");

    // Build a simple ResponsesRequest that will trigger the tool call
    let req = ResponsesRequest {
        background: false,
        include: None,
        input: ResponseInput::Text("search something".to_string()),
        instructions: Some("Be brief".to_string()),
        max_output_tokens: Some(64),
        max_tool_calls: None,
        metadata: None,
        model: Some("mock-model".to_string()),
        parallel_tool_calls: true,
        previous_response_id: None,
        reasoning: None,
        service_tier: sglang_router_rs::protocols::spec::ServiceTier::Auto,
        store: true,
        stream: false,
        temperature: Some(0.2),
        tool_choice: sglang_router_rs::protocols::spec::ToolChoice::default(),
        tools: vec![ResponseTool {
            r#type: ResponseToolType::Mcp,
            server_url: Some(mcp.url()),
            authorization: None,
            server_label: Some("mock".to_string()),
            server_description: None,
            require_approval: None,
            allowed_tools: None,
        }],
        top_logprobs: 0,
        top_p: None,
        truncation: sglang_router_rs::protocols::spec::Truncation::Disabled,
        user: None,
        request_id: "resp_test_mcp_e2e".to_string(),
        priority: 0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop: None,
        top_k: -1,
        min_p: 0.0,
        repetition_penalty: 1.0,
        conversation: None,
    };

    let resp = router
        .route_responses(None, &req, req.model.as_deref())
        .await;

    assert_eq!(resp.status(), axum::http::StatusCode::OK);

    let body_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .expect("Failed to read response body");
    let body_json: serde_json::Value =
        serde_json::from_slice(&body_bytes).expect("Failed to parse response JSON");

    let output = body_json
        .get("output")
        .and_then(|v| v.as_array())
        .expect("response output missing");
    assert!(!output.is_empty(), "expected at least one output item");

    // Verify mcp_list_tools item is present
    let list_tools_item = output
        .iter()
        .find(|entry| {
            entry.get("type") == Some(&serde_json::Value::String("mcp_list_tools".into()))
        })
        .expect("missing mcp_list_tools output item");

    assert_eq!(
        list_tools_item.get("server_label").and_then(|v| v.as_str()),
        Some("mock"),
        "server_label should match"
    );
    let tools_list = list_tools_item
        .get("tools")
        .and_then(|v| v.as_array())
        .expect("tools array missing in mcp_list_tools");
    assert!(
        !tools_list.is_empty(),
        "mcp_list_tools should contain at least one tool"
    );

    // Verify mcp_call item is present
    let mcp_call_item = output
        .iter()
        .find(|entry| entry.get("type") == Some(&serde_json::Value::String("mcp_call".into())))
        .expect("missing mcp_call output item");

    assert_eq!(
        mcp_call_item.get("status").and_then(|v| v.as_str()),
        Some("completed"),
        "mcp_call status should be completed"
    );
    assert_eq!(
        mcp_call_item.get("server_label").and_then(|v| v.as_str()),
        Some("mock"),
        "server_label should match"
    );
    assert!(
        mcp_call_item.get("name").is_some(),
        "mcp_call should have a tool name"
    );
    assert!(
        mcp_call_item.get("arguments").is_some(),
        "mcp_call should have arguments"
    );
    assert!(
        mcp_call_item.get("output").is_some(),
        "mcp_call should have output"
    );

    let final_text = output
        .iter()
        .rev()
        .filter_map(|entry| entry.get("content"))
        .filter_map(|content| content.as_array())
        .flat_map(|parts| parts.iter())
        .filter_map(|part| part.get("text"))
        .filter_map(|v| v.as_str())
        .next();

    if let Some(text) = final_text {
        assert_eq!(text, "Tool result consumed; here is the final answer.");
    } else {
        let call_entry = output.iter().find(|entry| {
            entry.get("type") == Some(&serde_json::Value::String("function_tool_call".into()))
        });
        assert!(call_entry.is_some(), "missing function tool call entry");
        if let Some(entry) = call_entry {
            assert_eq!(
                entry.get("status").and_then(|v| v.as_str()),
                Some("in_progress"),
                "function call should be in progress when no content is returned"
            );
        }
    }

    let tools = body_json
        .get("tools")
        .and_then(|v| v.as_array())
        .expect("tools array missing");
    assert_eq!(tools.len(), 1);
    let tool = tools.first().unwrap();
    assert_eq!(tool.get("type").and_then(|v| v.as_str()), Some("mcp"));
    assert_eq!(
        tool.get("server_label").and_then(|v| v.as_str()),
        Some("mock")
    );

    // Cleanup
    worker.stop().await;
    mcp.stop().await;
}

#[tokio::test]
async fn test_conversations_crud_basic() {
    // Router in OpenAI mode (no actual upstream calls in these tests)
    let router_cfg = RouterConfig {
        mode: RoutingMode::OpenAI {
            worker_urls: vec!["http://localhost".to_string()],
        },
        connection_mode: ConnectionMode::Http,
        policy: PolicyConfig::Random,
        host: "127.0.0.1".to_string(),
        port: 0,
        max_payload_size: 8 * 1024 * 1024,
        request_timeout_secs: 60,
        worker_startup_timeout_secs: 1,
        worker_startup_check_interval_secs: 1,
        dp_aware: false,
        api_key: None,
        discovery: None,
        metrics: None,
        log_dir: None,
        log_level: Some("warn".to_string()),
        request_id_headers: None,
        max_concurrent_requests: 8,
        queue_size: 0,
        queue_timeout_secs: 5,
        rate_limit_tokens_per_second: None,
        cors_allowed_origins: vec![],
        retry: RetryConfig::default(),
        circuit_breaker: CircuitBreakerConfig::default(),
        disable_retries: false,
        disable_circuit_breaker: false,
        health_check: HealthCheckConfig::default(),
        enable_igw: false,
        model_path: None,
        tokenizer_path: None,
        history_backend: sglang_router_rs::config::HistoryBackend::Memory,
        oracle: None,
        reasoning_parser: None,
        tool_call_parser: None,
    };

    let ctx = AppContext::new(router_cfg, reqwest::Client::new(), 8, None).expect("ctx");
    let router = RouterFactory::create_router(&Arc::new(ctx))
        .await
        .expect("router");

    // Create
    let create_body = serde_json::json!({ "metadata": { "project": "alpha" } });
    let create_resp = router.create_conversation(None, &create_body).await;
    assert_eq!(create_resp.status(), axum::http::StatusCode::OK);
    let create_bytes = axum::body::to_bytes(create_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let create_json: serde_json::Value = serde_json::from_slice(&create_bytes).unwrap();
    let conv_id = create_json["id"].as_str().expect("id missing");
    assert!(conv_id.starts_with("conv_"));
    assert_eq!(create_json["object"], "conversation");

    // Get
    let get_resp = router.get_conversation(None, conv_id).await;
    assert_eq!(get_resp.status(), axum::http::StatusCode::OK);
    let get_bytes = axum::body::to_bytes(get_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let get_json: serde_json::Value = serde_json::from_slice(&get_bytes).unwrap();
    assert_eq!(get_json["metadata"]["project"], serde_json::json!("alpha"));

    // Update (merge)
    let update_body = serde_json::json!({ "metadata": { "owner": "alice" } });
    let upd_resp = router
        .update_conversation(None, conv_id, &update_body)
        .await;
    assert_eq!(upd_resp.status(), axum::http::StatusCode::OK);
    let upd_bytes = axum::body::to_bytes(upd_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let upd_json: serde_json::Value = serde_json::from_slice(&upd_bytes).unwrap();
    assert_eq!(upd_json["metadata"]["project"], serde_json::json!("alpha"));
    assert_eq!(upd_json["metadata"]["owner"], serde_json::json!("alice"));

    // Delete
    let del_resp = router.delete_conversation(None, conv_id).await;
    assert_eq!(del_resp.status(), axum::http::StatusCode::OK);
    let del_bytes = axum::body::to_bytes(del_resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let del_json: serde_json::Value = serde_json::from_slice(&del_bytes).unwrap();
    assert_eq!(del_json["deleted"], serde_json::json!(true));

    // Get again -> 404
    let not_found = router.get_conversation(None, conv_id).await;
    assert_eq!(not_found.status(), axum::http::StatusCode::NOT_FOUND);
}

#[test]
fn test_responses_request_creation() {
    let request = ResponsesRequest {
        background: false,
        include: None,
        input: ResponseInput::Text("Hello, world!".to_string()),
        instructions: Some("Be helpful".to_string()),
        max_output_tokens: Some(100),
        max_tool_calls: None,
        metadata: None,
        model: Some("test-model".to_string()),
        parallel_tool_calls: true,
        previous_response_id: None,
        reasoning: Some(ResponseReasoningParam {
            effort: Some(ReasoningEffort::Medium),
            summary: None,
        }),
        service_tier: ServiceTier::Auto,
        store: true,
        stream: false,
        temperature: Some(0.7),
        tool_choice: ToolChoice::Value(ToolChoiceValue::Auto),
        tools: vec![ResponseTool {
            r#type: ResponseToolType::WebSearchPreview,
            ..Default::default()
        }],
        top_logprobs: 5,
        top_p: Some(0.9),
        truncation: Truncation::Disabled,
        user: Some("test-user".to_string()),
        request_id: "resp_test123".to_string(),
        priority: 0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop: None,
        top_k: -1,
        min_p: 0.0,
        repetition_penalty: 1.0,
        conversation: None,
    };

    assert!(!request.is_stream());
    assert_eq!(request.get_model(), Some("test-model"));
    let routing_text = request.extract_text_for_routing();
    assert_eq!(routing_text, "Hello, world!");
}

#[test]
fn test_sampling_params_conversion() {
    let request = ResponsesRequest {
        background: false,
        include: None,
        input: ResponseInput::Text("Test".to_string()),
        instructions: None,
        max_output_tokens: Some(50),
        max_tool_calls: None,
        metadata: None,
        model: Some("test-model".to_string()),
        parallel_tool_calls: true, // Use default true
        previous_response_id: None,
        reasoning: None,
        service_tier: ServiceTier::Auto,
        store: true, // Use default true
        stream: false,
        temperature: Some(0.8),
        tool_choice: ToolChoice::Value(ToolChoiceValue::Auto),
        tools: vec![],
        top_logprobs: 0, // Use default 0
        top_p: Some(0.95),
        truncation: Truncation::Auto,
        user: None,
        request_id: "resp_test456".to_string(),
        priority: 0,
        frequency_penalty: 0.1,
        presence_penalty: 0.2,
        stop: None,
        top_k: 10,
        min_p: 0.05,
        repetition_penalty: 1.1,
        conversation: None,
    };

    let params = request.to_sampling_params(1000, None);

    // Check that parameters are converted correctly
    assert!(params.contains_key("temperature"));
    assert!(params.contains_key("top_p"));
    assert!(params.contains_key("frequency_penalty"));
    assert!(params.contains_key("max_new_tokens"));
}

#[test]
fn test_responses_response_creation() {
    let response = ResponsesResponse::new(
        "resp_test789".to_string(),
        "test-model".to_string(),
        ResponseStatus::Completed,
    );

    assert_eq!(response.id, "resp_test789");
    assert_eq!(response.model, "test-model");
    assert!(response.is_complete());
    assert!(!response.is_in_progress());
    assert!(!response.is_failed());
}

#[test]
fn test_usage_conversion() {
    let usage_info = UsageInfo::new_with_cached(15, 25, Some(8), 3);
    let response_usage = usage_info.to_response_usage();

    assert_eq!(response_usage.input_tokens, 15);
    assert_eq!(response_usage.output_tokens, 25);
    assert_eq!(response_usage.total_tokens, 40);

    // Check details are converted correctly
    assert!(response_usage.input_tokens_details.is_some());
    assert_eq!(
        response_usage
            .input_tokens_details
            .as_ref()
            .unwrap()
            .cached_tokens,
        3
    );

    assert!(response_usage.output_tokens_details.is_some());
    assert_eq!(
        response_usage
            .output_tokens_details
            .as_ref()
            .unwrap()
            .reasoning_tokens,
        8
    );

    let back_to_usage = response_usage.to_usage_info();
    assert_eq!(back_to_usage.prompt_tokens, 15);
    assert_eq!(back_to_usage.completion_tokens, 25);
    assert_eq!(back_to_usage.reasoning_tokens, Some(8));
}

#[test]
fn test_reasoning_param_default() {
    let param = ResponseReasoningParam {
        effort: Some(ReasoningEffort::Medium),
        summary: None,
    };

    let json = serde_json::to_string(&param).unwrap();
    let parsed: ResponseReasoningParam = serde_json::from_str(&json).unwrap();

    assert!(matches!(parsed.effort, Some(ReasoningEffort::Medium)));
}

#[test]
fn test_json_serialization() {
    let request = ResponsesRequest {
        background: true,
        include: None,
        input: ResponseInput::Text("Test input".to_string()),
        instructions: Some("Test instructions".to_string()),
        max_output_tokens: Some(200),
        max_tool_calls: Some(5),
        metadata: None,
        model: Some("gpt-4".to_string()),
        parallel_tool_calls: false,
        previous_response_id: None,
        reasoning: Some(ResponseReasoningParam {
            effort: Some(ReasoningEffort::High),
            summary: None,
        }),
        service_tier: ServiceTier::Priority,
        store: false,
        stream: true,
        temperature: Some(0.9),
        tool_choice: ToolChoice::Value(ToolChoiceValue::Required),
        tools: vec![ResponseTool {
            r#type: ResponseToolType::CodeInterpreter,
            ..Default::default()
        }],
        top_logprobs: 10,
        top_p: Some(0.8),
        truncation: Truncation::Auto,
        user: Some("test_user".to_string()),
        request_id: "resp_comprehensive_test".to_string(),
        priority: 1,
        frequency_penalty: 0.3,
        presence_penalty: 0.4,
        stop: None,
        top_k: 50,
        min_p: 0.1,
        repetition_penalty: 1.2,
        conversation: None,
    };

    let json = serde_json::to_string(&request).expect("Serialization should work");
    let parsed: ResponsesRequest =
        serde_json::from_str(&json).expect("Deserialization should work");

    assert_eq!(parsed.request_id, "resp_comprehensive_test");
    assert_eq!(parsed.model, Some("gpt-4".to_string()));
    assert!(parsed.background);
    assert!(parsed.stream);
    assert_eq!(parsed.tools.len(), 1);
}

#[tokio::test]
async fn test_multi_turn_loop_with_mcp() {
    // This test verifies the multi-turn loop functionality:
    // 1. Initial request with MCP tools
    // 2. Mock worker returns function_call
    // 3. Router executes MCP tool and resumes
    // 4. Mock worker returns final answer
    // 5. Verify the complete flow worked

    // Start mock MCP server
    let mut mcp = MockMCPServer::start().await.expect("start mcp");

    // Write a temp MCP config file
    let mcp_yaml = format!(
        "servers:\n  - name: mock\n    protocol: streamable\n    url: {}\n",
        mcp.url()
    );
    let dir = tempfile::tempdir().expect("tmpdir");
    let cfg_path = dir.path().join("mcp.yaml");
    std::fs::write(&cfg_path, mcp_yaml).expect("write mcp cfg");
    std::env::set_var("SGLANG_MCP_CONFIG", cfg_path.to_str().unwrap());

    // Start mock OpenAI worker
    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let worker_url = worker.start().await.expect("start worker");

    // Build router config
    let router_cfg = RouterConfig {
        mode: RoutingMode::OpenAI {
            worker_urls: vec![worker_url],
        },
        connection_mode: ConnectionMode::Http,
        policy: PolicyConfig::Random,
        host: "127.0.0.1".to_string(),
        port: 0,
        max_payload_size: 8 * 1024 * 1024,
        request_timeout_secs: 60,
        worker_startup_timeout_secs: 5,
        worker_startup_check_interval_secs: 1,
        dp_aware: false,
        api_key: None,
        discovery: None,
        metrics: None,
        log_dir: None,
        log_level: Some("info".to_string()),
        request_id_headers: None,
        max_concurrent_requests: 32,
        queue_size: 0,
        queue_timeout_secs: 5,
        rate_limit_tokens_per_second: None,
        cors_allowed_origins: vec![],
        retry: RetryConfig::default(),
        circuit_breaker: CircuitBreakerConfig::default(),
        disable_retries: false,
        disable_circuit_breaker: false,
        health_check: HealthCheckConfig::default(),
        enable_igw: false,
        model_path: None,
        tokenizer_path: None,
        history_backend: sglang_router_rs::config::HistoryBackend::Memory,
        oracle: None,
        reasoning_parser: None,
        tool_call_parser: None,
    };

    let ctx = AppContext::new(router_cfg, reqwest::Client::new(), 64, None).expect("ctx");
    let router = RouterFactory::create_router(&Arc::new(ctx))
        .await
        .expect("router");

    // Build request with MCP tools
    let req = ResponsesRequest {
        background: false,
        include: None,
        input: ResponseInput::Text("search for SGLang".to_string()),
        instructions: Some("Be helpful".to_string()),
        max_output_tokens: Some(128),
        max_tool_calls: None, // No limit - test unlimited
        metadata: None,
        model: Some("mock-model".to_string()),
        parallel_tool_calls: true,
        previous_response_id: None,
        reasoning: None,
        service_tier: ServiceTier::Auto,
        store: true,
        stream: false,
        temperature: Some(0.7),
        tool_choice: ToolChoice::Value(ToolChoiceValue::Auto),
        tools: vec![ResponseTool {
            r#type: ResponseToolType::Mcp,
            server_url: Some(mcp.url()),
            server_label: Some("mock".to_string()),
            server_description: Some("Mock MCP server for testing".to_string()),
            require_approval: Some("never".to_string()),
            ..Default::default()
        }],
        top_logprobs: 0,
        top_p: Some(1.0),
        truncation: Truncation::Disabled,
        user: None,
        request_id: "resp_multi_turn_test".to_string(),
        priority: 0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop: None,
        top_k: 50,
        min_p: 0.0,
        repetition_penalty: 1.0,
        conversation: None,
    };

    // Execute the request (this should trigger the multi-turn loop)
    let response = router.route_responses(None, &req, None).await;

    // Check status
    assert_eq!(
        response.status(),
        axum::http::StatusCode::OK,
        "Request should succeed"
    );

    // Read the response body
    use axum::body::to_bytes;
    let response_body = response.into_body();
    let body_bytes = to_bytes(response_body, usize::MAX).await.unwrap();
    let response_json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

    println!(
        "Multi-turn response: {}",
        serde_json::to_string_pretty(&response_json).unwrap()
    );

    // Verify the response structure
    assert_eq!(response_json["object"], "response");
    assert_eq!(response_json["status"], "completed");
    // Note: mock worker generates its own ID, so we just verify it exists
    assert!(
        response_json["id"].is_string(),
        "Response should have an id"
    );

    // Check that output contains final message
    let output = response_json["output"]
        .as_array()
        .expect("output should be array");
    assert!(!output.is_empty(), "output should not be empty");

    // Find the final message with text
    let has_final_text = output.iter().any(|item| {
        item.get("type")
            .and_then(|t| t.as_str())
            .map(|t| t == "message")
            .unwrap_or(false)
            && item
                .get("content")
                .and_then(|c| c.as_array())
                .map(|arr| {
                    arr.iter().any(|part| {
                        part.get("type")
                            .and_then(|t| t.as_str())
                            .map(|t| t == "output_text")
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
    });

    assert!(has_final_text, "Should have final text output");

    // Verify tools are masked back to MCP format
    let tools = response_json["tools"]
        .as_array()
        .expect("tools should be array");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "mcp");
    assert_eq!(tools[0]["server_label"], "mock");

    // Clean up
    std::env::remove_var("SGLANG_MCP_CONFIG");
    worker.stop().await;
    mcp.stop().await;
}

#[tokio::test]
async fn test_max_tool_calls_limit() {
    // This test verifies that max_tool_calls is respected
    // Note: The mock worker returns a final answer after one tool call,
    // so with max_tool_calls=1, it completes normally (doesn't exceed the limit)

    let mut mcp = MockMCPServer::start().await.expect("start mcp");
    let mcp_yaml = format!(
        "servers:\n  - name: mock\n    protocol: streamable\n    url: {}\n",
        mcp.url()
    );
    let dir = tempfile::tempdir().expect("tmpdir");
    let cfg_path = dir.path().join("mcp.yaml");
    std::fs::write(&cfg_path, mcp_yaml).expect("write mcp cfg");
    std::env::set_var("SGLANG_MCP_CONFIG", cfg_path.to_str().unwrap());

    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let worker_url = worker.start().await.expect("start worker");

    let router_cfg = RouterConfig {
        mode: RoutingMode::OpenAI {
            worker_urls: vec![worker_url],
        },
        connection_mode: ConnectionMode::Http,
        policy: PolicyConfig::Random,
        host: "127.0.0.1".to_string(),
        port: 0,
        max_payload_size: 8 * 1024 * 1024,
        request_timeout_secs: 60,
        worker_startup_timeout_secs: 5,
        worker_startup_check_interval_secs: 1,
        dp_aware: false,
        api_key: None,
        discovery: None,
        metrics: None,
        log_dir: None,
        log_level: Some("info".to_string()),
        request_id_headers: None,
        max_concurrent_requests: 32,
        queue_size: 0,
        queue_timeout_secs: 5,
        rate_limit_tokens_per_second: None,
        cors_allowed_origins: vec![],
        retry: RetryConfig::default(),
        circuit_breaker: CircuitBreakerConfig::default(),
        disable_retries: false,
        disable_circuit_breaker: false,
        health_check: HealthCheckConfig::default(),
        enable_igw: false,
        model_path: None,
        tokenizer_path: None,
        history_backend: sglang_router_rs::config::HistoryBackend::Memory,
        oracle: None,
        reasoning_parser: None,
        tool_call_parser: None,
    };

    let ctx = AppContext::new(router_cfg, reqwest::Client::new(), 64, None).expect("ctx");
    let router = RouterFactory::create_router(&Arc::new(ctx))
        .await
        .expect("router");

    let req = ResponsesRequest {
        background: false,
        include: None,
        input: ResponseInput::Text("test max calls".to_string()),
        instructions: None,
        max_output_tokens: Some(128),
        max_tool_calls: Some(1), // Limit to 1 call
        metadata: None,
        model: Some("mock-model".to_string()),
        parallel_tool_calls: true,
        previous_response_id: None,
        reasoning: None,
        service_tier: ServiceTier::Auto,
        store: false,
        stream: false,
        temperature: Some(0.7),
        tool_choice: ToolChoice::Value(ToolChoiceValue::Auto),
        tools: vec![ResponseTool {
            r#type: ResponseToolType::Mcp,
            server_url: Some(mcp.url()),
            server_label: Some("mock".to_string()),
            ..Default::default()
        }],
        top_logprobs: 0,
        top_p: Some(1.0),
        truncation: Truncation::Disabled,
        user: None,
        request_id: "resp_max_calls_test".to_string(),
        priority: 0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop: None,
        top_k: 50,
        min_p: 0.0,
        repetition_penalty: 1.0,
        conversation: None,
    };

    let response = router.route_responses(None, &req, None).await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    use axum::body::to_bytes;
    let response_body = response.into_body();
    let body_bytes = to_bytes(response_body, usize::MAX).await.unwrap();
    let response_json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();

    println!(
        "Max calls response: {}",
        serde_json::to_string_pretty(&response_json).unwrap()
    );

    // With max_tool_calls=1, the mock returns a final answer after 1 call
    // So it completes normally without exceeding the limit
    assert_eq!(response_json["status"], "completed");

    // Verify the basic response structure
    assert!(response_json["id"].is_string());
    assert_eq!(response_json["object"], "response");

    // The response should have tools masked back to MCP format
    let tools = response_json["tools"]
        .as_array()
        .expect("tools should be array");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "mcp");

    // Note: To test actual limit exceeding, we would need a mock that keeps
    // calling tools indefinitely, which would hit max_iterations (safety limit)

    std::env::remove_var("SGLANG_MCP_CONFIG");
    worker.stop().await;
    mcp.stop().await;
}

/// Helper function to set up common test infrastructure for streaming MCP tests
/// Returns (mcp_server, worker, router, temp_dir)
async fn setup_streaming_mcp_test() -> (
    MockMCPServer,
    MockWorker,
    Box<dyn sglang_router_rs::routers::RouterTrait>,
    tempfile::TempDir,
) {
    let mcp = MockMCPServer::start().await.expect("start mcp");
    let mcp_yaml = format!(
        "servers:\n  - name: mock\n    protocol: streamable\n    url: {}\n",
        mcp.url()
    );
    let dir = tempfile::tempdir().expect("tmpdir");
    let cfg_path = dir.path().join("mcp.yaml");
    std::fs::write(&cfg_path, mcp_yaml).expect("write mcp cfg");

    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let worker_url = worker.start().await.expect("start worker");

    let router_cfg = RouterConfig {
        mode: RoutingMode::OpenAI {
            worker_urls: vec![worker_url],
        },
        connection_mode: ConnectionMode::Http,
        policy: PolicyConfig::Random,
        host: "127.0.0.1".to_string(),
        port: 0,
        max_payload_size: 8 * 1024 * 1024,
        request_timeout_secs: 60,
        worker_startup_timeout_secs: 5,
        worker_startup_check_interval_secs: 1,
        dp_aware: false,
        api_key: None,
        discovery: None,
        metrics: None,
        log_dir: None,
        log_level: Some("info".to_string()),
        request_id_headers: None,
        max_concurrent_requests: 32,
        queue_size: 0,
        queue_timeout_secs: 5,
        rate_limit_tokens_per_second: None,
        cors_allowed_origins: vec![],
        retry: RetryConfig::default(),
        circuit_breaker: CircuitBreakerConfig::default(),
        disable_retries: false,
        disable_circuit_breaker: false,
        health_check: HealthCheckConfig::default(),
        enable_igw: false,
        model_path: None,
        tokenizer_path: None,
        history_backend: sglang_router_rs::config::HistoryBackend::Memory,
        oracle: None,
        reasoning_parser: None,
        tool_call_parser: None,
    };

    let ctx = AppContext::new(router_cfg, reqwest::Client::new(), 64, None).expect("ctx");
    let router = RouterFactory::create_router(&Arc::new(ctx))
        .await
        .expect("router");

    (mcp, worker, router, dir)
}

/// Parse SSE (Server-Sent Events) stream into structured events
fn parse_sse_events(body: &str) -> Vec<(Option<String>, serde_json::Value)> {
    let mut events = Vec::new();
    let blocks: Vec<&str> = body
        .split("\n\n")
        .filter(|s| !s.trim().is_empty())
        .collect();

    for block in blocks {
        let mut event_name: Option<String> = None;
        let mut data_lines: Vec<String> = Vec::new();

        for line in block.lines() {
            if let Some(rest) = line.strip_prefix("event:") {
                event_name = Some(rest.trim().to_string());
            } else if let Some(rest) = line.strip_prefix("data:") {
                let data = rest.trim_start();
                // Skip [DONE] marker
                if data != "[DONE]" {
                    data_lines.push(data.to_string());
                }
            }
        }

        if !data_lines.is_empty() {
            let data = data_lines.join("\n");
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&data) {
                events.push((event_name, parsed));
            }
        }
    }

    events
}

#[tokio::test]
async fn test_streaming_with_mcp_tool_calls() {
    // This test verifies that streaming works with MCP tool calls:
    // 1. Initial streaming request with MCP tools
    // 2. Mock worker streams text, then function_call deltas
    // 3. Router buffers function call, executes MCP tool
    // 4. Router resumes streaming with tool results
    // 5. Mock worker streams final answer
    // 6. Verify SSE events are properly formatted

    let (mut mcp, mut worker, router, _dir) = setup_streaming_mcp_test().await;

    // Build streaming request with MCP tools
    let req = ResponsesRequest {
        background: false,
        include: None,
        input: ResponseInput::Text("search for something interesting".to_string()),
        instructions: Some("Use tools when needed".to_string()),
        max_output_tokens: Some(256),
        max_tool_calls: Some(3),
        metadata: None,
        model: Some("mock-model".to_string()),
        parallel_tool_calls: true,
        previous_response_id: None,
        reasoning: None,
        service_tier: ServiceTier::Auto,
        store: true,
        stream: true, // KEY: Enable streaming
        temperature: Some(0.7),
        tool_choice: ToolChoice::Value(ToolChoiceValue::Auto),
        tools: vec![ResponseTool {
            r#type: ResponseToolType::Mcp,
            server_url: Some(mcp.url()),
            server_label: Some("mock".to_string()),
            server_description: Some("Mock MCP for streaming test".to_string()),
            require_approval: Some("never".to_string()),
            ..Default::default()
        }],
        top_logprobs: 0,
        top_p: Some(1.0),
        truncation: Truncation::Disabled,
        user: None,
        request_id: "resp_streaming_mcp_test".to_string(),
        priority: 0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop: None,
        top_k: 50,
        min_p: 0.0,
        repetition_penalty: 1.0,
        conversation: None,
    };

    let response = router.route_responses(None, &req, None).await;

    // Verify streaming response
    assert_eq!(
        response.status(),
        axum::http::StatusCode::OK,
        "Streaming request should succeed"
    );

    // Check Content-Type is text/event-stream
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok());
    assert_eq!(
        content_type,
        Some("text/event-stream"),
        "Should have SSE content type"
    );

    // Read the streaming body
    use axum::body::to_bytes;
    let response_body = response.into_body();
    let body_bytes = to_bytes(response_body, usize::MAX).await.unwrap();
    let body_text = String::from_utf8_lossy(&body_bytes);

    println!("Streaming SSE response:\n{}", body_text);

    // Parse all SSE events into structured format
    let events = parse_sse_events(&body_text);

    assert!(!events.is_empty(), "Should have at least one SSE event");
    println!("Total parsed SSE events: {}", events.len());

    // Check for [DONE] marker
    let has_done_marker = body_text.contains("data: [DONE]");
    assert!(has_done_marker, "Stream should end with [DONE] marker");

    // Track which events we've seen
    let mut found_mcp_list_tools = false;
    let mut found_mcp_list_tools_in_progress = false;
    let mut found_mcp_list_tools_completed = false;
    let mut found_response_created = false;
    let mut found_mcp_call_added = false;
    let mut found_mcp_call_in_progress = false;
    let mut found_mcp_call_arguments = false;
    let mut found_mcp_call_arguments_done = false;
    let mut found_mcp_call_done = false;
    let mut found_response_completed = false;

    for (event_name, data) in &events {
        let event_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match event_type {
            "response.output_item.added" => {
                // Check if it's an mcp_list_tools item
                if let Some(item) = data.get("item") {
                    if item.get("type").and_then(|v| v.as_str()) == Some("mcp_list_tools") {
                        found_mcp_list_tools = true;
                        println!("✓ Found mcp_list_tools added event");

                        // Verify tools array is present (should be empty in added event)
                        assert!(
                            item.get("tools").is_some(),
                            "mcp_list_tools should have tools array"
                        );
                    } else if item.get("type").and_then(|v| v.as_str()) == Some("mcp_call") {
                        found_mcp_call_added = true;
                        println!("✓ Found mcp_call added event");

                        // Verify mcp_call has required fields
                        assert!(item.get("name").is_some(), "mcp_call should have name");
                        assert_eq!(
                            item.get("server_label").and_then(|v| v.as_str()),
                            Some("mock"),
                            "mcp_call should have server_label"
                        );
                    }
                }
            }
            "response.mcp_list_tools.in_progress" => {
                found_mcp_list_tools_in_progress = true;
                println!("✓ Found mcp_list_tools.in_progress event");

                // Verify it has output_index and item_id
                assert!(
                    data.get("output_index").is_some(),
                    "mcp_list_tools.in_progress should have output_index"
                );
                assert!(
                    data.get("item_id").is_some(),
                    "mcp_list_tools.in_progress should have item_id"
                );
            }
            "response.mcp_list_tools.completed" => {
                found_mcp_list_tools_completed = true;
                println!("✓ Found mcp_list_tools.completed event");

                // Verify it has output_index and item_id
                assert!(
                    data.get("output_index").is_some(),
                    "mcp_list_tools.completed should have output_index"
                );
                assert!(
                    data.get("item_id").is_some(),
                    "mcp_list_tools.completed should have item_id"
                );
            }
            "response.mcp_call.in_progress" => {
                found_mcp_call_in_progress = true;
                println!("✓ Found mcp_call.in_progress event");

                // Verify it has output_index and item_id
                assert!(
                    data.get("output_index").is_some(),
                    "mcp_call.in_progress should have output_index"
                );
                assert!(
                    data.get("item_id").is_some(),
                    "mcp_call.in_progress should have item_id"
                );
            }
            "response.mcp_call_arguments.delta" => {
                found_mcp_call_arguments = true;
                println!("✓ Found mcp_call_arguments.delta event");

                // Delta should include arguments payload
                assert!(
                    data.get("delta").is_some(),
                    "mcp_call_arguments.delta should include delta text"
                );
            }
            "response.mcp_call_arguments.done" => {
                found_mcp_call_arguments_done = true;
                println!("✓ Found mcp_call_arguments.done event");

                assert!(
                    data.get("arguments").is_some(),
                    "mcp_call_arguments.done should include full arguments"
                );
            }
            "response.output_item.done" => {
                if let Some(item) = data.get("item") {
                    if item.get("type").and_then(|v| v.as_str()) == Some("mcp_call") {
                        found_mcp_call_done = true;
                        println!("✓ Found mcp_call done event");

                        // Verify mcp_call.done has output
                        assert!(
                            item.get("output").is_some(),
                            "mcp_call done should have output"
                        );
                    }
                }
            }
            "response.created" => {
                found_response_created = true;
                println!("✓ Found response.created event");

                // Verify response has required fields
                assert!(
                    data.get("response").is_some(),
                    "response.created should have response object"
                );
            }
            "response.completed" => {
                found_response_completed = true;
                println!("✓ Found response.completed event");
            }
            _ => {
                println!("  Other event: {}", event_type);
            }
        }

        if let Some(name) = event_name {
            println!("  Event name: {}", name);
        }
    }

    // Verify key events were present
    println!("\n=== Event Summary ===");
    println!("MCP list_tools added: {}", found_mcp_list_tools);
    println!(
        "MCP list_tools in_progress: {}",
        found_mcp_list_tools_in_progress
    );
    println!(
        "MCP list_tools completed: {}",
        found_mcp_list_tools_completed
    );
    println!("Response created: {}", found_response_created);
    println!("MCP call added: {}", found_mcp_call_added);
    println!("MCP call in_progress: {}", found_mcp_call_in_progress);
    println!("MCP call arguments delta: {}", found_mcp_call_arguments);
    println!("MCP call arguments done: {}", found_mcp_call_arguments_done);
    println!("MCP call done: {}", found_mcp_call_done);
    println!("Response completed: {}", found_response_completed);

    // Assert critical events are present
    assert!(
        found_mcp_list_tools,
        "Should send mcp_list_tools added event at the start"
    );
    assert!(
        found_mcp_list_tools_in_progress,
        "Should send mcp_list_tools.in_progress event"
    );
    assert!(
        found_mcp_list_tools_completed,
        "Should send mcp_list_tools.completed event"
    );
    assert!(found_response_created, "Should send response.created event");
    assert!(found_mcp_call_added, "Should send mcp_call added event");
    assert!(
        found_mcp_call_in_progress,
        "Should send mcp_call.in_progress event"
    );
    assert!(found_mcp_call_done, "Should send mcp_call done event");

    assert!(
        found_mcp_call_arguments,
        "Should send mcp_call_arguments.delta event"
    );
    assert!(
        found_mcp_call_arguments_done,
        "Should send mcp_call_arguments.done event"
    );

    // Verify no error events
    let has_error = body_text.contains("event: error");
    assert!(!has_error, "Should not have error events");

    worker.stop().await;
    mcp.stop().await;
}

#[tokio::test]
async fn test_streaming_multi_turn_with_mcp() {
    // Test streaming with multiple tool call rounds
    let (mut mcp, mut worker, router, _dir) = setup_streaming_mcp_test().await;

    let req = ResponsesRequest {
        background: false,
        include: None,
        input: ResponseInput::Text("complex query requiring multiple tool calls".to_string()),
        instructions: Some("Be thorough".to_string()),
        max_output_tokens: Some(512),
        max_tool_calls: Some(5), // Allow multiple rounds
        metadata: None,
        model: Some("mock-model".to_string()),
        parallel_tool_calls: true,
        previous_response_id: None,
        reasoning: None,
        service_tier: ServiceTier::Auto,
        store: true,
        stream: true,
        temperature: Some(0.8),
        tool_choice: ToolChoice::Value(ToolChoiceValue::Auto),
        tools: vec![ResponseTool {
            r#type: ResponseToolType::Mcp,
            server_url: Some(mcp.url()),
            server_label: Some("mock".to_string()),
            ..Default::default()
        }],
        top_logprobs: 0,
        top_p: Some(1.0),
        truncation: Truncation::Disabled,
        user: None,
        request_id: "resp_streaming_multiturn_test".to_string(),
        priority: 0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop: None,
        top_k: 50,
        min_p: 0.0,
        repetition_penalty: 1.0,
        conversation: None,
    };

    let response = router.route_responses(None, &req, None).await;
    assert_eq!(response.status(), axum::http::StatusCode::OK);

    use axum::body::to_bytes;
    let body_bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body_text = String::from_utf8_lossy(&body_bytes);

    println!("Multi-turn streaming response:\n{}", body_text);

    // Verify streaming completed successfully
    assert!(body_text.contains("data: [DONE]"));
    assert!(!body_text.contains("event: error"));

    // Count events
    let event_count = body_text
        .split("\n\n")
        .filter(|s| !s.trim().is_empty())
        .count();
    println!("Total events in multi-turn stream: {}", event_count);

    assert!(event_count > 0, "Should have received streaming events");

    worker.stop().await;
    mcp.stop().await;
}
