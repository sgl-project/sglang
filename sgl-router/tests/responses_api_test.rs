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
