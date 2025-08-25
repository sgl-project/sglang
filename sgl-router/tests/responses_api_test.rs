// Integration test for Responses API

use sglang_router_rs::protocols::spec::{
    GenerationRequest, ReasoningEffort, ResponseInput, ResponseReasoningParam, ResponseStatus,
    ResponseTool, ResponseToolType, ResponsesRequest, ResponsesResponse, ServiceTier, ToolChoice,
    ToolChoiceValue, Truncation, UsageInfo,
};

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

    // Test GenerationRequest trait implementation
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

    // Test reverse conversion
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

    // Test JSON serialization/deserialization preserves default
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

    // Test that everything can be serialized to JSON and back
    let json = serde_json::to_string(&request).expect("Serialization should work");
    let parsed: ResponsesRequest =
        serde_json::from_str(&json).expect("Deserialization should work");

    assert_eq!(parsed.request_id, "resp_comprehensive_test");
    assert_eq!(parsed.model, Some("gpt-4".to_string()));
    assert!(parsed.background);
    assert!(parsed.stream);
    assert_eq!(parsed.tools.len(), 1);
}
