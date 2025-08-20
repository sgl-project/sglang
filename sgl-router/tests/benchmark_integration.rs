// Integration test to ensure benchmarks compile and basic functionality works
// This prevents benchmarks from breaking in CI
//
// UPDATED: Removed deprecated ToPdRequest usage, now uses direct JSON serialization

use serde_json::{from_str, to_string, to_value};
use sglang_router_rs::core::{BasicWorker, WorkerType};
use sglang_router_rs::protocols::{
    common::StringOrArray,
    generate::{GenerateParameters, GenerateRequest, SamplingParams},
    openai::{
        chat::{ChatCompletionRequest, ChatMessage, UserMessageContent},
        completions::CompletionRequest,
    },
};

/// Create a default GenerateRequest for benchmarks with minimal fields set
fn default_generate_request() -> GenerateRequest {
    GenerateRequest {
        text: None,
        prompt: None,
        input_ids: None,
        stream: false,
        parameters: None,
        sampling_params: None,
        return_logprob: false,
        // SGLang Extensions
        lora_path: None,
        session_params: None,
        return_hidden_states: false,
        rid: None,
    }
}

/// Create a default ChatCompletionRequest for benchmarks with minimal fields set
fn default_chat_completion_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: String::new(),
        messages: vec![],
        max_tokens: None,
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        n: None,
        stream: false,
        stream_options: None,
        stop: None,
        presence_penalty: None,
        frequency_penalty: None,
        logit_bias: None,
        logprobs: false,
        top_logprobs: None,
        user: None,
        response_format: None,
        seed: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        function_call: None,
        functions: None,
        // SGLang Extensions
        top_k: None,
        min_p: None,
        min_tokens: None,
        repetition_penalty: None,
        regex: None,
        ebnf: None,
        stop_token_ids: None,
        no_stop_trim: false,
        ignore_eos: false,
        continue_final_message: false,
        skip_special_tokens: true,
        // SGLang Extensions
        lora_path: None,
        session_params: None,
        separate_reasoning: true,
        stream_reasoning: true,
        return_hidden_states: false,
    }
}

/// Create a default CompletionRequest for benchmarks with minimal fields set
fn default_completion_request() -> CompletionRequest {
    CompletionRequest {
        model: String::new(),
        prompt: StringOrArray::String(String::new()),
        suffix: None,
        max_tokens: None,
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
        // SGLang Extensions
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
        // SGLang Extensions
        lora_path: None,
        session_params: None,
        return_hidden_states: false,
        other: serde_json::Map::new(),
    }
}

#[allow(dead_code)]
fn create_test_worker() -> BasicWorker {
    BasicWorker::new(
        "http://test-server:8000".to_string(),
        WorkerType::Prefill {
            bootstrap_port: Some(5678),
        },
    )
}

#[test]
fn test_benchmark_request_creation() {
    // Ensure all benchmark request types can be created without panicking

    let generate_req = GenerateRequest {
        text: Some("Test prompt".to_string()),
        parameters: Some(GenerateParameters {
            max_new_tokens: Some(100),
            temperature: Some(0.8),
            top_p: Some(0.9),
            top_k: Some(50),
            repetition_penalty: Some(1.0),
            ..Default::default()
        }),
        sampling_params: Some(SamplingParams {
            temperature: Some(0.8),
            top_p: Some(0.9),
            top_k: Some(50),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            repetition_penalty: Some(1.0),
            ..Default::default()
        }),
        ..default_generate_request()
    };

    let chat_req = ChatCompletionRequest {
        model: "test-model".to_string(),
        messages: vec![ChatMessage::User {
            role: "user".to_string(),
            content: UserMessageContent::Text("Test message".to_string()),
            name: None,
        }],
        max_tokens: Some(150),
        max_completion_tokens: Some(150),
        temperature: Some(0.7),
        top_p: Some(1.0),
        n: Some(1),
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        parallel_tool_calls: Some(true),
        ..default_chat_completion_request()
    };

    let completion_req = CompletionRequest {
        model: "test-model".to_string(),
        prompt: StringOrArray::String("Test prompt".to_string()),
        max_tokens: Some(50),
        temperature: Some(0.8),
        top_p: Some(1.0),
        n: Some(1),
        presence_penalty: Some(0.0),
        frequency_penalty: Some(0.0),
        best_of: Some(1),
        ..default_completion_request()
    };

    // Test serialization works
    assert!(to_string(&generate_req).is_ok());
    assert!(to_string(&chat_req).is_ok());
    assert!(to_string(&completion_req).is_ok());
}

#[test]
fn test_benchmark_serialization_roundtrip() {
    // Test serialization/deserialization roundtrip for benchmark types

    let generate_req = GenerateRequest {
        text: Some("Test prompt".to_string()),
        ..default_generate_request()
    };

    // Serialize and deserialize
    let json = to_string(&generate_req).expect("Serialization should work");
    let deserialized: GenerateRequest = from_str(&json).expect("Deserialization should work");

    // Verify basic field equality
    assert_eq!(generate_req.text, deserialized.text);
    assert_eq!(generate_req.stream, deserialized.stream);
    assert_eq!(generate_req.return_logprob, deserialized.return_logprob);
}

#[test]
fn test_benchmark_direct_json_routing() {
    // Test direct JSON routing functionality for benchmark types (replaces regular routing)

    let generate_req = GenerateRequest {
        text: Some("Test prompt".to_string()),
        ..default_generate_request()
    };

    // Test direct JSON conversion (replaces regular routing methods)
    let json = to_value(&generate_req).unwrap();
    let json_string = to_string(&json).unwrap();
    let bytes = json_string.as_bytes();

    // Verify conversions work
    assert!(!json_string.is_empty());
    assert!(!bytes.is_empty());
}
