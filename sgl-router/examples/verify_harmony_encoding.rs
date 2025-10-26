//! Verify Harmony encoding matches Python implementation
//!
//! Run with: cargo run --example verify_harmony_encoding

use sglang_router_rs::{
    protocols::{
        chat::{ChatCompletionRequest, ChatMessage, UserMessageContent},
        common::{Function, FunctionCallResponse, Tool, ToolCall},
    },
    routers::grpc::harmony::HarmonyBuilder,
};

fn main() {
    println!("{}", "=".repeat(80));
    println!("Harmony Encoding Verification - Rust Implementation");
    println!("{}", "=".repeat(80));
    println!();

    let builder = HarmonyBuilder::new();

    // Test 1: Simple user message
    println!("Test 1: Simple user message");
    println!("{}", "-".repeat(80));

    let request1 = create_simple_request();
    match builder.build_from_chat(&request1) {
        Ok(output) => {
            println!("Input: User message 'Hello'");
            println!("Token IDs: {:?}", output.input_ids);
            println!("Token count: {}", output.input_ids.len());
        }
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // Test 2: Multi-turn with tool call
    println!("Test 2: Multi-turn conversation");
    println!("{}", "-".repeat(80));

    let request2 = create_tool_call_request();
    match builder.build_from_chat(&request2) {
        Ok(output) => {
            println!("Input: User + Tool call");
            println!("Token IDs: {:?}", output.input_ids);
            println!("Token count: {}", output.input_ids.len());
        }
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // Test 3: With tool response
    println!("Test 3: Tool response");
    println!("{}", "-".repeat(80));

    let request3 = create_tool_response_request();
    match builder.build_from_chat(&request3) {
        Ok(output) => {
            println!("Input: User + Tool call + Tool response");
            println!("Token IDs: {:?}", output.input_ids);
            println!("Token count: {}", output.input_ids.len());
        }
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // Test 4: Assistant with final response
    println!("Test 4: Assistant final response");
    println!("{}", "-".repeat(80));

    let request4 = create_assistant_response_request();
    match builder.build_from_chat(&request4) {
        Ok(output) => {
            println!("Input: User + Assistant final");
            println!("Token IDs: {:?}", output.input_ids);
            println!("Token count: {}", output.input_ids.len());
        }
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // Test 5: With tool definitions
    println!("Test 5: With tool definitions");
    println!("{}", "-".repeat(80));

    let request5 = create_request_with_tools();
    match builder.build_from_chat(&request5) {
        Ok(output) => {
            println!("Input: System + Developer (with tools) + User");
            println!("Token IDs: {:?}", output.input_ids);
            println!("Token count: {}", output.input_ids.len());
        }
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // JSON output for comparison
    println!("{}", "=".repeat(80));
    println!("JSON Output for Comparison");
    println!("{}", "=".repeat(80));

    let test_cases = serde_json::json!({
        "simple_user": {
            "messages": [{"role": "user", "content": "Hello"}],
            "token_ids": builder.build_from_chat(&request1).unwrap().input_ids,
        },
        "multi_turn": {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "recipient": "functions.get_weather",
                    "content_type": "json",
                    "content": r#"{"location": "SF"}"#,
                },
            ],
            "token_ids": builder.build_from_chat(&request2).unwrap().input_ids,
        },
    });

    println!("{}", serde_json::to_string_pretty(&test_cases).unwrap());
}

fn create_simple_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("Hello".to_string()),
            name: None,
        }],
        model: "gpt-4o".to_string(),
        ..Default::default()
    }
}

fn create_tool_call_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        messages: vec![
            ChatMessage::User {
                content: UserMessageContent::Text("What's the weather?".to_string()),
                name: None,
            },
            ChatMessage::Assistant {
                content: None,
                name: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: FunctionCallResponse {
                        name: "get_weather".to_string(),
                        arguments: Some(r#"{"location": "SF"}"#.to_string()),
                    },
                }]),
                reasoning_content: None,
            },
        ],
        model: "gpt-4o".to_string(),
        ..Default::default()
    }
}

fn create_tool_response_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        messages: vec![
            ChatMessage::User {
                content: UserMessageContent::Text("What's the weather?".to_string()),
                name: None,
            },
            ChatMessage::Assistant {
                content: None,
                name: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    tool_type: "function".to_string(),
                    function: FunctionCallResponse {
                        name: "get_weather".to_string(),
                        arguments: Some(r#"{"location": "SF"}"#.to_string()),
                    },
                }]),
                reasoning_content: None,
            },
            ChatMessage::Tool {
                content: r#"{"temperature": 72}"#.to_string(),
                tool_call_id: "call_1".to_string(),
            },
        ],
        model: "gpt-4o".to_string(),
        ..Default::default()
    }
}

fn create_assistant_response_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        messages: vec![
            ChatMessage::User {
                content: UserMessageContent::Text("Hello".to_string()),
                name: None,
            },
            ChatMessage::Assistant {
                content: Some("Hi there".to_string()),
                name: None,
                tool_calls: None,
                reasoning_content: None,
            },
        ],
        model: "gpt-4o".to_string(),
        ..Default::default()
    }
}

fn create_request_with_tools() -> ChatCompletionRequest {
    ChatCompletionRequest {
        messages: vec![ChatMessage::User {
            content: UserMessageContent::Text("What's the weather in SF?".to_string()),
            name: None,
        }],
        tools: Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get the current weather for a location".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }),
                strict: None,
            },
        }]),
        model: "gpt-4o".to_string(),
        ..Default::default()
    }
}
