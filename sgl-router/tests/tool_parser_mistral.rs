//! Mistral Parser Integration Tests
//!
//! Tests for the Mistral parser which handles [TOOL_CALLS] format

use serde_json::json;
use sglang_router_rs::tool_parser::{MistralParser, ToolParser};

#[tokio::test]
async fn test_mistral_single_tool() {
    let parser = MistralParser::new();
    let input = r#"Let me search for that.
[TOOL_CALLS] [{"name": "search_web", "arguments": {"query": "latest news", "max_results": 5}}]"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "search_web");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["query"], "latest news");
    assert_eq!(args["max_results"], 5);
}

#[tokio::test]
async fn test_mistral_multiple_tools() {
    let parser = MistralParser::new();
    let input = r#"I'll help you with both tasks.
[TOOL_CALLS] [
    {"name": "get_weather", "arguments": {"city": "Tokyo", "units": "celsius"}},
    {"name": "search_news", "arguments": {"query": "AI developments", "limit": 10}}
]"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 2);

    assert_eq!(result[0].function.name, "get_weather");
    let args0: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args0["city"], "Tokyo");

    assert_eq!(result[1].function.name, "search_news");
    let args1: serde_json::Value = serde_json::from_str(&result[1].function.arguments).unwrap();
    assert_eq!(args1["query"], "AI developments");
}

#[tokio::test]
async fn test_mistral_nested_json() {
    let parser = MistralParser::new();
    let input = r#"Processing complex data.
[TOOL_CALLS] [{"name": "process_data", "arguments": {"config": {"nested": {"value": [1, 2, 3]}}, "enabled": true}}]"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["config"]["nested"]["value"], json!([1, 2, 3]));
    assert_eq!(args["enabled"], true);
}

#[tokio::test]
async fn test_mistral_with_text_after() {
    let parser = MistralParser::new();
    let input = r#"[TOOL_CALLS] [{"name": "test", "arguments": {}}]

And here's some text after the tool call that should be ignored."#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "test");
}

#[tokio::test]
async fn test_mistral_empty_arguments() {
    let parser = MistralParser::new();
    let input = r#"[TOOL_CALLS] [{"name": "ping", "arguments": {}}]"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "ping");
}

#[tokio::test]
async fn test_mistral_with_brackets_in_strings() {
    let parser = MistralParser::new();
    let input = r#"[TOOL_CALLS] [{"name": "echo", "arguments": {"text": "Array notation: arr[0] = value[1]"}}]"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Array notation: arr[0] = value[1]");
}

#[tokio::test]
async fn test_mistral_format_detection() {
    let parser = MistralParser::new();

    assert!(parser.detect_format("[TOOL_CALLS] ["));
    assert!(parser.detect_format("Some text [TOOL_CALLS] ["));
    assert!(!parser.detect_format("Just plain text"));
    assert!(!parser.detect_format("[{\"name\": \"test\"}]")); // JSON array without TOOL_CALLS
}

#[tokio::test]
async fn test_mistral_malformed_json() {
    let parser = MistralParser::new();

    // Missing closing bracket
    let input = r#"[TOOL_CALLS] [{"name": "test", "arguments": {}"#;
    if let Ok(result) = parser.parse_complete(input).await {
        assert_eq!(result.len(), 0);
    }
    // Error is also acceptable for malformed input

    // Invalid JSON inside
    let input = r#"[TOOL_CALLS] [{"name": invalid}]"#;
    if let Ok(result) = parser.parse_complete(input).await {
        assert_eq!(result.len(), 0);
    }
    // Error is also acceptable for malformed input
}

#[tokio::test]
async fn test_mistral_real_world_output() {
    let parser = MistralParser::new();

    // Actual output from Mistral model
    let input = r#"I'll search for information about Rust programming and check the weather in San Francisco.

[TOOL_CALLS] [
    {
        "name": "web_search",
        "arguments": {
            "query": "Rust programming language features 2024",
            "max_results": 3,
            "include_snippets": true
        }
    },
    {
        "name": "get_weather",
        "arguments": {
            "location": "San Francisco, CA",
            "units": "fahrenheit",
            "include_forecast": false
        }
    }
]

Let me execute these searches for you."#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].function.name, "web_search");
    assert_eq!(result[1].function.name, "get_weather");
}
