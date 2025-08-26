//! Llama Parser Integration Tests
//!
//! Tests for the Llama parser which handles <|python_tag|> format and plain JSON

use sglang_router_rs::tool_parser::{LlamaParser, ToolParser};

#[tokio::test]
async fn test_llama_python_tag_format() {
    let parser = LlamaParser::new();
    let input = r#"<|python_tag|>{"name": "search", "arguments": {"query": "weather"}}"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "search");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["query"], "weather");
}

#[tokio::test]
async fn test_llama_plain_json_fallback() {
    let parser = LlamaParser::new();
    let input = r#"{"name": "calculate", "arguments": {"x": 5, "y": 10}}"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "calculate");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["x"], 5);
    assert_eq!(args["y"], 10);
}

#[tokio::test]
async fn test_llama_with_text_before() {
    let parser = LlamaParser::new();
    let input = r#"Let me help you with that. <|python_tag|>{"name": "get_time", "arguments": {"timezone": "UTC"}}"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "get_time");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["timezone"], "UTC");
}

#[tokio::test]
async fn test_llama_with_nested_json() {
    let parser = LlamaParser::new();
    let input = r#"<|python_tag|>{
        "name": "update_settings",
        "arguments": {
            "preferences": {
                "theme": "dark",
                "language": "en"
            },
            "notifications": true
        }
    }"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "update_settings");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["preferences"]["theme"], "dark");
    assert_eq!(args["notifications"], true);
}

#[tokio::test]
async fn test_llama_empty_arguments() {
    let parser = LlamaParser::new();

    // With python_tag
    let input = r#"<|python_tag|>{"name": "ping", "arguments": {}}"#;
    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "ping");

    // Plain JSON
    let input = r#"{"name": "ping", "arguments": {}}"#;
    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "ping");
}

#[tokio::test]
async fn test_llama_format_detection() {
    let parser = LlamaParser::new();

    assert!(parser.detect_format(r#"<|python_tag|>{"name": "test"}"#));
    assert!(parser.detect_format(r#"{"name": "test", "arguments": {}}"#));
    assert!(!parser.detect_format("plain text"));
    assert!(!parser.detect_format(r#"{"key": "value"}"#)); // No name field
}

#[tokio::test]
async fn test_llama_invalid_json_after_tag() {
    let parser = LlamaParser::new();

    let input = r#"<|python_tag|>{"name": invalid}"#;
    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 0);
}

#[tokio::test]
async fn test_llama_real_world_output() {
    let parser = LlamaParser::new();

    // Actual output from Llama 3.2 model - simplified for testing
    let input = r#"I'll search for that information for you.

<|python_tag|>{"name": "web_search", "arguments": {"query": "Llama 3.2 model capabilities", "num_results": 5, "search_type": "recent"}}"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "web_search");

    // Test with nicely formatted JSON
    let formatted_input = r#"<|python_tag|>{
    "name": "get_current_time",
    "arguments": {
        "timezone": "America/New_York",
        "format": "ISO8601"
    }
}"#;

    let result2 = parser.parse_complete(formatted_input).await.unwrap();
    assert_eq!(result2.len(), 1);
    assert_eq!(result2[0].function.name, "get_current_time");
}

#[tokio::test]
async fn test_llama_json_array_format() {
    let parser = LlamaParser::new();

    // Plain JSON array (should work as fallback)
    let input = r#"[{"name": "func1", "arguments": {}}, {"name": "func2", "arguments": {}}]"#;

    let result = parser.parse_complete(input).await.unwrap();
    // Current implementation might handle this through JSON fallback
    assert!(!result.is_empty());
}
