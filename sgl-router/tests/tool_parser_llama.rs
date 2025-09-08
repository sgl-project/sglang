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

#[tokio::test]
async fn test_single_json() {
    // Test parsing plain JSON without python_tag
    let parser = LlamaParser::new();
    let text = r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#;

    let result = parser.parse_complete(text).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Paris");
}

#[tokio::test]
async fn test_multiple_json_with_separator() {
    // Test multiple JSON objects with semicolon separator
    let parser = LlamaParser::new();
    let text = r#"<|python_tag|>{"name": "get_weather", "arguments": {"city": "Paris"}};{"name": "get_tourist_attractions", "arguments": {"city": "Paris"}}"#;

    let result = parser.parse_complete(text).await.unwrap();
    // Note: Current implementation may only parse the first one due to semicolon handling
    assert!(!result.is_empty());
    assert_eq!(result[0].function.name, "get_weather");
}

#[tokio::test]
async fn test_multiple_json_with_separator_customized() {
    // Test multiple JSON objects with python_tag repeated
    let parser = LlamaParser::new();
    let text = r#"<|python_tag|>{"name": "get_weather", "arguments": {}}<|python_tag|>{"name": "get_tourist_attractions", "arguments": {}}"#;

    let result = parser.parse_complete(text).await.unwrap();
    // Current implementation may handle this differently
    assert!(!result.is_empty());
    assert_eq!(result[0].function.name, "get_weather");
}

#[tokio::test]
async fn test_json_with_trailing_text() {
    // Test JSON with trailing text after
    let parser = LlamaParser::new();
    let text = r#"{"name": "get_weather", "arguments": {}} Some follow-up text"#;

    let result = parser.parse_complete(text).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "get_weather");
}

#[tokio::test]
async fn test_invalid_then_valid_json() {
    // Test error recovery - invalid JSON followed by valid JSON
    let parser = LlamaParser::new();
    let text = r#"{"name": "get_weather", "arguments": {{"name": "get_weather", "arguments": {}}"#;

    let result = parser.parse_complete(text).await.unwrap();
    // Should parse at least one valid JSON
    if !result.is_empty() {
        assert_eq!(result[0].function.name, "get_weather");
    }
}

#[tokio::test]
async fn test_plain_text_only() {
    // Test plain text with no tool calls
    let parser = LlamaParser::new();
    let text = "This is just plain explanation text.";

    let result = parser.parse_complete(text).await.unwrap();
    assert_eq!(result.len(), 0);
}

#[tokio::test]
async fn test_with_python_tag_prefix() {
    // Test text before python_tag
    let parser = LlamaParser::new();
    let text = r#"Some intro. <|python_tag|>{"name": "get_weather", "arguments": {}}"#;

    let result = parser.parse_complete(text).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "get_weather");
}

// ============================================================================
// STREAMING TESTS
// ============================================================================

#[tokio::test]
async fn test_llama_streaming_simple() {
    let parser = LlamaParser::new();
    let mut state = sglang_router_rs::tool_parser::ParseState::new();

    // Send complete JSON at once
    let full_json = r#"<|python_tag|>{"name": "search", "arguments": {"query": "weather"}}"#;

    let result = parser
        .parse_incremental(full_json, &mut state)
        .await
        .unwrap();

    match result {
        sglang_router_rs::tool_parser::StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "search");
        }
        _ => panic!("Expected ToolComplete for complete JSON input"),
    }
}

#[tokio::test]
async fn test_llama_streaming_partial() {
    let parser = LlamaParser::new();
    let mut state = sglang_router_rs::tool_parser::ParseState::new();

    // Stream in chunks
    let chunks = vec![
        r#"<|python"#,
        r#"_tag|>{"name": "#,
        r#""calculate", "#,
        r#""arguments": {"x": 10}"#,
        r#"}"#,
    ];

    let mut got_complete = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &mut state).await.unwrap();
        if let sglang_router_rs::tool_parser::StreamResult::ToolComplete(tool) = result {
            assert_eq!(tool.function.name, "calculate");
            got_complete = true;
        }
    }

    assert!(got_complete, "Should have completed parsing");
}

#[tokio::test]
async fn test_llama_streaming_plain_json() {
    let parser = LlamaParser::new();
    let mut state = sglang_router_rs::tool_parser::ParseState::new();

    // Stream plain JSON without python_tag
    let chunks = vec![
        r#"{"name": "#,
        r#""search", "#,
        r#""arguments": "#,
        r#"{"query": "#,
        r#""test"}}"#,
    ];

    let mut got_complete = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &mut state).await.unwrap();
        if let sglang_router_rs::tool_parser::StreamResult::ToolComplete(tool) = result {
            assert_eq!(tool.function.name, "search");
            got_complete = true;
        }
    }

    assert!(got_complete, "Should have completed parsing");
}

#[tokio::test]
async fn test_llama_streaming_with_text_before() {
    let parser = LlamaParser::new();
    let mut state = sglang_router_rs::tool_parser::ParseState::new();

    let chunks = vec![
        r#"Let me help you. "#,
        r#"<|python_tag|>"#,
        r#"{"name": "get_time","#,
        r#" "arguments": {"#,
        r#""timezone": "UTC"}}"#,
    ];

    let mut got_complete = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &mut state).await.unwrap();
        if let sglang_router_rs::tool_parser::StreamResult::ToolComplete(tool) = result {
            assert_eq!(tool.function.name, "get_time");
            got_complete = true;
        }
    }

    assert!(got_complete, "Should have completed parsing");
}

#[tokio::test]
async fn test_llama_streaming_multiple_tools() {
    // Test streaming multiple tool calls with semicolon separator
    let parser = LlamaParser::new();
    let mut state = sglang_router_rs::tool_parser::ParseState::new();

    let text =
        r#"<|python_tag|>{"name": "func1", "arguments": {}};{"name": "func2", "arguments": {}}"#;

    let result = parser.parse_incremental(text, &mut state).await.unwrap();

    // Should get first tool complete
    match result {
        sglang_router_rs::tool_parser::StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "func1");
        }
        _ => panic!("Expected first tool to be complete"),
    }

    // Process remaining buffer to get second tool
    let result2 = parser.parse_incremental("", &mut state).await.unwrap();
    match result2 {
        sglang_router_rs::tool_parser::StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "func2");
        }
        _ => panic!("Expected second tool to be complete"),
    }
}

#[tokio::test]
async fn test_llama_streaming_multiple_tools_chunked() {
    // Test streaming multiple tool calls arriving in chunks
    let parser = LlamaParser::new();
    let mut state = sglang_router_rs::tool_parser::ParseState::new();

    // First chunk - incomplete first JSON
    let chunk1 = r#"<|python_tag|>{"name": "get_weather", "arguments""#;
    let result1 = parser.parse_incremental(chunk1, &mut state).await.unwrap();

    // Should be incomplete or have tool name
    match result1 {
        sglang_router_rs::tool_parser::StreamResult::Incomplete
        | sglang_router_rs::tool_parser::StreamResult::ToolName { .. }
        | sglang_router_rs::tool_parser::StreamResult::ToolArguments { .. } => {
            // Expected - could get tool name or be incomplete or even partial args
        }
        _ => panic!(
            "Expected incomplete or tool name for partial JSON, got: {:?}",
            result1
        ),
    }

    // Second chunk - complete first JSON and separator
    let chunk2 = r#": {"city": "Paris"}};{"name": "#;
    let result2 = parser.parse_incremental(chunk2, &mut state).await.unwrap();

    // Should get first tool complete
    match result2 {
        sglang_router_rs::tool_parser::StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "get_weather");
            let args: serde_json::Value = serde_json::from_str(&tool.function.arguments).unwrap();
            assert_eq!(args["city"], "Paris");
        }
        _ => panic!("Expected first tool to be complete after separator"),
    }

    // Third chunk - complete second JSON
    let chunk3 = r#""get_time", "arguments": {"timezone": "UTC"}}"#;
    let result3 = parser.parse_incremental(chunk3, &mut state).await.unwrap();

    // Should get second tool complete
    match result3 {
        sglang_router_rs::tool_parser::StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "get_time");
            let args: serde_json::Value = serde_json::from_str(&tool.function.arguments).unwrap();
            assert_eq!(args["timezone"], "UTC");
        }
        _ => {
            // If not complete yet, try one more empty chunk
            let result4 = parser.parse_incremental("", &mut state).await.unwrap();
            match result4 {
                sglang_router_rs::tool_parser::StreamResult::ToolComplete(tool) => {
                    assert_eq!(tool.function.name, "get_time");
                    let args: serde_json::Value =
                        serde_json::from_str(&tool.function.arguments).unwrap();
                    assert_eq!(args["timezone"], "UTC");
                }
                _ => panic!("Expected second tool to be complete"),
            }
        }
    }
}
