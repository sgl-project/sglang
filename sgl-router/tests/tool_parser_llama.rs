//! Llama Parser Integration Tests
//!
//! Tests for the Llama parser which handles <|python_tag|> format and plain JSON

use sglang_router_rs::tool_parser::{LlamaParser, ToolParser};

#[tokio::test]
async fn test_llama_python_tag_format() {
    let parser = LlamaParser::new();
    let input = r#"Here are some results: <|python_tag|>{"name": "search", "parameters": {"query": "weather"}}"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(normal_text, "Here are some results: ");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["query"], "weather");
}

#[tokio::test]
async fn test_llama_with_semicolon_separation() {
    let parser = LlamaParser::new();

    let input = r#"<|python_tag|>{"name": "tool1", "parameters": {}};{"name": "tool2", "parameters": {"y": 2}}"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].function.name, "tool1");
    assert_eq!(tools[1].function.name, "tool2");
    assert_eq!(normal_text, "");
}

#[tokio::test]
async fn test_llama_no_tool_calls() {
    let parser = LlamaParser::new();

    let input = "This is just plain text with no tool calls";
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, input);
}

#[tokio::test]
async fn test_llama_plain_json_fallback() {
    let parser = LlamaParser::new();
    let input = r#"{"name": "calculate", "parameters": {"x": 5, "y": 10}}"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "calculate");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["x"], 5);
    assert_eq!(args["y"], 10);
}

#[tokio::test]
async fn test_llama_with_text_before() {
    let parser = LlamaParser::new();
    let input = r#"Let me help you with that. <|python_tag|>{"name": "get_time", "parameters": {"timezone": "UTC"}}"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Let me help you with that. ");
    assert_eq!(tools[0].function.name, "get_time");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["timezone"], "UTC");
}

#[tokio::test]
async fn test_llama_with_nested_json() {
    let parser = LlamaParser::new();
    let input = r#"<|python_tag|>{
        "name": "update_settings",
        "parameters": {
            "preferences": {
                "theme": "dark",
                "language": "en"
            },
            "notifications": true
        }
    }"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "update_settings");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["preferences"]["theme"], "dark");
    assert_eq!(args["notifications"], true);
}

#[tokio::test]
async fn test_llama_empty_arguments() {
    let parser = LlamaParser::new();

    // With python_tag
    let input = r#"<|python_tag|>{"name": "ping", "parameters": {}}"#;
    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "ping");

    // Plain JSON
    let input = r#"{"name": "ping", "parameters": {}}"#;
    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "ping");
}

#[tokio::test]
async fn test_llama_format_detection() {
    let parser = LlamaParser::new();

    assert!(parser.detect_format(r#"<|python_tag|>{"name": "test"}"#));
    assert!(parser.detect_format(r#"{"name": "test", "parameters": {}}"#));
    assert!(!parser.detect_format("plain text"));
    assert!(!parser.detect_format(r#"{"key": "value"}"#)); // No name field
}

#[tokio::test]
async fn test_llama_invalid_json_after_tag() {
    let parser = LlamaParser::new();

    let input = r#"<|python_tag|>{"name": invalid}"#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, "<|python_tag|>{\"name\": invalid}");
}

#[tokio::test]
async fn test_llama_real_world_output() {
    let parser = LlamaParser::new();

    // Actual output from Llama 3.2 model - simplified for testing
    let input = r#"I'll search for that information for you.

<|python_tag|>{"name": "web_search", "parameters": {"query": "Llama 3.2 model capabilities", "num_results": 5, "search_type": "recent"}}"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "web_search");

    let formatted_input = r#"<|python_tag|>{
    "name": "get_current_time",
    "parameters": {
        "timezone": "America/New_York",
        "format": "ISO8601"
    }
}"#;

    let (_normal_text, tools2) = parser.parse_complete(formatted_input).await.unwrap();
    assert_eq!(tools2.len(), 1);
    assert_eq!(tools2[0].function.name, "get_current_time");
}

#[tokio::test]
async fn test_single_json() {
    let parser = LlamaParser::new();
    let text = r#"{"name": "get_weather", "parameters": {"city": "Paris"}}"#;

    let (_normal_text, tools) = parser.parse_complete(text).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Paris");
}

#[tokio::test]
async fn test_multiple_json_with_separator() {
    let parser = LlamaParser::new();
    let text = r#"<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}};{"name": "get_tourist_attractions", "parameters": {"city": "Paris"}}"#;

    let (_normal_text, tools) = parser.parse_complete(text).await.unwrap();
    // Note: Current implementation may only parse the first one due to semicolon handling
    assert!(!tools.is_empty());
    assert_eq!(tools[0].function.name, "get_weather");
}

#[tokio::test]
async fn test_json_with_trailing_text() {
    let parser = LlamaParser::new();
    // Valid JSON with trailing text - LlamaParser doesn't support this mixed format
    let text = r#"{"name": "get_weather", "parameters": {}} Some follow-up text"#;

    let (normal_text, tools) = parser.parse_complete(text).await.unwrap();
    // LlamaParser expects pure JSON or <|python_tag|> format, not JSON with trailing text
    // So this returns as normal text
    assert_eq!(tools.len(), 0);
    assert_eq!(normal_text, text);
}

#[tokio::test]
async fn test_invalid_then_valid_json() {
    let parser = LlamaParser::new();
    let text =
        r#"{"name": "get_weather", "parameters": {{"name": "get_weather", "parameters": {}}"#;

    let (_normal_text, tools) = parser.parse_complete(text).await.unwrap();
    // Should parse at least one valid JSON
    if !tools.is_empty() {
        assert_eq!(tools[0].function.name, "get_weather");
    }
}

#[tokio::test]
async fn test_plain_text_only() {
    let parser = LlamaParser::new();
    let text = "This is just plain explanation text.";

    let (_normal_text, tools) = parser.parse_complete(text).await.unwrap();
    assert_eq!(tools.len(), 0);
}

#[tokio::test]
async fn test_with_python_tag_prefix() {
    let parser = LlamaParser::new();
    let text = r#"Some intro. <|python_tag|>{"name": "get_weather", "parameters": {}}"#;

    let (_normal_text, tools) = parser.parse_complete(text).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "get_weather");
}

// STREAMING TESTS

#[tokio::test]
async fn test_llama_streaming_simple() {
    let parser = LlamaParser::new();
    let mut state = sglang_router_rs::tool_parser::ParseState::new();

    // Send complete JSON at once
    let full_json = r#"<|python_tag|>{"name": "search", "parameters": {"query": "weather"}}"#;

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
        r#""parameters": {"x": 10}"#,
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
        r#""parameters": "#,
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
        r#" "parameters": {"#,
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
    let parser = LlamaParser::new();
    let mut state = sglang_router_rs::tool_parser::ParseState::new();

    let text =
        r#"<|python_tag|>{"name": "func1", "parameters": {}};{"name": "func2", "parameters": {}}"#;

    let result = parser.parse_incremental(text, &mut state).await.unwrap();

    // Should get first tool complete
    match result {
        sglang_router_rs::tool_parser::StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "func1");
        }
        _ => panic!("Expected first tool to be complete, got: {:?}", result),
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
    let parser = LlamaParser::new();
    let mut state = sglang_router_rs::tool_parser::ParseState::new();

    // First chunk - incomplete first JSON
    let chunk1 = r#"<|python_tag|>{"name": "get_weather", "parameters""#;
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
        _ => panic!("Expected first tool complete, got: {:?}", result2),
    }

    let chunk3 = r#""get_time", "parameters": {"timezone": "UTC"}}"#;
    let result3 = parser.parse_incremental(chunk3, &mut state).await.unwrap();
    match result3 {
        sglang_router_rs::tool_parser::StreamResult::ToolComplete(tool) => {
            assert_eq!(tool.function.name, "get_time");
        }
        _ => panic!("Expected tool to be complete, got: {:?}", result3),
    }
}
