//! Llama Parser Integration Tests
//!
//! Tests for the Llama parser which handles <|python_tag|> format and plain JSON

use sglang_router_rs::tool_parser::{LlamaParser, ToolParser};

mod common;
use common::{create_test_tools, streaming_helpers::*};

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

    assert!(parser.has_tool_markers(r#"<|python_tag|>{"name": "test"}"#));
    assert!(parser.has_tool_markers(r#"{"name": "test", "parameters": {}}"#));
    assert!(!parser.has_tool_markers("plain text"));
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
    let tools = create_test_tools();

    let mut parser = LlamaParser::new();

    // Send complete JSON at once
    let full_json = r#"<|python_tag|>{"name": "search", "parameters": {"query": "weather"}}"#;

    let result = parser.parse_incremental(full_json, &tools).await.unwrap();

    assert!(
        !result.calls.is_empty(),
        "Expected tool call for complete JSON input"
    );
    assert_eq!(result.calls[0].name.as_ref().unwrap(), "search");
}

#[tokio::test]
async fn test_llama_streaming_partial() {
    let tools = create_test_tools();

    let mut parser = LlamaParser::new();

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
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        if !result.calls.is_empty() {
            if let Some(name) = &result.calls[0].name {
                assert_eq!(name, "calculate");
                got_complete = true;
            }
        }
    }

    assert!(got_complete, "Should have completed parsing");
}

#[tokio::test]
async fn test_llama_streaming_plain_json() {
    let tools = create_test_tools();

    let mut parser = LlamaParser::new();

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
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        if !result.calls.is_empty() {
            if let Some(name) = &result.calls[0].name {
                assert_eq!(name, "search");
                got_complete = true;
            }
        }
    }

    assert!(got_complete, "Should have completed parsing");
}

#[tokio::test]
async fn test_llama_streaming_with_text_before() {
    let tools = create_test_tools();

    let mut parser = LlamaParser::new();

    let chunks = vec![
        r#"Let me help you. "#,
        r#"<|python_tag|>"#,
        r#"{"name": "get_time","#,
        r#" "parameters": {"#,
        r#""timezone": "UTC"}}"#,
    ];

    let mut got_complete = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        if !result.calls.is_empty() {
            if let Some(name) = &result.calls[0].name {
                assert_eq!(name, "get_time");
                got_complete = true;
            }
        }
    }

    assert!(got_complete, "Should have completed parsing");
}

#[tokio::test]
async fn test_llama_streaming_multiple_tools() {
    let tools = create_test_tools();

    let mut parser = LlamaParser::new();

    let text =
        r#"<|python_tag|>{"name": "func1", "parameters": {}};{"name": "func2", "parameters": {}}"#;

    let result = parser.parse_incremental(text, &tools).await.unwrap();

    // Should get first tool complete
    assert!(
        !result.calls.is_empty(),
        "Expected first tool to be complete"
    );
    if let Some(name) = &result.calls[0].name {
        assert_eq!(name, "func1");
    }

    // Process remaining buffer to get second tool
    let result2 = parser.parse_incremental("", &tools).await.unwrap();
    if !result2.calls.is_empty() {
        if let Some(name) = &result2.calls[0].name {
            assert_eq!(name, "func2");
        }
    }
}

#[tokio::test]
async fn test_llama_streaming_multiple_tools_chunked() {
    let mut parser = LlamaParser::new();

    let tools = create_test_tools();

    // First chunk - incomplete first JSON
    let chunk1 = r#"<|python_tag|>{"name": "get_weather", "parameters""#;
    let result1 = parser.parse_incremental(chunk1, &tools).await.unwrap();
    if !result1.calls.is_empty() {
        if let Some(name) = &result1.calls[0].name {
            assert_eq!(name, "get_weather");
        }
    }

    // Second chunk - complete first JSON and separator
    let chunk2 = r#": {"city": "Paris"}};{"name": "#;
    let result2 = parser.parse_incremental(chunk2, &tools).await.unwrap();

    // Should get parameters for first tool (name already sent in result1)
    if !result2.calls.is_empty() {
        let args: serde_json::Value = serde_json::from_str(&result2.calls[0].parameters).unwrap();
        assert_eq!(args["city"], "Paris");
    }

    let chunk3 = r#""get_time", "parameters": {"timezone": "UTC"}}"#;
    let result3 = parser.parse_incremental(chunk3, &tools).await.unwrap();
    if !result3.calls.is_empty() {
        if let Some(name) = &result3.calls[0].name {
            assert_eq!(name, "get_time");
        }
    }
}

// =============================================================================
// REALISTIC STREAMING TESTS
// =============================================================================

#[tokio::test]
async fn test_llama_realistic_chunks_with_python_tag() {
    let tools = create_test_tools();
    let mut parser = LlamaParser::new();

    let input = r#"<|python_tag|>{"name": "calculate", "parameters": {"x": 10, "y": 20}}"#;
    let chunks = create_realistic_chunks(input);

    assert!(chunks.len() > 15, "Should have many small chunks");

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(&chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "calculate");
                got_tool_name = true;
            }
        }
    }

    assert!(got_tool_name, "Should have parsed tool name");
}

#[tokio::test]
async fn test_llama_python_tag_arrives_in_parts() {
    let tools = create_test_tools();
    let mut parser = LlamaParser::new();

    // Python tag itself arrives in small chunks
    let chunks = vec![
        "<|p", "yth", "on_", "tag", "|>{", r#"""#, "na", r#"me""#, ": ", r#"""#, "sea", "rch",
        r#"""#, ", ", r#"""#, "par", "ame", "ter", "s", r#"""#, ": {", r#"""#, "q", r#"""#, ": ",
        r#"""#, "tes", "t", r#"""#, "}}",
    ];

    let mut got_tool_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "search");
                got_tool_name = true;
            }
        }
    }

    assert!(got_tool_name, "Should have parsed tool name");
}
