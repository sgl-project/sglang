//! JSON Parser Integration Tests
//!
//! Tests for the JSON parser which handles OpenAI, Claude, and generic JSON formats

use serde_json::json;
use sglang_router_rs::tool_parser::{JsonParser, ToolParser};

#[tokio::test]
async fn test_simple_json_tool_call() {
    let parser = JsonParser::new();
    let input = r#"{"name": "get_weather", "arguments": {"location": "San Francisco"}}"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "San Francisco");
}

#[tokio::test]
async fn test_json_array_of_tools() {
    let parser = JsonParser::new();
    let input = r#"Hello, here are the results: [
        {"name": "get_weather", "arguments": {"location": "SF"}},
        {"name": "search", "arguments": {"query": "news"}}
    ]"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(normal_text, "Hello, here are the results: ");
    assert_eq!(tools[0].function.name, "get_weather");
    assert_eq!(tools[1].function.name, "search");
}

#[tokio::test]
async fn test_json_with_parameters_key() {
    let parser = JsonParser::new();
    let input = r#"{"name": "calculate", "parameters": {"x": 10, "y": 20}}"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "calculate");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["x"], 10);
    assert_eq!(args["y"], 20);
}

#[tokio::test]
async fn test_json_extraction_from_text() {
    let parser = JsonParser::new();
    let input = r#"I'll help you with that. {"name": "search", "arguments": {"query": "rust"}} Let me search for that."#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(
        normal_text,
        "I'll help you with that.  Let me search for that."
    );
    assert_eq!(tools[0].function.name, "search");
}

#[tokio::test]
async fn test_json_with_nested_objects() {
    let parser = JsonParser::new();
    let input = r#"{
        "name": "update_config",
        "arguments": {
            "settings": {
                "theme": "dark",
                "language": "en",
                "notifications": {
                    "email": true,
                    "push": false
                }
            }
        }
    }"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "update_config");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["settings"]["theme"], "dark");
    assert_eq!(args["settings"]["notifications"]["email"], true);
}

#[tokio::test]
async fn test_json_with_special_characters() {
    let parser = JsonParser::new();
    let input = r#"{"name": "echo", "arguments": {"text": "Line 1\nLine 2\tTabbed", "path": "C:\\Users\\test"}}"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Line 1\nLine 2\tTabbed");
    assert_eq!(args["path"], "C:\\Users\\test");
}

#[tokio::test]
async fn test_json_with_unicode() {
    let parser = JsonParser::new();
    let input = r#"{"name": "translate", "arguments": {"text": "Hello 世界 🌍", "emoji": "😊"}}"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["text"], "Hello 世界 🌍");
    assert_eq!(args["emoji"], "😊");
}

#[tokio::test]
async fn test_json_empty_arguments() {
    let parser = JsonParser::new();
    let input = r#"{"name": "ping", "arguments": {}}"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "ping");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args, json!({}));
}

#[tokio::test]
async fn test_json_invalid_format() {
    let parser = JsonParser::new();

    // Missing closing brace
    let input = r#"{"name": "test", "arguments": {"key": "value""#;
    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
    assert_eq!(
        normal_text,
        "{\"name\": \"test\", \"arguments\": {\"key\": \"value\""
    );

    // Not JSON at all
    let input = "This is just plain text";
    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 0);
}

#[tokio::test]
async fn test_json_format_detection() {
    let parser = JsonParser::new();

    assert!(parser.has_tool_markers(r#"{"name": "test", "arguments": {}}"#));
    assert!(parser.has_tool_markers(r#"[{"name": "test"}]"#));
    assert!(!parser.has_tool_markers("plain text"));
}

// Streaming tests for JSON array format
#[tokio::test]
async fn test_json_array_streaming_required_mode() {
    use sglang_router_rs::protocols::common::Tool;

    // Test that simulates the exact streaming pattern from required mode
    let mut parser = JsonParser::new();

    // Define test tools
    let tools = vec![Tool {
        tool_type: "function".to_string(),
        function: sglang_router_rs::protocols::common::Function {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: serde_json::json!({}),
            strict: None,
        },
    }];

    // Simulate the EXACT chunks from the debug log
    let chunks = vec![
        "[{",
        " \"",
        "name",
        "\":",
        " \"",
        "get",
        "_weather",
        "\",",
        " \"",
        "parameters",
        "\":",
        " {",
        " \"",
        "city",
        "\":",
        " \"",
        "Paris",
        "\"",
        " }",
        " }]",
    ];

    let mut all_results = Vec::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        println!("Chunk: '{}' -> Calls: {}", chunk, result.calls.len());
        all_results.extend(result.calls);
    }

    // We should have gotten tool call chunks
    assert!(
        !all_results.is_empty(),
        "Should have emitted tool call chunks"
    );

    // Check that we got the function name
    let has_name = all_results
        .iter()
        .any(|item| item.name.as_ref().is_some_and(|n| n == "get_weather"));
    assert!(has_name, "Should have emitted function name");

    // Check that we got the parameters
    let has_params = all_results.iter().any(|item| !item.parameters.is_empty());
    assert!(has_params, "Should have emitted parameters");
}

#[tokio::test]
async fn test_json_array_multiple_tools_streaming() {
    use sglang_router_rs::protocols::common::Tool;

    // Test with multiple tools in array
    let mut parser = JsonParser::new();

    let tools = vec![
        Tool {
            tool_type: "function".to_string(),
            function: sglang_router_rs::protocols::common::Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: serde_json::json!({}),
                strict: None,
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: sglang_router_rs::protocols::common::Function {
                name: "get_news".to_string(),
                description: Some("Get news".to_string()),
                parameters: serde_json::json!({}),
                strict: None,
            },
        },
    ];

    // Split into smaller, more realistic chunks
    let chunks = vec![
        "[{",
        "\"name\":",
        "\"get_weather\"",
        ",\"parameters\":",
        "{\"city\":",
        "\"SF\"}",
        "}",
        ",",
        "{\"name\":",
        "\"get_news\"",
        ",\"parameters\":",
        "{\"topic\":",
        "\"tech\"}",
        "}]",
    ];

    let mut all_results = Vec::new();

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();
        println!("Chunk: '{}' -> Calls: {}", chunk, result.calls.len());
        all_results.extend(result.calls);
    }

    // Should have gotten tool calls for both functions
    let has_weather = all_results
        .iter()
        .any(|item| item.name.as_ref().is_some_and(|n| n == "get_weather"));
    let has_news = all_results
        .iter()
        .any(|item| item.name.as_ref().is_some_and(|n| n == "get_news"));

    assert!(has_weather, "Should have get_weather tool call");
    assert!(has_news, "Should have get_news tool call");
}
