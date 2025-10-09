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
