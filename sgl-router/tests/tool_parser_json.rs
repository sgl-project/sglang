//! JSON Parser Integration Tests
//!
//! Tests for the JSON parser which handles OpenAI, Claude, and generic JSON formats

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
async fn test_json_format_detection() {
    let parser = JsonParser::new();

    assert!(parser.has_tool_markers(r#"{"name": "test", "arguments": {}}"#));
    assert!(parser.has_tool_markers(r#"[{"name": "test"}]"#));
    assert!(!parser.has_tool_markers("plain text"));
}

// =============================================================================
// COMMON TEST SUITE IMPLEMENTATION
// =============================================================================

mod common;
use common::test_suite::CommonParserTests;

struct JsonTestSuite;

impl CommonParserTests for JsonTestSuite {
    type Parser = JsonParser;

    fn create_parser() -> Self::Parser {
        JsonParser::new()
    }

    fn format_empty_args_input() -> &'static str {
        r#"{"name": "ping", "arguments": {}}"#
    }

    fn format_single_tool_input(name: &str, args_json: &str) -> String {
        format!(r#"{{"name": "{}", "arguments": {}}}"#, name, args_json)
    }

    fn format_compact_json_input() -> &'static str {
        r#"{"name":"test","arguments":{"key":"value"}}"#
    }
}

// Common tests
#[tokio::test]
async fn test_json_common_empty_input() {
    JsonTestSuite::test_empty_input_impl().await;
}

#[tokio::test]
async fn test_json_common_plain_text() {
    JsonTestSuite::test_plain_text_impl().await;
}

#[tokio::test]
async fn test_json_common_empty_arguments() {
    JsonTestSuite::test_empty_arguments_impl().await;
}

#[tokio::test]
async fn test_json_common_nested_json() {
    JsonTestSuite::test_nested_json_impl().await;
}

#[tokio::test]
async fn test_json_common_unicode() {
    JsonTestSuite::test_unicode_impl().await;
}

#[tokio::test]
async fn test_json_common_special_json_values() {
    JsonTestSuite::test_special_json_values_impl().await;
}

#[tokio::test]
async fn test_json_common_escaped_chars() {
    JsonTestSuite::test_escaped_chars_impl().await;
}

#[tokio::test]
async fn test_json_common_long_string() {
    JsonTestSuite::test_long_string_impl().await;
}

#[tokio::test]
async fn test_json_common_compact_json() {
    JsonTestSuite::test_compact_json_impl().await;
}

#[tokio::test]
async fn test_json_common_malformed() {
    JsonTestSuite::test_malformed_json_impl().await;
}
