//! Kimi K2 Parser Integration Tests

use sglang_router_rs::tool_parser::{KimiK2Parser, ToolParser};

mod common;
use common::create_test_tools;

#[tokio::test]
async fn test_kimik2_complete_parsing() {
    let parser = KimiK2Parser::new();

    let input = r#"Let me help you with that.
<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location": "Tokyo", "units": "celsius"}<|tool_call_end|>
<|tool_calls_section_end|>
The weather in Tokyo is..."#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "Let me help you with that.\n");
    assert_eq!(tools[0].function.name, "get_weather");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Tokyo");
    assert_eq!(args["units"], "celsius");
}

#[tokio::test]
async fn test_kimik2_multiple_tools() {
    let parser = KimiK2Parser::new();

    let input = r#"<|tool_calls_section_begin|>
<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{"query": "rust tutorials"}<|tool_call_end|>
<|tool_call_begin|>functions.translate:1<|tool_call_argument_begin|>{"text": "Hello", "to": "ja"}<|tool_call_end|>
<|tool_calls_section_end|>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "translate");
}

#[tokio::test]
async fn test_kimik2_with_whitespace() {
    let parser = KimiK2Parser::new();

    let input = r#"<|tool_calls_section_begin|>
<|tool_call_begin|> functions.test:0 <|tool_call_argument_begin|> {"key": "value", "num": 42} <|tool_call_end|>
<|tool_calls_section_end|>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "test");

    let args: serde_json::Value = serde_json::from_str(&tools[0].function.arguments).unwrap();
    assert_eq!(args["key"], "value");
    assert_eq!(args["num"], 42);
}

#[tokio::test]
async fn test_kimik2_streaming() {
    let tools = create_test_tools();

    let mut parser = KimiK2Parser::new();

    // Simulate streaming chunks
    let chunks = vec![
        "<|tool_calls_section_begin|>\n",
        "<|tool_call_begin|>functions.",
        "calculate:0",
        "<|tool_call_argument_begin|>",
        r#"{"x": 10, "#,
        r#""y": 20}"#,
        "<|tool_call_end|>\n",
        "<|tool_calls_section_end|>",
    ];

    let mut found_name = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &tools).await.unwrap();

        for call in result.calls {
            if let Some(name) = call.name {
                assert_eq!(name, "calculate");
                found_name = true;
            }
        }
    }

    assert!(found_name, "Should have found tool name during streaming");
}

#[test]
fn test_kimik2_format_detection() {
    let parser = KimiK2Parser::new();

    // Should detect Kimi K2 format
    assert!(parser.has_tool_markers("<|tool_calls_section_begin|>"));
    assert!(parser.has_tool_markers("text with <|tool_calls_section_begin|> marker"));

    // Should not detect other formats
    assert!(!parser.has_tool_markers("[TOOL_CALLS]"));
    assert!(!parser.has_tool_markers("<tool_call>"));
    assert!(!parser.has_tool_markers("plain text"));
}

#[tokio::test]
async fn test_kimik2_sequential_indices() {
    let parser = KimiK2Parser::new();

    let input = r#"<|tool_calls_section_begin|>
<|tool_call_begin|>functions.first:0<|tool_call_argument_begin|>{"param": "a"}<|tool_call_end|>
<|tool_call_begin|>functions.second:1<|tool_call_argument_begin|>{"param": "b"}<|tool_call_end|>
<|tool_call_begin|>functions.third:2<|tool_call_argument_begin|>{"param": "c"}<|tool_call_end|>
<|tool_calls_section_end|>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 3);
    assert_eq!(normal_text, "");
    assert_eq!(tools[0].function.name, "first");
    assert_eq!(tools[1].function.name, "second");
    assert_eq!(tools[2].function.name, "third");
}

#[tokio::test]
async fn test_function_index_extraction() {
    let parser = KimiK2Parser::new();

    let input = r#"Text before tool calls.
<|tool_calls_section_begin|>
<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{"query": "rust"}<|tool_call_end|>
<|tool_call_begin|>functions.calc:1<|tool_call_argument_begin|>{"x": 10}<|tool_call_end|>
<|tool_calls_section_end|>"#;

    let (normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 2);
    assert_eq!(normal_text, "Text before tool calls.\n");
    assert_eq!(tools[0].function.name, "search");
    assert_eq!(tools[1].function.name, "calc");
    // TODO: Verify indices are preserved: 0 and 1
}

#[tokio::test]
async fn test_namespace_extraction() {
    let parser = KimiK2Parser::new();

    let input = r#"<|tool_calls_section_begin|>
<|tool_call_begin|>api.tools.search:0<|tool_call_argument_begin|>{"q": "test"}<|tool_call_end|>
<|tool_calls_section_end|>"#;

    let (_normal_text, tools) = parser.parse_complete(input).await.unwrap();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].function.name, "api.tools.search"); // Includes full namespace
}
