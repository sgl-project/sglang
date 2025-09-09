//! Kimi K2 Parser Integration Tests

use sglang_router_rs::tool_parser::{KimiK2Parser, ParseState, StreamResult, ToolParser};

#[tokio::test]
async fn test_kimik2_complete_parsing() {
    let parser = KimiK2Parser::new();

    // Test single tool call
    let input = r#"Let me help you with that.
<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location": "Tokyo", "units": "celsius"}<|tool_call_end|>
<|tool_calls_section_end|>
The weather in Tokyo is..."#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "get_weather");

    // Verify arguments
    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
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

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].function.name, "search");
    assert_eq!(result[1].function.name, "translate");
}

#[tokio::test]
async fn test_kimik2_with_whitespace() {
    let parser = KimiK2Parser::new();

    // Test with extra whitespace
    let input = r#"<|tool_calls_section_begin|>
<|tool_call_begin|> functions.test:0 <|tool_call_argument_begin|> {"key": "value", "num": 42} <|tool_call_end|>
<|tool_calls_section_end|>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "test");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert_eq!(args["key"], "value");
    assert_eq!(args["num"], 42);
}

#[tokio::test]
async fn test_kimik2_streaming() {
    let parser = KimiK2Parser::new();
    let mut state = ParseState::new();

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
    let mut found_complete = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &mut state).await.unwrap();

        match result {
            StreamResult::ToolName { name, .. } => {
                assert_eq!(name, "calculate");
                found_name = true;
            }
            StreamResult::ToolComplete(tool) => {
                assert_eq!(tool.function.name, "calculate");
                found_complete = true;
            }
            _ => {}
        }
    }

    assert!(found_name || found_complete);
}

#[test]
fn test_kimik2_format_detection() {
    let parser = KimiK2Parser::new();

    // Should detect Kimi K2 format
    assert!(parser.detect_format("<|tool_calls_section_begin|>"));
    assert!(parser.detect_format("<|tool_call_begin|>"));
    assert!(parser.detect_format("text with <|tool_calls_section_begin|> marker"));

    // Should not detect other formats
    assert!(!parser.detect_format("[TOOL_CALLS]"));
    assert!(!parser.detect_format("<tool_call>"));
    assert!(!parser.detect_format("plain text"));
}

#[tokio::test]
async fn test_kimik2_sequential_indices() {
    let parser = KimiK2Parser::new();

    // Test with proper sequential indexing
    let input = r#"<|tool_calls_section_begin|>
<|tool_call_begin|>functions.first:0<|tool_call_argument_begin|>{"param": "a"}<|tool_call_end|>
<|tool_call_begin|>functions.second:1<|tool_call_argument_begin|>{"param": "b"}<|tool_call_end|>
<|tool_call_begin|>functions.third:2<|tool_call_argument_begin|>{"param": "c"}<|tool_call_end|>
<|tool_calls_section_end|>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].function.name, "first");
    assert_eq!(result[1].function.name, "second");
    assert_eq!(result[2].function.name, "third");
}

#[tokio::test]
async fn test_function_index_extraction() {
    let parser = KimiK2Parser::new();

    let input = r#"Text before tool calls.
<|tool_calls_section_begin|>
<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>{"query": "rust"}<|tool_call_end|>
<|tool_call_begin|>functions.calc:1<|tool_call_argument_begin|>{"x": 10}<|tool_call_end|>
<|tool_calls_section_end|>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].function.name, "search");
    assert_eq!(result[1].function.name, "calc");
    // TODO: Verify indices are preserved: 0 and 1
    // TODO: Verify normal text = "Text before tool calls."
}

#[tokio::test]
async fn test_namespace_extraction() {
    let parser = KimiK2Parser::new();

    let input = r#"<|tool_calls_section_begin|>
<|tool_call_begin|>api.tools.search:0<|tool_call_argument_begin|>{"q": "test"}<|tool_call_end|>
<|tool_calls_section_end|>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "search"); // Should extract after last dot
}
