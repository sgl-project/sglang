//! DeepSeek V3 Parser Integration Tests

use sglang_router_rs::tool_parser::{DeepSeekParser, ParseState, StreamResult, ToolParser};

#[tokio::test]
async fn test_deepseek_complete_parsing() {
    let parser = DeepSeekParser::new();

    // Test single tool call
    let input = r#"Let me help you with that.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "Tokyo", "units": "celsius"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
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
async fn test_deepseek_multiple_tools() {
    let parser = DeepSeekParser::new();

    let input = r#"<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>search
```json
{"query": "rust programming"}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>translate
```json
{"text": "Hello World", "to": "ja"}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].function.name, "search");
    assert_eq!(result[1].function.name, "translate");
}

#[tokio::test]
async fn test_deepseek_streaming() {
    let parser = DeepSeekParser::new();
    let mut state = ParseState::new();

    // Simulate streaming chunks
    let chunks = vec![
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>",
        "function<｜tool▁sep｜>get_weather\n",
        "```json\n",
        r#"{"location": "#,
        r#""Beijing", "#,
        r#""units": "metric"}"#,
        "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    ];

    let mut found_name = false;
    let mut found_complete = false;

    for chunk in chunks {
        let result = parser.parse_incremental(chunk, &mut state).await.unwrap();

        match result {
            StreamResult::ToolName { name, .. } => {
                assert_eq!(name, "get_weather");
                found_name = true;
            }
            StreamResult::ToolComplete(tool) => {
                assert_eq!(tool.function.name, "get_weather");
                found_complete = true;
            }
            _ => {}
        }
    }

    assert!(found_name || found_complete);
}

#[tokio::test]
async fn test_deepseek_nested_json() {
    let parser = DeepSeekParser::new();

    let input = r#"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>process
```json
{
    "data": {
        "nested": {
            "deep": [1, 2, 3]
        }
    }
}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "process");

    let args: serde_json::Value = serde_json::from_str(&result[0].function.arguments).unwrap();
    assert!(args["data"]["nested"]["deep"].is_array());
}

#[test]
fn test_deepseek_format_detection() {
    let parser = DeepSeekParser::new();

    // Should detect DeepSeek format
    assert!(parser.detect_format("<｜tool▁calls▁begin｜>"));
    assert!(parser.detect_format("text with <｜tool▁calls▁begin｜> marker"));

    // Should not detect other formats
    assert!(!parser.detect_format("[TOOL_CALLS]"));
    assert!(!parser.detect_format("<tool_call>"));
    assert!(!parser.detect_format("plain text"));
}

#[tokio::test]
async fn test_deepseek_malformed_json_handling() {
    let parser = DeepSeekParser::new();

    // Malformed JSON should be skipped
    let input = r#"<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>broken
```json
{invalid json}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>valid
```json
{"key": "value"}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"#;

    let result = parser.parse_complete(input).await.unwrap();
    // Only the valid tool call should be parsed
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "valid");
}

#[tokio::test]
async fn test_normal_text_extraction() {
    let parser = DeepSeekParser::new();

    // Python extracts text before tool calls as normal_text
    let input = r#"Let me help you with that.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "Tokyo"}
```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].function.name, "get_weather");

    // TODO: Verify normal text extraction when parser returns it
    // In Python: normal_text = "Let me help you with that."
}

#[tokio::test]
async fn test_multiple_tool_calls() {
    let parser = DeepSeekParser::new();

    let input = r#"<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "Tokyo"}
```<｜tool▁call▁end｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "Paris"}
```<｜tool▁call▁end｜>
<｜tool▁calls▁end｜><｜end▁of▁sentence｜>"#;

    let result = parser.parse_complete(input).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].function.name, "get_weather");
    assert_eq!(result[1].function.name, "get_weather");
}
